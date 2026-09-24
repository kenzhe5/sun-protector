"""get_shade_route tool implementation.

Real, if simplified, shade-aware pedestrian routing:

1. Ask the FOSSGIS OSRM foot-routing server for walking route alternatives
   between origin and destination (routing.openstreetmap.de; the
   router.project-osrm.org demo only has a car profile, whatever the URL says).
2. Sample up to 40 points per alternative and fetch buildings near all of
   them with a single OpenStreetMap Overpass request (60 m buffer).
3. Compute the sun's azimuth/elevation for "now" at the route's location
   using a standard NOAA solar-position approximation (no external API,
   pure math — this is the "UV/skin" domain reasoning the agent needs).
4. Score each route segment: a segment is considered "shaded" if there is
   a building on the side of the path facing the sun (i.e. between the
   walker and the sun) within ~15m. Aggregate to a shade_fraction per
   route.
5. Return routes sorted by shade_fraction (most shade first), each with
   distance/duration/shade_fraction, so the agent/LLM can trade off
   "10% longer but 40% more shade".

This is intentionally a heuristic, not true shadow ray-casting (which
would need building heights + a full solar shadow model) — documented
as a known simplification in ARCHITECTURE.md.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import httpx

from .pharmacy import _post_with_retry

OSRM_URL = "https://routing.openstreetmap.de/routed-foot/route/v1/foot"


def _solar_position(lat: float, lon: float, when: datetime) -> tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) of the sun. NOAA simplified algorithm."""
    when = when.astimezone(timezone.utc)
    day_of_year = when.timetuple().tm_yday
    hour_utc = when.hour + when.minute / 60 + when.second / 3600

    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hour_utc - 12) / 24)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    time_offset = eqtime + 4 * lon
    tst = hour_utc * 60 + time_offset
    hour_angle = math.radians(tst / 4 - 180)

    lat_r = math.radians(lat)
    zenith = math.acos(
        math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    )
    elevation = 90 - math.degrees(zenith)

    az_cos = (math.sin(lat_r) * math.cos(zenith) - math.sin(decl)) / (
        math.cos(lat_r) * math.sin(zenith) + 1e-9
    )
    az_cos = max(-1, min(1, az_cos))
    azimuth = math.degrees(math.acos(az_cos))
    if hour_angle > 0:
        azimuth = 360 - azimuth

    return azimuth, elevation


def _bearing(lat1, lon1, lat2, lon2) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


SHADE_RADIUS_M = 60
# Публичный Overpass бывает перегружен (504) — не держим пользователя дольше этого.
BUILDINGS_TIMEOUT_S = 20
MAX_SAMPLES_PER_ROUTE = 40


def _dist_m(lat1, lon1, lat2, lon2) -> float:
    """Equirectangular approximation — accurate enough at ~60 m scale."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6371000 * math.hypot(x, y)


async def _fetch_buildings_along(client: httpx.AsyncClient, paths: list[list[tuple[float, float]]]) -> list[dict]:
    """One Overpass request for buildings near all sampled route points
    (the `around` filter accepts a whole polyline), instead of one request
    per point — that was 100+ sequential calls on a long walking route."""
    parts = []
    for path in paths:
        coords = ",".join(f"{lat:.6f},{lon:.6f}" for lat, lon in path)
        parts.append(f'way["building"](around:{SHADE_RADIUS_M},{coords});')
    query = f"[out:json][timeout:25];({''.join(parts)});out center;"
    resp = await _post_with_retry(client, {"data": query})
    return [
        {"lat": el["center"]["lat"], "lon": el["center"]["lon"]}
        for el in resp.json().get("elements", [])
        if el.get("center")
    ]


async def get_shade_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> dict:
    """Return walking route alternatives ranked by estimated shade coverage."""
    headers = {"User-Agent": "sun-protector-course-project/0.1 (educational)"}
    async with httpx.AsyncClient(timeout=25.0, headers=headers) as client:
        osrm_resp = await client.get(
            f"{OSRM_URL}/{origin_lon},{origin_lat};{dest_lon},{dest_lat}",
            params={"overview": "full", "geometries": "geojson", "alternatives": "true", "steps": "false"},
        )
        osrm_resp.raise_for_status()
        osrm_data = osrm_resp.json()

        if osrm_data.get("code") != "Ok" or not osrm_data.get("routes"):
            return {"error": "no_route_found", "detail": osrm_data.get("code")}

        now = datetime.now(timezone.utc)
        sun_az, sun_el = _solar_position(origin_lat, origin_lon, now)

        routes = osrm_data["routes"][:3]
        samples = []
        for route in routes:
            coords = route["geometry"]["coordinates"]  # [lon, lat] pairs
            step = max(1, len(coords) // MAX_SAMPLES_PER_ROUTE)
            samples.append([(lat, lon) for lon, lat in coords[::step]] or [(lat, lon) for lon, lat in coords])

        buildings = []
        shade_known = True
        if sun_el > 0:
            try:
                buildings = await asyncio.wait_for(_fetch_buildings_along(client, samples), BUILDINGS_TIMEOUT_S)
            except Exception:  # noqa: BLE001 — таймаут/504: маршрут отдаём без оценки тени
                shade_known = False

        scored_routes = []
        for route, sampled in zip(routes, samples):
            if sun_el <= 0:
                # sun below horizon -> treat whole route as "shaded" (no direct UV)
                shade_fraction = 1.0
            elif not shade_known:
                shade_fraction = None
            else:
                shaded_count = 0
                for lat1, lon1 in sampled:
                    # shaded if a building within SHADE_RADIUS_M lies roughly
                    # towards the sun (within 45° of the solar azimuth)
                    for b in buildings:
                        if _dist_m(lat1, lon1, b["lat"], b["lon"]) > SHADE_RADIUS_M:
                            continue
                        b_bearing = _bearing(lat1, lon1, b["lat"], b["lon"])
                        diff = min((b_bearing - sun_az) % 360, (sun_az - b_bearing) % 360)
                        if diff < 45:
                            shaded_count += 1
                            break
                shade_fraction = shaded_count / len(sampled) if sampled else 0.0

            scored_routes.append(
                {
                    "distance_m": round(route["distance"]),
                    "duration_min": round(route["duration"] / 60, 1),
                    "shade_fraction": round(shade_fraction, 2) if shade_fraction is not None else None,
                    "geometry": route["geometry"],
                }
            )

    scored_routes.sort(key=lambda r: (-(r["shade_fraction"] or 0), r["distance_m"]))
    return {
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "destination": {"lat": dest_lat, "lon": dest_lon},
        "sun_azimuth_deg": round(sun_az, 1),
        "sun_elevation_deg": round(sun_el, 1),
        "routes": scored_routes,
        "recommended_route_index": 0 if scored_routes else None,
        "method": "OSRM alternatives + OSM building proximity + NOAA solar position heuristic",
    }
