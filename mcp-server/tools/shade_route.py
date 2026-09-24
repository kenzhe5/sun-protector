"""get_shade_route tool implementation.

Shade-aware pedestrian routing with building heights:

1. Ask the FOSSGIS OSRM foot-routing server for walking route alternatives
   between origin and destination (routing.openstreetmap.de; the
   router.project-osrm.org demo only has a car profile, whatever the URL says).
2. Compute the sun's azimuth/elevation for "now" at the route's location
   using a standard NOAA solar-position approximation (pure math, no API).
3. Sample up to 30 points per alternative and fetch the buildings near all
   of them with a single OpenStreetMap Overpass request — with their
   outlines (`out geom`) and tags: `height`, else `building:levels` × 3 m,
   else a default by building type (house ~6 m, garage/kiosk ~3 m, other ~9 m).
4. Shadow test per point (2D ray casting on flat ground): cast a ray from
   the point towards the sun; the point is in shade if the ray hits a
   building outline closer than that building's shadow length,
   L = height / tan(sun elevation). Low sun -> long shadows, noon -> short.
5. Return routes sorted by shade_fraction (most shade first), each with
   distance/duration/shade_fraction, so the agent/LLM can trade off
   "10% longer but 40% more shade".

Known simplifications: flat roofs and flat terrain, trees/awnings ignored,
buildings mapped as OSM relations (multipolygons) are skipped, and many
buildings lack height tags, so defaults are used — the share of buildings
with a known height is returned as `height_known_share`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

import httpx


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

    # азимут от севера по часовой: cos(Az) = (sin δ − sin φ·cos z) / (cos φ·sin z);
    # в полдень в северном полушарии даёт 180° (солнце на юге)
    az_cos = (math.sin(decl) - math.sin(lat_r) * math.cos(zenith)) / (
        math.cos(lat_r) * math.sin(zenith) + 1e-9
    )
    az_cos = max(-1, min(1, az_cos))
    azimuth = math.degrees(math.acos(az_cos))
    if hour_angle > 0:
        azimuth = 360 - azimuth

    return azimuth, elevation


# Публичный Overpass бывает перегружен (504) — не держим пользователя дольше этого.
BUILDINGS_TIMEOUT_S = 25
MAX_SAMPLES_PER_ROUTE = 30

LEVEL_HEIGHT_M = 3.0
FALLBACK_HEIGHT_M = 9.0  # ~3 этажа, если в OSM нет ни высоты, ни этажности
DEFAULT_HEIGHT_BY_TYPE = {
    "house": 6.0, "detached": 6.0, "semidetached_house": 6.0, "terrace": 6.0,
    "bungalow": 4.0, "cabin": 4.0, "hut": 3.0, "shed": 3.0, "kiosk": 3.0,
    "garage": 3.0, "garages": 3.0, "roof": 4.0, "carport": 3.0, "service": 4.0,
}
# Радиус поиска домов вокруг маршрута: тень дома высотой ~30 м при текущем
# солнце, но не меньше 30 м и не больше 150 м (иначе ответ Overpass огромный).
SEARCH_HEIGHT_M = 30.0
MIN_SEARCH_M, MAX_SEARCH_M = 30.0, 150.0
MAX_SHADOW_M = 200.0

_M_PER_DEG = 6371000 * math.pi / 180


def _parse_number(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"\s*([\d.]+)", value.replace(",", "."))
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def _building_height(tags: dict) -> tuple[float, bool]:
    """(высота в метрах, известна ли она из OSM, а не взята по умолчанию)."""
    raw = tags.get("height") or tags.get("building:height")
    h = _parse_number(raw)
    if h and h > 0:
        if raw and ("ft" in raw or "'" in raw):
            h *= 0.3048
        return h, True
    levels = _parse_number(tags.get("building:levels"))
    if levels and levels > 0:
        return levels * LEVEL_HEIGHT_M, True
    return DEFAULT_HEIGHT_BY_TYPE.get(tags.get("building", ""), FALLBACK_HEIGHT_M), False


def _search_radius(sun_el: float) -> float:
    """Округляем вверх до шага 30 м, чтобы запрос (и ключ кэша) не менялся
    каждые несколько минут вместе с высотой солнца."""
    if sun_el <= 0:
        return MIN_SEARCH_M
    raw = SEARCH_HEIGHT_M / math.tan(math.radians(sun_el))
    return max(MIN_SEARCH_M, min(MAX_SEARCH_M, math.ceil(raw / 30) * 30))


# Если Overpass не отдал контуры (504 под нагрузкой), берём только центры домов
# и считаем дом квадратом такого размера вокруг центра.
FALLBACK_HALF_SIZE_M = 8.0
ATTEMPT_TIMEOUT_S = 12


def _square_around(lat: float, lon: float, half: float) -> list[dict]:
    k = math.cos(math.radians(lat)) * _M_PER_DEG
    return [
        {"lat": lat + dy / _M_PER_DEG, "lon": lon + dx / k}
        for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))
    ]


# Дома меняются редко, а публичный Overpass часто отвечает 504 — успешные
# ответы кладём в файловый кэш на сутки (MCP-сервер живёт один вызов, поэтому
# кэш в памяти процесса не пережил бы следующий запрос).
CACHE_DIR = os.path.join(tempfile.gettempdir(), "sun_protector_overpass")
CACHE_TTL_S = 24 * 3600


def _cache_path(query: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.sha1(query.encode()).hexdigest() + ".json")


def _cached(query: str) -> list[dict] | None:
    path = _cache_path(query)
    try:
        if time.time() - os.path.getmtime(path) < CACHE_TTL_S:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except (OSError, ValueError):
        pass
    return None


async def _overpass(client: httpx.AsyncClient, query: str) -> list[dict]:
    cached = _cached(query)
    if cached is not None:
        return cached
    path = _cache_path(query)
    resp = await client.post(
        "https://overpass-api.de/api/interpreter", data={"data": query}, timeout=ATTEMPT_TIMEOUT_S
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(elements, f)
    except OSError:
        pass  # кэш — только ускорение, без него всё работает
    return elements


async def _fetch_buildings_along(
    client: httpx.AsyncClient, paths: list[list[tuple[float, float]]], radius_m: float
) -> tuple[list[dict], str]:
    """One Overpass request for buildings (outline + tags) near all sampled
    route points — the `around` filter accepts a whole polyline. Returns
    (buildings, mode): mode "outline" or "center" (fallback, lighter query)."""
    parts = []
    for path in paths:
        coords = ",".join(f"{lat:.6f},{lon:.6f}" for lat, lon in path)
        parts.append(f'way["building"](around:{radius_m:.0f},{coords});')
    selection = f"[out:json][timeout:25];({''.join(parts)});"
    q_outline, q_center = selection + "out geom;", selection + "out center;"
    # сначала то, что уже есть в кэше, — без обращения к сети
    if _cached(q_outline) is not None:
        elements, mode = _cached(q_outline), "outline"
    elif _cached(q_center) is not None:
        elements, mode = _cached(q_center), "center"
    else:
        try:
            elements, mode = await _overpass(client, q_outline), "outline"
        except (httpx.HTTPError, ValueError):
            elements, mode = await _overpass(client, q_center), "center"
    if mode == "center":
        for el in elements:
            if el.get("center"):
                el["geometry"] = _square_around(el["center"]["lat"], el["center"]["lon"], FALLBACK_HALF_SIZE_M)
    buildings = []
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 3:
            continue
        height, known = _building_height(el.get("tags", {}))
        lat_c = sum(p["lat"] for p in geom) / len(geom)
        lon_c = sum(p["lon"] for p in geom) / len(geom)
        k = math.cos(math.radians(lat_c)) * _M_PER_DEG
        pts = [((p["lon"] - lon_c) * k, (p["lat"] - lat_c) * _M_PER_DEG) for p in geom]
        buildings.append({
            "lat": lat_c, "lon": lon_c, "height": height, "height_known": known,
            "radius": max(math.hypot(x, y) for x, y in pts),
            "outline": [(p["lat"], p["lon"]) for p in geom],
        })
    return buildings, mode


def _ray_hit_distance(point: tuple[float, float], outline: list[tuple[float, float]], direction: tuple[float, float]) -> float | None:
    """Расстояние (м) от точки до ближайшего пересечения луча с контуром дома."""
    lat0, lon0 = point
    k = math.cos(math.radians(lat0)) * _M_PER_DEG
    xy = [((lon - lon0) * k, (lat - lat0) * _M_PER_DEG) for lat, lon in outline]
    dx, dy = direction
    best = None
    for (ax, ay), (bx, by) in zip(xy, xy[1:] + xy[:1]):
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            continue  # луч параллелен стене
        t = (ax * ey - ay * ex) / denom  # расстояние вдоль луча
        u = (ax * dy - ay * dx) / denom  # положение на стене (0..1)
        if t >= 0 and 0 <= u <= 1 and (best is None or t < best):
            best = t
    return best


def _is_shaded(point: tuple[float, float], buildings: list[dict], sun_az: float, sun_el: float) -> bool:
    tan_el = math.tan(math.radians(sun_el))
    direction = (math.sin(math.radians(sun_az)), math.cos(math.radians(sun_az)))  # восток, север
    lat0, lon0 = point
    k = math.cos(math.radians(lat0)) * _M_PER_DEG
    for b in buildings:
        shadow = min(MAX_SHADOW_M, b["height"] / tan_el)
        center_dist = math.hypot((b["lon"] - lon0) * k, (b["lat"] - lat0) * _M_PER_DEG)
        if center_dist > shadow + b["radius"]:
            continue  # дом слишком далеко, чтобы его тень дотянулась
        hit = _ray_hit_distance(point, b["outline"], direction)
        if hit is not None and hit <= shadow:
            return True
    return False


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
            step = max(1, math.ceil(len(coords) / MAX_SAMPLES_PER_ROUTE))
            samples.append([(lat, lon) for lon, lat in coords[::step]] or [(lat, lon) for lon, lat in coords])

        buildings, outline_mode = [], None
        shade_known, shade_error = True, None
        if sun_el > 0:
            try:
                buildings, outline_mode = await asyncio.wait_for(
                    _fetch_buildings_along(client, samples, _search_radius(sun_el)), BUILDINGS_TIMEOUT_S
                )
            except Exception as exc:  # noqa: BLE001 — таймаут/504: маршрут отдаём без оценки тени
                shade_known = False
                shade_error = f"{type(exc).__name__}: {exc}"[:300]
                print(f"shade: buildings fetch failed: {shade_error}", file=sys.stderr, flush=True)

        scored_routes = []
        for route, sampled in zip(routes, samples):
            if sun_el <= 0:
                # sun below horizon -> treat whole route as "shaded" (no direct UV)
                shade_fraction = 1.0
            elif not shade_known:
                shade_fraction = None
            else:
                shaded_count = sum(_is_shaded(p, buildings, sun_az, sun_el) for p in sampled)
                shade_fraction = shaded_count / len(sampled) if sampled else 0.0

            scored_routes.append(
                {
                    "distance_m": round(route["distance"]),
                    "duration_min": round(route["duration"] / 60, 1),
                    "shade_fraction": round(shade_fraction, 2) if shade_fraction is not None else None,
                    "shade_error": shade_error,
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
        "shade_error": shade_error,
        "buildings_used": len(buildings),
        "building_shapes": outline_mode,  # "outline" — контуры из OSM, "center" — квадраты вокруг центров
        "height_known_share": round(sum(b["height_known"] for b in buildings) / len(buildings), 2) if buildings else None,
        "method": "OSRM alternatives + OSM building outlines & heights + NOAA sun position, 2D shadow ray casting",
    }
