"""find_nearby_pharmacy tool implementation.

Data source: OpenStreetMap Overpass API — free, no key, community-maintained
POI data. Good enough for "nearest pharmacy to buy sunscreen" without
needing a Google Places billing account.
"""
from __future__ import annotations

import asyncio

import httpx

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


async def _post_with_retry(client: httpx.AsyncClient, data: dict, attempts: int = 3) -> httpx.Response:
    last_exc: Exception | None = None
    for i in range(attempts):
        url = OVERPASS_MIRRORS[i % len(OVERPASS_MIRRORS)]
        try:
            resp = await client.post(url, data=data)
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            await asyncio.sleep(1.5 * (i + 1))
    raise last_exc  # type: ignore[misc]


async def find_nearby_pharmacy(latitude: float, longitude: float, radius_m: int = 1200) -> dict:
    """Find pharmacies (amenity=pharmacy) within radius_m meters of a point."""
    query = f"""
    [out:json][timeout:15];
    node["amenity"="pharmacy"](around:{radius_m},{latitude},{longitude});
    out center 10;
    """
    headers = {"User-Agent": "sun-protector-course-project/0.1 (educational)"}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        resp = await _post_with_retry(client, {"data": query})
        data = resp.json()

    def haversine_m(lat1, lon1, lat2, lon2):
        import math

        r = 6371000
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        return 2 * r * math.asin(min(1, a**0.5))

    results = []
    for el in data.get("elements", []):
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        results.append(
            {
                "name": tags.get("name", "Pharmacy"),
                "latitude": lat,
                "longitude": lon,
                "distance_m": round(haversine_m(latitude, longitude, lat, lon)),
                "opening_hours": tags.get("opening_hours"),
            }
        )
    results.sort(key=lambda r: r["distance_m"])

    return {
        "query_point": {"latitude": latitude, "longitude": longitude},
        "radius_m": radius_m,
        "count": len(results),
        "pharmacies": results[:10],
        "source": "openstreetmap.org (Overpass API)",
    }
