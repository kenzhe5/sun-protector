"""Геокодинг через OpenStreetMap Nominatim: адрес → координаты и обратно.

Используется и формой поиска на карте (/api/geocode), и чат-агентом
(«проложи маршрут до Абая 150»).
"""
from __future__ import annotations

import asyncio

import httpx

NOMINATIM = "https://nominatim.openstreetmap.org"
NOMINATIM_HEADERS = {"User-Agent": "sun-protector-study/1.0"}


def _short_address(item: dict) -> str:
    a = item.get("address", {})
    street = " ".join(x for x in [a.get("road"), a.get("house_number")] if x)
    parts = [item.get("name"), street, a.get("city") or a.get("town") or a.get("village")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out) or item.get("display_name", "")


STREET_TYPES = ("улица", "ул", "проспект", "пр", "микрорайон", "мкр", "переулок", "бульвар", "шоссе")


def _query_variants(q: str) -> list[str]:
    """«Абая 150» → ещё «проспект Абая 150» и «улица Абая 150»: Nominatim плохо
    понимает адрес без типа улицы, а так пишут почти все."""
    words = q.lower().replace(".", " ").split()
    has_number = any(ch.isdigit() for ch in q)
    if has_number and words and words[0] not in STREET_TYPES and "," not in q:
        return [q, f"проспект {q}", f"улица {q}"]
    return [q]


def _rank(item: dict, q: str) -> int:
    """Выше — результаты, где совпали и улица, и номер дома из запроса."""
    a = item.get("address", {})
    tokens = [t for t in q.lower().replace(",", " ").split() if t not in STREET_TYPES]
    number = next((t for t in tokens if t[0].isdigit()), None)
    names = [t for t in tokens if not t[0].isdigit()]
    road = (a.get("road") or "").lower()
    score = 0
    if names and all(n in road or n in (item.get("name") or "").lower() for n in names):
        score += 2
    if number and (a.get("house_number") or "").lower() == number:
        score += 1
    return score


async def _nominatim_search(client: httpx.AsyncClient, q: str, viewbox: str | None, bounded: bool) -> list[dict]:
    params = {"q": q, "format": "jsonv2", "addressdetails": 1, "limit": 5, "accept-language": "ru"}
    if viewbox:
        params["viewbox"] = viewbox
        if bounded:
            params["bounded"] = 1
    r = await client.get(f"{NOMINATIM}/search", params=params)
    r.raise_for_status()
    return r.json()


async def geocode(q: str, lat: float | None = None, lon: float | None = None):
    """Поиск адреса/места по тексту (OSM Nominatim): сначала в пределах города
    вокруг центра карты, если пусто — в радиусе ~100 км (не по всему миру:
    иначе на «кайсар» находится «Кас» в Красноярском крае)."""
    viewbox = f"{lon - 0.3},{lat + 0.2},{lon + 0.3},{lat - 0.2}" if lat is not None and lon is not None else None
    async with httpx.AsyncClient(timeout=10, headers=NOMINATIM_HEADERS) as client:
        batches = await asyncio.gather(
            *(_nominatim_search(client, v, viewbox, bounded=True) for v in _query_variants(q)),
            return_exceptions=True,
        )
        found = [i for b in batches if isinstance(b, list) for i in b]
        if not found and lat is not None and lon is not None:
            wide = f"{lon - 1},{lat + 1},{lon + 1},{lat - 1}"
            found = await _nominatim_search(client, q, wide, bounded=True)

    seen, results = set(), []
    for i in sorted(found, key=lambda i: -_rank(i, q)):
        if i["place_id"] in seen:
            continue
        seen.add(i["place_id"])
        results.append({"lat": float(i["lat"]), "lon": float(i["lon"]), "label": _short_address(i), "full": i["display_name"]})
    return results[:6]


async def reverse_label(lat: float, lon: float):
    """Адрес по координатам — чтобы показать клик по карте человеческим текстом."""
    params = {"lat": lat, "lon": lon, "format": "jsonv2", "addressdetails": 1, "accept-language": "ru", "zoom": 18}
    try:
        async with httpx.AsyncClient(timeout=10, headers=NOMINATIM_HEADERS) as client:
            r = await client.get(f"{NOMINATIM}/reverse", params=params)
        r.raise_for_status()
        return _short_address(r.json())
    except Exception:
        return f"{lat:.5f}, {lon:.5f}"
