"""Проверки геометрии тени на примерах с заранее известным ответом.

Запуск: python mcp-server/tests/test_shade.py  (или pytest)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, timezone  # noqa: E402

from tools.shade_route import _M_PER_DEG, _building_height, _is_shaded, _solar_position  # noqa: E402

LAT0, LON0 = 43.24, 76.88  # точка пешехода (Алматы)
K = math.cos(math.radians(LAT0)) * _M_PER_DEG


def square(south_m: float, north_m: float, half_width_m: float = 5.0) -> list[tuple[float, float]]:
    """Квадратный дом: y от south_m до north_m (метры к северу от пешехода)."""
    corners = [(-half_width_m, south_m), (half_width_m, south_m), (half_width_m, north_m), (-half_width_m, north_m)]
    return [(LAT0 + y / _M_PER_DEG, LON0 + x / K) for x, y in corners]


def building(south_m: float, north_m: float, height: float) -> dict:
    outline = square(south_m, north_m)
    return {
        "lat": sum(p[0] for p in outline) / 4, "lon": sum(p[1] for p in outline) / 4,
        "height": height, "height_known": True, "radius": 10.0, "outline": outline,
    }


P = (LAT0, LON0)
SOUTH, NORTH = 180.0, 0.0


def test_close_building_on_sun_side_gives_shade():
    # дом 10 м высотой в 5 м к югу, солнце на юге под 45° → тень 10 м, накрывает
    assert _is_shaded(P, [building(-15, -5, 10)], SOUTH, 45)


def test_far_building_does_not_reach():
    # тот же дом в 20 м — тень 10 м не дотягивается
    assert not _is_shaded(P, [building(-30, -20, 10)], SOUTH, 45)


def test_low_sun_makes_long_shadow():
    # солнце под 20° → тень 10/tan(20°) ≈ 27 м, теперь дом в 20 м накрывает
    assert _is_shaded(P, [building(-30, -20, 10)], SOUTH, 20)


def test_taller_building_reaches_further():
    # 27-метровый дом (9 этажей) в 20 м при солнце 45° → тень 27 м
    assert _is_shaded(P, [building(-30, -20, 27)], SOUTH, 45)


def test_building_on_other_side_gives_no_shade():
    # дом к югу, а солнце на севере — тень падает от пешехода, а не на него
    assert not _is_shaded(P, [building(-15, -5, 30)], NORTH, 45)


def test_height_from_tags():
    assert _building_height({"height": "12 m"}) == (12.0, True)
    assert _building_height({"building:levels": "5"}) == (15.0, True)
    assert _building_height({"building": "garage"}) == (3.0, False)
    assert _building_height({"building": "yes"}) == (9.0, False)


def test_sun_is_south_at_noon_north_hemisphere():
    # Алматы, 24 сентября, ~12:59 по солнечному времени (07:00 UTC): солнце на юге
    az, el = _solar_position(43.24, 76.88, datetime(2026, 9, 24, 7, 0, tzinfo=timezone.utc))
    assert 170 < az < 190 and 40 < el < 50, (az, el)


def test_sun_is_west_in_afternoon_and_east_in_morning():
    # Нью-Йорк, 24 сентября: 14:30 местного (18:30 UTC) — юго-запад; 09:00 (13:00 UTC) — юго-восток
    az_pm, _ = _solar_position(40.758, -73.985, datetime(2026, 9, 24, 18, 30, tzinfo=timezone.utc))
    az_am, _ = _solar_position(40.758, -73.985, datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc))
    assert 200 < az_pm < 250, az_pm
    assert 110 < az_am < 160, az_am


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"все {len(tests)} проверок пройдены")
