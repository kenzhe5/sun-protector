"""get_uv_forecast tool implementation.

Data source: Open-Meteo (https://open-meteo.com) — free, no API key,
provides hourly and daily UV index forecasts. Chosen over paid
alternatives (OpenWeather One Call, OpenUV) because it needs zero
credentials for a course project and is reliable enough for a demo.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def get_uv_forecast(latitude: float, longitude: float) -> dict:
    """Fetch current + hourly UV index forecast for a location.

    Returns a dict with the current UV index, a risk band, and the
    next 6 hours of forecasted UV so the agent can reason about
    "should I go out now vs later".
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "uv_index,is_day",
        "hourly": "uv_index,uv_index_clear_sky,temperature_2m,cloud_cover",
        "daily": "uv_index_max,sunrise,sunset",
        "timezone": "auto",
        "forecast_days": 2,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    # Время считаем в часовом поясе точки (Open-Meteo отдаёт его в
    # utc_offset_seconds), а не по часам сервера: в Docker они в UTC, и
    # вечером в Алматы раньше подставлялся дневной UV.
    tz = timezone(timedelta(seconds=data.get("utc_offset_seconds", 0)))
    now_local = datetime.now(tz)
    now_prefix = now_local.strftime("%Y-%m-%dT%H")

    hourly_times = data.get("hourly", {}).get("time", [])
    hourly_uv = data.get("hourly", {}).get("uv_index", [])
    current_idx = next((i for i, t in enumerate(hourly_times) if t.startswith(now_prefix)), 0)

    current = data.get("current", {})
    current_uv = current.get("uv_index")
    if current_uv is None and hourly_uv:
        current_uv = hourly_uv[current_idx]
    is_day = bool(current.get("is_day", 1))

    next_6h = [
        {"time": hourly_times[i], "uv_index": hourly_uv[i]}
        for i in range(current_idx, min(current_idx + 6, len(hourly_times)))
    ]

    daily = data.get("daily", {})
    sunrises, sunsets = daily.get("sunrise", []), daily.get("sunset", [])
    now_str = now_local.strftime("%Y-%m-%dT%H:%M")
    next_sunrise = next((t for t in sunrises if t > now_str), None)

    def risk_band(uv: float | None) -> str:
        if uv is None:
            return "unknown"
        if uv < 3:
            return "low"
        if uv < 6:
            return "moderate"
        if uv < 8:
            return "high"
        if uv < 11:
            return "very_high"
        return "extreme"

    return {
        "latitude": latitude,
        "longitude": longitude,
        "current_uv_index": current_uv,
        "current_risk_band": risk_band(current_uv),
        "daily_uv_max": (daily.get("uv_index_max") or [None])[0],
        "is_day": is_day,
        "local_time": now_str,
        "sunrise": (sunrises or [None])[0],
        "sunset": (sunsets or [None])[0],
        "next_sunrise": next_sunrise,
        "next_6h_forecast": next_6h,
        "source": "open-meteo.com",
    }
