"""get_uv_forecast tool implementation.

Data source: Open-Meteo (https://open-meteo.com) — free, no API key,
provides hourly and daily UV index forecasts. Chosen over paid
alternatives (OpenWeather One Call, OpenUV) because it needs zero
credentials for a course project and is reliable enough for a demo.
"""
from __future__ import annotations

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
        "hourly": "uv_index,uv_index_clear_sky,temperature_2m,cloud_cover",
        "daily": "uv_index_max",
        "timezone": "auto",
        "forecast_days": 1,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly_times = data.get("hourly", {}).get("time", [])
    hourly_uv = data.get("hourly", {}).get("uv_index", [])

    # find the current-hour index (Open-Meteo returns 24h, pick closest to "now" in local tz)
    from datetime import datetime

    now_iso_prefix = datetime.now().strftime("%Y-%m-%dT%H")
    current_idx = 0
    for i, t in enumerate(hourly_times):
        if t.startswith(now_iso_prefix):
            current_idx = i
            break

    current_uv = hourly_uv[current_idx] if hourly_uv else None
    next_6h = [
        {"time": hourly_times[i], "uv_index": hourly_uv[i]}
        for i in range(current_idx, min(current_idx + 6, len(hourly_times)))
    ]

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
        "daily_uv_max": data.get("daily", {}).get("uv_index_max", [None])[0],
        "next_6h_forecast": next_6h,
        "source": "open-meteo.com",
    }
