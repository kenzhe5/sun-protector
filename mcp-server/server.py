"""Sun Protector MCP server.

Exposes three tools over the Model Context Protocol (stdio transport):

  - get_uv_forecast(latitude, longitude)
  - find_nearby_pharmacy(latitude, longitude, radius_m=1200)
  - get_shade_route(origin_lat, origin_lon, dest_lat, dest_lon)

Run standalone for debugging:
    python server.py

Run as an MCP server (stdio), e.g. from Claude Desktop / the agent service:
    python -m mcp_server.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import find_nearby_pharmacy, get_shade_route, get_uv_forecast

mcp = FastMCP("sun-protector")


@mcp.tool()
async def get_uv_forecast_tool(latitude: float, longitude: float) -> dict:
    """Get the current and next-6-hour UV index forecast for a location.

    Use this to decide whether it's currently safe to be outside without
    protection, and when UV will drop to a safer level.
    """
    return await get_uv_forecast(latitude, longitude)


@mcp.tool()
async def find_nearby_pharmacy_tool(latitude: float, longitude: float, radius_m: int = 1200) -> dict:
    """Find the nearest pharmacies/drugstores to a location (to buy sunscreen).

    radius_m: search radius in meters, default 1200m (~15 min walk).
    """
    return await find_nearby_pharmacy(latitude, longitude, radius_m)


@mcp.tool()
async def get_shade_route_tool(
    origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float
) -> dict:
    """Get walking route alternatives between two points, ranked by estimated
    shade coverage, not just shortest distance. Shade = 2D shadow ray casting:
    current sun position (NOAA) + OpenStreetMap building outlines and heights
    (height / building:levels, defaults by type). Returns each alternative's
    distance, duration and shade_fraction (0-1), plus how many buildings had
    a known height.
    """
    return await get_shade_route(origin_lat, origin_lon, dest_lat, dest_lon)


if __name__ == "__main__":
    mcp.run(transport="stdio")
