"""Bridges the agent service to our own sun-protector MCP server.

Uses langchain-mcp-adapters to spawn the MCP server as a stdio subprocess
and load its tools as LangChain-compatible tools the LangGraph nodes can
call directly (deterministic control-flow) or hand to an LLM for
ReAct-style tool calling (used in the free-form /api/ask path).
"""
from __future__ import annotations

import json

from langchain_mcp_adapters.client import MultiServerMCPClient

from . import config

_MCP_SERVER_PATH = config.MCP_SERVER_ARGS


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "sun_protector": {
                "command": "python",
                "args": [_MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )


async def get_mcp_tools():
    """Return the 3 sun-protector MCP tools as LangChain Tool objects."""
    client = _client()
    return await client.get_tools()


def _unwrap(raw):
    """MCP tools return a list of content blocks (e.g. [{"type": "text",
    "text": "<json>"}]). Our tools always return a single JSON object, so
    unwrap + parse it back into a plain dict for the graph nodes."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        try:
            return json.loads(raw[0]["text"])
        except json.JSONDecodeError:
            return raw[0]["text"]
    return raw


async def call_tool(name: str, **kwargs):
    """Call a single MCP tool by name and return its parsed (dict) result."""
    tools = await get_mcp_tools()
    for tool in tools:
        if tool.name == name:
            raw = await tool.ainvoke(kwargs)
            return _unwrap(raw)
    raise ValueError(f"Unknown MCP tool: {name}. Available: {[t.name for t in tools]}")
