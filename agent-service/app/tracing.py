"""LangSmith tracing setup.

LangChain/LangGraph auto-trace to LangSmith when these env vars are set:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=...
  LANGCHAIN_PROJECT=sun-protector

We just validate + surface a clear log line so it's obvious on startup
whether tracing is actually active (useful for the defense demo, where
you need to point at a live LangSmith dashboard).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("sun_protector.tracing")


def setup_tracing() -> bool:
    os.environ.setdefault("LANGCHAIN_PROJECT", "sun-protector")
    enabled = bool(os.getenv("LANGCHAIN_API_KEY")) and os.getenv("LANGCHAIN_TRACING_V2", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if enabled:
        logger.info("LangSmith tracing ENABLED (project=%s)", os.environ["LANGCHAIN_PROJECT"])
    else:
        logger.warning(
            "LangSmith tracing DISABLED — set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY "
            "in .env to see traces at https://smith.langchain.com"
        )
    return enabled
