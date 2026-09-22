from __future__ import annotations

from typing import Any, Optional, TypedDict


class LatLon(TypedDict, total=False):
    lat: float
    lon: float


class SunProtectorState(TypedDict, total=False):
    # --- inputs ---
    phototype: int  # Fitzpatrick I-VI (1-6)
    origin: LatLon
    destination: Optional[LatLon]
    minutes_outside: int
    minutes_since_last_spf: Optional[int]
    sweating_or_swimming: bool
    question: Optional[str]  # free-form question routed through RAG

    # --- derived / working memory ---
    uv_data: dict[str, Any]
    risk: dict[str, Any]
    route_options: list[dict[str, Any]]
    rag_answer: Optional[str]
    rag_sources: list[dict[str, Any]]

    # --- human-in-the-loop ---
    needs_confirmation: bool
    urgent_message: Optional[str]
    confirmed: Optional[bool]

    # --- loop control ---
    recheck_count: int

    # --- output ---
    final_recommendation: Optional[str]
    model_used: Optional[str]
    trace: list[str]  # human-readable node trace, shown in the UI for transparency
