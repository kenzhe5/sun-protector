"""Builds the LangGraph workflow.

Flow:

    intake -> fetch_uv -> assess_risk --[risk_branch]-->
        "urgent" -> urgent_advice (срочное «уйдите в тень») -> plan_route
        "normal" -> plan_route
    plan_route -> rag_answer -> compose_response --[recheck_branch]-->
        "recheck" -> recheck_update -> assess_risk   (cycle)
        "end" -> END

This gives the required "multi-step workflow with conditional logic":
branching (risk_branch) and a cycle (recheck loop, capped). Night is
handled in assess_risk (is_day from Open-Meteo): no UV → low risk.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import nodes
from .state import SunProtectorState


def build_graph():
    graph = StateGraph(SunProtectorState)

    graph.add_node("intake", nodes.intake)
    graph.add_node("fetch_uv", nodes.fetch_uv)
    graph.add_node("assess_risk", nodes.assess_risk_node)
    graph.add_node("urgent_advice", nodes.urgent_advice)
    graph.add_node("plan_route", nodes.plan_route)
    graph.add_node("rag_answer", nodes.rag_answer_node)
    graph.add_node("compose_response", nodes.compose_response)
    graph.add_node("recheck_update", nodes.recheck_update)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "fetch_uv")
    graph.add_edge("fetch_uv", "assess_risk")

    graph.add_conditional_edges(
        "assess_risk",
        nodes.risk_branch,
        {"urgent": "urgent_advice", "normal": "plan_route"},
    )
    graph.add_edge("urgent_advice", "plan_route")
    graph.add_edge("plan_route", "rag_answer")
    graph.add_edge("rag_answer", "compose_response")

    graph.add_conditional_edges(
        "compose_response",
        nodes.recheck_branch,
        {"recheck": "recheck_update", "end": END},
    )
    graph.add_edge("recheck_update", "assess_risk")

    return graph.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
