from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .. import config, mcp_client
from ..guardrails import check_output
from ..rag.retriever import retrieve
from .risk import assess_risk
from .state import SunProtectorState

logger = logging.getLogger("sun_protector.graph")

URGENT_RISK_BANDS = {"very_high", "extreme"}
MAX_RECHECKS = 1


def _trace(state: SunProtectorState, msg: str) -> list[str]:
    return [*state.get("trace", []), msg]


async def intake(state: SunProtectorState) -> dict:
    return {
        "trace": _trace(state, "intake: normalized request"),
        "recheck_count": state.get("recheck_count", 0),
        "rag_sources": state.get("rag_sources", []),
    }


async def fetch_uv(state: SunProtectorState) -> dict:
    origin = state["origin"]
    uv = await mcp_client.call_tool("get_uv_forecast_tool", latitude=origin["lat"], longitude=origin["lon"])
    return {"uv_data": uv, "trace": _trace(state, f"fetch_uv: UV={uv.get('current_uv_index')} ({uv.get('current_risk_band')})")}


async def assess_risk_node(state: SunProtectorState) -> dict:
    uv = state["uv_data"].get("current_uv_index")
    if uv is None:
        uv = state["uv_data"].get("daily_uv_max") or 0
    result = assess_risk(
        phototype=state["phototype"],
        uv_index=uv,
        minutes_outside=state.get("minutes_outside", 30),
        minutes_since_last_spf=state.get("minutes_since_last_spf"),
        sweating_or_swimming=state.get("sweating_or_swimming", False),
    )
    risk_dict = dict(result.__dict__)
    if not state["uv_data"].get("is_day", True):
        # Ночь: UV нет — крем обновлять не нужно, даже если он нанесён давно.
        risk_dict.update(
            risk_band="low",
            reapply_due=False,
            minutes_until_reapply=-1,
            recommendation_summary="Сейчас ночь — ультрафиолета нет, солнцезащитный крем не нужен. Можно спокойно выходить.",
        )
        return {"risk": risk_dict, "trace": _trace(state, "assess_risk: night (is_day=0), UV-риска нет")}
    return {"risk": risk_dict, "trace": _trace(state, f"assess_risk: band={result.risk_band} ratio={result.risk_ratio}")}


def risk_branch(state: SunProtectorState) -> str:
    band = state["risk"]["risk_band"]
    return "urgent" if band in URGENT_RISK_BANDS else "normal"


async def urgent_advice(state: SunProtectorState) -> dict:
    """Очень высокий/экстремальный риск: добавляем срочное предупреждение
    («уйдите в тень сейчас»), которое compose_response ставит первым."""
    risk = state["risk"]
    message = f"⚠️ {risk['recommendation_summary']} (UV {risk['uv_index']})"
    return {
        "urgent_message": message,
        "trace": _trace(state, f"urgent_advice: {risk['risk_band']}"),
    }


async def plan_route(state: SunProtectorState) -> dict:
    dest = state.get("destination")
    if not dest:
        return {"route_options": [], "trace": _trace(state, "plan_route: no destination, skipped")}
    origin = state["origin"]
    result = await mcp_client.call_tool(
        "get_shade_route_tool",
        origin_lat=origin["lat"],
        origin_lon=origin["lon"],
        dest_lat=dest["lat"],
        dest_lon=dest["lon"],
    )
    routes = result.get("routes", [])
    return {
        "route_options": routes,
        "trace": _trace(state, f"plan_route: {len(routes)} alternatives, best shade={routes[0]['shade_fraction'] if routes else 'n/a'}"),
    }


async def rag_answer_node(state: SunProtectorState) -> dict:
    question = state.get("question")
    if not question:
        return {"rag_answer": None, "rag_sources": [], "trace": _trace(state, "rag: no question, skipped")}
    docs = retrieve(question, k=4)
    return {
        "rag_sources": [{"source": d.metadata.get("source"), "text": d.page_content[:300]} for d in docs],
        "trace": _trace(state, f"rag: retrieved {len(docs)} chunks"),
    }


SYSTEM_PROMPT = """Ты — Sun Protector, ассистент по безопасности на солнце.
Ты НЕ дерматолог и не ставишь диагнозы. Твои советы основаны на публичных
гайдлайнах (WHO, AAD) и на структурированных данных ниже (UV-индекс,
риск-скор, маршрут в тени, релевантные фрагменты гайдлайнов).

Дай короткий, конкретный ответ на русском языке:
1. Текущий риск и что делать прямо сейчас.
2. Когда повторно нанести крем (если применимо).
3. Если есть маршрут — какой выбрать и почему (тень vs расстояние).
4. Если есть цитаты из гайдлайнов — кратко используй их для обоснования.

Если сейчас НОЧЬ: скажи прямо, что ультрафиолета нет и выходить безопасно,
крем и тень не нужны (маршрут — просто самый удобный). Можно одной фразой
подсказать про завтра: во сколько восход и какой будет максимум UV.

Пиши по-человечески, 3-6 предложений, без воды и без нумерованных пунктов."""


async def compose_response(state: SunProtectorState) -> dict:
    if state.get("skip_summary"):
        # Чат сам пишет ответ по данным графа — второй вызов LLM тут лишний.
        return {"trace": _trace(state, "compose_response: skipped (chat mode)")}
    risk = state.get("risk", {})
    uv = state.get("uv_data", {})
    routes = state.get("route_options", [])
    rag_sources = state.get("rag_sources", [])
    question = state.get("question")

    context_parts = [
        f"Местное время: {uv.get('local_time')}, сейчас {'день' if uv.get('is_day', True) else 'НОЧЬ (солнце село)'}; "
        f"восход {uv.get('sunrise')}, закат {uv.get('sunset')}, следующий восход {uv.get('next_sunrise')}, "
        f"максимум UV за день {uv.get('daily_uv_max')}",
        f"UV сейчас: {uv.get('current_uv_index')} ({uv.get('current_risk_band')})",
        f"Риск: {risk.get('risk_band')} (ratio={risk.get('risk_ratio')}), safe_exposure_minutes={risk.get('safe_exposure_minutes')}",
        f"Повторное нанесение нужно: {risk.get('reapply_due')}, через {risk.get('minutes_until_reapply')} мин",
    ]
    if routes:
        best = routes[0]
        context_parts.append(
            f"Лучший маршрут: {best['distance_m']}м, {best['duration_min']}мин, доля тени={best['shade_fraction']*100:.0f}%"
        )
    if state.get("urgent_message"):
        context_parts.append(f"СРОЧНО (скажи первым): {state['urgent_message']}")
    if rag_sources:
        cites = "\n".join(f"- ({s['source']}) {s['text']}" for s in rag_sources)
        context_parts.append(f"Релевантные фрагменты гайдлайнов:\n{cites}")
    if question:
        context_parts.append(f"Вопрос пользователя: {question}")

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(context_parts))]

    result, model_used = await config.call_with_fallback(messages)
    guarded = check_output(result.content)

    return {
        "final_recommendation": guarded.text,
        "model_used": model_used,
        "trace": _trace(state, f"compose_response: model={model_used}"),
    }


def recheck_branch(state: SunProtectorState) -> str:
    routes = state.get("route_options") or []
    if not routes or state.get("recheck_count", 0) >= MAX_RECHECKS:
        return "end"
    if not state.get("uv_data", {}).get("is_day", True):
        return "end"  # ночью за время прогулки риск не вырастет
    best = routes[0]
    projected_minutes_outside = state.get("minutes_outside", 0) + best.get("duration_min", 0)
    reapply_interval = 60 if state.get("sweating_or_swimming") else 120
    since_spf = state.get("minutes_since_last_spf") or 0
    if projected_minutes_outside > 45 and (since_spf + best.get("duration_min", 0)) >= reapply_interval:
        return "recheck"
    return "end"


async def recheck_update(state: SunProtectorState) -> dict:
    """Simulates time passing during the walk (this is the workflow's
    cycle: assess -> plan -> (if the walk will run past the reapply
    window) -> loop back to re-assess with updated elapsed time)."""
    best = state["route_options"][0]
    added_minutes = best.get("duration_min", 0)
    return {
        "minutes_outside": state.get("minutes_outside", 0) + added_minutes,
        "minutes_since_last_spf": (state.get("minutes_since_last_spf") or 0) + added_minutes,
        "recheck_count": state.get("recheck_count", 0) + 1,
        "trace": _trace(state, f"recheck_update: +{added_minutes} min elapsed, re-assessing risk"),
    }
