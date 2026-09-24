"""Чат-агент Sun Protector.

LLM с tool calling поверх тех же компонентов, что и остальное приложение:
  - plan_walk / check_sun_now → LangGraph workflow (UV, риск, маршрут в тени)
  - search_guidelines        → RAG по гайдлайнам WHO/AAD/EPA
  - save_place               → геокодинг адреса дома/работы

Сервер без состояния: историю переписки, геолокацию, сохранённые места и
профиль присылает браузер в каждом запросе. Побочные результаты (маршруты
для карты, новые места) возвращаются отдельно от текста ответа.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from . import config, geo
from .graph.graph import get_graph
from .rag.retriever import retrieve

logger = logging.getLogger("sun_protector.chat")

MAX_TOOL_ROUNDS = 4

HERE_WORDS = {"", "я", "здесь", "тут", "моё местоположение", "мое местоположение", "текущее местоположение", "current"}
HOME_WORDS = {"дом", "домой", "home", "из дома", "от дома"}
WORK_WORDS = {"работа", "работу", "на работу", "work", "офис"}
MAP_WORDS = {"точка на карте", "map", "карта"}


@dataclass
class ChatContext:
    location: dict | None
    map_point: dict | None
    places: dict[str, dict | None]
    phototype: int | None  # None — пользователь ещё не говорил, считаем II
    minutes_since_last_spf: int | None
    sweating: bool
    # то, что уходит в браузер помимо текста
    routes: list[dict] = field(default_factory=list)
    route_points: dict[str, dict] = field(default_factory=dict)
    places_update: dict[str, dict] = field(default_factory=dict)
    profile_update: dict[str, Any] = field(default_factory=dict)
    sun: dict | None = None


class PlaceNotFound(Exception):
    pass


async def _resolve(ctx: ChatContext, place: str) -> dict:
    key = place.strip().lower()
    if key in HERE_WORDS:
        if not ctx.location:
            raise PlaceNotFound("Геолокация недоступна — попроси пользователя написать адрес, откуда он идёт.")
        return {**ctx.location, "label": "Моё местоположение"}
    if key in HOME_WORDS or key in WORK_WORDS:
        kind = "home" if key in HOME_WORDS else "work"
        if not ctx.places.get(kind):
            name = "дома" if kind == "home" else "работы"
            raise PlaceNotFound(f"Адрес {name} не сохранён — спроси адрес и вызови save_place.")
        return ctx.places[kind]
    if key in MAP_WORDS:
        if not ctx.map_point:
            raise PlaceNotFound("Точка на карте не выбрана.")
        return ctx.map_point
    near = ctx.location or ctx.places.get("home") or ctx.places.get("work") or {}
    found = await geo.geocode(place, near.get("lat"), near.get("lon"))
    if not found:
        raise PlaceNotFound(
            f"«{place}» не найдено на карте OpenStreetMap. Попроси точный адрес (улица и дом) "
            "или предложи кликнуть точку на карте."
        )
    return {"lat": found[0]["lat"], "lon": found[0]["lon"], "label": found[0]["label"]}


async def _run_workflow(ctx: ChatContext, origin: dict, destination: dict | None) -> dict:
    state = await get_graph().ainvoke(
        {
            "phototype": ctx.phototype or 2,
            "origin": {"lat": origin["lat"], "lon": origin["lon"]},
            "destination": {"lat": destination["lat"], "lon": destination["lon"]} if destination else None,
            "minutes_outside": 0,
            "minutes_since_last_spf": ctx.minutes_since_last_spf,
            "sweating_or_swimming": ctx.sweating,
            "question": None,
            "skip_summary": True,
        }
    )
    uv, risk = state.get("uv_data", {}), state.get("risk", {})
    ctx.sun = {
        "is_day": uv.get("is_day", True),
        "uv_index": uv.get("current_uv_index"),
        "risk_band": risk.get("risk_band"),
        "sunrise": uv.get("next_sunrise") or uv.get("sunrise"),
        "sunset": uv.get("sunset"),
        "minutes_until_reapply": risk.get("minutes_until_reapply"),
        "reapply_due": risk.get("reapply_due"),
    }
    return state


def _sun_summary(state: dict) -> dict:
    uv, risk = state.get("uv_data", {}), state.get("risk", {})
    return {
        "местное_время": uv.get("local_time"),
        "сейчас": "день" if uv.get("is_day", True) else "ночь",
        "восход": uv.get("next_sunrise") or uv.get("sunrise"),
        "закат": uv.get("sunset"),
        "uv_сейчас": uv.get("current_uv_index"),
        "uv_максимум_сегодня": uv.get("daily_uv_max"),
        "риск": risk.get("risk_band"),
        "минут_до_ожога_без_защиты": risk.get("safe_exposure_minutes"),
        "обновить_крем_сейчас": risk.get("reapply_due"),
        "минут_до_обновления_крема": risk.get("minutes_until_reapply"),
        "вывод": risk.get("recommendation_summary"),
        "срочно": state.get("urgent_message"),
    }


async def _plan(ctx: ChatContext, o: dict, d: dict) -> str:
    state = await _run_workflow(ctx, o, d)
    routes = state.get("route_options") or []
    ctx.routes = routes
    ctx.route_points = {"origin": o, "destination": d}
    result = _sun_summary(state)
    result["откуда"], result["куда"] = o["label"], d["label"]
    is_day = state.get("uv_data", {}).get("is_day", True)
    result["маршруты"] = [
        {
            "минут": r["duration_min"],
            "км": round(r["distance_m"] / 1000, 1),
            # ночью солнца нет — доля тени бессмысленна, не даём модели её упоминать
            "доля_тени": (
                "ночь, тень не важна" if not is_day
                else "не удалось посчитать (сервис карт не ответил)" if r["shade_fraction"] is None
                else r["shade_fraction"]
            ),
        }
        for r in routes
    ] or "пеший маршрут не найден"
    return json.dumps(result, ensure_ascii=False)


def _build_tools(ctx: ChatContext):
    @tool
    async def plan_walk(origin: str, destination: str) -> str:
        """Построить пеший маршрут (с учётом тени) и оценить UV-риск на прогулке.

        origin/destination: адрес или название места, либо одно из слов:
        "моё местоположение", "дом", "работа", "точка на карте".
        """
        try:
            o = await _resolve(ctx, origin)
            d = await _resolve(ctx, destination)
        except PlaceNotFound as exc:
            return f"ОШИБКА: {exc}"
        return await _plan(ctx, o, d)

    @tool
    async def check_sun_now(place: str = "моё местоположение") -> str:
        """UV-индекс и риск обгореть прямо сейчас: день или ночь, нужен ли крем,
        когда его обновить. place — адрес или "моё местоположение"/"дом"/"работа"."""
        try:
            p = await _resolve(ctx, place)
        except PlaceNotFound as exc:
            return f"ОШИБКА: {exc}"
        state = await _run_workflow(ctx, p, None)
        return json.dumps(_sun_summary(state), ensure_ascii=False)

    @tool
    def search_guidelines(question: str) -> str:
        """Найти ответ в рекомендациях WHO, AAD, EPA, SkinCancer.org
        (SPF, как и когда наносить крем, одежда, дети, плавание и т.п.)."""
        docs = retrieve(question, k=4)
        return "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)

    @tool
    async def save_place(kind: Literal["home", "work"], address: str) -> str:
        """Запомнить адрес дома (home) или работы (work).
        address — адрес, либо "моё местоположение" / "точка на карте"."""
        try:
            p = await _resolve(ctx, address)
        except PlaceNotFound as exc:
            return f"ОШИБКА: {exc}"
        if p.get("label") == "Моё местоположение":
            p = {**p, "label": await geo.reverse_label(p["lat"], p["lon"])}
        ctx.places[kind] = p
        ctx.places_update[kind] = p
        return f"Сохранено: {p['label']}"

    @tool
    def update_profile(
        phototype: int | None = None,
        sunscreen_applied_minutes_ago: int | None = None,
        no_sunscreen_yet: bool = False,
        sweating_or_swimming: bool | None = None,
    ) -> str:
        """Запомнить данные о пользователе из переписки.

        phototype — тип кожи по Фицпатрику 1-6 (1 очень светлая, всегда сгорает;
        2 светлая; 3 средняя; 4 смуглая; 5 тёмная; 6 очень тёмная).
        sunscreen_applied_minutes_ago — сколько минут назад нанесён крем.
        no_sunscreen_yet — крем сегодня не наносил.
        sweating_or_swimming — потеет / купается.
        """
        saved = []
        if phototype is not None:
            ctx.phototype = max(1, min(6, int(phototype)))
            ctx.profile_update["phototype"] = ctx.phototype
            saved.append(f"тип кожи {ctx.phototype}")
        if no_sunscreen_yet:
            ctx.minutes_since_last_spf = None
            ctx.profile_update["sunscreen_applied_minutes_ago"] = None
            saved.append("крем не нанесён")
        elif sunscreen_applied_minutes_ago is not None:
            ctx.minutes_since_last_spf = max(0, int(sunscreen_applied_minutes_ago))
            ctx.profile_update["sunscreen_applied_minutes_ago"] = ctx.minutes_since_last_spf
            saved.append(f"крем {ctx.minutes_since_last_spf} мин назад")
        if sweating_or_swimming is not None:
            ctx.sweating = bool(sweating_or_swimming)
            ctx.profile_update["sweating_or_swimming"] = ctx.sweating
            saved.append("потеет/купается" if ctx.sweating else "не потеет")
        return "Запомнено: " + (", ".join(saved) or "ничего")

    return [plan_walk, check_sun_now, search_guidelines, save_place, update_profile]


SYSTEM_PROMPT = """Ты — Sun Protector, дружелюбный помощник в чате: помогаешь дойти пешком
по тени и не сгореть на солнце. Отвечай по-русски, коротко (1–4 предложения),
как живой человек в мессенджере. Без заголовков и списков, без координат.

Как действовать:
- Спрашивают про дорогу («как дойти», «на работу», «проложи путь») → plan_walk.
  Если откуда не сказано — "моё местоположение". «Домой» → destination "дом",
  «на работу» → destination "работа".
- «Можно выйти?», «нужен ли крем?», «какой UV?» → check_sun_now.
- Вопросы про защиту от солнца (SPF, как наносить, дети, плавание) →
  search_guidelines, отвечай по найденному и кратко упомяни источник (WHO/AAD/EPA).
- Пользователь называет, где его дом или работа («я живу на…», «работаю в…»,
  «моя работа — …») → СНАЧАЛА save_place, потом (если просили) plan_walk
  с "дом"/"работа". Не спрашивай разрешения запомнить — просто запомни.
- Пользователь говорит про свою кожу («светлая», «легко сгораю», «смуглая»),
  когда мазался кремом или что купается → update_profile, затем отвечай
  уже с учётом новых данных (при необходимости вызови check_sun_now).
- Если тип кожи неизвестен и ты даёшь совет про время на солнце или крем —
  в конце одной фразой спроси, какая у человека кожа.
- Если инструмент вернул ОШИБКА — объясни по-человечески и спроси, что нужно.

Про солнце:
- Ночью (сейчас = ночь) ультрафиолета нет: прямо скажи, что выходить безопасно
  и крем не нужен. Можно добавить, во сколько восход.
- Днём скажи главное: уровень риска, нужен ли крем и когда обновить,
  какой маршрут выбрать (больше тени vs короче). Маршрут пользователь видит на
  карте — называй только время, расстояние и долю тени.
- Ты не врач и не ставишь диагнозы; про родинки и изменения кожи — к дерматологу.

Что известно о пользователе:
{context}"""


def _context_text(ctx: ChatContext) -> str:
    lines = [
        f"- фототип кожи: {ctx.phototype}" if ctx.phototype else "- фототип кожи: неизвестен (в расчётах берём II)",
        "- крем: ещё не наносил" if ctx.minutes_since_last_spf is None else f"- крем нанесён {ctx.minutes_since_last_spf} мин назад",
        f"- геолокация: {'есть' if ctx.location else 'нет'}",
        f"- дом: {ctx.places['home']['label'] if ctx.places.get('home') else 'не сохранён'}",
        f"- работа: {ctx.places['work']['label'] if ctx.places.get('work') else 'не сохранена'}",
    ]
    if ctx.map_point:
        lines.append(f"- на карте выбрана точка: {ctx.map_point.get('label')}")
    return "\n".join(lines)


def _model_name(msg: AIMessage) -> str:
    meta = msg.response_metadata or {}
    return meta.get("model_name") or meta.get("model") or "?"


async def run_chat(
    history: list[dict[str, str]], ctx: ChatContext, route: tuple[dict, dict] | None = None
) -> dict[str, Any]:
    """route — точки из формы «Откуда/Куда»: маршрут строим сразу, без
    раунда, где модель выбирает инструмент (быстрее на один LLM-вызов)."""
    tools = _build_tools(ctx)
    by_name = {t.name: t for t in tools}
    llm = config.get_primary_llm().bind_tools(tools).with_fallbacks(
        [config.get_fallback_llm().bind_tools(tools)]
    )

    messages: list = [SystemMessage(content=SYSTEM_PROMPT.format(context=_context_text(ctx)))]
    for m in history[-20:]:
        messages.append(HumanMessage(content=m["text"]) if m["role"] == "user" else AIMessage(content=m["text"]))

    used_tools: list[str] = []
    if route:
        origin, destination = route
        messages.append(AIMessage(content="", tool_calls=[{
            "name": "plan_walk",
            "args": {"origin": origin["label"], "destination": destination["label"]},
            "id": "form_route",
        }]))
        messages.append(ToolMessage(content=await _plan(ctx, origin, destination), tool_call_id="form_route"))
        used_tools.append("plan_walk")

    for _ in range(MAX_TOOL_ROUNDS):
        reply: AIMessage = await llm.ainvoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            used_tools.append(call["name"])
            try:
                output = await by_name[call["name"]].ainvoke(call["args"])
            except Exception as exc:  # noqa: BLE001 — ошибку инструмента отдаём модели
                logger.exception("tool %s failed", call["name"])
                output = f"ОШИБКА: инструмент не сработал ({exc.__class__.__name__})"
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    else:
        reply = await llm.ainvoke(messages)

    text = reply.content if isinstance(reply.content, str) else "".join(
        part.get("text", "") for part in reply.content if isinstance(part, dict)
    )
    return {
        "text": text,
        "routes": ctx.routes,
        "route_points": ctx.route_points,
        "places_update": ctx.places_update,
        "profile_update": ctx.profile_update,
        "sun": ctx.sun,
        "tools": used_tools,
        "model_used": _model_name(reply),
    }
