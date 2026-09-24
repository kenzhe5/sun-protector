from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from . import config, geo, mcp_client
from .chat import ChatContext, run_chat
from .graph.graph import get_graph
from .guardrails import check_input, check_output
from .multimodal.label_ocr import analyze_label_photo
from .multimodal.skin_analysis import analyze_skin_photo
from .rag.retriever import retrieve
from .tracing import setup_tracing

logging.basicConfig(level=logging.INFO)
setup_tracing()

app = FastAPI(title="Sun Protector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LatLon(BaseModel):
    lat: float
    lon: float


class RecommendRequest(BaseModel):
    phototype: int
    origin: LatLon
    destination: LatLon | None = None
    minutes_outside: int = 30
    minutes_since_last_spf: int | None = None
    sweating_or_swimming: bool = False
    question: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class Place(BaseModel):
    lat: float
    lon: float
    label: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    location: LatLon | None = None
    map_point: Place | None = None
    places: dict[str, Place | None] = {}
    phototype: int | None = None
    minutes_since_last_spf: int | None = None
    sweating_or_swimming: bool = False
    route_from: Place | None = None  # точки из формы «Откуда/Куда»
    route_to: Place | None = None


class AskRequest(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def _format_result(state: dict) -> dict:
    return {
        "status": "done",
        "risk": state.get("risk"),
        "uv_data": state.get("uv_data"),
        "route_options": state.get("route_options"),
        "rag_sources": state.get("rag_sources"),
        "final_recommendation": state.get("final_recommendation"),
        "model_used": state.get("model_used"),
        "trace": state.get("trace"),
    }


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    if req.question:
        guarded = check_input(req.question)
        if not guarded.allowed:
            raise HTTPException(400, "Input blocked by guardrails (possible prompt injection).")
        req.question = guarded.text

    graph = get_graph()
    graph_input = {
        "phototype": req.phototype,
        "origin": req.origin.model_dump(),
        "destination": req.destination.model_dump() if req.destination else None,
        "minutes_outside": req.minutes_outside,
        "minutes_since_last_spf": req.minutes_since_last_spf,
        "sweating_or_swimming": req.sweating_or_swimming,
        "question": req.question,
    }
    state = await graph.ainvoke(graph_input)
    return _format_result(state)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(400, "Последнее сообщение должно быть от пользователя.")
    guarded = check_input(req.messages[-1].text)
    if not guarded.allowed:
        return {"text": "Не могу выполнить такую просьбу. Спросите про солнце, крем или маршрут 🙂", "routes": []}
    history = [m.model_dump() for m in req.messages]
    history[-1]["text"] = guarded.text

    ctx = ChatContext(
        location=req.location.model_dump() if req.location else None,
        map_point=req.map_point.model_dump() if req.map_point else None,
        places={k: v.model_dump() if v else None for k, v in req.places.items()},
        phototype=req.phototype,
        minutes_since_last_spf=req.minutes_since_last_spf,
        sweating=req.sweating_or_swimming,
    )
    try:
        route = (req.route_from.model_dump(), req.route_to.model_dump()) if req.route_from and req.route_to else None
        result = await run_chat(history, ctx, route)
    except Exception:
        logging.exception("chat failed")
        raise HTTPException(502, "Модель сейчас недоступна, попробуйте ещё раз.")
    result["text"] = check_output(result["text"], add_disclaimer=False).text
    return result


@app.post("/api/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return await analyze_skin_photo(data, file.content_type or "image/jpeg")
    except Exception as exc:
        logging.exception("analyze_skin failed")
        raise HTTPException(502, f"Модель не смогла обработать фото: {exc.__class__.__name__}")


@app.post("/api/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return await analyze_label_photo(data, file.content_type or "image/jpeg")
    except Exception as exc:
        logging.exception("analyze_label failed")
        raise HTTPException(502, f"Модель не смогла обработать фото: {exc.__class__.__name__}")


ASK_SYSTEM_PROMPT = (
    "Ты — Sun Protector. Отвечай на вопросы про SPF/UV/защиту кожи кратко и по делу "
    "(3-5 предложений), опираясь ТОЛЬКО на приведённые фрагменты гайдлайнов. "
    "Если фрагменты не отвечают на вопрос, скажи об этом честно."
)


@app.post("/api/ask")
async def ask(req: AskRequest):
    guarded_in = check_input(req.question)
    if not guarded_in.allowed:
        raise HTTPException(400, "Input blocked by guardrails (possible prompt injection).")

    docs = retrieve(guarded_in.text, k=4)
    context = "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)
    messages = [
        SystemMessage(content=ASK_SYSTEM_PROMPT),
        HumanMessage(content=f"Контекст гайдлайнов:\n{context}\n\nВопрос: {guarded_in.text}"),
    ]
    result, model_used = await config.call_with_fallback(messages)
    guarded_out = check_output(config.text_of(result))
    return {
        "answer": guarded_out.text,
        "sources": [{"source": d.metadata.get("source"), "excerpt": d.page_content[:200]} for d in docs],
        "model_used": model_used,
    }


@app.get("/api/sun")
async def sun(lat: float, lon: float):
    """День/ночь и UV прямо сейчас — для строки статуса, без вызова LLM."""
    uv = await mcp_client.call_tool("get_uv_forecast_tool", latitude=lat, longitude=lon)
    return {
        "is_day": uv.get("is_day", True),
        "uv_index": uv.get("current_uv_index"),
        "risk_band": uv.get("current_risk_band"),
        "sunrise": uv.get("next_sunrise") or uv.get("sunrise"),
        "sunset": uv.get("sunset"),
        "daily_uv_max": uv.get("daily_uv_max"),
    }


@app.get("/api/geocode")
async def geocode(q: str, lat: float | None = None, lon: float | None = None):
    return await geo.geocode(q, lat, lon)


@app.get("/api/reverse")
async def reverse(lat: float, lon: float):
    return {"label": await geo.reverse_label(lat, lon)}


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Без Cache-Control браузер кэширует index.html/app.js «на глазок» и после
    обновления показывает старый фронтенд. no-cache = всегда сверяться с сервером
    (по ETag — если файл не менялся, ответ 304 без повторной загрузки)."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
