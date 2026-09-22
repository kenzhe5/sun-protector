from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel

from . import config
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


class ResumeRequest(BaseModel):
    thread_id: str
    confirmed: bool


class AskRequest(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def _format_result(state: dict, thread_id: str) -> dict:
    if "__interrupt__" in state and state["__interrupt__"]:
        interrupt = state["__interrupt__"][0]
        return {
            "status": "needs_confirmation",
            "thread_id": thread_id,
            "interrupt": interrupt.value,
        }
    return {
        "status": "done",
        "thread_id": thread_id,
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

    thread_id = str(uuid.uuid4())
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
    state = await graph.ainvoke(graph_input, config={"configurable": {"thread_id": thread_id}})
    return _format_result(state, thread_id)


@app.post("/api/resume")
async def resume(req: ResumeRequest):
    graph = get_graph()
    state = await graph.ainvoke(
        Command(resume=req.confirmed), config={"configurable": {"thread_id": req.thread_id}}
    )
    return _format_result(state, req.thread_id)


@app.post("/api/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    data = await file.read()
    result = await analyze_skin_photo(data, file.content_type or "image/jpeg")
    return result


@app.post("/api/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    data = await file.read()
    result = await analyze_label_photo(data, file.content_type or "image/jpeg")
    return result


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
    guarded_out = check_output(result.content)
    return {
        "answer": guarded_out.text,
        "sources": [{"source": d.metadata.get("source"), "excerpt": d.page_content[:200]} for d in docs],
        "model_used": model_used,
    }


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
