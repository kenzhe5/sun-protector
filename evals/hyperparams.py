"""Подбор гиперпараметров LLM: temperature, max_tokens, top_p.

Один и тот же промпт (как в продакшене: предрасчитанный риск передаётся
модели) прогоняется по всему golden dataset (30 сценариев) с разными
настройками. Каждый сценарий — 2 раза, чтобы измерить стабильность.

Метрики для каждой конфигурации:
  - band_accuracy   — доля ответов, где назван правильный уровень риска
                      (детерминированная проверка по ключевым словам);
  - avg_judge_score — оценка LLM-судьи 1-5 (корректность + безопасность);
  - stability       — средняя похожесть двух ответов на один сценарий (0-1);
  - truncated_rate  — доля ответов, оборванных лимитом max_tokens;
  - avg_chars, avg_latency_s.

Запуск: python evals/hyperparams.py  ->  evals/results/hyperparams_results.json
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent-service"))
from app import config  # noqa: E402
from app.graph.risk import assess_risk  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "hyperparams_results.json")
REPEATS = 2
CONCURRENCY = 6

# (название, temperature, max_tokens, top_p)
CONFIGS = [
    ("t0.0", 0.0, 300, None),
    ("t0.2 (текущая)", 0.2, 300, None),
    ("t0.7", 0.7, 300, None),
    ("t1.0", 1.0, 300, None),
    ("t0.7 + top_p 0.5", 0.7, 300, 0.5),
    ("t0.2, max_tokens 120", 0.2, 120, None),
    ("t0.2, max_tokens 1024", 0.2, 1024, None),
]

PROMPT = """Ты — Sun Protector. Дан профиль: фототип {phototype}, UV индекс {uv},
{minutes_outside} минут уже на улице, крем нанесён {since_spf} мин назад ({sweating}).

Предрасчитанная риск-оценка (используй её, не пересчитывай риск сам):
risk_band={risk_band}, risk_ratio={risk_ratio}, safe_exposure_minutes={safe_exposure_minutes},
reapply_due={reapply_due}, minutes_until_reapply={minutes_until_reapply}.

Дай короткую рекомендацию (2-4 предложения) на основе этой оценки: назови уровень
риска словами (низкий / умеренный / высокий / очень высокий / экстремальный),
что делать, когда повторно наносить крем."""

JUDGE_PROMPT = """Score this sun-safety advice 1-5 for correctness+safety.
Scenario: phototype={phototype}, UV={uv}, minutes_outside={minutes_outside},
expected_risk_band={expected_band}. Advice: "{advice}"
Respond with ONLY the integer 1-5."""

# от самого сильного к слабому: «очень высокий» содержит «высок»
BAND_KEYWORDS = [
    ("extreme", "экстрем"),
    ("very_high", "очень высок"),
    ("high", "высок"),
    ("moderate", "умерен"),
    ("low", "низк"),
]


def detect_band(text: str) -> str | None:
    t = text.lower()
    for band, kw in BAND_KEYWORDS:
        if kw in t:
            return band
    return None


def make_llms(temperature: float, max_tokens: int, top_p: float | None):
    extra = {"top_p": top_p} if top_p is not None else {}
    primary = ChatAnthropic(
        model=config.PRIMARY_MODEL, api_key=config.ANTHROPIC_API_KEY,
        temperature=temperature, max_tokens=max_tokens, **extra,
    )
    fallback = ChatOpenAI(
        model=config.FALLBACK_MODEL, api_key=config.OPENAI_API_KEY,
        temperature=temperature, max_tokens=max_tokens, **extra,
    )
    return primary, fallback


def is_truncated(msg) -> bool:
    meta = msg.response_metadata or {}
    return meta.get("finish_reason") == "length" or meta.get("stop_reason") == "max_tokens"


async def generate(llms, messages) -> tuple:
    primary, fallback = llms
    start = time.monotonic()
    try:
        msg, model = await primary.ainvoke(messages), config.PRIMARY_MODEL
    except Exception:  # noqa: BLE001 — та же стратегия fallback, что в продакшене
        msg, model = await fallback.ainvoke(messages), config.FALLBACK_MODEL
    return msg, model, time.monotonic() - start


async def judge(case: dict, advice: str) -> int:
    inp = case["input"]
    res = await config.call_fast_with_fallback([HumanMessage(content=JUDGE_PROMPT.format(
        phototype=inp["phototype"], uv=inp["uv_index"], minutes_outside=inp["minutes_outside"],
        expected_band=case["expected_risk_band"], advice=advice,
    ))])
    digits = "".join(c for c in res.content if c.isdigit())
    return int(digits[:1]) if digits else 0


async def run_case(sem, llms, case: dict) -> dict:
    inp = case["input"]
    risk = assess_risk(
        phototype=inp["phototype"], uv_index=inp["uv_index"], minutes_outside=inp["minutes_outside"],
        minutes_since_last_spf=inp["minutes_since_last_spf"], sweating_or_swimming=inp["sweating_or_swimming"],
    )
    messages = [
        SystemMessage(content="Отвечай кратко, по-русски."),
        HumanMessage(content=PROMPT.format(
            phototype=inp["phototype"], uv=inp["uv_index"], minutes_outside=inp["minutes_outside"],
            since_spf=inp["minutes_since_last_spf"] or "ещё не наносил(а)",
            sweating="потеет/плавает" if inp["sweating_or_swimming"] else "обычная активность",
            risk_band=risk.risk_band, risk_ratio=risk.risk_ratio,
            safe_exposure_minutes=risk.safe_exposure_minutes, reapply_due=risk.reapply_due,
            minutes_until_reapply=risk.minutes_until_reapply,
        )),
    ]
    async with sem:
        outs = [await generate(llms, messages) for _ in range(REPEATS)]
        texts = [m.content for m, _, _ in outs]
        score = await judge(case, texts[0])
    print(f"  {case['id']}: {detect_band(texts[0])} / ожидалось {case['expected_risk_band']}", flush=True)
    return {
        "id": case["id"],
        "expected_band": case["expected_risk_band"],
        "detected_bands": [detect_band(t) for t in texts],
        "judge_score": score,
        "similarity": difflib.SequenceMatcher(None, texts[0], texts[1]).ratio(),
        "truncated": [is_truncated(m) for m, _, _ in outs],
        "chars": [len(t) for t in texts],
        "latency_s": [round(lat, 2) for _, _, lat in outs],
        "model": outs[0][1],
        "answer": texts[0],
    }


async def run():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    sem = asyncio.Semaphore(CONCURRENCY)
    summaries, all_rows = [], {}

    for name, temperature, max_tokens, top_p in CONFIGS:
        print(f"== {name}", flush=True)
        llms = make_llms(temperature, max_tokens, top_p)
        rows = await asyncio.gather(*(run_case(sem, llms, c) for c in dataset))
        n_answers = len(rows) * REPEATS
        summary = {
            "config": name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p if top_p is not None else 1.0,
            "band_accuracy": round(sum(b == r["expected_band"] for r in rows for b in r["detected_bands"]) / n_answers, 3),
            "avg_judge_score": round(sum(r["judge_score"] for r in rows) / len(rows), 2),
            "stability": round(sum(r["similarity"] for r in rows) / len(rows), 3),
            "truncated_rate": round(sum(t for r in rows for t in r["truncated"]) / n_answers, 3),
            "avg_chars": round(sum(c for r in rows for c in r["chars"]) / n_answers),
            "avg_latency_s": round(sum(x for r in rows for x in r["latency_s"]) / n_answers, 2),
            "model": rows[0]["model"],
        }
        summaries.append(summary)
        all_rows[name] = rows
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summaries, "rows": all_rows}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
