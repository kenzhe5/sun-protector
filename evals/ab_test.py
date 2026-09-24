"""A/B test: prompt v1 vs v2 for the advice-generation step.

Backstory (this is a real finding, not a staged one): the first version
of the advice prompt gave the LLM only the raw inputs (phototype, UV,
minutes outside, etc.) and asked it to reason about sunburn risk itself.
Running it over the golden dataset (see run_evals.py's first draft)
produced visibly wrong risk framing in several cases — e.g. UV 7 / high
exposure called "umeренный" (moderate) — because the model was doing
ad hoc arithmetic instead of using the app's own risk.py formula.

v2 instead passes the *precomputed* deterministic risk assessment
(risk_band, risk_ratio, reapply_due, ...) from app/graph/risk.py into
the prompt and asks the model to phrase it, not compute it — which is
what app/graph/nodes.py:compose_response actually does in production.

This script quantifies that decision: same model both times, only the
prompt changes, scored by the same LLM-as-judge rubric as run_evals.py.

Usage: python ab_test.py
Writes evals/results/ab_test_results.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent-service"))
from app import config  # noqa: E402
from app.graph.risk import assess_risk  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "ab_test_results.json")

PROMPT_V1 = """Ты — Sun Protector. Дан профиль: фототип {phototype}, UV индекс {uv},
{minutes_outside} минут уже на улице, крем нанесён {since_spf} мин назад ({sweating}).
Дай короткую рекомендацию (2-4 предложения): текущий риск, что делать, когда повторно наносить крем."""

PROMPT_V2 = """Ты — Sun Protector. Дан профиль: фототип {phototype}, UV индекс {uv},
{minutes_outside} минут уже на улице, крем нанесён {since_spf} мин назад ({sweating}).

Предрасчитанная риск-оценка (используй её, не пересчитывай риск сам):
risk_band={risk_band}, risk_ratio={risk_ratio}, safe_exposure_minutes={safe_exposure_minutes},
reapply_due={reapply_due}, minutes_until_reapply={minutes_until_reapply}.

Дай короткую рекомендацию (2-4 предложения) на основе этой оценки: текущий риск, что делать,
когда повторно наносить крем."""

JUDGE_PROMPT = """Score this sun-safety advice 1-5 for correctness+safety.
Scenario: phototype={phototype}, UV={uv}, minutes_outside={minutes_outside},
expected_risk_band={expected_band}. Advice: "{advice}"
Respond with ONLY the integer 1-5."""


async def _score_prompt(prompt_template: str, case: dict, risk) -> dict:
    inp = case["input"]
    start = time.monotonic()
    try:
        result, model_used = await config.call_with_fallback(
            [
                SystemMessage(content="Отвечай кратко."),
                HumanMessage(
                    content=prompt_template.format(
                        phototype=inp["phototype"],
                        uv=inp["uv_index"],
                        minutes_outside=inp["minutes_outside"],
                        since_spf=inp["minutes_since_last_spf"] or "ещё не наносил(а)",
                        sweating="потеет/плавает" if inp["sweating_or_swimming"] else "обычная активность",
                        risk_band=risk.risk_band,
                        risk_ratio=risk.risk_ratio,
                        safe_exposure_minutes=risk.safe_exposure_minutes,
                        reapply_due=risk.reapply_due,
                        minutes_until_reapply=risk.minutes_until_reapply,
                    )
                ),
            ],
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001 - a variant failing outright is a valid A/B outcome
        return {"latency_s": None, "judge_score": 0, "advice": None, "error": str(exc)[:200]}
    latency = time.monotonic() - start

    judge_result = await config.call_fast_with_fallback(
        [
            HumanMessage(
                content=JUDGE_PROMPT.format(
                    phototype=inp["phototype"],
                    uv=inp["uv_index"],
                    minutes_outside=inp["minutes_outside"],
                    expected_band=case["expected_risk_band"],
                    advice=config.text_of(result),
                )
            )
        ]
    )
    try:
        score = int("".join(c for c in config.text_of(judge_result) if c.isdigit())[:1] or "0")
    except ValueError:
        score = 0
    return {"latency_s": round(latency, 2), "judge_score": score, "advice": config.text_of(result), "model_used": model_used}


async def run():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    results_a, results_b = [], []
    for case in dataset:
        inp = case["input"]
        risk = assess_risk(
            phototype=inp["phototype"],
            uv_index=inp["uv_index"],
            minutes_outside=inp["minutes_outside"],
            minutes_since_last_spf=inp["minutes_since_last_spf"],
            sweating_or_swimming=inp["sweating_or_swimming"],
        )
        results_a.append(await _score_prompt(PROMPT_V1, case, risk))
        results_b.append(await _score_prompt(PROMPT_V2, case, risk))
        print(f"{case['id']}: v1(raw)={results_a[-1]['judge_score']} v2(risk-injected)={results_b[-1]['judge_score']}")

    def summarize(name, rows):
        n = len(rows)
        latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
        errors = sum(1 for r in rows if r.get("error"))
        return {
            "variant": name,
            "n": n,
            "errors": errors,
            "avg_judge_score": round(sum(r["judge_score"] for r in rows) / n, 2),
            "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        }

    summary_a = summarize("v1_raw_inputs", results_a)
    summary_b = summarize("v2_risk_injected", results_b)

    print("\n=== A/B SUMMARY ===")
    print(json.dumps({"v1_raw_inputs": summary_a, "v2_risk_injected": summary_b}, indent=2))

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": {"v1_raw_inputs": summary_a, "v2_risk_injected": summary_b}, "rows_v1": results_a, "rows_v2": results_b},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
