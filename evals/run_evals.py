"""Runs the golden dataset through the system and reports two metrics:

1. accuracy: does app/graph/risk.py's risk_band/reapply_due still match
   the golden dataset (regression guard on the formula).
2. llm_judge_score: for each scenario, generate a natural-language
   recommendation (same prompt style as compose_response) with the
   primary model, then ask a judge model to score 1-5 on correctness +
   safety, and average across the dataset.

Usage: python run_evals.py
Writes evals/results/results.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent-service"))
from app import config  # noqa: E402
from app.graph.risk import assess_risk  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "results.json")

ADVICE_PROMPT = """Ты — Sun Protector. Дан профиль: фототип {phototype}, UV индекс {uv},
{minutes_outside} минут уже на улице, крем нанесён {since_spf} мин назад ({sweating}).

Предрасчитанная риск-оценка (используй её, не пересчитывай риск сам):
risk_band={risk_band}, risk_ratio={risk_ratio}, safe_exposure_minutes={safe_exposure_minutes},
reapply_due={reapply_due}, minutes_until_reapply={minutes_until_reapply}.

Дай короткую рекомендацию (2-4 предложения) на основе этой оценки: текущий риск, что делать,
когда повторно наносить крем."""

JUDGE_PROMPT = """You are grading sun-safety advice for correctness and safety.
Scenario: phototype={phototype}, UV={uv}, minutes_outside={minutes_outside},
expected_risk_band={expected_band}, expected_reapply_due={expected_reapply}.
Advice given: "{advice}"

Score 1-5 (integer only, no other text):
5 = advice matches the expected risk level and reapply timing, gives safe actionable guidance
3 = roughly right direction but vague or missing the reapply point
1 = wrong/unsafe (e.g. says low risk when it should be high, or omits urgent shade advice for extreme risk)
Respond with ONLY the integer."""


async def run():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    accuracy_hits = 0
    judge_scores = []
    rows = []

    for case in dataset:
        inp = case["input"]
        result = assess_risk(
            phototype=inp["phototype"],
            uv_index=inp["uv_index"],
            minutes_outside=inp["minutes_outside"],
            minutes_since_last_spf=inp["minutes_since_last_spf"],
            sweating_or_swimming=inp["sweating_or_swimming"],
        )
        band_ok = result.risk_band == case["expected_risk_band"]
        reapply_ok = result.reapply_due == case["expected_reapply_due"]
        is_correct = band_ok and reapply_ok
        accuracy_hits += int(is_correct)

        advice_messages = [
            SystemMessage(content="Отвечай кратко и по делу."),
            HumanMessage(
                content=ADVICE_PROMPT.format(
                    phototype=inp["phototype"],
                    uv=inp["uv_index"],
                    minutes_outside=inp["minutes_outside"],
                    since_spf=inp["minutes_since_last_spf"] or "ещё не наносил(а)",
                    sweating="потеет/плавает" if inp["sweating_or_swimming"] else "обычная активность",
                    risk_band=result.risk_band,
                    risk_ratio=result.risk_ratio,
                    safe_exposure_minutes=result.safe_exposure_minutes,
                    reapply_due=result.reapply_due,
                    minutes_until_reapply=result.minutes_until_reapply,
                )
            ),
        ]
        advice_result, model_used = await config.call_with_fallback(advice_messages, max_tokens=200)
        advice_text = config.text_of(advice_result)

        judge_result = await config.call_fast_with_fallback(
            [
                HumanMessage(
                    content=JUDGE_PROMPT.format(
                        phototype=inp["phototype"],
                        uv=inp["uv_index"],
                        minutes_outside=inp["minutes_outside"],
                        expected_band=case["expected_risk_band"],
                        expected_reapply=case["expected_reapply_due"],
                        advice=advice_text,
                    )
                )
            ]
        )
        try:
            score = int("".join(c for c in config.text_of(judge_result) if c.isdigit())[:1] or "0")
        except ValueError:
            score = 0
        judge_scores.append(score)

        rows.append(
            {
                "id": case["id"],
                "formula_correct": is_correct,
                "computed_band": result.risk_band,
                "expected_band": case["expected_risk_band"],
                "advice_model": model_used,
                "advice": advice_text,
                "judge_score": score,
            }
        )
        print(f"{case['id']}: formula_ok={is_correct} judge={score}/5")

    summary = {
        "n": len(dataset),
        "accuracy": round(accuracy_hits / len(dataset), 3),
        "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else 0,
        "avg_judge_score_normalized": round((sum(judge_scores) / len(judge_scores)) / 5, 3) if judge_scores else 0,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
