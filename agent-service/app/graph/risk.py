"""Deterministic sun-exposure risk scoring.

This module is the code implementation of the same formula documented in
skills/sunscreen-reapplication-advisor/SKILL.md. It's kept as plain,
testable Python (not an LLM call) because it's a numeric/lookup formula —
an LLM would be slower, costlier, and less reliable at arithmetic than a
lookup table + division. The Skill exists *in parallel* for the case
where a user asks a free-form question in chat ("if I'm type III and UV
is 7, when do I need to reapply?") and the agent should reason through
the formula in natural language rather than call a structured tool.

Reference baseline: approximate minutes-to-erythema (sunburn onset) at
UV index = 1, by Fitzpatrick skin phototype (commonly cited dermatology
teaching values, e.g. summarized in AAD/WHO patient education material).
Real-world burn time is influenced by many more factors (altitude,
reflection off snow/water/sand, medication, individual variation) — this
is an educational approximation, not a clinical instrument.
"""
from __future__ import annotations

from dataclasses import dataclass

# minutes to sunburn at UV index 1, by Fitzpatrick phototype (I-VI)
BASELINE_MINUTES_AT_UV1: dict[int, float] = {
    1: 67,
    2: 100,
    3: 200,
    4: 300,
    5: 400,
    6: 500,
}

REAPPLY_INTERVAL_MINUTES = 120  # standard "every 2 hours" sunscreen guidance
SWEAT_SWIM_REAPPLY_MINUTES = 60  # halved if swimming / heavy sweating


@dataclass
class RiskAssessment:
    phototype: int
    uv_index: float
    minutes_outside: int
    safe_exposure_minutes: float
    risk_ratio: float
    risk_band: str  # low | moderate | high | very_high | extreme
    reapply_due: bool
    minutes_until_reapply: float
    recommendation_summary: str


def assess_risk(
    phototype: int,
    uv_index: float,
    minutes_outside: int,
    minutes_since_last_spf: int | None = None,
    sweating_or_swimming: bool = False,
) -> RiskAssessment:
    phototype = max(1, min(6, phototype))
    uv_index = max(0.0, uv_index)

    baseline = BASELINE_MINUTES_AT_UV1[phototype]
    # time-to-burn scales ~inversely with UV intensity
    safe_exposure_minutes = baseline / uv_index if uv_index > 0 else float("inf")

    risk_ratio = (minutes_outside / safe_exposure_minutes) if safe_exposure_minutes != float("inf") else 0.0

    if uv_index >= 11 or risk_ratio >= 1.5:
        band = "extreme"
    elif uv_index >= 8 or risk_ratio >= 1.0:
        band = "very_high"
    elif uv_index >= 6 or risk_ratio >= 0.75:
        band = "high"
    elif uv_index >= 3 or risk_ratio >= 0.4:
        band = "moderate"
    else:
        band = "low"

    interval = SWEAT_SWIM_REAPPLY_MINUTES if sweating_or_swimming else REAPPLY_INTERVAL_MINUTES
    if minutes_since_last_spf is None:
        reapply_due = uv_index >= 3  # no SPF applied yet and UV is meaningful
        minutes_until_reapply = 0 if reapply_due else float("inf")
    else:
        minutes_until_reapply = max(0.0, interval - minutes_since_last_spf)
        reapply_due = minutes_until_reapply <= 0

    summaries = {
        "low": "Низкий риск — защита не обязательна для коротких прогулок.",
        "moderate": "Умеренный риск — рекомендуется SPF 30+, головной убор.",
        "high": "Высокий риск — обязателен SPF 30-50+, ищите тень, ограничьте время на солнце.",
        "very_high": "Очень высокий риск — SPF 50+, тень обязательна, минимизируйте пребывание на солнце.",
        "extreme": "Экстремальный риск — уйдите в тень/помещение как можно скорее.",
    }

    return RiskAssessment(
        phototype=phototype,
        uv_index=uv_index,
        minutes_outside=minutes_outside,
        safe_exposure_minutes=round(safe_exposure_minutes, 1) if safe_exposure_minutes != float("inf") else -1,
        risk_ratio=round(risk_ratio, 2),
        risk_band=band,
        reapply_due=reapply_due,
        minutes_until_reapply=round(minutes_until_reapply, 1) if minutes_until_reapply != float("inf") else -1,
        recommendation_summary=summaries[band],
    )
