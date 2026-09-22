"""Builds evals/golden_dataset.json.

30 hand-picked (phototype, UV, minutes_outside, minutes_since_last_spf,
sweating) scenarios chosen to cover: every Fitzpatrick phototype, low/
moderate/high/extreme UV, short/long exposure, overdue vs fresh
sunscreen, no-sunscreen-yet, and swimming/sweating. expected_risk_band
and expected_reapply_due are computed from the *same documented formula*
in app/graph/risk.py / SKILL.md — this is intentional: the dataset is a
regression guard on the formula's behavior, not an independent oracle
for it (the LLM-as-judge metric in run_evals.py is the independent
check, since it scores the free-text advice, not the formula).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent-service"))
from app.graph.risk import assess_risk  # noqa: E402

SCENARIOS = [
    # (id, note, phototype, uv, minutes_outside, minutes_since_spf, sweating)
    ("t1_low_uv_type1", "edge: lowest UV, most burn-prone skin", 1, 1, 30, None, False),
    ("t2_low_uv_type6", "edge: lowest UV, least burn-prone skin", 6, 1, 30, None, False),
    ("t3_moderate_type3_short", "typical short walk, moderate UV", 3, 4, 15, None, False),
    ("t4_moderate_type3_long", "typical long walk, moderate UV", 3, 4, 90, None, False),
    ("t5_high_uv_type2", "high UV, fair skin", 2, 7, 30, None, False),
    ("t6_high_uv_type5", "high UV, deep skin", 5, 7, 30, None, False),
    ("t7_extreme_uv_type1", "edge: extreme UV, most vulnerable skin", 1, 11, 20, None, False),
    ("t8_extreme_uv_type6", "edge: extreme UV, least vulnerable skin", 6, 11, 20, None, False),
    ("t9_very_high_uv_type4", "very high UV, medium skin", 4, 9, 40, None, False),
    ("t10_zero_uv", "edge: UV index 0 (night/heavy overcast)", 3, 0, 60, None, False),
    ("t11_boundary_uv3", "boundary: UV exactly 3 (moderate threshold)", 3, 3, 20, None, False),
    ("t12_boundary_uv6", "boundary: UV exactly 6 (high threshold)", 3, 6, 20, None, False),
    ("t13_boundary_uv8", "boundary: UV exactly 8 (very_high threshold)", 3, 8, 20, None, False),
    ("t14_boundary_uv11", "boundary: UV exactly 11 (extreme threshold)", 3, 11, 20, None, False),
    ("t15_spf_fresh", "sunscreen applied 10 min ago, not due", 2, 6, 30, 10, False),
    ("t16_spf_overdue", "sunscreen applied 130 min ago, overdue", 2, 6, 30, 130, False),
    ("t17_spf_exact_boundary", "sunscreen applied exactly 120 min ago", 2, 6, 30, 120, False),
    ("t18_no_spf_yet_low_uv", "no sunscreen applied, but UV is low", 2, 2, 30, None, False),
    ("t19_no_spf_yet_high_uv", "no sunscreen applied, UV is high", 2, 8, 30, None, False),
    ("t20_swimming_fresh", "swimming, applied 40 min ago (under 60 limit)", 3, 6, 30, 40, True),
    ("t21_swimming_overdue", "swimming, applied 70 min ago (over 60 limit)", 3, 6, 30, 70, True),
    ("t22_type1_long_exposure", "very burn-prone, long exposure, moderate UV", 1, 5, 120, None, False),
    ("t23_type6_long_exposure", "very sun-tolerant, long exposure, high UV", 6, 8, 180, None, False),
    ("t24_short_exposure_extreme_uv", "short exposure but extreme UV", 2, 12, 5, None, False),
    ("t25_long_exposure_low_uv", "long exposure but very low UV", 4, 1, 240, None, False),
    ("t26_type3_typical_commute", "typical commute scenario", 3, 5, 10, None, False),
    ("t27_type2_beach_day", "beach day, high UV, long exposure, swimming", 2, 9, 180, 30, True),
    ("t28_type5_midday_sport", "outdoor sport at midday, high UV", 5, 8, 60, None, False),
    ("t29_type4_cloudy_moderate", "cloudy but still moderate UV", 4, 3, 45, None, False),
    ("t30_type1_extreme_worst_case", "worst case: most vulnerable skin, extreme UV, long exposure, no SPF", 1, 13, 60, None, False),
]


def build():
    rows = []
    for sid, note, phototype, uv, minutes_outside, since_spf, sweating in SCENARIOS:
        result = assess_risk(
            phototype=phototype,
            uv_index=uv,
            minutes_outside=minutes_outside,
            minutes_since_last_spf=since_spf,
            sweating_or_swimming=sweating,
        )
        rows.append(
            {
                "id": sid,
                "note": note,
                "input": {
                    "phototype": phototype,
                    "uv_index": uv,
                    "minutes_outside": minutes_outside,
                    "minutes_since_last_spf": since_spf,
                    "sweating_or_swimming": sweating,
                },
                "expected_risk_band": result.risk_band,
                "expected_reapply_due": result.reapply_due,
            }
        )
    out_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} scenarios to {out_path}")


if __name__ == "__main__":
    build()
