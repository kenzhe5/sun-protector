---
name: sunscreen-reapplication-advisor
description: Calculates safe sun-exposure time and sunscreen reapplication timing from Fitzpatrick skin phototype, UV index, and time already spent outside. Use whenever the user asks about sunburn risk, "when should I reapply sunscreen", "how long can I stay in the sun", or gives skin type + UV index + time outside in a free-form question rather than through the app's structured /api/recommend call.
---

# Sunscreen Reapplication Advisor

This Skill is useful specifically for **free-form natural-language questions**
("I'm skin type II, UV is 7, I've been out for an hour, when do I need
sunscreen?") where there's no structured API call to make — the agent
should reason through the formula itself, in text, rather than call a tool.
For the app's own UI, the identical formula runs as deterministic code in
`agent-service/app/graph/risk.py` (faster and cheaper than an LLM call for
a pure arithmetic lookup) — this file documents that same formula so an
LLM can apply it correctly when talking to a user directly.

## Step 1 — Baseline minutes to sunburn at UV index 1, by Fitzpatrick phototype

| Phototype | Description | Baseline minutes (at UV=1) |
|---|---|---|
| I | Very pale, always burns, never tans | 67 |
| II | Fair, burns easily | 100 |
| III | Light-medium, sometimes burns | 200 |
| IV | Medium/olive, rarely burns | 300 |
| V | Brown, very rarely burns | 400 |
| VI | Deeply pigmented, almost never burns | 500 |

## Step 2 — Safe exposure time at the current UV index

```
safe_exposure_minutes = baseline_minutes / current_UV_index
```

(Burn time scales roughly inversely with UV intensity.)

## Step 3 — Risk ratio and band

```
risk_ratio = minutes_already_outside / safe_exposure_minutes
```

- ratio < 0.4 and UV < 3 → **low**
- ratio < 0.75 and UV < 6 → **moderate**
- ratio < 1.0 and UV < 8 → **high**
- ratio < 1.5 and UV < 11 → **very_high**
- otherwise → **extreme**

## Step 4 — Reapplication timing

- Standard interval: reapply sunscreen every **120 minutes**.
- If swimming or sweating heavily: reapply every **60 minutes**.
- `minutes_until_reapply = interval - minutes_since_last_application` (0 if none applied yet and UV ≥ 3 → reapply now).

## Step 5 — Respond

State plainly:
1. The risk band and what it means practically (seek shade? SPF level?).
2. How many more minutes they can safely stay out unprotected at this UV.
3. When they next need to reapply sunscreen (or apply it now).

Never phrase this as a medical diagnosis. Always allow that individual
factors (medication, altitude, reflection off water/sand/snow, skin
history) can change real risk — this is an educational estimate, not a
clinical instrument.

## Example

> Input: phototype II, UV index 7, 45 minutes outside already, sunscreen applied 100 minutes ago.
> safe_exposure_minutes = 100 / 7 ≈ 14.3
> risk_ratio = 45 / 14.3 ≈ 3.1 → UV 7 alone already puts this at **high**, and the ratio pushes it to **very_high**.
> minutes_until_reapply = 120 - 100 = 20 minutes.
> Response: "This is very-high risk for your skin type at UV 7 — you're well past your safe unprotected window. Move to shade now if possible, and your sunscreen is due for reapplication in about 20 minutes regardless."
