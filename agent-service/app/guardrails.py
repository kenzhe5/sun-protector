"""Lightweight, dependency-free guardrails (bonus requirement 4).

Deliberately simple / hand-rolled rather than pulling in Guardrails AI or
NeMo Guardrails, so the logic is auditable in one file for the defense.
Two layers:

  1. Input filtering: block obvious prompt-injection attempts aimed at the
     system prompt, and redact PII (emails, phone numbers) before it's
     sent to an LLM provider or logged to LangSmith.
  2. Output filtering: this app gives health-adjacent advice (sun/skin),
     so we hard-block outputs that read as a medical diagnosis ("you have
     skin cancer") and always keep the "not a dermatologist" disclaimer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|the) (previous|prior|above) instructions", re.I),
    re.compile(r"disregard (your|the) (system|previous) prompt", re.I),
    re.compile(r"you are now (in )?(dan|developer mode|jailbreak)", re.I),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.I),
    re.compile(r"act as if you have no (restrictions|rules|guardrails)", re.I),
]

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")

_DIAGNOSIS_PATTERNS = [
    re.compile(r"\byou (definitely |certainly )?have (skin )?cancer\b", re.I),
    re.compile(r"\bthis is (definitely |certainly )?melanoma\b", re.I),
    re.compile(r"\bI diagnose you with\b", re.I),
]

DISCLAIMER = (
    "\n\n_Это информационный ассистент на основе публичных гайдлайнов "
    "(WHO/AAD и др.), а не диагностика и не замена консультации дерматолога. "
    "При подозрительных родинках/новообразованиях — обратитесь к врачу._"
)


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    flags: list[str] = field(default_factory=list)


def check_input(text: str) -> GuardrailResult:
    flags = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append(f"prompt_injection:{pattern.pattern[:30]}")

    redacted = _EMAIL_RE.sub("[email redacted]", text)
    redacted = _PHONE_RE.sub("[phone redacted]", redacted)
    if redacted != text:
        flags.append("pii_redacted")

    allowed = not any(f.startswith("prompt_injection") for f in flags)
    return GuardrailResult(allowed=allowed, text=redacted, flags=flags)


def check_output(text: str, *, add_disclaimer: bool = True) -> GuardrailResult:
    flags = []
    safe_text = text
    for pattern in _DIAGNOSIS_PATTERNS:
        if pattern.search(text):
            flags.append(f"blocked_diagnosis_claim:{pattern.pattern[:30]}")
            safe_text = pattern.sub(
                "this may need a dermatologist's evaluation (I can't diagnose)", safe_text
            )

    if add_disclaimer and DISCLAIMER.strip() not in safe_text:
        safe_text = safe_text.rstrip() + DISCLAIMER

    return GuardrailResult(allowed=True, text=safe_text, flags=flags)
