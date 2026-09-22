"""Photo of skin -> estimated Fitzpatrick phototype (1-6).

Uses Claude vision with a structured-output prompt. This is the
multimodality requirement: the estimate feeds straight into risk.py's
formula (baseline burn-time lookup by phototype), so it's not a
decoration — without it the user would have to know their own
Fitzpatrick type, which almost nobody does by name.
"""
from __future__ import annotations

import base64
import json

from langchain_core.messages import HumanMessage

from .. import config

PROMPT = """You are estimating Fitzpatrick skin phototype (I-VI) from a photo of skin
(forearm/face), for sun-safety advice purposes only — NOT a medical or
cosmetic judgment. Fitzpatrick scale reference:
I: very pale, always burns, never tans. II: fair, burns easily, tans minimally.
III: light-medium, sometimes mild burn, tans gradually. IV: medium/olive, rarely burns, tans well.
V: brown, very rarely burns, tans darkly. VI: deeply pigmented, never burns.

Look at the skin tone in the image and respond with ONLY a JSON object:
{"phototype": <1-6 integer>, "confidence": <0-1 float>, "note": "<one short sentence>"}
"""


async def analyze_skin_photo(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": PROMPT},
            {
                "type": "image",
                "source_type": "base64",
                "data": b64,
                "mime_type": media_type,
            },
        ]
    )
    result, _model = await config.call_with_fallback([message], temperature=0.0, max_tokens=200)
    text = result.content if isinstance(result.content, str) else str(result.content)
    text = text.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"phototype": 3, "confidence": 0.0, "note": "parse_failed", "raw": text}
