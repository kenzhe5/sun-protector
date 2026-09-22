"""Photo of a sunscreen label -> extracted SPF / broad-spectrum / expiry via OCR+vision.

Cross-checked in app/main.py against the AAD/skincancer.org guidance in
the RAG corpus (e.g. "SPF 30+ broad-spectrum recommended") so the agent
can tell the user if their actual product is under-protective.
"""
from __future__ import annotations

import base64
import json

from langchain_core.messages import HumanMessage

from .. import config

PROMPT = """This is a photo of a sunscreen product label. Read the label (OCR) and
extract what you can. Respond with ONLY a JSON object:
{"spf": <integer or null>, "broad_spectrum": <true/false/null>,
 "water_resistant": <true/false/null>, "expiry_date": "<string or null>",
 "active_ingredients": ["..."], "raw_text_snippet": "<short excerpt of what you read>"}
If the image isn't a sunscreen label, set all fields to null and note that in raw_text_snippet.
"""


async def analyze_label_photo(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
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
    result, _model = await config.call_with_fallback([message], temperature=0.0, max_tokens=400)
    text = result.content if isinstance(result.content, str) else str(result.content)
    text = text.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"spf": None, "raw_text_snippet": text, "note": "parse_failed"}
