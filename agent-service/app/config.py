"""Environment/config loading + LLM client factories.

LLM choice (documented in ARCHITECTURE.md in full):
  - Primary: Anthropic Claude (claude-sonnet-5) — used for the final
    recommendation synthesis and vision analysis, where reasoning quality
    and instruction-following on safety-sensitive advice matters most.
  - Cheap/fast path: claude-haiku-4-5 — used for lightweight structured
    parsing (intent extraction) where latency/cost matter more than depth.
  - Fallback: OpenAI gpt-4o-mini — used automatically if the Anthropic
    call fails/times out/rate-limits, and used as the "B" arm in the
    A/B experiment (see evals/ab_test.py).
  - Embeddings: OpenAI text-embedding-3-small — Anthropic has no first
    party embeddings API, and this model is cheap + good enough for a
    guideline-document corpus of this size.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

# repo_root/.env (one level above agent-service/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(os.path.dirname(_ROOT), ".env"))  # no-op safety
load_dotenv()  # also allow agent-service/.env or process env


def _clean(v: str | None) -> str | None:
    if v is None:
        return None
    return v.strip().strip('"').strip("'")


ANTHROPIC_API_KEY = _clean(os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "claude-sonnet-5")
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-4-5-20251001")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# temperature=0 — для моделей, которые её принимают (fallback gpt-4o-mini,
# судья Claude Haiku 4.5): по эксперименту evals/hyperparams.py качество как
# при 0.2 (судья 4.17 vs 4.20 — в пределах шума), а стабильность ответов выше
# (0.79 vs 0.60). У основной модели Claude Sonnet 5 temperature/top_p/top_k
# убраны самим Anthropic (запрос с ними — ошибка 400), поэтому ей их не
# передаём. Подробности — EVALS.md.
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma")
CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus")

MCP_SERVER_CMD = os.getenv("MCP_SERVER_CMD", "python")
MCP_SERVER_ARGS = os.getenv("MCP_SERVER_ARGS", os.path.join(_ROOT, "mcp-server", "server.py"))


@lru_cache
def get_primary_llm(temperature: float | None = None, max_tokens: int | None = None):
    from langchain_anthropic import ChatAnthropic

    # Sonnet 5: без temperature (не поддерживается) и без размышлений —
    # ответы короткие, риск считает формула, а размышления добавляют задержку.
    return ChatAnthropic(
        model=PRIMARY_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        thinking={"type": "disabled"},
    )


@lru_cache
def get_fast_llm(temperature: float | None = None, max_tokens: int | None = None):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=FAST_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens or 512,
    )


@lru_cache
def get_fallback_llm(temperature: float | None = None, max_tokens: int | None = None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=FALLBACK_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
    )


@lru_cache
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


async def call_with_fallback(messages, temperature: float | None = None, max_tokens: int | None = None):
    """Invoke the primary (Claude) model; on any error, fall back to OpenAI.

    This is the "fallback strategy between models" recommended requirement:
    if Anthropic is down, rate-limited, or the key is missing/invalid, the
    user still gets an answer instead of a 500.
    """
    primary = get_primary_llm(temperature, max_tokens)
    try:
        return await primary.ainvoke(messages), PRIMARY_MODEL
    except Exception as primary_exc:  # noqa: BLE001 - intentional broad fallback
        try:
            fallback = get_fallback_llm(temperature, max_tokens)
            result = await fallback.ainvoke(messages)
            return result, f"{FALLBACK_MODEL} (fallback, primary_error={primary_exc.__class__.__name__})"
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Both primary and fallback LLMs failed: {primary_exc} / {fallback_exc}"
            ) from fallback_exc


async def call_fast_with_fallback(messages, temperature: float = 0.0, max_tokens: int = 8):
    """Same fallback pattern as call_with_fallback, but for the cheap/fast
    judge-style calls (evals, A/B judge) — falls back Claude Haiku -> gpt-4o-mini."""
    primary = get_fast_llm(temperature, max_tokens)
    try:
        return await primary.ainvoke(messages)
    except Exception:  # noqa: BLE001
        fallback = get_fallback_llm(temperature, max_tokens)
        return await fallback.ainvoke(messages)


def text_of(message) -> str:
    """Текст ответа модели: content бывает строкой или списком блоков."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))
