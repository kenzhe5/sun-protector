# ☀️ Sun Protector

An AI assistant for sun safety: tells you the current UV risk for your skin type,
finds the shadiest walking route to where you're going, and reminds you when to
reapply sunscreen — grounded in public dermatology guidelines (WHO, AAD, EPA,
SkinCancer.org), not a diagnosis.

Built as a course capstone project (see `final_task.md` / `my_project.md` for the
original brief). Full technical rationale is in [ARCHITECTURE.md](ARCHITECTURE.md);
eval methodology and results are in [EVALS.md](EVALS.md).

![Recommendation screenshot](docs/screenshot-recommendation.png)
![Human-in-the-loop confirmation](docs/screenshot-confirm.png)

## What it does

- **Risk assessment**: Fitzpatrick skin phototype + live UV index (Open-Meteo) →
  deterministic risk score (low/moderate/high/very_high/extreme) and reapply timer.
- **Shade-aware routing**: given a destination, ranks walking-route alternatives
  by estimated shade coverage (OSRM routing + OpenStreetMap building data + real
  solar position — not just shortest distance).
- **Photo → phototype**: upload a skin photo, get an estimated Fitzpatrick type.
- **Photo → label check**: upload a sunscreen label photo, get its SPF/ingredients
  read via OCR and checked against guideline recommendations.
- **Guideline Q&A**: ask a free-text question, answered via RAG over a real
  WHO/AAD/EPA/SkinCancer.org corpus with cited sources.
- **Urgent-risk confirmation**: at very-high/extreme risk, the workflow pauses
  and asks you to confirm before "sending" an urgent notification
  (human-in-the-loop).

## Course requirement checklist

| Requirement | Where |
|---|---|
| LangGraph multi-step workflow (branching, cycle, human-in-the-loop) | `agent-service/app/graph/` |
| MCP server, 3 tools | `mcp-server/` |
| Skill (SKILL.md) | `skills/sunscreen-reapplication-advisor/SKILL.md` |
| RAG pipeline (chunking, embeddings, vector DB) | `agent-service/app/rag/` |
| Document/web scraping for RAG corpus | `agent-service/data/corpus/` (fetched from WHO/AAD/EPA/SkinCancer.org) |
| Multimodality (vision) | `agent-service/app/multimodal/` |
| LangSmith tracing | `agent-service/app/tracing.py` (set `LANGCHAIN_API_KEY` in `.env`) |
| Golden dataset (30 examples) + evals | `evals/golden_dataset.json`, `evals/run_evals.py`, [EVALS.md](EVALS.md) |
| A/B experiment | `evals/ab_test.py`, [EVALS.md](EVALS.md) |
| LLM + hyperparameter choice, documented | [ARCHITECTURE.md](ARCHITECTURE.md) §3, `app/config.py` |
| Guardrails (bonus) | `agent-service/app/guardrails.py` |
| Fallback between models (bonus) | `app/config.py:call_with_fallback` — exercised for real, see EVALS.md |
| Docker / docker-compose (bonus) | `Dockerfile`, `docker-compose.yml` |
| Web frontend | `frontend/` (static, no build step) |

## Run it

### Option A — Docker (simplest)

```bash
cp .env.example .env   # then fill in CLAUDE_API_KEY / OPENAI_API_KEY
docker compose up --build
# first time only, in another shell: ingest the RAG corpus
docker compose exec sun-protector python -m app.rag.ingest
```

Open http://localhost:8000

### Option B — local

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r agent-service/requirements.txt -r mcp-server/requirements.txt

cp .env.example .env   # then fill in CLAUDE_API_KEY / OPENAI_API_KEY

cd agent-service
python -m app.rag.ingest        # one-time: embed the guideline corpus into Chroma
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 — the same FastAPI process serves both the API and
the static frontend. Allow location access for the map to center on you, or
just click anywhere on the map to set your start point.

### Run evals / A/B test

```bash
cd evals
python run_evals.py     # -> results/results.json
python ab_test.py       # -> results/ab_test_results.json
```

### Use the MCP server standalone (e.g. from Claude Desktop)

```json
{
  "mcpServers": {
    "sun-protector": {
      "command": "python",
      "args": ["/absolute/path/to/mcp-server/server.py"]
    }
  }
}
```

## Manual verification performed during this build

Every path below was exercised against the live app (not just unit-tested) while
building this: MCP tool calls over the real stdio protocol (UV forecast, pharmacy
search, shade routing — all against live Open-Meteo/OSRM/OpenStreetMap APIs);
the full LangGraph workflow including the human-in-the-loop `interrupt()`/resume
cycle; RAG retrieval returning correctly-cited guideline chunks; the automatic
Claude→OpenAI fallback (triggered for real — the Anthropic key ran out of credit
mid-build, see EVALS.md); guardrails blocking a prompt-injection attempt; and the
frontend end-to-end via a real browser (map render, route drawing, confirm
dialog) — screenshots above are from that session, not mockups.

## Known limitations

- Shade routing is a heuristic (sun position + building proximity), not true
  shadow ray-casting — see `mcp-server/tools/shade_route.py`.
- Risk formula uses commonly-cited dermatology teaching baselines, not a
  clinically calibrated per-person model — informational, not diagnostic.
- No auth/roles, no CI pipeline, no production deployment — out of scope for
  the course timeline (see ARCHITECTURE.md §5 for the full list, said out loud
  rather than hidden).
- **Security note**: this repo's `.env` at one point contained a live API key
  pasted directly into a chat session; that key should be treated as
  compromised and rotated before any real use. `.env` is gitignored.

## Repo layout

```
agent-service/   FastAPI app, LangGraph workflow, RAG, multimodal, config
mcp-server/      Standalone MCP server (3 tools)
skills/          Claude Skill (SKILL.md)
frontend/        Static web UI (no build step)
evals/           Golden dataset, eval runner, A/B test
docs/            Screenshots
ARCHITECTURE.md  Design rationale, request-flow diagram, trade-offs
EVALS.md         Metrics, golden dataset rationale, A/B results
```
