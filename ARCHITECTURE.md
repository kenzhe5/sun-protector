# Architecture — Sun Protector

## 1. Problem and user

**User:** someone planning to be outside (walking, commuting, beach day) who wants
a plain answer to "is it currently safe for my skin, and how do I minimize sun
exposure on this specific walk" — instead of manually checking a UV widget,
remembering their own skin type, and guessing whether a shaded route exists.

**Not a medical device.** All advice is derived from public guidelines (WHO, AAD,
EPA, SkinCancer.org) plus a documented formula — not a dermatologist, not a
diagnosis. Every response carries a disclaimer (see `guardrails.py`).

## 2. Request flow (one user request, end to end)

```
Browser (frontend/app.js)
   │  POST /api/recommend {phototype, origin, destination?, minutes_outside, ...}
   ▼
FastAPI (agent-service/app/main.py)
   │  graph.ainvoke(input, thread_id)
   ▼
LangGraph workflow (agent-service/app/graph/graph.py)

  intake
    │
    ▼
  fetch_uv  ───────────────► MCP tool: get_uv_forecast_tool ───► Open-Meteo API
    │
    ▼
  assess_risk  (deterministic formula, app/graph/risk.py)
    │
    ├─[risk_band in {very_high, extreme}]──► urgent_advice ──► interrupt()
    │                                           │                  │
    │                                           │      (frontend shows confirm dialog,
    │                                           │       POST /api/resume {confirmed})
    │                                           ▼
    │                                     plan_route ◄───────────┘
    └─[else]───────────────────────────────────┤
                                                 ▼
                                    plan_route ──► MCP tool: get_shade_route_tool
                                                     (OSRM alternatives + OSM buildings
                                                      + solar position → shade_fraction)
                                                 │
                                                 ▼
                                          rag_answer  (if user asked a free-text question)
                                                 │       Chroma similarity_search over
                                                 │       WHO/AAD/EPA/SkinCancer.org corpus
                                                 ▼
                                       compose_response
                                          (Claude primary → OpenAI fallback,
                                           guardrails.check_output)
                                                 │
                                    ┌────────────┴─────────────┐
                              [walk still going, reapply due]  │ [else]
                                    ▼                          ▼
                             recheck_update ──► assess_risk   END
                             (loops back, capped at 1 cycle)
```

This single graph is the "multi-step workflow with conditional logic" required by
the course: **branching** (`risk_branch`), a **cycle** (`recheck_branch` →
`recheck_update` → back to `assess_risk`, capped at `MAX_RECHECKS=1` to keep the
demo bounded), and a **human-in-the-loop gate** (`urgent_advice`'s
`langgraph.types.interrupt()`, resumed via `Command(resume=...)` from
`/api/resume`).

## 3. Components and why each exists

| Component | What | Why this, not an alternative |
|---|---|---|
| **Orchestration** | LangGraph `StateGraph` | Needed explicit branching + a cycle + a pause/resume gate. CrewAI's role-based crews fit multi-agent delegation better than a single deterministic state machine; Parlant is optimized for conversational guardrail-heavy agents. LangGraph's graph model maps directly onto the risk→route→advice pipeline with the checkpointer giving human-in-the-loop "for free". |
| **MCP server** (`mcp-server/`) | Standalone MCP server, stdio transport, 3 tools: `get_uv_forecast`, `find_nearby_pharmacy`, `get_shade_route` | MCP vs. a plain internal function call: the tools are genuinely reusable outside this app (e.g. loadable directly into Claude Desktop) and the boundary forces a clean, versioned tool contract instead of ad hoc function imports. `get_shade_route` in particular does real work (OSRM + Overpass + a NOAA solar-position formula), which justifies a dedicated service boundary. |
| **RAG** | Chroma (local, file-based) + OpenAI `text-embedding-3-small`, recursive character chunking (800/100) | Qdrant/Pinecone need running infra or a hosted account for what is currently a ~20-chunk corpus — not worth the ops cost yet; Chroma is a drop-in swap later if the corpus grows. Embedding choice: Anthropic has no embeddings API, and `text-embedding-3-small` is cheap/good enough for short guideline prose. No reranker — corpus is small enough that top-k similarity is already precise; documented as a explicit simplification, not an oversight. |
| **Multimodality** | Vision (skin-tone photo → Fitzpatrick phototype; sunscreen label photo → OCR of SPF/ingredients/expiry), via the same Claude→OpenAI fallback path | Chosen because it feeds directly into the risk formula (phototype) and into a real guideline cross-check (label SPF vs. AAD's "SPF 30+" recommendation) — not a bolted-on demo feature. |
| **Skill** (`skills/sunscreen-reapplication-advisor/SKILL.md`) | Documents the exact formula in `risk.py` for an LLM to apply when a user asks a free-form question with no structured API call to make | The formula runs as plain Python inside the app (faster/cheaper/exact for arithmetic); the Skill exists for the case an LLM is reasoning conversationally and needs the same formula in natural language. |
| **LLM** | Primary: Claude (`claude-sonnet-5`, `claude-haiku-4-5` for cheap/fast judge calls). Fallback: `gpt-4o-mini`. | See `agent-service/app/config.py` docstring. Claude for the main synthesis step (safety-sensitive, instruction-following matters); Haiku for cheap structured/judge calls; OpenAI as the automatic fallback if Anthropic errors, rate-limits, or (as happened during this build — see EVALS.md) runs out of credit. `call_with_fallback()` wraps every LLM call in the app, including vision. |
| **Guardrails** | Hand-rolled regex-based input/output filters (`app/guardrails.py`) | Simple, auditable in one file for the defense, rather than pulling in a framework for ~5 patterns: blocks prompt-injection phrasing, redacts email/phone from inputs, blocks language that reads as a cancer diagnosis, and appends a "not a dermatologist" disclaimer to every output. |
| **Tracing** | LangSmith via env vars (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`) | No custom code needed — LangChain/LangGraph auto-instrument every LLM call and every graph node when these are set. |
| **Frontend** | One static page (`frontend/`, vanilla JS + Leaflet, no build step), served directly by FastAPI's `StaticFiles` | The course only requires "a real frontend, not CLI" — not a specific framework. A single HTML/JS/CSS bundle with no npm/webpack step means the whole app is one process (`uvicorn`) to run and one Docker image to build. Leaflet + OpenStreetMap needs no API key, unlike Google Maps JS. |

## 4. What's independent / replaceable vs. intentionally coupled

- **Independent, swappable:** the MCP server is a separate process reachable over
  stdio — it could be swapped for a hosted MCP server without touching the graph
  code beyond `mcp_client.py`. The vector store (`rag/retriever.py`) is isolated
  behind `retrieve()` — swapping Chroma for Qdrant/pgvector only touches that file.
  The LLM provider is isolated behind `config.call_with_fallback()` — swapping
  models means changing env vars, not code.
- **Deliberately coupled:** `risk.py`'s formula is inlined into `assess_risk_node`
  rather than exposed as an MCP tool or a separate microservice — it's a pure,
  fast, free function; wrapping it in a network boundary would add latency and
  failure modes for no benefit. The frontend talks to the backend same-origin
  (no separate API gateway) — appropriate for a single-team course project, not
  necessarily for a multi-team production system.

## 5. Known simplifications (said out loud, not hidden)

- `get_shade_route`'s shade scoring is a heuristic (building proximity + sun
  azimuth, not true ray-cast shadows from building height data) — documented in
  `mcp-server/tools/shade_route.py`'s docstring.
- The risk formula's baseline burn-times are commonly-cited dermatology teaching
  values, not a per-user calibrated model — real burn time varies with altitude,
  reflection (water/sand/snow), medication, and individual variation.
- RAG corpus is 4 documents / ~20 chunks — enough to demonstrate a real pipeline
  with real sources, not a production-scale knowledge base.
- No reranker, no semantic cache, no auth/roles, no CI pipeline — out of scope
  given the timeline; see the course's "recommended, not required" list.

## 6. Cost / latency / fallback

- A typical `/api/recommend` call: 1 MCP call (UV, ~0.3s), optionally 1 more (route,
  ~1-3s depending on Overpass load), 1 RAG retrieval (~0.1s, local), 1 LLM call for
  synthesis. Dominant cost/latency is the final LLM call.
- If the primary (Claude) LLM is unavailable for any reason, `call_with_fallback()`
  transparently retries on `gpt-4o-mini` and tags the response with which model
  actually answered (`model_used` field) — this is not hypothetical: it was
  exercised for real during this build when the Anthropic key ran out of credit
  (see EVALS.md).
