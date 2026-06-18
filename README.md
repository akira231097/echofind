# 🎙️ EchoFind

**A memory-aware, conversational RAG engine that finds the single best podcast clip or episode for a natural-language question.**

[![CI](https://github.com/akira231097/echofind/actions/workflows/ci.yml/badge.svg)](https://github.com/akira231097/echofind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/vector-Pinecone-111111)
![Cohere](https://img.shields.io/badge/rerank-Cohere-39594C)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

EchoFind is the conversational front door to a large podcast catalog. Ask it
something the way you'd ask a friend — *"what did that founder say about pricing
last month?"*, *"find the latest episode with the neuroscientist"*, *"who was
that guy again?"* — and it returns one precise audio/video **clip** (with exact
start/end timestamps) or a whole **episode**, plus a few one-tap follow-ups.

It is **not** keyword search. EchoFind routes intent, expands the query with
hypothetical documents, runs hybrid dense + sparse retrieval over time-aware
buckets, reranks, re-scores with metadata, and asks an LLM to pick the winning
clip — all while keeping a structured memory of the conversation so it can
resolve pronouns, follow-ups, and topic shifts across turns.

> ⚙️ Built with FastAPI + Server-Sent Events, Google Gemini, OpenAI embeddings,
> Pinecone hybrid search, Cohere reranking, and PostgreSQL. ~17k lines of Python.

> 📐 **[Architecture diagram & design deep-dive →](docs/DESIGN.md)**

---

## Table of contents

- [Why it's interesting](#why-its-interesting)
- [How it works](#how-it-works)
- [Standout engineering](#standout-engineering)
- [Performance](#performance)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [Testing](#testing)
- [Design notes & limitations](#design-notes--limitations)

---

## Why it's interesting

Most "chat with your data" demos are a single embedding lookup glued to an LLM.
EchoFind is a full **agentic retrieval pipeline** that tackles the problems that
actually show up in production conversational search:

| Problem | How EchoFind handles it |
|---|---|
| "Find a clip" vs "explain this" vs "find an episode" are different tasks | A 3-branch **LLM router** dispatches each query to a dedicated pipeline |
| Users speak in pronouns and follow-ups ("what else did *he* say?") | A deterministic **conversation memory** state machine resolves entities, topics, and "the other one" |
| The system keeps surfacing the *same* result | A 5-turn **exclusion window** hard-drops already-shown clips |
| "Latest" / "before the election" / "last month" need time reasoning | A **multi-bucket time-search planner** builds and fuses temporal buckets |
| One query rarely matches the best transcript wording | **HyDE** expansion + **multi-level Reciprocal Rank Fusion** over dense & sparse vectors |
| Semantic similarity alone misranks by recency/person/show | **Hybrid metadata-aware re-scoring** with intent-driven weight profiles |
| LLMs return malformed JSON | A **robustness ladder**: structured output → JSON mode → a hand-written JSON repairer |
| Latency matters | Selection **and** the memory update happen in a *single* LLM call; recommendations are pre-computed |

---

## How it works

### High-level architecture

```
                 ┌──────────────────────────┐
                 │   Web UI (web/index.html) │  single-file, no build step
                 └────────────┬─────────────┘
                              │  POST /api/chat/stream  (Server-Sent Events)
                 ┌────────────▼─────────────┐
                 │   FastAPI  (server.py)    │  streams staged progress + result
                 │   API routes (api/)       │
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │  Agent orchestrator       │  engine/agent.py
                 │  + conversation memory    │  engine/memory.py
                 └────────────┬─────────────┘
                              │  Stage 0: LLM ROUTER (engine/router.py)
            ┌─────────────────┼─────────────────────┐
            ▼                 ▼                     ▼
     ┌────────────┐   ┌───────────────┐    ┌──────────────────┐
     │ small_talk │   │ episode_search│    │   clip_search     │  ← default RAG
     │ (+grounding)│  │               │    │   full pipeline   │
     └────────────┘   └───────────────┘    └──────────────────┘
```

### The clip-search pipeline (the main event)

A single request — *"what did the neuroscientist say recently about sleep?"* —
flows through these stages, each streamed to the UI as an SSE progress event:

1. **Route** — Gemini classifies the query into `small_talk` / `episode_search` /
   `clip_search` with a confidence score (low confidence → safe fallback).
2. **Analyze** — resolve pronouns against memory, detect follow-ups, extract
   guests/hosts/show/time filters via a **gazetteer** (fuzzy entity index), and
   generate **HyDE** hypothetical transcript snippets — all in parallel.
3. **Embed** — one batched OpenAI `text-embedding-3-large` call (dense) plus
   **BM25 sparse** vectors, in parallel. HyDE vectors are weighted by their
   cosine-similarity rank to the original query.
4. **Retrieve** — a **multi-bucket time-search plan** (`latest` / `relative` /
   `between` / `before` / `after`, with pre/post-event splitting) runs each
   bucket in parallel against Pinecone's **dense + sparse** indexes, fused with
   **three-level RRF** (dense+sparse per query, then original+HyDE across queries,
   then weighted across buckets). Already-shown clips are hard-excluded.
5. **Hydrate** — fetch full rows (transcript, titles, speakers, timestamps,
   media URLs) from **PostgreSQL** and merge with vector scores.
6. **Rerank** — **Cohere rerank** over a constraint-enriched query.
7. **Re-score** — `apply_hybrid_metadata_scoring` blends semantic + date +
   person-match + show-match using a weight profile chosen from detected intent.
8. **Select** — Gemini reads the top candidates as XML documents, extracts
   supporting quotes *before* choosing, and returns the winning clip, a
   user-facing answer, a confidence score, **and the conversation-memory update —
   in one call.**
9. **Recommend** — the top-3 alternative clips are pre-computed and cached so a
   follow-up tap resolves instantly.

The `episode_search` and `small_talk` branches are analogous; `small_talk`
optionally uses Google Search **grounding** for explanatory answers.

For a much deeper walkthrough of every subsystem, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Standout engineering

These are the parts worth a closer look:

- **Single-call select + memory.** The selection step makes the LLM both pick the
  best chunk and emit the full structured memory update at once — deliberately
  trading one round-trip for ~1–2s of latency.
- **Deterministic memory as a separate source of truth.** Pronoun resolution,
  topic-drift detection, "the other one" disambiguation, and topic-thread
  tracking live in explicit, testable state machinery (`engine/memory.py`) — not
  left to the model. Per-component renderers tailor exactly what each LLM sees.
- **Multi-level Reciprocal Rank Fusion with weighted HyDE.** Three fusion stages:
  dense + sparse fused per query, then original + HyDE variants fused across
  queries (each HyDE document weighted by its cosine-similarity rank to the base
  query), then time buckets fused by bucket weight.
- **Time-aware multi-bucket retrieval.** A planner builds latest/oldest/relative/
  before/after/between buckets (and can regex-infer event anchors), each
  independently filtered, fused, and re-weighted.
- **Intent-driven metadata re-scoring.** One of several weight profiles
  (pure-recency / recency+topic / person-focused / show-focused / standard) is
  selected per query, with tiered person matching and multiplicative penalties.
- **A robustness ladder for unreliable LLM JSON:** structured output → JSON-object
  mode → a hand-written character-state-machine JSON repairer.
- **Pervasive concurrency & graceful degradation:** batched embeddings,
  `asyncio.gather` over HyDE calls / buckets / branches, exponential-backoff
  retries, and fallbacks at every stage (optional grounding, entity-filter and
  recall fallbacks, per-stage fallback selections).

---

## Performance

A design goal is that the **deterministic memory layer is never the bottleneck** —
LLM calls should dominate latency, not bookkeeping. A reproducible micro-benchmark
([`benchmarks/memory_bench.py`](benchmarks/memory_bench.py)) measures the pure-Python
per-turn cost at a realistic steady state (~50 turns of history), on commodity hardware:

| Operation (per turn) | p50 | p95 |
| --- | --- | --- |
| Full memory update (entity decay, topic thread, exclusion window, compression) | ~0.04 ms | ~0.14 ms |
| Render router context | ~0.005 ms | ~0.008 ms |
| Render query-analyzer context | ~0.009 ms | ~0.027 ms |
| Render full memory prompt | ~0.006 ms | ~0.018 ms |
| Exclusion-window lookup | <0.001 ms | <0.001 ms |

The entire memory layer costs **well under 0.2 ms per turn** — three to four orders
of magnitude below a single model round-trip. Reproduce: `python benchmarks/memory_bench.py`.

---

## Tech stack

| Layer | Technology |
|---|---|
| API & streaming | **FastAPI**, **Uvicorn**, **Server-Sent Events** (`sse-starlette`) |
| Frontend | Vanilla HTML/CSS/JS — single file, no build step |
| LLM | **Google Gemini** (`gemini-3-flash-preview` + `gemini-2.5-flash-lite`) via OpenAI-compatible API; `google-genai` for Search grounding |
| Embeddings | **OpenAI** `text-embedding-3-large` (dense) |
| Sparse | **BM25** via `pinecone-text` |
| Vector search | **Pinecone** hybrid (dense + sparse indexes) |
| Reranking | **Cohere rerank** via Pinecone inference |
| Relational store | **PostgreSQL** (clip + episode metadata) via `psycopg2` |
| Object storage | **AWS S3** (BM25 model) via `boto3` |
| Retrieval techniques | HyDE, multi-level RRF, fuzzy matching (`rapidfuzz`/`thefuzz`) |
| Validation | **Pydantic v2** structured outputs |
| Concurrency | `asyncio` throughout; in-RAM session memory |

---

## Project structure

```
EchoFind/
├── server.py                 # FastAPI app: clients, lifespan, CORS, serves the UI
├── config.py                 # All settings via env vars (no secrets committed)
├── run_local.py              # Local dev server launcher
├── api/
│   └── routes.py             # /chat, /chat/stream (SSE), recommendations, sessions
├── engine/                   # The agentic RAG engine
│   ├── agent.py              # Orchestrator: routing, the clip pipeline, fusion, scoring
│   ├── router.py             # 3-branch LLM query router
│   ├── query_analyzer.py     # Pronoun resolution, entity/time extraction, HyDE
│   ├── selection.py          # Single-call clip selection + memory update
│   ├── episode_search.py     # Episode-level retrieval branch
│   ├── recommendations.py    # Pre-computed follow-up clips
│   ├── episode_recommendations.py
│   ├── small_talk.py         # Greetings / explanations (+ optional grounding)
│   ├── memory.py             # Deterministic conversation-memory state machine
│   └── schemas.py            # Pydantic request/response/memory schemas
├── retrieval/                # Search & data-access layer
│   ├── data_fetcher.py       # Pinecone search, RRF, PostgreSQL hydration
│   ├── search.py             # Cohere reranking
│   ├── search_filter.py      # Metadata + date filter construction
│   ├── sparse_encoder.py     # BM25 sparse encoding (graceful default fallback)
│   └── gazetteer.py          # Fast fuzzy entity lookup (hosts/guests/shows)
├── web/
│   └── index.html            # Streaming chat UI
├── data/
│   └── entities.sample.json  # Synthetic catalog for local dev/demo
├── tests/                    # Memory-behavior verification scripts
├── docs/
│   └── ARCHITECTURE.md       # Deep-dive design documentation
├── .env.example              # Copy to .env and fill in
└── requirements.txt
```

---

## Getting started

EchoFind talks to several managed services (Gemini, OpenAI, Pinecone, Cohere
via Pinecone, PostgreSQL). To run it end-to-end you need accounts/keys for those
and a populated index + database. The code, structure, and pipeline are fully
readable without them.

```bash
# 1. Clone & enter
git clone https://github.com/akira231097/echofind.git
cd echofind

# 2. (Recommended) create a virtualenv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env        # then fill in your keys

# 5. Run the dev server
python run_local.py
#   → UI:        http://localhost:8000
#   → API docs:  http://localhost:8000/docs
#   → Health:    http://localhost:8000/api/health
```

> The BM25 sparse encoder gracefully falls back to a default model if no trained
> model is available, so the service still boots without the (proprietary) index.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Non-streaming chat (full response) |
| `POST` | `/api/chat/stream` | **Streaming** chat via Server-Sent Events |
| `POST` | `/api/recommendation/click` | Resolve a pre-computed clip recommendation |
| `POST` | `/api/episode-recommendation/click` | Resolve an episode recommendation |
| `POST` | `/api/session/reset` | Clear a session's memory |
| `GET`  | `/api/session/{id}` | Session info (turns, entities, themes) |
| `DELETE` | `/api/session/{id}` | Delete a session |
| `GET`  | `/api/sessions` | List active sessions |
| `POST` | `/api/cleanup` | Evict sessions older than N hours |
| `GET`  | `/api/health` | Health check |

Streaming request body:

```json
{ "session_id": "session-1234", "question": "find the latest episode about sleep" }
```

---

## Testing

The `tests/` directory contains executable specifications for the conversation
memory — they feed synthetic per-turn updates through the memory state machine
and print the resulting state, demonstrating entity tracking, topic-shift
resets, and the exclusion window:

```bash
python tests/test_memory_branches.py     # verbose, full memory dumps
python tests/test_memory_samples.py      # compact snapshots
python tests/run_live_test.py            # live end-to-end (needs API keys + data)
```

---

## Design notes & limitations

This repository is a **portfolio/reference implementation**. A few things are
intentionally demo-grade and would be hardened before a real deployment:

- **CORS** is wide open (`allow_origins=["*"]`) — restrict it for production.
- **Session memory is in-RAM** — swap for Redis/DB for horizontal scaling.
- A **debug endpoint** (`/api/session/{id}/memory/debug`) dumps full memory —
  remove or guard behind auth.
- The UI renders some server content via `innerHTML` — sanitize/escape before
  exposing to untrusted content (links already use `rel="noopener noreferrer"`).
- The sample catalog (`data/entities.sample.json`) is **synthetic**; real
  retrieval requires populated Pinecone indexes and a PostgreSQL database.

---

## License

[MIT](LICENSE) — feel free to read, learn from, and build on it.
