# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-17

Initial public release of **EchoFind** — a memory-aware, conversational RAG engine
that returns the single best podcast clip or episode for a natural-language question.

### Added

- **FastAPI + SSE service** (`server.py`, `api/routes.py`): streaming `POST /api/chat/stream`
  endpoint that emits staged pipeline progress and a final result via Server-Sent Events,
  plus a non-streaming `POST /api/chat` endpoint. Serves a single-file, build-step-free web UI.
- **3-branch LLM router** (`engine/router.py`): classifies each query into `small_talk`,
  `episode_search`, or `clip_search` with a confidence score and a safe low-confidence fallback.
- **Clip-search RAG pipeline** (`engine/agent.py`): route → analyze → embed → retrieve →
  hydrate → rerank → re-score → select → recommend, streamed stage-by-stage to the UI.
- **Query analysis with memory** (`engine/query_analyzer.py`): pronoun/follow-up resolution,
  gazetteer-based fuzzy entity extraction (guests/hosts/shows), time-filter detection, and
  HyDE hypothetical-transcript generation (including person-specific angles), run concurrently.
- **Multi-level Reciprocal Rank Fusion** (`retrieval/data_fetcher.py`): a single `rrf_fuse_lists`
  primitive (RRF_K=60) applied at three levels — dense+sparse per query, original+HyDE across
  queries, and across time buckets — with coalesced retrieval-source labels.
- **Weighted HyDE**: each HyDE document is weighted by the cosine similarity rank of its
  embedding to the base query, with the original query weighted highest.
- **Time-aware multi-bucket retrieval** (`make_time_search_plan`): latest / relative-recent /
  between / before / after planning, pre/post-event splitting (with US-election anchor inference),
  per-bucket date gating, and per-bucket RRF weights; buckets execute in parallel.
- **Two-stage hybrid search**: dense (OpenAI `text-embedding-3-large`) + BM25 sparse over
  Pinecone, with dynamic score thresholds, filtered/recall budgets, and relaxed-recall fallbacks.
- **PostgreSQL hydration** (`batch_get_rds_items`): fetches full transcript, titles, speakers,
  timestamps, and media (audio + MP4/HLS video) URLs and merges them with vector scores.
- **Cohere reranking** (`retrieval/search.py`) via Pinecone inference over a constraint-enriched query.
- **Intent-driven hybrid metadata scoring** (`apply_hybrid_metadata_scoring`): weighted blend of
  semantic, date, person-match, and show-match signals across selectable weight profiles
  (pure-recency / recency+topic / person-focused / show-focused / standard), with tiered person
  matching, episode-diversity caps, and multiplicative penalties.
- **Single-call select + memory** (`engine/selection.py`): one Gemini call that extracts quotes,
  selects the best clip, writes the user-facing answer and confidence, and emits the conversation
  memory update — guarded by a robustness ladder (structured output → JSON-object mode → a
  hand-written JSON repairer).
- **Deterministic conversation memory** (`engine/memory.py`): entity/topic tracking, follow-up
  and topic-drift handling, and a 5-turn exclusion window that hard-drops already-shown clips
  (with soft deprioritization for older ones).
- **Pre-computed recommendations** (`engine/recommendations.py`, `engine/episode_recommendations.py`):
  cached top-3 alternative clips/episodes resolved instantly via dedicated click endpoints.
- **Episode-search branch** (`engine/episode_search.py`) and **small-talk branch**
  (`engine/small_talk.py`) with optional Google Search grounding.
- **Session management endpoints**: reset, info, delete, list, cleanup, health, and a memory
  debug dump.
- **Graceful degradation**: BM25 sparse encoder falls back to a default model when no trained
  model is available, with fallbacks at routing, retrieval, reranking, and selection stages.
- **Local dev runner** (`run_local.py`), env-driven configuration (`config.py`, `.env.example`),
  synthetic demo catalog (`data/entities.sample.json`), memory-behavior tests (`tests/`),
  architecture docs (`docs/ARCHITECTURE.md`), and a CI workflow (lint + byte-compile).

[1.0.0]: https://github.com/akira231097/echofind/releases/tag/v1.0.0
