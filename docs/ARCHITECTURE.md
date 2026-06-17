# EchoFind — Architecture

A deep dive into how EchoFind turns a natural-language question into a single
best podcast clip or episode. Read [`../README.md`](../README.md) first for the
high-level picture.

---

## 1. Request lifecycle

```
Browser ──POST /api/chat/stream──▶ FastAPI (server.py)
                                      │
                                      ▼
                          api/routes.py: event_generator
                                      │  async for update in agent.ask_streaming(...)
                                      ▼
                          engine/agent.py: orchestrator
                                      │
   Stage 0 ── route_query (engine/router.py) ───────────────┐
                                      │                       │ low confidence
            ┌─────────────────────────┼───────────────┐      ▼ → small_talk fallback
            ▼                         ▼               ▼
       small_talk              episode_search     clip_search
   (engine/small_talk)     (engine/episode_search)  (engine/agent.py pipeline)
                                      │
                                      ▼
                          SSE: stage events → complete → recommendations
                                      │
                                      ▼
            background_tasks → agent.compress_memory_background(session)
```

Every stage yields a `StageUpdate` (`stage`, `message`, `progress`, `data`) which
`api/routes.py` serializes as a Server-Sent Event. The browser parses the byte
stream manually (`getReader()` + `TextDecoder`) and animates a progress ring +
six stage pills until the terminal `complete` event renders the clip/episode card.

---

## 2. Components

### server.py
Builds the API clients (OpenAI for embeddings, Pinecone for vectors, Gemini via
the OpenAI-compatible endpoint), loads the entity catalog, constructs the
`EchoFindAgent`, wires CORS + routes, and serves `web/index.html`. A FastAPI
`lifespan` context performs startup/shutdown; Google Search grounding is
initialized opportunistically and degrades cleanly if `google-genai` is absent.

### api/routes.py
The HTTP surface. The streaming endpoint adapts the agent's async generator into
SSE frames, maps granular stages to event types (`stage`/`complete`/`error`/
`recommendations`), and — on completion — schedules background memory compression
via FastAPI `BackgroundTasks`. Also exposes recommendation-click, session reset/
info/delete, and a memory-debug endpoint.

### engine/router.py — the 3-branch router
A temperature-0 Gemini call with forced JSON output over a few-shot prompt plus a
memory-rendered context. It returns a `RouterOutput`: a `route`
(`small_talk` | `episode_search` | `clip_search`), `confidence`, `reasoning`,
resolved entities, and a `fallback_route`. If confidence falls below the
threshold (0.70), the orchestrator overrides the route with the safe fallback.

### engine/query_analyzer.py — query understanding + HyDE
Resolves pronouns and follow-ups against memory, extracts guests/hosts/show and a
structured `time_filter`, and generates **HyDE** hypothetical transcript snippets.
A **gazetteer** (`retrieval/gazetteer.py`) provides an authoritative fuzzy index
of known hosts/guests/shows so extracted names are validated and host-vs-guest is
disambiguated. The metadata-extraction call and the HyDE calls run concurrently.

### engine/agent.py — orchestrator & clip pipeline
The heart of the system. Beyond dispatching branches, it implements the
clip-search pipeline:

- **HyDE weighting** — each HyDE vector's RRF weight is interpolated between
  configured max/min by its cosine similarity to the original query
  (`_rank_hyde_weights`); the original query is pinned at a fixed higher weight.
- **Time-search planning** — `make_time_search_plan` converts the `time_filter`
  into one or more retrieval **buckets** (`latest`, `oldest`, `relative_recent`,
  `between`, `before`, `after`), including pre/post-event splitting that can
  regex-infer an event anchor. Each bucket carries its own weight and date range.
- **Parallel bucket retrieval** — every bucket builds its own metadata + date
  filter and runs hybrid search concurrently.
- **Recency strategy & metadata scoring** — `apply_recency_boost`,
  `apply_recency_strategy`, and `apply_hybrid_metadata_scoring` blend semantic,
  date, person-match, and show-match signals using an intent-selected weight
  profile, with `enforce_episode_cap_and_bucket_quota` for diversity.
- **Exclusion** — `memory.get_excluded_ids()` hard-drops clips shown in the last
  5 turns (with a soft fallback when too few candidates remain).

### retrieval/data_fetcher.py — search + fusion + hydration
- `concurrent_embedding_generation` / `concurrent_sparse_embedding_generation` —
  batched dense + sparse encoding.
- `concurrent_pinecone_search` — per-query dense and sparse queries with
  two-stage recall budgeting and dynamic score thresholds
  (`min(base, max(minimum, max_score * relative_factor))`).
- **Multi-level RRF** — dense and sparse results are fused per query
  (`RRF_K = 60`), then the per-query lists (original + HyDE) are fused across
  queries with per-query weights, then buckets are fused by bucket weight.
- `batch_get_rds_items` — opens a PostgreSQL connection and `SELECT`s full rows
  (transcript, titles, speakers, guests/hosts, `startMs`/`endMs`, media URLs),
  filtering soft-deleted / non-feed rows.
- `merge_db_and_pinecone_data` — joins relational rows with vector scores.

### retrieval/search.py — reranking
`rerank_chunks_cohere` calls Cohere rerank (via Pinecone inference) over a query
enriched with `Constraints:` lines (guest/host/time/follow-up), capped at a fixed
candidate budget.

### retrieval/search_filter.py — filter construction
Builds Pinecone metadata filters: fuzzy-matches requested people against the
gazetteer (WRatio + token-set with a specificity guard) and translates the
`time_filter` into a numeric `pdnumeric` (YYYYMMDD) date clause.

### retrieval/sparse_encoder.py — BM25
Loads a trained BM25 model if available and otherwise **falls back to a default
encoder**, so the service boots without the proprietary model. Exposes async
batch encoding used by the dense+sparse parallel path.

### engine/selection.py — single-call selection + memory
Sends the top candidates to Gemini as XML `<document>` blocks (full transcripts +
recency/match markers), instructs the model to **extract 2–4 supporting quotes
before choosing**, and returns — in one structured-output call — the chosen index,
a user-facing answer, a confidence score, and the full `BranchMemoryUpdate`.

### engine/memory.py — deterministic conversation memory
The non-LLM source of truth. Key pieces:

- **`SearchState`** — current entities, current/thread topic, topic history,
  participants, conversation phase, route pattern, last action, follow-up flag.
- **`apply_branch_memory_update`** — the single entry point all branches call.
  It increments the turn, records the route, merges/replaces entities (replace on
  topic shift), runs a heuristic person detector, updates the topic thread,
  records the shown artifact into the **exclusion window**, appends a
  `ConversationTurn`, decays entity relevance (`mention_count * 0.8^turns_ago`),
  and compresses the oldest turn once the recent window overflows.
- **Per-component renderers** — `render_for_router`, `render_for_query_analyzer`,
  `render_for_small_talk` produce exactly the context each LLM needs.

### engine/recommendations.py & episode_recommendations.py
After a result is chosen, the top-3 alternatives are generated with pre-written
click prompts/answers/memory updates and cached keyed by `session:turn`, so a
follow-up tap resolves instantly and writes a contextual memory turn.

### engine/small_talk.py & episode_search.py
- **small_talk** handles greetings, clarifications, and explanations; explanatory
  answers can use Google Search **grounding** for fresh facts.
- **episode_search** aggregates chunk-level Pinecone hits to **episode-level**
  scores, enriches from the episodes table, and selects the best whole episode.

---

## 3. Key data contracts (`engine/schemas.py`)

- **`ChatRequest` / `ChatResponse`** — the public API shape.
- **`RouterOutput`** — routing decision with confidence + fallback.
- **`QueryAnalysisResult`** — resolved query, follow-up flag, extracted entities,
  HyDE docs, time filter.
- **`BranchMemoryUpdate`** — the *unified* memory contract every branch emits
  (turn summary, action type/target, entities, topics, topic-shift flag, plus
  enriched `key_quotes` / `topics_covered` / `notable_examples`).
- **`SelectionResult`**, **`SmallTalkResponse`**, **`EpisodeSearchResponse`**.

---

## 4. Concurrency & resilience

- **Parallelism:** batched single-call embeddings; `asyncio.gather` across HyDE
  calls, retrieval buckets, and dense/sparse tasks; pre-computed recommendations.
- **Graceful degradation:** optional grounding, entity-filter fallback when a
  filter is empty, relaxed-recall fallback when a bucket underdelivers, per-stage
  fallback selections, and exponential-backoff-with-jitter LLM retries that skip
  non-retryable auth/quota errors.
- **JSON robustness ladder:** structured output → JSON-object mode → a hand-written
  character-state-machine repairer that escapes stray interior quotes.

---

## 5. Configuration

All runtime configuration lives in `config.py` and is sourced from environment
variables (see `.env.example`). It covers model selection, per-stage reasoning
effort, retrieval budgets and score thresholds, RRF and HyDE weights, recency
strategy, fuzzy-match thresholds, episode-scoring weight profiles, and memory
limits. No secrets are committed — every credential is `os.getenv(...)` with no
fallback.
