---
title: "Three-Level Reciprocal Rank Fusion: combining retrieval signals that don't share a scale"
published: false
description: How EchoFind fuses dense, sparse, multi-query, and time-bucketed retrieval into one ranking using rank-only fusion — and why it deliberately throws magnitude away, then earns it back.
tags: rag, machinelearning, python, search
canonical_url:
---

> Engineering notes from **EchoFind**, a memory-aware conversational RAG engine
> that returns the single best podcast *clip* (with timestamps) for a natural-language
> question. This post is about the retrieval core. Code references are real;
> constants are the ones the system actually ships with.

## The problem: four good signals, four incomparable scales

A single embedding of a user's question rarely matches the wording of the *best*
transcript segment. Someone asks *"what did that founder say about pricing?"* and
the transcript says *"…we landed on usage-based billing because…"*. No shared
keywords, and the dense vector for the literal question isn't perfectly aligned
with the vector for the answer.

So EchoFind doesn't rely on one signal. For every query it retrieves along **four
independent axes** at once:

1. **Dense** semantic similarity (OpenAI `text-embedding-3-large` in Pinecone).
2. **Sparse** lexical match (BM25), to catch names, jargon, and exact phrases the
   embedding smooths over.
3. **Multiple query formulations** — the original question plus several
   [HyDE](https://arxiv.org/abs/2212.10496) documents (short *hypothetical*
   transcript snippets that look like the answer we're hoping to find).
4. **Time windows** — for "latest", "before the election", "back in 2021" style
   questions, several date-bucketed searches run in parallel.

Each axis produces its own ranked list. And here's the catch that makes naïve
combination impossible: **the scores aren't comparable.** Pinecone dense cosine
scores live on one scale, BM25 sparse scores on another, and a date-sorted bucket
has no relevance score at all. Adding them is meaningless; even min-max
normalizing them per-list quietly invents structure that isn't there.

## The fix: fuse on rank, not score

[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf)
(RRF) sidesteps the whole problem by ignoring scores and using only **rank**. An
item that appears at position `r` in a list contributes:

```python
def rrf_contribution(rank: int, weight: float, k: int = 60) -> float:
    # rank is 1-based; k dampens how fast the contribution decays
    return weight * (1.0 / (k + rank))
```

Sum each item's contribution across every list it appears in, sort by the total,
and you have a fused ranking. `k = 60` is the long-standing SIGIR/Elastic default
and EchoFind keeps it (`RRF_K = 60`). Because the formula depends only on ordinal
position, dense, sparse, and date-sorted lists become safely combinable — scale
drift in any one signal can't blow up the fusion.

The whole thing is one primitive, `rrf_fuse_lists`, applied at **three levels**.

### Level 1 — dense + sparse, per query

Inside `concurrent_hybrid_search`, each query's dense and sparse hit lists are
fused with `DENSE_RRF_WEIGHT` / `SPARSE_RRF_WEIGHT`. A small but useful detail:
while fusing, the code coalesces a `retrieval_source` label (`"both"` > `dense` >
`sparse`) so later stages — and debugging humans — know *why* a chunk surfaced.

### Level 2 — original query + HyDE, across formulations

Now we fuse the per-query lists together. But not all query formulations deserve
equal trust. A HyDE document is a *hallucinated* snippet; a good one pulls
retrieval toward the answer, a bad one pulls it toward a tangent. So before
fusing, EchoFind weights them:

```python
# Each HyDE embedding is scored against the base-query embedding.
# Closer to the user's intent => higher fusion weight.
sims = [(i, cosine_similarity(hyde_vec[i], query_vec)) for i in range(n_hyde)]
weights = rank_hyde_weights(sims, high=HYDE_WEIGHT_MAX, low=HYDE_WEIGHT_MIN)
# the original query always keeps the highest weight of all
weights["original"] = ORIGINAL_QUERY_WEIGHT
```

with the constants the system ships:

- `ORIGINAL_QUERY_WEIGHT = 1.25` — the user's real question leads.
- `HYDE_WEIGHT_MAX = 1.1` down to `HYDE_WEIGHT_MIN = 0.85` — HyDE docs are sorted
  by cosine similarity to the base query and assigned a **linearly interpolated**
  weight in that band. The HyDE doc closest to the user's intent pulls hardest;
  the most speculative one barely participates.

`combine_pinecone_results` then fuses the original-query list with the weighted
HyDE lists.

### Level 3 — across time buckets

The same `combine_pinecone_results` fuses each *time bucket's* result list using
per-bucket weights from `make_time_search_plan`. A "latest episode about X" query,
for example, produces a tight recency bucket weighted `~2.0` and a broad
backstop "all" bucket weighted `~0.3`. Buckets run concurrently via
`asyncio.gather`, so the extra recall costs latency only equal to the *slowest*
bucket, not their sum.

After fusion, an episode-cap / min-per-bucket quota
(`enforce_episode_cap_and_bucket_quota`) keeps one chatty episode from
monopolizing the candidate set — diversity matters when the next stage is a
reranker with a fixed budget.

## The deliberate trade-off: throw away magnitude, then buy it back

RRF's strength is also its cost: it is **score-agnostic**. It keeps ordinal rank
and discards the *magnitude* of similarity. That makes heterogeneous signals
combinable and robust — but it cannot express *"this one dense hit is vastly
better than everything else below it."* That information is genuinely gone after
fusion.

EchoFind accepts that loss on purpose, because magnitude is cheap to recover
**later**, in a controlled place, on a small candidate set:

- **Cohere rerank** (via Pinecone inference) re-scores the survivors directly
  against the query — restoring a real, comparable relevance magnitude.
- **`apply_hybrid_metadata_scoring`** then applies an intent-driven weighted blend
  of semantic + date + person + show signals, with the weights chosen by what the
  query actually asked for (a "latest Lex episode" weights date and person; a pure
  topic query weights semantics at 0.85).

The mental model is a funnel:

> **Rank fusion maximizes recall** of plausible candidates across every signal.
> **Rerank-and-rescore restores precision** at the top, where it's cheap.

Fusion is allowed to be blurry because it's followed by something sharp. You get
the recall of four retrieval strategies without ever having to pretend their
scores mean the same thing.

## Takeaways

- **When you have multiple retrievers, fuse on rank, not score.** RRF with
  `k = 60` is a one-line primitive that makes incomparable signals composable and
  immune to per-signal scale drift.
- **Weight your query expansions by trust.** HyDE is powerful but speculative;
  scoring each hypothetical against the real query (and capping its weight below
  the original) keeps a bad expansion from hijacking retrieval.
- **Separate the recall stage from the precision stage.** Let fusion be
  high-recall and magnitude-blind; recover magnitude afterward with a reranker on
  a small set. Trying to do both at once is where hybrid search usually goes
  wrong.

*EchoFind is open source. The retrieval core lives in `retrieval/data_fetcher.py`
and the orchestration in `engine/agent.py`; a reproducible evaluation harness is
in `evals/`.*
