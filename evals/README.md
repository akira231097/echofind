# EchoFind — evaluation harness

A reproducible harness that measures the **real** clip-search pipeline against a
labeled query set: retrieval quality (Hit@k, Recall@k, MRR), end-to-end
selection accuracy, latency percentiles, and — optionally — answer quality via
[RAGAS](https://docs.ragas.io).

It exercises the full path the production agent runs:

```
router → query analysis → HyDE → hybrid dense+sparse search
       → three-level RRF → Cohere rerank → hybrid metadata scoring
       → single-call select + memory
```

The harness does **not** modify production code. It captures the ranked
candidate set by wrapping `engine.agent.apply_hybrid_metadata_scoring` at
runtime, purely for measurement.

## Two modes

| Mode | When | Behavior |
| --- | --- | --- |
| **Skeleton** | no credentials | validates the labeled set + metric wiring, prints what *would* run, emits **zero** numbers |
| **Live** | `OPENAI_API_KEY`, `PINECONE_API_KEY`, `GEMINI_API_KEY`, `RDS_HOST` set | boots the real agent, runs every query, prints real metrics |

Skeleton mode is the default on a fresh clone — by design, the harness will
**never fabricate a metric**. EchoFind's retrieval quality depends on a private
podcast index and relational store, so the numbers must come from your own data.

```bash
python evals/retrieval_eval.py                 # skeleton check (no creds)
python evals/retrieval_eval.py --repeats 3     # live; 3 latency repeats/query
python evals/retrieval_eval.py --ragas         # live + answer faithfulness/relevancy
python evals/retrieval_eval.py --out evals/results.json
```

## Metrics

| Metric | Meaning |
| --- | --- |
| **Hit@1** | the rank-1 candidate is relevant |
| **Hit@3 / Hit@k** | a relevant clip appears in the top-3 / top-k |
| **Recall@k** | fraction of the candidate-relevant set captured in the top-k |
| **MRR** | mean reciprocal rank of the first relevant clip |
| **Selection acc.** | the single clip the LLM actually returned is relevant |
| **Latency p50/p95/p99** | end-to-end wall-clock per query |
| **RAGAS** (opt) | `faithfulness` + `answer_relevancy` from an LLM judge |

## Labeled query set

`queries.sample.jsonl` is a **template** wired to the fictional catalog in
`data/entities.sample.json`. Replace it with your own catalog. One JSON object
per line:

```json
{
  "query": "what did the guest say about reliable AI agents",
  "relevant": {
    "episode_title_contains": ["agent reliability"],
    "guests_any": ["dr. elena voss"],
    "shows_any": ["frontier talks"]
  },
  "reference_answer": "Reliable agents need typed tools, retries, and self-correction."
}
```

`relevant` needs **at least one** matcher. A candidate (or the chosen) clip
counts as relevant if its episode title contains any `episode_title_contains`
substring (case-insensitive), **or** its guests/speakers intersect `guests_any`,
**or** its podcast title contains any `shows_any` entry. Human-readable matchers
mean a labeler never has to look up opaque chunk IDs.

`reference_answer` is optional and only used by `--ragas`.

## RAGAS (optional)

```bash
pip install ragas datasets
python evals/retrieval_eval.py --ragas
```

RAGAS uses an LLM judge (set via its own env, e.g. `OPENAI_API_KEY`) to score
each answer's `faithfulness` to the retrieved contexts and its
`answer_relevancy` to the question. Only queries carrying a `reference_answer`
are scored.

## Notes

- Queries that the router sends to `small_talk` or `episode_search` (rather than
  `clip_search`) produce no candidate ranking and are reported separately —
  keep the labeled set focused on clip-retrieval intents.
- `--repeats N` runs each query `N` times for stable latency percentiles; the
  first warm-up query is excluded.
- The harness is excluded from CI byte-compilation because the optional `ragas`
  import is lazy; nothing here runs unless you invoke it.
