"""Retrieval + answer-quality evaluation harness for EchoFind.

This runs the **real** clip-search pipeline end to end — router → query analysis →
HyDE → hybrid dense+sparse search → three-level RRF → Cohere rerank → hybrid
metadata scoring → single-call selection — against a labeled query set, then
reports retrieval quality, selection accuracy, and latency percentiles. With the
optional ``--ragas`` flag it additionally scores answer faithfulness and
relevancy with an LLM judge.

Unlike the sibling Clip'O'pedia eval (which ships a deterministic in-memory
backend and therefore runs with zero credentials), EchoFind talks to live
OpenAI + Pinecone + PostgreSQL. So this harness has two modes:

* **Live mode** — credentials and a populated index are present. It boots the
  same agent ``server.py`` builds, runs every labeled query, and prints real
  numbers.
* **Skeleton mode** — credentials are absent. It validates the labeled query set
  and the metric wiring, prints exactly what *would* run, and exits WITHOUT
  emitting a single fabricated metric. This is the default when you clone the
  repo without secrets.

The harness never modifies production code: it captures the ranked candidate set
by wrapping ``engine.agent.apply_hybrid_metadata_scoring`` at runtime, purely for
measurement.

Run:
    python evals/retrieval_eval.py                       # skeleton check (no creds)
    python evals/retrieval_eval.py --repeats 3           # live, 3 latency repeats
    python evals/retrieval_eval.py --ragas               # live + answer-quality
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Any

# Make the repo root importable when run as `python evals/retrieval_eval.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_QUERIES = os.path.join(os.path.dirname(__file__), "queries.sample.jsonl")

# Env vars that must hold real (non-placeholder) values for live mode.
REQUIRED_KEYS = ("OPENAI_API_KEY", "PINECONE_API_KEY", "GEMINI_API_KEY")
# A relational store is also needed to hydrate chunk rows.
REQUIRED_DB = ("RDS_HOST",)


# ---------------------------------------------------------------------------
# Labeled query set
# ---------------------------------------------------------------------------

def load_queries(path: str) -> list[dict[str, Any]]:
    """Load and validate the JSONL labeled query set.

    Each line is an object: ``{"query": str, "relevant": {...}, "reference_answer"?: str}``
    where ``relevant`` carries one or more human-readable matchers:
    ``episode_title_contains`` / ``guests_any`` / ``shows_any``.
    """
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})")
            if not obj.get("query"):
                raise SystemExit(f"{path}:{lineno}: missing 'query'")
            rel = obj.get("relevant") or {}
            if not any(rel.get(k) for k in ("episode_title_contains", "guests_any", "shows_any")):
                raise SystemExit(
                    f"{path}:{lineno}: 'relevant' needs at least one of "
                    "episode_title_contains / guests_any / shows_any"
                )
            items.append(obj)
    if not items:
        raise SystemExit(f"{path}: no labeled queries found")
    return items


def is_relevant(meta: dict[str, Any], rel: dict[str, Any]) -> bool:
    """True if a candidate (or chosen) clip matches the labeled relevance rule."""
    title = (meta.get("episode_title") or "").lower()
    for sub in rel.get("episode_title_contains", []):
        if sub.lower() in title:
            return True

    people = [str(x).lower() for x in (meta.get("guests") or [])]
    people += [str(x).lower() for x in (meta.get("speakers") or [])]
    for guest in rel.get("guests_any", []):
        gl = guest.lower()
        if any(gl in p or p in gl for p in people):
            return True

    podcast = (meta.get("podcast_title") or "").lower()
    for show in rel.get("shows_any", []):
        if show.lower() in podcast:
            return True
    return False


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------

def _has_real_value(name: str) -> bool:
    val = (os.environ.get(name) or "").strip()
    return bool(val) and not val.lower().startswith("your-")


def credentials_present() -> bool:
    return all(_has_real_value(k) for k in REQUIRED_KEYS) and all(
        _has_real_value(k) for k in REQUIRED_DB
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def score_ranking(ranked_meta: list[dict[str, Any]], rel: dict[str, Any], k: int) -> dict[str, float]:
    """Hit@1, Hit@3, Hit@k, Recall@k, reciprocal rank for one query."""
    flags = [is_relevant(m, rel) for m in ranked_meta]
    first = next((i + 1 for i, ok in enumerate(flags) if ok), None)
    total_relevant = max(1, sum(flags))  # candidate-set recall denominator
    return {
        "hit@1": 1.0 if flags[:1] == [True] else 0.0,
        "hit@3": 1.0 if any(flags[:3]) else 0.0,
        f"hit@{k}": 1.0 if any(flags[:k]) else 0.0,
        f"recall@{k}": sum(flags[:k]) / total_relevant,
        "rr": (1.0 / first) if first else 0.0,
    }


# ---------------------------------------------------------------------------
# Skeleton mode
# ---------------------------------------------------------------------------

def run_skeleton(queries: list[dict[str, Any]], args: argparse.Namespace) -> int:
    print("EchoFind - retrieval evaluation harness")
    print("=" * 64)
    print("MODE: SKELETON (no live credentials detected)")
    print("-" * 64)
    missing = [k for k in REQUIRED_KEYS + REQUIRED_DB if not _has_real_value(k)]
    print(f"Missing/placeholder env vars : {', '.join(missing)}")
    print(f"Labeled query set            : {args.queries}")
    print(f"Queries validated            : {len(queries)}")
    print(f"Latency repeats (live)       : {args.repeats}")
    print(f"k for Hit@k / Recall@k       : {args.k}")
    print(f"RAGAS answer scoring         : {'requested' if args.ragas else 'off'}")
    print("-" * 64)
    print("Metrics this harness will report in live mode:")
    for line in (
        "  Hit@1        chosen-rank-1 candidate is relevant",
        "  Hit@3 / Hit@k  a relevant clip appears in the top-3 / top-k",
        "  Recall@k     fraction of the candidate-relevant set captured in top-k",
        "  MRR          mean reciprocal rank of the first relevant clip",
        "  Selection    the single clip the LLM actually returned is relevant",
        "  Latency      end-to-end p50 / p95 / p99 / mean (wall clock)",
        "  RAGAS*       faithfulness + answer relevancy (with --ragas)",
    ):
        print(line)
    print("-" * 64)
    print("To produce real numbers:")
    print("  1. cp .env.example .env  and fill in OPENAI / PINECONE / GEMINI / RDS_*")
    print("  2. Point the labeled set at your catalog (see evals/README.md)")
    print("  3. python evals/retrieval_eval.py --repeats 3 [--ragas]")
    print("=" * 64)
    print("No metrics emitted: skeleton mode never fabricates numbers.")
    return 0


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

async def run_live(queries: list[dict[str, Any]], args: argparse.Namespace) -> int:
    # Heavy imports happen only in live mode so skeleton mode stays stdlib-only.
    import engine.agent as agent_mod
    from engine.agent import EchoFindAgent
    from engine.schemas import PipelineStage
    from server import create_clients, load_entity_data

    # --- Capture the ranked candidate set without touching production code. ---
    capture: dict[str, list[dict[str, Any]]] = {"ranked": []}
    _orig_score = agent_mod.apply_hybrid_metadata_scoring

    def _capturing_score(chunks, llm_analysis, max_per_episode=3):
        result = _orig_score(chunks, llm_analysis, max_per_episode)
        capture["ranked"] = list(result)
        return result

    agent_mod.apply_hybrid_metadata_scoring = _capturing_score

    personalities, authors, shows = load_entity_data()
    openai_client, pinecone_client, gemini_client = create_clients()
    agent = EchoFindAgent(
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        gemini_client=gemini_client,
        unique_personalities=personalities,
        unique_authors=authors,
        unique_shows=shows,
    )

    async def ask_once(session_id: str, question: str) -> dict[str, Any]:
        capture["ranked"] = []
        final: dict[str, Any] = {}
        async for update in agent.ask_streaming(session_id, question):
            if update.stage == PipelineStage.COMPLETE.value:
                final = update.data or {}
        return final

    print("EchoFind - retrieval evaluation harness")
    print("=" * 64)
    print(f"MODE: LIVE | queries: {len(queries)} | repeats: {args.repeats} | k={args.k}")
    print("-" * 64)

    # Warm caches / connection pools so the first query doesn't skew latency.
    await ask_once("eval-warmup", "what did the guest say about reliability?")

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    ragas_rows: list[dict[str, Any]] = []
    off_branch = 0

    for idx, item in enumerate(queries):
        session_id = f"eval-{idx}"
        query, rel = item["query"], item.get("relevant", {})

        t0 = time.perf_counter()
        final = await ask_once(session_id, query)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        branch = final.get("branch", "clip_search")
        ranked_meta = list(capture["ranked"])
        chosen = final.get("chunk") or {}

        if branch != "clip_search" or not ranked_meta:
            off_branch += 1

        metrics = score_ranking(ranked_meta, rel, args.k) if ranked_meta else {}
        selection_hit = 1.0 if (chosen and is_relevant(chosen, rel)) else 0.0
        per_query.append({"metrics": metrics, "selection": selection_hit, "branch": branch})

        if args.ragas and item.get("reference_answer"):
            ragas_rows.append({
                "question": query,
                "answer": final.get("answer", ""),
                "contexts": [m.get("chunk", "") for m in ranked_meta[: args.k] if m.get("chunk")],
                "ground_truth": item["reference_answer"],
            })

        # Extra latency repeats for stable percentiles.
        for _ in range(max(0, args.repeats - 1)):
            t0 = time.perf_counter()
            await ask_once(f"{session_id}-r", query)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        agent.reset_session(session_id)

    scored = [p for p in per_query if p["metrics"]]
    n = max(1, len(scored))

    def mean_of(key: str) -> float:
        return sum(p["metrics"].get(key, 0.0) for p in scored) / n

    print(f"Scored (clip_search) queries : {len(scored)}/{len(queries)}")
    if off_branch:
        print(f"  (note: {off_branch} routed off clip_search or returned no candidates)")
    print("-" * 64)
    print(f"Hit@1            {mean_of('hit@1'):.3f}")
    print(f"Hit@3            {mean_of('hit@3'):.3f}")
    print(f"Hit@{args.k}            {mean_of(f'hit@{args.k}'):.3f}")
    print(f"Recall@{args.k}         {mean_of(f'recall@{args.k}'):.3f}")
    print(f"MRR              {mean_of('rr'):.3f}")
    print(f"Selection acc.   {sum(p['selection'] for p in per_query) / max(1, len(per_query)):.3f}")
    print("-" * 64)
    print(f"Latency p50      {_pct(latencies, 50):.1f} ms")
    print(f"Latency p95      {_pct(latencies, 95):.1f} ms")
    print(f"Latency p99      {_pct(latencies, 99):.1f} ms")
    print(f"Latency mean     {statistics.mean(latencies):.1f} ms")

    if args.ragas:
        _run_ragas(ragas_rows)

    if args.out:
        summary = {
            "n_queries": len(queries),
            "n_scored": len(scored),
            "hit@1": mean_of("hit@1"),
            "hit@3": mean_of("hit@3"),
            f"hit@{args.k}": mean_of(f"hit@{args.k}"),
            f"recall@{args.k}": mean_of(f"recall@{args.k}"),
            "mrr": mean_of("rr"),
            "selection_acc": sum(p["selection"] for p in per_query) / max(1, len(per_query)),
            "latency_p50_ms": _pct(latencies, 50),
            "latency_p95_ms": _pct(latencies, 95),
            "latency_p99_ms": _pct(latencies, 99),
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary -> {args.out}")
    return 0


def _run_ragas(rows: list[dict[str, Any]]) -> None:
    print("-" * 64)
    if not rows:
        print("RAGAS: skipped (no labeled queries carried a 'reference_answer').")
        return
    try:
        from datasets import Dataset  # noqa: F401
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print("RAGAS: requested but not installed. `pip install ragas datasets` to enable.")
        return
    try:
        from datasets import Dataset
        dataset = Dataset.from_list(rows)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        print(f"RAGAS  faithfulness     {result['faithfulness']:.3f}")
        print(f"RAGAS  answer_relevancy {result['answer_relevancy']:.3f}")
    except Exception as exc:  # judge-LLM/network failures shouldn't crash the run
        print(f"RAGAS: evaluation failed ({exc!s}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EchoFind retrieval evaluation harness")
    p.add_argument("--queries", default=DEFAULT_QUERIES, help="JSONL labeled query set")
    p.add_argument("--repeats", type=int, default=1, help="latency repeats per query (live)")
    p.add_argument("--k", type=int, default=5, help="k for Hit@k / Recall@k")
    p.add_argument("--ragas", action="store_true", help="also score answers with RAGAS")
    p.add_argument("--out", default=None, help="optional path to write a JSON summary")
    p.add_argument("--force-skeleton", action="store_true", help="run skeleton check even with creds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    queries = load_queries(args.queries)
    if args.force_skeleton or not credentials_present():
        return run_skeleton(queries, args)
    return asyncio.run(run_live(queries, args))


if __name__ == "__main__":
    raise SystemExit(main())
