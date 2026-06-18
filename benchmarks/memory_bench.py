"""Micro-benchmark for EchoFind's deterministic conversation-memory layer.

The memory state machine (engine/memory.py) runs on every turn: it applies the
unified branch update (entity merge/decay, topic-thread tracking, exclusion
window, turn compression) and renders tailored context strings for the router
and query analyzer. This benchmark measures the pure-Python CPU cost of those
hot paths at a realistic steady state (~50 turns of history). No network, no
models — just the orchestration overhead the user pays per turn.

Run:  python benchmarks/memory_bench.py
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
import time

logging.disable(logging.CRITICAL)  # keep the benchmark output clean
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import ConversationMemory  # noqa: E402
from engine.schemas import BranchMemoryUpdate  # noqa: E402

ITERS = 3000


def make_update(i: int) -> BranchMemoryUpdate:
    return BranchMemoryUpdate(
        turn_summary=f"User asked about topic {i}; showed a clip on AI agents and reliability.",
        action_type="clip_shown",
        action_target_id=f"clip-{i:04d}",
        action_target_title=f"Episode {i}: autonomy and tooling",
        published_date="2026-06-10",
        entities_mentioned=["Dr. Lena Ortiz", "AI agents", "reliability", "tooling"],
        topics_discussed=["ai agents", "autonomy"],
        is_topic_shift=(i % 5 == 0),
        suggested_phase="deep_dive",
        key_quotes=["agents need reliable tools", "let them retry and self-correct"],
        topics_covered=["tooling", "evaluation"],
        notable_examples=["API self-correction loop"],
    )


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def bench(label: str, fn) -> None:
    fn()  # warm up
    samples = [(_t(fn)) for _ in range(ITERS)]
    print(f"{label:<34} p50 {pct(samples,50):7.3f} ms   p95 {pct(samples,95):7.3f} ms   mean {statistics.mean(samples):7.3f} ms")


def _t(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    mem = ConversationMemory("bench-session")
    # Warm to steady state (memory compresses past MAX_RECENT_TURNS).
    for i in range(120):
        mem.apply_branch_memory_update(
            make_update(i), "clip_search", 0.92, f"find clips about topic {i}",
            f"question {i}", f"resolved query {i}",
        )

    print(f"EchoFind — conversation-memory micro-benchmark ({ITERS} iters, steady state)")
    print("-" * 78)

    counter = {"i": 1000}

    def do_apply():
        counter["i"] += 1
        mem.apply_branch_memory_update(
            make_update(counter["i"]), "clip_search", 0.9,
            "find clips about agent reliability", "what else did she say?",
            "what else did Dr. Lena Ortiz say about agent reliability?",
        )

    bench("apply_branch_memory_update (per turn)", do_apply)
    bench("search_state.render_for_router", lambda: mem.search_state.render_for_router(mem.recent_turns))
    bench("search_state.render_for_query_analyzer", lambda: mem.search_state.render_for_query_analyzer(mem.recent_turns))
    bench("render_for_prompt_enhanced (full)", lambda: mem.render_for_prompt_enhanced())
    bench("get_excluded_ids", lambda: mem.get_excluded_ids())


if __name__ == "__main__":
    main()
