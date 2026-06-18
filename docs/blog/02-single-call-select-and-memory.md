---
title: "One LLM call instead of two: fusing answer selection and memory in a conversational RAG agent"
published: false
description: How EchoFind cuts a round-trip off the user's critical path by having a single model call pick the answer AND update conversation memory — plus the four-layer reliability ladder that keeps structured output from ever hard-failing.
tags: llm, rag, python, ai
canonical_url:
---

> Engineering notes from **EchoFind**, a memory-aware conversational RAG engine
> for podcast clips. This post is about the last step of every query — the one on
> the user's critical path — and a reliability pattern for structured LLM output
> that's useful well beyond this project.

## Two jobs at the end of every turn

By the time EchoFind has retrieved, fused, and reranked candidate clips, two
things still have to happen before the user sees an answer:

1. **Select + answer.** Pick the single best clip from the candidates and write
   the user-facing answer that cites it.
2. **Update memory.** Record what this turn was about — a summary, the entities
   mentioned, the themes, notable examples — so the *next* turn can resolve
   *"what else did **he** say about that?"* without re-deriving everything.

The obvious implementation runs these as two LLM calls in sequence. It works, and
it's clean. It also puts **two round-trips (~1–2s of avoidable latency)** on the
critical path, right at the end where the user is already waiting.

## The optimization: fold both into one structured call

`select_and_update_memory` (in `engine/selection.py`) does both jobs in **one**
Gemini call. A [PTCF-structured](https://cloud.google.com/blog/products/application-development/five-best-practices-for-prompt-engineering)
prompt presents the candidates as XML `<document>` blocks and asks the model —
via a single Pydantic schema, `SelectionWithMemoryOutput` — to emit everything
together:

```python
class SelectionWithMemoryOutput(BaseModel):
    supporting_quotes: list[str]   # extracted FIRST — grounds the choice
    chosen_index: int              # which candidate wins
    answer: str                    # the user-facing answer
    confidence: float
    memory_update: BranchMemoryUpdate   # summary, entities, themes, examples
```

The ordering in the schema is deliberate: quotes first, then the index, then the
answer. Making the model surface its supporting evidence *before* committing to a
choice is a lightweight chain-of-thought that measurably steadies the selection —
and it costs nothing extra because it's the same call.

Two wins fall out of the fusion:

- **Latency.** One round-trip instead of two, on the exact stretch of the
  pipeline the user is waiting on.
- **Consistency by construction.** Memory is written in the *same* call that
  chose the clip, so it can never describe a different clip than the one the user
  was shown. With two calls, that drift is a real failure mode; here it's
  impossible.

## The hard part: structured output that never hard-fails

Folding two jobs into one schema makes the output larger and more structured —
and structured LLM output fails in annoying, runtime-specific ways. A robust
agent can't crash the user's turn because a model returned *almost*-valid JSON.

So selection sits on a **four-layer reliability ladder**, each rung catching the
failure of the one above:

```python
async def select_and_update_memory(client, prompt, schema):
    # 1) Native structured output — the happy path.
    try:
        return client.beta.chat.completions.parse(..., response_format=schema)
    except AttributeError:
        pass  # SDK/endpoint doesn't support .parse on this path

    # 2) Ask for a raw JSON object and validate it ourselves.
    raw = client.chat.completions.create(..., response_format={"type": "json_object"})
    try:
        return schema.model_validate_json(raw)
    except ValidationError:
        pass

    # 3) Repair common malformations, then re-validate.
    try:
        return schema.model_validate_json(_repair_json(raw))
    except ValidationError:
        pass

    # 4) Typed fallback — pick the top candidate, minimal memory update.
    return _safe_default_selection(...)
```

Layer 3 is where most of the engineering went. The single most common real-world
failure is **unescaped quotes inside string values** — the model writes
`"answer": "She said "ship it" and moved on"` and strict parsers choke. So
`_fix_unescaped_quotes_in_strings` is a small **character-level state machine**:
it walks the JSON tracking whether it's inside a string, and escapes interior
double-quotes that aren't structural. It's boring code, and it's exactly the kind
of boring code that turns a flaky agent into a dependable one.

The point of the ladder isn't any single rung — it's that **every rung has a typed
fallback below it**, so the pipeline degrades gracefully instead of throwing. In
the worst case the user still gets the top-reranked clip and a sensible answer;
they never get a stack trace.

## The trade-off, stated honestly

Fusing selection and memory **couples two concerns**. A bug in the
memory-extraction part of the prompt can perturb the selection part, and the
schema is bigger and a little harder to evolve. That's a real cost.

EchoFind judges it worth paying, for two reasons: the latency win lands on the
user's critical path (the most valuable place to save time), and the
*can't-drift* guarantee — memory always reflecting the clip actually shown — is
worth more than the decoupling. To keep follow-ups instant, the top-3 follow-up
recommendations are then **pre-computed and cached**, so when the user taps one,
it resolves with no further model call at all.

## Takeaways

- **Look for LLM calls you can fuse.** If two sequential calls operate on the same
  context and one's output naturally contains the other's, a single structured
  call can remove a round-trip *and* a class of inconsistency bugs. Check whether
  the coupling it introduces is acceptable — here it was.
- **Order fields in your schema to think before committing.** Emitting evidence
  (quotes) before the decision (index) is free chain-of-thought.
- **Treat structured output as fallible I/O, not a contract.** Wrap it in a ladder
  — native parse → JSON mode → repair → typed default — so a malformed response
  degrades instead of crashing. A character-level quote-escaper handles the single
  most common malformation.

*EchoFind is open source. Selection and the memory schema live in
`engine/selection.py` and `engine/schemas.py`; the full pipeline orchestration is
in `engine/agent.py`.*
