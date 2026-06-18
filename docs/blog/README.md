# EchoFind — engineering write-ups

Long-form posts on the design decisions behind EchoFind. Each is written to be
cross-posted as-is (dev.to / Medium frontmatter included) and links back to the
real code it describes.

1. **[Three-Level Reciprocal Rank Fusion](01-multi-level-rrf-weighted-hyde.md)** —
   how the retrieval core fuses dense, sparse, multi-query (weighted HyDE), and
   time-bucketed signals into one ranking using rank-only fusion, and why it
   throws magnitude away on purpose and earns it back with a reranker.

2. **[One LLM call instead of two](02-single-call-select-and-memory.md)** —
   fusing answer selection and conversation-memory updates into a single
   structured call to cut a round-trip off the critical path, plus a four-layer
   reliability ladder for structured LLM output that never hard-fails.
