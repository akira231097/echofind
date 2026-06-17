"""
Combined Selection + Memory Update in single LLM call.

Optimization: Instead of separate selection and memory update,
we do both in ONE Gemini call to save ~1-2 seconds latency.

Prompt Engineering: PTCF Framework (Persona, Task, Context, Format)
with thinking tokens enabled for better reasoning.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import config
from engine.memory import ConversationMemory
from retrieval.llm_utils import llm_call_with_retry

logger = logging.getLogger(__name__)

# Model & reasoning - load from config (centralized in config.py for easy switching)
GEMINI_MODEL = config.CLIP_SELECTION_MODEL
CLIP_SELECTION_REASONING_EFFORT = config.CLIP_SELECTION_REASONING_EFFORT


class SelectionWithMemoryOutput(BaseModel):
    """Combined output schema with enhanced context extraction (Phase 2)."""

    # ==========================================================================
    # PHASE 2: Quote Extraction Protocol
    # Research shows extracting quotes before selection improves accuracy 27% → 98%
    # ==========================================================================
    relevant_quotes: List[str] = Field(
        default_factory=list,
        description="2-4 direct quotes from documents that answer the query. "
        "Format: 'Doc X: \"exact quote...\"' - Extract BEFORE making selection!"
    )

    # Selection fields
    chosen_index: int = Field(
        description="Index (0-based) of the best chunk from candidates"
    )
    answer: str = Field(
        description="Natural response describing the chosen clip (2-4 sentences). "
        "Reference specific content from the transcript."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence that this clip answers the question"
    )

    # Memory update fields (generated in same call)
    turn_summary: str = Field(
        description="Brief summary of this answer for memory (2-3 sentences, max 500 chars)"
    )
    extracted_entities: List[str] = Field(
        default_factory=list,
        description="Key people, topics, or concepts mentioned (max 10)"
    )
    turn_themes: List[str] = Field(
        default_factory=list,
        description="High-level themes for this turn (max 5)"
    )

    # ==========================================================================
    # OPTION A: Enhanced Turn Context (for Query Analyzer)
    # These fields help resolve "that example", "the monkey thing", etc.
    # ==========================================================================
    key_quotes: List[str] = Field(
        default_factory=list,
        description="2-3 memorable/notable quotes from the SELECTED clip that user might reference later"
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="Specific topics/subtopics discussed in the clip (max 5)"
    )
    notable_examples: List[str] = Field(
        default_factory=list,
        description="Specific examples, stories, or anecdotes mentioned (max 3). "
        "E.g., 'Neuralink monkey experiment', 'OpenAI founding story'"
    )


def _repair_json(content: str) -> str:
    import re

    if not content:
        return content

    # Strip any markdown code blocks
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    content = re.sub(r',(\s*[}\]])', r'\1', content)


    content = re.sub(r'"\s*\n\s*"', '",\n"', content)


    content = re.sub(r'(\d+|true|false|null|"[^"]*"|]|})\s*\n\s*"([^"]+)":', r'\1,\n"\2":', content)

    # =======================================================================
    # FIX UNESCAPED QUOTES INSIDE STRING VALUES
    # =======================================================================
    content = _fix_unescaped_quotes_in_strings(content)

    return content


def _fix_unescaped_quotes_in_strings(content: str) -> str:
    result = []
    i = 0
    in_string = False
    string_start_col = 0  # Track column position for debugging

    while i < len(content):
        char = content[i]

        if char == '\\' and i + 1 < len(content):
            # Escaped character - copy both and skip
            result.append(char)
            result.append(content[i + 1])
            i += 2
            continue

        if char == '"':
            if not in_string:
                # Starting a string
                in_string = True
                string_start_col = i
                result.append(char)
            else:

                rest = content[i + 1:i + 20].lstrip()  # Look at next ~20 chars after whitespace


                is_terminator = False

                if rest:
                    first_char = rest[0] if rest else ''
                    # Direct terminators
                    if first_char in ',}]\n:':
                        is_terminator = True
                    # Check for newline followed by field pattern
                    elif first_char == '\n' or (i + 1 < len(content) and content[i + 1] in ' \t\n'):
                        # Look for "fieldname": pattern after whitespace
                        rest_stripped = content[i + 1:].lstrip()
                        if rest_stripped.startswith('"') or rest_stripped.startswith('}') or rest_stripped.startswith(']'):
                            is_terminator = True

                if is_terminator:
                    # This is the end of the string
                    in_string = False
                    result.append(char)
                else:
                    # This is an unescaped quote INSIDE the string - escape it
                    result.append('\\')
                    result.append('"')
        else:
            result.append(char)

        i += 1

    return ''.join(result)


async def _fallback_json_object_call(
    gemini_client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str = "low",
) -> Dict[str, Any]:
    """
    Fallback to json_object mode when structured output (beta.parse) is not available.
    Includes JSON repair logic for malformed responses.
    """
    from retrieval.llm_utils import llm_call_with_retry

    resp = await llm_call_with_retry(
        gemini_client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        reasoning_effort=reasoning_effort,
        response_format={"type": "json_object"},
        operation_name="Clip Selection (JSON Object Fallback)"
    )

    raw_content = resp.choices[0].message.content
    finish_reason = getattr(resp.choices[0], 'finish_reason', 'unknown')
    logger.debug(f"[SELECTION] Fallback raw response (first 500 chars): {raw_content[:500] if raw_content else 'None'}")
    logger.debug(f"[SELECTION] Fallback finish reason: {finish_reason}")

    if not raw_content:
        raise ValueError("Empty LLM response in fallback mode")

    # Try to parse JSON, with repair attempt on failure
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as json_err:
        logger.warning(f"[SELECTION] Fallback JSON parse failed: {json_err}")
        logger.warning(f"[SELECTION] Problematic content around error (char {json_err.pos}): ...{raw_content[max(0, json_err.pos-50):json_err.pos+50]}...")

        # Attempt to repair common JSON issues
        repaired_content = _repair_json(raw_content)
        if repaired_content != raw_content:
            logger.info("[SELECTION] Attempting JSON repair...")
            result = json.loads(repaired_content)
            logger.info("[SELECTION] JSON repair successful!")
            return result
        else:
            logger.error(f"[SELECTION] Could not repair JSON. Full raw content:\n{raw_content}")
            raise


def _build_recency_context(recency_metadata: Optional[Dict[str, Any]]) -> str:
    """
    Build context string for recency transparency in selection prompt.

    Provides guidance based on what the recency-first strategy determined.
    """
    if not recency_metadata:
        return ""

    recency_satisfied = recency_metadata.get("recency_satisfied", True)
    fallback_triggered = recency_metadata.get("fallback_triggered", False)
    topic_present = recency_metadata.get("topic_present", False)
    recency_priority = recency_metadata.get("recency_priority", "none")

    context_parts = []

    # Add recency strategy info
    if recency_priority == "hard":
        context_parts.append("""
🎯 RECENCY MODE: HARD (Pure Recency Query)
The user wants the NEWEST content. The candidates have been pre-sorted by recency.
Clips marked ⭐ MOST RECENT are the actual newest - strongly prefer these.
For this query type, RECENCY IS MORE IMPORTANT THAN TOPIC DEPTH.
Choose the newest clip that reasonably matches the entity/show requested.
""")
    elif recency_priority == "soft":
        context_parts.append("""
🎯 RECENCY MODE: SOFT (Topic + Recency Query)
The user wants recent content on a specific topic. Balance recency with relevance.
Clips marked 🕐 RECENT have recency boost applied.
Choose the NEWEST clip that ALSO matches the topic well.
""")

    # Add fallback warning if needed
    if not recency_satisfied:
        if fallback_triggered and topic_present:
            context_parts.append("""
⚠️ RECENCY FALLBACK: No recent clips matched the topic well enough.
The results below are the most RELEVANT available but may not be recent.
In your response, acknowledge this honestly.
Example: "I couldn't find recent clips on [topic], but here's a relevant discussion from [date]..."
""")
        elif fallback_triggered:
            context_parts.append("""
⚠️ RECENCY FALLBACK: The search fell back to relevance-based results.
The clips may not be the most recent available.
""")

    return "\n".join(context_parts)


def _build_match_type_context(chunks: List[Dict[str, Any]], recency_metadata: Optional[Dict[str, Any]]) -> str:
    """
    Build context string for match type transparency in selection prompt.

    This helps the LLM understand when we have exact guest matches vs mentions.
    Critical for calibrating confidence correctly.
    """
    if not chunks:
        return ""

    # Count match types
    exact_guest_count = 0
    mention_only_count = 0
    no_match_count = 0

    for chunk in chunks:
        match_type = chunk.get('person_match_type', 'unknown')
        if match_type in ('exact_guest_speaking', 'exact_guest_present'):
            exact_guest_count += 1
        elif match_type == 'mention_only':
            mention_only_count += 1
        elif match_type in ('no_match', 'host_match', 'neutral'):
            no_match_count += 1

    context_parts = []

    # Check if this is a person-specific query
    hybrid_scoring = recency_metadata.get('hybrid_scoring_applied', False) if recency_metadata else False

    if exact_guest_count > 0:
        context_parts.append(f"""
✅ GUEST MATCH STATUS: FOUND ({exact_guest_count} clips with exact guest match)
Clips marked 🎤 GUEST SPEAKING or 👤 GUEST PRESENT have the requested person as an actual guest.
STRONGLY PREFER these over clips that just MENTION the person.
""")
    elif mention_only_count > 0 and hybrid_scoring:
        context_parts.append(f"""
⚠️ GUEST MATCH STATUS: NO EXACT MATCH
The requested guest was NOT found as an actual guest in any candidate clips.
All matches are "mention only" - clips where the person is DISCUSSED but not present.

CRITICAL: In your response:
1. Acknowledge this honestly (e.g., "I couldn't find a clip of [Person] as a guest, but here's a discussion about them...")
2. SET CONFIDENCE ≤ 0.60 since this is a fallback result, not an exact match
3. If no suitable mention clip exists, say so clearly
""")
    elif hybrid_scoring and no_match_count == len(chunks):
        context_parts.append("""
❌ GUEST MATCH STATUS: NO MATCHES
None of the candidate clips feature the requested person as guest or mention them.
SET CONFIDENCE ≤ 0.40 and acknowledge the mismatch in your response.
""")

    return "\n".join(context_parts)


async def select_and_update_memory(
    gemini_client,
    user_query: str,
    resolved_query: str,
    chunks: List[Dict[str, Any]],
    memory: ConversationMemory,
    is_followup: bool = False,
    recency_metadata: Optional[Dict[str, Any]] = None,
) -> SelectionWithMemoryOutput:
    """
    Single LLM call for selection + memory update.

    Args:
        gemini_client: Gemini client
        user_query: Original user question
        resolved_query: Query with pronouns resolved
        chunks: Reranked chunks from Cohere
        memory: Current conversation memory
        is_followup: Whether this is a follow-up question
        recency_metadata: Optional dict with recency_satisfied, fallback_triggered flags

    Returns:
        SelectionWithMemoryOutput with chosen chunk and memory update data
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("[CLIP SELECTION] 🎯 STARTING CLIP SELECTION PROCESS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[CLIP SELECTION] 📥 INPUT SUMMARY:")
    logger.info(f"  ├─ Original question: \"{user_query}\"")
    logger.info(f"  ├─ After pronoun resolution: \"{resolved_query}\"")
    logger.info(f"  ├─ Is this a follow-up? {'Yes ✓' if is_followup else 'No (new topic)'}")
    logger.info(f"  ├─ Candidate clips to evaluate: {len(chunks)}")
    logger.info(f"  └─ Conversation turn: #{memory.turn_count + 1}")

    # Log recency context if present
    if recency_metadata:
        recency_mode = recency_metadata.get('recency_priority', 'none')
        logger.info("")
        logger.info("[CLIP SELECTION] ⏰ RECENCY CONTEXT:")
        if recency_mode == 'hard':
            logger.info("  └─ Mode: HARD RECENCY - User wants the NEWEST clip (e.g., 'latest', 'most recent')")
        elif recency_mode == 'soft':
            logger.info("  └─ Mode: SOFT RECENCY - Balance between topic relevance and recency")
        else:
            logger.info("  └─ Mode: Standard - Prioritizing topic relevance over recency")

    if not chunks:
        logger.warning("[SELECTION] No chunks provided, returning empty result")
        return SelectionWithMemoryOutput(
            relevant_quotes=[],
            chosen_index=0,
            answer="I couldn't find a relevant clip for this question.",
            confidence=0.0,
            turn_summary="No results found",
            extracted_entities=[],
            turn_themes=[],
            key_quotes=[],
            topics_covered=[],
            notable_examples=[],
        )

    # Format chunks for prompt - use config for limit (Phase 1: increased from 25 to 40)
    chunk_limit = getattr(config, 'LLM_TOP_K', 40)
    chunks_to_evaluate = chunks[:chunk_limit]
    chunks_text = _format_chunks_for_selection(chunks_to_evaluate)

    logger.info("")
    logger.info("[CLIP SELECTION] 📋 PREPARING CONTEXT FOR LLM:")
    logger.info(f"  ├─ Clips being sent to LLM: {len(chunks_to_evaluate)} (limit: {chunk_limit})")

    # Log top candidates for visibility
    logger.info("  ├─ Top 5 candidate clips:")
    for i, chunk in enumerate(chunks_to_evaluate[:5]):
        podcast = chunk.get('podcast_title', 'Unknown')[:25]
        episode = chunk.get('episode_title', 'Unknown')[:35]
        score = chunk.get('rerank_score', chunk.get('score', 0))
        date = chunk.get('published_date', 'Unknown')[:10]
        logger.info(f"  │   [{i}] {podcast} - {episode}... (score: {score:.3f}, date: {date})")

    # Memory context - no truncation with 1M context window
    memory_context = memory.render_for_prompt()
    logger.info(f"  └─ Memory context size: {len(memory_context):,} chars (~{len(memory_context)//4:,} tokens)")

    # ========================================================================
    # PTCF Framework Prompt: Persona, Task, Context, Format
    # Using XML-style sections for clarity with Gemini
    # ========================================================================

    # Get today's date for temporal context
    today_str = datetime.now().strftime('%Y-%m-%d')

    system_prompt = f"""<persona>
You are an expert podcast clip curator and conversation memory manager. You have deep expertise in:
- Understanding podcast content and conversational context
- Matching user questions to relevant audio clips with high precision
- Tracking conversation history for coherent multi-turn interactions
- Extracting key entities and themes for future reference
</persona>

<task>
Your primary objective is to perform THREE critical tasks in one response:

TASK 1 - CLIP SELECTION:
Analyze each candidate clip and select the ONE that BEST answers the user's question.

**Definition - Chunk:**
A chunk is a contiguous conversation segment from a single podcast episode.
It contains full transcript text plus metadata (podcast, episode, published date, guests, hosts, speakers).

**What "depth & nuance" means:**
- Multi-step reasoning and causal chains (why/how, not just what)
- Specifics: examples, mechanisms, numbers, named entities, caveats, trade-offs
- Clear stance or insight (not just superficial mentions or teasers)

**Selection Rubric (score each 0-1, then decide):**
- Relevance: Directly answers the user's exact question
- Depth/Nuance: Demonstrates reasoning, trade-offs, mechanisms, or rich detail
- Completeness: The chunk alone substantially addresses the query
- Authority: First-person from the named subject(s) or authoritative analysis
- Temporal_fit: Align with time intent if implied (see below)
- Coherence: Understandable standalone (minimal missing context)
- Length_preference: Prefer longer, meatier segments (2-7 minutes); use text density as proxy if duration unknown

**Context-Aware Selection:**
- Consider the podcast and episode context when evaluating relevance
- Prioritize chunks from episodes that thematically align with the query
- Use guest/host/speakers metadata to ensure the right voices are present
- The "speakers" field shows who is actually speaking in this specific chunk

**Speaker Priority (CRITICAL IF APPLICABLE):**
- If the query mentions a specific person (e.g., "Elon Musk's ideas"), strongly prioritize chunks where that person is listed as a guest AND is speaking directly (look for first-person: I, my, we)
- Use the Guests/Hosts/Speakers metadata to identify who is speaking
- Avoid chunks merely ABOUT the person unless no direct speech is available

**Temporal Guidance (CRITICAL FOR RECENCY QUERIES):**
- TODAY'S DATE IS {today_str}. Use this to evaluate recency and temporal alignment.
- Treat words like "latest", "newest", "recent", "up to date", "current", "most recent" as a demand for the MOST RECENTLY PUBLISHED chunk that still answers the question
- Among otherwise qualified chunks, you MUST pick the one with the NEWEST Published Date
- Treat words like "oldest", "earliest", "first", "original" as a demand for the EARLIEST Published Date
- When the query specifies a timeframe (e.g., "before 2022", "after June 2023", "last 6 months"), reject chunks whose Published Date falls outside that window
- If a chunk has no Published Date while another satisfies the temporal intent with a known date, prefer the dated chunk

*** CRITICAL RECENCY INDICATORS ***
- Clips marked with ⭐ MOST RECENT or 🕐 RECENT have been pre-sorted by our recency algorithm
- For "latest" queries: Strongly prefer clips with these markers - they are the ACTUAL newest content
- Episode numbers like "#2419" vs "#2409" indicate recency - HIGHER numbers = MORE RECENT
- Compare Published Dates directly: 2024-11-25 is newer than 2024-11-10
- For pure recency queries (no topic), the clip at index [0] with MOST RECENT marker should usually win

*** IMPORTANT: FOR RECENCY/LATEST QUERIES, CHECK PUBLISHED DATE AND CHOOSE THE NEWEST AMONG EQUIVALENTLY RELEVANT CHUNKS ***
- Whenever temporal intent is present, MENTION the chosen chunk's Published Date in your answer

**Hard Preferences and Tie-breakers:**
1. The chunk must directly answer the query (hard filter)
2. If multiple are equally relevant, pick the one with greater depth/nuance
3. If still tied, pick the one with first-person authority from the named person
4. If still tied, prefer longer (2-7 min) while staying on-topic and coherent
5. For explicit temporal language ("latest", "newest", "recent"), break ties in favor of newest Published Date; for "oldest"/"earliest" language, break ties using the oldest Published Date; otherwise avoid penalizing older material that better answers the question

TASK 2 - RESPONSE GENERATION:
Write the "answer" field as a conversational response (2-4 sentences) that:
- Answers the user's question based ONLY on the SELECTED chunk (chosen_index)
- Speaks naturally as a helpful assistant - DO NOT mention "document", "clip", "Doc X", "chunk", or any internal references
- Includes specific quotes or paraphrased content from the SELECTED transcript
- For temporal queries, mention the date naturally (e.g., "In a November 2024 conversation...")
- For follow-ups, connects naturally to what was discussed before

CRITICAL - SPEAKER ATTRIBUTION RULES:
⚠️ USE THE METADATA to identify WHO is speaking in the selected chunk!
- Check <guests>, <hosts>, and <speakers_in_chunk> fields
- The GUEST is usually the expert/interviewee - attribute their words to THEM, not just the host
- If the transcript shows the guest explaining something, write: "[Guest Name] explains that..."
- Don't just default to "Joe Rogan discusses..." - check who the GUEST is!

Examples:
❌ WRONG: "Joe Rogan discusses IShowSpeed..." (when Ehsan Ahmad is the guest speaking)
✅ CORRECT: "In his conversation with Ehsan Ahmad, Joe Rogan and his guest discuss IShowSpeed..."
✅ CORRECT: "Ehsan Ahmad tells Joe Rogan about IShowSpeed's origins..."

❌ NEVER write: "In Document 1...", "Doc 7 states...", "The clip shows..."
❌ NEVER reference multiple documents in the answer - use ONLY the selected one
✅ DO write: "Elon Musk explains to Joe Rogan that..." (when Elon is the guest)
✅ DO write: "Naval Ravikant shares his perspective on..." (when Naval is the guest)

TASK 3 - MEMORY UPDATE:
Extract information for conversation continuity:
- turn_summary: Brief summary for future context (max 150 chars)
- extracted_entities: Key people, topics, concepts (max 5)
- turn_themes: High-level themes like "AI safety", "entrepreneurship" (max 3)
</task>

<context>
TODAY'S DATE: {today_str}

CONVERSATION MEMORY:
{memory_context}

This is {"a FOLLOW-UP question - the user is continuing a previous topic" if is_followup else "a NEW topic - no direct connection to previous conversation"}.
{_build_recency_context(recency_metadata)}
{_build_match_type_context(chunks, recency_metadata)}
</context>

<format>
You MUST respond with valid JSON matching this exact schema:
{{
  "relevant_quotes": ["Doc 0: \"exact quote...\"", "Doc 3: \"another quote...\""],
  "chosen_index": <integer 0 to N-1>,
  "answer": "<conversational 2-4 sentence response - NO document references!>",
  "confidence": <float 0.0 to 1.0>,
  "turn_summary": "<summary for memory, max 500 chars>",
  "extracted_entities": ["<entity1>", "<entity2>", ...],
  "turn_themes": ["<theme1>", "<theme2>", ...],
  "key_quotes": ["memorable quote 1", "memorable quote 2"],
  "topics_covered": ["specific topic 1", "specific topic 2", ...],
  "notable_examples": ["example/story 1", "example/story 2", ...]
}}

=== PHASE 2: QUOTE EXTRACTION PROTOCOL ===
CRITICAL: Extract relevant_quotes FIRST to help you select the best document.
1. Scan ALL documents for quotes that answer the query
2. List 2-4 exact quotes with "Doc X:" prefix (for YOUR internal selection process)
3. Select the best document based on these quotes
4. Write the "answer" field based ONLY on the selected document - NEVER mention "Doc X" in the answer!

NOTE: The "relevant_quotes" field is for YOUR selection process only.
The "answer" field is shown directly to the user - it must be conversational with NO document references.

=== OPTION A: ENHANCED CONTEXT FIELDS ===
These fields help future queries resolve references like "that example", "the monkey thing":
- key_quotes: 2-3 memorable quotes from the SELECTED clip (for future reference)
- topics_covered: Specific topics discussed (not just themes)
- notable_examples: Stories, experiments, anecdotes mentioned (e.g., "Neuralink monkey experiment")

CONFIDENCE SCORING GUIDE:
- 0.9-1.0: Perfect match, directly answers the question with authority
- 0.7-0.9: Good match, answers most of the question well
- 0.5-0.7: Partial match, related but incomplete
- 0.3-0.5: Weak match, tangentially related
- 0.0-0.3: Poor match, only superficial connection
</format>"""

    # ==========================================================================
    # PROMPT STRUCTURE OPTIMIZATION (Phase 1)
    # Research: "Put longform data at the top... Queries at the end can improve
    # response quality by up to 30%" - Anthropic Long Context Tips
    # ==========================================================================
    user_prompt = f"""<documents>
{chunks_text}
</documents>

<conversation_memory>
{memory_context}
</conversation_memory>

<context>
  <is_followup>{is_followup}</is_followup>
  <total_candidates>{len(chunks)}</total_candidates>
  <valid_index_range>0 to {len(chunks)-1}</valid_index_range>
</context>

<instructions>
=== QUOTE EXTRACTION PROTOCOL (Do this FIRST!) ===
STEP 1: Search ALL {len(chunks)} documents for quotes that answer: "{resolved_query}"
STEP 2: Extract 2-4 exact quotes with their document index (e.g., "Doc 3: ...")
STEP 3: Put these in the "relevant_quotes" field

=== THEN SELECT AND RESPOND ===
STEP 4: Based on your quotes, select the BEST document (chosen_index = 0 to {len(chunks)-1})
STEP 5: Write a natural response (2-4 sentences) referencing the content
STEP 6: Extract memory fields:
  - turn_summary: 2-3 sentence summary (max 500 chars)
  - extracted_entities: People, topics mentioned (max 10)
  - turn_themes: High-level themes (max 5)
  - key_quotes: 2-3 memorable quotes from SELECTED clip
  - topics_covered: Specific topics discussed (max 5)
  - notable_examples: Stories/anecdotes mentioned (max 3)

Return valid JSON with ALL fields.
</instructions>

<query>
  <original>{user_query}</original>
  <resolved>{resolved_query}</resolved>
</query>"""

    try:
        logger.info("")
        logger.info("[CLIP SELECTION] 🤖 CALLING LLM FOR SELECTION:")
        logger.info(f"  ├─ Model: {GEMINI_MODEL}")
        logger.info("  ├─ Using Quote Extraction Protocol (extracts quotes BEFORE selecting)")
        logger.info("  └─ Waiting for response...")
        import time
        llm_start = time.time()

        # =======================================================================
        # STRUCTURED OUTPUT: Use beta.chat.completions.parse() with Pydantic
        # This GUARANTEES valid JSON matching the schema - no parsing errors!
        # See: https://ai.google.dev/gemini-api/docs/structured-output
        # =======================================================================
        try:
            # Try structured output first (guarantees schema adherence)
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                reasoning_effort=CLIP_SELECTION_REASONING_EFFORT,
                response_format=SelectionWithMemoryOutput,  # Pydantic model
                operation_name="Clip Selection (Structured)"
            )

            llm_time = time.time() - llm_start
            logger.info(f"[SELECTION] LLM response received in {llm_time:.2f}s (structured output)")

            # With structured output, parsing is automatic and guaranteed
            finish_reason = getattr(resp.choices[0], 'finish_reason', 'unknown')
            logger.debug(f"[SELECTION] Finish reason: {finish_reason}")

            # Get the pre-parsed result directly
            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                logger.info("[SELECTION] Structured output parsed successfully!")
                result = parsed_result.model_dump()
            else:
                # Fallback to content if parsed is None (shouldn't happen)
                logger.warning("[SELECTION] Parsed result is None, falling back to content parsing")
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError as attr_err:
            # beta.chat.completions.parse not available - fall back to json_object mode
            logger.warning(f"[SELECTION] Structured output not available ({attr_err}), using json_object fallback")
            resp = await _fallback_json_object_call(
                gemini_client, system_prompt, user_prompt, GEMINI_MODEL, CLIP_SELECTION_REASONING_EFFORT
            )
            llm_time = time.time() - llm_start
            logger.info(f"[SELECTION] LLM response received in {llm_time:.2f}s (json_object fallback)")
            result = resp  # Already parsed by fallback function

        # Validate index
        chosen_index = result.get("chosen_index", 0)
        if chosen_index < 0 or chosen_index >= len(chunks):
            logger.warning(f"[SELECTION] Invalid index {chosen_index}, using 0")
            chosen_index = 0

        # Validate and normalize confidence (LLM sometimes returns 1-10 scale)
        raw_confidence = result.get("confidence", 0.7)
        if raw_confidence > 1.0:
            confidence = raw_confidence / 10.0  # Convert 1-10 scale to 0-1
            logger.info(f"[SELECTION] Normalized confidence: {raw_confidence} -> {confidence}")
        else:
            confidence = max(0.0, min(1.0, raw_confidence))

        # Log the selection result with Phase 2 fields
        chosen_chunk = chunks[chosen_index] if chosen_index < len(chunks) else {}
        relevant_quotes = result.get("relevant_quotes", [])

        logger.info("")
        logger.info("[CLIP SELECTION] ✅ SELECTION COMPLETE!")
        logger.info("=" * 70)

        # Quote Extraction Results (Phase 2)
        logger.info("")
        logger.info("[CLIP SELECTION] 📝 QUOTE EXTRACTION RESULTS:")
        logger.info(f"  ├─ Quotes extracted: {len(relevant_quotes)}")
        if relevant_quotes:
            for i, quote in enumerate(relevant_quotes[:3]):
                logger.info(f"  │   [{i+1}] \"{quote[:100]}{'...' if len(quote) > 100 else ''}\"")
            logger.info("  └─ (Quotes were extracted BEFORE selection to improve accuracy)")
        else:
            logger.info("  └─ No quotes extracted (may affect selection accuracy)")

        # Selection Decision
        logger.info("")
        logger.info("[CLIP SELECTION] 🎬 SELECTED CLIP:")
        logger.info(f"  ├─ Index: [{chosen_index}] out of {len(chunks)} candidates")
        logger.info(f"  ├─ Episode: \"{chosen_chunk.get('episode_title', 'Unknown')[:60]}\"")
        logger.info(f"  ├─ Podcast: {chosen_chunk.get('podcast_title', 'Unknown')}")
        logger.info(f"  ├─ Published: {chosen_chunk.get('published_date', 'Unknown')[:10]}")
        logger.info(f"  └─ Confidence: {confidence:.1%} {'✓ High' if confidence >= 0.7 else '⚠ Medium' if confidence >= 0.5 else '⚠ Low'}")

        # Response Preview
        logger.info("")
        logger.info("[CLIP SELECTION] 💬 RESPONSE TO USER:")
        answer_preview = result.get('answer', '')[:150]
        logger.info(f"  └─ \"{answer_preview}{'...' if len(result.get('answer', '')) > 150 else ''}\"")

        # Memory Update Preview (Option A Enhanced Fields)
        key_quotes_mem = result.get('key_quotes', [])
        topics_covered_mem = result.get('topics_covered', [])
        notable_examples_mem = result.get('notable_examples', [])

        logger.info("")
        logger.info("[CLIP SELECTION] 🧠 MEMORY UPDATE (for follow-up queries):")
        logger.info(f"  ├─ Summary: \"{result.get('turn_summary', '')[:80]}...\"")
        logger.info(f"  ├─ Entities: {result.get('extracted_entities', [])[:5]}")
        logger.info(f"  ├─ Themes: {result.get('turn_themes', [])[:3]}")
        logger.info("  │")
        logger.info("  │  [ENHANCED CONTEXT - Option A] (helps resolve 'that example', 'the thing he said')")
        if key_quotes_mem:
            logger.info(f"  ├─ Key quotes stored: {len(key_quotes_mem)}")
            for q in key_quotes_mem[:2]:
                logger.info(f"  │     • \"{q[:60]}...\"")
        if topics_covered_mem:
            logger.info(f"  ├─ Topics covered: {topics_covered_mem[:4]}")
        if notable_examples_mem:
            logger.info(f"  └─ Notable examples: {notable_examples_mem[:3]}")
        else:
            logger.info(f"  └─ Notable examples: (none extracted)")

        logger.info("")
        logger.info("=" * 70)

        # Use config limits for entities and themes
        max_entities = getattr(config, 'MAX_TRACKED_ENTITIES', 15)
        max_themes = getattr(config, 'MAX_TRACKED_THEMES', 8)
        max_summary = getattr(config, 'MAX_TURN_SUMMARY_CHARS', 500)

        # Fix: Use 'or' to handle empty string answers, not just missing keys
        answer_text = result.get("answer", "").strip()
        if not answer_text:
            logger.warning("[SELECTION] ⚠️ LLM returned empty answer! Using fallback response.")
            answer_text = "Here's a relevant clip that addresses your question."

        return SelectionWithMemoryOutput(
            # Phase 2: Quote extraction results
            relevant_quotes=relevant_quotes[:4],
            # Selection fields
            chosen_index=chosen_index,
            answer=answer_text,
            confidence=confidence,
            # Memory fields with updated limits
            turn_summary=result.get("turn_summary", user_query[:100])[:max_summary],
            extracted_entities=result.get("extracted_entities", [])[:max_entities],
            turn_themes=result.get("turn_themes", [])[:max_themes],
            # Option A: Enhanced context fields
            key_quotes=result.get("key_quotes", [])[:3],
            topics_covered=result.get("topics_covered", [])[:5],
            notable_examples=result.get("notable_examples", [])[:3],
        )

    except json.JSONDecodeError as json_err:
        logger.error(f"[SELECTION] JSON parsing failed: {json_err}")
        logger.error(f"[SELECTION] Error at position {json_err.pos}, line {json_err.lineno}, column {json_err.colno}")
        # Fallback to first chunk with context from query
        fallback_entities = _extract_entities_from_query(resolved_query)
        return SelectionWithMemoryOutput(
            relevant_quotes=[],
            chosen_index=0,
            answer="Here's a clip that may be relevant to your question.",
            confidence=0.5,
            turn_summary=f"Search: {user_query[:200]}",
            extracted_entities=fallback_entities,
            turn_themes=[],
            key_quotes=[],
            topics_covered=[],
            notable_examples=[],
        )
    except Exception as e:
        import traceback
        logger.error(f"[SELECTION] LLM failed: {type(e).__name__}: {e}")
        logger.error(f"[SELECTION] Traceback: {traceback.format_exc()}")
        # Fallback to first chunk with context from query
        fallback_entities = _extract_entities_from_query(resolved_query)
        return SelectionWithMemoryOutput(
            relevant_quotes=[],
            chosen_index=0,
            answer="Here's a clip that may be relevant.",
            confidence=0.5,
            turn_summary=f"Search: {user_query[:200]}",
            extracted_entities=fallback_entities,
            turn_themes=[],
            key_quotes=[],
            topics_covered=[],
            notable_examples=[],
        )


def select_and_update_memory_sync(
    gemini_client,
    user_query: str,
    resolved_query: str,
    chunks: List[Dict[str, Any]],
    memory: ConversationMemory,
    is_followup: bool = False,
    recency_metadata: Optional[Dict[str, Any]] = None,
) -> SelectionWithMemoryOutput:
    """
    Synchronous version of select_and_update_memory.
    Use this when not in an async context.
    """
    return asyncio.run(select_and_update_memory(
        gemini_client,
        user_query,
        resolved_query,
        chunks,
        memory,
        is_followup,
        recency_metadata,
    ))


def _extract_entities_from_query(query: str) -> List[str]:
    """
    Extract likely entity names from query for fallback memory update.
    Simple heuristic: capitalized multi-word phrases that look like names.
    """
    import re

    entities = []
    # Find capitalized words that could be names (2+ consecutive capitalized words)
    name_patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
    entities.extend(name_patterns[:3])

    # Also find single capitalized words that are likely names (not common words)
    common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}
    single_caps = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
    for word in single_caps:
        if word.lower() not in common_words and word not in entities and len(entities) < 5:
            entities.append(word)

    return entities[:5]


def _extract_episode_number(title: str) -> Optional[int]:
    """Extract episode number from title (e.g., 'JRE #2419' -> 2419)."""
    import re
    if not title:
        return None
    # Match patterns like #1234, Episode 1234, Ep 1234, E1234
    patterns = [
        r'#(\d+)',           # #2419
        r'Episode\s*(\d+)',  # Episode 2419
        r'Ep\.?\s*(\d+)',    # Ep 2419, Ep. 2419
        r'\bE(\d+)\b',       # E2419
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _format_chunks_for_selection(chunks: List[Dict[str, Any]]) -> str:
    """
    Format chunks with XML structure for optimal LLM comprehension.

    Phase 1 Optimization:
    - NO transcript truncation (use full content)
    - XML tags for clear document boundaries
    - Structured metadata for better parsing

    Research: XML structure improves LLM accuracy by providing clear boundaries.
    """
    formatted = []
    for i, chunk in enumerate(chunks):
        # NO TRUNCATION - use full transcript for maximum context
        # Gemini 2.5 Flash Lite has 1M token context window
        transcript = chunk.get("chunk", "") or chunk.get("text", "") or ""

        podcast = chunk.get("podcast_title", chunk.get("podcastTitle", "Unknown"))
        episode = chunk.get("episode_title", chunk.get("episodeTitle", "Unknown"))

        # Handle speakers - could be list or string
        speakers = chunk.get("speakers", chunk.get("speaker", []))
        if isinstance(speakers, list):
            speakers_str = ", ".join(speakers) if speakers else "Unknown"
        else:
            speakers_str = str(speakers) if speakers else "Unknown"

        # Get guests and hosts
        guests = chunk.get("guests", [])
        if isinstance(guests, list):
            guests_str = ", ".join(guests) if guests else ""
        else:
            guests_str = str(guests) if guests else ""

        hosts = chunk.get("hosts", [])
        if isinstance(hosts, list):
            hosts_str = ", ".join(hosts) if hosts else ""
        else:
            hosts_str = str(hosts) if hosts else ""

        # Get published date
        pub_date = chunk.get("published_date", chunk.get("publishedDate", ""))

        # Extract episode number for recency comparison
        ep_num = _extract_episode_number(episode)

        # Get recency metadata if available (from recency-first strategy)
        recency_boost = chunk.get("recency_boost")
        recency_marker = ""
        if recency_boost is not None:
            if recency_boost >= 1.5:
                recency_marker = "MOST_RECENT"
            elif recency_boost >= 1.3:
                recency_marker = "VERY_RECENT"
            elif recency_boost >= 1.1:
                recency_marker = "RECENT"

        # Get match type metadata (from hybrid scoring)
        match_type = chunk.get("person_match_type", "unknown")
        match_marker = ""
        if match_type == "exact_guest_speaking":
            match_marker = "GUEST_SPEAKING"
        elif match_type == "exact_guest_present":
            match_marker = "GUEST_PRESENT"
        elif match_type == "mention_only":
            match_marker = "MENTIONED_ONLY"
        elif match_type == "host_match":
            match_marker = "HOST_MATCH"

        # Get scores for transparency
        hybrid_score = chunk.get("hybrid_score", 0)
        rerank_score = chunk.get("rerank_score", 0)

        # XML-structured format for better LLM parsing
        formatted.append(f"""<document index="{i}">
  <metadata>
    <podcast>{podcast}</podcast>
    <episode>{episode}</episode>
    <episode_number>{ep_num or 'N/A'}</episode_number>
    <published_date>{pub_date}</published_date>
    <hosts>{hosts_str}</hosts>
    <guests>{guests_str}</guests>
    <speakers_in_chunk>{speakers_str}</speakers_in_chunk>
    <relevance_score>{hybrid_score:.3f}</relevance_score>
    <rerank_score>{rerank_score:.3f}</rerank_score>
    <recency_marker>{recency_marker}</recency_marker>
    <match_type>{match_marker}</match_type>
  </metadata>
  <transcript>
{transcript}
  </transcript>
</document>""")

    return "\n".join(formatted)
