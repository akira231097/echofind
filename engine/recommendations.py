"""
Recommendation system for suggesting alternative clips after selection.

Runs asynchronously AFTER the main selection to generate top 3 alternative
recommendations with pre-computed prompts and memory updates.

Uses gemini-2.5-flash-lite with no reasoning for speed.
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
RECOMMENDATION_MODEL = config.CLIP_RECOMMENDATION_MODEL
RECOMMENDATION_REASONING_EFFORT = config.CLIP_RECOMMENDATION_REASONING_EFFORT


class RecommendationItem(BaseModel):
    """Single recommendation with pre-computed response and memory updates."""

    chunk_index: int = Field(
        description="Index of the recommended chunk in the candidates list"
    )
    prompt: str = Field(
        description="Short clickable prompt (max 50 chars) that user sees"
    )
    answer: str = Field(
        description="Pre-computed natural response for this clip (2-4 sentences)"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence that this clip is relevant"
    )
    # Pre-computed memory updates
    turn_summary: str = Field(
        description="Brief summary for memory (max 150 chars)"
    )
    extracted_entities: List[str] = Field(
        default_factory=list,
        description="Key entities to add to memory (max 5)"
    )
    turn_themes: List[str] = Field(
        default_factory=list,
        description="Themes to add to memory (max 3)"
    )


class RecommendationsOutput(BaseModel):
    """Output from recommendation LLM containing top 3 alternatives."""

    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Top 3 alternative clip recommendations"
    )


def _format_chunks_for_recommendations(chunks: List[Dict[str, Any]], exclude_index: int) -> str:
    """
    Format chunks for recommendation prompt, excluding the selected one.

    Phase 1 Optimization: NO transcript truncation for full context.
    """
    formatted = []

    for i, chunk in enumerate(chunks):
        if i == exclude_index:
            continue  # Skip the already-selected chunk

        # NO TRUNCATION - use full transcript (Phase 1)
        transcript = chunk.get("chunk", "") or chunk.get("text", "") or ""
        podcast = chunk.get("podcast_title", chunk.get("podcastTitle", "Unknown"))
        episode = chunk.get("episode_title", chunk.get("episodeTitle", "Unknown"))

        # Handle speakers
        speakers = chunk.get("speakers", chunk.get("speaker", []))
        if isinstance(speakers, list):
            speakers_str = ", ".join(speakers) if speakers else "Unknown"
        else:
            speakers_str = str(speakers) if speakers else "Unknown"

        # Get guests
        guests = chunk.get("guests", [])
        if isinstance(guests, list):
            guests_str = ", ".join(guests) if guests else ""
        else:
            guests_str = str(guests) if guests else ""

        pub_date = chunk.get("published_date", chunk.get("publishedDate", ""))

        formatted.append(
            f"[{i}] Podcast: {podcast}\n"
            f"    Episode: {episode}\n"
            f"    Speakers: {speakers_str}\n"
            f"    Guests: {guests_str}\n"
            f"    Published: {pub_date}\n"
            f"    Transcript: {transcript}\n"
        )

    return "\n".join(formatted)


async def generate_recommendations(
    gemini_client,
    user_query: str,
    resolved_query: str,
    chunks: List[Dict[str, Any]],
    selected_index: int,
    selected_answer: str,
    memory: ConversationMemory,
    is_followup: bool = False,
) -> RecommendationsOutput:
    logger.info("=" * 70)
    logger.info("[RECOMMENDATIONS] Starting recommendation generation")
    logger.info("=" * 70)
    logger.info(f"[RECOMMENDATIONS] Query: {resolved_query[:60]}...")
    logger.info(f"[RECOMMENDATIONS] Excluding selected index: {selected_index}")
    logger.info(f"[RECOMMENDATIONS] Total chunks available: {len(chunks)}")

    # Need at least 2 chunks (1 selected + 1 for recommendation)
    if len(chunks) < 2:
        logger.warning("[RECOMMENDATIONS] Not enough chunks for recommendations")
        return RecommendationsOutput(recommendations=[])

    # Format remaining chunks
    chunks_text = _format_chunks_for_recommendations(chunks[:30], selected_index)

    # Memory context - no truncation with 1M context window
    memory_context = memory.render_for_prompt()

    today_str = datetime.now().strftime('%Y-%m-%d')

    # ========================================================================
    # RECOMMENDATION PROMPT - Similar structure to selection but focused on
    # generating SHORT prompts and pre-computing memory updates
    # ========================================================================

    system_prompt = f"""<persona>
You are a podcast recommendation assistant. Your job is to suggest 3 alternative clips
that might also interest the user based on their question.
</persona>

<task>
The user asked: "{user_query}"
We already showed them one clip. Now pick the TOP 3 BEST alternative clips from the remaining candidates.

For EACH recommendation, you must provide:
1. **prompt**: A SHORT clickable prompt (MAX 50 characters!) that entices the user
   - Should be a complete, natural phrase
   - Examples: "More on AI risks", "Naval's take", "Earlier discussion", "Different perspective"
   - Must relate to the clip content

2. **answer**: A natural response (2-4 sentences) describing the clip
   - Reference specific content from the transcript
   - Explain why this clip is relevant

3. **Memory updates** (for when user clicks):
   - turn_summary: Brief summary (max 150 chars)
   - extracted_entities: Key people/topics (max 5)
   - turn_themes: High-level themes (max 3)

IMPORTANT:
- Pick clips that offer DIFFERENT perspectives or additional value
- Don't pick clips that are too similar to each other
- Prompts must be SHORT and compelling (50 chars max!)
- The answer should stand alone - assume user clicked this recommendation
</task>

<context>
TODAY'S DATE: {today_str}
SELECTED CLIP ANSWER (already shown): {selected_answer[:200]}...

CONVERSATION MEMORY:
{memory_context}

This is {"a FOLLOW-UP question" if is_followup else "a NEW topic"}.
</context>

<format>
Respond with valid JSON:
{{
  "recommendations": [
    {{
      "chunk_index": <integer - the [N] index from candidates>,
      "prompt": "<SHORT clickable text, max 50 chars>",
      "answer": "<2-4 sentence response>",
      "confidence": <float 0.0-1.0>,
      "turn_summary": "<brief summary, max 150 chars>",
      "extracted_entities": ["entity1", "entity2"],
      "turn_themes": ["theme1", "theme2"]
    }},
    ... (exactly 3 recommendations)
  ]
}}
</format>"""

    user_prompt = f"""<user_question>{user_query}</user_question>

<resolved_query>{resolved_query}</resolved_query>

<remaining_candidates>
{chunks_text}
</remaining_candidates>

<instructions>
1. Review the remaining candidate clips (excluding the already-selected one)
2. Pick the TOP 3 that would best complement the user's query
3. For each, write a SHORT prompt (max 50 chars!) and pre-compute the response
4. Ensure variety - pick clips offering different angles/perspectives
5. Return valid JSON with exactly 3 recommendations
</instructions>"""

    # =======================================================================
    # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
    # =======================================================================
    try:
        logger.info(f"[RECOMMENDATIONS] Calling Gemini LLM ({RECOMMENDATION_MODEL})...")
        import time
        llm_start = time.time()

        try:
            # Try structured output first
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=RECOMMENDATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                reasoning_effort=RECOMMENDATION_REASONING_EFFORT,
                response_format=RecommendationsOutput,
                operation_name="Clip Recommendations (Structured)"
            )

            llm_time = time.time() - llm_start
            logger.info(f"[RECOMMENDATIONS] LLM response received in {llm_time:.2f}s")

            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                logger.info("[RECOMMENDATIONS] Structured output parsed successfully!")
                result = parsed_result.model_dump()
            else:
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError:
            # Fallback to json_object mode
            logger.warning("[RECOMMENDATIONS] Structured output not available, using json_object")
            resp = await llm_call_with_retry(
                gemini_client.chat.completions.create,
                model=RECOMMENDATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                reasoning_effort=RECOMMENDATION_REASONING_EFFORT,
                max_tokens=2500,
                response_format={"type": "json_object"},
                operation_name="Clip Recommendations (JSON Fallback)"
            )
            llm_time = time.time() - llm_start
            logger.info(f"[RECOMMENDATIONS] LLM response received in {llm_time:.2f}s")
            result = json.loads(resp.choices[0].message.content)

        recommendations = []
        for rec in result.get("recommendations", [])[:3]:  # Max 3
            chunk_idx = rec.get("chunk_index", 0)

            # Validate index
            if chunk_idx < 0 or chunk_idx >= len(chunks) or chunk_idx == selected_index:
                logger.warning(f"[RECOMMENDATIONS] Invalid index {chunk_idx}, skipping")
                continue

            # Truncate prompt to 50 chars
            prompt = rec.get("prompt", "Related clip")[:50]

            recommendations.append(RecommendationItem(
                chunk_index=chunk_idx,
                prompt=prompt,
                answer=rec.get("answer", "Here's another relevant clip."),
                confidence=min(1.0, max(0.0, rec.get("confidence", 0.7))),
                turn_summary=rec.get("turn_summary", "Alternative clip shown")[:150],
                extracted_entities=rec.get("extracted_entities", [])[:5],
                turn_themes=rec.get("turn_themes", [])[:3],
            ))

        logger.info(f"[RECOMMENDATIONS] Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations):
            logger.info(f"  [{i+1}] idx={rec.chunk_index} | prompt=\"{rec.prompt}\" | conf={rec.confidence}")

        return RecommendationsOutput(recommendations=recommendations)

    except Exception as e:
        logger.error(f"[RECOMMENDATIONS] LLM failed: {e}")
        return RecommendationsOutput(recommendations=[])



_pending_recommendations: Dict[str, Dict[str, Any]] = {}
_recommendations_lock = asyncio.Lock()


async def store_recommendations(
    session_id: str,
    turn_id: str,
    recommendations: RecommendationsOutput,
    chunks: List[Dict[str, Any]],
    original_question: str = "",
    resolved_query: str = "",
) -> None:
    """Store recommendations for later retrieval when user clicks."""
    async with _recommendations_lock:
        key = f"{session_id}:{turn_id}"
        _pending_recommendations[key] = {
            "recommendations": recommendations,
            "chunks": chunks,
            "original_question": original_question,
            "resolved_query": resolved_query,
            "created_at": datetime.utcnow(),
        }
        logger.info(f"[RECOMMENDATIONS] Stored {len(recommendations.recommendations)} recommendations for {key}")

        # Cleanup old entries (keep last 100)
        if len(_pending_recommendations) > 100:
            # Remove oldest entries
            sorted_keys = sorted(
                _pending_recommendations.keys(),
                key=lambda k: _pending_recommendations[k]["created_at"]
            )
            for old_key in sorted_keys[:-100]:
                del _pending_recommendations[old_key]


async def get_recommendation(
    session_id: str,
    turn_id: str,
    recommendation_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific recommendation by index.

    Returns the recommendation item, full chunk data, and original context.
    """
    async with _recommendations_lock:
        key = f"{session_id}:{turn_id}"
        stored = _pending_recommendations.get(key)

        if not stored:
            logger.warning(f"[RECOMMENDATIONS] No stored recommendations for {key}")
            return None

        recs = stored["recommendations"].recommendations
        chunks = stored["chunks"]

        if recommendation_index < 0 or recommendation_index >= len(recs):
            logger.warning(f"[RECOMMENDATIONS] Invalid recommendation index {recommendation_index}")
            return None

        rec = recs[recommendation_index]
        chunk = chunks[rec.chunk_index] if rec.chunk_index < len(chunks) else None

        if not chunk:
            logger.warning(f"[RECOMMENDATIONS] Chunk not found for index {rec.chunk_index}")
            return None

        return {
            "recommendation": rec,
            "chunk": chunk,
            "original_question": stored.get("original_question", ""),
            "resolved_query": stored.get("resolved_query", ""),
        }
