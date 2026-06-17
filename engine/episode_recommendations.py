"""
Episode recommendation system for suggesting alternative episodes after selection.

Runs asynchronously AFTER the main episode selection to generate top 3 alternative
episode recommendations with pre-computed prompts and memory updates.

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
EPISODE_RECOMMENDATION_MODEL = config.EPISODE_RECOMMENDATION_MODEL
EPISODE_RECOMMENDATION_REASONING_EFFORT = config.EPISODE_RECOMMENDATION_REASONING_EFFORT


class EpisodeRecommendationItem(BaseModel):
    """Single episode recommendation with pre-computed response and memory updates."""

    episode_index: int = Field(
        description="Index of the recommended episode in the candidates list"
    )
    prompt: str = Field(
        description="Short clickable prompt (max 50 chars) that user sees"
    )
    answer: str = Field(
        description="Pre-computed natural response for this episode (2-4 sentences)"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence that this episode is relevant"
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


class EpisodeRecommendationsOutput(BaseModel):
    """Output from recommendation LLM containing top 3 alternative episodes."""

    recommendations: List[EpisodeRecommendationItem] = Field(
        default_factory=list,
        description="Top 3 alternative episode recommendations"
    )


def _format_episodes_for_recommendations(
    episodes: List[Dict[str, Any]],
    episode_descriptions: Dict[str, Dict],
    exclude_index: int
) -> str:
    """
    Format episodes for recommendation prompt, excluding the selected one.

    Phase 1 Optimization: NO description truncation for full context.
    """
    formatted = []

    for i, ep in enumerate(episodes):
        if i == exclude_index:
            continue  # Skip the already-selected episode

        # Get description from RDS data - NO TRUNCATION (Phase 1)
        desc_data = episode_descriptions.get(ep.get("episode_id"), {})
        description = desc_data.get("description", "")

        podcast = ep.get("podcast_title", "Unknown")
        episode_title = ep.get("episode_title", "Unknown")

        # Handle hosts
        hosts = ep.get("hosts", [])
        if isinstance(hosts, list):
            hosts_str = ", ".join(hosts) if hosts else "Unknown"
        else:
            hosts_str = str(hosts) if hosts else "Unknown"

        # Get guests
        guests = ep.get("guests", [])
        if isinstance(guests, list):
            guests_str = ", ".join(guests) if guests else ""
        else:
            guests_str = str(guests) if guests else ""

        pub_date = ep.get("published_date") or desc_data.get("published_date") or "Unknown"

        formatted.append(
            f"[{i}] Episode: {episode_title}\n"
            f"    Podcast: {podcast}\n"
            f"    Published: {pub_date}\n"
            f"    Hosts: {hosts_str}\n"
            f"    Guests: {guests_str}\n"
            f"    Description: {description or 'No description available'}\n"
        )

    return "\n".join(formatted)


async def generate_episode_recommendations(
    gemini_client,
    user_query: str,
    resolved_query: str,
    episodes: List[Dict[str, Any]],
    episode_descriptions: Dict[str, Dict],
    selected_index: int,
    selected_response: str,
    memory: ConversationMemory,
) -> EpisodeRecommendationsOutput:
    """
    Generate top 3 alternative episode recommendations from remaining candidates.

    This runs AFTER the main episode selection completes and should not block the response.
    Uses a fast model with no reasoning for minimal latency.

    Args:
        gemini_client: Gemini client
        user_query: Original user question
        resolved_query: Query with pronouns resolved
        episodes: All candidate episodes (scored)
        episode_descriptions: Episode descriptions from RDS
        selected_index: Index of the episode that was already selected
        selected_response: The response text that was given for the selected episode
        memory: Current conversation memory

    Returns:
        EpisodeRecommendationsOutput with up to 3 alternative recommendations
    """
    logger.info("=" * 70)
    logger.info("[EPISODE_RECOMMENDATIONS] Starting episode recommendation generation")
    logger.info("=" * 70)
    logger.info(f"[EPISODE_RECOMMENDATIONS] Query: {resolved_query[:60]}...")
    logger.info(f"[EPISODE_RECOMMENDATIONS] Excluding selected index: {selected_index}")
    logger.info(f"[EPISODE_RECOMMENDATIONS] Total episodes available: {len(episodes)}")

    # Need at least 2 episodes (1 selected + 1 for recommendation)
    if len(episodes) < 2:
        logger.warning("[EPISODE_RECOMMENDATIONS] Not enough episodes for recommendations")
        return EpisodeRecommendationsOutput(recommendations=[])

    # Format remaining episodes (top 15 for recommendations)
    episodes_text = _format_episodes_for_recommendations(
        episodes[:15], episode_descriptions, selected_index
    )

    # Memory context - no truncation with 1M context window
    memory_context = memory.render_for_prompt()

    today_str = datetime.now().strftime('%Y-%m-%d')

    # ========================================================================
    # EPISODE RECOMMENDATION PROMPT
    # ========================================================================

    system_prompt = f"""<persona>
You are a podcast recommendation assistant. Your job is to suggest 3 alternative episodes
that might also interest the user based on their question.
</persona>

<task>
The user asked: "{user_query}"
We already showed them one episode. Now pick the TOP 3 BEST alternative episodes from the remaining candidates.

For EACH recommendation, you must provide:
1. **prompt**: A SHORT clickable prompt (MAX 50 characters!) that entices the user
   - Should be a complete, natural phrase
   - Examples: "Another take on weight loss", "Earlier interview", "More from this guest"
   - Must relate to the episode content

2. **answer**: A natural response (2-4 sentences) describing the episode
   - Reference specific content from the description
   - Explain WHAT the episode covers and WHY it's relevant
   - DO NOT use generic phrases like "I found X" - be specific about the content!

3. **Memory updates** (for when user clicks):
   - turn_summary: Brief summary (max 150 chars)
   - extracted_entities: Key people/topics (max 5)
   - turn_themes: High-level themes (max 3)

IMPORTANT:
- Pick episodes that offer DIFFERENT perspectives or additional value
- Don't pick episodes that are too similar to each other
- Prompts must be SHORT and compelling (50 chars max!)
- The answer should describe what the episode is about, not just announce it
</task>

<context>
TODAY'S DATE: {today_str}
SELECTED EPISODE RESPONSE (already shown): {selected_response[:250]}...

CONVERSATION MEMORY:
{memory_context}
</context>

<format>
Respond with valid JSON:
{{
  "recommendations": [
    {{
      "episode_index": <integer - the [N] index from candidates>,
      "prompt": "<SHORT clickable text, max 50 chars>",
      "answer": "<2-4 sentence response describing episode content>",
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

<remaining_episodes>
{episodes_text}
</remaining_episodes>

<instructions>
1. Review the remaining episode candidates (excluding the already-selected one)
2. Pick the TOP 3 that would best complement the user's query
3. For each, write a SHORT prompt (max 50 chars!) and pre-compute a response that describes the episode content
4. Ensure variety - pick episodes offering different angles/perspectives/guests
5. Return valid JSON with exactly 3 recommendations
</instructions>"""

    # =======================================================================
    # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
    # =======================================================================
    try:
        logger.info(f"[EPISODE_RECOMMENDATIONS] Calling Gemini LLM ({EPISODE_RECOMMENDATION_MODEL})...")
        import time
        llm_start = time.time()

        try:
            # Try structured output first
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=EPISODE_RECOMMENDATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                reasoning_effort=EPISODE_RECOMMENDATION_REASONING_EFFORT,
                response_format=EpisodeRecommendationsOutput,
                operation_name="Episode Recommendations (Structured)"
            )

            llm_time = time.time() - llm_start
            logger.info(f"[EPISODE_RECOMMENDATIONS] LLM response received in {llm_time:.2f}s")

            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                logger.info("[EPISODE_RECOMMENDATIONS] Structured output parsed successfully!")
                result = parsed_result.model_dump()
            else:
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError:
            # Fallback to json_object mode
            logger.warning("[EPISODE_RECOMMENDATIONS] Structured output not available, using json_object")
            resp = await llm_call_with_retry(
                gemini_client.chat.completions.create,
                model=EPISODE_RECOMMENDATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                reasoning_effort=EPISODE_RECOMMENDATION_REASONING_EFFORT,
                max_tokens=2500,
                response_format={"type": "json_object"},
                operation_name="Episode Recommendations (JSON Fallback)"
            )
            llm_time = time.time() - llm_start
            logger.info(f"[EPISODE_RECOMMENDATIONS] LLM response received in {llm_time:.2f}s")
            result = json.loads(resp.choices[0].message.content)

        recommendations = []
        for rec in result.get("recommendations", [])[:3]:  # Max 3
            episode_idx = rec.get("episode_index", 0)

            # Validate index
            if episode_idx < 0 or episode_idx >= len(episodes) or episode_idx == selected_index:
                logger.warning(f"[EPISODE_RECOMMENDATIONS] Invalid index {episode_idx}, skipping")
                continue

            # Truncate prompt to 50 chars
            prompt = rec.get("prompt", "Related episode")[:50]

            recommendations.append(EpisodeRecommendationItem(
                episode_index=episode_idx,
                prompt=prompt,
                answer=rec.get("answer", "Here's another relevant episode."),
                confidence=min(1.0, max(0.0, rec.get("confidence", 0.7))),
                turn_summary=rec.get("turn_summary", "Alternative episode shown")[:150],
                extracted_entities=rec.get("extracted_entities", [])[:5],
                turn_themes=rec.get("turn_themes", [])[:3],
            ))

        logger.info(f"[EPISODE_RECOMMENDATIONS] Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations):
            logger.info(f"  [{i+1}] idx={rec.episode_index} | prompt=\"{rec.prompt}\" | conf={rec.confidence}")

        return EpisodeRecommendationsOutput(recommendations=recommendations)

    except Exception as e:
        logger.error(f"[EPISODE_RECOMMENDATIONS] LLM failed: {e}")
        return EpisodeRecommendationsOutput(recommendations=[])


# Store for pending episode recommendations (keyed by session_id + turn)
_pending_episode_recommendations: Dict[str, Dict[str, Any]] = {}
_episode_recommendations_lock = asyncio.Lock()


async def store_episode_recommendations(
    session_id: str,
    turn_id: str,
    recommendations: EpisodeRecommendationsOutput,
    episodes: List[Dict[str, Any]],
    episode_descriptions: Dict[str, Dict],
    original_question: str = "",
    resolved_query: str = "",
) -> None:
    """Store episode recommendations for later retrieval when user clicks."""
    async with _episode_recommendations_lock:
        key = f"{session_id}:{turn_id}"
        _pending_episode_recommendations[key] = {
            "recommendations": recommendations,
            "episodes": episodes,
            "episode_descriptions": episode_descriptions,
            "original_question": original_question,
            "resolved_query": resolved_query,
            "created_at": datetime.utcnow(),
        }
        logger.info(f"[EPISODE_RECOMMENDATIONS] Stored {len(recommendations.recommendations)} recommendations for {key}")

        # Cleanup old entries (keep last 100)
        if len(_pending_episode_recommendations) > 100:
            # Remove oldest entries
            sorted_keys = sorted(
                _pending_episode_recommendations.keys(),
                key=lambda k: _pending_episode_recommendations[k]["created_at"]
            )
            for old_key in sorted_keys[:-100]:
                del _pending_episode_recommendations[old_key]


async def get_episode_recommendation(
    session_id: str,
    turn_id: str,
    recommendation_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific episode recommendation by index.

    Returns the recommendation item, full episode data, and original context.
    """
    async with _episode_recommendations_lock:
        key = f"{session_id}:{turn_id}"
        stored = _pending_episode_recommendations.get(key)

        if not stored:
            logger.warning(f"[EPISODE_RECOMMENDATIONS] No stored recommendations for {key}")
            return None

        recs = stored["recommendations"].recommendations
        episodes = stored["episodes"]
        episode_descriptions = stored["episode_descriptions"]

        if recommendation_index < 0 or recommendation_index >= len(recs):
            logger.warning(f"[EPISODE_RECOMMENDATIONS] Invalid recommendation index {recommendation_index}")
            return None

        rec = recs[recommendation_index]
        episode = episodes[rec.episode_index] if rec.episode_index < len(episodes) else None

        if not episode:
            logger.warning(f"[EPISODE_RECOMMENDATIONS] Episode not found for index {rec.episode_index}")
            return None

        # Enrich with description data
        desc_data = episode_descriptions.get(episode.get("episode_id"), {})
        episode_with_desc = episode.copy()
        episode_with_desc["description"] = desc_data.get("description")
        episode_with_desc["uri"] = desc_data.get("uri")
        episode_with_desc["image"] = desc_data.get("images", [None])[0] if desc_data.get("images") else None

        return {
            "recommendation": rec,
            "episode": episode_with_desc,
            "original_question": stored.get("original_question", ""),
            "resolved_query": stored.get("resolved_query", ""),
        }
