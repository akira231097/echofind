"""
Query Router for EchoFind Conversational Agent.

Routes incoming queries to one of three branches:
- small_talk: Greetings, explanations, off-topic
- episode_search: Finding episodes by metadata (no topic)
- clip_search: Finding clips by topic/theme/moment

Runs on the configured router model (gemini-3-flash-preview) with
reasoning_effort="none" for fast, low-latency classification.
"""

import asyncio
import json
import logging
from typing import Optional

import config
from engine.memory import ConversationMemory
from engine.schemas import RouterOutput

logger = logging.getLogger(__name__)

# Model & reasoning - load from config (centralized in config.py for easy switching)
ROUTER_MODEL = config.ROUTER_MODEL
ROUTER_REASONING_EFFORT = config.ROUTER_REASONING_EFFORT
CONFIDENCE_THRESHOLD = getattr(config, 'ROUTER_CONFIDENCE_THRESHOLD', 0.70)
DEFAULT_FALLBACK = getattr(config, 'ROUTER_DEFAULT_FALLBACK', 'small_talk')

# Response schema with sub_intent for downstream handling
# Includes "thinking" field for Chain-of-Thought reasoning (improves accuracy by ~30%)
ROUTER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {
            "type": "string",
            "description": "Step-by-step reasoning: Query analysis → Context check → Route evaluation → Decision"
        },
        "route": {
            "type": "string",
            "enum": ["small_talk", "episode_search", "clip_search"]
        },
        "confidence": {
            "type": "number"
        },
        "reasoning": {
            "type": "string"
        },
        "sub_intent": {
            "type": "string",
            "description": "Specific intent within route: greeting|explanation|clarification|off_topic for small_talk"
        },
        "resolved_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Resolved pronouns/references to actual entity names"
        }
    },
    "required": ["thinking", "route", "confidence", "reasoning"]
}


def _build_router_system_prompt() -> str:
    """Build the system prompt for router."""
    return """You are the routing brain for Echo, a podcast discovery agent. Your job is to understand what the user wants and route to the correct handler.

ROUTES:
- clip_search: Find content by TOPIC/THEME/MEANING (semantic search). Default when uncertain.
- episode_search: Find content by METADATA ONLY (person, show, date - no topic).
- small_talk: User wants to UNDERSTAND/DISCUSS, not find new content.

ROUTING PRINCIPLES:

1. EXPLICIT REQUESTS WIN
   If query contains explicit intent words, route based on those:
   - "find clips about X", "show me X topic" → clip_search
   - "latest episodes", "what shows has X been on" → episode_search
   - "explain that", "who is X", "is that true" → small_talk

2. CONTINUATIONS FOLLOW PREVIOUS ROUTE
   Short responses continue the conversation flow:
   - Check ROUTE_HISTORY in context - that's what "more/yes/next" continues
   - Affirmatives, negatives, navigation all stay on same route
   - Only explicit new requests change routes

3. CONTEXT RESOLVES AMBIGUITY
   When query is ambiguous, use context:
   - Pronouns (he/she/it/that) → resolve from PRIMARY entity or CURRENT_TOPIC
   - "more", "another" → continue ROUTE_HISTORY
   - Questions about shown content → small_talk/explanation

4. GREETING ONLY AT START
   "greeting" sub_intent ONLY when there's NO prior context (TYPE: none).
   With any context, short responses are continuations, not greetings.

5. WHEN UNCERTAIN → clip_search
   clip_search is the most comprehensive route. When in doubt, route there.

6. PERSONAL QUESTIONS ABOUT PODCAST PERSONALITIES → clip_search
   When user asks personal questions about someone IN CONTEXT:
   - "Does he drink?", "Does she meditate?", "Is he married?"
   - "What time does he wake up?", "What's her routine?"
   - "What does he think about X?"

   These should route to clip_search because:
   - Podcast hosts discuss personal details in their content
   - User wants CLIPS where that person discusses this topic
   - NOT off-topic - these ARE searchable podcast content

   ONLY route personal questions to small_talk if:
   - There are NO entities in context (asking about Claude/general)
   - Question is completely unrelated to any known personality

=== CHAIN-OF-THOUGHT REASONING (CRITICAL FOR ACCURACY) ===
Research shows step-by-step reasoning improves classification accuracy by 30%.
Before outputting your decision, you MUST think through these steps:

STEP 1 - QUERY ANALYSIS: What is the user literally asking? Is it seeking NEW content or asking about something already discussed?
STEP 2 - CONTEXT CHECK: What was just shown (TYPE)? What entities are in focus? What was the previous route?
STEP 3 - ROUTE EVALUATION: For each route, would it satisfy what the user is asking?
STEP 4 - DECISION: Based on your analysis, which route is best?

OUTPUT FORMAT:
{
  "thinking": "<Your step-by-step reasoning through Steps 1-4. Be specific about what you observe and conclude.>",
  "route": "clip_search|episode_search|small_talk",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of detected intent",
  "sub_intent": "greeting|explanation|contextual_knowledge|clarification|off_topic (only for small_talk)",
  "resolved_entities": ["Names resolved from pronouns"]
}

The "thinking" field forces you to reason BEFORE deciding, which significantly improves accuracy."""


def _build_router_user_prompt(query: str, memory: ConversationMemory) -> str:
    """Build user prompt with rich context from memory."""

    # Get rich context using the new render_for_router method
    rich_context = memory.search_state.render_for_router(memory.recent_turns)

    return f"""Below are examples showing CONTEXT, QUERY, and correct OUTPUT format.

---
EXAMPLE 1 - Greeting at conversation start:
CONTEXT:
TYPE: none (conversation start)
QUERY: "hey, what can you help me with?"
OUTPUT: {{"route": "small_talk", "confidence": 0.95, "reasoning": "New conversation greeting", "sub_intent": "greeting", "resolved_entities": []}}

---
EXAMPLE 2 - Topic search (semantic):
CONTEXT:
TYPE: none (conversation start)
QUERY: "clips about artificial intelligence"
OUTPUT: {{"route": "clip_search", "confidence": 0.95, "reasoning": "Explicit topic search for AI content", "resolved_entities": []}}

---
EXAMPLE 3 - Metadata search (no topic):
CONTEXT:
TYPE: none (conversation start)
QUERY: "latest Joe Rogan episodes"
OUTPUT: {{"route": "episode_search", "confidence": 0.95, "reasoning": "Pure metadata - host + recency, no topic", "resolved_entities": ["Joe Rogan"]}}

---
EXAMPLE 4 - Person + topic (semantic search):
CONTEXT:
TYPE: none (conversation start)
QUERY: "what does Elon Musk think about Mars colonization"
OUTPUT: {{"route": "clip_search", "confidence": 0.95, "reasoning": "Person + topic needs semantic search", "resolved_entities": ["Elon Musk"]}}

---
EXAMPLE 5 - Explanation request:
CONTEXT:
TYPE: clip_shown
CONTENT: "Huberman on dopamine"
ABOUT: Huberman claims cold showers boost dopamine by 250%
PRIMARY: Andrew Huberman
QUERY: "is that actually true?"
OUTPUT: {{"route": "small_talk", "confidence": 0.92, "reasoning": "Asking to verify claim from shown content", "sub_intent": "explanation", "resolved_entities": []}}

---
EXAMPLE 6 - Contextual knowledge:
CONTEXT:
TYPE: clip_shown
CONTENT: "Jelly Roll on Joe Rogan"
ABOUT: Jelly Roll discusses prison, addiction recovery, weight loss
PRIMARY: Jelly Roll
QUERY: "who is this guy?"
OUTPUT: {{"route": "small_talk", "confidence": 0.95, "reasoning": "Asking about entity in current context", "sub_intent": "contextual_knowledge", "resolved_entities": ["Jelly Roll"]}}

---
EXAMPLE 7 - Affirmative continuation (clip_search):
CONTEXT:
TYPE: clip_shown
CONTENT: "Naval on wealth"
ABOUT: Naval explains building wealth through leverage
PRIMARY: Naval Ravikant
CURRENT_TOPIC: "wealth building"
ROUTE_HISTORY: clip_search → clip_search
QUERY: "yes"
OUTPUT: {{"route": "clip_search", "confidence": 0.95, "reasoning": "Affirmative continues previous clip_search on wealth topic", "resolved_entities": ["Naval Ravikant"]}}

---
EXAMPLE 8 - Affirmative continuation (episode_search):
CONTEXT:
TYPE: episode_shown
CONTENT: "Tim Ferriss Show #500"
PRIMARY: Tim Ferriss
ROUTE_HISTORY: episode_search
QUERY: "sure, more please"
OUTPUT: {{"route": "episode_search", "confidence": 0.92, "reasoning": "Affirmative continues previous episode_search", "resolved_entities": ["Tim Ferriss"]}}

---
EXAMPLE 9 - Implicit affirmative:
CONTEXT:
TYPE: clip_shown
CONTENT: "Peterson on meaning"
ABOUT: Finding meaning through responsibility
PRIMARY: Jordan Peterson
CURRENT_TOPIC: "meaning"
ROUTE_HISTORY: clip_search
QUERY: "sounds good"
OUTPUT: {{"route": "clip_search", "confidence": 0.90, "reasoning": "Implicit affirmative continues clip_search", "resolved_entities": ["Jordan Peterson"]}}

---
EXAMPLE 10 - Negative continuation (wants different, same route):
CONTEXT:
TYPE: clip_shown
CONTENT: "Rogan on hunting"
ABOUT: Joe Rogan discusses bow hunting experiences
PRIMARY: Joe Rogan
CURRENT_TOPIC: "hunting"
ROUTE_HISTORY: clip_search
QUERY: "no, something else"
OUTPUT: {{"route": "clip_search", "confidence": 0.88, "reasoning": "Negative but continues clip_search with different content", "resolved_entities": ["Joe Rogan"]}}

---
EXAMPLE 11 - Navigation (next/skip):
CONTEXT:
TYPE: clip_shown
CONTENT: "Lex on AI safety"
PRIMARY: Lex Fridman
CURRENT_TOPIC: "AI safety"
ROUTE_HISTORY: clip_search → clip_search
QUERY: "next"
OUTPUT: {{"route": "clip_search", "confidence": 0.92, "reasoning": "Navigation continues same route", "resolved_entities": ["Lex Fridman"]}}

---
EXAMPLE 12 - Feedback (liked it):
CONTEXT:
TYPE: clip_shown
CONTENT: "Huberman sleep tips"
ABOUT: Science-based sleep optimization
PRIMARY: Andrew Huberman
CURRENT_TOPIC: "sleep"
ROUTE_HISTORY: clip_search
QUERY: "I liked that, more like this"
OUTPUT: {{"route": "clip_search", "confidence": 0.95, "reasoning": "Positive feedback requests similar content", "resolved_entities": ["Andrew Huberman"]}}

---
EXAMPLE 13 - Feedback (didn't like):
CONTEXT:
TYPE: clip_shown
CONTENT: "Crypto discussion"
ABOUT: Bitcoin price predictions
CURRENT_TOPIC: "crypto"
ROUTE_HISTORY: clip_search
QUERY: "boring, something different"
OUTPUT: {{"route": "clip_search", "confidence": 0.88, "reasoning": "Negative feedback requests different content", "resolved_entities": []}}

---
EXAMPLE 14 - Compound query (explicit wins):
CONTEXT:
TYPE: clip_shown
PRIMARY: Naval Ravikant
CURRENT_TOPIC: "startups"
ROUTE_HISTORY: clip_search
QUERY: "yes, but also what episodes has he been on?"
OUTPUT: {{"route": "episode_search", "confidence": 0.90, "reasoning": "Explicit episode request overrides continuation", "resolved_entities": ["Naval Ravikant"]}}

---
EXAMPLE 15 - Correction:
CONTEXT:
TYPE: clip_shown
CONTENT: "Joe Biden speech"
PRIMARY: Joe Biden
QUERY: "no I meant Joe Rogan"
OUTPUT: {{"route": "clip_search", "confidence": 0.88, "reasoning": "Correction - user wants Joe Rogan content", "resolved_entities": ["Joe Rogan"]}}

---
EXAMPLE 16 - Pronoun with topic:
CONTEXT:
TYPE: clip_shown
CONTENT: "Musk on AI"
PRIMARY: Elon Musk
CURRENT_TOPIC: "AI"
QUERY: "what else did he say about that?"
OUTPUT: {{"route": "clip_search", "confidence": 0.95, "reasoning": "More clips from Elon Musk about AI", "resolved_entities": ["Elon Musk"]}}

---
EXAMPLE 17 - Guest appearances (metadata):
CONTEXT:
TYPE: clip_shown
PRIMARY: Sam Altman
QUERY: "what podcasts has he been on?"
OUTPUT: {{"route": "episode_search", "confidence": 0.95, "reasoning": "Asking for episode metadata - guest appearances", "resolved_entities": ["Sam Altman"]}}

---
EXAMPLE 18 - Reaction as continuation:
CONTEXT:
TYPE: clip_shown
CONTENT: "Mind-blowing physics fact"
ABOUT: Quantum entanglement explanation
CURRENT_TOPIC: "physics"
ROUTE_HISTORY: clip_search
QUERY: "wow, really?"
OUTPUT: {{"route": "clip_search", "confidence": 0.85, "reasoning": "Reaction with context continues exploration", "resolved_entities": []}}

---
EXAMPLE 19 - Minimal input with context:
CONTEXT:
TYPE: clip_shown
CONTENT: "Meditation benefits"
PRIMARY: Sam Harris
CURRENT_TOPIC: "meditation"
ROUTE_HISTORY: clip_search
QUERY: "more"
OUTPUT: {{"route": "clip_search", "confidence": 0.92, "reasoning": "Minimal input continues previous search", "resolved_entities": ["Sam Harris"]}}

---
EXAMPLE 20 - Minimal input without context:
CONTEXT:
TYPE: none (conversation start)
QUERY: "?"
OUTPUT: {{"route": "small_talk", "confidence": 0.80, "reasoning": "Unclear input with no context - ask for clarification", "sub_intent": "clarification", "resolved_entities": []}}

---
EXAMPLE 21 - Topic pivot:
CONTEXT:
TYPE: clip_shown
CURRENT_TOPIC: "productivity"
ROUTE_HISTORY: clip_search → clip_search
QUERY: "actually, what about relationships?"
OUTPUT: {{"route": "clip_search", "confidence": 0.95, "reasoning": "Explicit topic pivot to relationships", "resolved_entities": []}}

---
EXAMPLE 22 - Asking about system:
CONTEXT:
TYPE: clip_shown
QUERY: "what else can you search for?"
OUTPUT: {{"route": "small_talk", "confidence": 0.88, "reasoning": "Meta question about capabilities", "sub_intent": "greeting", "resolved_entities": []}}

---
EXAMPLE 23 - Typo/casual language:
CONTEXT:
TYPE: none (conversation start)
QUERY: "hubermn sleep stuff"
OUTPUT: {{"route": "clip_search", "confidence": 0.92, "reasoning": "Topic search despite typo - Huberman sleep content", "resolved_entities": ["Andrew Huberman"]}}

---
EXAMPLE 24 - Off-topic:
CONTEXT:
TYPE: clip_shown
CURRENT_TOPIC: "fitness"
QUERY: "what's the weather today?"
OUTPUT: {{"route": "small_talk", "confidence": 0.85, "reasoning": "Off-topic question unrelated to podcasts", "sub_intent": "off_topic", "resolved_entities": []}}

---
EXAMPLE 25 - Open-ended recommendation (no criteria):
CONTEXT:
TYPE: none (conversation start)
QUERY: "what should I listen to today?"
OUTPUT: {{"route": "small_talk", "confidence": 0.88, "reasoning": "Asking for recommendation without topic/criteria - ask what interests them", "sub_intent": "clarification", "resolved_entities": []}}

---
EXAMPLE 26 - Open-ended with context (has prior topic):
CONTEXT:
TYPE: clip_shown
CURRENT_TOPIC: "productivity"
PRIMARY: Tim Ferriss
ROUTE_HISTORY: clip_search
QUERY: "what should I listen to next?"
OUTPUT: {{"route": "clip_search", "confidence": 0.90, "reasoning": "Has context - continue with productivity/Tim Ferriss content", "resolved_entities": ["Tim Ferriss"]}}

---
EXAMPLE 27 - Vague discovery request:
CONTEXT:
TYPE: none (conversation start)
QUERY: "show me something interesting"
OUTPUT: {{"route": "small_talk", "confidence": 0.85, "reasoning": "Vague request without criteria - ask what topics interest them", "sub_intent": "clarification", "resolved_entities": []}}

---
EXAMPLE 28 - Discovery with hint:
CONTEXT:
TYPE: none (conversation start)
QUERY: "I'm bored, show me something funny"
OUTPUT: {{"route": "clip_search", "confidence": 0.92, "reasoning": "Has implicit topic - funny/comedy content", "resolved_entities": []}}

---
EXAMPLE 29 - Personal question about entity in context:
CONTEXT:
TYPE: clip_shown
CONTENT: "Huberman on alcohol and sleep"
ABOUT: Dr. Huberman explains how alcohol affects sleep quality
PRIMARY: Andrew Huberman
CURRENT_TOPIC: "alcohol effects on sleep"
QUERY: "does he drink?"
OUTPUT: {{"route": "clip_search", "confidence": 0.92, "reasoning": "Personal question about podcast personality - search for clips where Huberman discusses his own drinking habits", "resolved_entities": ["Andrew Huberman"]}}

---
EXAMPLE 30 - Personal question with NO context (truly off-topic):
CONTEXT:
TYPE: none (conversation start)
QUERY: "does he drink?"
OUTPUT: {{"route": "small_talk", "confidence": 0.85, "reasoning": "Personal question with no context - need clarification about who", "sub_intent": "clarification", "resolved_entities": []}}

---
NOW ROUTE THIS QUERY:

CONTEXT:
{rich_context}

QUERY: "{query}"

OUTPUT:"""


def _create_fallback_router_output(query: str, error_reason: str) -> RouterOutput:
    """
    Create a fallback RouterOutput when routing fails.

    Defaults to clip_search as it's the most general route.
    """
    logger.warning(f"[ROUTER] Creating fallback output due to: {error_reason}")

    # Detect simple patterns for basic routing even on failure
    query_lower = query.lower()

    # Simple heuristics for fallback routing
    if any(greeting in query_lower for greeting in ['hello', 'hi ', 'hey', 'thanks', 'thank you']):
        fallback_route = "small_talk"
        sub_intent = "greeting"
    elif any(word in query_lower for word in ['what is', 'who is', 'explain', 'how does']):
        fallback_route = "small_talk"
        sub_intent = "explanation"
    elif any(word in query_lower for word in ['episode', 'latest', 'newest', 'recent']):
        fallback_route = "episode_search"
        sub_intent = None
    else:
        fallback_route = DEFAULT_FALLBACK
        sub_intent = None

    return RouterOutput(
        route=fallback_route,
        sub_intent=sub_intent,
        confidence=0.5,  # Low confidence for fallback
        reasoning=f"Fallback routing due to error: {error_reason[:100]}",
        resolved_entities=[],
        query_intent=query[:50],
        key_signals=["fallback"],
        fallback_route=None,
    )


async def route_query(
    gemini_client,
    query: str,
    memory: ConversationMemory,
) -> RouterOutput:
    """
    Route user query to appropriate branch (small_talk, clip_search, episode_search).

    This is the FIRST decision point - if routing is wrong, everything downstream fails.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("[ROUTER] 🧭 ROUTING USER QUERY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[ROUTER] 📥 INPUT:")
    logger.info(f"  ├─ Query: \"{query[:70]}{'...' if len(query) > 70 else ''}\"")
    logger.info(f"  ├─ Session: {memory.session_id}")
    logger.info(f"  └─ Turn: #{memory.turn_count + 1}")

    # Build context
    system_prompt = _build_router_system_prompt()
    user_prompt = _build_router_user_prompt(query, memory)

    logger.info("")
    logger.info("[ROUTER] 📋 CONTEXT FOR DECISION:")
    logger.info(f"  ├─ Prompt size: {len(user_prompt):,} chars")

    # Log recent route history for context
    route_pattern = memory.search_state.get_route_pattern() if hasattr(memory, 'search_state') else "N/A"
    logger.info(f"  ├─ Route history: {route_pattern}")

    # Log current entities/topic for pronoun resolution context
    current_entities = memory.search_state.current_entities if hasattr(memory, 'search_state') else []
    current_topic = memory.search_state.current_topic if hasattr(memory, 'search_state') else None
    if current_entities:
        logger.info(f"  ├─ Current entities: {current_entities[:3]}")
    if current_topic:
        logger.info(f"  └─ Current topic: {current_topic}")
    else:
        logger.info("  └─ Current topic: (none)")

    # Use asyncio.to_thread for sync OpenAI client compatibility
    try:
        response = await asyncio.to_thread(
            gemini_client.chat.completions.create,
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # Deterministic routing
            reasoning_effort=ROUTER_REASONING_EFFORT,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        if not raw_content:
            logger.error("[ROUTER] Empty response from LLM")
            return _create_fallback_router_output(query, "Empty LLM response")

        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as json_err:
            logger.error(f"[ROUTER] JSON parse error: {json_err}")
            logger.error(f"[ROUTER] Raw content: {raw_content[:200]}...")
            return _create_fallback_router_output(query, f"JSON parse error: {json_err}")

        # Ensure resolved_entities defaults to empty list if not provided
        if "resolved_entities" not in result:
            result["resolved_entities"] = []

        # Log routing decision
        route = result['route']
        confidence = result['confidence']
        sub_intent = result.get('sub_intent', 'N/A')
        resolved = result.get('resolved_entities', [])
        reasoning = result.get('reasoning', '')
        thinking = result.get('thinking', '')

        logger.info("")
        logger.info("[ROUTER] ✅ ROUTING DECISION:")

        # Log Chain-of-Thought reasoning (if available)
        if thinking:
            logger.info("")
            logger.info("[ROUTER] 🧠 CHAIN-OF-THOUGHT REASONING:")
            # Split thinking into lines for better readability
            for line in thinking.split('. '):
                if line.strip():
                    logger.info(f"  │ {line.strip()}")
            logger.info("")

        # Route with emoji
        route_emoji = {"small_talk": "💬", "clip_search": "🎬", "episode_search": "🎧"}
        logger.info(f"  ├─ Route: {route_emoji.get(route, '📌')} {route.upper()}")
        logger.info(f"  ├─ Confidence: {confidence:.1%} {'✓ High' if confidence >= 0.8 else '⚠ Medium' if confidence >= CONFIDENCE_THRESHOLD else '⚠ Low'}")

        # Sub-intent explanation
        if route == "small_talk":
            intent_desc = {
                "greeting": "User said hello/hi",
                "explanation": "User wants more info/explanation",
                "clarification": "User wants to clarify something",
                "off_topic": "Query not related to podcasts"
            }
            logger.info(f"  ├─ Sub-intent: {sub_intent} - {intent_desc.get(sub_intent, '')}")
        elif route == "clip_search":
            logger.info(f"  ├─ Sub-intent: {sub_intent} - Will search by topic/theme")
        elif route == "episode_search":
            logger.info(f"  ├─ Sub-intent: {sub_intent} - Will search by person/show/date")

        # Resolved entities (pronoun resolution)
        if resolved:
            logger.info(f"  ├─ Resolved entities: {resolved}")
            logger.info("  │   (Pronouns like 'he/she/they' resolved to actual names)")

        # Reasoning
        logger.info(f"  └─ Reasoning: \"{reasoning[:80]}{'...' if len(reasoning) > 80 else ''}\"")

        logger.debug(f"[ROUTER] Full reasoning: {reasoning}")

        # Fallback logic
        if result["confidence"] < CONFIDENCE_THRESHOLD:
            original_route = result["route"]
            result["route"] = DEFAULT_FALLBACK
            result["fallback_route"] = original_route
            logger.info("")
            logger.warning(f"[ROUTER] ⚠️ LOW CONFIDENCE FALLBACK:")
            logger.warning(f"  ├─ Original route: {original_route}")
            logger.warning(f"  ├─ Confidence: {result['confidence']:.1%} (threshold: {CONFIDENCE_THRESHOLD:.1%})")
            logger.warning(f"  └─ Falling back to: {DEFAULT_FALLBACK}")

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"[ROUTER] ✅ ROUTING COMPLETE → {result['route'].upper()}")
        logger.info("=" * 70)

        return RouterOutput(**result)

    except Exception as e:
        logger.error("")
        logger.error("[ROUTER] ❌ ROUTING FAILED:")
        logger.error(f"  ├─ Error: {type(e).__name__}: {e}")
        logger.error(f"  └─ Falling back to: {DEFAULT_FALLBACK}")
        return _create_fallback_router_output(query, str(e))


def route_query_sync(
    gemini_client,
    query: str,
    memory: ConversationMemory,
) -> RouterOutput:
    logger.debug("[ROUTER] Using synchronous route_query wrapper")
    return asyncio.run(route_query(gemini_client, query, memory))


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_route_description(route: str) -> str:
    """Get a human-readable description of a route."""
    descriptions = {
        "small_talk": "Handle greeting, explanation, or off-topic conversation",
        "episode_search": "Find episodes by metadata (host, guest, show, date)",
        "clip_search": "Find clips by topic, theme, or semantic content",
    }
    return descriptions.get(route, "Unknown route")


def should_use_fallback(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Check if confidence is below threshold and fallback should be used."""
    return confidence < threshold


def log_routing_summary(output: RouterOutput, query: str) -> None:
    """Log a concise routing summary for debugging."""
    logger.info(
        f"[ROUTER SUMMARY] '{query[:40]}...' -> {output.route} "
        f"(conf={output.confidence:.2f}, intent='{output.query_intent}')"
    )
