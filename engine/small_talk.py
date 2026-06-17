"""
Small Talk Branch for EchoFind Conversational Agent.

Handles:
1. Greetings and persona introduction
2. Contextual knowledge questions (about entities in conversation context)
3. Explanation requests (with optional search grounding)
4. Clarification requests (no prior context)
5. Off-topic questions (redirect to podcasts)

Key Features:
- Intelligent response type detection using query analysis + memory context
- Rich context rendering for each response type
- Follow-up suggestions to keep conversation flowing
- Synthesizes knowledge from conversation history for contextual questions
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, Optional, Tuple, List

from pydantic import BaseModel, Field
import config
from engine.memory import ConversationMemory
from engine.schemas import SmallTalkResponse, BranchMemoryUpdate, RouterOutput
from retrieval.llm_utils import llm_call_with_retry

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class SmallTalkMemoryUpdateOutput(BaseModel):
    """Memory update fields from small talk responses."""
    turn_summary: str = Field(default="", description="Brief summary (max 150 chars)")
    action_target_title: Optional[str] = Field(default=None, description="Title of explained topic")
    entities_mentioned: List[str] = Field(default_factory=list, description="Key entities (max 5)")
    topics_discussed: List[str] = Field(default_factory=list, description="Topics discussed (max 3)")
    is_topic_shift: bool = Field(default=False, description="Whether this is a new topic")
    suggested_phase: str = Field(default="discovery", description="Conversation phase")


class SmallTalkResponseOutput(BaseModel):
    """Structured output for small talk responses."""
    response_text: str = Field(description="Natural response text")
    follow_up_suggestions: List[str] = Field(default_factory=list, description="Follow-up prompts (max 3)")
    memory_update: SmallTalkMemoryUpdateOutput = Field(default_factory=SmallTalkMemoryUpdateOutput)


class StandardExplanationOutput(BaseModel):
    """Structured output for standard explanation responses."""
    response_text: str = Field(description="Explanation response text")
    follow_up_suggestions: List[str] = Field(default_factory=list, description="Follow-up prompts")

# Models & reasoning - load from config (centralized in config.py for easy switching)
SIMPLE_RESPONSE_MODEL = config.SMALL_TALK_MODEL
GROUNDING_MODEL = config.SMALL_TALK_WITH_GROUNDING_MODEL
MEMORY_UPDATE_MODEL = config.MEMORY_UPDATE_MODEL

# Reasoning effort settings
SMALL_TALK_REASONING_EFFORT = config.SMALL_TALK_REASONING_EFFORT
GROUNDING_REASONING_EFFORT = config.SMALL_TALK_GROUNDING_REASONING_EFFORT
MEMORY_UPDATE_REASONING_EFFORT = config.MEMORY_UPDATE_REASONING_EFFORT

# Optional: Native Gemini client for grounding (set via init_grounding_client)
_native_genai_client = None


def init_grounding_client(api_key: Optional[str] = None) -> bool:
    global _native_genai_client

    try:
        from google import genai

        api_key = api_key or getattr(config, 'GEMINI_API_KEY', None)
        if not api_key:
            logger.warning("[SMALL_TALK] No API key provided for native Gemini client")
            return False

        _native_genai_client = genai.Client(api_key=api_key)
        logger.info("[SMALL_TALK] Native Gemini client initialized - grounding enabled")
        return True

    except ImportError:
        logger.info("[SMALL_TALK] google-genai not installed - grounding disabled")
        logger.info("[SMALL_TALK] To enable grounding: pip install google-genai")
        return False
    except Exception as e:
        logger.warning(f"[SMALL_TALK] Failed to init native client: {e}")
        return False


def is_grounding_available() -> bool:
    """Check if grounding is available."""
    return _native_genai_client is not None


def _detect_response_type(
    query: str,
    memory: ConversationMemory,
    router_output: RouterOutput,
) -> tuple[str, dict]:
    query_lower = query.lower().strip()

    # Get structured context from memory
    small_talk_context = memory.search_state.render_for_small_talk(
        recent_turns=memory.recent_turns,
        query=query
    )
    small_talk_context["current_themes"] = memory.conversation_themes

    logger.debug(f"[SMALL_TALK] Is contextual question: {small_talk_context['is_contextual_question']}")
    logger.debug(f"[SMALL_TALK] Queried entities: {[e['name'] for e in small_talk_context['queried_entities']]}")
    logger.debug(f"[SMALL_TALK] Uses pronoun: {small_talk_context.get('uses_pronoun', False)}")
    logger.debug(f"[SMALL_TALK] Resolved entity: {small_talk_context.get('resolved_entity')}")

    # =========================================================================
    # ROUTER HINT: Trust router's sub_intent when available (it has more context)
    # =========================================================================
    if router_output.sub_intent:
        router_sub_intent = router_output.sub_intent.lower()
        logger.info(f"[SMALL_TALK] Router provided sub_intent: {router_sub_intent}")

        # Map router sub_intents to our response types
        sub_intent_map = {
            "greeting": "greeting",
            "explanation": "explanation",
            "contextual_knowledge": "contextual_knowledge",
            "clarification": "clarification",
            "off_topic": "off_topic",
        }

        if router_sub_intent in sub_intent_map:
            mapped_type = sub_intent_map[router_sub_intent]
            # For explanation/contextual_knowledge, verify we have context
            if mapped_type == "explanation" and memory.search_state.last_action.action_type:
                logger.info(f"[SMALL_TALK] Using router sub_intent: {mapped_type}")
                return mapped_type, small_talk_context
            elif mapped_type == "contextual_knowledge":
                logger.info(f"[SMALL_TALK] Using router sub_intent: {mapped_type}")
                return mapped_type, small_talk_context
            elif mapped_type in ["greeting", "clarification", "off_topic"]:
                # For these, check if there's actually content context
                # "what is this episode about" is NOT a greeting even if pattern matches
                has_content_context = memory.search_state.last_action.action_type in ["clip_shown", "episode_shown"]
                content_question_words = ["episode", "clip", "podcast", "video", "about", "this"]
                is_asking_about_content = any(word in query_lower for word in content_question_words)

                if has_content_context and is_asking_about_content and mapped_type == "greeting":
                    # Override: this is really an explanation request
                    logger.info(f"[SMALL_TALK] Override router greeting -> explanation (has content context)")
                    return "explanation", small_talk_context
                else:
                    logger.info(f"[SMALL_TALK] Using router sub_intent: {mapped_type}")
                    return mapped_type, small_talk_context

    # =========================================================================
    # SAFETY CHECK: Personal questions about entities should NOT be here
    # =========================================================================
    if memory.search_state.current_entities:
        personal_question_patterns = [
            r'\bdoes\s+(he|she|they)\s+\w+',      # "does he drink"
            r'\bis\s+(he|she|they)\s+\w+',         # "is he married"
            r'\bwhat\s+does\s+(he|she|they)\b',    # "what does he think"
            r'\bwhere\s+does\s+(he|she|they)\b',   # "where does he live"
            r'\bhow\s+(old|tall)\s+is\b',          # "how old is he"
        ]

        import re
        if any(re.search(p, query_lower) for p in personal_question_patterns):
            logger.warning(f"[SMALL_TALK] Personal question about entity detected - "
                          f"treating as contextual_knowledge. Query: '{query[:50]}', "
                          f"Entities: {memory.search_state.current_entities[:2]}")
            return "contextual_knowledge", small_talk_context

    # =========================================================================
    # FALLBACK: Local pattern matching (when router doesn't provide sub_intent)
    # =========================================================================

    # === GREETING DETECTION ===
    greeting_patterns = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "what can you do", "what can you help", "who are you", "introduce yourself",
        "help me", "how does this work"
    ]
    if any(pattern in query_lower for pattern in greeting_patterns):
        if small_talk_context["is_contextual_question"] and len(query_lower) > 20:
            pass  # Fall through to contextual check
        else:
            has_content_context = memory.search_state.last_action.action_type in ["clip_shown", "episode_shown"]
            content_question_words = ["episode", "clip", "podcast", "video", "about"]
            is_asking_about_content = any(word in query_lower for word in content_question_words)

            if has_content_context and is_asking_about_content:
                logger.info("[SMALL_TALK] Detected: explanation (not greeting - asking about content)")
                return "explanation", small_talk_context

            logger.info("[SMALL_TALK] Detected: greeting")
            return "greeting", small_talk_context

    # === CONTEXTUAL KNOWLEDGE DETECTION ===
    # User is asking about something IN the conversation context
    knowledge_patterns = [
        "who is", "who's", "what is", "what's", "tell me about",
        "explain who", "describe", "what do you know about"
    ]
    has_knowledge_pattern = any(pattern in query_lower for pattern in knowledge_patterns)

    if has_knowledge_pattern and small_talk_context["is_contextual_question"]:
        queried = [e['name'] for e in small_talk_context['queried_entities']]
        resolved = small_talk_context.get('resolved_entity')
        uses_pronoun = small_talk_context.get('uses_pronoun', False)

        if uses_pronoun and resolved:
            logger.info(f"[SMALL_TALK] Detected: contextual_knowledge via pronoun resolution -> {resolved}")
        else:
            logger.info(f"[SMALL_TALK] Detected: contextual_knowledge about {queried}")
        return "contextual_knowledge", small_talk_context

    # === EXPLANATION DETECTION ===
    # User wants us to explain something about the last shown content
    explanation_patterns = [
        "explain", "what does that mean", "what did he mean", "what did she mean",
        "clarify", "what's that about", "tell me more about that",
        "is that true", "is that accurate", "break that down", "simplify",
        "what was that", "i don't understand", "can you elaborate"
    ]
    has_explanation_pattern = any(pattern in query_lower for pattern in explanation_patterns)

    if has_explanation_pattern:
        if memory.search_state.last_action.action_type:
            logger.info("[SMALL_TALK] Detected: explanation (has prior context)")
            return "explanation", small_talk_context
        else:
            logger.info("[SMALL_TALK] Detected: clarification (no prior context)")
            return "clarification", small_talk_context

    # === GENERAL KNOWLEDGE (with search grounding) ===
    if has_knowledge_pattern and not small_talk_context["is_contextual_question"]:
        logger.info(f"[SMALL_TALK] Detected: general_knowledge (not in context, will use grounding)")
        return "explanation", small_talk_context

    # === OFF-TOPIC DETECTION ===
    logger.info("[SMALL_TALK] Detected: off_topic (default fallback)")
    return "off_topic", small_talk_context


def _get_response_type_from_router(
    router_output: RouterOutput,
    memory: ConversationMemory,
) -> str:
    """
    DEPRECATED: Use _detect_response_type instead.
    Kept for backward compatibility.
    """
    sub_intent = router_output.sub_intent

    logger.debug(f"[SMALL_TALK] Router sub_intent: {sub_intent}")
    logger.debug(f"[SMALL_TALK] Last action type: {memory.search_state.last_action.action_type}")

    if sub_intent == "greeting":
        return "greeting"
    elif sub_intent == "explanation":
        if memory.search_state.last_action.action_type:
            return "explanation"
        else:
            logger.info("[SMALL_TALK] Explanation requested but no prior context - treating as clarification")
            return "clarification"
    elif sub_intent == "off_topic":
        return "off_topic"
    else:
        logger.warning(f"[SMALL_TALK] Unknown sub_intent: {sub_intent}, defaulting to off_topic")
        return "off_topic"


def _build_simple_response_prompt(
    query: str,
    response_type: str,
    memory: ConversationMemory,
    context_data: Dict[str, Any] = None,
) -> Tuple[str, str]:
    """Build prompts for simple (non-grounding) responses with rich context."""

    memory_context = memory.search_state.render_for_prompt()
    context_data = context_data or {}

    # Build conversation summary for context
    convo_summary = ""
    if context_data.get("conversation_summary"):
        convo_parts = []
        for turn in context_data["conversation_summary"][-3:]:
            if turn.get("user_asked"):
                convo_parts.append(f"• User: {turn['user_asked'][:60]}...")
                if turn.get("content_shown"):
                    convo_parts.append(f"  → Showed: {turn['content_shown']}")
        if convo_parts:
            convo_summary = "\n".join(convo_parts)

    # Get follow-up suggestions
    follow_ups = context_data.get("follow_up_suggestions", [])
    follow_up_text = "\n".join([f"  - {f}" for f in follow_ups]) if follow_ups else ""

    # Get current entities and topic
    current_entities = list(context_data.get("all_entity_knowledge", {}).keys())[:5]
    current_topic = context_data.get("current_topic", "")

    system_prompt = """<persona>
You are Echo, an AI assistant that helps users discover podcast content. You can find clips and episodes from popular shows.
</persona>

<response_guidelines>
1. Be warm, conversational, and helpful
2. Keep responses concise (2-3 sentences max for greetings)
3. For greetings: briefly mention you help find podcast clips, then ASK what they're interested in
4. If off-topic: acknowledge briefly, then suggest how podcasts might relate
5. Reference conversation context naturally if it exists
6. ALWAYS end with a question or suggestion to keep the conversation flowing
7. DO NOT over-explain what you can do - focus on engaging the user
</response_guidelines>

<examples>
GOOD greeting: "Hey! I help you discover interesting podcast clips. What topics or guests are you curious about?"
BAD greeting: "Hello! I'm Echo, your friendly podcast discovery AI agent. I help users find interesting podcast clips and episodes from shows like Joe Rogan Experience, Lex Fridman Podcast, and many more. I can search by topic, guest, host, or show name..."
</examples>

<output_format>
Respond with valid JSON:
{
  "response_text": "Your concise, engaging response",
  "response_type": "greeting|off_topic|clarification",
  "follow_up_suggestions": ["Suggestion based on context", "Another suggestion"],
  "memory_update": {
    "turn_summary": "Brief summary (max 150 chars)",
    "action_type": "greeting",
    "action_target_id": null,
    "action_target_title": null,
    "entities_mentioned": [],
    "topics_discussed": [],
    "is_topic_shift": false,
    "suggested_phase": "discovery"
  }
}
</output_format>"""

    if response_type == "greeting":
        context_block = ""
        if convo_summary:
            context_block = f"""
CONVERSATION SO FAR:
{convo_summary}

NOTE: Welcome them back! Reference what you were discussing.
"""
        else:
            context_block = "New conversation - be brief and engaging."

        user_prompt = f"""User says: "{query}"

Respond to this greeting. Keep it SHORT (2-3 sentences max).
- Briefly say you help find podcast clips
- ASK what they're interested in (topic, person, or show)
- Do NOT list all your capabilities

{context_block}

CURRENT ENTITIES: {current_entities if current_entities else "None yet"}
CURRENT TOPIC: {current_topic or "None yet"}

FOLLOW-UP OPTIONS:
{follow_up_text if follow_up_text else "  - Ask what topics interest them"}

Respond with JSON."""

    elif response_type == "clarification":
        user_prompt = f"""User says: "{query}"

User is asking for clarification, but there's no previous context to clarify.
Politely explain you're ready to help them find podcast content.
Ask what topic or show they're interested in.

CONVERSATION CONTEXT:
{convo_summary if convo_summary else "Conversation just started."}

CURRENT STATE:
{memory_context}

ADAPT FOLLOW-UP SUGGESTIONS:
{follow_up_text if follow_up_text else "  - Ask what they want to explore"}

Respond with JSON."""

    else:  # off_topic
        redirect_hint = ""
        if current_topic:
            redirect_hint = f"You could mention you were just exploring {current_topic}."
        elif current_entities:
            redirect_hint = f"You could offer to find clips about {current_entities[0] if current_entities else 'various topics'}."

        user_prompt = f"""User says: "{query}"

This question isn't directly about podcasts.
Acknowledge their question briefly, then gently redirect to how you can help with podcast discovery.
Don't be dismissive - be friendly and suggest how podcasts might relate to their interest.

CONVERSATION CONTEXT:
{convo_summary if convo_summary else "New conversation."}

REDIRECT HINT: {redirect_hint}

ADAPT FOLLOW-UP SUGGESTIONS:
{follow_up_text if follow_up_text else "  - Suggest related podcast topics"}

Respond with JSON."""

    return system_prompt, user_prompt


def _build_explanation_prompt(
    query: str,
    memory: ConversationMemory,
    context_data: Dict[str, Any] = None,
) -> Tuple[str, str]:
    """
    Build prompt for explanation response with rich context.
    Used for both grounded and non-grounded explanations.
    """

    last_action = memory.search_state.last_action
    # Memory context - no truncation with 1M context window
    memory_context = memory.render_for_prompt()
    context_data = context_data or {}

    # Get follow-up suggestions
    follow_ups = context_data.get("follow_up_suggestions", [])
    follow_up_text = "\n".join([f"  - {f}" for f in follow_ups]) if follow_ups else ""

    # Get current entities for context
    current_entities = list(context_data.get("all_entity_knowledge", {}).keys())[:5]

    system_prompt = """<persona>
You are Echo, a knowledgeable podcast discovery AI.
The user wants you to explain or expand on something from your previous response.
</persona>

<response_guidelines>
1. Provide a clear, helpful explanation
2. Reference the previous context when relevant
3. Keep your response conversational but informative (3-6 sentences typically)
4. If discussing complex topics, break them down simply
5. Encourage the user to explore related podcast content
6. ALWAYS include follow_up_suggestions to keep the conversation flowing
</response_guidelines>

<output_format>
Respond with valid JSON:
{
  "response_text": "Your explanation here",
  "follow_up_suggestions": ["Natural follow-up 1", "Natural follow-up 2"],
  "memory_update": {
    "turn_summary": "Explained [topic] (max 150 chars)",
    "action_type": "explanation",
    "action_target_id": null,
    "action_target_title": "Topic that was explained",
    "entities_mentioned": ["list of people/shows mentioned"],
    "topics_discussed": ["list of topics discussed"],
    "is_topic_shift": false,
    "suggested_phase": "deep_dive"
  }
}
</output_format>"""

    context_description = ""
    if last_action.action_type == "clip_shown":
        pub_date_str = f"\nPUBLISHED: {last_action.published_date}" if last_action.published_date else ""
        context_description = f"""
PREVIOUS ACTION: You showed a podcast clip
CLIP TITLE: "{last_action.target_title or 'a podcast clip'}"{pub_date_str}
CLIP SUMMARY: {last_action.target_summary or 'No summary available'}
"""
    elif last_action.action_type == "episode_shown":
        pub_date_str = f"\nPUBLISHED: {last_action.published_date}" if last_action.published_date else ""
        context_description = f"""
PREVIOUS ACTION: You showed a podcast episode
EPISODE TITLE: "{last_action.target_title or 'a podcast episode'}"{pub_date_str}
EPISODE SUMMARY: {last_action.target_summary or 'No summary available'}
"""
    elif last_action.action_type == "explanation":
        context_description = f"""
PREVIOUS ACTION: You provided an explanation
TOPIC: {last_action.target_title or 'previous topic'}
SUMMARY: {last_action.target_summary or 'Previous explanation'}
"""
    elif last_action.action_type == "greeting":
        context_description = """
PREVIOUS ACTION: You greeted the user
The user may be asking about something you mentioned in your greeting.
"""
    elif last_action.action_type == "contextual_knowledge":
        context_description = f"""
PREVIOUS ACTION: You explained something from conversation context
TOPIC: {last_action.target_title or 'previous topic'}
SUMMARY: {last_action.target_summary or 'Previous explanation'}
"""
    else:
        context_description = f"""
PREVIOUS ACTION: {last_action.action_type or 'Unknown'}
SUMMARY: {last_action.target_summary or 'No summary available'}
"""

    user_prompt = f"""User asks: "{query}"

CONTEXT:
{context_description}

CURRENT ENTITIES IN CONVERSATION: {current_entities if current_entities else "None tracked"}

FULL CONVERSATION MEMORY:
{memory_context}

SUGGESTED FOLLOW-UPS TO ADAPT:
{follow_up_text if follow_up_text else "  - Offer to find more related clips"}

Provide a helpful explanation based on the context above.
Be conversational and informative.

Respond with JSON."""

    return system_prompt, user_prompt


def _build_grounded_explanation_prompt(
    query: str,
    memory: ConversationMemory,
) -> str:
    """
    Build prompt for grounded explanation (used with native Gemini API).
    Returns just the user prompt since system prompts work differently with native API.
    """

    last_action = memory.search_state.last_action

    context = ""
    if last_action.target_title:
        context = f'The user is asking about "{last_action.target_title}".'
        if last_action.published_date:
            context += f' (Published: {last_action.published_date})'
    if last_action.target_summary:
        context += f' Context: {last_action.target_summary}'  # No truncation - 1M context window

    return f"""You are Echo, a podcast discovery AI assistant.

The user asks: "{query}"

{context}

Search for relevant information to provide a clear, helpful explanation.
Keep your response conversational (3-6 sentences).
NOTE: The grounding tool automatically handles citation formatting - do not add your own citations."""


def _build_contextual_knowledge_prompt(
    query: str,
    context_data: Dict[str, Any],
    memory: ConversationMemory,
) -> Tuple[str, str]:

    queried_entities = context_data.get("queried_entities", [])
    conversation_summary = context_data.get("conversation_summary", [])
    current_topic = context_data.get("current_topic", "")
    last_action = context_data.get("last_action", {})
    resolved_entity = context_data.get("resolved_entity")
    uses_pronoun = context_data.get("uses_pronoun", False)

    # Determine the target entity (from queried_entities or resolved pronoun)
    target_entity_name = None
    if queried_entities:
        target_entity_name = queried_entities[0]["name"]
    elif resolved_entity:
        target_entity_name = resolved_entity

    # Build entity context
    entity_context_parts = []

    # If we resolved a pronoun, explain that
    if uses_pronoun and resolved_entity:
        entity_context_parts.append(f"USER IS ASKING ABOUT: {resolved_entity} (resolved from pronoun in their question)")

    for entity_data in queried_entities:
        entity_name = entity_data["name"]
        facts = entity_data.get("facts", [])
        appeared_in = entity_data.get("appeared_in", [])
        themes = entity_data.get("themes", [])

        entity_context_parts.append(f"\n=== ABOUT {entity_name.upper()} ===")
        if appeared_in:
            entity_context_parts.append(f"Appeared in content shown to user:")
            for title in appeared_in[:3]:
                entity_context_parts.append(f"  - {title}")
        if facts:
            entity_context_parts.append(f"What we learned from clips:")
            for fact in facts[:3]:
                entity_context_parts.append(f"  - {fact}")
        if themes:
            entity_context_parts.append(f"Topics discussed: {', '.join(themes[:5])}")

    # If no facts from queried_entities but we have last action, use that
    if not entity_context_parts or (len(entity_context_parts) == 1 and uses_pronoun):
        if last_action.get("title") or last_action.get("summary"):
            entity_context_parts.append(f"\n=== WHAT WE JUST SHOWED ===")
            if last_action.get("title"):
                entity_context_parts.append(f"Content: {last_action['title']}")
            if last_action.get("published_date"):
                entity_context_parts.append(f"Published: {last_action['published_date']}")
            if last_action.get("summary"):
                entity_context_parts.append(f"About: {last_action['summary']}")

    entity_context = "\n".join(entity_context_parts) if entity_context_parts else "Limited context available."

    # Build conversation context with more detail
    convo_parts = []
    for turn in conversation_summary[-3:]:
        user_q = turn.get("user_asked", "")
        result = turn.get("result", "")
        shown = turn.get("content_shown", "")
        entities = turn.get("entities", [])
        if user_q:
            convo_parts.append(f"User asked: {user_q}")
            if shown:
                convo_parts.append(f"  → Showed: {shown}")
            if result:
                convo_parts.append(f"  → Summary: {result}")
            if entities:
                convo_parts.append(f"  → People/topics: {', '.join(entities[:3])}")
    convo_context = "\n".join(convo_parts) if convo_parts else "Conversation just started."

    # Build suggested follow-ups
    follow_ups = context_data.get("follow_up_suggestions", [])
    follow_up_text = "\n".join([f"  - {f}" for f in follow_ups]) if follow_ups else f"  - Find more clips about {target_entity_name or 'this topic'}"

    # REWRITTEN SYSTEM PROMPT: Use GENERAL KNOWLEDGE + context to fully answer
    system_prompt = """You are a knowledgeable AI assistant helping users discover podcast content.

<CRITICAL_RULES>
1. FULLY ANSWER THE USER'S QUESTION using your general knowledge
2. Supplement with conversation context when relevant (what clips were shown, what was discussed)
3. Do NOT limit your answer to only what was in the clips - use your full knowledge!
4. Be informative and complete (3-5 sentences)
5. End with a natural follow-up suggestion

IMPORTANT: You have general knowledge about public figures, streamers, celebrities, etc.
USE IT! Don't say "based on the clips we watched" if you know more about the person.
</CRITICAL_RULES>

<BAD_RESPONSE_EXAMPLES>
User: "Who is IShowSpeed, his origins?"
BAD: "Based on the clips we just watched, John Cena and Joe Rogan discussed IShowSpeed's appearance at the Royal Rumble. However, the clips didn't go into detail about his origins."
(This is bad because you KNOW who IShowSpeed is - don't pretend you only know what was in the clips!)

User: "Who is this guy?"
BAD: "Hey there! I'm Echo, your friendly podcast discovery AI..."
(Don't introduce yourself - answer the question!)
</BAD_RESPONSE_EXAMPLES>

<GOOD_RESPONSE_EXAMPLES>
User: "Who is IShowSpeed, his origins?"
GOOD: "IShowSpeed (real name Darren Watkins Jr.) is a popular American YouTuber and streamer known for his energetic personality and viral moments. He started streaming in 2017 at age 12, initially playing NBA 2K and Fortnite before blowing up during the pandemic. He's known for his over-the-top reactions and has become one of the biggest streamers on YouTube. In the clip you just watched, John Cena was impressed by his athleticism at the Royal Rumble. Want me to find more clips where people discuss him?"

User: "Who is Jelly Roll?" (after watching his Joe Rogan clips)
GOOD: "Jelly Roll is a country rapper and singer from Nashville. He had a troubled past involving prison and addiction, which he's been very open about. In the Joe Rogan clips we watched, he discussed his incredible 300-pound weight loss journey and transformation. Would you like to hear more about his music career or his recovery story?"
</GOOD_RESPONSE_EXAMPLES>

<output_format>
Respond with valid JSON:
{
  "response_text": "Your direct answer about the entity/topic (NO self-introduction)",
  "used_conversation_context": true,
  "follow_up_suggestions": ["Natural follow-up 1", "Natural follow-up 2"],
  "memory_update": {
    "turn_summary": "Brief summary (max 150 chars)",
    "action_type": "contextual_knowledge",
    "action_target_id": null,
    "action_target_title": "Entity/topic explained",
    "entities_mentioned": ["list of people/shows mentioned"],
    "topics_discussed": ["list of topics"],
    "is_topic_shift": false,
    "suggested_phase": "deep_dive"
  }
}
</output_format>"""

    user_prompt = f"""User asks: "{query}"

TARGET ENTITY: {target_entity_name or "Unknown - use context to determine"}

=== WHAT WE DISCUSSED (for context) ===
{entity_context}

=== RECENT CONVERSATION ===
{convo_context}

=== CURRENT TOPIC ===
{current_topic or "Established from conversation above"}

INSTRUCTIONS:
1. FULLY ANSWER the question about {target_entity_name or "the topic"} using YOUR KNOWLEDGE
2. The conversation context above is SUPPLEMENTARY - use it to add relevance, not to limit your answer
3. If you know facts about {target_entity_name or "the topic"} from your training, INCLUDE THEM
4. Reference the clips shown when relevant, but don't ONLY talk about what was in clips
5. End with a natural follow-up suggestion

EXAMPLE OF WHAT TO DO:
- User asks "Who is IShowSpeed?" after watching a clip where he was mentioned
- You should explain who IShowSpeed IS (streamer, real name, how he got famous)
- THEN connect to what was discussed in the clip

SUGGESTED FOLLOW-UPS TO ADAPT:
{follow_up_text}

Respond with JSON."""

    return system_prompt, user_prompt


def _build_memory_update_prompt(
    query: str,
    response_text: str,
    memory: ConversationMemory,
) -> Tuple[str, str]:
    """Build prompt for memory update call after grounding response."""

    system_prompt = """You are a memory extraction system.
Extract structured memory fields from a conversation turn.
Output valid JSON matching the specified schema."""

    user_prompt = f"""Extract memory update from this conversation turn:

USER QUERY: {query}

ASSISTANT RESPONSE: {response_text}

PREVIOUS CONTEXT:
- Last action: {memory.search_state.last_action.action_type or 'none'}
- Current entities: {memory.search_state.current_entities}
- Current topic: {memory.search_state.current_topic}

Extract and return JSON:
{{
  "turn_summary": "Brief summary of what was explained (max 150 chars)",
  "action_type": "explanation",
  "action_target_id": null,
  "action_target_title": "Topic that was explained",
  "entities_mentioned": ["list of people/shows/topics mentioned"],
  "topics_discussed": ["list of main topics discussed"],
  "is_topic_shift": false,
  "suggested_phase": "deep_dive"
}}"""

    return system_prompt, user_prompt


async def _call_grounded_search(
    query: str,
    memory: ConversationMemory,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Call native Gemini API with search grounding.

    Returns:
        Tuple of (response_text, sources)
    """
    global _native_genai_client

    if not _native_genai_client:
        raise RuntimeError("Native Gemini client not initialized")

    try:
        from google.genai import types

        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        gen_config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )

        prompt = _build_grounded_explanation_prompt(query, memory)

        response = await llm_call_with_retry(
            _native_genai_client.models.generate_content,
            model=GROUNDING_MODEL,
            contents=prompt,
            config=gen_config,
            operation_name="Grounded Search"
        )

        response_text = response.text

        # Extract sources from grounding metadata
        sources = []
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_metadata = candidate.grounding_metadata
                    if hasattr(grounding_metadata, 'grounding_chunks'):
                        for chunk in grounding_metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                sources.append({
                                    "title": getattr(chunk.web, 'title', 'Source'),
                                    "uri": getattr(chunk.web, 'uri', '')
                                })
        except Exception as e:
            logger.debug(f"[SMALL_TALK] Could not extract grounding sources: {e}")

        return response_text, sources

    except Exception as e:
        logger.error(f"[SMALL_TALK] Grounded search failed: {e}")
        raise


async def handle_small_talk(
    gemini_client,
    query: str,
    memory: ConversationMemory,
    router_output: RouterOutput,
) -> SmallTalkResponse:
    branch_start = time.time()

    logger.info("=" * 60)
    logger.info("[SMALL_TALK] ========== SMALL TALK START ==========")
    logger.info(f"[SMALL_TALK] Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    logger.info(f"[SMALL_TALK] Session: {memory.session_id}")
    logger.info(f"[SMALL_TALK] Router sub_intent: {router_output.sub_intent}")
    logger.info("=" * 60)

    # Intelligent response type detection using query analysis + memory context
    response_type, context_data = _detect_response_type(query, memory, router_output)
    logger.info(f"[SMALL_TALK] Response type: {response_type}")
    logger.debug(f"[SMALL_TALK] Context data keys: {list(context_data.keys())}")

    try:
        if response_type in ("greeting", "clarification", "off_topic"):
            # === SIMPLE RESPONSE PATH ===
            logger.info(f"[SMALL_TALK] Using simple response path ({response_type})")
            logger.info(f"[SMALL_TALK] Model: {SIMPLE_RESPONSE_MODEL}")

            system_prompt, user_prompt = _build_simple_response_prompt(
                query, response_type, memory, context_data
            )

            # =======================================================================
            # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
            # =======================================================================
            call_start = time.time()
            try:
                # Try structured output first
                resp = await llm_call_with_retry(
                    gemini_client.beta.chat.completions.parse,
                    model=SIMPLE_RESPONSE_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    reasoning_effort=SMALL_TALK_REASONING_EFFORT,
                    response_format=SmallTalkResponseOutput,
                    operation_name="Small Talk Response (Structured)"
                )
                call_time = time.time() - call_start
                logger.info(f"[SMALL_TALK] LLM call completed in {call_time:.2f}s")

                parsed_result = resp.choices[0].message.parsed
                if parsed_result:
                    logger.info("[SMALL_TALK] Structured output parsed successfully!")
                    result = parsed_result.model_dump()
                else:
                    raw_response = resp.choices[0].message.content
                    if raw_response:
                        result = json.loads(raw_response)
                    else:
                        raise ValueError("Both parsed and content are empty")

            except AttributeError:
                # Fallback to json_object mode
                logger.warning("[SMALL_TALK] Structured output not available, using json_object")
                resp = await llm_call_with_retry(
                    gemini_client.chat.completions.create,
                    model=SIMPLE_RESPONSE_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    reasoning_effort=SMALL_TALK_REASONING_EFFORT,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                    operation_name="Small Talk Response (JSON Fallback)"
                )
                call_time = time.time() - call_start
                logger.info(f"[SMALL_TALK] LLM call completed in {call_time:.2f}s")
                raw_response = resp.choices[0].message.content
                logger.debug(f"[SMALL_TALK] Raw response: {raw_response[:200]}...")
                result = json.loads(raw_response)

            # Extract follow-up suggestions (from LLM or fallback to context)
            follow_ups = result.get("follow_up_suggestions", context_data.get("follow_up_suggestions", []))

            # Extract memory update with defaults (Phase 2 enhanced)
            mem_data = result.get("memory_update", {})
            memory_update = BranchMemoryUpdate(
                turn_summary=mem_data.get("turn_summary", f"Responded to {response_type}")[:500],
                action_type=response_type,
                action_target_id=None,
                action_target_title=None,
                entities_mentioned=mem_data.get("entities_mentioned", [])[:10],
                topics_discussed=mem_data.get("topics_discussed", [])[:5],
                is_topic_shift=mem_data.get("is_topic_shift", False),
                suggested_phase=mem_data.get("suggested_phase", "discovery"),
                # Option A: Enhanced fields (empty for small talk - no media shown)
                key_quotes=[],
                topics_covered=mem_data.get("topics_discussed", [])[:5],
                notable_examples=[],
            )

            response = SmallTalkResponse(
                response_text=result.get("response_text", "Hello! I'm Echo, your podcast discovery assistant."),
                response_type=response_type,
                sources=[],
                memory_update=memory_update,
                follow_up_suggestions=follow_ups[:3],
            )

            branch_time = time.time() - branch_start
            logger.info(f"[SMALL_TALK] Response: {response.response_text[:100]}...")
            logger.info(f"[SMALL_TALK] Follow-ups: {follow_ups[:2]}")
            logger.info(f"[SMALL_TALK] Memory update action: {memory_update.action_type}")
            logger.info(f"[SMALL_TALK] Total time: {branch_time:.2f}s")
            logger.info("[SMALL_TALK] ========== SMALL TALK END ==========")

            return response

        elif response_type == "contextual_knowledge":
            # === CONTEXTUAL KNOWLEDGE PATH ===
            # User is asking about something in our conversation context
            logger.info("[SMALL_TALK] Using contextual knowledge path")
            logger.info(f"[SMALL_TALK] Model: {SIMPLE_RESPONSE_MODEL}")

            queried = [e["name"] for e in context_data.get("queried_entities", [])]
            resolved = context_data.get("resolved_entity")
            uses_pronoun = context_data.get("uses_pronoun", False)

            if uses_pronoun:
                logger.info(f"[SMALL_TALK] Pronoun resolution: query used pronoun -> resolved to '{resolved}'")
            logger.info(f"[SMALL_TALK] Queried entities: {queried}")

            system_prompt, user_prompt = _build_contextual_knowledge_prompt(
                query, context_data, memory
            )

            # =======================================================================
            # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
            # =======================================================================
            call_start = time.time()
            try:
                # Try structured output first
                resp = await llm_call_with_retry(
                    gemini_client.beta.chat.completions.parse,
                    model=SIMPLE_RESPONSE_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    reasoning_effort=SMALL_TALK_REASONING_EFFORT,
                    response_format=SmallTalkResponseOutput,
                    operation_name="Contextual Knowledge Response (Structured)"
                )
                call_time = time.time() - call_start
                logger.info(f"[SMALL_TALK] LLM call completed in {call_time:.2f}s")

                parsed_result = resp.choices[0].message.parsed
                if parsed_result:
                    logger.info("[SMALL_TALK] Structured output parsed successfully!")
                    result = parsed_result.model_dump()
                else:
                    raw_response = resp.choices[0].message.content
                    if raw_response:
                        result = json.loads(raw_response)
                    else:
                        raise ValueError("Both parsed and content are empty")

            except AttributeError:
                # Fallback to json_object mode
                logger.warning("[SMALL_TALK] Structured output not available, using json_object")
                resp = await llm_call_with_retry(
                    gemini_client.chat.completions.create,
                    model=SIMPLE_RESPONSE_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    reasoning_effort=SMALL_TALK_REASONING_EFFORT,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                    operation_name="Contextual Knowledge Response (JSON Fallback)"
                )
                call_time = time.time() - call_start
                logger.info(f"[SMALL_TALK] LLM call completed in {call_time:.2f}s")
                raw_response = resp.choices[0].message.content
                logger.debug(f"[SMALL_TALK] Raw response: {raw_response[:300]}...")
                result = json.loads(raw_response)

            # Extract follow-up suggestions from response
            follow_ups = result.get("follow_up_suggestions", context_data.get("follow_up_suggestions", []))

            # Extract memory update (Phase 2 enhanced)
            mem_data = result.get("memory_update", {})
            memory_update = BranchMemoryUpdate(
                turn_summary=mem_data.get("turn_summary", f"Explained {queried[0] if queried else 'entity'}")[:500],
                action_type="contextual_knowledge",
                action_target_id=None,
                action_target_title=queried[0] if queried else mem_data.get("action_target_title"),
                entities_mentioned=mem_data.get("entities_mentioned", queried)[:10],
                topics_discussed=mem_data.get("topics_discussed", [])[:5],
                is_topic_shift=mem_data.get("is_topic_shift", False),
                suggested_phase=mem_data.get("suggested_phase", "deep_dive"),
                # Option A: Enhanced fields (empty for small talk - no media shown)
                key_quotes=[],
                topics_covered=mem_data.get("topics_discussed", [])[:5],
                notable_examples=[],
            )

            response = SmallTalkResponse(
                response_text=result.get("response_text", f"I showed you some clips related to {queried[0] if queried else 'that topic'}."),
                response_type="contextual_knowledge",
                sources=[],
                memory_update=memory_update,
                follow_up_suggestions=follow_ups[:3],
            )

            branch_time = time.time() - branch_start
            logger.info(f"[SMALL_TALK] Response: {response.response_text[:100]}...")
            logger.info(f"[SMALL_TALK] Follow-ups: {follow_ups[:2]}")
            logger.info(f"[SMALL_TALK] Memory update action: {memory_update.action_type}")
            logger.info(f"[SMALL_TALK] Total time: {branch_time:.2f}s")
            logger.info("[SMALL_TALK] ========== SMALL TALK END ==========")

            return response

        else:
            # === EXPLANATION PATH ===
            logger.info("[SMALL_TALK] Using explanation path")

            sources = []
            follow_ups = context_data.get("follow_up_suggestions", [])

            if is_grounding_available():
                # Use native Gemini API with grounding
                logger.info("[SMALL_TALK] Grounding available - using native Gemini API")
                logger.info(f"[SMALL_TALK] Model: {GROUNDING_MODEL}")

                try:
                    grounding_start = time.time()
                    response_text, sources = await _call_grounded_search(query, memory)
                    grounding_time = time.time() - grounding_start
                    logger.info(f"[SMALL_TALK] Grounded search completed in {grounding_time:.2f}s")
                    logger.info(f"[SMALL_TALK] Sources found: {len(sources)}")
                    # For grounded search, use context follow-ups since grounding doesn't return them

                except Exception as e:
                    logger.warning(f"[SMALL_TALK] Grounding failed, falling back to standard: {e}")
                    # Fall back to standard response
                    response_text, follow_ups = await _get_standard_explanation(
                        gemini_client, query, memory, context_data
                    )

            else:
                # Use standard response (no grounding)
                logger.info("[SMALL_TALK] Grounding not available - using standard explanation")
                logger.info(f"[SMALL_TALK] Model: {GROUNDING_MODEL}")
                response_text, follow_ups = await _get_standard_explanation(
                    gemini_client, query, memory, context_data
                )

            # Memory update call (structured)
            logger.info("[SMALL_TALK] Extracting memory update...")
            memory_start = time.time()

            mem_system, mem_user = _build_memory_update_prompt(query, response_text, memory)

            try:
                # Try structured output first
                mem_resp = await llm_call_with_retry(
                    gemini_client.beta.chat.completions.parse,
                    model=MEMORY_UPDATE_MODEL,
                    messages=[
                        {"role": "system", "content": mem_system},
                        {"role": "user", "content": mem_user}
                    ],
                    temperature=0.0,
                    reasoning_effort=MEMORY_UPDATE_REASONING_EFFORT,
                    response_format=SmallTalkMemoryUpdateOutput,
                    operation_name="Memory Update Extraction (Structured)"
                )

                memory_time = time.time() - memory_start
                logger.info(f"[SMALL_TALK] Memory update call completed in {memory_time:.2f}s")

                parsed_result = mem_resp.choices[0].message.parsed
                if parsed_result:
                    logger.info("[SMALL_TALK] Memory update structured output parsed successfully!")
                    mem_result = parsed_result.model_dump()
                else:
                    raw_content = mem_resp.choices[0].message.content
                    if raw_content:
                        mem_result = json.loads(raw_content)
                    else:
                        raise ValueError("Both parsed and content are empty")

            except AttributeError:
                # Fallback to json_object mode
                logger.warning("[SMALL_TALK] Structured output not available for memory update, using json_object")
                mem_resp = await llm_call_with_retry(
                    gemini_client.chat.completions.create,
                    model=MEMORY_UPDATE_MODEL,
                    messages=[
                        {"role": "system", "content": mem_system},
                        {"role": "user", "content": mem_user}
                    ],
                    temperature=0.0,
                    reasoning_effort=MEMORY_UPDATE_REASONING_EFFORT,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    operation_name="Memory Update Extraction (JSON Fallback)"
                )
                memory_time = time.time() - memory_start
                logger.info(f"[SMALL_TALK] Memory update call completed in {memory_time:.2f}s")
                mem_result = json.loads(mem_resp.choices[0].message.content)

            except json.JSONDecodeError as e:
                logger.warning(f"[SMALL_TALK] Memory update JSON parse failed: {e}")
                mem_result = {}  # Use empty dict, will use defaults below

            memory_update = BranchMemoryUpdate(
                turn_summary=mem_result.get("turn_summary", f"Explained: {query[:50]}")[:500],
                action_type="explanation",
                action_target_id=None,
                action_target_title=mem_result.get("action_target_title", "Explanation"),
                entities_mentioned=mem_result.get("entities_mentioned", [])[:10],
                topics_discussed=mem_result.get("topics_discussed", [])[:5],
                is_topic_shift=mem_result.get("is_topic_shift", False),
                suggested_phase=mem_result.get("suggested_phase", "deep_dive"),
                # Option A: Enhanced fields (empty for small talk - no media shown)
                key_quotes=[],
                topics_covered=mem_result.get("topics_discussed", [])[:5],
                notable_examples=[],
            )

            response = SmallTalkResponse(
                response_text=response_text,
                response_type="explanation",
                sources=sources[:5],
                memory_update=memory_update,
                follow_up_suggestions=follow_ups[:3],
            )

            branch_time = time.time() - branch_start
            logger.info(f"[SMALL_TALK] Response: {response.response_text[:100]}...")
            logger.info(f"[SMALL_TALK] Follow-ups: {follow_ups[:2]}")
            logger.info(f"[SMALL_TALK] Memory update action: {memory_update.action_type}")
            logger.info(f"[SMALL_TALK] Memory update topic: {memory_update.action_target_title}")
            logger.info(f"[SMALL_TALK] Total time: {branch_time:.2f}s")
            logger.info("[SMALL_TALK] ========== SMALL TALK END ==========")

            return response

    except json.JSONDecodeError as e:
        logger.error(f"[SMALL_TALK] JSON parse error: {e}")
        return _create_fallback_response(query, "json_error")

    except Exception as e:
        logger.error(f"[SMALL_TALK] Error: {type(e).__name__}: {e}", exc_info=True)
        return _create_fallback_response(query, str(e)[:50])


async def _get_standard_explanation(
    gemini_client,
    query: str,
    memory: ConversationMemory,
    context_data: Dict[str, Any] = None,
) -> Tuple[str, List[str]]:
    """
    Get explanation response without grounding.

    Returns:
        Tuple of (response_text, follow_up_suggestions)
    """
    context_data = context_data or {}
    system_prompt, user_prompt = _build_explanation_prompt(query, memory, context_data)

    # Default follow-ups from context
    default_follow_ups = context_data.get("follow_up_suggestions", [])

    call_start = time.time()

    try:
        # Try structured output first
        resp = await llm_call_with_retry(
            gemini_client.beta.chat.completions.parse,
            model=GROUNDING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            reasoning_effort=GROUNDING_REASONING_EFFORT,
            response_format=StandardExplanationOutput,
            operation_name="Standard Explanation (Structured)"
        )
        call_time = time.time() - call_start
        logger.info(f"[SMALL_TALK] Standard explanation call completed in {call_time:.2f}s")

        parsed_result = resp.choices[0].message.parsed
        if parsed_result:
            logger.info("[SMALL_TALK] Standard explanation structured output parsed successfully!")
            response_text = parsed_result.response_text
            follow_ups = parsed_result.follow_up_suggestions or default_follow_ups
            logger.info(f"[SMALL_TALK] Extracted response_text: {response_text[:100]}...")
            logger.info(f"[SMALL_TALK] Extracted follow_ups: {follow_ups[:2]}")
            return response_text, follow_ups
        else:
            raw_content = resp.choices[0].message.content
            if raw_content:
                result = json.loads(raw_content)
                response_text = result.get("response_text", "I'd be happy to explain that further.")
                follow_ups = result.get("follow_up_suggestions", default_follow_ups)
                logger.info(f"[SMALL_TALK] Extracted response_text: {response_text[:100]}...")
                logger.info(f"[SMALL_TALK] Extracted follow_ups: {follow_ups[:2]}")
                return response_text, follow_ups
            else:
                raise ValueError("Both parsed and content are empty")

    except AttributeError:
        # Fallback to json_object mode
        logger.warning("[SMALL_TALK] Structured output not available for explanation, using json_object")
        resp = await llm_call_with_retry(
            gemini_client.chat.completions.create,
            model=GROUNDING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            reasoning_effort=GROUNDING_REASONING_EFFORT,
            max_tokens=800,
            response_format={"type": "json_object"},
            operation_name="Standard Explanation (JSON Fallback)"
        )
        call_time = time.time() - call_start
        logger.info(f"[SMALL_TALK] Standard explanation call completed in {call_time:.2f}s")

        raw_content = resp.choices[0].message.content
        if not raw_content:
            logger.error("[SMALL_TALK] Empty response from model")
            return "I'd be happy to help you with that. What would you like to know?", default_follow_ups

        try:
            result = json.loads(raw_content)
            response_text = result.get("response_text", "I'd be happy to explain that further.")
            follow_ups = result.get("follow_up_suggestions", default_follow_ups)
            logger.info(f"[SMALL_TALK] Extracted response_text: {response_text[:100]}...")
            logger.info(f"[SMALL_TALK] Extracted follow_ups: {follow_ups[:2]}")
            return response_text, follow_ups
        except json.JSONDecodeError as e:
            logger.warning(f"[SMALL_TALK] JSON parse failed in fallback: {e}")
            logger.warning(f"[SMALL_TALK] Raw content (first 500 chars): {raw_content[:500]}")

            # Try regex extraction as last resort
            if '"response_text"' in raw_content:
                patterns = [
                    r'"response_text"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'"response_text"\s*:\s*"([^"]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, raw_content, re.DOTALL)
                    if match:
                        extracted = match.group(1)
                        extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                        logger.info(f"[SMALL_TALK] Extracted response via regex: {extracted[:100]}...")
                        return extracted, default_follow_ups

            return "I'd be happy to explain that further. Could you tell me more about what you'd like to know?", default_follow_ups

    except json.JSONDecodeError as e:
        logger.warning(f"[SMALL_TALK] JSON parse failed in structured output: {e}")
        return "I'd be happy to explain that further. Could you tell me more about what you'd like to know?", default_follow_ups


def _create_fallback_response(query: str, error_hint: str) -> SmallTalkResponse:
    """Create a fallback response when processing fails."""

    logger.info(f"[SMALL_TALK] Creating fallback response (error: {error_hint})")

    return SmallTalkResponse(
        response_text="I apologize, I had a little trouble there. I'm Echo, your podcast discovery assistant. How can I help you find interesting podcast content today?",
        response_type="error",
        sources=[],
        memory_update=BranchMemoryUpdate(
            turn_summary=f"Error in small talk: {error_hint[:50]}",
            action_type="error",
            action_target_id=None,
            action_target_title=None,
            entities_mentioned=[],
            topics_discussed=[],
            is_topic_shift=False,
            suggested_phase="discovery",
            # Option A: Enhanced fields (empty for error case)
            key_quotes=[],
            topics_covered=[],
            notable_examples=[],
        ),
    )


def handle_small_talk_sync(
    gemini_client,
    query: str,
    memory: ConversationMemory,
    router_output: RouterOutput,
) -> SmallTalkResponse:
    """Synchronous version of handle_small_talk."""
    logger.debug("[SMALL_TALK] Using synchronous wrapper")
    return asyncio.run(handle_small_talk(gemini_client, query, memory, router_output))


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_response_type_description(response_type: str) -> str:
    """Get human-readable description of response type."""
    descriptions = {
        "greeting": "Greeting and persona introduction",
        "contextual_knowledge": "Answer about entity from conversation context",
        "explanation": "Explanation with context (optionally grounded)",
        "clarification": "Clarification request (no prior context)",
        "off_topic": "Off-topic redirect to podcasts",
        "error": "Error fallback response",
    }
    return descriptions.get(response_type, "Unknown response type")


def log_small_talk_summary(response: SmallTalkResponse, query: str) -> None:
    """Log a concise summary of the small talk interaction."""
    logger.info(
        f"[SMALL_TALK SUMMARY] '{query[:30]}...' -> {response.response_type} "
        f"(sources={len(response.sources)}, summary='{response.memory_update.turn_summary[:50]}...')"
    )
