"""
Episode Search Branch for EchoFind Conversational Agent.

Finds specific episodes by metadata (host, guest, show, date).
Does NOT handle topic-based queries (those go to clip_search).

Pipeline:
1. Extract intent (host, guest, show, time)
2. Generate 3 focused HyDE embeddings for episode-level content
3. Search Pinecone with metadata filters
4. Group chunks by EpisodeId, score at episode level
5. Fetch episode descriptions from RDS
6. Rerank episodes (not chunks)
7. LLM selects best episode + generates memory update

Key Design Principles:
- Episode-Level Scoring: Score at episode level, not chunk level
- Aggregation Strategy: Group search results by EpisodeId
- Rich Context: Fetch episode descriptions from RDS Episodes table
- Unified Memory: Uses BranchMemoryUpdate schema like all branches
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from pydantic import BaseModel, Field

import config
from retrieval.llm_utils import llm_call_with_retry


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class EpisodeTimeFilter(BaseModel):
    """Time filter for episode search."""
    mode: str = Field(default="none", description="Time mode: none, latest, oldest, between, before")
    start_date_utc: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD")
    end_date_utc: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    date_gating: str = Field(default="soft", description="Date filtering: hard or soft")
    recency_priority: str = Field(default="none", description="Recency priority: none, soft, hard")
    topic_present: bool = Field(default=False, description="Whether a topic is in the query")


class EpisodeIntentOutput(BaseModel):
    """Structured output for episode intent extraction."""
    resolved_query: str = Field(description="Fully resolved query with context applied")
    hosts: List[str] = Field(default_factory=list, description="Host names (lowercase)")
    guests: List[str] = Field(default_factory=list, description="Guest names (lowercase)")
    show_name: Optional[str] = Field(default=None, description="Show/podcast name (lowercase)")
    time_filter: EpisodeTimeFilter = Field(default_factory=EpisodeTimeFilter)
    intent_summary: str = Field(default="", description="Brief intent summary (10 words max)")


class EpisodeHydeOutput(BaseModel):
    """Structured output for HyDE document generation."""
    hyde_docs: List[str] = Field(default_factory=list, description="3 hypothetical episode intros")


class EpisodeMemoryUpdate(BaseModel):
    """Memory update from episode selection (Phase 2 enhanced)."""
    turn_summary: str = Field(default="", description="Brief summary (max 500 chars)")
    entities_mentioned: List[str] = Field(default_factory=list, description="People/shows mentioned (max 10)")
    topics_discussed: List[str] = Field(default_factory=list, description="Topics discussed (max 5)")
    is_topic_shift: bool = Field(default=True, description="Whether this is a new topic")
    suggested_phase: str = Field(default="discovery", description="Conversation phase")

    # Option A: Enhanced context fields for query analyzer
    key_quotes: List[str] = Field(
        default_factory=list,
        description="2-3 memorable quotes from episode description/content"
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="Specific topics/subtopics in the episode (max 5)"
    )
    notable_examples: List[str] = Field(
        default_factory=list,
        description="Notable guests, topics, or highlights (max 3)"
    )


class EpisodeSelectionOutput(BaseModel):
    """Structured output for episode selection (Phase 2 with Quote Extraction)."""

    # Phase 2: Quote extraction for better selection
    relevant_quotes: List[str] = Field(
        default_factory=list,
        description="2-4 quotes from episode descriptions that match the query"
    )

    chosen_index: int = Field(default=0, description="Index of selected episode (0-based)")
    response_text: str = Field(description="Natural response describing the episode")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Selection confidence")
    memory_update: EpisodeMemoryUpdate = Field(default_factory=EpisodeMemoryUpdate)

import psycopg2
from psycopg2.extras import RealDictCursor
from thefuzz import process, fuzz

import config
from engine.memory import ConversationMemory
from engine.schemas import EpisodeSearchResponse, BranchMemoryUpdate, RouterOutput
from retrieval.search_filter import build_filter, build_episode_filter, _build_date_clause, normalize_name
from retrieval.gazetteer import get_gazetteer
from retrieval.data_fetcher import (
    concurrent_embedding_generation,
    concurrent_sparse_embedding_generation,
    concurrent_pinecone_search,
    combine_pinecone_results,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION (from config.py for easy model switching)
# ==============================================================================

# Models - load from config (centralized in config.py for easy switching)
EPISODE_SEARCH_MODEL = config.EPISODE_SEARCH_MODEL
EPISODE_INTENT_MODEL = config.EPISODE_INTENT_MODEL
EPISODE_HYDE_COUNT = getattr(config, 'EPISODE_SEARCH_HYDE_COUNT', 3)
EPISODE_MAX_CANDIDATES = getattr(config, 'EPISODE_SEARCH_MAX_EPISODES', 15)
EPISODE_PINECONE_K = 100  # Top K from Pinecone per query
EPISODE_TARGET_PER_QUERY = 30  # Target chunks per query

# Reasoning effort settings
EPISODE_INTENT_REASONING_EFFORT = config.EPISODE_INTENT_REASONING_EFFORT
EPISODE_HYDE_REASONING_EFFORT = config.EPISODE_HYDE_REASONING_EFFORT
EPISODE_SELECTION_REASONING_EFFORT = config.EPISODE_SELECTION_REASONING_EFFORT

# RDS Configuration
RDS_HOST = getattr(config, 'RDS_HOST', 'localhost')
RDS_PORT = getattr(config, 'RDS_PORT', 5432)
RDS_DATABASE = getattr(config, 'RDS_DATABASE', 'podcast_content')
RDS_USERNAME = getattr(config, 'RDS_USERNAME', 'postgres')
RDS_PASSWORD = getattr(config, 'RDS_PASSWORD', '')


# ==============================================================================
# STEP 1: INTENT EXTRACTION
# ==============================================================================

async def extract_episode_intent(
    gemini_client,
    query: str,
    memory: ConversationMemory,
    router_output: Optional[RouterOutput] = None,
) -> Dict[str, Any]:
    """
    Extract episode search intent: host, guest, show, time constraints.

    This is simpler than query_analyzer because we only need metadata,
    not topic analysis or HyDE document generation.

    Args:
        gemini_client: Gemini client (OpenAI-compatible)
        query: User's query
        memory: Conversation memory for context
        router_output: Optional router output with sub_intent

    Returns:
        Dict with resolved_query, hosts, guests, show_name, time_filter, intent_summary
    """
    intent_start = time.time()

    logger.info(f"[EPISODE_SEARCH] Extracting intent from: {query[:80]}...")

    today = datetime.now(timezone.utc)
    today_str = today.strftime('%Y-%m-%d')

    # Date context for relative time parsing
    date_context = {
        "today": today_str,
        "recent_window": (today - timedelta(days=config.RECENT_WINDOW_DAYS_DEFAULT)).strftime('%Y-%m-%d'),
        "one_week_ago": (today - timedelta(days=7)).strftime('%Y-%m-%d'),
        "one_month_ago": (today - timedelta(days=30)).strftime('%Y-%m-%d'),
        "three_months_ago": (today - timedelta(days=90)).strftime('%Y-%m-%d'),
        "six_months_ago": (today - timedelta(days=180)).strftime('%Y-%m-%d'),
        "one_year_ago": (today - timedelta(days=365)).strftime('%Y-%m-%d'),
    }

    # =========================================================================
    # GAZETTEER LOOKUP - Find known entities in query (like clip search)
    # =========================================================================
    gazetteer = get_gazetteer()
    relevant_candidates = gazetteer.search(query, top_k=15)

    # Categorize candidates by type
    authors_lower = {a.lower() for a in gazetteer.authors}
    personalities_lower = {p.lower() for p in gazetteer.personalities}
    shows_lower = {s.lower() for s in gazetteer.shows}

    host_candidates = [c for c in relevant_candidates if c.lower() in authors_lower]
    guest_candidates = [c for c in relevant_candidates if c.lower() in personalities_lower]
    show_candidates = [c for c in relevant_candidates if c.lower() in shows_lower]

    logger.info(f"[EPISODE_SEARCH] Gazetteer lookup:")
    logger.info(f"  ├─ Host candidates: {host_candidates[:5]}")
    logger.info(f"  ├─ Guest candidates: {guest_candidates[:5]}")
    logger.info(f"  └─ Show candidates: {show_candidates[:5]}")

    # Build entity context for LLM prompt
    entity_context_parts = []
    if host_candidates:
        entity_context_parts.append(f"Known Hosts in database: {', '.join(host_candidates[:8])}")
    if guest_candidates:
        entity_context_parts.append(f"Known Guests in database: {', '.join(guest_candidates[:8])}")
    if show_candidates:
        entity_context_parts.append(f"Known Shows in database: {', '.join(show_candidates[:5])}")

    entity_context = "\n".join(entity_context_parts) if entity_context_parts else "No specific entity matches found in database."

    # Memory context for pronoun resolution - USE FULL CONTEXT like query_analyzer
    state_context = memory.search_state.render_for_query_analyzer(memory.recent_turns)

    # Get sub_intent hint from router
    sub_intent_hint = ""
    if router_output and router_output.sub_intent:
        sub_intent_hint = f"Router sub_intent: {router_output.sub_intent}"

    system_prompt = f"""<task>
Extract episode search intent from the query.

ONLY extract: hosts, guests, show names, and time constraints.
DO NOT extract topics or themes (those would go to clip_search).

Today's date: {today_str}
</task>

<known_entities>
{entity_context}

CRITICAL RULES FOR ENTITY EXTRACTION:
1. If a person is listed as a "Known Host", put them in the "hosts" array
2. If a person is listed as a "Known Guest", put them in the "guests" array
3. If query mentions a show by name OR by host name, use the exact show name from "Known Shows"
4. "Joe Rogan" = host of "The Joe Rogan Experience" → hosts: ["joe rogan"], show_name: "the joe rogan experience"
5. "Lex Fridman" = host of "Lex Fridman Podcast" → hosts: ["lex fridman"], show_name: "lex fridman podcast"
6. ALWAYS prefer exact matches from the known entities lists above
</known_entities>

<context>
{state_context}
{sub_intent_hint}
</context>

<output>
Return valid JSON:
{{
  "resolved_query": "Query with pronouns resolved (use context)",
  "hosts": ["host1", "host2"],
  "guests": ["guest1", "guest2"],
  "show_name": "exact show name from known_entities or null",
  "episode_identifier": {{
    "has_specific_episode": true|false,
    "episode_number": <integer> or null,
    "episode_title_hint": "<partial title or guest name>" or null
  }},
  "time_filter": {{
    "has_time_constraint": true/false,
    "mode": "latest|oldest|before|after|between|none",
    "start_date_utc": "YYYY-MM-DD" or null,
    "end_date_utc": "YYYY-MM-DD" or null,
    "date_gating": "hard|soft",
    "recency_priority": "hard|soft|none",
    "topic_present": true|false
  }},
  "intent_summary": "Brief summary of what user wants (max 50 chars)"
}}
</output>

<episode_identifier_rules>
EXTRACT SPECIFIC EPISODE REFERENCES:
- "JRE 2422" → has_specific_episode=true, episode_number=2422
- "episode #150" → has_specific_episode=true, episode_number=150
- "the Elon Musk episode" → has_specific_episode=true, episode_title_hint="Elon Musk"
- "Lex #100" → has_specific_episode=true, episode_number=100
- "latest JRE" → has_specific_episode=false (not a specific episode)

Episode numbers appear in titles as "#<number>" - this helps identify the exact episode.
</episode_identifier_rules>

<topic_present_rules>
CRITICAL: Determines search strategy (semantic vs pure metadata).

topic_present = TRUE when query has a SEMANTIC SUBJECT beyond entity + time:
- "latest JRE about AI" → topic_present=true (topic: AI)
- "Joe Rogan episode on fitness" → topic_present=true (topic: fitness)
- "Lex Fridman discussing consciousness" → topic_present=true

topic_present = FALSE when query is ONLY entity/show + time + guest:
- "latest Joe Rogan episode" → topic_present=false
- "JRE 2422" → topic_present=false
- "Joe Rogan episode with Elon Musk" → topic_present=false (guest filter, not topic)
- "Invest Like the Best with Ari Emanuel" → topic_present=false

RULE: If the query could be answered by filtering metadata ONLY (host, guest, show, date, episode number), then topic_present=false.
</topic_present_rules>

<pronoun_resolution>
Use the context above to resolve pronouns:
- "his/her/their episode" -> Use ENTITIES from context
- "that show" -> Use TOPIC or LAST_TARGET from context
- "another one" -> Use previous host/guest from context
</pronoun_resolution>

<context_persistence_for_relative_queries>
CRITICAL RULE: Relative references INHERIT host/show from context.

Relative references include:
- "the one before that", "the previous episode", "before that"
- "from last month", "from 2023", "older episodes"
- "the next one", "what came after"
- "another one", "more from them"

RULE: If query has a relative time/sequence reference BUT NO explicit new host/guest/show,
you MUST inherit host/show from the LAST_ACTION context provided above.

Examples:
CONTEXT: Last episode = "Huberman Lab" hosted by "Dr. Andrew Huberman"
QUERY: "the one before that"
CORRECT OUTPUT: hosts=["dr. andrew huberman"], show_name="huberman lab", time_filter.mode="before"
WRONG OUTPUT: hosts=[], show_name=null  ← This loses context!

CONTEXT: Last episode = "Lex Fridman Podcast #69 - David Chalmers"
QUERY: "from last month"
CORRECT OUTPUT: hosts=["lex fridman"], show_name="lex fridman podcast", time_filter.mode="between"
WRONG OUTPUT: hosts=[], show_name=null  ← User clearly wants Lex episodes from last month!

CONTEXT: Last clip showed "Andrew Huberman" discussing "gut health"
QUERY: "his episode from when he first talked about this"
CORRECT OUTPUT: hosts=["dr. andrew huberman"], topic_present=true (topic: gut health), time_filter.mode="oldest"
</context_persistence_for_relative_queries>

<explicit_reference_resolution>
CRITICAL: "that guest", "the guest", "that host" must resolve from LAST_ACTION context.

RULE: Look at the Content/Title in the context to identify who "that guest" or "that host" refers to.

Episode title patterns:
- "David Chalmers: The Hard Problem | Lex Fridman Podcast #69" → guest = "David Chalmers", host = "Lex Fridman"
- "Dr. Peter Attia on Longevity | Huberman Lab" → guest = "Dr. Peter Attia", host = "Dr. Andrew Huberman"
- "essentials: build a healthy gut | huberman lab" → no guest, host = "Dr. Andrew Huberman"

Examples:
CONTEXT: Last clip = "David Chalmers: Hard Problem | Lex Fridman #69"
QUERY: "who is that guest, and more from him on other shows"
CORRECT: guests=["david chalmers"], show_name=null (user wants OTHER shows)
WRONG: guests=["geoffrey hinton"]  ← Do NOT hallucinate different people!

CONTEXT: Last episode showed Jelly Roll on Joe Rogan
QUERY: "more from that guest"
CORRECT: guests=["jelly roll"]
WRONG: guests=[]  ← Must resolve "that guest" from context!
</explicit_reference_resolution>

<date_reference>
- "latest", "most recent", "newest" -> mode="latest", start_date={date_context['recent_window']}, date_gating="hard", recency_priority="hard"
- "last week" -> between {date_context['one_week_ago']} and {date_context['today']}, date_gating="hard"
- "last month" -> between {date_context['one_month_ago']} and {date_context['today']}, date_gating="hard"
- "this year" -> between {today.year}-01-01 and {date_context['today']}, date_gating="hard"
- "2023" or "2024" -> between YEAR-01-01 and YEAR-12-31, date_gating="hard"
- No date mentioned -> mode="none", date_gating="soft", recency_priority="none"

DATE GATING RULES:
- "hard": User explicitly mentioned a time. Filter MUST match strictly.
- "soft": No date mentioned. Semantic search only, no date filter.
- DEFAULT to "hard" for any time-related query in Episode Search.

RECENCY PRIORITY:
- "hard": Recency is the PRIMARY concern (e.g., "latest episode")
- "soft": Recency matters but topic also matters
- "none": No recency preference
</date_reference>"""

    # Build user prompt with few-shot examples for better accuracy
    user_prompt = f"""=== FEW-SHOT EXAMPLES ===

Example 1 - Relative query inherits context:
CONTEXT: Last episode = "Huberman Lab - Sleep Essentials", PRIMARY entity = "Dr. Andrew Huberman"
QUERY: "the one before that"
OUTPUT: {{"resolved_query": "Previous Huberman Lab episode", "hosts": ["dr. andrew huberman"], "guests": [], "show_name": "huberman lab", "time_filter": {{"mode": "before", "date_gating": "hard", "recency_priority": "hard", "topic_present": false}}, "intent_summary": "Previous Huberman episode"}}

---
Example 2 - "That guest" resolves from title:
CONTEXT: Last clip = "David Chalmers: Hard Problem | Lex Fridman #69", entities = ["Lex Fridman", "David Chalmers"]
QUERY: "more from that guest on other shows"
OUTPUT: {{"resolved_query": "David Chalmers episodes on other podcasts", "hosts": [], "guests": ["david chalmers"], "show_name": null, "time_filter": {{"mode": "none", "topic_present": false}}, "intent_summary": "David Chalmers other appearances"}}

---
Example 3 - Date filter inherits ALL context (host AND guest):
CONTEXT: Last clip = "Joe Rogan Experience #2404 - Elon Musk", entities = ["Joe Rogan", "Elon Musk"]
QUERY: "from 2023"
OUTPUT: {{"resolved_query": "Joe Rogan with Elon Musk episodes from 2023", "hosts": ["joe rogan"], "guests": ["elon musk"], "show_name": "the joe rogan experience", "time_filter": {{"mode": "between", "start_date_utc": "2023-01-01", "end_date_utc": "2023-12-31", "date_gating": "hard", "topic_present": false}}, "intent_summary": "JRE with Elon from 2023"}}
NOTE: User wants the SAME host+guest combination, just from a different time period. Inherit BOTH.

---
Example 4 - Date filter inherits host only (no guest in context):
CONTEXT: Last episode = "Huberman Lab - Sleep Essentials", PRIMARY = "Dr. Andrew Huberman"
QUERY: "from last month"
OUTPUT: {{"resolved_query": "Huberman Lab episodes from last month", "hosts": ["dr. andrew huberman"], "guests": [], "show_name": "huberman lab", "time_filter": {{"mode": "between", "start_date_utc": "{date_context['one_month_ago']}", "end_date_utc": "{date_context['today']}", "date_gating": "hard", "topic_present": false}}, "intent_summary": "Huberman from last month"}}

---
Example 5 - Pronoun with topic inherits context:
CONTEXT: PRIMARY = "Scott Galloway", TOPIC = "economics"
QUERY: "his episode from when he first talked about this"
OUTPUT: {{"resolved_query": "Scott Galloway earliest episode on economics", "hosts": ["scott galloway"], "guests": [], "show_name": null, "time_filter": {{"mode": "oldest", "date_gating": "hard", "topic_present": true}}, "intent_summary": "Galloway first economics episode"}}

---
Example 5 - Fresh query (no inheritance needed):
CONTEXT: None
QUERY: "Latest Huberman"
OUTPUT: {{"resolved_query": "Latest Huberman Lab episode", "hosts": ["dr. andrew huberman"], "guests": [], "show_name": "huberman lab", "time_filter": {{"mode": "latest", "start_date_utc": "{date_context['six_months_ago']}", "date_gating": "hard", "recency_priority": "hard", "topic_present": false}}, "intent_summary": "Latest Huberman Lab"}}

=== CURRENT TASK ===

SIMPLE MEMORY (for quick reference):
- Current entities: {json.dumps(memory.search_state.current_entities or [])}
- Current topic: {json.dumps(memory.search_state.current_topic) if memory.search_state.current_topic else 'null'}
- Thread topic: {json.dumps(memory.search_state.conversation_thread_topic) if memory.search_state.conversation_thread_topic else 'null'}
- Conversation participants: {json.dumps(memory.search_state.conversation_participants or [])}
- Last action: {memory.search_state.last_action.action_type or 'none'}
- Last shown content: {json.dumps(memory.search_state.last_action.target_title) if memory.search_state.last_action.target_title else 'null'}

FULL CONVERSATION CONTEXT (use this to resolve "that guest", "the host", pronouns, and relative queries):
{state_context}

QUERY: "{query}"

IMPORTANT INSTRUCTIONS:
1. Use FULL CONVERSATION CONTEXT above to resolve pronouns (he/she/they/him/her) and references ("that guest", "the host")
2. "that guest" = look at last shown content title, extract the GUEST name (person before ":" or "|" who isn't the host)
3. "the host" / "that host" = look at show name in last content and map to host (Huberman Lab → Dr. Andrew Huberman)
4. For relative queries ("the one before", "from last month") with NO new entity → INHERIT host/show from context
5. Check "Last shown content" to see what episode/clip was displayed - this tells you WHO was in it
6. "more from him/her" → resolve pronoun to person from context, then search for their content

OUTPUT:"""

    # =======================================================================
    # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
    # =======================================================================
    try:
        try:
            # Try structured output first
            logger.info(f"[EPISODE_SEARCH] Calling intent extraction with structured output...")
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=EPISODE_INTENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                reasoning_effort=EPISODE_INTENT_REASONING_EFFORT,
                response_format=EpisodeIntentOutput,
                operation_name="Episode Intent Extraction (Structured)"
            )

            if not resp.choices:
                logger.error("[EPISODE_SEARCH] Empty response from intent LLM")
                return _create_fallback_intent(query)

            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                logger.info("[EPISODE_SEARCH] Structured output parsed successfully!")
                result = parsed_result.model_dump()
            else:
                # Fallback to content parsing
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError as attr_err:
            # beta.chat.completions.parse not available - fall back to json_object mode
            logger.warning(f"[EPISODE_SEARCH] Structured output not available ({attr_err}), using json_object")
            resp = await llm_call_with_retry(
                gemini_client.chat.completions.create,
                model=EPISODE_INTENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                reasoning_effort=EPISODE_INTENT_REASONING_EFFORT,
                max_tokens=400,
                response_format={"type": "json_object"},
                operation_name="Episode Intent Extraction (JSON Fallback)"
            )
            result = json.loads(resp.choices[0].message.content)

        # =====================================================================
        # POST-PROCESS: Apply context inheritance if LLM failed to do so
        # =====================================================================
        result = _apply_context_inheritance_fallback(result, query, memory)

        intent_time = time.time() - intent_start
        logger.info(f"[EPISODE_SEARCH] Intent extracted in {intent_time:.3f}s:")
        logger.info(f"  ├─ Resolved: {result.get('resolved_query', query)[:60]}...")
        logger.info(f"  ├─ Hosts: {result.get('hosts', [])}")
        logger.info(f"  ├─ Guests: {result.get('guests', [])}")
        logger.info(f"  ├─ Show: {result.get('show_name')}")
        logger.info(f"  ├─ Time: mode={result.get('time_filter', {}).get('mode', 'none')}, gating={result.get('time_filter', {}).get('date_gating', 'soft')}")
        logger.info(f"  └─ Summary: {result.get('intent_summary', 'N/A')}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"[EPISODE_SEARCH] Intent JSON parse error: {e}")
        return _create_fallback_intent(query)
    except Exception as e:
        logger.error(f"[EPISODE_SEARCH] Intent extraction failed: {e}")
        return _create_fallback_intent(query)


def _create_fallback_intent(query: str) -> Dict[str, Any]:
    """Create fallback intent when extraction fails."""
    import re

    # Detect topic presence using word boundaries to avoid false positives
    # e.g., "Elon" should not match "on", "with" should not trigger topic
    topic_patterns = [
        r'\babout\b',
        r'\bon\b',  # "on" as standalone word
        r'\bdiscussing\b',
        r'\btalked about\b',
        r'\bsaid about\b',
        r'\bopinion on\b',
        r'\bviews on\b',
        r'\bthoughts on\b',
    ]
    query_lower = query.lower()
    has_topic = any(re.search(pattern, query_lower) for pattern in topic_patterns)

    # Detect episode number (e.g., "JRE 2422", "#150", "episode 100")
    episode_number = None
    ep_match = re.search(r'#?(\d{2,4})(?:\s|$)', query)
    if ep_match:
        episode_number = int(ep_match.group(1))

    return {
        "resolved_query": query,
        "hosts": [],
        "guests": [],
        "show_name": None,
        "episode_identifier": {
            "has_specific_episode": episode_number is not None,
            "episode_number": episode_number,
            "episode_title_hint": None,
        },
        "time_filter": {
            "has_time_constraint": False,
            "mode": "none",
            "date_gating": "soft",
            "recency_priority": "none",
            "topic_present": has_topic,
        },
        "intent_summary": query[:50],
    }


def _apply_context_inheritance_fallback(
    result: Dict[str, Any],
    query: str,
    memory: ConversationMemory,
) -> Dict[str, Any]:
    """
    Post-process LLM result to apply context inheritance if it failed to do so.

    This catches cases where LLM returned empty hosts/show for relative queries
    that should have inherited from context.
    """
    import re

    query_lower = query.lower().strip()

    # Check if this looks like a relative/continuation query
    relative_patterns = [
        r'\bthe one before\b', r'\bbefore that\b', r'\bprevious\b',
        r'\bthe next\b', r'\bafter that\b',
        r'\banother one\b', r'\bmore from\b',
        r'\bfrom last\b', r'\bfrom this\b', r'\blast month\b', r'\blast week\b',
        r'\bthat guest\b', r'\bthe guest\b', r'\bthat host\b', r'\bthe host\b',
    ]

    # Check if this is a DATE-ONLY query (just a year or time reference)
    # These should inherit FULL context (host AND guest)
    date_only_patterns = [
        r'^from\s+\d{4}$',           # "from 2023"
        r'^in\s+\d{4}$',              # "in 2023"
        r'^\d{4}$',                   # just "2023"
        r'^from\s+last\s+(month|week|year)$',  # "from last month"
        r'^last\s+(month|week|year)$',         # "last month"
        r'^this\s+(month|week|year)$',         # "this year"
    ]

    is_relative = any(re.search(p, query_lower) for p in relative_patterns)
    is_date_only = any(re.search(p, query_lower) for p in date_only_patterns)

    # Get current result values
    hosts = result.get("hosts", [])
    show_name = result.get("show_name")
    guests = result.get("guests", [])

    # For date-only queries, we should inherit BOTH host AND guest from context
    # Even if host was already extracted
    if is_date_only and not guests:
        state = memory.search_state
        # Try to get guest from current entities (excluding the host)
        current_entities = state.current_entities or []
        last_title = state.last_action.target_title or ""

        # Extract potential guest from entities (entity that isn't the host)
        host_names = [h.lower() for h in hosts] if hosts else []
        for entity in current_entities:
            entity_lower = entity.lower()
            # Skip if this entity is already in hosts
            if any(host_name in entity_lower or entity_lower in host_name for host_name in host_names):
                continue
            # This entity is likely the guest
            result["guests"] = [entity_lower]
            logger.info(f"[EPISODE_SEARCH] Applied date-only inheritance: guest='{entity_lower}' from context")
            break

        return result

    # For other relative queries, only apply fallback if NOTHING was extracted
    if not is_relative:
        return result

    if hosts or show_name or guests:
        # LLM did extract something, trust it
        return result

    # LLM returned nothing - try to inherit from context
    state = memory.search_state
    last_title = state.last_action.target_title or ""
    entities = state.current_entities or []

    # Try to extract host/show from last title
    inherited_host = None
    inherited_show = None
    inherited_guest = None

    # Known show patterns
    show_host_map = {
        'huberman lab': 'dr. andrew huberman',
        'lex fridman': 'lex fridman',
        'joe rogan': 'joe rogan',
        'tim ferriss': 'tim ferriss',
        'prof g': 'scott galloway',
        'all-in': 'chamath palihapitiya',
    }

    last_title_lower = last_title.lower()
    for show_pattern, host in show_host_map.items():
        if show_pattern in last_title_lower:
            inherited_host = host
            inherited_show = show_pattern
            break

    # If "that guest" pattern, try to extract guest from title
    if 'that guest' in query_lower or 'the guest' in query_lower:
        # Title pattern: "Guest Name: Topic | Show" or "Guest Name | Show"
        if ':' in last_title:
            potential_guest = last_title.split(':')[0].strip()
            if potential_guest and len(potential_guest.split()) <= 4:
                inherited_guest = potential_guest.lower()
        elif '|' in last_title:
            potential_guest = last_title.split('|')[0].strip()
            if potential_guest and len(potential_guest.split()) <= 4:
                inherited_guest = potential_guest.lower()

    # Apply inheritance
    if inherited_host and not hosts:
        result["hosts"] = [inherited_host]
        logger.info(f"[EPISODE_SEARCH] Applied inheritance fallback: host='{inherited_host}'")

    if inherited_show and not show_name:
        result["show_name"] = inherited_show
        logger.info(f"[EPISODE_SEARCH] Applied inheritance fallback: show='{inherited_show}'")

    if inherited_guest and not guests:
        result["guests"] = [inherited_guest]
        logger.info(f"[EPISODE_SEARCH] Applied inheritance fallback: guest='{inherited_guest}'")

    return result


# ==============================================================================
# STEP 2: GENERATE EPISODE-FOCUSED HYDE EMBEDDINGS
# ==============================================================================

async def generate_episode_hyde(
    gemini_client,
    intent: Dict[str, Any],
) -> List[str]:
    """
    Generate HyDE documents focused on episode-level content.

    Unlike clip search HyDE (which emulates transcript snippets),
    these emulate episode introductions/descriptions - how a host
    would introduce an episode at the start.

    Args:
        gemini_client: Gemini client
        intent: Extracted intent with hosts, guests, show_name

    Returns:
        List of HyDE document strings (3 by default)
    """
    hyde_start = time.time()

    resolved_query = intent.get("resolved_query", "")
    hosts = intent.get("hosts", [])
    guests = intent.get("guests", [])
    show_name = intent.get("show_name")

    # Build context for HyDE generation
    context_parts = []
    if hosts:
        context_parts.append(f"Host(s): {', '.join(hosts)}")
    if guests:
        context_parts.append(f"Guest(s): {', '.join(guests)}")
    if show_name:
        context_parts.append(f"Show: {show_name}")

    context = "\n".join(context_parts) if context_parts else "General podcast episode"

    system_prompt = """You are a podcast content generator.
Generate hypothetical podcast episode introductions that sound natural.

Guidelines:
- Write how a host would introduce an episode at the start
- Include guest and host names naturally in the text
- Keep each document to 40-60 words
- Sound conversational, not like metadata

Output JSON:
{
  "hyde_docs": ["doc1", "doc2", "doc3"]
}
"""

    user_prompt = f"""Generate 3 different hypothetical podcast episode introductions.

Query intent: {resolved_query}
Context:
{context}

Generate 3 varied introductions (opening statements, descriptions, announcements)."""

    # =======================================================================
    # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
    # =======================================================================
    try:
        try:
            # Try structured output first
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=EPISODE_INTENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,  # Higher temp for diversity
                reasoning_effort=EPISODE_HYDE_REASONING_EFFORT,
                response_format=EpisodeHydeOutput,
                operation_name="Episode HyDE Generation (Structured)"
            )

            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                hyde_docs = parsed_result.hyde_docs
            else:
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                    hyde_docs = result.get("hyde_docs", [])
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError:
            # Fallback to json_object mode
            resp = await llm_call_with_retry(
                gemini_client.chat.completions.create,
                model=EPISODE_INTENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                reasoning_effort=EPISODE_HYDE_REASONING_EFFORT,
                max_tokens=500,
                response_format={"type": "json_object"},
                operation_name="Episode HyDE Generation (JSON Fallback)"
            )
            result = json.loads(resp.choices[0].message.content)
            hyde_docs = result.get("hyde_docs", [])

        # Ensure we have the right count
        hyde_docs = hyde_docs[:EPISODE_HYDE_COUNT]

        hyde_time = time.time() - hyde_start
        logger.info(f"[EPISODE_SEARCH] Generated {len(hyde_docs)} HyDE documents in {hyde_time:.3f}s")
        for i, doc in enumerate(hyde_docs):
            logger.debug(f"  HyDE[{i}]: {doc[:60]}...")

        return hyde_docs if hyde_docs else [resolved_query]

    except Exception as e:
        logger.error(f"[EPISODE_SEARCH] HyDE generation failed: {e}")
        # Fallback: use resolved query variations
        return [
            resolved_query,
            f"Welcome to today's episode featuring {', '.join(guests) if guests else 'our special guest'}",
            f"On this episode of {show_name or 'the podcast'}, we talk with {', '.join(guests) if guests else 'our guest'}",
        ][:EPISODE_HYDE_COUNT]


# ==============================================================================
# STEP 3: EPISODE-LEVEL SCORING (Aggregation Strategy)
# ==============================================================================

def _determine_weight_profile(intent: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
    """
    Determine scoring weights based on query intent.

    Returns: (weights_dict, profile_name)

    Weight profiles are tuned for different query types:
    - PURE_RECENCY: "latest JRE" - recency dominates, semantic meaningless
    - SPECIFIC_EPISODE: "JRE 2422" - episode_number_boost handles this
    - GUEST_FOCUSED: "JRE with Elon" - person match is critical
    - TOPIC_RECENCY: "latest JRE about AI" - balance semantic + recency
    """
    time_filter = intent.get("time_filter", {})
    mode = (time_filter.get("mode") or "none").lower()
    topic_present = time_filter.get("topic_present", False)
    recency_priority = (time_filter.get("recency_priority") or "none").lower()

    episode_identifier = intent.get("episode_identifier", {})
    has_specific_episode = episode_identifier.get("has_specific_episode", False)

    intent_guests = intent.get("guests", [])
    intent_hosts = intent.get("hosts", [])
    intent_show = intent.get("show_name")

    # ================================================================
    # PRIORITY 1: Specific Episode Number (e.g., "JRE 2422")
    # ================================================================
    if has_specific_episode:
        # Low weights because episode_number_boost (+0.5) does the heavy lifting
        return {
            "semantic": 0.15,
            "recency": 0.10,
            "person": 0.20,
            "show": 0.15,
        }, "SPECIFIC_EPISODE"

    # ================================================================
    # PRIORITY 2: Pure Recency - No Topic (e.g., "latest JRE")
    # ================================================================
    if mode == "latest" and not topic_present:
        return {
            "semantic": 0.05,
            "recency": 0.70,
            "person": 0.15,
            "show": 0.10,
        }, "PURE_RECENCY"

    if mode == "oldest" and not topic_present:
        return {
            "semantic": 0.05,
            "recency": 0.70,  # Will be inverted (oldest = highest)
            "person": 0.15,
            "show": 0.10,
        }, "PURE_OLDEST"

    # ================================================================
    # PRIORITY 3: Guest-Focused (e.g., "JRE with Elon")
    # ================================================================
    if intent_guests:
        if mode in ["latest", "between", "relative_recent"]:
            # Guest + time constraint
            return {
                "semantic": 0.15,
                "recency": 0.25,
                "person": 0.45,
                "show": 0.15,
            }, "GUEST_TIME"
        else:
            # Guest only
            return {
                "semantic": 0.20,
                "recency": 0.15,
                "person": 0.50,
                "show": 0.15,
            }, "GUEST_FOCUSED"

    # ================================================================
    # PRIORITY 4: Topic + Recency (e.g., "latest JRE about AI")
    # ================================================================
    if mode in ["latest", "relative_recent"] and topic_present:
        return {
            "semantic": 0.40,
            "recency": 0.30,
            "person": 0.15,
            "show": 0.15,
        }, "TOPIC_RECENCY"

    # ================================================================
    # PRIORITY 5: Time Range (e.g., "JRE 2024 episodes")
    # ================================================================
    if mode in ["between", "before", "after"]:
        return {
            "semantic": 0.20,
            "recency": 0.35,
            "person": 0.25,
            "show": 0.20,
        }, "TIME_RANGE"

    # ================================================================
    # PRIORITY 6: Show-Focused (e.g., "Invest Like the Best episodes")
    # ================================================================
    if intent_show and not intent_guests:
        return {
            "semantic": 0.30,
            "recency": 0.25,
            "person": 0.15,
            "show": 0.30,
        }, "SHOW_FOCUSED"

    # ================================================================
    # DEFAULT: Standard balanced
    # ================================================================
    return {
        "semantic": 0.40,
        "recency": 0.25,
        "person": 0.20,
        "show": 0.15,
    }, "STANDARD"


def _compute_episode_person_match(
    episode_hosts: List[str],
    episode_guests: List[str],
    intent_hosts: List[str],
    intent_guests: List[str],
) -> Tuple[float, str]:
    """
    Compute person match score for an episode.

    Returns:
        (score, match_type) where score is 0.0-1.0
    """
    # Normalize all names for comparison
    def normalize(names):
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',') if n.strip()]
        return {n.lower().strip() for n in (names or [])}

    ep_hosts = normalize(episode_hosts)
    ep_guests = normalize(episode_guests)
    want_hosts = normalize(intent_hosts)
    want_guests = normalize(intent_guests)

    # Check for guest matches (highest priority)
    if want_guests:
        guest_overlap = want_guests & ep_guests
        if guest_overlap:
            return (1.0, "exact_guest")
        # Check if wanted guest appears as host (e.g., "Joe Rogan" could be guest on another show)
        guest_as_host = want_guests & ep_hosts
        if guest_as_host:
            return (0.7, "guest_as_host")

    # Check for host matches
    if want_hosts:
        host_overlap = want_hosts & ep_hosts
        if host_overlap:
            return (0.85, "exact_host")
        # Check if wanted host appears as guest
        host_as_guest = want_hosts & ep_guests
        if host_as_guest:
            return (0.6, "host_as_guest")

    # No specific person requested
    if not want_guests and not want_hosts:
        return (0.5, "no_filter")

    return (0.0, "no_match")


def _compute_episode_show_match(
    episode_podcast_title: str,
    intent_show_name: Optional[str],
    intent_hosts: List[str],
) -> Tuple[float, str]:
    """
    Compute show match score for an episode.
    """
    if not episode_podcast_title:
        return (0.3, "unknown_show")

    podcast_lower = episode_podcast_title.lower()

    # Direct show name match
    if intent_show_name:
        show_lower = intent_show_name.lower()
        if show_lower in podcast_lower or podcast_lower in show_lower:
            return (1.0, "exact_show")

    # Host name in podcast title (e.g., "joe rogan" in "The Joe Rogan Experience")
    for host in (intent_hosts or []):
        if host.lower() in podcast_lower:
            return (0.9, "host_in_title")

    return (0.3, "different_show")


def _compute_episode_number_boost(
    episode_title: str,
    target_episode_number: Optional[int],
    target_title_hint: Optional[str],
) -> Tuple[float, bool]:
    """
    Compute boost for specific episode number matches.

    Looks for patterns like "#2422" or "2422" in episode title.

    Returns: (boost_value, matched)
    """
    import re

    if not episode_title:
        return (0.0, False)

    title_lower = episode_title.lower()

    # Check episode number match
    if target_episode_number:
        patterns = [
            f"#{target_episode_number}",
            f"#{target_episode_number} ",
            f"#{target_episode_number}-",
            f" {target_episode_number} -",
            f" {target_episode_number}:",
            f"episode {target_episode_number}",
            f"ep {target_episode_number}",
            f"e{target_episode_number} ",
        ]
        for pattern in patterns:
            if pattern in title_lower:
                return (0.5, True)  # Massive boost

        # Looser check: number appears in title
        if str(target_episode_number) in title_lower:
            # Verify it's not part of a larger number
            if re.search(rf'\b{target_episode_number}\b', title_lower):
                return (0.4, True)

    # Check title hint match (e.g., "the Elon Musk episode")
    if target_title_hint:
        hint_lower = target_title_hint.lower()
        if hint_lower in title_lower:
            return (0.3, True)

    return (0.0, False)


def score_episodes_from_chunks(
    chunks: List[Dict[str, Any]],
    intent: Dict[str, Any],
    max_chunks_per_episode: int = 3,
) -> List[Dict[str, Any]]:
    """
    Group chunks by EpisodeId and score at episode level.

    UPDATED: Uses hybrid scoring with person/show match and episode number boost.

    Args:
        chunks: Pinecone results with hybrid_score
        intent: Extracted intent with time constraints
        max_chunks_per_episode: Cap chunks per episode (default 3)

    Returns:
        List of episode dicts sorted by episode_score
    """
    scoring_start = time.time()

    # Extract intent details
    time_filter = intent.get("time_filter", {})
    mode = (time_filter.get("mode") or "none").lower()
    topic_present = time_filter.get("topic_present", True)

    episode_identifier = intent.get("episode_identifier", {})
    target_episode_number = episode_identifier.get("episode_number")
    target_title_hint = episode_identifier.get("episode_title_hint")

    intent_hosts = intent.get("hosts", [])
    intent_guests = intent.get("guests", [])
    intent_show = intent.get("show_name")

    # Get weight profile
    weights, weight_profile = _determine_weight_profile(intent)

    logger.info(f"[EPISODE_SCORING] Input: {len(chunks)} chunks, profile={weight_profile}")
    logger.info(f"[EPISODE_SCORING] Intent: mode={mode}, topic={topic_present}, hosts={intent_hosts}, guests={intent_guests}")

    # ================================================================
    # PURE RECENCY: Pre-sort chunks by date BEFORE grouping
    # ================================================================
    if weight_profile in ["PURE_RECENCY", "PURE_OLDEST"]:
        logger.info("[EPISODE_SCORING] Pre-sorting chunks by date")
        reverse_sort = (weight_profile == "PURE_RECENCY")
        chunks = sorted(
            chunks,
            key=lambda c: (c.get("metadata") or {}).get("pdnumeric", 0) or c.get("pdnumeric", 0),
            reverse=reverse_sort
        )

    # ================================================================
    # Group chunks by episode (with per-episode cap)
    # ================================================================
    episodes: Dict[str, Dict] = defaultdict(lambda: {
        "chunks": [],
        "episode_id": None,
        "episode_title": None,
        "podcast_title": None,
        "published_date": None,
        "pdnumeric": 0,
        "guests": [],
        "hosts": [],
    })

    episode_chunk_counts: Dict[str, int] = defaultdict(int)

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        episode_id = meta.get("episodeId") or chunk.get("episodeId")

        if not episode_id:
            continue

        # Cap chunks per episode for performance
        if episode_chunk_counts[episode_id] >= max_chunks_per_episode:
            continue

        episode_chunk_counts[episode_id] += 1
        ep = episodes[episode_id]
        ep["episode_id"] = episode_id
        ep["chunks"].append(chunk)

        # Take metadata from first chunk
        if not ep["episode_title"]:
            ep["episode_title"] = meta.get("episodeTitle") or chunk.get("episode_title")
            ep["podcast_title"] = meta.get("channelTitle") or chunk.get("podcast_title")
            ep["published_date"] = meta.get("publishedDate") or chunk.get("published_date")
            ep["pdnumeric"] = meta.get("pdnumeric") or chunk.get("pdnumeric", 0)

            # Parse guests/hosts arrays
            raw_guests = meta.get("guests", []) or chunk.get("guests", [])
            raw_hosts = meta.get("hosts", []) or chunk.get("hosts", [])

            if isinstance(raw_guests, str):
                ep["guests"] = [g.strip() for g in raw_guests.split(',') if g.strip()]
            elif isinstance(raw_guests, list):
                ep["guests"] = raw_guests
            else:
                ep["guests"] = []

            if isinstance(raw_hosts, str):
                ep["hosts"] = [h.strip() for h in raw_hosts.split(',') if h.strip()]
            elif isinstance(raw_hosts, list):
                ep["hosts"] = raw_hosts
            else:
                ep["hosts"] = []

    logger.info(f"[EPISODE_SCORING] Grouped into {len(episodes)} episodes (max {max_chunks_per_episode} chunks/ep)")

    if not episodes:
        return []

    # ================================================================
    # Calculate date range for recency normalization
    # ================================================================
    all_dates = [ep["pdnumeric"] for ep in episodes.values() if ep["pdnumeric"]]
    max_date = max(all_dates) if all_dates else 0
    min_date = min(all_dates) if all_dates else 0
    date_range = max(max_date - min_date, 1)

    # ================================================================
    # Score each episode
    # ================================================================
    scored_episodes = []
    match_stats = {"exact_guest": 0, "exact_host": 0, "no_match": 0, "episode_number_match": 0}

    for episode_id, ep in episodes.items():
        chunk_scores = [c.get("hybrid_score", c.get("score", 0)) for c in ep["chunks"]]

        # Semantic score (best chunk)
        semantic_score = max(chunk_scores) if chunk_scores else 0

        # Recency score (0-1, newest = 1)
        pd = ep["pdnumeric"] or 0
        recency_score = (pd - min_date) / date_range if pd > 0 and date_range > 0 else 0.5

        # Invert for "oldest" mode
        if mode == "oldest":
            recency_score = 1.0 - recency_score

        # Person match score
        person_score, person_match_type = _compute_episode_person_match(
            ep["hosts"], ep["guests"], intent_hosts, intent_guests
        )

        # Show match score
        show_score, show_match_type = _compute_episode_show_match(
            ep["podcast_title"], intent_show, intent_hosts
        )

        # Episode number boost (additive, not weighted)
        episode_number_boost, ep_number_matched = _compute_episode_number_boost(
            ep["episode_title"], target_episode_number, target_title_hint
        )

        # Track stats
        if "exact" in person_match_type:
            match_stats["exact_guest" if "guest" in person_match_type else "exact_host"] += 1
        elif person_match_type == "no_match":
            match_stats["no_match"] += 1
        if ep_number_matched:
            match_stats["episode_number_match"] += 1

        # Compute weighted score
        base_score = (
            weights["semantic"] * semantic_score +
            weights["recency"] * recency_score +
            weights["person"] * person_score +
            weights["show"] * show_score
        )

        # Add episode number boost (not weighted, direct addition)
        episode_score = base_score + episode_number_boost

        # Store all scores for debugging
        ep["episode_score"] = episode_score
        ep["base_score"] = base_score
        ep["semantic_score"] = semantic_score
        ep["recency_score"] = recency_score
        ep["person_score"] = person_score
        ep["person_match_type"] = person_match_type
        ep["show_score"] = show_score
        ep["show_match_type"] = show_match_type
        ep["episode_number_boost"] = episode_number_boost
        ep["episode_number_matched"] = ep_number_matched
        ep["chunk_count"] = len(ep["chunks"])
        ep["weight_profile"] = weight_profile
        # Keep backwards compatible field
        ep["best_chunk_score"] = semantic_score

        # Remove chunks from output
        ep_copy = {k: v for k, v in ep.items() if k != "chunks"}
        scored_episodes.append(ep_copy)

    # Sort by episode score
    scored_episodes.sort(key=lambda x: x["episode_score"], reverse=True)

    scoring_time = time.time() - scoring_start
    logger.info(f"[EPISODE_SCORING] Scored {len(scored_episodes)} episodes in {scoring_time:.3f}s")
    logger.info(f"[EPISODE_SCORING] Profile: {weight_profile} | Weights: {weights}")
    logger.info(f"[EPISODE_SCORING] Match stats: {match_stats}")

    # Log top 5
    logger.info(f"[EPISODE_SCORING] Top 5 episodes:")
    for i, ep in enumerate(scored_episodes[:5]):
        logger.info(
            f"  [{i}] score={ep['episode_score']:.3f} "
            f"(sem={ep['semantic_score']:.2f} rec={ep['recency_score']:.2f} "
            f"person={ep['person_score']:.2f}/{ep['person_match_type']} "
            f"show={ep['show_score']:.2f} ep_boost={ep['episode_number_boost']:.2f}) | "
            f"{ep['episode_title'][:45]}..."
        )

    return scored_episodes[:EPISODE_MAX_CANDIDATES]


# ==============================================================================
# STEP 3.5: LIGHTWEIGHT EPISODE RERANKING (Optional)
# ==============================================================================

async def lightweight_episode_rerank(
    pinecone_client,
    query: str,
    episodes: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Lightweight reranking using episode titles and descriptions only.

    FAST: ~200-400ms (vs ~1-2s for full transcript reranking)

    Only used for topic+recency queries where semantic relevance matters.
    """
    if not episodes or len(episodes) <= 1:
        return episodes

    rerank_start = time.time()

    # Build documents from episode metadata (NOT transcripts)
    documents = []
    for ep in episodes[:20]:  # Cap at 20 for speed
        # Combine title + podcast + guests for reranking
        title = ep.get("episode_title", "")
        podcast = ep.get("podcast_title", "")
        guests = ep.get("guests", [])
        guests_str = ", ".join(guests) if isinstance(guests, list) else str(guests)

        doc_text = f"{title}. {podcast}. Guests: {guests_str}"
        # Phase 1: Use config for description limit (increased for better reranker context)
        desc_limit = getattr(config, 'EPISODE_DESCRIPTION_MAX_CHARS', 2000)
        documents.append({
            "id": ep.get("episode_id", ""),
            "text": doc_text[:desc_limit] if desc_limit else doc_text,
        })

    if not documents:
        return episodes

    try:
        # Use Pinecone's Cohere reranker
        rerank_response = pinecone_client.inference.rerank(
            model="cohere-rerank-english-v3.0",
            query=query,
            documents=[d["text"] for d in documents],
            top_n=min(top_n, len(documents)),
            return_documents=False,
        )

        # Build reranked list
        reranked_episodes = []
        ep_by_idx = {i: ep for i, ep in enumerate(episodes[:20])}

        for item in rerank_response.data:
            idx = item.index
            if idx in ep_by_idx:
                ep = ep_by_idx[idx].copy()
                ep["rerank_score"] = item.score
                # Boost episode_score with rerank signal
                ep["episode_score"] = (ep.get("episode_score", 0) * 0.6) + (item.score * 0.4)
                reranked_episodes.append(ep)

        # Add any episodes not in rerank results
        reranked_ids = {ep["episode_id"] for ep in reranked_episodes}
        for ep in episodes:
            if ep.get("episode_id") not in reranked_ids:
                reranked_episodes.append(ep)

        rerank_time = time.time() - rerank_start
        logger.info(f"[EPISODE_SEARCH] Lightweight rerank: {len(documents)} episodes in {rerank_time:.3f}s")

        # Re-sort by updated episode_score
        reranked_episodes.sort(key=lambda x: x.get("episode_score", 0), reverse=True)

        return reranked_episodes

    except Exception as e:
        logger.warning(f"[EPISODE_SEARCH] Lightweight rerank failed: {e}, using original order")
        return episodes


# ==============================================================================
# STEP 4: FETCH EPISODE DESCRIPTIONS FROM RDS
# ==============================================================================

def fetch_episode_descriptions(episode_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch episode descriptions from RDS Episodes table.

    This enriches the scored episodes with full descriptions
    that aren't stored in the Pinecone vector index.

    Args:
        episode_ids: List of episode IDs to fetch

    Returns:
        Dict mapping episode_id to {description, uri, images, title}
    """
    if not episode_ids:
        return {}

    fetch_start = time.time()
    logger.info(f"[EPISODE_SEARCH] Fetching descriptions for {len(episode_ids)} episodes from RDS...")

    try:
        connection = psycopg2.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            database=RDS_DATABASE,
            user=RDS_USERNAME,
            password=RDS_PASSWORD,
            cursor_factory=RealDictCursor
        )

        with connection.cursor() as cursor:
            # Query Episodes table for descriptions
            # Note: Table name might be "Episodes" or need to query Shorts grouped by episodeId
            placeholders = ','.join(['%s'] * len(episode_ids))

            # Query Episodes table for full episode metadata
            query = f"""
                SELECT
                    "episodeId",
                    "episodeTitle",
                    "episodeDescription",
                    "episodeUri",
                    "episodeImages"
                FROM "Episodes"
                WHERE "episodeId" IN ({placeholders})
            """

            cursor.execute(query, episode_ids)
            results = cursor.fetchall()

            episode_map = {}
            for row in results:
                ep_id = row.get("episodeId")
                if ep_id and ep_id not in episode_map:
                    # Parse images array
                    images = row.get("episodeImages") or []
                    if isinstance(images, str):
                        images = [images]

                    episode_map[ep_id] = {
                        "title": row.get("episodeTitle"),
                        "description": row.get("episodeDescription"),
                        "uri": row.get("episodeUri"),
                        "images": images,
                        "published_date": None,  # Not in Episodes table, get from Shorts if needed
                    }

            fetch_time = time.time() - fetch_start
            logger.info(f"[EPISODE_SEARCH] Fetched {len(episode_map)} episode descriptions in {fetch_time:.3f}s")

            return episode_map

    except Exception as e:
        logger.error(f"[EPISODE_SEARCH] Failed to fetch episode descriptions: {e}")
        return {}
    finally:
        if 'connection' in locals() and connection:
            connection.close()


# ==============================================================================
# STEP 5: EPISODE SELECTION LLM
# ==============================================================================


def validate_results_match_filter(
    episodes: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Validate if any results actually match the requested filter.

    This detects when relaxed recall returned results that don't match
    the original query (e.g., user asked for "Naval on Lex Fridman" but
    we only found other Lex Fridman episodes without Naval).

    Args:
        episodes: Scored episode candidates
        intent: Extracted intent with guests/hosts

    Returns:
        Tuple of (has_exact_match, mismatch_description)
        - has_exact_match: True if at least one episode matches the filter
        - mismatch_description: Human-readable description of what's missing
    """
    requested_guests = [g.lower() for g in intent.get("guests", [])]
    requested_hosts = [h.lower() for h in intent.get("hosts", [])]
    requested_show = (intent.get("show_name") or "").lower()

    if not requested_guests and not requested_hosts:
        # No specific person filter - any result is valid
        return True, ""

    # Check if any episode matches the requested guest/host
    for ep in episodes[:10]:  # Check top 10
        ep_guests = [g.lower() for g in (ep.get("guests") or [])]
        ep_hosts = [h.lower() for h in (ep.get("hosts") or [])]
        ep_show = (ep.get("podcast_title") or "").lower()

        # Check guest match (fuzzy - name contained)
        guest_match = not requested_guests  # True if no guests requested
        for req_guest in requested_guests:
            for ep_guest in ep_guests:
                if req_guest in ep_guest or ep_guest in req_guest:
                    guest_match = True
                    break
            if guest_match:
                break

        # Check host match (fuzzy - name contained)
        host_match = not requested_hosts  # True if no hosts requested
        for req_host in requested_hosts:
            for ep_host in ep_hosts:
                if req_host in ep_host or ep_host in req_host:
                    host_match = True
                    break
            # Also check if host is in guests (sometimes hosts appear as guests)
            for ep_guest in ep_guests:
                if req_host in ep_guest or ep_guest in req_host:
                    host_match = True
                    break
            if host_match:
                break

        if guest_match and host_match:
            return True, ""

    # No match found - build description
    mismatch_parts = []
    if requested_guests:
        mismatch_parts.append(f"{', '.join(intent.get('guests', []))}")
    if requested_show:
        mismatch_parts.append(f"on {intent.get('show_name', '')}")
    elif requested_hosts:
        mismatch_parts.append(f"with {', '.join(intent.get('hosts', []))}")

    mismatch_description = " ".join(mismatch_parts)
    logger.warning(f"[EPISODE_SEARCH] No exact match found for: {mismatch_description}")

    return False, mismatch_description


async def select_episode(
    gemini_client,
    query: str,
    intent: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    episode_descriptions: Dict[str, Dict],
    memory: ConversationMemory,
    is_fallback: bool = False,
    fallback_description: str = "",
) -> Tuple[Optional[Dict[str, Any]], BranchMemoryUpdate]:
    """
    LLM selects the best episode and generates memory update.

    This is the final step: given scored candidates, the LLM
    picks the winner and generates a natural response.

    Args:
        gemini_client: Gemini client
        query: Original user query
        intent: Extracted intent
        episodes: Scored episode candidates
        episode_descriptions: Descriptions from RDS
        memory: Conversation memory
        is_fallback: True if no exact match found (showing related content)
        fallback_description: What we couldn't find (e.g., "Naval on Lex Fridman")

    Returns:
        Tuple of (selected_episode dict, BranchMemoryUpdate)
    """
    selection_start = time.time()

    logger.info(f"[EPISODE_SEARCH] Selecting from {len(episodes)} episode candidates...")
    if is_fallback:
        logger.info(f"[EPISODE_SEARCH] FALLBACK MODE: Couldn't find exact match for '{fallback_description}'")

    if not episodes:
        return None, BranchMemoryUpdate(
            turn_summary="No episodes found matching criteria",
            action_type="error",
            entities_mentioned=[],
            topics_discussed=[],
            is_topic_shift=False,
            suggested_phase="discovery",
        )

    # Format episodes for prompt - Phase 1: increased from 10 to config limit
    episode_limit = getattr(config, 'EPISODE_SELECTION_LIMIT', 15)
    desc_limit = getattr(config, 'EPISODE_DESCRIPTION_MAX_CHARS', 2000)

    episodes_text = []
    for i, ep in enumerate(episodes[:episode_limit]):
        desc_data = episode_descriptions.get(ep.get("episode_id"), {})
        description = desc_data.get("description", "")
        # Phase 1: Use full description (no truncation) for better LLM context
        # Description limit is now configurable via EPISODE_DESCRIPTION_MAX_CHARS

        # Format guests/hosts
        guests = ep.get("guests", [])
        guests_str = ", ".join(guests) if isinstance(guests, list) and guests else "N/A"
        hosts = ep.get("hosts", [])
        hosts_str = ", ".join(hosts) if isinstance(hosts, list) and hosts else "N/A"

        # Format published date
        pub_date = ep.get("published_date") or desc_data.get("published_date") or "Unknown"

        episodes_text.append(f"""[{i}] Episode: {ep.get('episode_title', 'Unknown')}
    Podcast: {ep.get('podcast_title', 'Unknown')}
    Published: {pub_date}
    Hosts: {hosts_str}
    Guests: {guests_str}
    Relevance Score: {ep.get('episode_score', 0):.3f}
    Description: {description or 'No description available'}
""")

    episodes_formatted = "\n".join(episodes_text)

    # Build selection prompt - different for fallback vs exact match
    if is_fallback:
        system_prompt = f"""<task>
You are Echo, a podcast discovery AI. The user asked for something we couldn't find an exact match for.
We couldn't find: {fallback_description}
But we have related episodes to suggest instead.

Select the best related episode and generate a TRANSPARENT response that:
1. FIRST acknowledges we couldn't find what they asked for
2. THEN suggests the related episode as an alternative
</task>

<quote_extraction_protocol>
CRITICAL: Before selecting an episode, you MUST first extract relevant quotes.
This dramatically improves selection accuracy.

Step 1: Read each episode description carefully
Step 2: Extract 2-4 quotes that are relevant to what the user might want
Step 3: ONLY AFTER extracting quotes, select the best episode based on evidence
</quote_extraction_protocol>

<selection_rules>
1. Select the most relevant alternative episode
2. **CRITICAL - AVOID REPEATS**: Do NOT select the same episode again unless the user specifically asked for it. Check PREVIOUSLY SHOWN EPISODES below and pick a DIFFERENT episode.
</selection_rules>

<output>
Return valid JSON:
{{
  "relevant_quotes": [
    "Quote from episode description showing relevance...",
    "Another quote supporting your selection..."
  ],
  "chosen_index": 0-9,
  "response_text": "Start with 'I couldn't find [what they asked for]...' then suggest the alternative. Be specific about the episode content.",
  "confidence": 0.0-1.0,
  "memory_update": {{
    "turn_summary": "Brief summary of what user asked for and what we suggested (max 500 chars)",
    "entities_mentioned": ["guest1", "host1", "show"],
    "topics_discussed": ["topic1", "topic2"],
    "is_topic_shift": true,
    "suggested_phase": "discovery",
    "key_quotes": ["Memorable quote from the episode"],
    "topics_covered": ["Specific topic discussed in the episode"],
    "notable_examples": ["Notable highlight from the episode"]
  }}
}}
</output>

<response_guidelines>
- ALWAYS start with "I couldn't find [X] on [Y]" or similar acknowledgment
- Then say "but here's a related episode you might enjoy..." or "however, I found..."
- Describe WHAT the suggested episode is about
- Keep to 2-4 sentences
- Example: "I couldn't find any episodes with Naval Ravikant on Lex Fridman's podcast. However, here's a recent episode where Lex speaks with Demis Hassabis about AI and the future of intelligence - you might find the AI discussion interesting."
</response_guidelines>"""
    else:
        system_prompt = """<task>
You are Echo, a podcast discovery AI. Select the best episode matching the user's query.
Generate a natural response that describes WHAT THE EPISODE IS ABOUT and WHY IT'S RELEVANT to the user.
Extract memory update fields including key quotes and topics for context.
</task>

<quote_extraction_protocol>
CRITICAL: Before selecting an episode, you MUST first extract relevant quotes.
This dramatically improves selection accuracy from 27% to 98%.

Step 1: Read each episode description carefully
Step 2: Extract 2-4 quotes from descriptions that directly relate to the user's query
Step 3: ONLY AFTER extracting quotes, select the best episode based on evidence
Step 4: Your selection must be justified by the quotes you extracted
</quote_extraction_protocol>

<selection_rules>
1. For "latest" queries, prioritize the most recent episode that matches
2. For specific guest queries, ensure the guest is actually in the episode
3. For specific show queries, prefer that show
4. If multiple episodes match equally, pick the most relevant/interesting one
5. Your chosen_index MUST correspond to an episode with strong supporting quotes
6. **CRITICAL - AVOID REPEATS**: Do NOT select the same episode again unless the user specifically asked for it. Check PREVIOUSLY SHOWN EPISODES below and pick a DIFFERENT episode.
</selection_rules>

<output>
Return valid JSON:
{
  "relevant_quotes": [
    "Exact quote from episode description showing relevance to query...",
    "Another quote supporting why this episode matches..."
  ],
  "chosen_index": 0-9,
  "response_text": "Natural response describing the episode content (2-4 sentences). Focus on WHAT the episode covers and WHY it matches their query.",
  "confidence": 0.0-1.0,
  "memory_update": {
    "turn_summary": "Summary of what user asked for and what we found (max 500 chars)",
    "entities_mentioned": ["guest1", "host1", "show"],
    "topics_discussed": ["topic1", "topic2"],
    "is_topic_shift": true,
    "suggested_phase": "discovery",
    "key_quotes": ["2-3 memorable quotes from the episode that could help resolve future queries"],
    "topics_covered": ["Specific topics/subtopics in the episode (max 5)"],
    "notable_examples": ["Notable guests, stories, or highlights (max 3)"]
  }
}
</output>

<response_guidelines>
- Start by describing WHAT THE EPISODE IS ABOUT - use the description to explain the episode's content and key topics
- Explain WHY this episode matches what the user is looking for
- Include guest/host names naturally where relevant
- Keep to 2-4 sentences
- DO NOT use generic phrases like "I found X which matches your search" - be specific about the content!
- Example: "In this episode, Joe Rogan and Jelly Roll discuss his incredible 300-pound weight loss journey, covering the mental and physical challenges, the importance of honesty and seeking help..."
</response_guidelines>"""

    # Build memory context and previously shown episodes list (no truncation - 1M context)
    memory_context = memory.render_for_prompt()

    # Get previously shown episode IDs from exclusion window
    excluded_ids = memory.get_excluded_ids()
    previously_shown = []
    for turn in memory.recent_turns[-10:]:  # Check last 10 turns
        if turn.artifact_id and turn.artifact_title:
            previously_shown.append(f"- {turn.artifact_title} (ID: {turn.artifact_id[:20]}...)")

    previously_shown_text = "\n".join(previously_shown) if previously_shown else "None"

    user_prompt = f"""User query: "{query}"

Intent Summary: {intent.get('intent_summary', query)}

<conversation_memory>
{memory_context}
</conversation_memory>

<previously_shown_episodes>
{previously_shown_text}
</previously_shown_episodes>

Episode Candidates:
{episodes_formatted}

Select the best episode and generate a natural response + memory update.
IMPORTANT: Avoid selecting any episode from the previously shown list above unless the user specifically asked for it."""

    # =======================================================================
    # STRUCTURED OUTPUT: Guarantees valid JSON matching schema
    # =======================================================================
    try:
        try:
            # Try structured output first
            logger.info(f"[EPISODE_SEARCH] Calling selection with structured output...")
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=EPISODE_SEARCH_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                reasoning_effort=EPISODE_SELECTION_REASONING_EFFORT,
                response_format=EpisodeSelectionOutput,
                operation_name="Episode Selection (Structured)"
            )

            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                logger.info("[EPISODE_SEARCH] Structured output parsed successfully!")
                result = parsed_result.model_dump()
            else:
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError:
            # Fallback to json_object mode
            logger.warning("[EPISODE_SEARCH] Structured output not available, using json_object")
            resp = await llm_call_with_retry(
                gemini_client.chat.completions.create,
                model=EPISODE_SEARCH_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                reasoning_effort=EPISODE_SELECTION_REASONING_EFFORT,
                response_format={"type": "json_object"},
                operation_name="Episode Selection (JSON Fallback)"
            )
            result = json.loads(resp.choices[0].message.content)

        # Get selected episode
        chosen_index = result.get("chosen_index", 0)
        if chosen_index < 0 or chosen_index >= len(episodes):
            logger.warning(f"[EPISODE_SEARCH] Invalid chosen_index {chosen_index}, using 0")
            chosen_index = 0

        selected_episode = episodes[chosen_index].copy()

        # Phase 2: Log extracted quotes for debugging
        relevant_quotes = result.get("relevant_quotes", [])
        if relevant_quotes:
            logger.info(f"[EPISODE_SEARCH] Quote Extraction Protocol - {len(relevant_quotes)} quotes extracted:")
            for i, quote in enumerate(relevant_quotes[:4]):
                logger.info(f"  [{i+1}] \"{quote[:100]}{'...' if len(quote) > 100 else ''}\"")

        # Build memory update
        mem_data = result.get("memory_update", {})

        # Ensure entities include guests and hosts from episode
        entities = mem_data.get("entities_mentioned", [])
        ep_guests = selected_episode.get("guests", [])
        ep_hosts = selected_episode.get("hosts", [])

        # Add guest/host entities
        if isinstance(ep_guests, list):
            entities.extend(ep_guests)
        if isinstance(ep_hosts, list):
            entities.extend(ep_hosts)

        # Dedupe and cap (increased to 10 for Phase 2)
        entities = list(dict.fromkeys(entities))[:10]

        # Phase 2: Enhanced memory update with key_quotes, topics_covered, notable_examples
        memory_update = BranchMemoryUpdate(
            turn_summary=mem_data.get("turn_summary", f"Found episode: {selected_episode.get('episode_title', 'Unknown')[:50]}")[:500],
            action_type="episode_shown",
            action_target_id=selected_episode.get("episode_id"),
            action_target_title=selected_episode.get("episode_title"),
            published_date=selected_episode.get("published_date"),  # From Pinecone chunk metadata
            entities_mentioned=entities,
            topics_discussed=mem_data.get("topics_discussed", [])[:5],
            is_topic_shift=mem_data.get("is_topic_shift", True),
            suggested_phase=mem_data.get("suggested_phase", "discovery"),
            # Option A: Enhanced context fields
            key_quotes=mem_data.get("key_quotes", [])[:3],
            topics_covered=mem_data.get("topics_covered", [])[:5],
            notable_examples=mem_data.get("notable_examples", [])[:3],
        )

        # Enrich selected episode with response and confidence
        # Fix: Use 'or' to handle empty string responses, not just missing keys
        selected_episode["response_text"] = result.get("response_text") or f"I found an episode about {selected_episode.get('episode_title', 'your topic')}"
        selected_episode["confidence"] = min(1.0, max(0.0, result.get("confidence", 0.8)))
        selected_episode["selected_index"] = chosen_index  # Track index for recommendations

        # Add description data from RDS
        desc_data = episode_descriptions.get(selected_episode.get("episode_id"), {})
        selected_episode["description"] = desc_data.get("description")
        selected_episode["uri"] = desc_data.get("uri")
        selected_episode["image"] = desc_data.get("images", [None])[0] if desc_data.get("images") else None

        selection_time = time.time() - selection_start

        logger.info("")
        logger.info("[EPISODE SEARCH] ✅ EPISODE SELECTION COMPLETE!")
        logger.info("=" * 70)

        # Quote extraction results
        if relevant_quotes:
            logger.info("")
            logger.info("[EPISODE SEARCH] 📝 QUOTE EXTRACTION (Phase 2):")
            logger.info(f"  ├─ Quotes extracted: {len(relevant_quotes)}")
            for i, q in enumerate(relevant_quotes[:2]):
                logger.info(f"  │   [{i+1}] \"{q[:80]}...\"")
            logger.info("  └─ (Quotes extracted BEFORE selection for accuracy)")

        # Selected episode
        logger.info("")
        logger.info("[EPISODE SEARCH] 🎧 SELECTED EPISODE:")
        logger.info(f"  ├─ Index: [{chosen_index}] out of {len(episodes)} candidates")
        logger.info(f"  ├─ Title: \"{selected_episode.get('episode_title', 'Unknown')[:60]}\"")
        logger.info(f"  ├─ Podcast: {selected_episode.get('podcast_title', 'Unknown')}")
        logger.info(f"  ├─ Published: {selected_episode.get('published_date', 'Unknown')[:10]}")
        confidence = selected_episode.get('confidence', 0)
        logger.info(f"  └─ Confidence: {confidence:.1%} {'✓ High' if confidence >= 0.7 else '⚠ Medium' if confidence >= 0.5 else '⚠ Low'}")

        # Response
        logger.info("")
        logger.info("[EPISODE SEARCH] 💬 RESPONSE TO USER:")
        logger.info(f"  └─ \"{selected_episode.get('response_text', '')[:120]}...\"")

        # Memory update
        logger.info("")
        logger.info("[EPISODE SEARCH] 🧠 MEMORY UPDATE:")
        logger.info(f"  ├─ Summary: \"{mem_data.get('turn_summary', '')[:60]}...\"")
        logger.info(f"  ├─ Entities: {entities[:5]}")
        key_quotes_mem = mem_data.get("key_quotes", [])
        topics_covered_mem = mem_data.get("topics_covered", [])
        notable_examples_mem = mem_data.get("notable_examples", [])
        if key_quotes_mem or topics_covered_mem or notable_examples_mem:
            logger.info("  │")
            logger.info("  │  [ENHANCED CONTEXT - Option A]")
            if key_quotes_mem:
                logger.info(f"  ├─ Key quotes: {len(key_quotes_mem)} stored")
            if topics_covered_mem:
                logger.info(f"  ├─ Topics covered: {topics_covered_mem[:3]}")
            if notable_examples_mem:
                logger.info(f"  └─ Notable: {notable_examples_mem[:2]}")
        else:
            logger.info("  └─ Enhanced context: (none extracted)")

        logger.info("")
        logger.info(f"[EPISODE SEARCH] ⏱️ Selection time: {selection_time:.3f}s")
        logger.info("=" * 70)

        return selected_episode, memory_update

    except json.JSONDecodeError as e:
        logger.error(f"[EPISODE_SEARCH] Selection JSON parse error: {e}")
        return _create_fallback_selection(
            episodes, episode_descriptions, intent,
            is_fallback=is_fallback, fallback_description=fallback_description
        )
    except Exception as e:
        logger.error(f"[EPISODE_SEARCH] Selection failed: {e}")
        return _create_fallback_selection(
            episodes, episode_descriptions, intent,
            is_fallback=is_fallback, fallback_description=fallback_description
        )


def _create_fallback_selection(
    episodes: List[Dict[str, Any]],
    episode_descriptions: Dict[str, Dict],
    intent: Dict[str, Any] = None,
    is_fallback: bool = False,
    fallback_description: str = "",
) -> Tuple[Optional[Dict[str, Any]], BranchMemoryUpdate]:
    """Create fallback selection when LLM fails.

    Args:
        episodes: Scored episode candidates
        episode_descriptions: Descriptions from RDS
        intent: Extracted intent
        is_fallback: True if no exact match found (showing related content)
        fallback_description: What we couldn't find (e.g., "Naval on Lex Fridman")
    """
    if not episodes:
        return None, BranchMemoryUpdate(
            turn_summary="Episode search error",
            action_type="error",
            entities_mentioned=[],
            topics_discussed=[],
            is_topic_shift=False,
            suggested_phase="discovery",
            # Option A: Enhanced fields (empty for error case)
            key_quotes=[],
            topics_covered=[],
            notable_examples=[],
        )

    # Determine selection strategy based on intent
    mode = "none"
    has_specific_episode = False

    if intent:
        mode = (intent.get("time_filter", {}).get("mode") or "none").lower()
        has_specific_episode = intent.get("episode_identifier", {}).get("has_specific_episode", False)

    # Select episode based on intent
    if has_specific_episode:
        # For specific episode queries, trust the scoring (episode_number_boost)
        selected = episodes[0].copy()
        logger.info(f"[EPISODE_SEARCH] Fallback: Using top-scored episode (specific episode query)")
    elif mode == "latest":
        # Sort by actual date for latest queries
        episodes_sorted = sorted(episodes, key=lambda x: x.get("pdnumeric", 0), reverse=True)
        selected = episodes_sorted[0].copy()
        logger.info(f"[EPISODE_SEARCH] Fallback: Selected newest by date (pdnumeric={selected.get('pdnumeric')})")
    elif mode == "oldest":
        episodes_sorted = sorted(episodes, key=lambda x: x.get("pdnumeric", 0), reverse=False)
        selected = episodes_sorted[0].copy()
        logger.info(f"[EPISODE_SEARCH] Fallback: Selected oldest by date")
    else:
        # Use scoring order
        selected = episodes[0].copy()
        logger.info(f"[EPISODE_SEARCH] Fallback: Using top-scored episode")

    desc_data = episode_descriptions.get(selected.get("episode_id"), {})
    description = desc_data.get("description", "")

    # Generate a response that describes the episode content
    episode_title = selected.get('episode_title', 'an episode')
    podcast_title = selected.get('podcast_title', 'this podcast')

    # Phase 1: Use config for description limits in responses
    response_desc_limit = getattr(config, 'EPISODE_DESCRIPTION_MAX_CHARS', 2000)

    if is_fallback and fallback_description:
        # FALLBACK MODE: Be transparent that we didn't find exact match
        logger.info(f"[EPISODE_SEARCH] Fallback response: Couldn't find '{fallback_description}'")

        # Build a fallback response that acknowledges the miss
        fallback_intro = f"I couldn't find any episodes with {fallback_description}."

        if description:
            # Use description with smart truncation for response
            preview_limit = min(600, response_desc_limit)  # Cap at 600 for response readability
            desc_preview = description[:preview_limit].rstrip()
            if len(description) > preview_limit:
                last_period = desc_preview.rfind('.')
                if last_period > 150:
                    desc_preview = desc_preview[:last_period + 1]
                else:
                    last_space = desc_preview.rfind(' ')
                    if last_space > 200:
                        desc_preview = desc_preview[:last_space] + "..."
                    else:
                        desc_preview += "..."

            selected["response_text"] = f"{fallback_intro} However, here's a recent episode from {podcast_title} you might enjoy: {desc_preview}"
        else:
            selected["response_text"] = f"{fallback_intro} However, here's \"{episode_title}\" from {podcast_title} which you might find interesting."
    elif description:
        # Normal case: Use description with smart truncation for response
        preview_limit = min(800, response_desc_limit)  # Cap at 800 for response readability
        desc_preview = description[:preview_limit].rstrip()
        # Add ellipsis only if truncated
        if len(description) > preview_limit:
            # Find last complete sentence to avoid mid-word truncation
            last_period = desc_preview.rfind('.')
            if last_period > 300:
                desc_preview = desc_preview[:last_period + 1]
            else:
                # No good sentence break, find last space
                last_space = desc_preview.rfind(' ')
                if last_space > 400:
                    desc_preview = desc_preview[:last_space] + "..."
                else:
                    desc_preview += "..."

        # Use the description directly since it usually starts with good context
        selected["response_text"] = desc_preview
    else:
        selected["response_text"] = f"I found \"{episode_title}\" which matches your search."

    selected["confidence"] = min(0.7, selected.get("episode_score", 0.5))
    selected["description"] = description
    selected["uri"] = desc_data.get("uri")
    selected["image"] = desc_data.get("images", [None])[0] if desc_data.get("images") else None

    # Build entities from episode metadata
    entities = []
    if selected.get("hosts"):
        entities.extend(selected["hosts"][:2] if isinstance(selected["hosts"], list) else [])
    if selected.get("guests"):
        entities.extend(selected["guests"][:2] if isinstance(selected["guests"], list) else [])

    return selected, BranchMemoryUpdate(
        turn_summary=f"Found episode: {selected.get('episode_title', 'Unknown')[:100]}",
        action_type="episode_shown",
        action_target_id=selected.get("episode_id"),
        action_target_title=selected.get("episode_title"),
        published_date=selected.get("published_date"),  # From Pinecone chunk metadata
        entities_mentioned=entities[:10],
        topics_discussed=[],
        is_topic_shift=True,
        suggested_phase="discovery",
        # Option A: Enhanced fields (basic extraction from metadata for fallback)
        key_quotes=[],  # No quotes available in fallback
        topics_covered=[],  # Could be populated from description if needed
        notable_examples=[f"{selected.get('podcast_title', '')}"] if selected.get('podcast_title') else [],
    )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

async def handle_episode_search(
    gemini_client,
    openai_client,
    pinecone_client,
    query: str,
    memory: ConversationMemory,
    unique_personalities: List[str],
    unique_authors: List[str],
    router_output: Optional[RouterOutput] = None,
) -> EpisodeSearchResponse:
    """
    Main entry point for episode search branch.

    Pipeline:
    1. Extract intent (host, guest, show, time)
    2. Generate HyDE embeddings
    3. Search Pinecone with metadata filters
    4. Group and score by episode
    5. Fetch episode descriptions from RDS
    6. LLM selects best episode
    7. Return response with memory update

    Args:
        gemini_client: Gemini client (OpenAI-compatible)
        openai_client: OpenAI client for embeddings
        pinecone_client: Pinecone client for vector search
        query: User's query
        memory: Conversation memory
        unique_personalities: Known guest names for fuzzy matching
        unique_authors: Known host names for fuzzy matching
        router_output: Router output with sub_intent

    Returns:
        EpisodeSearchResponse with episode details and memory_update
    """
    branch_start = time.time()

    logger.info("")
    logger.info("=" * 70)
    logger.info("[EPISODE SEARCH] 🎧 STARTING EPISODE SEARCH")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[EPISODE SEARCH] 📥 INPUT:")
    logger.info(f"  ├─ User query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
    logger.info(f"  ├─ Session: {memory.session_id}")
    logger.info(f"  ├─ Conversation turn: #{memory.turn_count + 1}")
    if router_output:
        logger.info(f"  └─ Router intent: {router_output.sub_intent or 'general_episode_search'}")
    else:
        logger.info("  └─ Router intent: (no router output)")

    try:
        # ================================================================
        # STEP 1: Extract Intent
        # ================================================================
        logger.info("")
        logger.info("[EPISODE SEARCH] 🔍 STEP 1: EXTRACTING USER INTENT")
        logger.info("  └─ Analyzing query to extract: hosts, guests, show, time filters...")

        intent = await extract_episode_intent(gemini_client, query, memory, router_output)

        resolved_query = intent.get("resolved_query", query)
        hosts = intent.get("hosts", [])
        guests = intent.get("guests", [])
        show_name = intent.get("show_name")
        time_filter = intent.get("time_filter", {})

        logger.info("")
        logger.info("[EPISODE SEARCH] 📋 INTENT EXTRACTION RESULT:")
        logger.info(f"  ├─ Resolved query: \"{resolved_query}...\"")
        logger.info(f"  ├─ Hosts/Creators: {hosts if hosts else '(any)'}")
        logger.info(f"  ├─ Guests: {guests if guests else '(any)'}")
        logger.info(f"  ├─ Show/Podcast: {show_name if show_name else '(any)'}")
        if time_filter:
            mode = time_filter.get('mode', 'none')
            if mode == 'latest':
                logger.info(f"  └─ Time filter: LATEST (most recent episode)")
            elif mode == 'between':
                logger.info(f"  └─ Time filter: Between {time_filter.get('start_date', '?')} and {time_filter.get('end_date', '?')}")
            else:
                logger.info(f"  └─ Time filter: {time_filter}")
        else:
            logger.info("  └─ Time filter: (none)")

        # ================================================================
        # STEP 1.5: POST-INTENT SHOW → HOST MAPPING
        # If show_name is extracted but hosts is empty, try to map show to host
        # ================================================================
        gazetteer = get_gazetteer()
        unique_shows = gazetteer.shows

        if show_name and not hosts:
            show_name_lower = show_name.lower().strip()
            logger.info(f"[EPISODE_SEARCH] Attempting show→host mapping for: {show_name}")

            # Check if show_name matches a known host name
            for author in unique_authors:
                author_lower = author.lower()
                # "joe rogan" in "joe rogan" or "joe rogan" in "the joe rogan experience"
                if show_name_lower in author_lower or author_lower in show_name_lower:
                    logger.info(f"[EPISODE_SEARCH] Mapped show_name '{show_name}' to host '{author}'")
                    hosts = [author]
                    intent["hosts"] = hosts
                    break

            # Also check aliases if no host found
            if not hosts and hasattr(gazetteer, 'aliases'):
                for alias, canonical in gazetteer.aliases.items():
                    if show_name_lower == alias.lower() or alias.lower() in show_name_lower:
                        # Check if canonical is a host
                        if canonical.lower() in [a.lower() for a in unique_authors]:
                            logger.info(f"[EPISODE_SEARCH] Mapped show_name '{show_name}' via alias '{alias}' to host '{canonical}'")
                            hosts = [canonical]
                            intent["hosts"] = hosts
                            break

        logger.info(f"[EPISODE_SEARCH] Final extraction - hosts: {hosts}, guests: {guests}, show: {show_name}")

        # ================================================================
        # STEP 2: Generate HyDE Embeddings
        # ================================================================
        hyde_docs = await generate_episode_hyde(gemini_client, intent)
        all_queries = [resolved_query] + hyde_docs

        # ================================================================
        # STEP 3: Generate Embeddings (Parallel)
        # ================================================================
        logger.info(f"[EPISODE_SEARCH] Generating embeddings for {len(all_queries)} queries...")

        valid_query_data, valid_sparse_data = await asyncio.gather(
            concurrent_embedding_generation(openai_client, all_queries),
            concurrent_sparse_embedding_generation(pinecone_client, all_queries),
        )

        logger.info(f"[EPISODE_SEARCH] Generated {len(valid_query_data)} dense, {len(valid_sparse_data)} sparse embeddings")

        # ================================================================
        # STEP 3.5: Determine Search Strategy Based on Intent
        # ================================================================
        topic_present = time_filter.get("topic_present", True)
        mode = (time_filter.get("mode") or "none").lower()

        logger.info(f"[EPISODE_SEARCH] Search strategy: mode={mode}, topic_present={topic_present}")

        # ================================================================
        # STEP 4: Build Filter and Search Pinecone
        # Using enhanced build_episode_filter with channelTitle support
        # ================================================================
        # For episode search, include date clause if time constraint exists
        # FIX: Also check mode - LLM sometimes sets mode=latest but forgets has_time_constraint
        include_date = time_filter.get("has_time_constraint", False) or mode in ("latest", "oldest", "between", "before", "after", "relative_recent")

        # Ensure has_time_constraint is set correctly for downstream use
        if include_date and not time_filter.get("has_time_constraint"):
            time_filter["has_time_constraint"] = True
            logger.info(f"[EPISODE_SEARCH] Auto-derived has_time_constraint=True from mode={mode}")

        # First try with strict filter (AND between all conditions)
        pinecone_filter, post_date_range = build_episode_filter(
            guests,              # extracted_guests_interviewees
            hosts,               # extracted_hosts_creators
            show_name,           # NEW: show name for channelTitle filter
            unique_personalities,
            unique_authors,
            unique_shows,        # NEW: for show name fuzzy matching
            time_filter=time_filter,
            include_date_clause=include_date,
            strict=True,         # Start with strict AND filter
        )

        logger.info(f"[EPISODE_SEARCH] Pinecone filter (strict): {pinecone_filter}")

        # Determine recall settings
        # For episode search with date constraints, be stricter
        date_gating = time_filter.get("date_gating", "soft")
        allow_relaxed = (date_gating != "hard")
        recall_ratio = 0.2 if allow_relaxed else 0.0

        # Execute search
        logger.info(f"[EPISODE_SEARCH] Searching Pinecone (K={EPISODE_PINECONE_K}, recall_ratio={recall_ratio})...")

        nested_results = await concurrent_pinecone_search(
            pinecone_client,
            valid_query_data,
            valid_sparse_data,
            pinecone_filter,
            EPISODE_PINECONE_K,
            EPISODE_TARGET_PER_QUERY,
            use_hybrid=True,
            post_date_range=post_date_range,
            recall_ratio=recall_ratio,
            allow_relaxed_recall=allow_relaxed,
        )

        # Combine results across queries
        combined = combine_pinecone_results(nested_results, top_k=100)
        logger.info(f"[EPISODE_SEARCH] Pinecone returned {len(combined)} chunks (strict filter)")

        # ================================================================
        # STEP 4.5: RETRY WITH RELAXED FILTER if strict returns too few results
        # ================================================================
        MIN_RESULTS_THRESHOLD = 5
        used_relaxed_filter = False

        if len(combined) < MIN_RESULTS_THRESHOLD and (guests or hosts or show_name):
            logger.warning(f"[EPISODE_SEARCH] Strict filter returned only {len(combined)} results - retrying with relaxed OR filter")

            # Rebuild filter with OR between entity conditions
            relaxed_filter, post_date_range = build_episode_filter(
                guests,
                hosts,
                show_name,
                unique_personalities,
                unique_authors,
                unique_shows,
                time_filter=time_filter,
                include_date_clause=include_date,
                strict=False,  # Use OR between entity filters
            )

            logger.info(f"[EPISODE_SEARCH] Pinecone filter (relaxed): {relaxed_filter}")

            # Execute relaxed search
            nested_results = await concurrent_pinecone_search(
                pinecone_client,
                valid_query_data,
                valid_sparse_data,
                relaxed_filter,
                EPISODE_PINECONE_K,
                EPISODE_TARGET_PER_QUERY,
                use_hybrid=True,
                post_date_range=post_date_range,
                recall_ratio=0.3,  # More permissive
                allow_relaxed_recall=True,
            )

            combined = combine_pinecone_results(nested_results, top_k=100)
            logger.info(f"[EPISODE_SEARCH] Pinecone returned {len(combined)} chunks (relaxed filter)")
            used_relaxed_filter = True

        # ================================================================
        # STEP 4.6: FINAL FALLBACK - Remove date filter if "latest" still returns 0
        # If user asked for "latest" but no episodes exist in the time window,
        # remove the date filter and just find the most recent available episodes.
        # ================================================================
        if len(combined) == 0 and mode == "latest" and (guests or hosts or show_name):
            logger.warning(f"[EPISODE_SEARCH] 'Latest' mode returned 0 results with date filter - removing date filter to find most recent available")

            # Rebuild filter WITHOUT date clause
            no_date_filter, _ = build_episode_filter(
                guests,
                hosts,
                show_name,
                unique_personalities,
                unique_authors,
                unique_shows,
                time_filter=time_filter,
                include_date_clause=False,  # NO date filter
                strict=False,  # Keep relaxed entity filters
            )

            logger.info(f"[EPISODE_SEARCH] Pinecone filter (no date): {no_date_filter}")

            # Execute search without date restriction
            nested_results = await concurrent_pinecone_search(
                pinecone_client,
                valid_query_data,
                valid_sparse_data,
                no_date_filter,
                EPISODE_PINECONE_K,
                EPISODE_TARGET_PER_QUERY,
                use_hybrid=True,
                post_date_range=None,  # No post-filter either
                recall_ratio=0.3,
                allow_relaxed_recall=True,
            )

            combined = combine_pinecone_results(nested_results, top_k=100)
            logger.info(f"[EPISODE_SEARCH] Pinecone returned {len(combined)} chunks (no date filter - will sort by recency)")

            # Mark that we removed the date filter (results will be sorted by date in scoring)
            used_relaxed_filter = True

        # ================================================================
        # STEP 5: Group and Score by Episode
        # ================================================================
        scored_episodes = score_episodes_from_chunks(
            combined,
            intent,
            max_chunks_per_episode=3,
        )

        # ================================================================
        # STEP 5.5: Optional Lightweight Reranking (controlled by RERANKER_ENABLED)
        # ================================================================
        if config.RERANKER_ENABLED and topic_present and mode in ["latest", "relative_recent"]:
            logger.info("[EPISODE_SEARCH] Applying lightweight episode reranking (topic+recency)")
            scored_episodes = await lightweight_episode_rerank(
                pinecone_client,
                resolved_query,
                scored_episodes,
                top_n=10,
            )
        elif not config.RERANKER_ENABLED:
            # Skip reranking - take top 25 directly
            scored_episodes = scored_episodes[:25]
            logger.info(f"[EPISODE_SEARCH] RERANKER DISABLED: Taking top {len(scored_episodes)} episodes directly")

        if not scored_episodes:
            logger.warning("[EPISODE_SEARCH] No episodes found after scoring")
            branch_time = time.time() - branch_start
            logger.info(f"[EPISODE_SEARCH] Completed with no results in {branch_time:.2f}s")
            logger.info(f"[EPISODE_SEARCH] ========== EPISODE SEARCH END ==========")

            return EpisodeSearchResponse(
                response_text="I couldn't find any episodes matching your criteria. Try broadening your search or asking about a different show/guest.",
                episode_id="",
                episode_title="",
                podcast_title="",
                confidence=0.0,
                memory_update=BranchMemoryUpdate(
                    turn_summary="No episodes found matching criteria",
                    action_type="error",
                    entities_mentioned=hosts + guests,
                    topics_discussed=[],
                    is_topic_shift=True,
                    suggested_phase="discovery",
                ),
            )

        # ================================================================
        # STEP 6: Fetch Episode Descriptions from RDS
        # ================================================================
        episode_ids = [ep.get("episode_id") for ep in scored_episodes if ep.get("episode_id")]
        episode_descriptions = fetch_episode_descriptions(episode_ids)

        # ================================================================
        # STEP 6.5: Validate Results Match Filter
        # ================================================================
        has_exact_match, mismatch_description = validate_results_match_filter(
            scored_episodes, intent
        )

        # If we used relaxed filter, log it
        if used_relaxed_filter:
            logger.info(f"[EPISODE_SEARCH] Used RELAXED filter - results may be related but not exact matches")

        if not has_exact_match:
            logger.info(f"[EPISODE_SEARCH] No exact match - showing related content. Missing: {mismatch_description}")

        # ================================================================
        # STEP 7: LLM Selection
        # ================================================================
        selected, memory_update = await select_episode(
            gemini_client, query, intent, scored_episodes, episode_descriptions, memory,
            is_fallback=not has_exact_match,
            fallback_description=mismatch_description,
        )

        if not selected:
            logger.error("[EPISODE_SEARCH] Selection returned None")
            branch_time = time.time() - branch_start
            logger.info(f"[EPISODE_SEARCH] Completed with error in {branch_time:.2f}s")
            logger.info(f"[EPISODE_SEARCH] ========== EPISODE SEARCH END ==========")

            return EpisodeSearchResponse(
                response_text="I encountered an issue while selecting the episode. Please try again.",
                episode_id="",
                episode_title="",
                podcast_title="",
                confidence=0.0,
                memory_update=memory_update,
            )

        # ================================================================
        # BUILD RESPONSE
        # ================================================================
        branch_time = time.time() - branch_start

        logger.info("")
        logger.info(f"[EPISODE_SEARCH] ========== EPISODE SEARCH COMPLETE ==========")
        logger.info(f"[EPISODE_SEARCH] Total time: {branch_time:.2f}s")
        logger.info(f"[EPISODE_SEARCH] Selected: {selected.get('episode_title', 'Unknown')[:60]}")
        logger.info(f"[EPISODE_SEARCH] Podcast: {selected.get('podcast_title', 'Unknown')}")
        logger.info(f"[EPISODE_SEARCH] Confidence: {selected.get('confidence', 0):.2f}")
        logger.info(f"[EPISODE_SEARCH] Response: {selected.get('response_text', '')[:80]}...")
        logger.info(f"[EPISODE_SEARCH] ========== EPISODE SEARCH END ==========")
        logger.info("")

        # Fix: Use 'or' to handle empty string responses, not just missing keys
        return EpisodeSearchResponse(
            response_text=selected.get("response_text") or f"Here's an episode about {selected.get('episode_title', 'your topic')}",
            episode_id=selected.get("episode_id", ""),
            episode_title=selected.get("episode_title", ""),
            podcast_title=selected.get("podcast_title", ""),
            published_date=selected.get("published_date"),
            episode_description=selected.get("description"),
            episode_uri=selected.get("uri"),
            episode_image=selected.get("image"),
            guests=selected.get("guests", []) if isinstance(selected.get("guests"), list) else [],
            hosts=selected.get("hosts", []) if isinstance(selected.get("hosts"), list) else [],
            confidence=selected.get("confidence", 0.8),
            memory_update=memory_update,
            # Recommendation support - pass candidates data for episode recommendations
            selected_index=selected.get("selected_index", 0),
            scored_episodes=scored_episodes[:15],  # Top 15 for recommendations
            episode_descriptions_data=episode_descriptions,
        )

    except Exception as e:
        logger.error(f"[EPISODE_SEARCH] Unhandled error: {type(e).__name__}: {e}", exc_info=True)
        branch_time = time.time() - branch_start
        logger.info(f"[EPISODE_SEARCH] Failed after {branch_time:.2f}s")
        logger.info(f"[EPISODE_SEARCH] ========== EPISODE SEARCH END (ERROR) ==========")

        return EpisodeSearchResponse(
            response_text="I encountered an error while searching for episodes. Please try again.",
            episode_id="",
            episode_title="",
            podcast_title="",
            confidence=0.0,
            memory_update=BranchMemoryUpdate(
                turn_summary=f"Episode search error: {str(e)[:50]}",
                action_type="error",
                entities_mentioned=[],
                topics_discussed=[],
                is_topic_shift=False,
                suggested_phase="discovery",
            ),
        )


def handle_episode_search_sync(
    gemini_client,
    openai_client,
    pinecone_client,
    query: str,
    memory: ConversationMemory,
    unique_personalities: List[str],
    unique_authors: List[str],
    router_output: Optional[RouterOutput] = None,
) -> EpisodeSearchResponse:
    """Synchronous version of handle_episode_search."""
    logger.debug("[EPISODE_SEARCH] Using synchronous wrapper")
    return asyncio.run(handle_episode_search(
        gemini_client, openai_client, pinecone_client,
        query, memory, unique_personalities, unique_authors, router_output
    ))


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_episode_search_description() -> str:
    """Get human-readable description of episode search branch."""
    return "Find specific episodes by metadata (host, guest, show, date)"


def log_episode_search_summary(response: EpisodeSearchResponse, query: str) -> None:
    """Log a concise summary of the episode search interaction."""
    logger.info(
        f"[EPISODE_SEARCH SUMMARY] '{query[:30]}...' -> "
        f"'{response.episode_title[:30]}...' (conf={response.confidence:.2f})"
    )
