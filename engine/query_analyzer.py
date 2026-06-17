
import asyncio
import time
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
import config
from engine.memory import ConversationMemory
from retrieval.gazetteer import get_gazetteer
from retrieval.llm_utils import llm_call_with_retry

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class TimeFilterOutput(BaseModel):
    """Time filter constraints extracted from query."""
    mode: str = Field(
        default="none",
        description="Time mode: none, latest, oldest, between, relative_recent"
    )
    start_date_utc: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format"
    )
    end_date_utc: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format"
    )
    sort_preference: Optional[str] = Field(
        default=None,
        description="Sort preference: latest or oldest"
    )
    recency_priority: str = Field(
        default="none",
        description="Recency priority: none, soft, hard"
    )


class QueryMetadataOutput(BaseModel):
    """Structured output schema for query metadata extraction."""
    reasoning: str = Field(
        default="",
        description="Chain of thought reasoning for extraction decisions"
    )
    resolved_query: str = Field(
        description="The fully resolved query with pronouns replaced and context added"
    )
    extracted_guests_interviewees: List[str] = Field(
        default_factory=list,
        description="Guest or interviewee names mentioned in the query"
    )
    extracted_hosts_creators: List[str] = Field(
        default_factory=list,
        description="Host or creator names mentioned in the query"
    )
    extracted_show: Optional[str] = Field(
        default=None,
        description="Show or podcast name if mentioned (lowercase)"
    )
    is_followup: bool = Field(
        default=False,
        description="Whether this is a follow-up query referencing previous context"
    )
    topic_present: bool = Field(
        default=False,
        description="Whether a semantic topic is present beyond entity names"
    )
    time_filter: TimeFilterOutput = Field(
        default_factory=TimeFilterOutput,
        description="Time filter constraints extracted from query"
    )

# =============================================================================
# JSON REPAIR FUNCTIONS (for fallback mode)
# =============================================================================

def _repair_json(content: str) -> str:
    """Repair common JSON formatting issues from LLM output."""
    if not content:
        return content

    # Strip markdown code blocks
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Remove trailing commas before closing brackets
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Add missing commas between string values
    content = re.sub(r'"\s*\n\s*"', '",\n"', content)

    # Add missing commas between values and new keys
    content = re.sub(r'(\d+|true|false|null|"[^"]*"|]|})\s*\n\s*"([^"]+)":', r'\1,\n"\2":', content)

    # Fix unescaped quotes inside string values
    content = _fix_unescaped_quotes_in_strings(content)

    return content


def _fix_unescaped_quotes_in_strings(content: str) -> str:
    """Fix unescaped quotes inside JSON string values."""
    result = []
    i = 0
    in_string = False

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
                result.append(char)
            else:
                # Check if this quote terminates the string
                rest = content[i + 1:i + 20].lstrip()
                is_terminator = False

                if rest:
                    first_char = rest[0] if rest else ''
                    # Direct terminators
                    if first_char in ',}]\n:':
                        is_terminator = True
                    # Check for newline followed by field pattern
                    elif first_char == '\n' or (i + 1 < len(content) and content[i + 1] in ' \t\n'):
                        rest_stripped = content[i + 1:].lstrip()
                        if rest_stripped.startswith('"') or rest_stripped.startswith('}') or rest_stripped.startswith(']'):
                            is_terminator = True

                if is_terminator:
                    # End of string
                    in_string = False
                    result.append(char)
                else:
                    # Unescaped quote inside string - escape it
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
    Fallback to json_object mode when structured output is unavailable.
    Includes JSON repair logic for malformed responses.
    """
    resp = await llm_call_with_retry(
        gemini_client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        reasoning_effort=reasoning_effort,
        max_tokens=775,
        response_format={"type": "json_object"},
        operation_name="Query Metadata Extraction (JSON Fallback)"
    )

    raw_content = resp.choices[0].message.content
    finish_reason = getattr(resp.choices[0], 'finish_reason', 'unknown')
    logger.debug(f"[META] Fallback raw response (first 500 chars): {raw_content[:500] if raw_content else 'None'}")
    logger.debug(f"[META] Fallback finish reason: {finish_reason}")

    if not raw_content:
        raise ValueError("Empty LLM response in fallback mode")

    # Try to parse JSON, with repair attempt on failure
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as json_err:
        logger.warning(f"[META] Fallback JSON parse failed: {json_err}")
        logger.warning(f"[META] Problematic content around error (char {json_err.pos}): ...{raw_content[max(0, json_err.pos-50):json_err.pos+50]}...")

        # Attempt to repair common JSON issues
        repaired_content = _repair_json(raw_content)
        if repaired_content != raw_content:
            logger.info("[META] Attempting JSON repair...")
            result = json.loads(repaired_content)
            logger.info("[META] JSON repair successful!")
            return result
        else:
            logger.error(f"[META] Could not repair JSON. Full raw content:\n{raw_content}")
            raise


# Models & reasoning - load from config (centralized in config.py for easy switching)
GEMINI_MODEL = config.QUERY_ANALYZER_MODEL
GEMINI_MODEL_LITE = config.QUERY_ANALYZER_LITE_MODEL  # For HyDE generation

# Reasoning effort settings
QUERY_ANALYZER_REASONING_EFFORT = config.QUERY_ANALYZER_REASONING_EFFORT
HYDE_REASONING_EFFORT = config.QUERY_ANALYZER_HYDE_REASONING_EFFORT


# =============================================================================
# SYSTEM PROMPT BUILDER (Concise, Positive Patterns, Clear Context)
# =============================================================================

def _build_system_prompt(today_str: str, date_context: Dict[str, str], entity_context: str) -> str:
    return f"""You extract structured metadata from podcast search queries.

TODAY: {today_str}

ENTITY EXTRACTION PATTERNS:

Pattern 1 - "Host ON Topic":
  Host discussing a topic. The topic subject is NOT a guest.
  "Joe Rogan on Trump" → hosts=["Joe Rogan"], guests=[], topic_present=true

Pattern 2 - "Host WITH Guest":
  Guest appearance on host's show.
  "Joe Rogan with Elon" → hosts=["Joe Rogan"], guests=["Elon Musk"], topic_present=false

Pattern 3 - "Guest ON Host's Show":
  Guest appearing on a show (when second name has a known show).
  "Naval on Tim Ferriss" → guests=["Naval Ravikant"], hosts=["Tim Ferriss"], show="the tim ferriss show"

Pattern 4 - "Show Name Only":
  Infer host from show name.
  "latest JRE" → hosts=["Joe Rogan"], show="the joe rogan experience"

Pattern 5 - "Guest + Show + Topic":
  All three present.
  "Elon on Lex podcast about AI" → guests=["Elon Musk"], hosts=["Lex Fridman"], show="lex fridman podcast", topic_present=true

PRONOUN RESOLUTION (use MEMORY context):

Singular: he/him/his → first male entity | she/her/hers → first female entity
Plural: they/them/their → all entities from memory (or first if ambiguous)
Topic References: "that/this/it/this stuff/that topic" → THREAD_TOPIC (root topic of conversation)
Possessive: "his podcast" / "her show" → entity becomes host

CRITICAL - THREAD TOPIC RESOLUTION:
- "this stuff", "that topic", "more about it" → resolve to CONVERSATION THREAD TOPIC (not current_topic)
- Thread topic is the ROOT topic that started the conversation thread
- Example: Thread started with "sleep", evolved through caffeine+sleep, alcohol+sleep
  → "who else talks about this stuff" = "who else talks about sleep" (thread root)
- Look for "THREAD_TOPIC" or "ROOT TOPIC" in the context - use that for these resolutions

CRITICAL - CORRECTION/DISAMBIGUATION PATTERN:
- "No, not him/her, the other one" → USER IS CORRECTING - exclude the primary entity, find OTHER person from conversation
- "Not that one, the other person" → Same - look at FULL CONVERSATION HISTORY for other candidates
- When user says "the other one" after mentioning two people (e.g., "Joe Rogan interviewing Elon Musk"):
  → If primary was Elon Musk, "the other one" = Joe Rogan
- ALWAYS check conversation history for multiple people mentioned together

MEMORY PERSISTENCE:

Topic pivot: "what about health?" → KEEP entity, CHANGE topic
Entity pivot: "what does Sam think about that?" → CHANGE entity, KEEP topic
Complete reset: "find me X about Y" (explicit new search) → IGNORE memory

TIME EXTRACTION (CRITICAL - READ CAREFULLY):

"latest/newest/recent" + NO topic → mode="latest", recency_priority="hard"
"latest/newest/recent" + HAS topic → mode="latest", recency_priority="soft"
"oldest/earliest/first" → mode="oldest"
Specific year: "from 2023" → mode="between", start/end dates
No time words → mode="none", recency_priority="none"

IMPORTANT - TIME FILTER RULES:
1. Time filters are QUERY-SPECIFIC, NOT conversation-wide
2. Each query is evaluated INDEPENDENTLY for time language
3. If the CURRENT query has NO time words ("latest", "recent", "newest", "oldest", "from 20XX"):
   → ALWAYS set mode="none", recency_priority="none" (even on follow-ups!)
4. Previous time filters do NOT carry over unless user explicitly references time again
5. Example: User asks "latest AGI thoughts" (recency applied), then "what does Demis think?"
   → Second query has NO time words → mode="none" (find Demis content from ANY time)

PRECOMPUTED DATES:
- 1 week ago: {date_context['one_week_ago']}
- 3 months ago: {date_context['three_months_ago']}
- 6 months ago: {date_context['six_months_ago']}
- 1 year ago: {date_context['one_year_ago']}

DATABASE ENTITIES (match partial names to full names):
{entity_context}

OUTPUT: Valid JSON with _reasoning FIRST, then other fields."""


# =============================================================================
# USER PROMPT BUILDER (Few-Shot Examples with Consistent Formatting)
# =============================================================================

def _build_user_prompt(
    query: str,
    memory: ConversationMemory,
    date_context: Dict[str, str]
) -> str:
    state = memory.search_state

    # Get rich context for the current task (includes recent turns, summaries, time filters)
    rich_context = state.render_for_query_analyzer(memory.recent_turns)

    # Also keep simple format for backward compatibility in reasoning
    current_entities = state.current_entities if state.current_entities else []
    current_topic = state.current_topic if state.current_topic else None
    thread_topic = state.conversation_thread_topic if state.conversation_thread_topic else None
    participants = state.conversation_participants if state.conversation_participants else []
    entities_json = json.dumps(current_entities)
    topic_json = json.dumps(current_topic) if current_topic else "null"
    thread_topic_json = json.dumps(thread_topic) if thread_topic else "null"
    participants_json = json.dumps(participants) if participants else "[]"

    six_months_ago = date_context['six_months_ago']
    three_months_ago = date_context['three_months_ago']

    return f"""=== FEW-SHOT EXAMPLES ===

MEMORY: entities=["Elon Musk"], topic="AI safety"
QUERY: "what else did he say?"
OUTPUT: {{"_reasoning": "he→Elon Musk (memory). 'else'→continuation of AI safety.", "resolved_query": "What else did Elon Musk say about AI safety?", "extracted_guests_interviewees": ["Elon Musk"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "Joe Rogan on Trump"
OUTPUT: {{"_reasoning": "'ON' pattern→Trump is TOPIC, not guest.", "resolved_query": "Joe Rogan discussing Trump", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Joe Rogan"], "extracted_show": null, "is_followup": false, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "Joe Rogan with Elon Musk"
OUTPUT: {{"_reasoning": "'WITH' pattern→Elon is GUEST appearance.", "resolved_query": "Joe Rogan episode with Elon Musk", "extracted_guests_interviewees": ["Elon Musk"], "extracted_hosts_creators": ["Joe Rogan"], "extracted_show": null, "is_followup": false, "topic_present": false, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "latest JRE"
OUTPUT: {{"_reasoning": "JRE→Joe Rogan Experience. 'latest'+no topic→hard recency.", "resolved_query": "latest Joe Rogan Experience episode", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Joe Rogan"], "extracted_show": "the joe rogan experience", "is_followup": false, "topic_present": false, "time_filter": {{"mode": "latest", "start_date_utc": "{six_months_ago}", "sort_preference": "latest", "recency_priority": "hard"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "Lex recent on consciousness"
OUTPUT: {{"_reasoning": "Lex→Lex Fridman. 'recent'+topic→soft recency.", "resolved_query": "Lex Fridman recent episodes on consciousness", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Lex Fridman"], "extracted_show": null, "is_followup": false, "topic_present": true, "time_filter": {{"mode": "relative_recent", "start_date_utc": "{three_months_ago}", "sort_preference": "latest", "recency_priority": "soft"}}}}

---
MEMORY: entities=["Andrew Huberman"], topic="sleep"
QUERY: "his latest episode"
OUTPUT: {{"_reasoning": "his→Andrew Huberman. 'latest'+no new topic→hard recency.", "resolved_query": "Andrew Huberman's latest episode", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Andrew Huberman"], "extracted_show": null, "is_followup": true, "topic_present": false, "time_filter": {{"mode": "latest", "start_date_utc": "{six_months_ago}", "sort_preference": "latest", "recency_priority": "hard"}}}}

---
MEMORY: entities=["Elon Musk", "Sam Altman"], topic="OpenAI"
QUERY: "what did they disagree about?"
OUTPUT: {{"_reasoning": "they→[Elon Musk, Sam Altman] (plural). Topic=OpenAI persists.", "resolved_query": "What did Elon Musk and Sam Altman disagree about regarding OpenAI?", "extracted_guests_interviewees": ["Elon Musk", "Sam Altman"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "Naval on Tim Ferriss"
OUTPUT: {{"_reasoning": "'ON'+known host→Guest appearance. Tim Ferriss has show.", "resolved_query": "Naval Ravikant on Tim Ferriss Show", "extracted_guests_interviewees": ["Naval Ravikant"], "extracted_hosts_creators": ["Tim Ferriss"], "extracted_show": "the tim ferriss show", "is_followup": false, "topic_present": false, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Naval Ravikant"], topic="wealth"
QUERY: "what about his views on happiness?"
OUTPUT: {{"_reasoning": "his→Naval. Topic PIVOT: wealth→happiness. Entity persists.", "resolved_query": "Naval Ravikant's views on happiness", "extracted_guests_interviewees": ["Naval Ravikant"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Elon Musk"], topic="AI"
QUERY: "what does Sam Altman think about that?"
OUTPUT: {{"_reasoning": "that→AI (topic). Entity PIVOT: Elon→Sam. Topic persists.", "resolved_query": "What does Sam Altman think about AI?", "extracted_guests_interviewees": ["Sam Altman"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Naval Ravikant"], topic="wealth"
QUERY: "find me Joe Rogan episodes about MMA"
OUTPUT: {{"_reasoning": "'find me'+explicit terms→RESET. New search, ignore memory.", "resolved_query": "Joe Rogan episodes about MMA", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Joe Rogan"], "extracted_show": null, "is_followup": false, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "Elon vs Zuckerberg on AI"
OUTPUT: {{"_reasoning": "'vs' pattern→comparison. Both are subjects. AI is topic.", "resolved_query": "Elon Musk versus Mark Zuckerberg discussing AI", "extracted_guests_interviewees": ["Elon Musk", "Mark Zuckerberg"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": false, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "Elon Musk on Lex Fridman podcast about AI"
OUTPUT: {{"_reasoning": "'on [Show]'→guest. 'about AI'→topic. All three: guest+show+topic.", "resolved_query": "Elon Musk on Lex Fridman podcast discussing AI", "extracted_guests_interviewees": ["Elon Musk"], "extracted_hosts_creators": ["Lex Fridman"], "extracted_show": "lex fridman podcast", "is_followup": false, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=[], topic=null
QUERY: "episodes from 2023 with Sam Altman"
OUTPUT: {{"_reasoning": "'from 2023'→date filter. Sam Altman is guest.", "resolved_query": "2023 episodes with Sam Altman", "extracted_guests_interviewees": ["Sam Altman"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": false, "topic_present": false, "time_filter": {{"mode": "between", "start_date_utc": "2023-01-01", "end_date_utc": "2023-12-31", "recency_priority": "none"}}}}

---
MEMORY: entities=["Sam Altman"], topic="AGI", PREVIOUS_TIME_FILTER: mode="latest" (from user's earlier query "latest thoughts on AGI")
QUERY: "what does Demis Hassabis think about that?"
OUTPUT: {{"_reasoning": "ENTITY PIVOT: Sam→Demis. Topic 'that'=AGI persists. CRITICAL: Query has NO time words ('latest', 'recent', etc.) so time_filter RESETS to mode='none'. Time filters are query-specific, not inherited!", "resolved_query": "What does Demis Hassabis think about AGI?", "extracted_guests_interviewees": ["Demis Hassabis"], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Andrew Huberman", "Matthew Walker"], topic="alcohol and sleep", THREAD_TOPIC="sleep optimization"
QUERY: "who else talks about this stuff?"
OUTPUT: {{"_reasoning": "'this stuff'→THREAD_TOPIC='sleep optimization' (not current topic). Who else = other experts.", "resolved_query": "Other experts discussing sleep optimization", "extracted_guests_interviewees": [], "extracted_hosts_creators": [], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Tim Ferriss"], topic="productivity hacks", THREAD_TOPIC="morning routines"
QUERY: "more about that topic"
OUTPUT: {{"_reasoning": "'that topic'→THREAD_TOPIC='morning routines'. Continue thread.", "resolved_query": "More about morning routines", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Tim Ferriss"], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Elon Musk"], topic="AI regulation", CONVERSATION_HISTORY: [T1: "Joe Rogan interviewing Elon Musk about AI", T2: "What does he think about regulation?" resolved to Elon Musk]
QUERY: "No, not him, the other one"
OUTPUT: {{"_reasoning": "CORRECTION PATTERN: 'not him'=exclude Elon Musk. 'the other one'=look at T1 which had TWO people: Joe Rogan AND Elon Musk. User wants Joe Rogan.", "resolved_query": "What does Joe Rogan think about AI regulation?", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Joe Rogan"], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

---
MEMORY: entities=["Naval Ravikant"], topic="wealth", CONVERSATION_HISTORY: [T1: "Naval and Tim Ferriss discussing wealth", T2: "his book recommendations"]
QUERY: "actually the other guy"
OUTPUT: {{"_reasoning": "CORRECTION: 'the other guy'=not Naval. T1 had Naval AND Tim Ferriss. User wants Tim Ferriss.", "resolved_query": "Tim Ferriss book recommendations about wealth", "extracted_guests_interviewees": [], "extracted_hosts_creators": ["Tim Ferriss"], "extracted_show": null, "is_followup": true, "topic_present": true, "time_filter": {{"mode": "none", "recency_priority": "none"}}}}

=== CURRENT TASK ===

SIMPLE MEMORY (for quick reference):
- entities: {entities_json}
- current_topic: {topic_json}
- thread_topic: {thread_topic_json}
- conversation_participants (PEOPLE): {participants_json}

FULL CONTEXT (use this for accurate resolution - includes thread topic and full history):
{rich_context}

QUERY: "{query}"

IMPORTANT INSTRUCTIONS:
1. Use FULL CONTEXT above to accurately resolve pronouns (he/she/they/him/her)
2. The "SUBJECT being discussed" hint tells you who the content was ABOUT
3. TIME FILTERS ARE QUERY-SPECIFIC: Only apply time filter if THIS query has time words!
   - "latest", "recent", "newest" → apply recency
   - "oldest", "earliest", "first" → apply oldest
   - "from 2023", "in 2024" → apply date range
   - NO time words in query → mode="none" (even if previous query had time filter!)
4. For "this stuff", "that topic", "more about it" → use ROOT TOPIC from conversation thread
5. ROOT TOPIC is the original topic that started the thread (e.g., "sleep" even if recent turns discussed caffeine/alcohol)
6. Review FULL CONVERSATION HISTORY to understand the thread topic, not just last turn
7. CORRECTION PATTERNS: "No, not him/her, the other one" means EXCLUDE the primary entity and find the OTHER person from earlier in conversation. Look at T1 for multiple people mentioned together!
8. When T1 mentioned TWO people (e.g., "Joe Rogan interviewing Elon Musk") and user says "the other one", resolve to the NON-primary person

OUTPUT:"""


# =============================================================================
# METADATA EXTRACTION FUNCTION (Optimized with _reasoning field)
# =============================================================================

async def extract_query_metadata(
    gemini_client,
    query: str,
    memory: ConversationMemory,
) -> Dict[str, Any]:
    start_time = time.time()

    # Prepare date context
    current_date = datetime.now()
    today_str = current_date.strftime('%Y-%m-%d')

    date_context = {
        "one_week_ago": (current_date - timedelta(days=7)).strftime('%Y-%m-%d'),
        "three_months_ago": (current_date - timedelta(days=90)).strftime('%Y-%m-%d'),
        "six_months_ago": (current_date - timedelta(days=180)).strftime('%Y-%m-%d'),
        "one_year_ago": (current_date - timedelta(days=365)).strftime('%Y-%m-%d'),
    }

    # Gazetteer lookup for entity matching
    gazetteer = get_gazetteer()
    relevant_candidates = gazetteer.search(query, top_k=10)

    # Categorize entities
    authors_lower = {a.lower() for a in gazetteer.authors}
    personalities_lower = {p.lower() for p in gazetteer.personalities}
    shows_lower = {s.lower() for s in gazetteer.shows}

    host_candidates = [c for c in relevant_candidates if c.lower() in authors_lower]
    guest_candidates = [c for c in relevant_candidates if c.lower() in personalities_lower]
    show_candidates = [c for c in relevant_candidates if c.lower() in shows_lower]

    # Build entity context for prompt
    entity_parts = []
    if host_candidates:
        entity_parts.append(f"Hosts: {', '.join(host_candidates[:6])}")
    if guest_candidates:
        entity_parts.append(f"Guests: {', '.join(guest_candidates[:6])}")
    if show_candidates:
        entity_parts.append(f"Shows: {', '.join(show_candidates[:4])}")
    entity_context = "\n".join(entity_parts) if entity_parts else "No database matches found."

    # Build prompts
    system_prompt = _build_system_prompt(today_str, date_context, entity_context)
    user_prompt = _build_user_prompt(query, memory, date_context)

    logger.info(f"[META] Extracting metadata for: {query[:80]}...")
    logger.debug(f"[META] Memory entities: {memory.search_state.current_entities}")
    logger.debug(f"[META] Memory topic: {memory.search_state.current_topic}")

    # =======================================================================
    # STRUCTURED OUTPUT: Use beta.chat.completions.parse() with Pydantic
    # This GUARANTEES valid JSON matching the schema - no parsing errors!
    # See: https://ai.google.dev/gemini-api/docs/structured-output
    # =======================================================================
    try:
        try:
            # Try structured output first (guarantees schema adherence)
            logger.info(f"[META] Calling Gemini LLM ({GEMINI_MODEL}) with structured output...")
            resp = await llm_call_with_retry(
                gemini_client.beta.chat.completions.parse,
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,  # Deterministic extraction
                reasoning_effort=QUERY_ANALYZER_REASONING_EFFORT,
                response_format=QueryMetadataOutput,  # Pydantic model
                operation_name="Query Metadata Extraction (Structured)"
            )

            if not resp.choices:
                logger.error("[META] Empty response from LLM")
                return _fallback_metadata(query, memory)

            # With structured output, parsing is automatic and guaranteed
            finish_reason = getattr(resp.choices[0], 'finish_reason', 'unknown')
            logger.debug(f"[META] Finish reason: {finish_reason}")

            # Get the pre-parsed result directly
            parsed_result = resp.choices[0].message.parsed
            if parsed_result:
                logger.info("[META] Structured output parsed successfully!")
                result = parsed_result.model_dump()
            else:
                # Fallback to content if parsed is None (shouldn't happen)
                logger.warning("[META] Parsed result is None, falling back to content parsing")
                raw_content = resp.choices[0].message.content
                if raw_content:
                    result = json.loads(raw_content)
                else:
                    raise ValueError("Both parsed and content are empty")

        except AttributeError as attr_err:
            # beta.chat.completions.parse not available - fall back to json_object mode
            logger.warning(f"[META] Structured output not available ({attr_err}), using json_object fallback")
            result = await _fallback_json_object_call(
                gemini_client, system_prompt, user_prompt, GEMINI_MODEL, QUERY_ANALYZER_REASONING_EFFORT
            )
            logger.info("[META] JSON object fallback completed")

        # Log reasoning for debugging
        reasoning_key = "reasoning" if "reasoning" in result else "_reasoning"
        if reasoning_key in result:
            logger.info(f"[META] Reasoning: {result[reasoning_key]}")

    except json.JSONDecodeError as e:
        logger.error(f"[META] JSON parse error: {e}")
        return _fallback_metadata(query, memory)
    except Exception as e:
        logger.error(f"[META] Extraction failed: {e}")
        return _fallback_metadata(query, memory)

    # Post-process into expected format
    final_result = _postprocess_result(result, query, memory)

    elapsed = time.time() - start_time
    logger.info(f"[META] Extraction complete in {elapsed:.2f}s")
    logger.info(f"[META] Resolved: {final_result.get('resolved_query', query)}")
    logger.info(f"[META] Guests: {final_result.get('extracted_guests_interviewees', [])}")
    logger.info(f"[META] Hosts: {final_result.get('extracted_hosts_creators', [])}")
    logger.info(f"[META] Time mode: {final_result.get('time_filter', {}).get('mode', 'none')}")

    return final_result


def _postprocess_result(result: Dict[str, Any], query: str, memory: ConversationMemory) -> Dict[str, Any]:
    """
    Transform LLM output into the expected format for downstream processing.

    IMPORTANT: Validates host/guest classification against gazetteer.
    The LLM sometimes incorrectly classifies guests as hosts (e.g., Demis Hassabis).
    We use the gazetteer's authoritative host list to correct this.
    """
    # Extract time_filter with defaults
    time_filter = result.get("time_filter", {})
    time_filter.setdefault("mode", "none")
    time_filter.setdefault("recency_priority", "none")
    time_filter.setdefault("start_date_utc", None)
    time_filter.setdefault("end_date_utc", None)
    time_filter.setdefault("sort_preference", None)
    time_filter["has_time_constraint"] = time_filter["mode"] != "none"
    time_filter["topic_present"] = result.get("topic_present", False)

    # =========================================================================
    # VALIDATE HOST/GUEST CLASSIFICATION AGAINST GAZETTEER
    # The gazetteer contains the authoritative list of hosts (unique_authors).
    # Any name the LLM put in hosts that's NOT in our host list is actually a guest.
    # =========================================================================
    extracted_hosts = result.get("extracted_hosts_creators", [])
    extracted_guests = result.get("extracted_guests_interviewees", [])

    # Get authoritative host list from gazetteer
    gazetteer = get_gazetteer()
    known_hosts_lower = {h.lower() for h in gazetteer.authors}

    # Validate hosts - move non-hosts to guests
    validated_hosts = []
    moved_to_guests = []

    for host in extracted_hosts:
        if host.lower() in known_hosts_lower:
            validated_hosts.append(host)
        else:
            # This person is NOT a known host - they're a guest
            moved_to_guests.append(host)
            logger.info(f"[META] Host correction: '{host}' is NOT a known host, moving to guests")

    # Merge moved hosts into guests (avoiding duplicates)
    final_guests = list(extracted_guests)  # Start with LLM's guests
    for guest in moved_to_guests:
        if guest.lower() not in {g.lower() for g in final_guests}:
            final_guests.append(guest)

    if moved_to_guests:
        logger.info(f"[META] Host validation: {len(moved_to_guests)} corrected | Hosts: {validated_hosts} | Guests: {final_guests}")

    return {
        "query_title": query[:50],
        "resolved_query": result.get("resolved_query", query),
        "query_complexity": "simple",
        "is_followup": result.get("is_followup", False),
        "extracted_guests_interviewees": final_guests,
        "extracted_hosts_creators": validated_hosts,
        "extracted_show": result.get("extracted_show"),
        "referenced_entities": memory.search_state.current_entities if result.get("is_followup") else [],
        "original_query": query,
        "time_filter": time_filter,
        "_reasoning": result.get("_reasoning", ""),
    }


# =============================================================================
# MAIN QUERY ANALYSIS FUNCTION (with HyDE generation)
# =============================================================================

async def analyze_query_with_memory(
    gemini_client,
    query: str,
    memory: ConversationMemory,
) -> Dict[str, Any]:
    """
    Memory-aware query analysis with parallel HyDE generation.

    Changes from original analyze_query_parallel:
    1. Memory context injected into metadata prompt
    2. Pronoun resolution using entity tracking
    3. Follow-up detection based on memory
    4. HyDE docs generated with conversation context

    Args:
        gemini_client: Gemini client (OpenAI-compatible)
        query: User's raw question
        memory: Current conversation memory

    Returns:
        Dict with resolved_query, hyde_documents, time_filter, etc.
    """
    section_start = time.time()

    # --- 1. Context Preparation ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("[QUERY ANALYZER] 🔍 ANALYZING QUERY FOR CLIP SEARCH")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[QUERY ANALYZER] 📥 INPUT:")
    logger.info(f"  └─ Query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
    logger.info("")
    logger.info("[QUERY ANALYZER] 🎯 PURPOSE:")
    logger.info("  ├─ Extract: people (hosts/guests), topics, time filters")
    logger.info("  ├─ Resolve pronouns: he/she/they/it → actual names")
    logger.info("  └─ Generate HyDE documents for semantic search")

    # Gazetteer lookup
    logger.info("")
    logger.info("[QUERY ANALYZER] 📚 STEP 1: GAZETTEER LOOKUP (known hosts/guests/shows)")
    gaz_start = time.time()
    gazetteer = get_gazetteer()
    relevant_candidates = gazetteer.search(query, top_k=15)
    gaz_time = time.time() - gaz_start
    logger.info(f"  ├─ Search time: {gaz_time:.3f}s")
    logger.info(f"  └─ Found candidates: {relevant_candidates[:5]}")

    # =========================================================================
    # CATEGORIZE GAZETTEER CANDIDATES BY TYPE (like episode_search.py)
    # =========================================================================
    authors_lower = {a.lower() for a in gazetteer.authors}
    personalities_lower = {p.lower() for p in gazetteer.personalities}
    shows_lower = {s.lower() for s in gazetteer.shows}

    host_candidates = [c for c in relevant_candidates if c.lower() in authors_lower]
    guest_candidates = [c for c in relevant_candidates if c.lower() in personalities_lower]
    show_candidates = [c for c in relevant_candidates if c.lower() in shows_lower]

    logger.info("")
    logger.info("[QUERY ANALYZER] 📊 CANDIDATE CATEGORIZATION:")
    logger.info(f"  ├─ Hosts (podcast creators): {host_candidates[:4] if host_candidates else '(none found)'}")
    logger.info(f"  ├─ Guests (interviewees): {guest_candidates[:4] if guest_candidates else '(none found)'}")
    logger.info(f"  └─ Shows (podcasts): {show_candidates[:3] if show_candidates else '(none found)'}")

    # --- Memory context for logging ---
    last_turn = memory.get_last_turn()
    recent_entities = memory.get_recent_entities(top_k=8)

    logger.info("")
    logger.info("[QUERY ANALYZER] 🧠 STEP 2: MEMORY CONTEXT (for pronoun resolution)")
    logger.info(f"  ├─ Conversation turn: #{memory.turn_count + 1}")
    logger.info(f"  ├─ Recent entities: {recent_entities[:5] if recent_entities else '(none)'}")
    logger.info(f"  ├─ Themes discussed: {memory.conversation_themes[:3] if memory.conversation_themes else '(none)'}")
    if last_turn:
        logger.info(f"  └─ Last question: \"{last_turn.user_question[:50]}...\"")
    else:
        logger.info("  └─ Last question: (first turn)")

    # Detect if this is likely a follow-up
    is_likely_followup = _detect_followup(query, memory)
    logger.info("")
    logger.info(f"[QUERY ANALYZER] 🔗 FOLLOW-UP DETECTION: {'Yes ✓ (continuing previous topic)' if is_likely_followup else 'No (new topic)'}")

    # Log SearchState context with enhanced fields
    state = memory.search_state
    if state.current_entities or state.current_topic:
        logger.info("")
        logger.info("[QUERY ANALYZER] 📍 CURRENT CONTEXT (for resolving 'he/she/it/that'):")
        logger.info(f"  ├─ Entities in focus: {state.current_entities[:4]}")
        logger.info(f"  ├─ Current topic: {state.current_topic or '(none)'}")
        logger.info(f"  └─ Thread topic: {state.conversation_thread_topic or '(none)'}")

    # ========================================================================
    # TASK B: HyDE Generators (Hypothetical Document Embedding)
    # ========================================================================
    hyde_context = ""
    if last_turn:
        hyde_context = f"Previous topic: {last_turn.answer_summary}"
    hyde_angles = [
        ("Direct_Expert", "Give a direct, authoritative answer with specific facts, examples, and concrete details."),
        ("Personal_Story", "Share a personal anecdote or first-hand experience related to this topic."),
        ("Deep_Analysis", "Analyze the underlying mechanisms, strategy, trade-offs, or implications in depth."),
    ]

    # ========================================================================
    # PERSON-SPECIFIC HYDE: Generate HyDE docs that simulate the target person speaking
    # ========================================================================
    extracted_guests = []
    extracted_hosts = []

    # Pre-extract guests/hosts from gazetteer matches for person-specific HyDE
    query_lower = query.lower()
    for candidate in relevant_candidates[:10]:
        candidate_lower = candidate.lower()
        if candidate_lower in query_lower:
            # Check if this looks like a guest (usually a full name)
            if ' ' in candidate:  # Full names are likely guests
                extracted_guests.append(candidate)
            else:
                extracted_hosts.append(candidate)


    person_hyde_angles = []
    if extracted_guests:
        primary_guest = extracted_guests[0]
        person_hyde_angles = [
            ("Guest_First_Person", f"Generate a hypothetical podcast transcript where {primary_guest} discusses their work and ideas in first person. Include specific details they might mention."),
            ("Guest_Interview", f"Generate a hypothetical interview transcript where {primary_guest} answers questions about their projects, insights, and experiences."),
            ("Guest_Story", f"Generate a hypothetical podcast segment where {primary_guest} shares a personal anecdote or behind-the-scenes story."),
        ]
        logger.info(f"[QUERY ANALYZER] Person-specific HyDE enabled for guest: {primary_guest}")


    if person_hyde_angles:
        # Use 2 person-specific + 1 general for person queries
        final_hyde_angles = person_hyde_angles[:2] + hyde_angles[:1]
    else:
        final_hyde_angles = hyde_angles[:3]

    async def fetch_single_hyde(angle_name: str, instruction: str):
        """Generate a hypothetical podcast transcript snippet for semantic matching."""
        # Use resolved query context for better HyDE
        resolved_topic = query
        if last_turn:
            resolved_topic = f"{query} (context: {last_turn.resolved_query[:50]})"


        sys_prompt = f"""<task>
Generate a SHORT hypothetical podcast transcript snippet (50-80 words) on the given topic.
The transcript should sound like a natural podcast conversation with specific details.

Style: {angle_name.replace('_', ' ')}
Approach: {instruction}
{f'Context: {hyde_context}' if hyde_context else ''}
</task>

<format>
- Write as a transcript excerpt (natural speech patterns)
- Start the content IMMEDIATELY (no "Well," or "So,")
- Include specific details, names, examples, or numbers when relevant
- Keep under 80 words
- Output ONLY the transcript text, no labels or prefixes
</format>"""

        usr_prompt = f"Topic: {resolved_topic}\n\n[BEGIN TRANSCRIPT]"

        try:

            resp = await llm_call_with_retry(
                gemini_client.chat.completions.create,
                model=GEMINI_MODEL_LITE,  # Use lite model without thinking overhead
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt}
                ],
                temperature=0.8,
                reasoning_effort=HYDE_REASONING_EFFORT,
                #max_tokens=300,  # Sufficient for 50-80 word transcript
                operation_name=f"HyDE Generation ({angle_name})"
            )

            # Detailed logging for debugging empty/blocked responses
            if not resp.choices:
                logger.warning(f"[HYDE] {angle_name}: No choices in response. Full response: {resp}")
                return None

            choice = resp.choices[0]
            finish_reason = getattr(choice, 'finish_reason', 'unknown')
            content = choice.message.content

            if content is None:
                # Log detailed info to understand WHY content is None
                logger.warning(
                    f"[HYDE] {angle_name}: Gemini returned None content | "
                    f"finish_reason={finish_reason} | "
                    f"model={getattr(resp, 'model', 'unknown')} | "
                    f"prompt_preview={instruction[:50]}..."
                )
                # Check if there's safety feedback or other metadata
                if hasattr(resp, 'prompt_feedback'):
                    logger.warning(f"[HYDE] {angle_name}: prompt_feedback={resp.prompt_feedback}")
                return None

            result = content.strip()
            if len(result) < 10:
                logger.warning(f"[HYDE] {angle_name}: Response too short ({len(result)} chars): '{result}'")
                return None

            logger.debug(f"[HYDE] {angle_name}: Generated {len(result)} chars (finish_reason={finish_reason})")
            return result

        except Exception as e:
            import traceback
            logger.warning(
                f"[HYDE] {angle_name} generation FAILED: {type(e).__name__}: {e} | "
                f"prompt_preview={instruction[:50]}..."
            )
            logger.debug(f"[HYDE] {angle_name} traceback: {traceback.format_exc()}")
            return None

    # --- 3. Parallel Execution ---
    logger.info(f"[QUERY ANALYZER] Launching {1 + len(final_hyde_angles)} parallel LLM calls (1 Meta + {len(final_hyde_angles)} HyDE)...")
    llm_start = time.time()

    # Use the new optimized metadata extraction
    meta_task = extract_query_metadata(gemini_client, query, memory)
    hyde_tasks = [fetch_single_hyde(name, instr) for name, instr in final_hyde_angles]

    results = await asyncio.gather(meta_task, *hyde_tasks)

    llm_time = time.time() - llm_start
    logger.info(f"[QUERY ANALYZER] Parallel LLM finished in {llm_time:.2f}s")

    # --- 4. Assembly ---
    meta_result = results[0]

    # Filter HyDE docs - keep those with actual content (> 10 chars)
    hyde_raw = results[1:]
    hyde_docs = [doc for doc in hyde_raw if doc and len(doc) > 10]

    # Log HyDE generation stats
    successful = len(hyde_docs)
    failed = len(hyde_raw) - successful
    if failed > 0:
        logger.warning(f"[QUERY ANALYZER] HyDE generation: {successful}/{len(hyde_raw)} succeeded, {failed} failed")
    else:
        logger.info(f"[QUERY ANALYZER] HyDE generation: {successful}/{len(hyde_raw)} docs generated successfully")

    if not hyde_docs:
        hyde_docs = [f"Podcast discussion about {query}."]
        logger.warning("[QUERY ANALYZER] All HyDE generations failed, using fallback")

    final_result = {
        **meta_result,
        "hyde_documents": hyde_docs,
        "original_query": query,
        "_timing_breakdown": {
            "total_time": time.time() - section_start,
            "llm_elapsed": llm_time,
            "gaz_time": gaz_time,
        }
    }

    # Ensure defaults
    final_result.setdefault("resolved_query", query)
    final_result.setdefault("is_followup", is_likely_followup)
    final_result.setdefault("extracted_guests_interviewees", [])
    final_result.setdefault("extracted_hosts_creators", [])
    final_result.setdefault("extracted_show", None)  # Show/podcast name for channelTitle filter
    final_result.setdefault("referenced_entities", [])
    final_result.setdefault("query_complexity", "simple")
    final_result.setdefault("time_filter", {"has_time_constraint": False, "mode": "none"})

    # Ensure time_filter has recency_priority and topic_present defaults
    tf = final_result.get("time_filter", {})

    # ========================================================================
    # IMPROVED TOPIC DETECTION: Infer topic_present from query content
    # ========================================================================
    if "topic_present" not in tf or tf.get("topic_present") is None:
        # Check if query has semantic content beyond entity names and time words
        tf["topic_present"] = _infer_topic_present(
            query,
            final_result.get("extracted_guests_interviewees", []),
            final_result.get("extracted_hosts_creators", []),
        )
        logger.info(f"[QUERY ANALYZER] Inferred topic_present={tf['topic_present']} from query analysis")

    if "recency_priority" not in tf:
        # Infer recency_priority from mode if not provided by LLM
        mode = tf.get("mode", "none").lower()
        topic_present = tf.get("topic_present", False)
        if mode in ("latest", "oldest"):
            tf["recency_priority"] = "soft" if topic_present else "hard"
        elif mode in ("relative_recent",):
            tf["recency_priority"] = "soft"
        else:
            tf["recency_priority"] = "none"
    final_result["time_filter"] = tf

    # ===========================================================================
    # Log SearchState-based Resolution
    # ===========================================================================
    resolved_query = final_result.get('resolved_query', query)
    if resolved_query and resolved_query != query:
        logger.info("")
        logger.info("[QUERY ANALYZER] 🔄 PRONOUN RESOLUTION APPLIED:")
        logger.info(f"  ├─ Original: \"{query}\"")
        logger.info(f"  ├─ Resolved: \"{resolved_query}\"")
        logger.info(f"  ├─ Entities from memory: {state.current_entities[:3]}")
        logger.info(f"  └─ Topic from memory: {state.current_topic or '(none)'}")

    # Log final analysis results
    logger.info("")
    logger.info("[QUERY ANALYZER] ✅ ANALYSIS COMPLETE!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[QUERY ANALYZER] 📋 EXTRACTION RESULTS:")
    logger.info(f"  ├─ Original query: \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
    logger.info(f"  ├─ Resolved query: \"{final_result.get('resolved_query', query)}{'...' if len(final_result.get('resolved_query', query)) > 60 else ''}\"")
    logger.info(f"  ├─ Is follow-up: {'Yes ✓' if final_result.get('is_followup', False) else 'No'}")

    # People extracted
    guests = final_result.get('extracted_guests_interviewees', [])
    hosts = final_result.get('extracted_hosts_creators', [])
    show = final_result.get('extracted_show')
    logger.info(f"  │")
    logger.info(f"  │  [PEOPLE EXTRACTED]")
    logger.info(f"  ├─ Hosts: {hosts if hosts else '(any)'}")
    logger.info(f"  ├─ Guests: {guests if guests else '(any)'}")
    logger.info(f"  ├─ Show: {show if show else '(any)'}")

    # Time filter
    logger.info(f"  │")
    logger.info(f"  │  [TIME FILTER]")
    time_mode = tf.get('mode', 'none')
    recency_priority = tf.get('recency_priority', 'none')
    topic_present = tf.get('topic_present', False)
    if time_mode == 'latest':
        logger.info(f"  ├─ Mode: LATEST (most recent)")
    elif time_mode == 'between':
        logger.info(f"  ├─ Mode: DATE RANGE ({tf.get('start_date', '?')} to {tf.get('end_date', '?')})")
    else:
        logger.info(f"  ├─ Mode: None (any time)")

    if recency_priority == 'hard':
        logger.info(f"  ├─ Recency: HARD - Newest wins (e.g., 'latest episode')")
    elif recency_priority == 'soft':
        logger.info(f"  ├─ Recency: SOFT - Balance topic + recency")
    else:
        logger.info(f"  ├─ Recency: Standard - Topic relevance first")

    logger.info(f"  └─ Has topic content: {'Yes' if topic_present else 'No (metadata-only query)'}")

    # Performance
    logger.info("")
    logger.info("[QUERY ANALYZER] 📊 PERFORMANCE:")
    logger.info(f"  ├─ HyDE documents generated: {len(hyde_docs)}")
    logger.info(f"  └─ Total analysis time: {final_result['_timing_breakdown']['total_time']:.2f}s")

    logger.info("")
    logger.info("=" * 70)

    return final_result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _infer_topic_present(query: str, guests: List[str], hosts: List[str]) -> bool:
    """
    Infer whether the query has a semantic topic beyond just entity names and time expressions.

    Returns True if the query contains topical content words (e.g., "AI", "Chernobyl", "politics").
    Returns False for pure entity/recency queries (e.g., "latest Joe Rogan episode").

    This is critical for the recency-first strategy to work correctly.
    """
    query_lower = query.lower()

    # Remove entity names from query
    query_cleaned = query_lower
    all_entities = (guests or []) + (hosts or [])
    for entity in all_entities:
        if entity:
            query_cleaned = query_cleaned.replace(entity.lower(), " ")

    # Remove common time/recency words
    time_words = [
        "latest", "newest", "recent", "most recent", "new", "current", "up to date",
        "oldest", "earliest", "first", "original",
        "last", "past", "this", "week", "month", "year", "today", "yesterday",
        "before", "after", "since", "until", "ago", "from", "to",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "2020", "2021", "2022", "2023", "2024", "2025",
    ]
    for word in time_words:
        query_cleaned = re.sub(rf'\b{re.escape(word)}\b', ' ', query_cleaned)

    # Remove common filler words
    filler_words = [
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
        "from", "about", "what", "when", "where", "who", "why", "how",
        "is", "are", "was", "were", "be", "been", "being",
        "show", "episode", "podcast", "clip", "video", "interview",
        "any", "some", "find", "get", "me", "tell", "give",
        "did", "does", "do", "can", "could", "would", "should",
        "his", "her", "their", "its", "my", "your", "our",
    ]
    for word in filler_words:
        query_cleaned = re.sub(rf'\b{re.escape(word)}\b', ' ', query_cleaned)

    # Clean up whitespace
    query_cleaned = ' '.join(query_cleaned.split())

    # If there are remaining meaningful words, there's a topic
    # Minimum 3 characters to filter out noise
    remaining_words = [w for w in query_cleaned.split() if len(w) >= 3]

    # Topic detection heuristics
    has_topic = len(remaining_words) >= 1

    logger.debug(f"[TOPIC DETECT] Query: {query}")
    logger.debug(f"[TOPIC DETECT] Cleaned: '{query_cleaned}'")
    logger.debug(f"[TOPIC DETECT] Remaining words: {remaining_words}")
    logger.debug(f"[TOPIC DETECT] Has topic: {has_topic}")

    return has_topic


def _detect_followup(query: str, memory: ConversationMemory) -> bool:
    """Heuristic follow-up detection."""
    if not memory.recent_turns:
        return False

    query_lower = query.lower()

    # Pronoun check
    pronouns = ["he", "she", "they", "it", "this", "that", "these", "those", "him", "her", "his", "their"]
    has_pronoun = any(f" {p} " in f" {query_lower} " or query_lower.startswith(f"{p} ") or query_lower.endswith(f" {p}") for p in pronouns)

    # Follow-up keywords
    followup_keywords = [
        "more", "else", "another", "also", "again", "continue",
        "expand", "elaborate", "detail", "explain", "tell me more",
        "what about", "how about", "same", "similar", "different",
        "other", "besides", "additionally", "further", "next"
    ]
    has_keyword = any(kw in query_lower for kw in followup_keywords)

    # Short query after context (likely a follow-up)
    is_short_query = len(query.split()) <= 5 and memory.turn_count > 0

    return has_pronoun or has_keyword or is_short_query


def _fallback_metadata(query: str, memory: ConversationMemory) -> Dict[str, Any]:
    """
    Fallback when LLM extraction fails.
    Uses basic heuristics and memory state.
    """
    resolved = query
    is_followup = False
    referenced = []

    # Simple pronoun detection
    query_lower = query.lower()
    pronouns = ["he ", "she ", "they ", "him ", "her ", "his ", "their "]

    if memory.search_state.current_entities:
        for pronoun in pronouns:
            if pronoun in query_lower or query_lower.startswith(pronoun.strip()):
                entity = memory.search_state.current_entities[0]
                resolved = f"{query} (referring to {entity})"
                referenced.append(entity)
                is_followup = True
                break

    return {
        "query_title": query[:50],
        "resolved_query": resolved,
        "query_complexity": "simple",
        "is_followup": is_followup,
        "extracted_guests_interviewees": [],
        "extracted_hosts_creators": [],
        "extracted_show": None,
        "referenced_entities": referenced,
        "original_query": query,
        "time_filter": {
            "has_time_constraint": False,
            "mode": "none",
            "recency_priority": "none",
            "topic_present": False,
            "start_date_utc": None,
            "end_date_utc": None,
            "sort_preference": None,
        },
        "_reasoning": "Fallback: LLM extraction failed",
    }
