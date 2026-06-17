"""
Design Principles:
1. Rolling window - keep last N turns to bound token usage
2. Semantic compression - summarize older turns into themes
3. Entity tracking - maintain referenced people/topics for pronoun resolution
4. Artifact history - track shown clips for context
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import threading
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Debug mode: Set MEMORY_DEBUG=1 to enable file dumps after each turn
MEMORY_DEBUG = os.environ.get("MEMORY_DEBUG", "0") == "1"
MEMORY_DEBUG_DIR = Path(os.environ.get("MEMORY_DEBUG_DIR", "debug_memory"))

# Constants
MAX_RECENT_TURNS = 50         # Full context for last 50 turns (was 5)
MAX_COMPRESSED_TURNS = 100    # Summarized older turns (was 10)
MAX_ENTITIES = 20             # Track top 20 referenced entities
MAX_MEMORY_CHARS = 50000      # ~12500 tokens max for memory context (was 8000)
ENTITY_DECAY_FACTOR = 0.8     # Older entity mentions decay in importance


@dataclass
class ConversationTurn:
    """Single turn in conversation history.

    Phase 2 Enhancement: Now includes key_quotes, topics_covered, and
    notable_examples for richer context in follow-up query resolution.
    """
    turn_id: str
    timestamp: datetime
    user_question: str
    resolved_query: str           # Query after pronoun resolution
    answer_summary: str           # Brief summary (max 500 chars) - increased from 200
    artifact_id: Optional[str]
    artifact_title: Optional[str]
    key_entities: List[str]       # People, topics mentioned (max 10)
    themes: List[str]             # High-level themes (max 5)

    # Option A: Enhanced context fields (Phase 2)
    key_quotes: List[str] = field(default_factory=list)      # 2-3 memorable quotes
    topics_covered: List[str] = field(default_factory=list)  # Specific topics (max 5)
    notable_examples: List[str] = field(default_factory=list)  # Notable highlights (max 3)


# ==============================================================================
# PHASE 0: UNIFIED MEMORY SCHEMA FOR AGENT ARCHITECTURE
# ==============================================================================

@dataclass
class RouteRecord:
    """Record of a single routing decision for debugging and context."""
    turn_id: str
    route_chosen: str  # "small_talk", "episode_search", "clip_search"
    route_confidence: float
    query_intent: str  # Brief description of interpreted intent
    outcome: str  # "success", "no_results", "clarification_needed", "error"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LastActionContext:
    """
    Context about the most recent action taken by any branch.
    This is CRITICAL for pronoun resolution and continuation queries.
    """
    action_type: Optional[str] = None  # "clip_shown", "episode_shown", "explanation", "greeting"
    target_id: Optional[str] = None    # Chunk ID, Episode ID, or None
    target_title: Optional[str] = None # Human-readable title
    target_summary: Optional[str] = None  # Brief description of what was shown/said
    source_branch: Optional[str] = None   # Which branch handled it
    published_date: Optional[str] = None  # Publication date of shown episode/clip (YYYY-MM-DD)

    def render_for_prompt(self) -> str:
        """Render last action for LLM context."""
        if not self.action_type:
            return "No previous action in this session."

        parts = [f"LAST ACTION: {self.action_type}"]
        if self.target_title:
            parts.append(f"  Target: {self.target_title}")
        if self.published_date:
            parts.append(f"  Published: {self.published_date}")
        if self.target_summary:
            parts.append(f"  Summary: {self.target_summary}")  # No truncation - 1M context window
        if self.source_branch:
            parts.append(f"  Handled by: {self.source_branch}")

        return "\n".join(parts)


# ==============================================================================
# PHASE 1: SEARCH STATE - Deterministic Conversation State Tracker
# ==============================================================================

@dataclass
class SearchState:
    """
    Deterministic state tracker - THE source of truth for conversation context.

    UPDATED FOR AGENT ARCHITECTURE:
    - Added last_action for cross-branch context
    - Added route_history for routing decisions
    - Added conversation_phase for intent understanding

    UPDATED FOR TOPIC DRIFT FIX:
    - Added conversation_thread_topic: ROOT topic that persists across related turns
    - Added topic_history: Evolution of topics in current thread
    - These prevent "this stuff" from resolving to wrong topic when search returns bad results
    """

    # === EXISTING FIELDS (Keep All) ===
    current_entities: List[str] = field(default_factory=list)
    current_topic: Optional[str] = None
    last_time_filter: Optional[Dict[str, Any]] = None
    last_artifact_id: Optional[str] = None
    last_artifact_title: Optional[str] = None
    turns_since_entity_update: int = 0
    last_was_followup: bool = False

    # === NEW: Last Action Context (CRITICAL FOR AGENT) ===
    last_action: LastActionContext = field(default_factory=LastActionContext)

    # === NEW: Route History (Last 5 turns) ===
    route_history: List[RouteRecord] = field(default_factory=list)
    MAX_ROUTE_HISTORY: int = 5

    # === NEW: Conversation Phase ===
    # WARNING: Do NOT rely on this for routing decisions yet!
    # LLMs are inconsistent at updating abstract state flags.
    # Use last_action (concrete facts) for routing, log conversation_phase for observation.
    # Once we have data showing LLMs consistently classify phase, we can use it.
    conversation_phase: str = "discovery"  # "discovery", "deep_dive", "comparison", "idle"

    # === NEW: CONVERSATION THREAD TRACKING (TOPIC DRIFT FIX) ===
    # conversation_thread_topic: The ROOT topic that started the current thread
    # This does NOT get overwritten by bad search results - it persists until explicit topic change
    # Example: "Huberman on sleep" starts thread, follow-ups about caffeine/alcohol keep "sleep" as thread topic
    conversation_thread_topic: Optional[str] = None

    # topic_history: Evolution of topics in current conversation thread
    # Helps LLMs understand "this stuff" = entire thread, not just last turn
    # Example: ["sleep optimization", "caffeine and sleep", "alcohol and sleep"]
    topic_history: List[str] = field(default_factory=list)
    MAX_TOPIC_HISTORY: int = 10

    # === NEW: CONVERSATION PARTICIPANTS (CORRECTION PATTERN FIX) ===
    # conversation_participants: PEOPLE mentioned together in the conversation
    # These persist across turns so "the other one" can resolve correctly
    # Example: "Joe Rogan interviewing Elon Musk" → ["Joe Rogan", "Elon Musk"] persists
    # This is SEPARATE from current_entities which may be overwritten with topics
    conversation_participants: List[str] = field(default_factory=list)
    MAX_PARTICIPANTS: int = 10

    # === EXISTING METHODS (Keep All) ===
    def update_entities(
        self,
        new_entities: List[str],
        is_topic_shift: bool = False
    ) -> None:
        """Update active entities based on new turn."""
        if is_topic_shift:
            self.current_entities = new_entities[:5] if new_entities else []
            self.turns_since_entity_update = 0
            logger.debug(f"[SearchState] Topic shift - replaced entities: {self.current_entities}")
        elif new_entities:
            merged = list(dict.fromkeys(new_entities + self.current_entities))
            self.current_entities = merged[:5]
            self.turns_since_entity_update = 0
            logger.debug(f"[SearchState] Merged entities: {self.current_entities}")
        else:
            self.turns_since_entity_update += 1
            logger.debug(f"[SearchState] No new entities, staleness: {self.turns_since_entity_update}")

    def update_topic(self, new_topic: Optional[str], is_topic_shift: bool = False) -> None:
        """
        Update current topic and manage thread topic tracking.

        Args:
            new_topic: The new topic from the current turn
            is_topic_shift: If True, this is a complete topic change (reset thread)
        """
        if not new_topic:
            return

        if is_topic_shift:
            # Complete topic change - reset thread
            self.conversation_thread_topic = new_topic
            self.topic_history = [new_topic]
            self.current_topic = new_topic
            logger.info(f"[SearchState] TOPIC SHIFT: New thread started with '{new_topic}'")
        else:
            # Follow-up or refinement - preserve thread topic
            if not self.conversation_thread_topic:
                # First topic in conversation - set as thread root
                self.conversation_thread_topic = new_topic
                logger.info(f"[SearchState] Thread topic established: '{new_topic}'")

            # Add to topic history if different from last
            if not self.topic_history or self.topic_history[-1] != new_topic:
                self.topic_history.append(new_topic)
                # Trim to max size
                if len(self.topic_history) > self.MAX_TOPIC_HISTORY:
                    self.topic_history = self.topic_history[-self.MAX_TOPIC_HISTORY:]
                logger.debug(f"[SearchState] Topic history: {self.topic_history}")

            self.current_topic = new_topic
            logger.debug(f"[SearchState] Updated topic: {self.current_topic} (thread: {self.conversation_thread_topic})")

    def get_thread_summary(self) -> str:
        """
        Get a summary of the current conversation thread for LLM context.

        Returns string like: "Thread: sleep optimization → caffeine & sleep → alcohol & sleep"
        """
        if not self.topic_history:
            return ""

        if len(self.topic_history) == 1:
            return f"Thread topic: {self.topic_history[0]}"

        # Show evolution
        evolution = " → ".join(self.topic_history[-5:])  # Last 5 topics
        return f"Thread: {evolution} (root: {self.conversation_thread_topic})"

    def reset_thread(self) -> None:
        """Reset conversation thread (called on explicit topic change)."""
        self.conversation_thread_topic = None
        self.topic_history = []
        self.conversation_participants = []  # Also reset participants on topic shift
        logger.info("[SearchState] Conversation thread reset")

    def update_participants(self, entities: List[str], gazetteer=None) -> None:
        """
        Update conversation participants - ONLY adds PEOPLE (not topics/concepts).

        This persists across turns so "the other one" correction pattern works.
        Merges new people with existing, keeping most recent first.

        Args:
            entities: List of entity names from the turn
            gazetteer: Optional gazetteer for person detection (if available)
        """
        if not entities:
            return

        # Heuristic to detect people: names typically have spaces (first + last name)
        # or are known authors/personalities if gazetteer is available
        people_detected = []
        for entity in entities:
            # Skip obvious non-people
            entity_lower = entity.lower()
            non_person_indicators = [
                'ai', 'regulation', 'safety', 'ethics', 'topic', 'theme',
                'technology', 'industry', 'protection', 'authority', 'bots',
                'government', 'intervention', 'media', 'podcast', 'show'
            ]
            if any(indicator in entity_lower for indicator in non_person_indicators):
                continue

            # Check if it looks like a person name (has space = likely first+last)
            if ' ' in entity and len(entity) > 5:
                people_detected.append(entity)
            # Single names that are capitalized and not common words
            elif entity[0].isupper() and len(entity) > 2 and entity_lower not in non_person_indicators:
                # Could be a single-name person like "Naval" or "Lex"
                people_detected.append(entity)

        if not people_detected:
            return

        # Merge: new people first, then existing (deduped)
        seen = set()
        merged = []
        for person in people_detected + self.conversation_participants:
            person_lower = person.lower()
            if person_lower not in seen:
                seen.add(person_lower)
                merged.append(person)

        self.conversation_participants = merged[:self.MAX_PARTICIPANTS]
        logger.debug(f"[SearchState] Conversation participants: {self.conversation_participants}")

    def update_time_filter(self, time_filter: Optional[Dict[str, Any]]) -> None:
        """Persist last time filter for follow-up queries."""
        if time_filter:
            self.last_time_filter = time_filter
            logger.debug(f"[SearchState] Updated time filter: {self.last_time_filter}")

    def update_artifact(self, artifact_id: Optional[str], artifact_title: Optional[str]) -> None:
        """Track last shown artifact for 'more like this' queries."""
        if artifact_id:
            self.last_artifact_id = artifact_id
            self.last_artifact_title = artifact_title
            logger.debug(f"[SearchState] Updated artifact: {artifact_id[:30]}... | {artifact_title or 'N/A'}")

    def decay_if_stale(self) -> None:
        """Called in background - decay entities if not refreshed in 20 turns."""
        if self.turns_since_entity_update > 20:
            logger.info(f"[SearchState] Decaying stale entities (not updated in {self.turns_since_entity_update} turns)")
            self.current_entities = []
            self.current_topic = None
            self.turns_since_entity_update = 0

    def get_primary_entity(self) -> Optional[str]:
        """Get the most relevant entity for pronoun resolution."""
        return self.current_entities[0] if self.current_entities else None

    # === NEW METHODS FOR AGENT ===

    def update_last_action(
        self,
        action_type: str,
        target_id: Optional[str] = None,
        target_title: Optional[str] = None,
        target_summary: Optional[str] = None,
        source_branch: Optional[str] = None,
        published_date: Optional[str] = None,
    ) -> None:
        """
        Update last action context. Called by ALL branches after responding.

        Args:
            action_type: "clip_shown", "episode_shown", "explanation", "greeting"
            target_id: Chunk ID, Episode ID, or None
            target_title: Human-readable title
            target_summary: Brief description
            source_branch: "small_talk", "episode_search", "clip_search"
            published_date: Publication date of episode/clip (YYYY-MM-DD format)
        """
        self.last_action = LastActionContext(
            action_type=action_type,
            target_id=target_id,
            target_title=target_title,
            target_summary=target_summary,
            source_branch=source_branch,
            published_date=published_date,
        )
        logger.info(f"[SearchState] Updated last_action: {action_type} | {target_title or 'N/A'} | {published_date or 'no date'}")

    def add_route_record(
        self,
        turn_id: str,
        route_chosen: str,
        route_confidence: float,
        query_intent: str,
        outcome: str,
    ) -> None:
        """Add routing decision to history."""
        record = RouteRecord(
            turn_id=turn_id,
            route_chosen=route_chosen,
            route_confidence=route_confidence,
            query_intent=query_intent,
            outcome=outcome,
        )
        self.route_history.append(record)

        # Keep only last N records
        if len(self.route_history) > self.MAX_ROUTE_HISTORY:
            self.route_history = self.route_history[-self.MAX_ROUTE_HISTORY:]

        logger.debug(f"[SearchState] Route recorded: {route_chosen} (conf={route_confidence:.2f})")

    def update_conversation_phase(self, new_phase: str) -> None:
        """Update conversation phase based on user behavior."""
        valid_phases = ["discovery", "deep_dive", "comparison", "idle"]
        if new_phase in valid_phases:
            self.conversation_phase = new_phase
            logger.debug(f"[SearchState] Phase updated: {self.conversation_phase}")

    def get_route_pattern(self) -> str:
        """Get recent routing pattern for context (e.g., 'clip->clip->small_talk')."""
        if not self.route_history:
            return "no_history"
        return "->".join([r.route_chosen[:4] for r in self.route_history[-3:]])

    def render_for_prompt(self) -> str:
        """
        Render state as context for LLM prompts.
        UPDATED to include last_action and route history.
        """
        parts = []

        # Last Action (HIGHEST PRIORITY for continuation queries)
        if self.last_action.action_type:
            parts.append(self.last_action.render_for_prompt())

        # Active Entities
        if self.current_entities:
            parts.append(f"ACTIVE ENTITIES (for pronoun resolution): {', '.join(self.current_entities)}")
            parts.append(f"  -> If user says 'he/she/they', resolve to: {self.current_entities[0]}")

        # Current Topic & Thread Topic
        if self.conversation_thread_topic:
            parts.append(f"CONVERSATION THREAD TOPIC: {self.conversation_thread_topic}")
            parts.append(f"  -> 'this stuff', 'that topic' = this thread topic")
        if self.current_topic and self.current_topic != self.conversation_thread_topic:
            parts.append(f"CURRENT TURN TOPIC: {self.current_topic}")

        # Last Shown Artifact
        if self.last_artifact_title:
            parts.append(f"LAST SHOWN CLIP: {self.last_artifact_title}")
            parts.append(f"  -> If user says 'more like that' or 'similar', reference this clip")

        # Route History (for router context)
        if self.route_history:
            recent_routes = self.get_route_pattern()
            parts.append(f"RECENT ROUTE PATTERN: {recent_routes}")

        # Conversation Phase
        parts.append(f"CONVERSATION PHASE: {self.conversation_phase}")

        return "\n".join(parts) if parts else "No active conversation state."

    def render_for_router(self, recent_turns: List = None) -> str:
        """
        Rich context rendering specifically for Router decisions.

        The router is the FIRST decision point - if it routes wrong, everything fails.
        This context must enable accurate routing by providing:

        1. LAST ACTION WITH SUMMARY - Not just "clip_shown" but WHAT the content was about
        2. RECENT TURN HISTORY - Last 3 turns showing conversation flow
        3. ENTITY-SUBJECT MAPPING - Who is being discussed (not just who exists)
        4. CONVERSATION FLOW INDICATORS - Drilling down vs pivoting vs exploring
        5. DISAMBIGUATION HINTS - What "this/that/more" refer to
        6. INTENT PATTERN - Recent routing patterns to predict behavior

        Args:
            recent_turns: List of ConversationTurn objects from memory

        Returns:
            Structured context string optimized for routing decisions.
        """
        parts = []

        # =================================================================
        # SECTION 1: LAST ACTION WITH CONTENT SUMMARY (Critical!)
        # =================================================================
        if self.last_action.action_type:
            parts.append("=== LAST ACTION (what user just saw/did) ===")
            parts.append(f"TYPE: {self.last_action.action_type}")
            if self.last_action.target_title:
                parts.append(f"CONTENT: \"{self.last_action.target_title}\"")
            # KEY: Include what the content was ABOUT
            if self.last_action.target_summary:
                parts.append(f"ABOUT: {self.last_action.target_summary}")  # No truncation - 1M context
            parts.append("")
        else:
            parts.append("=== LAST ACTION ===")
            parts.append("TYPE: none (conversation start)")
            parts.append("")

        # =================================================================
        # SECTION 2: RECENT CONVERSATION FLOW (Shows trajectory)
        # =================================================================
        if recent_turns and len(recent_turns) > 0:
            parts.append("=== RECENT CONVERSATION (last 3 turns) ===")
            for turn in recent_turns[-3:]:
                # Show what user asked and what they got
                parts.append(f"USER: \"{turn.user_question[:60]}{'...' if len(turn.user_question) > 60 else ''}\"")
                if turn.answer_summary:
                    parts.append(f"  GOT: {turn.answer_summary[:80]}...")
                if turn.artifact_title:
                    parts.append(f"  SHOWN: {turn.artifact_title}")
            parts.append("")

        # =================================================================
        # SECTION 3: ACTIVE ENTITIES WITH SUBJECT DETECTION
        # =================================================================
        if self.current_entities:
            parts.append("=== ACTIVE ENTITIES (for pronoun resolution) ===")
            parts.append(f"PRIMARY (default 'he/she/him/her'): {self.current_entities[0]}")
            if len(self.current_entities) > 1:
                parts.append(f"OTHERS: {', '.join(self.current_entities[1:4])}")

            # Detect WHO is the SUBJECT of discussion
            if self.last_action.target_summary:
                summary_lower = self.last_action.target_summary.lower()
                for entity in self.current_entities[:2]:
                    if entity.lower() in summary_lower:
                        parts.append(f"SUBJECT OF LAST CONTENT: {entity}")
                        break
            parts.append("")

        # =================================================================
        # SECTION 4: CURRENT TOPIC & THREAD TOPIC & DISAMBIGUATION
        # =================================================================
        if self.conversation_thread_topic:
            parts.append(f"THREAD_TOPIC: \"{self.conversation_thread_topic}\"")
            parts.append("  -> 'this stuff', 'that topic', 'more about it' = this thread topic")
            if self.topic_history and len(self.topic_history) > 1:
                parts.append(f"  -> Topic evolution: {' → '.join(self.topic_history[-4:])}")
        if self.current_topic and self.current_topic != self.conversation_thread_topic:
            parts.append(f"CURRENT_TOPIC: \"{self.current_topic}\"")

        # Conversation participants (for "the other one" corrections)
        if self.conversation_participants and len(self.conversation_participants) >= 2:
            parts.append(f"PEOPLE_IN_CONVERSATION: {', '.join(self.conversation_participants[:4])}")
            parts.append(f"  -> 'the other one' = {self.conversation_participants[1]} (not {self.conversation_participants[0]})")
        if self.last_artifact_title:
            parts.append(f"LAST_ARTIFACT: \"{self.last_artifact_title}\"")
            parts.append("  -> 'this clip', 'that episode', 'more like this' = this artifact")
        if self.current_topic or self.last_artifact_title:
            parts.append("")

        # =================================================================
        # SECTION 5: CONVERSATION FLOW INDICATORS
        # =================================================================
        parts.append("=== CONVERSATION STATE ===")
        parts.append(f"PHASE: {self.conversation_phase}")

        # Detect drilling down vs exploring
        if self.route_history:
            recent_routes = [r.route_chosen for r in self.route_history[-3:]]
            if recent_routes.count(recent_routes[0]) == len(recent_routes) and len(recent_routes) >= 2:
                parts.append(f"PATTERN: User is drilling down (same route {len(recent_routes)}x)")
            elif len(set(recent_routes)) == len(recent_routes) and len(recent_routes) >= 2:
                parts.append("PATTERN: User is exploring (different routes)")
            parts.append(f"ROUTE_HISTORY: {' → '.join(recent_routes)}")

        if self.last_was_followup:
            parts.append("FLOW: Previous query was a follow-up (likely this one too)")

        result = "\n".join(parts) if parts else "NEW_SESSION - no prior context"
        logger.debug(f"[SearchState] Router context rendered: {len(result)} chars")
        return result

    def render_for_query_analyzer(self, recent_turns: List = None) -> str:
        """
        Rich context rendering specifically for Query Analyzer.

        This provides MORE context than router_context because the analyzer
        needs to understand:
        1. WHO did WHAT (not just names, but what they discussed/did)
        2. Previous time filters (for temporal follow-ups)
        3. Entity roles (guest vs host vs subject of discussion)
        4. Last action summary (what the shown content was about)
        5. FULL conversation thread history (not just last 3 turns!)
        6. Thread topic that persists across related turns

        TOPIC DRIFT FIX: Now includes full conversation thread summary so
        "this stuff" / "that topic" resolves to the thread topic, not just
        the last turn's topic (which may be wrong if search returned bad results).

        Args:
            recent_turns: List of ConversationTurn objects from memory

        Returns:
            Structured context string for pronoun resolution and query building.
        """
        parts = []

        # =================================================================
        # SECTION 0: CONVERSATION THREAD SUMMARY (CRITICAL FOR "THIS STUFF")
        # =================================================================
        # This section provides the LLM with the full thread context so it knows
        # what "this stuff", "that topic", "more about this" actually refers to
        thread_summary = self.get_thread_summary()
        if thread_summary:
            parts.append("=== CONVERSATION THREAD (what 'this/that/it' refers to) ===")
            parts.append(thread_summary)
            if self.conversation_thread_topic:
                parts.append(f"ROOT TOPIC: \"{self.conversation_thread_topic}\"")
                parts.append("  -> 'this stuff', 'that topic', 'more about it' = this ROOT TOPIC")
            parts.append("")

        # =================================================================
        # SECTION 0.5: CONVERSATION PARTICIPANTS (CRITICAL FOR "THE OTHER ONE")
        # =================================================================
        # This section lists PEOPLE mentioned in the conversation thread
        # so "the other one" / "not him" corrections can resolve correctly
        if self.conversation_participants:
            parts.append("=== CONVERSATION PARTICIPANTS (PEOPLE mentioned - for 'the other one') ===")
            parts.append(f"People in this conversation: {', '.join(self.conversation_participants)}")
            if len(self.conversation_participants) >= 2:
                parts.append(f"  -> If user says 'not him, the other one': exclude {self.conversation_participants[0]}, use {self.conversation_participants[1]}")
            parts.append("")

        # =================================================================
        # SECTION 1: FULL CONVERSATION HISTORY (not just last 3!)
        # =================================================================
        # Show ALL recent turns so LLM understands the full conversation flow
        # This prevents topic drift when search returns wrong results
        # Phase 2: Now includes key_quotes, topics_covered, notable_examples for richer context
        if recent_turns and len(recent_turns) > 0:
            parts.append("=== FULL CONVERSATION HISTORY (use this for context) ===")
            # Show ALL turns (up to 10) for full context
            for turn in recent_turns[-10:]:
                parts.append(f"[{turn.turn_id}] User asked: \"{turn.user_question}\"")
                parts.append(f"     Resolved to: \"{turn.resolved_query}\"")
                parts.append(f"     Result: {turn.answer_summary}")
                if turn.key_entities:
                    parts.append(f"     Entities involved: {', '.join(turn.key_entities[:6])}")

                # Option A: Enhanced context fields (Phase 2)
                key_quotes = getattr(turn, 'key_quotes', [])
                topics_covered = getattr(turn, 'topics_covered', [])
                notable_examples = getattr(turn, 'notable_examples', [])

                if key_quotes:
                    parts.append(f"     Key Quotes: {'; '.join(key_quotes[:2])}")
                if topics_covered:
                    parts.append(f"     Topics Covered: {', '.join(topics_covered[:4])}")
                if notable_examples:
                    parts.append(f"     Notable: {', '.join(notable_examples[:2])}")

                parts.append("")  # blank line between turns

        # SECTION 2: Last Action with FULL context (critical for "him/that/it")
        if self.last_action.action_type:
            parts.append("=== LAST ACTION (what was just shown) ===")
            parts.append(f"Action: {self.last_action.action_type}")
            if self.last_action.target_title:
                parts.append(f"Content: {self.last_action.target_title}")
            if self.last_action.target_summary:
                # KEY - tells analyzer what the content was ABOUT
                parts.append(f"Summary: {self.last_action.target_summary}")

        # SECTION 3: Active Entities with context about who did what
        if self.current_entities:
            parts.append("\n=== ACTIVE ENTITIES (for pronoun resolution) ===")
            parts.append(f"Primary entity (default for 'he/she/him/her'): {self.current_entities[0]}")
            if len(self.current_entities) > 1:
                parts.append(f"Other entities: {', '.join(self.current_entities[1:5])}")

            # Infer who was the SUBJECT based on last action summary
            if self.last_action.target_summary:
                summary_lower = self.last_action.target_summary.lower()
                for entity in self.current_entities[:3]:
                    entity_lower = entity.lower()
                    if entity_lower in summary_lower:
                        # Check if entity appears as subject of action verbs
                        subject_patterns = [
                            f"{entity_lower} describes",
                            f"{entity_lower} explains",
                            f"{entity_lower} discusses",
                            f"{entity_lower} talks",
                            f"{entity_lower} shares",
                            f"{entity_lower} lost",
                            f"{entity_lower} gained",
                            f"{entity_lower}'s",
                        ]
                        for pattern in subject_patterns:
                            if pattern in summary_lower:
                                parts.append(f"  -> '{entity}' is the SUBJECT being discussed (mentioned in summary)")
                                break

        # SECTION 4: Current Topic (skip if already covered in thread summary above)
        # Only show if we have a topic that differs from thread topic
        if self.current_topic and self.current_topic != self.conversation_thread_topic:
            parts.append(f"\n=== CURRENT TURN TOPIC ===")
            parts.append(f"This turn's specific topic: {self.current_topic}")
            parts.append(f"(Thread root topic remains: {self.conversation_thread_topic or 'not set'})")

        # SECTION 5: Previous Time Filter (CRITICAL for temporal follow-ups)
        if self.last_time_filter and self.last_time_filter.get("mode", "none") != "none":
            parts.append("\n=== PREVIOUS TIME FILTER (preserve on follow-ups) ===")
            mode = self.last_time_filter.get("mode")
            parts.append(f"Mode: {mode}")
            if self.last_time_filter.get("start_date_utc"):
                parts.append(f"Start date: {self.last_time_filter['start_date_utc']}")
            if self.last_time_filter.get("end_date_utc"):
                parts.append(f"End date: {self.last_time_filter['end_date_utc']}")
            if self.last_time_filter.get("recency_priority"):
                parts.append(f"Recency priority: {self.last_time_filter['recency_priority']}")
            parts.append("-> IMPORTANT: If this is a follow-up question without new dates,")
            parts.append("   carry forward this time filter to maintain temporal context!")

        # SECTION 6: Last Artifact Reference
        if self.last_artifact_title:
            parts.append(f"\n=== LAST SHOWN CONTENT ===")
            parts.append(f"Title: {self.last_artifact_title}")
            parts.append("-> 'more like that', 'similar clips', 'that episode' refers to this")

        # SECTION 7: Conversation State
        parts.append(f"\n=== CONVERSATION STATE ===")
        parts.append(f"Phase: {self.conversation_phase}")
        if self.last_was_followup:
            parts.append("Previous query was a follow-up (likely this one is too)")

        result = "\n".join(parts) if parts else "NEW_SESSION - no prior context"
        logger.debug(f"[SearchState] Analyzer context rendered: {len(result)} chars")
        return result

    def render_for_small_talk(self, recent_turns: List = None, query: str = "") -> Dict[str, Any]:
        """
        Rich context rendering specifically for Small Talk branch.

        Returns structured data (not just string) so small_talk can:
        1. Check if query mentions an in-context entity
        2. Get synthesized knowledge about that entity from conversation
        3. Decide whether to use memory context vs search grounding
        4. Generate follow-up suggestions

        Args:
            recent_turns: List of ConversationTurn objects from memory
            query: The user's current query (for entity matching)

        Returns:
            Dict with structured context for intelligent small talk responses
        """
        query_lower = query.lower() if query else ""

        # =====================================================================
        # PRONOUN RESOLUTION: Map pronouns to primary entity or last action subject
        # =====================================================================
        pronoun_patterns = [
            "this guy", "that guy", "this person", "that person",
            "who is he", "who's he", "who is she", "who's she",
            "who are they", "who's they", "tell me about him", "tell me about her",
            "who is this", "who's this", "who is that", "who's that",
            "about him", "about her", "about them", "about this",
        ]

        # Check if query uses pronouns
        uses_pronoun = any(pattern in query_lower for pattern in pronoun_patterns)

        # Also check for simple pronoun patterns like just "he" or "him" in knowledge questions
        knowledge_with_pronoun = False
        simple_pronouns = ["he", "she", "him", "her", "they", "them", "this", "that"]
        knowledge_patterns = ["who is", "who's", "tell me about", "what about", "explain"]
        if any(kp in query_lower for kp in knowledge_patterns):
            for pronoun in simple_pronouns:
                # Check if pronoun appears as a word boundary
                if f" {pronoun}" in f" {query_lower}" or f"{pronoun} " in f"{query_lower} " or f" {pronoun}?" in f" {query_lower}":
                    knowledge_with_pronoun = True
                    break

        # Determine target entity for pronoun resolution
        resolved_entity = None
        if uses_pronoun or knowledge_with_pronoun:
            # First try: Subject of last action (most relevant)
            if self.last_action.target_summary:
                summary_lower = self.last_action.target_summary.lower()
                for entity in self.current_entities[:3]:
                    if entity.lower() in summary_lower:
                        resolved_entity = entity
                        logger.debug(f"[SearchState] Pronoun resolved via last_action subject: {entity}")
                        break

            # Second try: Primary entity (fallback)
            if not resolved_entity and self.current_entities:
                resolved_entity = self.current_entities[0]
                logger.debug(f"[SearchState] Pronoun resolved via primary entity: {resolved_entity}")

        # Build entity knowledge from recent turns
        entity_knowledge: Dict[str, Dict[str, Any]] = {}
        for entity in self.current_entities[:10]:
            entity_lower = entity.lower()
            # Mark as mentioned if literally in query OR if resolved via pronoun
            is_mentioned = entity_lower in query_lower or (resolved_entity and entity.lower() == resolved_entity.lower())
            entity_knowledge[entity_lower] = {
                "name": entity,
                "mentioned_in_query": is_mentioned,
                "resolved_via_pronoun": resolved_entity and entity.lower() == resolved_entity.lower(),
                "facts": [],
                "appeared_in": [],
                "themes": [],
            }

        # Extract facts about entities from recent turns
        if recent_turns:
            for turn in recent_turns[-5:]:  # Last 5 turns
                summary = turn.answer_summary or ""
                title = turn.artifact_title or ""
                summary_lower = summary.lower()
                title_lower = title.lower()

                for entity_key, entity_data in entity_knowledge.items():
                    entity_name = entity_data["name"]
                    entity_lower = entity_name.lower()

                    # Check if entity was involved in this turn (multiple ways)
                    entity_in_key_entities = entity_name in turn.key_entities
                    entity_in_summary = entity_lower in summary_lower
                    entity_in_title = entity_lower in title_lower

                    if entity_in_key_entities or entity_in_summary or entity_in_title:
                        # Extract facts from summary - this IS the fact about the entity
                        if summary:
                            # Don't duplicate identical summaries
                            if summary not in entity_data["facts"]:
                                entity_data["facts"].append(summary)
                        if title:
                            if title not in entity_data["appeared_in"]:
                                entity_data["appeared_in"].append(title)
                        # Add themes
                        for theme in turn.themes:
                            if theme not in entity_data["themes"]:
                                entity_data["themes"].append(theme)

        # Deduplicate
        for entity_data in entity_knowledge.values():
            entity_data["facts"] = list(dict.fromkeys(entity_data["facts"]))[:3]
            entity_data["appeared_in"] = list(dict.fromkeys(entity_data["appeared_in"]))[:3]
            entity_data["themes"] = list(dict.fromkeys(entity_data["themes"]))[:5]

        # Find entities mentioned in the query (including pronoun-resolved)
        queried_entities = [
            entity_data for entity_data in entity_knowledge.values()
            if entity_data["mentioned_in_query"]
        ]

        # Fallback: If pronoun was resolved but no queried_entities found,
        # create a synthetic entry so the prompt knows who we're talking about
        if not queried_entities and resolved_entity:
            logger.debug(f"[SearchState] Creating synthetic queried entity for {resolved_entity}")
            queried_entities = [{
                "name": resolved_entity,
                "mentioned_in_query": True,
                "resolved_via_pronoun": True,
                "facts": [],  # Will need to use last_action for context
                "appeared_in": [self.last_artifact_title] if self.last_artifact_title else [],
                "themes": [],
            }]

        # Build conversation summary
        conversation_summary = []
        if recent_turns:
            for turn in recent_turns[-3:]:
                conversation_summary.append({
                    "user_asked": turn.user_question,
                    "result": turn.answer_summary,
                    "content_shown": turn.artifact_title,
                    "entities": turn.key_entities[:4],
                })

        # Determine if this is a contextual question
        # True if: entity mentioned by name, OR pronoun resolved to entity, OR pronoun used with context
        is_contextual = len(queried_entities) > 0 or (
            (uses_pronoun or knowledge_with_pronoun) and (self.current_entities or self.last_action.target_title)
        )

        # Build suggested follow-ups based on context
        follow_up_suggestions = []
        if self.current_entities:
            primary = self.current_entities[0]
            follow_up_suggestions.append(f"Show more clips of {primary}")
        if self.current_topic:
            follow_up_suggestions.append(f"Explore more about {self.current_topic}")
        if self.last_artifact_title:
            follow_up_suggestions.append(f"Find similar content to what we just watched")

        return {
            "is_contextual_question": is_contextual,
            "queried_entities": queried_entities,
            "all_entity_knowledge": entity_knowledge,
            "conversation_summary": conversation_summary,
            "last_action": {
                "type": self.last_action.action_type,
                "title": self.last_action.target_title,
                "summary": self.last_action.target_summary,
                "published_date": self.last_action.published_date,
            },
            "current_topic": self.current_topic,
            "current_themes": [],  # Will be populated from memory.conversation_themes
            "conversation_phase": self.conversation_phase,
            "follow_up_suggestions": follow_up_suggestions[:3],
            # Pronoun resolution info
            "resolved_entity": resolved_entity,
            "uses_pronoun": uses_pronoun or knowledge_with_pronoun,
        }


@dataclass
class EntityMention:
    """Tracked entity with recency weighting."""
    name: str
    mention_count: int
    last_turn_index: int
    entity_type: str  # "person", "topic", "show"

    def relevance_score(self, current_turn: int) -> float:
        """Calculate relevance with recency decay."""
        turns_ago = current_turn - self.last_turn_index
        decay = ENTITY_DECAY_FACTOR ** turns_ago
        return self.mention_count * decay


@dataclass
class ConversationMemory:
    """
    Efficient conversation memory with automatic compression.

    Memory is structured in tiers:
    1. Recent turns (full context) - last 5 turns
    2. Compressed history (summaries) - turns 6-15
    3. Entity graph (key references) - always maintained
    4. Theme summary (rolling) - conversation-wide themes
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Tier 1: Recent turns (full detail)
    recent_turns: List[ConversationTurn] = field(default_factory=list)

    # Tier 2: Compressed history
    compressed_summary: str = ""

    # Tier 3: Entity tracking for pronoun resolution
    entities: Dict[str, EntityMention] = field(default_factory=dict)

    # Tier 4: Rolling themes
    conversation_themes: List[str] = field(default_factory=list)

    # Artifact history (for "show me something else" type requests)
    shown_artifacts: List[str] = field(default_factory=list)

    # Current turn counter
    turn_count: int = 0

    # ===========================================================================
    # NEW: SearchState - The Brain
    # ===========================================================================
    search_state: SearchState = field(default_factory=SearchState)

    # ===========================================================================
    # NEW: Strict Exclusion Window (artifact_id, turn_number)
    # Chunks in this window are EXCLUDED (not just deprioritized)
    # ===========================================================================
    excluded_artifact_window: List[Tuple[str, int]] = field(default_factory=list)
    EXCLUSION_WINDOW_TURNS: int = 5  # Exclude for 5 turns

    def add_turn(
        self,
        user_question: str,
        resolved_query: str,
        answer_summary: str,
        artifact_id: Optional[str],
        artifact_title: Optional[str],
        entities: List[str],
        themes: List[str],
    ) -> None:
        """Add a new turn and trigger compression if needed."""
        self.turn_count += 1

        turn = ConversationTurn(
            turn_id=self._generate_turn_id(),
            timestamp=datetime.utcnow(),
            user_question=user_question,
            resolved_query=resolved_query,
            answer_summary=answer_summary,
            artifact_id=artifact_id,
            artifact_title=artifact_title,
            key_entities=entities,
            themes=themes,
        )

        self.recent_turns.append(turn)

        logger.info(f"[MEMORY] Session {self.session_id} | Turn {self.turn_count} added")
        logger.info(f"  ├─ Question: {user_question[:80]}{'...' if len(user_question) > 80 else ''}")
        logger.info(f"  ├─ Resolved: {resolved_query[:80]}{'...' if len(resolved_query) > 80 else ''}")
        logger.info(f"  ├─ Summary: {answer_summary[:60]}{'...' if len(answer_summary) > 60 else ''}")
        logger.info(f"  ├─ Entities: {entities[:5]}")
        logger.info(f"  └─ Themes: {themes[:3]}")

        # Update entity tracking
        for entity in entities:
            self._update_entity(entity, "auto")

        # Update themes
        for theme in themes:
            if theme not in self.conversation_themes:
                self.conversation_themes.append(theme)
        self.conversation_themes = self.conversation_themes[-10:]  # Keep last 10

        # Track artifact
        if artifact_id:
            self.shown_artifacts.append(artifact_id)
            self.shown_artifacts = self.shown_artifacts[-20:]  # Keep last 20

        # Compress if needed
        if len(self.recent_turns) > MAX_RECENT_TURNS:
            self._compress_oldest_turn()

    def _compress_oldest_turn(self) -> None:
        """Move oldest turn from recent to compressed summary."""
        if not self.recent_turns:
            return

        oldest = self.recent_turns.pop(0)

        # Append to compressed summary
        turn_bullet = f"- Q{oldest.turn_id}: {oldest.user_question[:50]}... → {oldest.answer_summary[:100]}..."

        if self.compressed_summary:
            self.compressed_summary += f"\n{turn_bullet}"
        else:
            self.compressed_summary = turn_bullet

        logger.debug(f"[MEMORY] Compressed turn {oldest.turn_id} to summary")

        # Trim compressed summary if too long
        if len(self.compressed_summary) > 10000:
            lines = self.compressed_summary.split('\n')
            self.compressed_summary = '\n'.join(lines[-MAX_COMPRESSED_TURNS:])
            logger.debug(f"[MEMORY] Trimmed compressed summary to {len(lines)} entries")

    def _update_entity(self, entity: str, entity_type: str) -> None:
        """Update entity tracking with new mention."""
        key = entity.lower()
        if key in self.entities:
            self.entities[key].mention_count += 1
            self.entities[key].last_turn_index = self.turn_count
        else:
            self.entities[key] = EntityMention(
                name=entity,
                mention_count=1,
                last_turn_index=self.turn_count,
                entity_type=entity_type,
            )

        # Prune low-relevance entities
        if len(self.entities) > MAX_ENTITIES:
            scored = [(k, v.relevance_score(self.turn_count)) for k, v in self.entities.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            self.entities = {k: self.entities[k] for k, _ in scored[:MAX_ENTITIES]}

    def _generate_turn_id(self) -> str:
        """Generate unique turn ID."""
        return f"T{self.turn_count}"

    def get_recent_entities(self, top_k: int = 10) -> List[str]:
        """Get most relevant entities for pronoun resolution."""
        scored = [(v.name, v.relevance_score(self.turn_count)) for v in self.entities.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[:top_k]]

    def get_last_artifact(self) -> Optional[str]:
        """Get the most recently shown artifact ID."""
        return self.shown_artifacts[-1] if self.shown_artifacts else None

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent turn."""
        return self.recent_turns[-1] if self.recent_turns else None

    # ===========================================================================
    # NEW: Exclusion Window Methods
    # ===========================================================================

    def add_to_exclusion_window(self, artifact_id: str) -> None:
        """
        Add artifact to strict exclusion window.
        Artifacts in this window will be EXCLUDED (not shown) for EXCLUSION_WINDOW_TURNS.
        """
        if artifact_id:
            self.excluded_artifact_window.append((artifact_id, self.turn_count))
            # Clean up old entries
            self.excluded_artifact_window = [
                (aid, turn) for aid, turn in self.excluded_artifact_window
                if self.turn_count - turn < self.EXCLUSION_WINDOW_TURNS
            ]
            logger.debug(
                f"[Memory] Added {artifact_id[:20]}... to exclusion window. "
                f"Window size: {len(self.excluded_artifact_window)}"
            )

    def get_excluded_ids(self) -> Set[str]:
        """
        Get artifact IDs that should be EXCLUDED from search results.
        These are chunks shown in the last EXCLUSION_WINDOW_TURNS turns.
        """
        return {
            aid for aid, turn in self.excluded_artifact_window
            if self.turn_count - turn < self.EXCLUSION_WINDOW_TURNS
        }

    def is_artifact_excluded(self, artifact_id: str) -> bool:
        """Check if a specific artifact should be excluded."""
        return artifact_id in self.get_excluded_ids()

    # ===========================================================================
    # NEW: Enhanced render_for_prompt with SearchState
    # ===========================================================================

    def render_for_prompt_enhanced(self, max_chars: int = None) -> str:  # max_chars ignored - 1M context available
        """
        Enhanced memory rendering that includes SearchState with last_action context.

        UPDATED FOR AGENT: Now includes last_action for cross-branch context.
        NOTE: max_chars parameter kept for backwards compatibility but ignored.
              LLMs have 1M context window - no truncation needed.
        """
        _ = max_chars  # Suppress unused parameter warning - kept for backwards compatibility
        parts = []

        # SECTION 1: SearchState (HIGHEST PRIORITY - includes last_action)
        state_context = self.search_state.render_for_prompt()
        if state_context and state_context != "No active conversation state.":
            parts.append(f"**CURRENT CONTEXT STATE:**\n{state_context}")

        # SECTION 2: Active Entities (legacy support - deduplicated)
        if self.entities:
            top_entities = self.get_recent_entities(8)
            state_entities = set(e.lower() for e in self.search_state.current_entities)
            additional_entities = [e for e in top_entities if e.lower() not in state_entities]
            if additional_entities:
                parts.append(f"**OTHER MENTIONED ENTITIES:** {', '.join(additional_entities)}")

        # SECTION 3: Conversation Themes
        if self.conversation_themes:
            parts.append(f"**THEMES:** {', '.join(self.conversation_themes[-5:])}")

        # SECTION 4: Recent Turns (expanded context)
        if self.recent_turns:
            recent_text = []
            for turn in self.recent_turns[-5:]:
                recent_text.append(
                    f"[{turn.turn_id}] User: \"{turn.user_question}\"\n"
                    f"    -> Resolved: \"{turn.resolved_query}\"\n"
                    f"    -> Clip: {turn.artifact_title or 'None'}\n"
                    f"    -> Entities: {', '.join(turn.key_entities[:3]) if turn.key_entities else 'none'}"
                )
            parts.append(f"**RECENT CONVERSATION:**\n" + "\n".join(recent_text))

        # SECTION 5: Compressed History (no truncation - 1M context window available)
        if self.compressed_summary:
            parts.append(f"**EARLIER CONTEXT:**\n{self.compressed_summary}")

        rendered = "\n\n".join(parts)

        # No truncation - LLMs have 1M context window
        # Only truncate if explicitly requested with a very low max_chars

        return rendered if rendered else "No conversation history yet."

    def render_for_prompt(self, max_chars: int = None) -> str:  # max_chars ignored - 1M context available
        """
        Render memory as compact string for LLM prompt.
        Now delegates to enhanced version with SearchState support.
        NOTE: max_chars kept for backwards compatibility but ignored (1M context available).
        """
        return self.render_for_prompt_enhanced(max_chars)

    # ===========================================================================
    # NEW: Unified Memory Update Method for Agent Architecture
    # ===========================================================================

    def apply_branch_memory_update(
        self,
        update: 'BranchMemoryUpdate',  # Forward reference to schema
        route_chosen: str,
        route_confidence: float,
        query_intent: str,
        user_question: str,
        resolved_query: str,
    ) -> None:
        """
        Apply unified memory update from ANY branch.

        This is the SINGLE method all branches use to update memory,
        ensuring consistency across small_talk, episode_search, and clip_search.

        Args:
            update: BranchMemoryUpdate from the branch
            route_chosen: Which branch handled it
            route_confidence: Router's confidence
            query_intent: Interpreted intent
            user_question: Original user question
            resolved_query: Query after pronoun resolution
        """
        self.turn_count += 1
        turn_id = f"T{self.turn_count}"

        logger.info("")
        logger.info("=" * 70)
        logger.info("[MEMORY] 🧠 UPDATING CONVERSATION MEMORY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("[MEMORY] 📝 TURN DETAILS:")
        logger.info(f"  ├─ Turn ID: {turn_id}")
        logger.info(f"  ├─ Session: {self.session_id}")
        logger.info(f"  ├─ User asked: \"{user_question[:60]}{'...' if len(user_question) > 60 else ''}\"")
        logger.info(f"  ├─ Resolved to: \"{resolved_query[:60]}{'...' if len(resolved_query) > 60 else ''}\"")
        logger.info(f"  └─ Route: {route_chosen.upper()} (confidence: {route_confidence:.1%})")

        logger.info("")
        logger.info("[MEMORY] 📊 ACTION TAKEN:")
        action_emoji = {"clip_shown": "🎬", "episode_shown": "🎧", "greeting": "👋", "explanation": "📖", "error": "❌"}
        logger.info(f"  ├─ Type: {action_emoji.get(update.action_type, '📌')} {update.action_type}")
        if update.action_target_title:
            logger.info(f"  └─ Content: \"{update.action_target_title[:60]}{'...' if len(update.action_target_title or '') > 60 else ''}\"")
        else:
            logger.info("  └─ Content: (no specific content shown)")

        # 1. Update SearchState with last_action (including published_date for temporal context)
        self.search_state.update_last_action(
            action_type=update.action_type,
            target_id=update.action_target_id,
            target_title=update.action_target_title,
            target_summary=update.turn_summary,
            source_branch=route_chosen,
            published_date=getattr(update, 'published_date', None),
        )

        # 2. Add route record
        self.search_state.add_route_record(
            turn_id=turn_id,
            route_chosen=route_chosen,
            route_confidence=route_confidence,
            query_intent=query_intent,
            outcome="success" if update.action_type != "error" else "error",
        )

        # 3. Update entities
        self.search_state.update_entities(
            update.entities_mentioned,
            is_topic_shift=update.is_topic_shift,
        )

        # 3.5. Update conversation participants (PEOPLE only - for "the other one" pattern)
        self.search_state.update_participants(update.entities_mentioned)

        # 4. Update topic if provided (pass is_topic_shift for thread tracking)
        if update.topics_discussed:
            self.search_state.update_topic(
                update.topics_discussed[0],
                is_topic_shift=update.is_topic_shift
            )

        # 5. Update conversation phase if suggested
        if update.suggested_phase:
            self.search_state.update_conversation_phase(update.suggested_phase)

        # 6. Track artifact if clip/episode was shown
        if update.action_type in ("clip_shown", "episode_shown") and update.action_target_id:
            self.search_state.update_artifact(
                update.action_target_id,
                update.action_target_title,
            )
            self.shown_artifacts.append(update.action_target_id)
            self.shown_artifacts = self.shown_artifacts[-20:]

            # Add to exclusion window for both clip_shown and episode_shown
            # This prevents showing the same content again in the next few turns
            if update.action_type in ("clip_shown", "episode_shown"):
                self.add_to_exclusion_window(update.action_target_id)

        # 7. Create ConversationTurn record with Phase 2 enhanced fields
        # Extract enhanced fields from update (with fallback for backwards compatibility)
        key_quotes = getattr(update, 'key_quotes', []) or []
        topics_covered = getattr(update, 'topics_covered', []) or []
        notable_examples = getattr(update, 'notable_examples', []) or []

        logger.info("")
        logger.info("[MEMORY] 📚 STORING CONVERSATION TURN:")
        logger.info(f"  ├─ Summary: \"{update.turn_summary[:60]}...\"")
        logger.info(f"  ├─ Entities: {update.entities_mentioned[:5]}")
        logger.info(f"  ├─ Topics: {update.topics_discussed[:3]}")
        logger.info(f"  └─ Topic shift? {'Yes (new conversation thread)' if update.is_topic_shift else 'No (continuing topic)'}")

        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=datetime.utcnow(),
            user_question=user_question,
            resolved_query=resolved_query,
            answer_summary=update.turn_summary[:500],  # Phase 2: increased from 200 to 500
            artifact_id=update.action_target_id,
            artifact_title=update.action_target_title,
            key_entities=update.entities_mentioned[:10],  # Phase 2: increased from 5 to 10
            themes=update.topics_discussed[:5],  # Phase 2: increased from 3 to 5
            # Option A: Enhanced context fields
            key_quotes=key_quotes[:3],
            topics_covered=topics_covered[:5],
            notable_examples=notable_examples[:3],
        )
        self.recent_turns.append(turn)

        # Log enhanced context fields (Option A - Phase 2)
        logger.info("")
        logger.info("[MEMORY] 🔮 ENHANCED CONTEXT (Option A - for follow-up resolution):")
        if key_quotes:
            logger.info(f"  ├─ Key quotes stored: {len(key_quotes)}")
            for q in key_quotes[:2]:
                logger.info(f"  │     • \"{q[:50]}...\"")
        else:
            logger.info("  ├─ Key quotes: (none)")
        if topics_covered:
            logger.info(f"  ├─ Topics covered: {topics_covered[:4]}")
        else:
            logger.info("  ├─ Topics covered: (none)")
        if notable_examples:
            logger.info(f"  └─ Notable examples: {notable_examples[:3]}")
        else:
            logger.info("  └─ Notable examples: (none)")
        logger.info("")
        logger.info("  ℹ️  These fields help resolve queries like 'that example', 'the thing about X'")

        # 8. Update entity tracking (legacy)
        for entity in update.entities_mentioned:
            self._update_entity(entity, "auto")

        # 9. Update themes (legacy)
        for theme in update.topics_discussed:
            if theme not in self.conversation_themes:
                self.conversation_themes.append(theme)
        self.conversation_themes = self.conversation_themes[-10:]

        # 10. Compress if needed
        if len(self.recent_turns) > MAX_RECENT_TURNS:
            self._compress_oldest_turn()

        # Log final summary
        logger.info("")
        logger.info("[MEMORY] 📈 MEMORY STATE AFTER UPDATE:")
        logger.info(f"  ├─ Total turns: {self.turn_count}")
        logger.info(f"  ├─ Recent turns stored: {len(self.recent_turns)}")
        logger.info(f"  ├─ Current entities: {self.search_state.current_entities[:4]}")
        logger.info(f"  ├─ Current topic: {self.search_state.current_topic or '(none)'}")
        logger.info(f"  ├─ Thread topic: {self.search_state.conversation_thread_topic or '(none)'}")
        logger.info(f"  ├─ Conversation phase: {self.search_state.conversation_phase}")
        logger.info(f"  └─ Route pattern (last 5): {self.search_state.get_route_pattern()}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("[MEMORY] ✅ MEMORY UPDATE COMPLETE")
        logger.info("=" * 70)

        # Auto-dump if debug mode enabled
        if MEMORY_DEBUG:
            self.dump_to_file(label=f"after_{route_chosen}")

    # ===========================================================================
    # DEBUG: Full Memory Serialization for Inspection
    # ===========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize full memory state to dictionary for debugging/inspection.

        Returns complete memory state including:
        - Session metadata
        - All recent turns (full detail)
        - Compressed history
        - Entity tracking
        - Search state with last action
        - What's rendered for each LLM component
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": self.turn_count,

            # Recent turns (full detail with Phase 2 enhanced fields)
            "recent_turns": [
                {
                    "turn_id": t.turn_id,
                    "timestamp": t.timestamp.isoformat(),
                    "user_question": t.user_question,
                    "resolved_query": t.resolved_query,
                    "answer_summary": t.answer_summary,
                    "artifact_id": t.artifact_id,
                    "artifact_title": t.artifact_title,
                    "key_entities": t.key_entities,
                    "themes": t.themes,
                    # Option A: Enhanced context fields (Phase 2)
                    "key_quotes": getattr(t, 'key_quotes', []),
                    "topics_covered": getattr(t, 'topics_covered', []),
                    "notable_examples": getattr(t, 'notable_examples', []),
                }
                for t in self.recent_turns
            ],

            # Compressed history
            "compressed_summary": self.compressed_summary,

            # Entity tracking
            "entities": {
                k: {
                    "name": v.name,
                    "mention_count": v.mention_count,
                    "last_turn_index": v.last_turn_index,
                    "entity_type": v.entity_type,
                    "relevance_score": v.relevance_score(self.turn_count),
                }
                for k, v in self.entities.items()
            },

            # Themes
            "conversation_themes": self.conversation_themes,

            # Shown artifacts
            "shown_artifacts": self.shown_artifacts,

            # Exclusion window
            "excluded_artifact_window": [
                {"artifact_id": aid, "turn": turn}
                for aid, turn in self.excluded_artifact_window
            ],
            "currently_excluded_ids": list(self.get_excluded_ids()),

            # Search state (the brain)
            "search_state": {
                "current_entities": self.search_state.current_entities,
                "current_topic": self.search_state.current_topic,
                "last_time_filter": self.search_state.last_time_filter,
                "last_artifact_id": self.search_state.last_artifact_id,
                "last_artifact_title": self.search_state.last_artifact_title,
                "turns_since_entity_update": self.search_state.turns_since_entity_update,
                "last_was_followup": self.search_state.last_was_followup,
                "conversation_phase": self.search_state.conversation_phase,
                "route_pattern": self.search_state.get_route_pattern(),

                # Thread tracking (topic drift fix)
                "conversation_thread_topic": self.search_state.conversation_thread_topic,
                "topic_history": self.search_state.topic_history,
                "thread_summary": self.search_state.get_thread_summary(),

                # Conversation participants (correction pattern fix)
                "conversation_participants": self.search_state.conversation_participants,

                # Last action (critical for follow-ups)
                "last_action": {
                    "action_type": self.search_state.last_action.action_type,
                    "target_id": self.search_state.last_action.target_id,
                    "target_title": self.search_state.last_action.target_title,
                    "target_summary": self.search_state.last_action.target_summary,
                    "source_branch": self.search_state.last_action.source_branch,
                },

                # Route history
                "route_history": [
                    {
                        "turn_id": r.turn_id,
                        "route_chosen": r.route_chosen,
                        "route_confidence": r.route_confidence,
                        "query_intent": r.query_intent,
                        "outcome": r.outcome,
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in self.search_state.route_history
                ],
            },

            # What each LLM component receives (for debugging)
            "llm_contexts": {
                "router_context": self.search_state.render_for_router(self.recent_turns),
                "query_analyzer_context": self.search_state.render_for_query_analyzer(self.recent_turns),
                "small_talk_context": self.search_state.render_for_small_talk(self.recent_turns),
                "full_prompt_context": self.render_for_prompt(max_chars=50000),
                "search_state_prompt": self.search_state.render_for_prompt(),
            },
        }

    def dump_to_file(self, label: str = "") -> Optional[Path]:
        """
        Dump current memory state to a JSON file for debugging.

        Files are saved to MEMORY_DEBUG_DIR (default: debug_memory/)
        Format: {session_id}_turn{N}_{label}_{timestamp}.json

        Args:
            label: Optional label to add to filename (e.g., "after_routing")

        Returns:
            Path to the created file, or None if debug mode is disabled
        """
        if not MEMORY_DEBUG:
            return None

        try:
            MEMORY_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(tz=None).strftime("%H%M%S")
            label_part = f"_{label}" if label else ""
            filename = f"{self.session_id}_turn{self.turn_count}{label_part}_{timestamp}.json"
            filepath = MEMORY_DEBUG_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"[MEMORY DEBUG] Dumped to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"[MEMORY DEBUG] Failed to dump: {e}")
            return None


class MemoryStore:
    """Thread-safe session memory store."""

    def __init__(self):
        self._store: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """Get existing memory or create new one for session."""
        with self._lock:
            is_new = session_id not in self._store
            if is_new:
                self._store[session_id] = ConversationMemory(session_id=session_id)
                logger.info(f"[MEMORY STORE] Created new session: {session_id}")
            else:
                mem = self._store[session_id]
                logger.info(f"[MEMORY STORE] Retrieved session: {session_id} | turns={mem.turn_count}, entities={len(mem.entities)}")
            return self._store[session_id]

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """Get memory for session if exists."""
        with self._lock:
            return self._store.get(session_id)

    def reset(self, session_id: str) -> ConversationMemory:
        """Reset memory for session."""
        with self._lock:
            old_mem = self._store.get(session_id)
            old_turns = old_mem.turn_count if old_mem else 0
            self._store[session_id] = ConversationMemory(session_id=session_id)
            logger.info(f"[MEMORY STORE] Reset session: {session_id} | cleared {old_turns} turns")
            return self._store[session_id]

    def delete(self, session_id: str) -> None:
        """Delete memory for session."""
        with self._lock:
            self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._store.keys())

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Returns count removed."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0

        with self._lock:
            to_remove = [
                sid for sid, mem in self._store.items()
                if mem.created_at < cutoff
            ]
            for sid in to_remove:
                del self._store[sid]
                removed += 1

        return removed


# Singleton instance
memory_store = MemoryStore()
