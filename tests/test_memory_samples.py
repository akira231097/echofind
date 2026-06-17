import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
CONCISE MEMORY SAMPLES FOR EACH BRANCH
=======================================
Outputs clean JSON for each branch type to verify implementation.
"""

import json
from engine.memory import ConversationMemory
from engine.schemas import BranchMemoryUpdate

# Disable logging for clean output
import logging
logging.disable(logging.CRITICAL)


def get_memory_snapshot(memory: ConversationMemory) -> dict:
    """Get essential memory fields only."""
    ss = memory.search_state
    return {
        "turn_count": memory.turn_count,
        "search_state": {
            "current_entities": ss.current_entities,
            "current_topic": ss.current_topic,
            "conversation_thread_topic": ss.conversation_thread_topic,
            "topic_history": ss.topic_history,
            "conversation_participants": ss.conversation_participants,
            "conversation_phase": ss.conversation_phase,
            "route_pattern": ss.get_route_pattern(),
            "last_action": {
                "action_type": ss.last_action.action_type,
                "target_title": ss.last_action.target_title,
                "target_summary": ss.last_action.target_summary[:100] if ss.last_action.target_summary else None,
                "source_branch": ss.last_action.source_branch,
                "published_date": ss.last_action.published_date,
            }
        },
        "recent_turns": [
            {
                "turn_id": t.turn_id,
                "user_question": t.user_question[:50],
                "resolved_query": t.resolved_query[:50],
                "answer_summary": t.answer_summary[:80],
                "artifact_title": t.artifact_title,
                "key_entities": t.key_entities,
                "themes": t.themes,
                "key_quotes": getattr(t, 'key_quotes', []),
                "topics_covered": getattr(t, 'topics_covered', []),
            }
            for t in memory.recent_turns[-3:]  # Last 3 only
        ],
        "excluded_ids": list(memory.get_excluded_ids()),
    }


def apply_update(memory, update, route, confidence, intent, question, resolved):
    memory.apply_branch_memory_update(
        update=update,
        route_chosen=route,
        route_confidence=confidence,
        query_intent=intent,
        user_question=question,
        resolved_query=resolved,
    )


print("=" * 70)
print(" SAMPLE MEMORY OUTPUTS FOR EACH BRANCH")
print("=" * 70)

# =============================================================================
# SAMPLE 1: SMALL_TALK - GREETING
# =============================================================================
print("\n" + "=" * 70)
print(" BRANCH: small_talk (greeting)")
print("=" * 70)
print("\nINPUT:")
print('  Query: "Hey, what can you help me with?"')

memory = ConversationMemory(session_id="sample-001")
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="User greeted. Introduced Echo podcast discovery assistant.",
        action_type="greeting",
        action_target_id=None,
        action_target_title=None,
        published_date=None,
        entities_mentioned=[],
        topics_discussed=[],
        is_topic_shift=False,
        suggested_phase="discovery",
    ),
    route="small_talk",
    confidence=0.95,
    intent="greeting",
    question="Hey, what can you help me with?",
    resolved="Hey, what can you help me with?",
)

print("\nMEMORY AFTER:")
print(json.dumps(get_memory_snapshot(memory), indent=2))

# =============================================================================
# SAMPLE 2: CLIP_SEARCH
# =============================================================================
print("\n" + "=" * 70)
print(" BRANCH: clip_search")
print("=" * 70)
print("\nINPUT:")
print('  Query: "clips about AI safety"')

memory = ConversationMemory(session_id="sample-002")
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="Showed clip: Yudkowsky discusses AI alignment on Lex Fridman podcast.",
        action_type="clip_shown",
        action_target_id="chunk_yudkowsky_001",
        action_target_title="Lex Fridman #368: Eliezer Yudkowsky on AI Alignment",
        published_date="2023-03-27",
        entities_mentioned=["Eliezer Yudkowsky", "Lex Fridman"],
        topics_discussed=["AI safety", "alignment"],
        is_topic_shift=True,
        suggested_phase="deep_dive",
        key_quotes=["The alignment problem is unsolved"],
        topics_covered=["value alignment", "mesa-optimization"],
        notable_examples=["Paperclip maximizer"],
    ),
    route="clip_search",
    confidence=0.95,
    intent="find clips about AI safety",
    question="clips about AI safety",
    resolved="clips about AI safety",
)

print("\nMEMORY AFTER:")
print(json.dumps(get_memory_snapshot(memory), indent=2))

# =============================================================================
# SAMPLE 3: EPISODE_SEARCH
# =============================================================================
print("\n" + "=" * 70)
print(" BRANCH: episode_search")
print("=" * 70)
print("\nINPUT:")
print('  Query: "What shows has Andrew Huberman been on?"')

memory = ConversationMemory(session_id="sample-003")
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="Showed episode: Huberman on Joe Rogan Experience #1683.",
        action_type="episode_shown",
        action_target_id="episode_jre_1683",
        action_target_title="Joe Rogan Experience #1683 - Andrew Huberman",
        published_date="2021-07-27",
        entities_mentioned=["Andrew Huberman", "Joe Rogan"],
        topics_discussed=["neuroscience", "sleep"],
        is_topic_shift=True,
        suggested_phase="discovery",
        key_quotes=["Huberman discusses neural regeneration"],
        topics_covered=["sleep protocols", "dopamine"],
        notable_examples=["Stanford lab research"],
    ),
    route="episode_search",
    confidence=0.95,
    intent="find Huberman appearances",
    question="What shows has Andrew Huberman been on?",
    resolved="What shows has Andrew Huberman been on?",
)

print("\nMEMORY AFTER:")
print(json.dumps(get_memory_snapshot(memory), indent=2))

# =============================================================================
# SAMPLE 4: SMALL_TALK - EXPLANATION (after clip)
# =============================================================================
print("\n" + "=" * 70)
print(" BRANCH: small_talk (explanation after clip)")
print("=" * 70)
print("\nSETUP: User saw clip about cold exposure")
print('INPUT:')
print('  Query: "Is that 250% claim true?"')

memory = ConversationMemory(session_id="sample-004")

# First: show a clip
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="Showed Huberman clip on cold exposure boosting dopamine 250%.",
        action_type="clip_shown",
        action_target_id="chunk_cold_001",
        action_target_title="Huberman Lab: Cold Exposure",
        published_date="2024-03-15",
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["cold exposure", "dopamine"],
        is_topic_shift=True,
        suggested_phase="deep_dive",
        key_quotes=["Cold increases dopamine 250%"],
        topics_covered=["cold plunge protocol"],
        notable_examples=[],
    ),
    route="clip_search",
    confidence=0.92,
    intent="cold exposure clips",
    question="clips about cold exposure",
    resolved="clips about cold exposure",
)

# Then: explanation request
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="User asked to verify 250% dopamine claim. Provided research context.",
        action_type="explanation",
        action_target_id=None,
        action_target_title="Verification: Dopamine claim",
        published_date=None,
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["dopamine research"],
        is_topic_shift=False,  # Continuing same thread!
        suggested_phase="deep_dive",
        key_quotes=[],
        topics_covered=["dopamine measurement"],
        notable_examples=[],
    ),
    route="small_talk",
    confidence=0.92,
    intent="verify claim from clip",
    question="Is that 250% claim true?",
    resolved="Is the 250% dopamine claim from Andrew Huberman true?",
)

print("\nMEMORY AFTER (2 turns):")
print(json.dumps(get_memory_snapshot(memory), indent=2))

# =============================================================================
# SAMPLE 5: TOPIC SHIFT
# =============================================================================
print("\n" + "=" * 70)
print(" TOPIC SHIFT EXAMPLE")
print("=" * 70)
print("\nSETUP: User was exploring sleep topic, now shifts to productivity")

memory = ConversationMemory(session_id="sample-005")

# Turn 1: Sleep topic
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="Showed Huberman sleep toolkit clip.",
        action_type="clip_shown",
        action_target_id="chunk_sleep_001",
        action_target_title="Huberman Lab: Sleep Toolkit",
        published_date="2024-01-10",
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["sleep"],
        is_topic_shift=True,
        suggested_phase="discovery",
        key_quotes=[],
        topics_covered=["sleep protocol"],
        notable_examples=[],
    ),
    route="clip_search",
    confidence=0.95,
    intent="sleep clips",
    question="clips about sleep",
    resolved="clips about sleep",
)

print("\nAFTER TURN 1 (sleep):")
print(f"  thread_topic: {memory.search_state.conversation_thread_topic}")
print(f"  topic_history: {memory.search_state.topic_history}")

# Turn 2: Caffeine (same thread)
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="Showed caffeine and sleep clip.",
        action_type="clip_shown",
        action_target_id="chunk_caffeine_001",
        action_target_title="Huberman: Caffeine & Sleep",
        published_date="2024-01-15",
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["caffeine", "sleep"],
        is_topic_shift=False,  # Same thread!
        suggested_phase="deep_dive",
        key_quotes=[],
        topics_covered=[],
        notable_examples=[],
    ),
    route="clip_search",
    confidence=0.92,
    intent="caffeine and sleep",
    question="what about caffeine?",
    resolved="what about caffeine and sleep?",
)

print("\nAFTER TURN 2 (caffeine - same thread):")
print(f"  thread_topic: {memory.search_state.conversation_thread_topic}")
print(f"  topic_history: {memory.search_state.topic_history}")

# Turn 3: TOPIC SHIFT to productivity
apply_update(
    memory,
    BranchMemoryUpdate(
        turn_summary="Topic shift! Showed Tim Ferriss productivity clip.",
        action_type="clip_shown",
        action_target_id="chunk_productivity_001",
        action_target_title="Tim Ferriss: Deep Work",
        published_date="2024-02-01",
        entities_mentioned=["Tim Ferriss"],
        topics_discussed=["productivity"],
        is_topic_shift=True,  # NEW THREAD!
        suggested_phase="discovery",
        key_quotes=[],
        topics_covered=["time blocking"],
        notable_examples=[],
    ),
    route="clip_search",
    confidence=0.88,
    intent="productivity clips",
    question="Actually, show me productivity clips",
    resolved="Show me clips about productivity",
)

print("\nAFTER TURN 3 (TOPIC SHIFT to productivity):")
print(f"  thread_topic: {memory.search_state.conversation_thread_topic}")
print(f"  topic_history: {memory.search_state.topic_history}")
print(f"  current_entities: {memory.search_state.current_entities}")
print("\n  NOTE: Thread topic RESET, history CLEARED, entities REPLACED")

print("\n" + "=" * 70)
print(" SUMMARY: KEY FIELDS TO VERIFY IN reference-memory-service")
print("=" * 70)
print("""
1. BranchMemoryUpdate fields:
   - turn_summary (max 500 chars)
   - action_type: clip_shown | episode_shown | explanation | greeting | error
   - action_target_id, action_target_title
   - published_date (YYYY-MM-DD)
   - entities_mentioned (max 10)
   - topics_discussed (max 5)
   - is_topic_shift (controls thread reset)
   - suggested_phase
   - key_quotes, topics_covered, notable_examples (Phase 2)

2. SearchState updates:
   - current_entities (merged, max 5)
   - current_topic / conversation_thread_topic
   - topic_history (appended or reset based on is_topic_shift)
   - conversation_participants (PEOPLE only)
   - last_action (CRITICAL for follow-ups)
   - route_history (last 5 decisions)

3. Exclusion window:
   - Shown clips/episodes excluded for 5 turns
""")
