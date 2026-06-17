import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
COMPREHENSIVE AGENT MEMORY TEST SCRIPT
=====================================

This script runs real queries through each branch (small_talk, episode_search, clip_search)
and prints the FULL memory state after each response.

Use this to verify that the agent implementation in reference-memory-service matches the original.

EVALUATION METRIC: The new agent must be functionally equivalent to this implementation.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

# Import the actual memory and schema classes
from engine.memory import (
    ConversationMemory,
    SearchState,
    LastActionContext,
    ConversationTurn,
    RouteRecord,
)
from engine.schemas import BranchMemoryUpdate, RouterOutput


def print_separator(title: str, char: str = "="):
    """Print a formatted separator."""
    print(f"\n{char * 80}")
    print(f" {title}")
    print(f"{char * 80}\n")


def print_memory_full(memory: ConversationMemory, label: str = ""):
    """Print the FULL memory state in a readable format."""
    print_separator(f"FULL MEMORY STATE {label}")

    mem_dict = memory.to_dict()

    # 1. Session Metadata
    print("1. SESSION METADATA:")
    print(f"   - Session ID: {mem_dict['session_id']}")
    print(f"   - Created At: {mem_dict['created_at']}")
    print(f"   - Turn Count: {mem_dict['turn_count']}")
    print()

    # 2. Search State (The Brain)
    print("2. SEARCH STATE (The Brain):")
    ss = mem_dict['search_state']
    print(f"   - Current Entities: {ss['current_entities']}")
    print(f"   - Current Topic: {ss['current_topic']}")
    print(f"   - Thread Topic: {ss['conversation_thread_topic']}")
    print(f"   - Topic History: {ss['topic_history']}")
    print(f"   - Conversation Participants: {ss['conversation_participants']}")
    print(f"   - Conversation Phase: {ss['conversation_phase']}")
    print(f"   - Route Pattern: {ss['route_pattern']}")
    print(f"   - Last Was Followup: {ss['last_was_followup']}")
    print()

    # 3. Last Action Context
    print("3. LAST ACTION CONTEXT (Critical for follow-ups):")
    la = ss['last_action']
    print(f"   - Action Type: {la['action_type']}")
    print(f"   - Target ID: {la['target_id']}")
    print(f"   - Target Title: {la['target_title']}")
    print(f"   - Target Summary: {la['target_summary'][:100] if la['target_summary'] else 'N/A'}...")
    print(f"   - Source Branch: {la['source_branch']}")
    print()

    # 4. Route History
    print("4. ROUTE HISTORY (Last 5 decisions):")
    for i, rh in enumerate(ss['route_history']):
        print(f"   [{i}] Turn {rh['turn_id']}: {rh['route_chosen']} (conf={rh['route_confidence']:.2f}) - {rh['query_intent'][:50]}...")
    if not ss['route_history']:
        print("   (empty)")
    print()

    # 5. Recent Turns
    print("5. RECENT TURNS (Full Detail):")
    for i, turn in enumerate(mem_dict['recent_turns']):
        print(f"   [{turn['turn_id']}]")
        print(f"       User Question: \"{turn['user_question'][:60]}...\"")
        print(f"       Resolved Query: \"{turn['resolved_query'][:60]}...\"")
        print(f"       Answer Summary: \"{turn['answer_summary'][:80]}...\"")
        print(f"       Artifact Title: {turn['artifact_title']}")
        print(f"       Key Entities: {turn['key_entities']}")
        print(f"       Themes: {turn['themes']}")
        # Phase 2 enhanced fields
        if turn.get('key_quotes'):
            print(f"       Key Quotes: {turn['key_quotes'][:2]}")
        if turn.get('topics_covered'):
            print(f"       Topics Covered: {turn['topics_covered']}")
        if turn.get('notable_examples'):
            print(f"       Notable Examples: {turn['notable_examples']}")
        print()
    if not mem_dict['recent_turns']:
        print("   (empty)")
    print()

    # 6. Entity Tracking
    print("6. ENTITY TRACKING:")
    for entity_key, entity_data in list(mem_dict['entities'].items())[:10]:
        print(f"   - {entity_data['name']}: mentions={entity_data['mention_count']}, relevance={entity_data['relevance_score']:.2f}")
    if not mem_dict['entities']:
        print("   (empty)")
    print()

    # 7. Conversation Themes
    print("7. CONVERSATION THEMES:")
    print(f"   {mem_dict['conversation_themes']}")
    print()

    # 8. Exclusion Window
    print("8. EXCLUSION WINDOW (artifacts not to show again):")
    print(f"   Currently Excluded: {mem_dict['currently_excluded_ids']}")
    print()

    # 9. LLM Contexts (what each component receives)
    print("9. LLM CONTEXTS (what each component receives):")
    print()
    print("   --- ROUTER CONTEXT ---")
    # Replace special characters for Windows console compatibility
    router_ctx = mem_dict['llm_contexts']['router_context'][:500].replace('→', '->').replace('│', '|')
    print(router_ctx)
    print("   ...")
    print()
    print("   --- QUERY ANALYZER CONTEXT ---")
    analyzer_ctx = mem_dict['llm_contexts']['query_analyzer_context'][:500].replace('→', '->').replace('│', '|')
    print(analyzer_ctx)
    print("   ...")
    print()


def simulate_branch_memory_update(
    memory: ConversationMemory,
    update: BranchMemoryUpdate,
    route: str,
    confidence: float,
    query_intent: str,
    user_question: str,
    resolved_query: str,
):
    """Simulate a branch memory update (same as agent would do)."""
    memory.apply_branch_memory_update(
        update=update,
        route_chosen=route,
        route_confidence=confidence,
        query_intent=query_intent,
        user_question=user_question,
        resolved_query=resolved_query,
    )


# =============================================================================
# TEST CASE 1: SMALL_TALK BRANCH - Greeting
# =============================================================================

def test_small_talk_greeting():
    """Test case: User says hello (greeting)."""
    print_separator("TEST CASE 1: SMALL_TALK BRANCH - GREETING", "=")

    # Create fresh memory
    memory = ConversationMemory(session_id="test-session-001")

    print("INPUT:")
    print("  Query: 'Hey, what can you help me with?'")
    print("  Route: small_talk")
    print("  Sub-intent: greeting")
    print()

    # Simulate the memory update that small_talk branch would produce
    memory_update = BranchMemoryUpdate(
        turn_summary="User greeted the assistant. Responded with introduction about Echo podcast discovery.",
        action_type="greeting",
        action_target_id=None,
        action_target_title=None,
        published_date=None,
        entities_mentioned=[],
        topics_discussed=[],
        is_topic_shift=False,
        suggested_phase="discovery",
        key_quotes=[],
        topics_covered=[],
        notable_examples=[],
    )

    # Apply the memory update
    simulate_branch_memory_update(
        memory=memory,
        update=memory_update,
        route="small_talk",
        confidence=0.95,
        query_intent="greeting and capability inquiry",
        user_question="Hey, what can you help me with?",
        resolved_query="Hey, what can you help me with?",
    )

    # Print full memory state
    print_memory_full(memory, "AFTER SMALL_TALK GREETING")

    return memory


# =============================================================================
# TEST CASE 2: SMALL_TALK BRANCH - Explanation
# =============================================================================

def test_small_talk_explanation(memory: ConversationMemory):
    """Test case: User asks for explanation after seeing a clip."""
    print_separator("TEST CASE 2: SMALL_TALK BRANCH - EXPLANATION", "=")

    print("SETUP: First, simulate showing a clip to the user...")
    print()

    # First, simulate that a clip was shown (from clip_search)
    clip_update = BranchMemoryUpdate(
        turn_summary="Showed clip where Huberman explains how cold exposure boosts dopamine by 250% and improves focus.",
        action_type="clip_shown",
        action_target_id="chunk_huberman_cold_001",
        action_target_title="Huberman Lab: Cold Exposure Benefits",
        published_date="2024-03-15",
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["cold exposure", "dopamine", "neuroscience"],
        is_topic_shift=True,
        suggested_phase="deep_dive",
        key_quotes=["Cold exposure can increase dopamine by 250%", "The benefits last for hours"],
        topics_covered=["cold plunge protocol", "dopamine baseline", "focus enhancement"],
        notable_examples=["Huberman's own morning cold exposure routine"],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=clip_update,
        route="clip_search",
        confidence=0.92,
        query_intent="find clips about cold exposure benefits",
        user_question="Show me clips about cold exposure benefits",
        resolved_query="Show me clips about cold exposure benefits",
    )

    print("CLIP SHOWN: 'Huberman Lab: Cold Exposure Benefits'")
    print()

    # Now user asks for explanation
    print("INPUT:")
    print("  Query: 'Is that 250% claim actually true?'")
    print("  Route: small_talk")
    print("  Sub-intent: explanation")
    print()

    explanation_update = BranchMemoryUpdate(
        turn_summary="User asked to verify the 250% dopamine claim from cold exposure. Provided explanation with research context.",
        action_type="explanation",
        action_target_id=None,
        action_target_title="Verification: Cold exposure dopamine claim",
        published_date=None,
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["dopamine research", "cold exposure science"],
        is_topic_shift=False,  # Continuing same topic
        suggested_phase="deep_dive",
        key_quotes=["Based on research by Šrámek et al. (2000)"],
        topics_covered=["dopamine measurement", "cold exposure duration", "study methodology"],
        notable_examples=[],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=explanation_update,
        route="small_talk",
        confidence=0.92,
        query_intent="verify dopamine claim from Huberman clip",
        user_question="Is that 250% claim actually true?",
        resolved_query="Is the 250% dopamine increase claim from Andrew Huberman actually true?",
    )

    print_memory_full(memory, "AFTER SMALL_TALK EXPLANATION")

    return memory


# =============================================================================
# TEST CASE 3: EPISODE_SEARCH BRANCH
# =============================================================================

def test_episode_search():
    """Test case: User searches for episodes by person."""
    print_separator("TEST CASE 3: EPISODE_SEARCH BRANCH", "=")

    # Create fresh memory
    memory = ConversationMemory(session_id="test-session-002")

    print("INPUT:")
    print("  Query: 'What shows has Andrew Huberman been on?'")
    print("  Route: episode_search")
    print("  Sub-intent: guest appearances")
    print()

    # Simulate the memory update that episode_search branch would produce
    memory_update = BranchMemoryUpdate(
        turn_summary="Showed episode: Huberman appeared on Joe Rogan Experience #1683 discussing neuroscience, sleep, and performance optimization.",
        action_type="episode_shown",
        action_target_id="episode_jre_1683_huberman",
        action_target_title="Joe Rogan Experience #1683 - Andrew Huberman",
        published_date="2021-07-27",
        entities_mentioned=["Andrew Huberman", "Joe Rogan"],
        topics_discussed=["neuroscience", "sleep optimization", "performance"],
        is_topic_shift=True,
        suggested_phase="discovery",
        key_quotes=[
            "Huberman discusses his research on neural regeneration",
            "Joe and Andrew explore the science of habit formation"
        ],
        topics_covered=["sleep protocols", "dopamine systems", "light exposure", "habit formation"],
        notable_examples=["Huberman's Stanford lab research", "Joe's experience with cold plunges"],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=memory_update,
        route="episode_search",
        confidence=0.95,
        query_intent="find episode appearances of Andrew Huberman",
        user_question="What shows has Andrew Huberman been on?",
        resolved_query="What shows has Andrew Huberman been on?",
    )

    print_memory_full(memory, "AFTER EPISODE_SEARCH")

    # Now test a follow-up
    print_separator("FOLLOW-UP: 'Show me more of his appearances'", "-")

    print("INPUT:")
    print("  Query: 'Show me more of his appearances'")
    print("  Route: episode_search (continuation)")
    print()

    followup_update = BranchMemoryUpdate(
        turn_summary="Showed another Huberman episode: Rich Roll Podcast discussing protocols for optimizing brain function.",
        action_type="episode_shown",
        action_target_id="episode_richroll_huberman",
        action_target_title="Rich Roll Podcast - Andrew Huberman: Optimize Your Brain",
        published_date="2022-01-15",
        entities_mentioned=["Andrew Huberman", "Rich Roll"],
        topics_discussed=["brain optimization", "neuroplasticity"],
        is_topic_shift=False,  # Same person, different episode
        suggested_phase="discovery",
        key_quotes=["Neuroplasticity remains throughout life"],
        topics_covered=["brain protocols", "focus techniques"],
        notable_examples=[],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=followup_update,
        route="episode_search",
        confidence=0.92,
        query_intent="more episode appearances of Huberman",
        user_question="Show me more of his appearances",
        resolved_query="Show me more of Andrew Huberman's podcast appearances",
    )

    print_memory_full(memory, "AFTER EPISODE_SEARCH FOLLOW-UP")

    return memory


# =============================================================================
# TEST CASE 4: CLIP_SEARCH BRANCH
# =============================================================================

def test_clip_search():
    """Test case: User searches for clips by topic."""
    print_separator("TEST CASE 4: CLIP_SEARCH BRANCH", "=")

    # Create fresh memory
    memory = ConversationMemory(session_id="test-session-003")

    print("INPUT:")
    print("  Query: 'Show me clips about AI safety and alignment'")
    print("  Route: clip_search")
    print()

    # Simulate the memory update that clip_search branch would produce
    memory_update = BranchMemoryUpdate(
        turn_summary="Showed clip where Eliezer Yudkowsky discusses AI alignment challenges and existential risks on Lex Fridman podcast.",
        action_type="clip_shown",
        action_target_id="chunk_yudkowsky_ai_001",
        action_target_title="Lex Fridman #368: Eliezer Yudkowsky on AI Alignment",
        published_date="2023-03-27",
        entities_mentioned=["Eliezer Yudkowsky", "Lex Fridman"],
        topics_discussed=["AI safety", "alignment problem", "existential risk"],
        is_topic_shift=True,
        suggested_phase="deep_dive",
        key_quotes=[
            "The alignment problem is not solved by making AI smarter",
            "We need to solve alignment before we have powerful AI"
        ],
        topics_covered=["value alignment", "corrigibility", "mesa-optimization", "RLHF limitations"],
        notable_examples=["Paperclip maximizer thought experiment", "GPT capability concerns"],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=memory_update,
        route="clip_search",
        confidence=0.95,
        query_intent="find clips about AI safety and alignment",
        user_question="Show me clips about AI safety and alignment",
        resolved_query="Show me clips about AI safety and alignment",
    )

    print_memory_full(memory, "AFTER CLIP_SEARCH")

    # Follow-up: User asks about the person
    print_separator("FOLLOW-UP: 'Who is this guy?'", "-")

    print("INPUT:")
    print("  Query: 'Who is this guy?'")
    print("  Route: small_talk")
    print("  Sub-intent: contextual_knowledge")
    print()

    contextual_update = BranchMemoryUpdate(
        turn_summary="User asked about Eliezer Yudkowsky. Provided background on his role in AI safety research and MIRI.",
        action_type="explanation",
        action_target_id=None,
        action_target_title="Background: Eliezer Yudkowsky",
        published_date=None,
        entities_mentioned=["Eliezer Yudkowsky", "MIRI"],
        topics_discussed=["AI safety research", "MIRI organization"],
        is_topic_shift=False,
        suggested_phase="deep_dive",
        key_quotes=[],
        topics_covered=["rationalist community", "AI risk research history"],
        notable_examples=[],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=contextual_update,
        route="small_talk",
        confidence=0.95,
        query_intent="asking about Eliezer Yudkowsky from clip",
        user_question="Who is this guy?",
        resolved_query="Who is Eliezer Yudkowsky?",  # Pronoun resolved
    )

    print_memory_full(memory, "AFTER CONTEXTUAL_KNOWLEDGE FOLLOW-UP")

    # Follow-up: User wants more clips
    print_separator("FOLLOW-UP: 'Show me more about this topic'", "-")

    print("INPUT:")
    print("  Query: 'Show me more about this topic'")
    print("  Route: clip_search (continuation)")
    print()

    more_clips_update = BranchMemoryUpdate(
        turn_summary="Showed another AI safety clip: Sam Harris discussing AI risk scenarios and the control problem.",
        action_type="clip_shown",
        action_target_id="chunk_harris_ai_002",
        action_target_title="Sam Harris: AI and the Future of Intelligence",
        published_date="2023-06-15",
        entities_mentioned=["Sam Harris"],
        topics_discussed=["AI risk", "control problem"],
        is_topic_shift=False,  # Same topic thread
        suggested_phase="deep_dive",
        key_quotes=["The control problem may be the most important problem we face"],
        topics_covered=["superintelligence", "value loading problem"],
        notable_examples=[],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=more_clips_update,
        route="clip_search",
        confidence=0.90,
        query_intent="more clips about AI safety topic",
        user_question="Show me more about this topic",
        resolved_query="Show me more clips about AI safety and alignment",  # Topic resolved from thread
    )

    print_memory_full(memory, "AFTER CLIP_SEARCH CONTINUATION")

    return memory


# =============================================================================
# TEST CASE 5: MULTI-TURN CONVERSATION WITH TOPIC SHIFT
# =============================================================================

def test_multi_turn_with_topic_shift():
    """Test case: Multi-turn conversation with explicit topic shift."""
    print_separator("TEST CASE 5: MULTI-TURN WITH TOPIC SHIFT", "=")

    memory = ConversationMemory(session_id="test-session-004")

    # Turn 1: Initial clip search
    print("TURN 1: User searches for sleep clips")
    print("  Query: 'clips about sleep optimization'")
    print()

    turn1_update = BranchMemoryUpdate(
        turn_summary="Showed clip about sleep optimization from Huberman Lab discussing sleep protocols.",
        action_type="clip_shown",
        action_target_id="chunk_sleep_001",
        action_target_title="Huberman Lab: Sleep Toolkit",
        published_date="2024-01-10",
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["sleep", "circadian rhythm"],
        is_topic_shift=True,
        suggested_phase="discovery",
        key_quotes=["Morning light exposure is crucial for sleep"],
        topics_covered=["sleep protocol", "light exposure", "temperature"],
        notable_examples=["Huberman's personal sleep routine"],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=turn1_update,
        route="clip_search",
        confidence=0.95,
        query_intent="clips about sleep optimization",
        user_question="clips about sleep optimization",
        resolved_query="clips about sleep optimization",
    )

    # Turn 2: Follow-up about caffeine
    print("TURN 2: User asks related question")
    print("  Query: 'what about caffeine and sleep?'")
    print()

    turn2_update = BranchMemoryUpdate(
        turn_summary="Showed clip about caffeine's effect on sleep - recommends stopping 8-10 hours before bed.",
        action_type="clip_shown",
        action_target_id="chunk_caffeine_001",
        action_target_title="Huberman Lab: Caffeine & Sleep",
        published_date="2024-01-15",
        entities_mentioned=["Andrew Huberman"],
        topics_discussed=["caffeine", "sleep quality"],
        is_topic_shift=False,  # Still in sleep thread
        suggested_phase="deep_dive",
        key_quotes=["Caffeine has a half-life of 5-6 hours"],
        topics_covered=["caffeine timing", "adenosine"],
        notable_examples=[],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=turn2_update,
        route="clip_search",
        confidence=0.92,
        query_intent="caffeine and sleep relationship",
        user_question="what about caffeine and sleep?",
        resolved_query="what about caffeine and sleep optimization?",
    )

    # Turn 3: TOPIC SHIFT - completely new topic
    print("TURN 3: User explicitly shifts topic")
    print("  Query: 'Actually, show me something about productivity instead'")
    print()

    turn3_update = BranchMemoryUpdate(
        turn_summary="Topic shift! Showed clip about productivity from Tim Ferriss discussing deep work strategies.",
        action_type="clip_shown",
        action_target_id="chunk_productivity_001",
        action_target_title="Tim Ferriss: Deep Work Strategies",
        published_date="2024-02-01",
        entities_mentioned=["Tim Ferriss"],
        topics_discussed=["productivity", "deep work"],
        is_topic_shift=True,  # Explicit new topic!
        suggested_phase="discovery",
        key_quotes=["Deep work is the superpower of the 21st century"],
        topics_covered=["time blocking", "focus strategies", "batching tasks"],
        notable_examples=["Tim's morning routine"],
    )

    simulate_branch_memory_update(
        memory=memory,
        update=turn3_update,
        route="clip_search",
        confidence=0.88,
        query_intent="topic shift to productivity",
        user_question="Actually, show me something about productivity instead",
        resolved_query="Show me clips about productivity",
    )

    print_memory_full(memory, "AFTER TOPIC SHIFT")

    print("\nNOTE: Observe how the topic shift:")
    print("  - Reset conversation_thread_topic to 'productivity'")
    print("  - Cleared topic_history and started fresh")
    print("  - Updated current_entities to new person (Tim Ferriss)")
    print("  - Previous sleep-related context is now in recent_turns history")

    return memory


# =============================================================================
# MAIN: RUN ALL TEST CASES
# =============================================================================

def run_all_tests():
    """Run all test cases and print summary."""
    print_separator("ECHOFIND MEMORY TEST SUITE", "*")
    print("This test suite demonstrates the full memory flow for each branch.")
    print("Use this to verify reference-memory-service implementation matches this behavior.")
    print()

    print("Running tests...")
    print()

    # Test 1: Small Talk - Greeting
    memory1 = test_small_talk_greeting()

    # Test 2: Small Talk - Explanation (continues from Test 1 context)
    memory2 = test_small_talk_explanation(memory1)

    # Test 3: Episode Search
    memory3 = test_episode_search()

    # Test 4: Clip Search with follow-ups
    memory4 = test_clip_search()

    # Test 5: Multi-turn with topic shift
    memory5 = test_multi_turn_with_topic_shift()

    print_separator("TEST SUITE COMPLETE", "*")
    print("All test cases executed successfully.")
    print()
    print("SUMMARY OF KEY BEHAVIORS TO VERIFY:")
    print("="*60)
    print()
    print("1. SMALL_TALK GREETING:")
    print("   - action_type = 'greeting'")
    print("   - No entities/topics stored")
    print("   - suggested_phase = 'discovery'")
    print()
    print("2. SMALL_TALK EXPLANATION:")
    print("   - action_type = 'explanation'")
    print("   - is_topic_shift = False (continues context)")
    print("   - Entities from previous context preserved")
    print()
    print("3. EPISODE_SEARCH:")
    print("   - action_type = 'episode_shown'")
    print("   - published_date populated")
    print("   - Entities = [guest, host]")
    print("   - is_topic_shift = True (new person/show)")
    print()
    print("4. CLIP_SEARCH:")
    print("   - action_type = 'clip_shown'")
    print("   - published_date populated")
    print("   - key_quotes, topics_covered, notable_examples filled")
    print("   - Exclusion window updated")
    print()
    print("5. TOPIC SHIFT BEHAVIOR:")
    print("   - is_topic_shift = True resets thread topic")
    print("   - Topic history cleared and restarted")
    print("   - Conversation participants reset")
    print()
    print("6. CONTINUATION/FOLLOW-UP:")
    print("   - is_topic_shift = False preserves thread")
    print("   - Topic history appended (not replaced)")
    print("   - Pronouns resolve from context")
    print()


# =============================================================================
# MEMORY UPDATE SCHEMA DOCUMENTATION
# =============================================================================

def print_schema_documentation():
    """Print the BranchMemoryUpdate schema for reference."""
    print_separator("BRANCH MEMORY UPDATE SCHEMA", "=")
    print("""
Every branch MUST return a BranchMemoryUpdate with these fields:

REQUIRED FIELDS:
----------------
turn_summary: str (max 500 chars)
    - Summary of what happened - user query + what was found
    - Example: "Showed clip where Huberman explains cold exposure benefits"

action_type: str
    - One of: 'clip_shown', 'episode_shown', 'explanation', 'greeting', 'error'

OPTIONAL FIELDS:
----------------
action_target_id: Optional[str]
    - Chunk ID, Episode ID, or None

action_target_title: Optional[str]
    - Human-readable title of shown content

published_date: Optional[str]
    - Publication date in YYYY-MM-DD format

entities_mentioned: List[str] (max 10)
    - Key entities mentioned this turn
    - Example: ["Andrew Huberman", "Joe Rogan"]

topics_discussed: List[str] (max 5)
    - Topics/themes discussed
    - Example: ["sleep optimization", "circadian rhythm"]

is_topic_shift: bool
    - True if user started a completely new topic
    - Controls whether thread topic is reset

suggested_phase: Optional[str]
    - One of: 'discovery', 'deep_dive', 'comparison', 'idle'

PHASE 2 ENHANCED FIELDS:
------------------------
key_quotes: List[str] (max 3)
    - Memorable quotes from content
    - Helps resolve "that quote" queries

topics_covered: List[str] (max 5)
    - Specific subtopics covered
    - Helps resolve "that topic" queries

notable_examples: List[str] (max 3)
    - Notable examples, stories, or highlights
    - Helps resolve "that example" queries
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--schema":
        print_schema_documentation()
    else:
        run_all_tests()
        print()
        print("TIP: Run with --schema flag to see BranchMemoryUpdate schema documentation")
