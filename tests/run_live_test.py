import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
REAL AGENT TEST - Runs actual queries through each branch
==========================================================

This script runs REAL queries through the agent with LLM calls and database access,
then prints the FULL memory for verification.

Usage:
    python run_real_agent_test.py

Set environment variables or use .env file for API keys.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Configure logging - show key info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)

import openai
from pinecone import Pinecone

from engine.agent import EchoFindAgent
from engine.memory import memory_store, ConversationMemory
from engine.small_talk import init_grounding_client
from config import OPENAI_API_KEY, PINECONE_API_KEY, GEMINI_API_KEY


def load_entity_data():
    """Load unique authors, personalities, and shows from JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "data", "entities.sample.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return (
            data.get("unique_personalities", []),
            data.get("unique_authors", []),
            data.get("unique_shows", []),
        )
    except Exception as e:
        logger.error(f"Failed to load entity data: {e}")
        return [], [], []


def create_clients():
    """Create API clients."""
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    gemini_client = openai.OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    return openai_client, pinecone_client, gemini_client


def get_full_memory_json(memory: ConversationMemory) -> dict:
    """Extract full memory state as JSON-serializable dict."""
    return memory.to_dict()


def print_memory_compact(memory: ConversationMemory, label: str):
    """Print memory in a compact but complete format."""
    mem = memory.to_dict()
    ss = mem["search_state"]

    print(f"\n{'='*70}")
    print(f" MEMORY AFTER: {label}")
    print(f"{'='*70}")

    print(f"\n[SEARCH STATE]")
    print(f"  current_entities: {ss['current_entities']}")
    print(f"  current_topic: {ss['current_topic']}")
    print(f"  conversation_thread_topic: {ss['conversation_thread_topic']}")
    print(f"  topic_history: {ss['topic_history']}")
    print(f"  conversation_participants: {ss['conversation_participants']}")
    print(f"  conversation_phase: {ss['conversation_phase']}")
    print(f"  route_pattern: {ss['route_pattern']}")

    print(f"\n[LAST ACTION]")
    la = ss["last_action"]
    print(f"  action_type: {la['action_type']}")
    print(f"  target_title: {la['target_title']}")
    print(f"  target_summary: {la['target_summary'][:100] if la['target_summary'] else None}...")
    print(f"  source_branch: {la['source_branch']}")
    print(f"  published_date: {la.get('published_date')}")

    print(f"\n[ROUTE HISTORY]")
    for rh in ss["route_history"]:
        intent = rh['query_intent'] or "N/A"
        print(f"  - {rh['turn_id']}: {rh['route_chosen']} (conf={rh['route_confidence']:.2f}) | {intent[:40]}...")

    print(f"\n[RECENT TURNS]")
    for turn in mem["recent_turns"][-3:]:
        print(f"  [{turn['turn_id']}]")
        print(f"    Q: {turn['user_question'][:50]}...")
        print(f"    Resolved: {turn['resolved_query'][:50]}...")
        print(f"    Summary: {turn['answer_summary'][:60]}...")
        print(f"    Artifact: {turn['artifact_title']}")
        print(f"    Entities: {turn['key_entities']}")
        print(f"    Themes: {turn['themes']}")
        if turn.get("key_quotes"):
            print(f"    Key Quotes: {turn['key_quotes']}")
        if turn.get("topics_covered"):
            print(f"    Topics Covered: {turn['topics_covered']}")

    print(f"\n[EXCLUSION WINDOW]")
    print(f"  excluded_ids: {mem['currently_excluded_ids']}")

    print(f"\n{'='*70}\n")


async def run_query(agent: EchoFindAgent, session_id: str, query: str) -> dict:
    """Run a single query and return the final response data."""
    logger.info(f"\n>>> QUERY: \"{query}\"")

    final_data = None
    async for update in agent.ask_streaming(session_id, query):
        if update.stage == "complete":
            final_data = update.data
        elif update.stage == "error":
            logger.error(f"Error: {update.data}")
            return None

    if final_data:
        logger.info(f"<<< RESPONSE: {final_data.get('answer', '')[:100]}...")
        logger.info(f"    Branch: {final_data.get('branch', 'unknown')}")
        logger.info(f"    Confidence: {final_data.get('confidence', 0):.2f}")

    return final_data


async def test_small_talk_greeting(agent: EchoFindAgent):
    """Test small_talk branch with greeting."""
    print("\n" + "#" * 70)
    print(" TEST: SMALL_TALK - GREETING")
    print("#" * 70)

    session_id = f"test-greeting-{datetime.now().strftime('%H%M%S')}"

    # Run query
    result = await run_query(agent, session_id, "Hey, what can you help me with?")

    # Get memory
    memory = memory_store.get(session_id)
    if memory:
        print_memory_compact(memory, "small_talk GREETING")
        return memory.to_dict()
    return None


async def test_clip_search(agent: EchoFindAgent):
    """Test clip_search branch."""
    print("\n" + "#" * 70)
    print(" TEST: CLIP_SEARCH")
    print("#" * 70)

    session_id = f"test-clip-{datetime.now().strftime('%H%M%S')}"

    # Run query
    result = await run_query(agent, session_id, "Show me clips about artificial intelligence")

    # Get memory
    memory = memory_store.get(session_id)
    if memory:
        print_memory_compact(memory, "clip_search")
        return memory.to_dict()
    return None


async def test_episode_search(agent: EchoFindAgent):
    """Test episode_search branch."""
    print("\n" + "#" * 70)
    print(" TEST: EPISODE_SEARCH")
    print("#" * 70)

    session_id = f"test-episode-{datetime.now().strftime('%H%M%S')}"

    # Run query
    result = await run_query(agent, session_id, "What shows has Joe Rogan done recently?")

    # Get memory
    memory = memory_store.get(session_id)
    if memory:
        print_memory_compact(memory, "episode_search")
        return memory.to_dict()
    return None


async def test_multi_turn_with_explanation(agent: EchoFindAgent):
    """Test multi-turn conversation with clip -> explanation."""
    print("\n" + "#" * 70)
    print(" TEST: MULTI-TURN (clip_search -> small_talk explanation)")
    print("#" * 70)

    session_id = f"test-multiturn-{datetime.now().strftime('%H%M%S')}"

    # Turn 1: Clip search
    print("\n--- TURN 1: Clip Search ---")
    result1 = await run_query(agent, session_id, "clips about sleep optimization")

    memory = memory_store.get(session_id)
    if memory:
        print_memory_compact(memory, "TURN 1 - clip_search")

    # Turn 2: Follow-up explanation
    print("\n--- TURN 2: Explanation Request ---")
    result2 = await run_query(agent, session_id, "Is that actually effective?")

    memory = memory_store.get(session_id)
    if memory:
        print_memory_compact(memory, "TURN 2 - small_talk explanation")
        return memory.to_dict()
    return None


async def main():
    """Main test runner."""
    print("=" * 70)
    print(" REAL AGENT TEST - Running queries through each branch")
    print("=" * 70)
    print()

    # Load entity data
    logger.info("Loading entity data...")
    unique_personalities, unique_authors, unique_shows = load_entity_data()
    logger.info(f"Loaded {len(unique_personalities)} personalities, {len(unique_authors)} authors")

    # Create clients
    logger.info("Creating API clients...")
    openai_client, pinecone_client, gemini_client = create_clients()

    # Initialize grounding
    init_grounding_client()

    # Create agent
    logger.info("Initializing EchoFindAgent...")
    agent = EchoFindAgent(
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        gemini_client=gemini_client,
        unique_personalities=unique_personalities,
        unique_authors=unique_authors,
        unique_shows=unique_shows,
        config={
            "pinecone_k": 100,
            "target_per_query": 30,
            "max_chunks_rerank": 35,
            "max_chunks_selection": 21,
            "reranker_top_n": 35,
        }
    )
    logger.info("Agent ready!\n")

    # Store all results
    all_results = {}

    # Run tests
    try:
        # Test 1: Small Talk Greeting
        all_results["small_talk_greeting"] = await test_small_talk_greeting(agent)

        # Test 2: Clip Search
        all_results["clip_search"] = await test_clip_search(agent)

        # Test 3: Episode Search
        all_results["episode_search"] = await test_episode_search(agent)

        # Test 4: Multi-turn with explanation
        all_results["multi_turn_explanation"] = await test_multi_turn_with_explanation(agent)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

    # Save all results to JSON file
    output_file = "agent_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] Full memory dumps saved to: {output_file}")

    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)
    print("\nKey fields to verify in reference-memory-service:")
    print("  1. search_state.last_action - CRITICAL for follow-ups")
    print("  2. search_state.current_entities - for pronoun resolution")
    print("  3. search_state.conversation_thread_topic - for 'this topic' resolution")
    print("  4. route_history - for continuation patterns")
    print("  5. recent_turns with key_quotes, topics_covered - Phase 2 fields")
    print()


if __name__ == "__main__":
    asyncio.run(main())
