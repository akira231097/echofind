"""
Main Chatbot Agent - Orchestrates the 7-step RAG pipeline with memory.

This is the primary entry point for chatbot queries.
Optimized for streaming and parallel execution.

Implements multi-bucket time search strategy from reference pipeline.
"""

import asyncio
import time
import logging
import re
import numpy as np
from datetime import datetime, timezone, timedelta, date
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass

from engine.memory import memory_store, ConversationMemory, MAX_RECENT_TURNS
from engine.query_analyzer import analyze_query_with_memory
from engine.selection import select_and_update_memory
from engine.schemas import ChatResponse, PipelineStage, RouterOutput
from engine.recommendations import (
    generate_recommendations,
    store_recommendations,
    get_recommendation,
)
from engine.episode_recommendations import (
    generate_episode_recommendations,
    store_episode_recommendations,
    get_episode_recommendation,
)

# Import router and branch handlers
from engine.router import route_query, CONFIDENCE_THRESHOLD
from engine.small_talk import handle_small_talk, init_grounding_client
from engine.episode_search import handle_episode_search

# Import existing pipeline components
from retrieval.data_fetcher import (
    concurrent_pinecone_search,
    combine_pinecone_results,
    concurrent_embedding_generation,
    concurrent_sparse_embedding_generation,
    get_final_chunk_keys,
    batch_get_rds_items,
    merge_db_and_pinecone_data,
)
from retrieval.search_filter import build_filter
from retrieval.search import rerank_chunks_cohere

from config import (
    MAX_CHUNKS_BEFORE_RERANK,
    RECENT_WINDOW_DAYS_DEFAULT,
    RECENT_WINDOW_DAYS_MAX,
    RECENT_BUCKET_WEIGHT,
    RECENT_BACKSTOP_WEIGHT,
    RECENT_DEFAULT_LIMIT,
    RECENT_MIN_RESULTS,
    ORIGINAL_QUERY_WEIGHT,
    HYDE_WEIGHT_MAX,
    HYDE_WEIGHT_MIN,
    PER_EPISODE_CAP,
    MIN_PER_BUCKET,
    RELEVANCE_FLOOR,
    RELEVANCE_FLOOR_HARD,
    MIN_RELEVANT_RESULTS,
    RECENCY_BOOST_HARD,
    RECENCY_BOOST_SOFT,
    RECENCY_FIRST_BUCKET_LIMIT,
    RERANKER_ENABLED,
)

logger = logging.getLogger(__name__)

# ============================================================================
# TIME SEARCH PLAN CONSTANTS (imported from config.py)
# ============================================================================
FALLBACK_HYDE_WEIGHT = (HYDE_WEIGHT_MAX + HYDE_WEIGHT_MIN) / 2.0  # Fallback weight


def _date_to_numeric(dt_obj: date) -> int:
    """Convert date object to YYYYMMDD integer."""
    return dt_obj.year * 10000 + dt_obj.month * 100 + dt_obj.day


def _yyyymmdd(s: str | None) -> int | None:
    """Convert ISO date string to YYYYMMDD integer."""
    if not s:
        return None
    try:
        cleaned = s.strip().strip('"').strip("'").rstrip(',').replace('\xa0', '').strip()
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.year * 10000 + dt.month * 100 + dt.day
    except Exception as e:
        logger.warning(f"Failed to parse date string '{s}': {e}")
        return None


def _yyyymmdd_to_iso(n: int | None) -> str | None:
    """Convert YYYYMMDD integer back to ISO date string."""
    if n is None:
        return None
    y = n // 10000
    m = (n % 10000) // 100
    d = n % 100
    return f"{y:04d}-{m:02d}-{d:02d}"


def _recent_numeric_range(time_filter: dict | None) -> dict:
    """
    Calculate numeric date range for recent content.
    Uses explicit dates from analyzer or defaults to 6 months.
    """
    explicit_start = None
    explicit_end = None
    approx_window = None

    if time_filter:
        explicit_start = _yyyymmdd(time_filter.get('start_date_utc')) if time_filter.get('start_date_utc') else None
        explicit_end = _yyyymmdd(time_filter.get('end_date_utc')) if time_filter.get('end_date_utc') else None
        approx_window = time_filter.get('approx_window_days')

    today = datetime.now(timezone.utc).date()
    end_num = explicit_end or _date_to_numeric(today)

    if explicit_start:
        start_num = explicit_start
    else:
        days_window = None
        if approx_window:
            try:
                days_window = max(1, min(int(approx_window), RECENT_WINDOW_DAYS_MAX))
            except (TypeError, ValueError):
                days_window = None
        if days_window is None:
            days_window = RECENT_WINDOW_DAYS_DEFAULT

        start_date = today - timedelta(days=days_window)
        start_num = _date_to_numeric(start_date)

    if start_num > end_num:
        start_num, end_num = end_num, start_num

    return {'start': start_num, 'end': end_num}


def _derive_bucket_time_filter(base_tf: dict | None, post_date_range: dict | None) -> dict | None:
    """
    If a bucket provides a numeric date range, synthesize a per-bucket time_filter
    so Pinecone also gates by date (not just client-side).
    """
    if post_date_range and (post_date_range.get("start") is not None or post_date_range.get("end") is not None):
        start_iso = _yyyymmdd_to_iso(post_date_range.get("start"))
        end_iso = _yyyymmdd_to_iso(post_date_range.get("end"))

        if start_iso and end_iso:
            return {
                "has_time_constraint": True,
                "mode": "between",
                "start_date_utc": start_iso,
                "end_date_utc": end_iso,
            }
        elif end_iso:
            return {
                "has_time_constraint": True,
                "mode": "before",
                "end_date_utc": end_iso,
            }
        elif start_iso:
            return {
                "has_time_constraint": True,
                "mode": "after",
                "start_date_utc": start_iso,
            }
    return base_tf


def _is_strict_time_request(user_query: str, time_filter: dict | None) -> bool:
    """Determine if query requires strict time filtering."""
    ql = (user_query or "").lower()
    if any(w in ql for w in [" only", " strictly", " exactly"]):
        return True
    mode = ((time_filter or {}).get("mode") or "none").lower()
    return mode in ("on", "between")


def _us_election_day(year: int) -> str:
    """Calculate US election day (first Tuesday after first Monday in November)."""
    d = date(year, 11, 1)
    # Find first Monday
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:  # Nov 1 is Monday
        first_monday = d
    else:
        first_monday = d + timedelta(days=days_until_monday)
    # Election is the Tuesday after first Monday
    election_day = first_monday + timedelta(days=1)
    return election_day.isoformat()


def _cosine_similarity(vec_a, vec_b) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec_a or not vec_b:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rank_hyde_weights(idx_sim_pairs, high: float = HYDE_WEIGHT_MAX, low: float = HYDE_WEIGHT_MIN):
    if not idx_sim_pairs:
        return {}, []
    if high < low:
        high, low = low, high
    ranked = sorted(idx_sim_pairs, key=lambda x: x[1], reverse=True)
    n = len(ranked)
    step = (high - low) / max(n - 1, 1) if n > 1 else 0.0
    weight_lookup = {}
    for position, (idx, sim, _) in enumerate(ranked):
        weight = high - (step * position)
        weight_lookup[idx] = float(max(low, round(weight, 4)))
    return weight_lookup, ranked


def enforce_episode_cap_and_bucket_quota(items: list, per_episode_cap: int = PER_EPISODE_CAP, min_per_bucket: int = MIN_PER_BUCKET) -> list:
    by_episode = {}
    by_bucket = {}
    selected = []

    # First pass: apply per-episode cap in rank order
    for m in items:
        epi = (m.get('metadata') or {}).get('episodeId') or m.get('episodeId')
        b = m.get('bucket', 'all')
        if epi:
            if by_episode.get(epi, 0) >= per_episode_cap:
                continue
            by_episode[epi] = by_episode.get(epi, 0) + 1
        selected.append(m)
        by_bucket[b] = by_bucket.get(b, 0) + 1

    # Ensure min per bucket by pulling earlier items if needed
    if min_per_bucket > 0:
        need = {b: max(0, min_per_bucket - by_bucket.get(b, 0)) for b in set(m.get('bucket', 'all') for m in items)}
        if any(v > 0 for v in need.values()):
            out, seen = [], set()
            # Seed with items from buckets needing fill first
            for b in need:
                if need[b] <= 0:
                    continue
                for m in selected:
                    if m.get('bucket', 'all') == b and m['id'] not in seen:
                        out.append(m)
                        seen.add(m['id'])
                        need[b] -= 1
                        if need[b] <= 0:
                            break
            # Append remaining in order (dedup)
            for m in selected:
                if m['id'] not in seen:
                    out.append(m)
                    seen.add(m['id'])
            selected = out

    return selected


# ============================================================================
# RECENCY-FIRST STRATEGY FUNCTIONS
# ============================================================================

def apply_recency_boost(
    items: list,
    time_filter: dict,
    boost_factor: float | None = None
) -> list:
    if not items:
        return items

    priority = (time_filter or {}).get("recency_priority", "none")

    if priority == "none":
        return items

    # Determine boost factor based on priority
    if boost_factor is None:
        boost_factor = RECENCY_BOOST_HARD if priority == "hard" else RECENCY_BOOST_SOFT

    # Extract dates and find range
    dates = []
    for item in items:
        pd = (item.get('metadata') or {}).get('pdnumeric', 0)
        if not pd:
            pd = item.get('pdnumeric', 0)
        dates.append(pd)

    if not dates or max(dates) == 0:
        return items

    max_date = max(dates)
    min_date = min(d for d in dates if d > 0) if any(d > 0 for d in dates) else max_date
    date_range = max(max_date - min_date, 1)

    # Apply boost based on relative recency
    for i, item in enumerate(items):
        pd = dates[i]
        if pd > 0:
            recency_ratio = (pd - min_date) / date_range  # 0 to 1, newest = 1
            boost = 1.0 + (boost_factor * recency_ratio)
        else:
            boost = 1.0  # No boost for items without dates

        original_score = item.get('hybrid_score', 0) or item.get('score', 0) or 0.5
        item['original_score'] = original_score
        item['hybrid_score'] = original_score * boost
        item['recency_boost'] = boost
        item['recency_ratio'] = recency_ratio if pd > 0 else None

    # Sort by boosted score
    return sorted(items, key=lambda x: x.get('hybrid_score', 0), reverse=True)


def apply_recency_strategy(
    candidates: list,
    time_filter: dict | None,
    semantic_scores: dict | None = None
) -> tuple[list, dict]:
    metadata = {
        "recency_satisfied": True,
        "fallback_triggered": False,
        "relevance_floor_applied": False,
        "original_count": len(candidates),
        "post_floor_count": len(candidates),
        "recency_priority": "none",
        "topic_present": False,
    }

    if not candidates:
        return candidates, metadata

    if not time_filter:
        return candidates, metadata

    priority = time_filter.get("recency_priority", "none")
    topic_present = time_filter.get("topic_present", False)
    metadata["recency_priority"] = priority
    metadata["topic_present"] = topic_present

    if priority == "none":
        return candidates, metadata

    # ========================================================================
    # STEP 1: Tag candidates with semantic scores if provided
    # ========================================================================
    if semantic_scores:
        for item in candidates:
            item_id = item.get('id')
            if item_id and item_id in semantic_scores:
                item['semantic_score'] = semantic_scores[item_id]

    # ========================================================================
    # STEP 2: Apply recency boost
    # ========================================================================
    boosted = apply_recency_boost(candidates, time_filter)

    # ========================================================================
    # STEP 3: Determine relevance floor
    # ========================================================================

    if priority == "hard" and not topic_present:
        floor = RELEVANCE_FLOOR_HARD
    else:
        floor = RELEVANCE_FLOOR

    # ========================================================================
    # STEP 4: Apply relevance floor if topic is present
    # ========================================================================
    if topic_present:
        metadata["relevance_floor_applied"] = True

        # Filter to items above relevance floor
        relevant_items = []
        for item in boosted:
            relevance = (
                item.get('rerank_score') or
                item.get('original_score') or
                item.get('semantic_score') or
                item.get('hybrid_score', 0)
            )
            if relevance >= floor:
                item['passed_relevance_floor'] = True
                relevant_items.append(item)
            else:
                item['passed_relevance_floor'] = False

        metadata["post_floor_count"] = len(relevant_items)
        logger.info(f"[RECENCY] Relevance floor ({floor}): {len(relevant_items)}/{len(boosted)} passed")

        # Log items that failed the floor for diagnostics
        failed_items = [item for item in boosted if not item.get('passed_relevance_floor', True)]
        if failed_items[:3]:
            logger.info(f"[RECENCY] Items REJECTED by relevance floor ({floor}):")
            for item in failed_items[:3]:
                meta = item.get('metadata', {})
                title = meta.get('episodeTitle', meta.get('title', 'Unknown'))[:40]
                score = item.get('rerank_score') or item.get('original_score') or item.get('hybrid_score', 0)
                pd = meta.get('pdnumeric', 0)
                logger.info(f"  - score={score:.3f} | date={pd} | {title}")

        # ====================================================================
        # STEP 5: Check if sufficient relevant results
        # ====================================================================
        if len(relevant_items) >= MIN_RELEVANT_RESULTS:
            # Success: return boosted relevant items
            logger.info(f"[RECENCY] Success: {len(relevant_items)} relevant items found")
            # Log top items that passed
            if relevant_items[:3]:
                logger.info(f"[RECENCY] Top items PASSED relevance floor:")
                for item in relevant_items[:3]:
                    meta = item.get('metadata', {})
                    title = meta.get('episodeTitle', meta.get('title', 'Unknown'))[:40]
                    score = item.get('rerank_score') or item.get('original_score') or item.get('hybrid_score', 0)
                    pd = meta.get('pdnumeric', 0)
                    boost = item.get('recency_boost', 1.0)
                    logger.info(f"  + score={score:.3f} | date={pd} | boost={boost:.2f} | {title}")
            return relevant_items, metadata

        # ====================================================================
        # STEP 6: Fallback - insufficient relevant recent content
        # ====================================================================
        metadata["recency_satisfied"] = False
        metadata["fallback_triggered"] = True

        if relevant_items:
            # Have some relevant items, use them (even if < MIN)
            logger.warning(f"[RECENCY] Partial fallback: only {len(relevant_items)} relevant items")
            return relevant_items, metadata

        # No relevant items at all - return semantic-only (drop recency boost)
        logger.warning("[RECENCY] Full fallback: no relevant items, returning by original score")
        # Sort by original score (semantic relevance) instead of boosted
        semantic_sorted = sorted(
            boosted,
            key=lambda x: x.get('original_score', x.get('semantic_score', 0)),
            reverse=True
        )
        return semantic_sorted, metadata

    # ========================================================================
    # No topic present (pure recency query) - return boosted results as-is
    # ========================================================================
    logger.info(f"[RECENCY] Pure recency query: returning {len(boosted)} boosted items")
    return boosted, metadata


# ============================================================================
# HYBRID METADATA-AWARE SCORING (Zero Latency - Pure Python Math)
# ============================================================================

def _normalize_to_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip().lower() for v in value.split(',') if v.strip()]
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if v]
    return []


def _compute_person_match_score(
    chunk: dict,
    target_guests: List[str],
    target_hosts: List[str],
) -> tuple[float, str]:
    if not target_guests and not target_hosts:
        return 0.5, 'neutral'  # Neutral score when no person filter

    # Normalize chunk metadata
    chunk_speakers = _normalize_to_list(chunk.get('speakers', []))
    chunk_guests = _normalize_to_list(chunk.get('guests', []))
    chunk_hosts = _normalize_to_list(chunk.get('hosts', []))

    # Normalize targets (separate guests from hosts for priority scoring)
    guest_targets_lower = [g.lower() for g in target_guests]
    host_targets_lower = [h.lower() for h in target_hosts]
    all_targets_lower = guest_targets_lower + host_targets_lower

    # ====================================================================
    # PRIORITY 1: Target GUEST is speaking in this chunk (highest value!)
    # ====================================================================
    for target in guest_targets_lower:
        for speaker in chunk_speakers:
            if target in speaker or speaker in target:
                return 1.0, 'exact_guest_speaking'

    # ====================================================================
    # PRIORITY 2: Target GUEST is in the guests metadata field
    # This is the KEY FIX - exact guest match should rank much higher
    # than just being mentioned in the text!
    # ====================================================================
    for target in guest_targets_lower:
        for guest in chunk_guests:
            if target in guest or guest in target:
                return 0.85, 'exact_guest_present'

    # ====================================================================
    # PRIORITY 3: Target HOST matches (for show-specific queries)
    # ====================================================================
    for target in host_targets_lower:
        for host in chunk_hosts:
            if target in host or host in target:
                return 0.7, 'host_match'

    # ====================================================================
    # PRIORITY 4: Target is just mentioned in transcript text
    # This is a WEAK signal - should not rank highly for "guest X" queries
    # ====================================================================
    transcript = (chunk.get('chunk', '') or '').lower()
    for target in all_targets_lower:
        if target in transcript:
            return 0.3, 'mention_only'

    return 0.0, 'no_match'


def _compute_show_match_score(
    chunk: dict,
    target_hosts: List[str],
    extracted_show: Optional[str] = None,
) -> float:
    chunk_hosts = _normalize_to_list(chunk.get('hosts', []))
    chunk_podcast = (chunk.get('podcast_title', '') or '').lower()
    chunk_channel = (chunk.get('channelTitle', '') or '').lower()

    # Priority 1: Direct show name match (most reliable)
    if extracted_show:
        show_lower = extracted_show.lower()
        # Check against channelTitle
        if show_lower in chunk_channel or chunk_channel in show_lower:
            return 1.0
        # Check against podcast_title
        if show_lower in chunk_podcast or chunk_podcast in show_lower:
            return 1.0
        # Show requested but no match - penalize
        return 0.2

    # Priority 2: Host-based inference
    if target_hosts:
        for host in target_hosts:
            host_lower = host.lower()
            if host_lower in chunk_podcast:
                return 0.9
            for ch in chunk_hosts:
                if host_lower in ch or ch in host_lower:
                    return 0.9
        return 0.3

    # No show filter - return None to signal "not applicable"
    return None


def apply_hybrid_metadata_scoring(
    chunks: List[dict],
    llm_analysis: dict,
    max_per_episode: int = 3,
) -> List[dict]:
    if not chunks:
        return chunks

    # ========================================================================
    # EXTRACT INTENT FROM ANALYSIS
    # ========================================================================
    time_filter = llm_analysis.get('time_filter', {}) or {}
    extracted_guests = llm_analysis.get('extracted_guests_interviewees', []) or []
    extracted_hosts = llm_analysis.get('extracted_hosts_creators', []) or []
    extracted_show = llm_analysis.get('extracted_show')  # NEW: Show name from query

    recency_priority = time_filter.get('recency_priority', 'none')
    topic_present = time_filter.get('topic_present', False)
    has_person_intent = bool(extracted_guests or extracted_hosts)
    has_show_intent = bool(extracted_show or extracted_hosts)
    has_recency_intent = recency_priority in ('hard', 'soft')

    # ========================================================================
    # DYNAMIC WEIGHTS - Only include active signals
    # ========================================================================
    # Base: semantic always included
    weights = {'semantic': 0.0, 'date': 0.0, 'person': 0.0, 'show': 0.0}

    if recency_priority == 'hard' and not topic_present:
        # "latest episode" - pure recency
        weights = {'semantic': 0.20, 'date': 0.50, 'person': 0.20 if has_person_intent else 0.0, 'show': 0.10 if has_show_intent else 0.0}
        weight_type = "PURE RECENCY"
    elif recency_priority == 'hard' and topic_present:
        # "latest episode about AGI"
        weights = {'semantic': 0.35, 'date': 0.35, 'person': 0.20 if has_person_intent else 0.0, 'show': 0.10 if has_show_intent else 0.0}
        weight_type = "RECENCY+TOPIC"
    elif has_person_intent and has_recency_intent:
        # "latest Lex Fridman episode"
        weights = {'semantic': 0.15, 'date': 0.30, 'person': 0.35, 'show': 0.20 if has_show_intent else 0.0}
        weight_type = "PERSON+RECENCY"
    elif has_person_intent:
        # "what did Elon say about X"
        weights = {'semantic': 0.40, 'date': 0.05, 'person': 0.40, 'show': 0.15 if has_show_intent else 0.0}
        weight_type = "PERSON FOCUS"
    elif has_show_intent:
        # "YC podcast about AI"
        weights = {'semantic': 0.50, 'date': 0.10 if has_recency_intent else 0.05, 'person': 0.0, 'show': 0.35}
        weight_type = "SHOW FOCUS"
    else:
        # Pure topic query
        weights = {'semantic': 0.85, 'date': 0.10 if has_recency_intent else 0.05, 'person': 0.0, 'show': 0.0}
        weight_type = "PURE TOPIC"

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    logger.info(f"[HYBRID] Weights: {weight_type} | sem={weights['semantic']:.2f} date={weights['date']:.2f} person={weights['person']:.2f} show={weights['show']:.2f}")

    # ========================================================================
    # SEMANTIC NORMALIZATION (0-1 based on actual distribution)
    # ========================================================================
    raw_semantic_scores = [c.get('rerank_score', 0) or c.get('hybrid_score', 0.5) or 0.5 for c in chunks]
    min_sem = min(raw_semantic_scores) if raw_semantic_scores else 0
    max_sem = max(raw_semantic_scores) if raw_semantic_scores else 1
    sem_range = max(max_sem - min_sem, 0.001)  
    # ========================================================================
    # COMPUTE DATE SCORES (Normalized 0-1, newest = 1)
    # ========================================================================
    dates = []
    for chunk in chunks:
        pd = chunk.get('pdnumeric', 0)
        if not pd:
            pd = (chunk.get('metadata', {}) or {}).get('pdnumeric', 0)
        if not pd:
            pub_str = chunk.get('published_date', '')
            if pub_str:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                    pd = dt.year * 10000 + dt.month * 100 + dt.day
                except:
                    pd = 0
        dates.append(pd or 0)

    max_date = max(dates) if dates else 0
    min_date = min(d for d in dates if d > 0) if any(d > 0 for d in dates) else max_date
    date_range = max(max_date - min_date, 1)

    # ========================================================================
    # COMPUTE SCORES FOR EACH CHUNK
    # ========================================================================
    exact_guest_matches = 0
    mention_only_matches = 0
    show_matches = 0

    for i, chunk in enumerate(chunks):
        # --- Normalized Semantic Score ---
        raw_sem = chunk.get('rerank_score', 0) or chunk.get('hybrid_score', 0.5) or 0.5
        semantic_score = (raw_sem - min_sem) / sem_range

        # --- Date Score ---
        pd = dates[i]
        date_score = (pd - min_date) / date_range if pd > 0 else 0.0

        # --- Person Match Score ---
        person_score, match_type = _compute_person_match_score(chunk, extracted_guests, extracted_hosts)

        # Track match types
        if match_type in ('exact_guest_speaking', 'exact_guest_present'):
            exact_guest_matches += 1
        elif match_type == 'mention_only':
            mention_only_matches += 1

        # --- Show Match Score (now with extracted_show) ---
        show_score_raw = _compute_show_match_score(chunk, extracted_hosts, extracted_show)
        show_score = show_score_raw if show_score_raw is not None else 0.0
        if show_score_raw is not None and show_score_raw >= 0.9:
            show_matches += 1

        # --- Content Quality Boost ---
        chunk_text = chunk.get('chunk', '') or ''
        quality_boost = 0.0
        if len(chunk_text) > 500:
            quality_boost = 0.03  
        elif len(chunk_text) < 100:
            quality_boost = -0.05  

        # --- Compute Base Score ---
        base_score = (
            weights['semantic'] * semantic_score +
            weights['date'] * date_score +
            weights['person'] * person_score +
            weights['show'] * show_score +
            quality_boost
        )

        # --- Hard Filters: Penalize mismatches when filter was requested ---
        penalty = 1.0

        # Person requested but no match
        if has_person_intent and match_type in ('no_match', 'neutral'):
            penalty *= 0.4  # 60% penalty

        # Show requested but no match
        if extracted_show and show_score_raw is not None and show_score_raw < 0.5:
            penalty *= 0.5  # 50% penalty

        final_score = base_score * penalty

        # --- Store all scores ---
        chunk['semantic_score'] = semantic_score
        chunk['semantic_score_raw'] = raw_sem
        chunk['date_score'] = date_score
        chunk['person_score'] = person_score
        chunk['person_match_type'] = match_type
        chunk['show_score'] = show_score
        chunk['quality_boost'] = quality_boost
        chunk['penalty'] = penalty
        chunk['final_score'] = final_score
        chunk['hybrid_weights'] = weights

        # Recency markers for UI
        if date_score >= 0.9:
            chunk['recency_boost'] = 1.5
        elif date_score >= 0.7:
            chunk['recency_boost'] = 1.3
        elif date_score >= 0.5:
            chunk['recency_boost'] = 1.1

    # ========================================================================
    # SORT AND APPLY EPISODE DIVERSITY CAP
    # ========================================================================
    sorted_by_score = sorted(chunks, key=lambda x: x.get('final_score', 0), reverse=True)

    # Episode diversity: max N chunks per episode
    episode_counts = {}
    diverse_results = []
    overflow_results = []  # Chunks that exceeded per-episode cap

    for chunk in sorted_by_score:
        ep_title = chunk.get('episode_title', '') or chunk.get('id', '')
        current_count = episode_counts.get(ep_title, 0)

        if current_count < max_per_episode:
            diverse_results.append(chunk)
            episode_counts[ep_title] = current_count + 1
        else:
            overflow_results.append(chunk)

    # If we need more chunks, add from overflow (already sorted by score)
    final_results = diverse_results + overflow_results

    # ========================================================================
    # LOGGING
    # ========================================================================
    if has_person_intent:
        logger.info(f"[HYBRID] Person match: {exact_guest_matches} exact, {mention_only_matches} mention, {len(chunks) - exact_guest_matches - mention_only_matches} none")
    if has_show_intent:
        logger.info(f"[HYBRID] Show match: {show_matches}/{len(chunks)} chunks")

    logger.info(f"[HYBRID] Episode diversity: {len(episode_counts)} unique episodes, max {max_per_episode}/ep")
    logger.info(f"[HYBRID] Scored {len(chunks)} chunks. Top 5:")

    for i, chunk in enumerate(final_results[:5]):
        episode = chunk.get('episode_title', 'Unknown')[:35]
        match_type = chunk.get('person_match_type', 'unknown')
        penalty = chunk.get('penalty', 1.0)
        penalty_str = f" PEN={penalty:.1f}" if penalty < 1.0 else ""
        logger.info(
            f"  [{i}] final={chunk.get('final_score', 0):.3f} | "
            f"sem={chunk.get('semantic_score', 0):.2f} date={chunk.get('date_score', 0):.2f} "
            f"person={chunk.get('person_score', 0):.2f} show={chunk.get('show_score', 0):.2f}{penalty_str} | {episode}"
        )

    return final_results


def make_time_search_plan(user_query: str, time_filter: dict | None) -> List[dict]:
    if not time_filter or not time_filter.get("has_time_constraint"):
        return [{'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 1.0}]

    mode = (time_filter.get("mode") or "none").lower()
    start_iso = time_filter.get("start_date_utc")
    end_iso = time_filter.get("end_date_utc")
    anchor_iso = time_filter.get("anchor_date_utc")

    start_num = _yyyymmdd(start_iso) if start_iso else None
    end_num = _yyyymmdd(end_iso) if end_iso else None

    strict = _is_strict_time_request(user_query, time_filter)
    prepost_hint = any(w in user_query.lower() for w in ["before and after", "pre and post", "before/after"])

    # ========== PRE/POST EVENT SPLITTING ==========
    if prepost_hint and mode == "none":
        ql = user_query.lower()
        # Use analyzer-provided anchor if present
        inferred_anchor = anchor_iso
        # Else try to infer US election anchor from year in query
        if not inferred_anchor:
            m = re.search(r"(?:us )?election[s]?\s+(\d{4})", ql)
            if m:
                try:
                    inferred_anchor = _us_election_day(int(m.group(1)))
                except Exception:
                    inferred_anchor = None

        if inferred_anchor:
            a = _yyyymmdd(inferred_anchor)
            if a:
                logger.info(f"[TIME PLAN] Pre/post event splitting with anchor={inferred_anchor} ({a})")
                return [
                    {'label': 'pre', 'include_date_clause': True, 'post_date_range': {'start': None, 'end': a - 1}, 'weight': 1.0},
                    {'label': 'post', 'include_date_clause': True, 'post_date_range': {'start': a, 'end': None}, 'weight': 1.0},
                    {'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 0.4},
                ]

    # ========== LATEST MODE ==========
    if mode == "latest":
        recent_rng = _recent_numeric_range(time_filter)
        recency_priority = time_filter.get("recency_priority", "soft")
        topic_present = time_filter.get("topic_present", False)

        logger.info(f"[TIME PLAN] Latest mode: recent_range={recent_rng}, priority={recency_priority}, topic={topic_present}")


        if recency_priority == "hard" and not topic_present:
            # For pure recency, use a tighter date range (last 21 days) to ensure newest items
            today = datetime.now(timezone.utc).date()
            recent_90_days = {
                'start': _date_to_numeric(today - timedelta(days=21)),
                'end': _date_to_numeric(today)
            }
            return [
                # Primary: Fetch from last 90 days and sort by date - ensures truly recent items
                {'label': 'recency_first', 'include_date_clause': True, 'post_date_range': recent_90_days,
                 'weight': 2.0, 'sort_by_date': True, 'limit': RECENCY_FIRST_BUCKET_LIMIT * 2},
                # Secondary: Broader 6-month window for semantic matching
                {'label': 'recent_semantic', 'include_date_clause': True, 'post_date_range': recent_rng,
                 'weight': RECENT_BUCKET_WEIGHT},
                # Fallback: All content
                {'label': 'all', 'include_date_clause': False, 'post_date_range': None,
                 'weight': RECENT_BACKSTOP_WEIGHT},
            ]
        else:
            today = datetime.now(timezone.utc).date()
            recent_60_days = {
                'start': _date_to_numeric(today - timedelta(days=21)),
                'end': _date_to_numeric(today)
            }
            return [
                # Primary: Semantic search in recent window
                {'label': 'recent_semantic', 'include_date_clause': True, 'post_date_range': recent_rng,
                 'weight': RECENT_BUCKET_WEIGHT},
                # Secondary: Pure recency bucket with tight date filter
                {'label': 'recency_first', 'include_date_clause': True, 'post_date_range': recent_60_days,
                 'weight': 1.3, 'sort_by_date': True, 'limit': RECENCY_FIRST_BUCKET_LIMIT},
                # Fallback: All content
                {'label': 'all', 'include_date_clause': False, 'post_date_range': None,
                 'weight': RECENT_BACKSTOP_WEIGHT},
            ]

    # ========== OLDEST MODE ==========
    if mode == "oldest":
        logger.info("[TIME PLAN] Oldest mode: using all bucket, will sort post-fusion")
        return [{'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 1.0}]

    # ========== RELATIVE RECENT MODE ==========
    if mode == "relative_recent" and not strict:
        rng = {'start': start_num, 'end': end_num}
        if rng['start'] is None and rng['end'] is None:
            rng = _recent_numeric_range(time_filter)
        logger.info(f"[TIME PLAN] Relative recent (soft): range={rng}")
        return [
            {'label': 'recent', 'include_date_clause': True, 'post_date_range': rng, 'weight': max(RECENT_BUCKET_WEIGHT - 0.05, 0.5)},
            {'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': RECENT_BACKSTOP_WEIGHT},
        ]

    # ========== PRE/POST WITH EXPLICIT MODE ==========
    if prepost_hint and mode in ("between", "before", "after"):
        if not anchor_iso and start_num and end_num:
            mid = (start_num + end_num) // 2
            anchor_iso = f"{mid // 10000:04d}-{(mid % 10000) // 100:02d}-{mid % 100:02d}"
        buckets = []
        if anchor_iso:
            a = _yyyymmdd(anchor_iso)
            if a:
                logger.info(f"[TIME PLAN] Pre/post event mode={mode} with anchor={anchor_iso}")
                buckets.append({'label': 'pre', 'include_date_clause': True, 'post_date_range': {'start': None, 'end': a - 1}, 'weight': 1.0})
                buckets.append({'label': 'post', 'include_date_clause': True, 'post_date_range': {'start': a, 'end': None}, 'weight': 1.0})
                buckets.append({'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 0.9})
                return buckets

    # ========== STRICT DATE MODES ==========
    if mode in ("on", "between", "before", "after") and (start_num or end_num):
        rng = {'start': start_num, 'end': end_num}
        if strict:
            logger.info(f"[TIME PLAN] Strict mode={mode}: range={rng}")
            return [{'label': 'in_range', 'include_date_clause': True, 'post_date_range': rng, 'weight': 1.0}]
        else:
            logger.info(f"[TIME PLAN] Soft mode={mode}: range={rng} + fallback")
            return [
                {'label': 'in_range', 'include_date_clause': True, 'post_date_range': rng, 'weight': 1.1},
                {'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 0.4},
            ]

    # Fallback
    logger.info("[TIME PLAN] No specific time plan, using all bucket")
    return [{'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 1.0}]


def apply_time_sort_and_limit(results: list, time_filter: dict | None) -> list:
    if not results or not time_filter:
        return results

    mode = (time_filter.get("mode") or "none").lower()
    pref = (time_filter.get("sort_preference") or "").lower()
    latest_n = time_filter.get("latest_n")
    oldest_n = time_filter.get("oldest_n")

    if mode not in ("latest", "oldest") and pref not in ("latest", "oldest"):
        return results

    effective_pref = pref if pref in ("latest", "oldest") else (mode if mode in ("latest", "oldest") else "")

    def get_day(meta):
        if not meta:
            return None
        if "pdnumeric" in meta:
            try:
                return int(meta["pdnumeric"])
            except (ValueError, TypeError):
                pass
        iso = meta.get("publishedDate")
        if iso:
            try:
                dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(timezone.utc)
                return dt.year * 10000 + dt.month * 100 + dt.day
            except:
                pass
        return None

    if effective_pref in ("latest", "oldest"):
        reverse = (effective_pref == "latest")

        with_dates = []
        without_dates = []

        for item in results:
            day = get_day(item.get("metadata", {}))
            if day is not None:
                with_dates.append((item, day))
            else:
                without_dates.append(item)

        with_dates.sort(key=lambda x: (x[1], (x[0].get("hybrid_score") or 0.0)), reverse=reverse)
        sorted_results = [item for item, _ in with_dates]
        sorted_results.extend(without_dates)

        logger.info(f"[TIME SORT] Sorted {len(with_dates)} dated items ({effective_pref}), {len(without_dates)} undated")

        limit_count = None
        if effective_pref == "latest":
            limit_count = max(1, int(latest_n)) if latest_n and latest_n > 1 else RECENT_DEFAULT_LIMIT
        elif effective_pref == "oldest":
            limit_count = max(1, int(oldest_n)) if oldest_n and oldest_n > 1 else RECENT_DEFAULT_LIMIT

        if limit_count:
            sorted_results = sorted_results[:limit_count]
            logger.info(f"[TIME SORT] Limited to top {limit_count} items for '{effective_pref}'")

        return sorted_results

    return results


@dataclass
class StageUpdate:
    stage: str
    message: str
    progress: float  # 0.0 to 1.0
    data: Optional[Dict[str, Any]] = None


class EchoFindAgent:


    def __init__(
        self,
        openai_client,
        pinecone_client,
        gemini_client,
        unique_personalities: List[str],
        unique_authors: List[str],
        unique_shows: List[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.gemini_client = gemini_client
        self.unique_personalities = unique_personalities
        self.unique_authors = unique_authors
        self.unique_shows = unique_shows or []
        self.config = config or {}

        # Pipeline parameters with defaults
        self.pinecone_k = self.config.get("pinecone_k", 100)
        self.target_per_query = self.config.get("target_per_query", 30)
        self.max_chunks_before_rerank = self.config.get("max_chunks_rerank", MAX_CHUNKS_BEFORE_RERANK)
        self.max_chunks_for_selection = self.config.get("max_chunks_selection", 25)
        self.reranker_top_n = self.config.get("reranker_top_n", 50)

    # ==========================================================================
    # AGENT BRANCH HANDLERS (Phase 4)
    # ==========================================================================

    async def _handle_small_talk(
        self,
        session_id: str,
        question: str,
        memory: ConversationMemory,
        router_output: RouterOutput,
        pipeline_start: float,
    ) -> AsyncGenerator[StageUpdate, None]:
        logger.info("[AGENT] Dispatching to SMALL_TALK branch")

        yield StageUpdate(
            stage=PipelineStage.ANALYZING.value,
            message="Processing your message...",
            progress=0.3,
        )

        try:
            # Call small talk handler (uses sub_intent from router, no regex)
            response = await handle_small_talk(
                self.gemini_client,
                question,
                memory,
                router_output,
            )

            # Apply unified memory update (Phase 0 pattern)
            memory.apply_branch_memory_update(
                update=response.memory_update,
                route_chosen="small_talk",
                route_confidence=router_output.confidence,
                query_intent=router_output.query_intent,
                user_question=question,
                resolved_query=question,  # No resolution needed for small talk
            )

            total_time = time.time() - pipeline_start

            logger.info(f"[AGENT] SMALL_TALK complete in {total_time:.2f}s")
            logger.info(f"[AGENT] Response type: {response.response_type}")
            logger.info(f"[AGENT] Memory action: {response.memory_update.action_type}")

            yield StageUpdate(
                stage=PipelineStage.COMPLETE.value,
                message="Done!",
                progress=1.0,
                data={
                    "answer": response.response_text,
                    "confidence": 0.9 if response.response_type != "error" else 0.3,
                    "response_type": response.response_type,
                    "sources": response.sources,
                    "chunk": None,  # No clip for small talk
                    "total_time": total_time,
                    "is_followup": False,
                    "resolved_query": question,
                    "branch": "small_talk",
                }
            )

        except Exception as e:
            logger.error(f"[AGENT] Small talk handler failed: {e}", exc_info=True)
            raise  # Re-raise to let caller handle fallback

    async def _handle_episode_search(
        self,
        session_id: str,
        question: str,
        memory: ConversationMemory,
        router_output: RouterOutput,
        pipeline_start: float,
    ) -> AsyncGenerator[StageUpdate, None]:
        logger.info("[AGENT] Dispatching to EPISODE_SEARCH branch")

        yield StageUpdate(
            stage=PipelineStage.EPISODE_SEARCHING.value,
            message="Searching for episodes...",
            progress=0.3,
        )

        try:
            # Call episode search handler
            response = await handle_episode_search(
                self.gemini_client,
                self.openai_client,
                self.pinecone_client,
                question,
                memory,
                self.unique_personalities,
                self.unique_authors,
                router_output,
            )

            yield StageUpdate(
                stage=PipelineStage.EPISODE_SELECTED.value,
                message=f"Found: {response.episode_title[:50]}..." if response.episode_title else "Searching...",
                progress=0.8,
            )

            # Apply unified memory update (Phase 0 pattern)
            memory.apply_branch_memory_update(
                update=response.memory_update,
                route_chosen="episode_search",
                route_confidence=router_output.confidence,
                query_intent=router_output.query_intent,
                user_question=question,
                resolved_query=question,  # Intent extraction handles resolution internally
            )

            total_time = time.time() - pipeline_start

            logger.info(f"[AGENT] EPISODE_SEARCH complete in {total_time:.2f}s")
            logger.info(f"[AGENT] Episode: {response.episode_title[:60] if response.episode_title else 'None'}")
            logger.info(f"[AGENT] Confidence: {response.confidence:.2f}")
            logger.info(f"[AGENT] Memory action: {response.memory_update.action_type}")

            # Get current turn ID for recommendations storage
            current_turn_id = f"turn_{memory.turn_count}"

            yield StageUpdate(
                stage=PipelineStage.COMPLETE.value,
                message="Done!",
                progress=1.0,
                data={
                    "answer": response.response_text,
                    "confidence": response.confidence,
                    "episode": {
                        "episode_id": response.episode_id,
                        "episode_title": response.episode_title,
                        "podcast_title": response.podcast_title,
                        "published_date": response.published_date,
                        "description": response.episode_description,
                        "uri": response.episode_uri,
                        "image": response.episode_image,
                        "guests": response.guests,
                        "hosts": response.hosts,
                    },
                    "chunk": None,  # No clip for episode search
                    "total_time": total_time,
                    "is_followup": False,
                    "resolved_query": question,
                    "branch": "episode_search",
                    "turn_id": current_turn_id,  # Include turn_id for recommendations
                }
            )

            # ================================================================
            # EPISODE RECOMMENDATIONS (runs after main response)
            # ================================================================
            # Generate episode recommendations if we have enough candidates
            if (response.scored_episodes and
                len(response.scored_episodes) > 1 and
                response.episode_descriptions_data):

                logger.info("[AGENT] Generating episode recommendations...")
                try:
                    episode_recommendations = await generate_episode_recommendations(
                        self.gemini_client,
                        question,
                        question,  # resolved_query same as question for episode search
                        response.scored_episodes,
                        response.episode_descriptions_data,
                        response.selected_index,
                        response.response_text,
                        memory,
                    )

                    if episode_recommendations.recommendations:
                        # Store recommendations for later retrieval
                        await store_episode_recommendations(
                            session_id,
                            current_turn_id,
                            episode_recommendations,
                            response.scored_episodes,
                            response.episode_descriptions_data,
                            original_question=question,
                            resolved_query=question,
                        )

                        # Send episode recommendations to frontend
                        yield StageUpdate(
                            stage=PipelineStage.EPISODE_RECOMMENDATIONS.value,
                            message="Episode recommendations ready",
                            progress=1.0,
                            data={
                                "turn_id": current_turn_id,
                                "recommendations": [
                                    {
                                        "index": i,
                                        "prompt": rec.prompt,
                                        "episode_index": rec.episode_index,
                                    }
                                    for i, rec in enumerate(episode_recommendations.recommendations)
                                ],
                            }
                        )
                        logger.info(f"[AGENT] Generated {len(episode_recommendations.recommendations)} episode recommendations")
                    else:
                        logger.info("[AGENT] No episode recommendations generated")

                except Exception as rec_err:
                    logger.warning(f"[AGENT] Episode recommendations failed (non-fatal): {rec_err}")
                    # Don't raise - recommendations are optional

        except Exception as e:
            logger.error(f"[AGENT] Episode search handler failed: {e}", exc_info=True)
            raise  # Re-raise to let caller handle fallback

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    async def ask_streaming(
        self,
        session_id: str,
        question: str,
    ) -> AsyncGenerator[StageUpdate, None]:
        pipeline_start = time.time()

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[AGENT] NEW REQUEST")
        logger.info("=" * 80)
        logger.info(f"[AGENT] Session: {session_id}")
        logger.info(f"[AGENT] Question: {question}")
        logger.info("=" * 80)

        # Get or create memory
        memory = memory_store.get_or_create(session_id)

        # ================================================================
        # STAGE 0: ROUTING - Determine which branch handles this query
        # ================================================================
        yield StageUpdate(
            stage=PipelineStage.ROUTING.value,
            message="Understanding your request...",
            progress=0.05,
        )

        router_start = time.time()
        try:
            router_output = await route_query(self.gemini_client, question, memory)
        except Exception as e:
            logger.error(f"[AGENT] Routing failed: {e}")
            # Fallback to clip_search
            router_output = RouterOutput(
                route="clip_search",
                sub_intent=None,
                confidence=0.3,
                reasoning=f"Routing error: {str(e)[:50]}",
                query_intent=question[:50],
                key_signals=["routing_error"],
                fallback_route=None,
            )
        router_time = time.time() - router_start

        logger.info(f"[AGENT] Router decision: {router_output.route} (conf={router_output.confidence:.2f}) in {router_time:.2f}s")
        logger.info(f"[AGENT] Router reasoning: {router_output.reasoning}")

        yield StageUpdate(
            stage=PipelineStage.ROUTED.value,
            message=f"Route: {router_output.route}",
            progress=0.1,
            data={
                "route": router_output.route,
                "confidence": router_output.confidence,
                "reasoning": router_output.reasoning,
            },
        )

        # ================================================================
        # BRANCH DISPATCH (Phase 4)
        # ================================================================

        if router_output.route == "small_talk":
            # === SMALL TALK BRANCH ===
            try:
                async for update in self._handle_small_talk(
                    session_id, question, memory, router_output, pipeline_start
                ):
                    yield update
                return
            except Exception as e:
                logger.error(f"[AGENT] Small talk branch failed: {e}", exc_info=True)
                # Fall through to clip_search as fallback
                router_output = RouterOutput(
                    route="clip_search",
                    sub_intent=None,
                    confidence=0.3,
                    reasoning="Small talk branch failed, falling back to clip_search",
                    query_intent=router_output.query_intent,
                    key_signals=["small_talk_error"],
                    fallback_route=None,
                )
                logger.info("[AGENT] Falling back to CLIP_SEARCH after small_talk error")

        elif router_output.route == "episode_search":
            # === EPISODE SEARCH BRANCH ===
            try:
                async for update in self._handle_episode_search(
                    session_id, question, memory, router_output, pipeline_start
                ):
                    yield update
                return
            except Exception as e:
                logger.error(f"[AGENT] Episode search branch failed: {e}", exc_info=True)
                # Fall through to clip_search as fallback
                router_output = RouterOutput(
                    route="clip_search",
                    sub_intent=None,
                    confidence=0.3,
                    reasoning="Episode search branch failed, falling back to clip_search",
                    query_intent=router_output.query_intent,
                    key_signals=["episode_search_error"],
                    fallback_route=None,
                )
                logger.info("[AGENT] Falling back to CLIP_SEARCH after episode_search error")

        # ================================================================
        # CLIP SEARCH BRANCH (Default - existing RAG pipeline)
        # ================================================================
        logger.info("[AGENT] Dispatching to CLIP_SEARCH branch (full RAG pipeline)")
        # ================================================================
        # STAGE 1: Query Analysis with Memory
        # ================================================================
        yield StageUpdate(
            stage=PipelineStage.ANALYZING.value,
            message="Understanding your question...",
            progress=0.1,
        )

        stage_start = time.time()
        try:
            llm_analysis = await analyze_query_with_memory(
                self.gemini_client,
                question,
                memory,
            )
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            yield StageUpdate(
                stage=PipelineStage.ERROR.value,
                message=f"Failed to analyze query: {str(e)}",
                progress=0.1,
                data={"error": str(e)},
            )
            return

        stage_time = time.time() - stage_start
        logger.info(f"[PIPELINE] Stage 1 (Query Analysis) completed in {stage_time:.2f}s")

        resolved_query = llm_analysis.get("resolved_query", question)
        is_followup = llm_analysis.get("is_followup", False)
        hyde_docs = llm_analysis.get("hyde_documents", [])

        logger.info(f"[PIPELINE] Stage 1 Results:")
        logger.info(f"  ├─ Resolved query: {resolved_query}...")
        logger.info(f"  ├─ Is follow-up: {is_followup}")
        logger.info(f"  └─ HyDE documents: {len(hyde_docs)}")

        yield StageUpdate(
            stage=PipelineStage.ANALYZED.value,
            message=f"Query understood: {resolved_query}..." if len(resolved_query) > 50 else f"Query understood: {resolved_query}",
            progress=0.2,
            data={"resolved_query": resolved_query, "is_followup": is_followup},
        )

        # ================================================================
        # STAGE 2-3: Embedding Generation (Parallel)
        # ================================================================
        yield StageUpdate(
            stage=PipelineStage.EMBEDDING.value,
            message="Encoding your question...",
            progress=0.25,
        )

        all_queries = [resolved_query] + hyde_docs
        logger.info(f"[PIPELINE] Stage 2-3: Generating embeddings for {len(all_queries)} queries")

        try:
            # Parallel embedding generation
            dense_task = concurrent_embedding_generation(self.openai_client, all_queries)
            sparse_task = concurrent_sparse_embedding_generation(self.pinecone_client, all_queries)

            valid_query_data, valid_sparse_data = await asyncio.gather(dense_task, sparse_task)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            yield StageUpdate(
                stage=PipelineStage.ERROR.value,
                message=f"Failed to generate embeddings: {str(e)}",
                progress=0.25,
                data={"error": str(e)},
            )
            return

        logger.info(f"[PIPELINE] Stage 2-3 complete: {len(valid_query_data)} dense, {len(valid_sparse_data)} sparse embeddings")

        # ================================================================
        # HYDE Similarity-Based Weighting
        # ================================================================
        query_weight_lookup = {0: ORIGINAL_QUERY_WEIGHT}  
        hyde_similarity_ranking = []

        base_entry = next((item for item in valid_query_data if item.get("index") == 0), None)
        hyde_entries = [item for item in valid_query_data if (item.get("index") or 0) > 0]

        if base_entry and hyde_entries:
            idx_sim_pairs = []
            for entry in hyde_entries:
                sim = _cosine_similarity(base_entry["vector"], entry["vector"])
                idx_sim_pairs.append((entry["index"], sim, entry.get("query", "")))

            hyde_weight_lookup, hyde_similarity_ranking = _rank_hyde_weights(idx_sim_pairs)

            for idx, weight in hyde_weight_lookup.items():
                query_weight_lookup[idx] = weight

            logger.info("[PIPELINE] HyDE similarity ranking (higher = closer to original query):")
            for rank_position, (idx, sim, text) in enumerate(hyde_similarity_ranking, start=1):
                weight = query_weight_lookup[idx]
                preview = (text or "").replace("\n", " ")[:60] + "..." if len(text or "") > 60 else (text or "").replace("\n", " ")
                logger.info(f"  Rank {rank_position}: idx={idx} sim={sim:.4f} weight={weight:.2f}")
        else:
            # Fallback: use uniform weights for HyDE documents
            if not base_entry:
                logger.warning("[PIPELINE] Original query embedding missing; using fallback HYDE weights")
            if not hyde_entries:
                logger.warning("[PIPELINE] No valid HyDE embeddings available; using fallback HYDE weights")
            for entry in hyde_entries:
                query_weight_lookup[entry["index"]] = FALLBACK_HYDE_WEIGHT

        # Build per-query RRF weights list
        per_query_rrf_weights = [
            query_weight_lookup.get(item["index"], FALLBACK_HYDE_WEIGHT)
            for item in valid_query_data
        ]
        logger.info(f"[PIPELINE] Per-query RRF weights: {per_query_rrf_weights}")

        yield StageUpdate(
            stage=PipelineStage.EMBEDDED.value,
            message=f"Generated {len(valid_query_data)} dense + {len(valid_sparse_data)} sparse embeddings",
            progress=0.35,
        )

        # ================================================================
        # STAGE 4: Hybrid Search with Multi-Bucket Time Strategy
        # ================================================================
        logger.info("[PIPELINE] Stage 4: Hybrid Search")
        yield StageUpdate(
            stage=PipelineStage.SEARCHING.value,
            message="Searching podcast database...",
            progress=0.4,
        )

        try:
            # Get time filter from analysis
            time_filter = llm_analysis.get('time_filter')

            # ================================================================
            # Determine recall_ratio and allow_relaxed_recall based on query intent
            # ================================================================
            recall_ratio = 0.0
            strict_time_constraints = False

            if time_filter and time_filter.get("has_time_constraint"):
                mode = (time_filter.get("mode") or "").lower()
                gating = (time_filter.get("gating") or "").lower()

                # Handle gating preference from LLM
                if gating == "soft":
                    recall_ratio = time_filter.get("recall_ratio", 0.3)
                elif gating == "hard":
                    strict_time_constraints = True
                    recall_ratio = 0.0

                # Determine strictness based on mode
                if mode in ["latest", "oldest", "on"]:
                    strict_time_constraints = True
                elif mode in ["before", "after", "between", "relative_recent"]:
                    if _is_strict_time_request(question, time_filter):
                        strict_time_constraints = True
                    elif gating != "hard" and recall_ratio == 0.0:
                        recall_ratio = 0.3  

            # Check if we'll have entity filters
            will_have_filter = bool(
                llm_analysis.get('extracted_guests_interviewees') or
                llm_analysis.get('extracted_hosts_creators')
            )
            if will_have_filter and recall_ratio == 0.0 and not strict_time_constraints:
                recall_ratio = 0.3
                logger.info(f"[PIPELINE] Entity filter detected, setting recall_ratio to {recall_ratio:.2f}")

            strict_constraints = strict_time_constraints or (will_have_filter and recall_ratio == 0.0)
            allow_relaxed_recall = not strict_constraints
            logger.info(f"[PIPELINE] Search params: recall_ratio={recall_ratio:.2f}, allow_relaxed={allow_relaxed_recall}, strict={strict_constraints}")

            # Create multi-bucket search plan based on temporal intent
            search_plan = make_time_search_plan(question, time_filter)
            logger.info(f"[PIPELINE] Time search plan: {len(search_plan)} buckets")
            for b in search_plan:
                logger.info(f"  ├─ Bucket '{b['label']}': include_date={b['include_date_clause']}, weight={b['weight']}, range={b.get('post_date_range')}")

            # ================================================================
            # Define async bucket processor for parallel execution
            # ================================================================
            async def process_bucket(bucket: dict) -> tuple:
                """Process a single bucket and return (results, weight, label)."""
                # Derive time filter for this bucket
                bucket_tf = _derive_bucket_time_filter(time_filter, bucket.get('post_date_range'))

                # Build filter for this bucket (including channelTitle for show filtering)
                bucket_filter, bucket_post_range = build_filter(
                    llm_analysis.get('extracted_guests_interviewees', []),
                    llm_analysis.get('extracted_hosts_creators', []),
                    self.unique_personalities,
                    self.unique_authors,
                    time_filter=bucket_tf,
                    include_date_clause=bucket['include_date_clause'],
                    show_name=llm_analysis.get('extracted_show'),
                    unique_shows=self.unique_shows,
                )

                logger.info(f"[PIPELINE] Bucket '{bucket['label']}' filter: {bucket_filter}")

                # Per-bucket allow_relaxed: relax if no filter or global flag allows
                bucket_allow_relaxed = allow_relaxed_recall or (bucket_filter is None)

                # Execute search for this bucket
                nested_results = await concurrent_pinecone_search(
                    self.pinecone_client,
                    valid_query_data,
                    valid_sparse_data,
                    bucket_filter,
                    self.pinecone_k,
                    self.target_per_query,
                    use_hybrid=True,
                    post_date_range=bucket.get('post_date_range'),
                    recall_ratio=recall_ratio,
                    allow_relaxed_recall=bucket_allow_relaxed,
                )

                # Combine results for this bucket using per-query RRF weights
                bucket_weights = per_query_rrf_weights if len(per_query_rrf_weights) == len(nested_results) else [1.0] * len(nested_results)
                if len(per_query_rrf_weights) != len(nested_results):
                    logger.warning(f"[PIPELINE] Mismatch between query weights ({len(per_query_rrf_weights)}) and nested results ({len(nested_results)}); using uniform weights for bucket '{bucket['label']}'")

                bucket_combined = combine_pinecone_results(
                    nested_results,
                    per_list_weights=bucket_weights,
                    top_k=self.max_chunks_before_rerank
                )

                # ============================================================
                # RECENCY-FIRST BUCKET: Sort by date descending
                # ============================================================
                if bucket.get('sort_by_date'):
                    def get_pdnumeric(item):
                        pd = (item.get('metadata') or {}).get('pdnumeric', 0)
                        if not pd:
                            pd = item.get('pdnumeric', 0)
                        return pd or 0

                    # Log dates before sorting for diagnostics
                    if bucket_combined:
                        dates_found = [get_pdnumeric(item) for item in bucket_combined]
                        valid_dates = [d for d in dates_found if d > 0]
                        if valid_dates:
                            logger.info(f"[RECENCY] Bucket '{bucket['label']}' date range: {min(valid_dates)} to {max(valid_dates)}")
                        else:
                            logger.warning(f"[RECENCY] Bucket '{bucket['label']}' has NO valid dates!")

                    # Sort by date descending (newest first)
                    bucket_combined = sorted(bucket_combined, key=get_pdnumeric, reverse=True)

                    # Limit to configured amount
                    bucket_limit = bucket.get('limit', RECENCY_FIRST_BUCKET_LIMIT)
                    bucket_combined = bucket_combined[:bucket_limit]

                    # Log top items after sorting
                    if bucket_combined:
                        top_items = bucket_combined[:3]
                        logger.info(f"[RECENCY] Bucket '{bucket['label']}' top 3 after date sort:")
                        for i, item in enumerate(top_items):
                            meta = item.get('metadata', {})
                            title = meta.get('episodeTitle', meta.get('title', 'Unknown'))[:50]
                            pd = get_pdnumeric(item)
                            score = item.get('hybrid_score', item.get('score', 0))
                            logger.info(f"  [{i}] {pd} | score={score:.3f} | {title}")

                    logger.info(f"[PIPELINE] Recency-first bucket '{bucket['label']}': sorted by date, limited to {len(bucket_combined)} items")

                # Tag results with bucket label
                for item in bucket_combined:
                    item['bucket'] = bucket['label']

                return bucket_combined, bucket['weight'], bucket['label']

            # ================================================================
            # Execute all buckets in PARALLEL
            # ================================================================
            logger.info(f"[PIPELINE] Processing {len(search_plan)} buckets in parallel...")
            bucket_tasks = [process_bucket(b) for b in search_plan]
            bucket_results = await asyncio.gather(*bucket_tasks)

            # Unpack results
            per_bucket_results = []
            per_bucket_weights = []
            bucket_counts = {}
            for bucket_fused, weight, label in bucket_results:
                per_bucket_results.append(bucket_fused)
                per_bucket_weights.append(weight)
                bucket_counts[label] = len(bucket_fused)

            logger.info(f"[PIPELINE] Parallel bucket processing complete. Bucket counts: {bucket_counts}")

            # Fuse all buckets using RRF with weights
            combined_results = combine_pinecone_results(
                per_bucket_results,
                per_list_weights=per_bucket_weights,
                top_k=self.max_chunks_before_rerank
            )

            # ================================================================
            # SAFETY SEARCH FALLBACK: When entity filters return zero results
            # ================================================================
            if len(combined_results) == 0 and will_have_filter:
                logger.warning("[PIPELINE] Zero results with entity filter, running unfiltered safety search...")
                safety_results = await concurrent_pinecone_search(
                    self.pinecone_client,
                    valid_query_data[:1],  # Just use original query
                    valid_sparse_data[:1] if valid_sparse_data else [],
                    None,  
                    150,   
                    50,    
                    use_hybrid=True,
                    post_date_range=None,
                    recall_ratio=0.0,
                    allow_relaxed_recall=True,
                )
                if safety_results and safety_results[0]:
                    combined_results = safety_results[0][:self.max_chunks_before_rerank]
                    for item in combined_results:
                        item.setdefault('retrieval_stage', 'safety_recall')
                        item['bucket'] = 'safety'
                    bucket_counts['safety_recall'] = len(combined_results)
                    logger.info(f"[PIPELINE] Safety search recovered {len(combined_results)} results")

            # ================================================================
            # RECENT BUCKET FALLBACK: When recent bucket has insufficient results
            # ================================================================
            recent_hits = bucket_counts.get('recent', 0)
            recent_plan_present = any(b.get('label') == 'recent' for b in search_plan)
            mode_lower = (time_filter or {}).get('mode', 'none').lower() if time_filter else 'none'
            pref_lower = ((time_filter or {}).get('sort_preference') or '').lower() if time_filter else ''
            fallback_triggered = False

            if recent_plan_present and (mode_lower == 'latest' or pref_lower == 'latest') and recent_hits < RECENT_MIN_RESULTS:
                logger.info(
                    f"[PIPELINE] Recent bucket delivered {recent_hits} results (< {RECENT_MIN_RESULTS}). "
                    "Running relaxed fallback search without recency clamp."
                )
                # Create fallback plan without time constraints
                fallback_plan = [{'label': 'all', 'include_date_clause': False, 'post_date_range': None, 'weight': 1.0}]

                # Process fallback bucket
                fallback_results = await asyncio.gather(*[process_bucket(b) for b in fallback_plan])
                fallback_bucket_results = []
                for bucket_fused, weight, label in fallback_results:
                    fallback_bucket_results.append(bucket_fused)
                    bucket_counts[f'fallback_{label}'] = len(bucket_fused)

                combined_results = combine_pinecone_results(
                    fallback_bucket_results,
                    top_k=self.max_chunks_before_rerank
                )
                fallback_triggered = True
                logger.info(f"[PIPELINE] Recency fallback activated; using {len(combined_results)} relaxed search results")

            # Apply time-based sorting for latest/oldest after fusion
            combined_results = apply_time_sort_and_limit(combined_results, time_filter if not fallback_triggered else None)

            # ================================================================
            # EPISODE CAP AND BUCKET QUOTA ENFORCEMENT
            # ================================================================
            combined_results = enforce_episode_cap_and_bucket_quota(
                combined_results,
                per_episode_cap=PER_EPISODE_CAP,
                min_per_bucket=MIN_PER_BUCKET
            )

            logger.info(f"[PIPELINE] Stage 4 complete: {len(combined_results)} candidates after RRF + time sort + episode cap")

            # ==================================================================
            # PHASE 4: HARD CHUNK EXCLUSION (with Safety Fallback)
            # ==================================================================

            excluded_ids = memory.get_excluded_ids()  # Last 5 turns

            if excluded_ids:
                original_count = len(combined_results)

                # HARD FILTER: Remove excluded chunks entirely
                fresh_results = [
                    c for c in combined_results
                    if c.get("id") not in excluded_ids
                ]

                excluded_count = original_count - len(fresh_results)
                logger.info(f"[FILTER] Hard exclusion: {excluded_count} chunks excluded (window: 5 turns)")

                # SAFETY FALLBACK: If we filtered too aggressively, relax the filter
                MIN_RESULTS_THRESHOLD = 3

                if len(fresh_results) < MIN_RESULTS_THRESHOLD and original_count > 0:
                    logger.warning(
                        f"[FILTER] Only {len(fresh_results)} results after exclusion. "
                        f"Falling back to deprioritization."
                    )
                    # Fallback: Deprioritize instead of exclude
                    non_excluded = [c for c in combined_results if c.get("id") not in excluded_ids]
                    excluded = [c for c in combined_results if c.get("id") in excluded_ids]
                    combined_results = non_excluded + excluded
                    logger.info(f"[FILTER] Fallback: {len(excluded)} chunks moved to end (deprioritized)")
                else:
                    combined_results = fresh_results
                    logger.info(f"[FILTER] Retained {len(fresh_results)} fresh chunks")

            # ==================================================================
            # SOFT DEPRIORITIZATION for older shown chunks (> 5 turns ago)
            # ==================================================================

            shown_artifact_ids = set(memory.shown_artifacts) if memory.shown_artifacts else set()
            soft_deprioritize_ids = shown_artifact_ids - excluded_ids if excluded_ids else shown_artifact_ids

            if soft_deprioritize_ids:
                non_shown = [c for c in combined_results if c.get("id") not in soft_deprioritize_ids]
                shown_old = [c for c in combined_results if c.get("id") in soft_deprioritize_ids]
                combined_results = non_shown + shown_old
                if shown_old:
                    logger.info(f"[FILTER] Soft deprioritized {len(shown_old)} older shown chunks")

            # ==================================================================
            # END EXCLUSION LOGIC
            # ==================================================================
        except Exception as e:
            logger.error(f"[PIPELINE] Search failed: {e}")
            yield StageUpdate(
                stage=PipelineStage.ERROR.value,
                message=f"Search failed: {str(e)}",
                progress=0.4,
                data={"error": str(e)},
            )
            return

        yield StageUpdate(
            stage=PipelineStage.SEARCHED.value,
            message=f"Found {len(combined_results)} candidate clips",
            progress=0.55,
        )

        if not combined_results:
            yield StageUpdate(
                stage=PipelineStage.COMPLETE.value,
                message="No results found",
                progress=1.0,
                data={
                    "answer": "I couldn't find any podcast clips matching your question. Try rephrasing or asking about a different topic.",
                    "confidence": 0.0,
                    "chunk": None,
                    "total_time": time.time() - pipeline_start,
                    "is_followup": is_followup,
                    "resolved_query": resolved_query,
                }
            )
            return

        # ================================================================
        # STAGE 5: RDS Fetch
        # ================================================================
        logger.info("[PIPELINE] Stage 5: RDS Fetch")
        yield StageUpdate(
            stage=PipelineStage.FETCHING.value,
            message="Loading clip details...",
            progress=0.6,
        )

        try:
            final_keys = get_final_chunk_keys(combined_results, self.max_chunks_before_rerank)
            logger.info(f"[PIPELINE] Fetching {len(final_keys)} chunks from RDS")
            db_results = batch_get_rds_items(final_keys)
            enriched_chunks = merge_db_and_pinecone_data(final_keys, db_results)
            logger.info(f"[PIPELINE] Stage 5 complete: {len(enriched_chunks)} enriched chunks")
        except Exception as e:
            logger.error(f"[PIPELINE] RDS fetch failed: {e}")
            yield StageUpdate(
                stage=PipelineStage.ERROR.value,
                message=f"Failed to load clip details: {str(e)}",
                progress=0.6,
                data={"error": str(e)},
            )
            return

        yield StageUpdate(
            stage=PipelineStage.FETCHED.value,
            message=f"Loaded {len(enriched_chunks)} clips with full details",
            progress=0.7,
        )

        if not enriched_chunks:
            yield StageUpdate(
                stage=PipelineStage.COMPLETE.value,
                message="No clips found in database",
                progress=1.0,
                data={
                    "answer": "I found some potential matches but couldn't load the clip details. Please try again.",
                    "confidence": 0.0,
                    "chunk": None,
                    "total_time": time.time() - pipeline_start,
                    "is_followup": is_followup,
                    "resolved_query": resolved_query,
                }
            )
            return

        # ================================================================
        # STAGE 6: Reranking
        # ================================================================
        logger.info("[PIPELINE] Stage 6: Reranking with Cohere")
        yield StageUpdate(
            stage=PipelineStage.RERANKING.value,
            message="Finding the best matches...",
            progress=0.75,
        )

        try:
            MIN_CHUNK_LENGTH_MS = 30000   # 30 seconds
            MAX_CHUNK_LENGTH_MS = 600000  # 10 minutes

            filtered_chunks = [
                c for c in enriched_chunks
                if MIN_CHUNK_LENGTH_MS <= (c.get('chunkLength') or 0) < MAX_CHUNK_LENGTH_MS
            ]

            if not filtered_chunks:
                # Fallback: try with just max filter (some clips may not have length data)
                filtered_chunks = [
                    c for c in enriched_chunks
                    if (c.get('chunkLength') or 0) < MAX_CHUNK_LENGTH_MS
                ]

            if not filtered_chunks:
                filtered_chunks = enriched_chunks  

            logger.info(f"[PIPELINE] Duration filter: {len(enriched_chunks)} → {len(filtered_chunks)} chunks (30s-10min)")

            logger.info(f"[PIPELINE] Duration filter: {len(filtered_chunks)} chunks after filter")

            # =================================================================
            # STAGE 6: COHERE RERANKING (controlled by RERANKER_ENABLED)
            # =================================================================
            if RERANKER_ENABLED:
                logger.info(f"[PIPELINE] Reranking {len(filtered_chunks)} chunks (filtered by duration)")
                reranked = await asyncio.to_thread(
                    rerank_chunks_cohere,
                    self.pinecone_client,
                    resolved_query,
                    filtered_chunks[:self.max_chunks_before_rerank],
                    self.reranker_top_n,
                    llm_analysis,
                    list(shown_artifact_ids) if shown_artifact_ids else None,
                )
                logger.info(f"[PIPELINE] Stage 6 complete: {len(reranked)} reranked results")
            else:
                # Skip reranking - take top 25 directly
                reranked = filtered_chunks[:25]
                logger.info(f"[PIPELINE] Stage 6 (RERANKER DISABLED): Taking top {len(reranked)} chunks directly")
        except Exception as e:
            logger.error(f"[PIPELINE] Reranking failed: {e}")
            # Fallback to unreranked results
            reranked = filtered_chunks[:self.max_chunks_for_selection]

        yield StageUpdate(
            stage=PipelineStage.RERANKED.value,
            message=f"Reranked to top {len(reranked)} clips",
            progress=0.85,
        )

        # ================================================================
        # STAGE 6.5: Apply Hybrid Metadata-Aware Scoring
        # ================================================================
        logger.info("[PIPELINE] Stage 6.5: Applying hybrid metadata-aware scoring")

        # Apply hybrid scoring - this re-sorts by weighted combination of signals
        reranked = apply_hybrid_metadata_scoring(reranked, llm_analysis)

        # Build recency_metadata for selection LLM transparency
        recency_priority = (time_filter or {}).get("recency_priority", "none")
        topic_present = (time_filter or {}).get("topic_present", False)

        recency_metadata = {
            "recency_satisfied": True,  
            "fallback_triggered": False,
            "relevance_floor_applied": False,  
            "original_count": len(reranked),
            "post_floor_count": len(reranked),
            "recency_priority": recency_priority,
            "topic_present": topic_present,
            "hybrid_scoring_applied": True,
        }

        logger.info(f"[PIPELINE] Stage 6.5 complete: hybrid_scoring=True, "
                   f"recency_priority={recency_priority}, topic={topic_present}")

        # ================================================================
        # STAGE 7: Selection + Memory Update
        # ================================================================
        logger.info("[PIPELINE] Stage 7: Selection + Memory Update")
        yield StageUpdate(
            stage=PipelineStage.SELECTING.value,
            message="Selecting the perfect clip...",
            progress=0.9,
        )

        try:
            selection_result = await select_and_update_memory(
                self.gemini_client,
                question,
                resolved_query,
                reranked[:self.max_chunks_for_selection],
                memory,
                is_followup,
                recency_metadata=recency_metadata,  
            )

            # Get chosen chunk
            chosen_index = selection_result.chosen_index
            if chosen_index < len(reranked):
                chosen_chunk = reranked[chosen_index]
            else:
                logger.warning(f"[PIPELINE] Invalid selection index {chosen_index}, using first chunk")
                chosen_chunk = reranked[0] if reranked else {}

            # ==================================================================
            # PHASE 3: SYNCHRONOUS STATE UPDATE
            # ==================================================================

            logger.info("[STATE UPDATE] Synchronously updating SearchState...")

            # 1. Extract data from Selection Result (already computed by selection LLM)
            new_entities = selection_result.extracted_entities if selection_result else []
            new_themes = selection_result.turn_themes if selection_result else []


            chunk_guests_raw = chosen_chunk.get('guests', []) if chosen_chunk else []
            chunk_hosts_raw = chosen_chunk.get('hosts', []) if chosen_chunk else []

            # Normalize to lists (handle both string and list formats)
            if isinstance(chunk_guests_raw, str):
                chunk_guests = [g.strip() for g in chunk_guests_raw.split(',') if g.strip()]
            else:
                chunk_guests = chunk_guests_raw if chunk_guests_raw else []

            if isinstance(chunk_hosts_raw, str):
                chunk_hosts = [h.strip() for h in chunk_hosts_raw.split(',') if h.strip()]
            else:
                chunk_hosts = chunk_hosts_raw if chunk_hosts_raw else []

            chunk_entities = chunk_guests + chunk_hosts

            # Combine: selection entities + chunk entities (deduplicated)
            all_new_entities = list(dict.fromkeys(new_entities + chunk_entities))[:5]

            is_topic_shift = not is_followup
            if not is_topic_shift and all_new_entities:
                current_entities_lower = {e.lower() for e in memory.search_state.current_entities}
                new_entities_lower = {e.lower() for e in all_new_entities}
                overlap = current_entities_lower.intersection(new_entities_lower)
                if len(overlap) == 0 and len(current_entities_lower) > 0:
                    is_topic_shift = True
                    logger.info("[STATE UPDATE] Detected implicit topic shift (no entity overlap)")

            # 3. Update SearchState (instant Python operation - microseconds)
            memory.search_state.update_entities(all_new_entities, is_topic_shift)

            if new_themes:
                memory.search_state.update_topic(new_themes[0])

            memory.search_state.update_time_filter(llm_analysis.get('time_filter'))
            memory.search_state.update_artifact(
                chosen_chunk.get('id') if chosen_chunk else None,
                chosen_chunk.get('episode_title') if chosen_chunk else None
            )
            memory.search_state.last_was_followup = is_followup

            # 4. Add to Exclusion Window (instant operation)
            if chosen_chunk and chosen_chunk.get('id'):
                memory.add_to_exclusion_window(chosen_chunk.get('id'))

            # 5. Log the state update
            logger.info("[STATE UPDATE] Complete:")
            logger.info(f"  ├─ Entities: {memory.search_state.current_entities}")
            logger.info(f"  ├─ Topic: {memory.search_state.current_topic}")
            logger.info(f"  ├─ Was follow-up: {is_followup}")
            logger.info(f"  ├─ Topic shift: {is_topic_shift}")
            logger.info(f"  └─ Exclusion window size: {len(memory.excluded_artifact_window)}")

            # ==================================================================
            # END SYNCHRONOUS STATE UPDATE
            # ==================================================================


            from engine.schemas import BranchMemoryUpdate

            # Phase 2: Extract enhanced fields from selection result
            key_quotes = getattr(selection_result, 'key_quotes', []) or []
            topics_covered = getattr(selection_result, 'topics_covered', []) or []
            notable_examples = getattr(selection_result, 'notable_examples', []) or []

            clip_memory_update = BranchMemoryUpdate(
                turn_summary=selection_result.turn_summary[:500],  # Phase 2: increased from 200 to 500
                action_type="clip_shown",
                action_target_id=chosen_chunk.get('id'),
                action_target_title=chosen_chunk.get('episode_title'),
                published_date=chosen_chunk.get('published_date') or chosen_chunk.get('publishedDate'),  # From Pinecone metadata
                entities_mentioned=selection_result.extracted_entities[:10],  # Phase 2: increased from 5 to 10
                topics_discussed=selection_result.turn_themes[:5],  # Phase 2: increased from 3 to 5
                is_topic_shift=is_topic_shift,
                suggested_phase="deep_dive" if is_followup else "discovery",
                # Option A: Enhanced context fields (Phase 2)
                key_quotes=key_quotes[:3],
                topics_covered=topics_covered[:5],
                notable_examples=notable_examples[:3],
            )


            memory.apply_branch_memory_update(
                update=clip_memory_update,
                route_chosen="clip_search",
                route_confidence=router_output.confidence,
                query_intent=router_output.query_intent,
                user_question=question,
                resolved_query=resolved_query,
            )

            logger.info("[AGENT] Clip search memory updated via unified BranchMemoryUpdate")
            logger.info(f"[AGENT] Action: {clip_memory_update.action_type} | Target: {clip_memory_update.action_target_title[:50] if clip_memory_update.action_target_title else 'N/A'}...")

        except Exception as e:
            logger.error(f"[PIPELINE] Selection failed: {e}")
            chosen_chunk = reranked[0] if reranked else {}
            selection_result = None

        total_time = time.time() - pipeline_start

        # ================================================================
        # FINAL: Complete Response
        # ================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[PIPELINE] COMPLETE")
        logger.info("=" * 80)
        logger.info(f"[PIPELINE] Total time: {total_time:.2f}s")
        logger.info(f"[PIPELINE] Final result:")
        logger.info(f"  ├─ Chosen clip: {chosen_chunk.get('episode_title', 'Unknown')[:60]}...")
        logger.info(f"  ├─ Podcast: {chosen_chunk.get('podcast_title', 'Unknown')}")
        logger.info(f"  ├─ Confidence: {selection_result.confidence if selection_result else 0.5}")
        logger.info(f"  ├─ Is follow-up: {is_followup}")
        logger.info(f"  └─ Memory state: {memory.turn_count} turns, {len(memory.entities)} entities")
        logger.info("=" * 80)

        # Get current turn ID for recommendations
        current_turn_id = f"T{memory.turn_count}"

        # Safety: ensure answer is never empty (use 'or' to handle empty strings)
        final_answer = (selection_result.answer if selection_result else None) or "Here's a relevant clip that addresses your question."

        yield StageUpdate(
            stage=PipelineStage.COMPLETE.value,
            message="Done!",
            progress=1.0,
            data={
                "answer": final_answer,
                "confidence": selection_result.confidence if selection_result else 0.7,
                "chunk": chosen_chunk,
                "total_time": total_time,
                "is_followup": is_followup,
                "resolved_query": resolved_query,
                "turn_id": current_turn_id,  
                "branch": "clip_search",  
            }
        )

        # ================================================================
        # STAGE 8: Generate Recommendations (non-blocking, after complete)
        # ================================================================
        if selection_result and len(reranked) > 1:
            logger.info("[PIPELINE] Stage 8: Generating recommendations...")
            try:
                recommendations = await generate_recommendations(
                    self.gemini_client,
                    question,
                    resolved_query,
                    reranked[:self.max_chunks_for_selection],
                    selection_result.chosen_index,
                    selection_result.answer,
                    memory,
                    is_followup,
                )

                if recommendations.recommendations:
                    # Store recommendations for later retrieval when user clicks
                    # Include original context for accurate memory updates
                    await store_recommendations(
                        session_id,
                        current_turn_id,
                        recommendations,
                        reranked[:self.max_chunks_for_selection],
                        original_question=question,
                        resolved_query=resolved_query,
                    )

                    # Send recommendations to frontend
                    yield StageUpdate(
                        stage=PipelineStage.RECOMMENDATIONS.value,
                        message="Recommendations ready",
                        progress=1.0,
                        data={
                            "turn_id": current_turn_id,
                            "recommendations": [
                                {
                                    "index": i,
                                    "prompt": rec.prompt,
                                    "chunk_index": rec.chunk_index,
                                }
                                for i, rec in enumerate(recommendations.recommendations)
                            ],
                        }
                    )
                    logger.info(f"[PIPELINE] Stage 8 complete: {len(recommendations.recommendations)} recommendations sent")
                else:
                    logger.info("[PIPELINE] Stage 8: No recommendations generated")

            except Exception as e:
                logger.error(f"[PIPELINE] Recommendation generation failed: {e}")
                

    async def ask(self, session_id: str, question: str) -> ChatResponse:
        """
        Non-streaming version - processes all stages and returns final result.

        Args:
            session_id: Unique session identifier for memory
            question: User's question

        Returns:
            ChatResponse with answer, clip details, and metadata
        """
        final_result = None
        error_message = None

        async for update in self.ask_streaming(session_id, question):
            if update.stage == PipelineStage.COMPLETE.value:
                final_result = update.data
            elif update.stage == PipelineStage.ERROR.value:
                error_message = update.message

        if error_message:
            return ChatResponse(
                answer=error_message,
                confidence=0.0,
                total_time=0.0,
                is_followup=False,
            )

        if not final_result:
            raise RuntimeError("Pipeline failed to complete")

        chunk = final_result.get("chunk", {}) or {}

        return ChatResponse(
            answer=final_result["answer"],
            confidence=final_result["confidence"],
            video_url=chunk.get("chunkAudioUrl"),
            video_chunk_path=chunk.get("videoChunkPath"),
            episode_title=chunk.get("episode_title"),
            podcast_title=chunk.get("podcast_title"),
            speakers=chunk.get("speakers", []) or [],
            guests=chunk.get("guests", []) or [],
            hosts=[chunk.get("hosts")] if isinstance(chunk.get("hosts"), str) else (chunk.get("hosts") or []),
            published_date=chunk.get("published_date"),
            chunk_start_time=chunk.get("startMs") / 1000.0 if chunk.get("startMs") else None,
            chunk_end_time=chunk.get("endMs") / 1000.0 if chunk.get("endMs") else None,
            chunk_length_ms=int(chunk.get("chunkLength")) if chunk.get("chunkLength") else None,
            total_time=final_result["total_time"],
            is_followup=final_result["is_followup"],
            resolved_query=final_result["resolved_query"],
        )

    def reset_session(self, session_id: str) -> None:
        """Clear conversation memory for a session."""
        memory_store.reset(session_id)
        logger.info(f"[AGENT] Reset memory for session {session_id}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session's current state."""
        memory = memory_store.get(session_id)
        if not memory:
            return None

        return {
            "session_id": session_id,
            "turn_count": memory.turn_count,
            "recent_entities": memory.get_recent_entities(10),
            "themes": memory.conversation_themes,
            "shown_artifacts_count": len(memory.shown_artifacts),
            "created_at": memory.created_at.isoformat(),
        }

    # ==========================================================================
    # PHASE 5: BACKGROUND MEMORY COMPRESSION
    # ==========================================================================

    async def compress_memory_background(self, session_id: str) -> None:
        """
        Background task for heavy memory operations.

        This runs AFTER the response is sent, so it doesn't affect latency.
        Only handles non-critical operations:
        - Compressing old turns
        - Decaying stale entities
        - Cleaning up exclusion window

        Critical operations (state updates) are done synchronously in Phase 3.
        """
        try:
            memory = memory_store.get(session_id)
            if not memory:
                logger.debug(f"[BACKGROUND] No memory found for session {session_id}")
                return

            logger.debug(f"[BACKGROUND] Starting compression for session {session_id}")

            # 1. Decay stale entities (not updated in 5+ turns)
            memory.search_state.decay_if_stale()

            # 2. Compress old turns if needed
            if len(memory.recent_turns) > MAX_RECENT_TURNS:
                # Existing compression logic
                turns_to_compress = memory.recent_turns[:-MAX_RECENT_TURNS]
                memory.recent_turns = memory.recent_turns[-MAX_RECENT_TURNS:]

                # Create compressed summary
                compressed_entries = []
                for turn in turns_to_compress:
                    compressed_entries.append(
                        f"- T{turn.turn_id}: \"{turn.user_question[:40]}...\" -> {turn.answer_summary[:60]}..."
                    )

                if memory.compressed_summary:
                    memory.compressed_summary += "\n" + "\n".join(compressed_entries)
                else:
                    memory.compressed_summary = "\n".join(compressed_entries)

                # Truncate if too long
                MAX_COMPRESSED_CHARS = 3000
                if len(memory.compressed_summary) > MAX_COMPRESSED_CHARS:
                    memory.compressed_summary = memory.compressed_summary[-MAX_COMPRESSED_CHARS:]

                logger.info(f"[BACKGROUND] Compressed {len(turns_to_compress)} old turns")

            # 3. Clean up exclusion window (already done in add_to_exclusion_window, but double-check)
            original_window_size = len(memory.excluded_artifact_window)
            memory.excluded_artifact_window = [
                (aid, turn) for aid, turn in memory.excluded_artifact_window
                if memory.turn_count - turn < memory.EXCLUSION_WINDOW_TURNS
            ]
            if len(memory.excluded_artifact_window) < original_window_size:
                cleaned = original_window_size - len(memory.excluded_artifact_window)
                logger.debug(f"[BACKGROUND] Cleaned {cleaned} old exclusion entries")

            logger.debug(f"[BACKGROUND] Compression complete for session {session_id}")

        except Exception as e:
            # Non-critical - log and continue
            logger.error(f"[BACKGROUND] Compression failed for {session_id}: {e}")
