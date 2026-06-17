# retrieval/search_filter.py

import re
import json
import time
import logging
from thefuzz import process, fuzz
import os
from datetime import datetime, timezone
from config import (
    WRATIO_THRESHOLD, TOKEN_SET_MIN_SCORE, SPECIFICITY_RATIO_THRESHOLD,
    FUZZY_MATCH_CANDIDATE_LIMIT, PERSONALITY_METADATA_FIELD, AUTHOR_METADATA_FIELD,
    RECENT_WINDOW_DAYS_DEFAULT
)
PD_NUMERIC_FIELD = os.getenv("PUBLISHED_NUMERIC_FIELD", "pdnumeric")

def _parse_iso_to_yyyymmdd(iso_s: str) -> int:
    """Convert ISO date string to YYYYMMDD integer for pdnumeric comparison"""
    if not iso_s:
        return None
    try:
        # Handle both YYYY-MM-DD and full ISO formats
        if 'T' in iso_s:
            dt = datetime.fromisoformat(iso_s.replace("Z", "+00:00")).astimezone(timezone.utc)
        else:
            dt = datetime.strptime(iso_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.year * 10000 + dt.month * 100 + dt.day
    except Exception as e:
        logging.warning(f"Failed to parse date '{iso_s}': {e}")
        return None

def _build_date_clause(time_filter: dict | None):
    """
    Build Pinecone filter clause for pdnumeric field and post-filter range.
    Returns: (pinecone_date_clause, post_date_range)
    """
    if not time_filter or not time_filter.get("has_time_constraint"):
        return None, None

    mode = (time_filter.get("mode") or "none").lower()
    start_iso = time_filter.get("start_date_utc")
    end_iso = time_filter.get("end_date_utc")

    # Convert to YYYYMMDD integers
    start_num = _parse_iso_to_yyyymmdd(start_iso) if start_iso else None
    end_num = _parse_iso_to_yyyymmdd(end_iso) if end_iso else None

    pinecone_clause = None
    
    # Build appropriate Pinecone filter based on mode
    if mode == "before" and end_num is not None:
        # Exclusive upper bound
        pinecone_clause = {PD_NUMERIC_FIELD: {"$lt": end_num}}
        logging.info(f"Time filter: before {end_iso} (pdnumeric < {end_num})")
        
    elif mode == "after" and start_num is not None:
        # Inclusive lower bound
        pinecone_clause = {PD_NUMERIC_FIELD: {"$gte": start_num}}
        logging.info(f"Time filter: after {start_iso} (pdnumeric >= {start_num})")
        
    elif mode == "between" and start_num is not None and end_num is not None:
        # Inclusive range
        pinecone_clause = {"$and": [
            {PD_NUMERIC_FIELD: {"$gte": start_num}},
            {PD_NUMERIC_FIELD: {"$lte": end_num}}
        ]}
        logging.info(f"Time filter: between {start_iso} and {end_iso} ({start_num} <= pdnumeric <= {end_num})")
        
    elif mode == "on" and start_num is not None:
        # Exact match for single day
        pinecone_clause = {PD_NUMERIC_FIELD: {"$eq": start_num}}
        logging.info(f"Time filter: on {start_iso} (pdnumeric = {start_num})")
        
    elif mode == "relative_recent" and start_num is not None:
        # Recent items (last N days/months)
        pinecone_clause = {PD_NUMERIC_FIELD: {"$gte": start_num}}
        if end_num:
            pinecone_clause = {"$and": [
                {PD_NUMERIC_FIELD: {"$gte": start_num}},
                {PD_NUMERIC_FIELD: {"$lte": end_num}}
            ]}
        logging.info(f"Time filter: recent from {start_iso} (pdnumeric >= {start_num})")

    elif mode == "latest":
        # LATEST mode: Filter to recent content
        # Validate LLM-provided date - if invalid (today/future), use default 21-day window
        from datetime import timedelta
        today = datetime.now(timezone.utc)
        today_num = today.year * 10000 + today.month * 100 + today.day
        week_ago_num = today_num - 7  # If within 7 days of today, likely wrong

        # Default: 21 days ago (RECENT_WINDOW_DAYS_DEFAULT from config)
        default_cutoff = today - timedelta(days=RECENT_WINDOW_DAYS_DEFAULT)
        default_start = default_cutoff.year * 10000 + default_cutoff.month * 100 + default_cutoff.day

        # Validate: LLM date must be in the past (more than 7 days ago) to be trusted
        if start_num is not None and start_num < week_ago_num:
            # Valid past date - use it
            pinecone_clause = {PD_NUMERIC_FIELD: {"$gte": start_num}}
            logging.info(f"Time filter: latest (LLM-provided, pdnumeric >= {start_num})")
        else:
            # Invalid date (today/future/within 7 days) or no date - use default
            if start_num is not None:
                logging.warning(f"Time filter: latest but LLM returned invalid date={start_num} (today={today_num}), using {RECENT_WINDOW_DAYS_DEFAULT}-day default")
            pinecone_clause = {PD_NUMERIC_FIELD: {"$gte": default_start}}
            # CRITICAL FIX: Update start_num to match Pinecone filter so post_date_range is consistent!
            start_num = default_start
            logging.info(f"Time filter: latest (default {RECENT_WINDOW_DAYS_DEFAULT} days, pdnumeric >= {default_start})")

    elif mode == "oldest":
        # OLDEST mode: Only filter if explicit end_date provided
        # Otherwise, sorting is handled post-fusion by apply_time_sort_and_limit()
        if end_num is not None:
            pinecone_clause = {PD_NUMERIC_FIELD: {"$lte": end_num}}
            logging.info(f"Time filter: oldest with explicit date (pdnumeric <= {end_num})")
        else:
            # No default filter for oldest - let sorting handle it
            logging.info("Time filter: oldest mode (no Pinecone filter, will sort post-fusion)")

    # Post-filter range for client-side safety net
    post_date_range = None
    if mode in {"before", "after", "between", "on", "relative_recent", "latest", "oldest"}:
        post_date_range = {"start": start_num, "end": end_num}

    return pinecone_clause, post_date_range

def normalize_name(name: str) -> str:
    if not isinstance(name, str): return ""
    name = name.lower()
    name = re.sub(r'\b(dr|prof|gen|mr|mrs|ms|rev|hon|jr|sr|[ivx]+)\b\.?', '', name, flags=re.IGNORECASE)
    name = name.replace('.', '')
    # ADD THESE TWO LINES:
    name = re.sub(r'\b(podcast|show|experience)\b', '', name)  # Remove show words
    name = re.sub(r'^\bthe\b\s+', '', name)  # Remove leading "the"
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    return name

# --- Enhanced Helper Function for Fuzzy Matching (Hybrid Approach) ---
def find_closest_metadata_matches(
    input_items,
    known_metadata_values,
    field_name,
    limit = FUZZY_MATCH_CANDIDATE_LIMIT
    ):
    """
    Uses a hybrid fuzzy matching approach checking top N candidates:
    WRatio primary, token_set_ratio check, and a specificity guard.
    Allows multiple valid matches per input item if scores are close.
    """
    matched_exact_values = set()
    valid_input_items = [str(item) for item in input_items if item and isinstance(item, str)]

    if not valid_input_items:
        logging.info(f"No valid input items provided for field '{field_name}'.")
        return []
    if not known_metadata_values:
        logging.warning(f"Known metadata value list is empty for field '{field_name}'. Cannot perform matching.")
        return []

    normalized_to_original_map = {}
    normalized_choices = []
    for k in known_metadata_values:
        normalized_k = normalize_name(str(k))
        if normalized_k:
            if normalized_k not in normalized_to_original_map:
                 normalized_to_original_map[normalized_k] = str(k)
            if normalized_k not in normalized_choices:
                 normalized_choices.append(normalized_k)

    if not normalized_choices:
         logging.warning(f"Known metadata values list yielded no valid normalized choices for field '{field_name}'.")
         return []

    logging.info(f"Attempting Hybrid fuzzy match for {field_name}: {valid_input_items} against {len(normalized_choices)} choices (checking top {limit}).")

    for item in valid_input_items:
        normalized_item = normalize_name(item)
        if not normalized_item: continue

        candidates = process.extract(
            normalized_item,
            normalized_choices,
            scorer=fuzz.WRatio,
            limit=limit
            )

        logging.info(f"Input '{item}' (Normalized='{normalized_item}'): Top {len(candidates)} WRatio candidates: {candidates}")

        found_match_for_item = False
        processed_candidates = 0
        for best_normalized_match, wratio_score in candidates:
            processed_candidates += 1
            if wratio_score < WRATIO_THRESHOLD:
                logging.info(f"  Candidate {processed_candidates}/'{best_normalized_match}' WRatio {wratio_score} below threshold {WRATIO_THRESHOLD}. Stopping checks for input '{item}'.")
                break

            logging.info(f"  Checking Candidate {processed_candidates}/'{best_normalized_match}' (WRatio: {wratio_score})...")


            token_set_score = fuzz.token_set_ratio(normalized_item, best_normalized_match)

            if token_set_score < TOKEN_SET_MIN_SCORE:
                logging.info(f"    Check Failed: TokenSetRatio {token_set_score} < {TOKEN_SET_MIN_SCORE}.")
                continue

            passes_specificity_guard = True
            if token_set_score == 100 and wratio_score >= WRATIO_THRESHOLD:
                passes_specificity_guard = True
            elif wratio_score == 100 or token_set_score == 100:
                simple_ratio = fuzz.ratio(normalized_item, best_normalized_match)
                if len(best_normalized_match) < len(normalized_item) and simple_ratio < SPECIFICITY_RATIO_THRESHOLD:
                    passes_specificity_guard = False
                    logging.warning(f"    Check Failed: Specificity Guard failed for candidate '{normalized_to_original_map.get(best_normalized_match)}'. Input='{item}', WR={wratio_score}, TS={token_set_score}, Ratio={simple_ratio} < {SPECIFICITY_RATIO_THRESHOLD}.")


            if passes_specificity_guard:
                original_casing_match = normalized_to_original_map.get(best_normalized_match)
                if original_casing_match:
                    matched_exact_values.add(original_casing_match)
                    found_match_for_item = True
                    logging.info(f"  + Matched '{field_name}': Input='{item}' -> Candidate Exact='{original_casing_match}' (WR={wratio_score}, TS={token_set_score}, Checks Passed)")
                else:
                     logging.error(f"    Internal Error: Could not find original value for normalized match '{best_normalized_match}'")
            else:
                logging.info(f"    Check Passed Thresholds but Failed Specificity Guard for candidate '{best_normalized_match}'.")

        if not found_match_for_item:
             logging.info(f"No suitable match passed all checks for input '{item}' after checking top {len(candidates)} WRatio candidates.")

    logging.info(f"Final collected exact matches for field '{field_name}': {list(matched_exact_values)}")
    return list(matched_exact_values)

def build_filter(
    llm_extracted_guests,
    llm_extracted_hosts,
    unique_personalities,
    unique_authors,
    time_filter: dict | None = None,
    include_date_clause: bool = True,
    show_name: str | None = None,
    unique_shows: list | None = None,
):
    """
    Build Pinecone filter for clip search.

    Args:
        llm_extracted_guests: Guest names from query analysis
        llm_extracted_hosts: Host names from query analysis
        unique_personalities: Known guest names for fuzzy matching
        unique_authors: Known host names for fuzzy matching
        time_filter: Time filter dict with mode, dates, etc.
        include_date_clause: Whether to include date in Pinecone filter
        show_name: Show/podcast name from query analysis (for channelTitle filter)
        unique_shows: Known show names for fuzzy matching

    Returns:
        (pinecone_filter, post_date_range) tuple
    """
    filter_build_start = time.time()
    filter_parts = []
    final_filter = None

    if llm_extracted_guests or llm_extracted_hosts:
        logging.info(f"Filter Input - Guests: {llm_extracted_guests}, Hosts: {llm_extracted_hosts}")
        guest_input = [str(g) for g in llm_extracted_guests if isinstance(g, str) and g]
        host_input = [str(h) for h in llm_extracted_hosts if isinstance(h, str) and h]


        exact_p_matches = find_closest_metadata_matches(guest_input, unique_personalities, PERSONALITY_METADATA_FIELD) if guest_input else []
        logging.info(f"Fuzzy Matched Personalities: {exact_p_matches}")

        exact_a_matches = find_closest_metadata_matches(host_input, unique_authors, AUTHOR_METADATA_FIELD) if host_input else []
        logging.info(f"Fuzzy Matched Authors: {exact_a_matches}")

        if exact_p_matches or exact_a_matches:
            unique_authors_set = set(unique_authors)
            unique_personalities_set = set(unique_personalities)
            ambiguous_canonical_names = unique_authors_set.intersection(unique_personalities_set)

            if ambiguous_canonical_names:
                logging.info(f"Identified {len(ambiguous_canonical_names)} ambiguous canonical names. Examples: {list(ambiguous_canonical_names)[:5]}")

            p_set = set(exact_p_matches)
            a_set = set(exact_a_matches)
            matched_ambiguous = (p_set | a_set).intersection(ambiguous_canonical_names)
            pure_p_matches = list(p_set - matched_ambiguous)
            pure_a_matches = list(a_set - matched_ambiguous)
            ambiguous_names_list = list(matched_ambiguous)

            logging.info(f"Pure Personality Matches (for $in): {pure_p_matches}")
            logging.info(f"Pure Author Matches (for $in): {pure_a_matches}")
            logging.info(f"Ambiguous Matches (for $or): {ambiguous_names_list}")

            # ================================================================
            # TWO-STAGE FILTER LOGIC:
            # When BOTH guest AND host are specified, use AND for precision
            # When only ONE is specified, use OR for flexibility
            # ================================================================
            has_guest_filter = bool(pure_p_matches)
            has_host_filter = bool(pure_a_matches) or bool(ambiguous_names_list)
            user_wants_both = bool(guest_input) and bool(host_input)

            if user_wants_both and has_guest_filter:
                # ============================================================
                # STRICT AND MODE: User asked for "Guest X on Host Y's show"
                # Example: "Elon Musk episode with Joe Rogan"
                # Filter: guests CONTAINS Elon AND hosts CONTAINS Joe Rogan
                # ============================================================
                logging.info("[FILTER] Using STRICT AND mode - both guest and host specified")

                guest_clause = {PERSONALITY_METADATA_FIELD: {"$in": [p.lower() for p in pure_p_matches]}}

                # Build host clause (handle ambiguous names)
                if pure_a_matches and not ambiguous_names_list:
                    host_clause = {AUTHOR_METADATA_FIELD: {"$in": [a.lower() for a in pure_a_matches]}}
                elif ambiguous_names_list:
                    # Host might be in guests or hosts field
                    host_clause = {"$or": [
                        {PERSONALITY_METADATA_FIELD: {"$in": [a.lower() for a in ambiguous_names_list]}},
                        {AUTHOR_METADATA_FIELD: {"$in": [a.lower() for a in ambiguous_names_list]}}
                    ]}
                else:
                    host_clause = None

                if host_clause:
                    final_filter = {"$and": [guest_clause, host_clause]}
                    logging.info("Constructed STRICT AND filter: guests AND hosts")
                else:
                    final_filter = guest_clause
                    logging.info("Constructed guest-only filter (no host match found)")
            else:
                # ============================================================
                # RELAXED OR MODE: Only one of guest/host specified
                # Example: "episodes about Elon Musk" or "Joe Rogan podcasts"
                # Filter: guests CONTAINS X OR hosts CONTAINS X
                # ============================================================
                logging.info("[FILTER] Using RELAXED OR mode - single entity or no guest match")

                filter_parts = []
                if pure_p_matches:
                    filter_parts.append({PERSONALITY_METADATA_FIELD: {"$in": [p.lower() for p in pure_p_matches]}})
                if pure_a_matches:
                    filter_parts.append({AUTHOR_METADATA_FIELD: {"$in": [a.lower() for a in pure_a_matches]}})
                if ambiguous_names_list:
                    filter_parts.append({"$or": [
                        {PERSONALITY_METADATA_FIELD: {"$in": [a.lower() for a in ambiguous_names_list]}},
                        {AUTHOR_METADATA_FIELD: {"$in": [a.lower() for a in ambiguous_names_list]}}
                    ]})

                if not filter_parts:
                    final_filter = None
                    logging.info("No valid filter parts generated after resolving ambiguities.")
                elif len(filter_parts) == 1:
                    final_filter = filter_parts[0]
                    logging.info("Constructed single-part final filter.")
                else:
                    final_filter = {"$or": filter_parts}
                    logging.info("Constructed multi-part $or final filter (relaxed for people queries).")
        else:
            logging.info("Fuzzy matching did not yield any results for guests or hosts.")
            final_filter = None
    else:
        logging.info("No guests or hosts extracted by LLM. Skipping filter creation.")
        final_filter = None

    # =========================================================================
    # CHANNEL TITLE FILTER: Add show/podcast filter if show_name is provided
    # This enables clip search to filter by podcast (like episode search does)
    # =========================================================================
    channel_filter = None
    if show_name and unique_shows:
        logging.info(f"[CLIP_FILTER] Building channelTitle filter for show: {show_name}")

        normalized_show = normalize_name(show_name)

        # Create normalized to original mapping
        normalized_to_original = {}
        normalized_choices = []
        for s in unique_shows:
            norm_s = normalize_name(str(s))
            if norm_s and norm_s not in normalized_to_original:
                normalized_to_original[norm_s] = str(s)
                normalized_choices.append(norm_s)

        if normalized_show and normalized_choices:
            raw_matches = process.extract(
                normalized_show,
                normalized_choices,
                scorer=fuzz.WRatio,
                limit=5
            )
            # Filter by score threshold manually for compatibility
            show_matches = [(match, score) for match, score in raw_matches if score >= 60]

            logging.info(f"[CLIP_FILTER] Show fuzzy matches: {show_matches}")

            if show_matches:
                matched_shows = []
                for norm_match, score in show_matches:
                    original = normalized_to_original.get(norm_match)
                    if original:
                        matched_shows.append(original)

                if matched_shows:
                    # Build channelTitle filter with case variants
                    channel_variants = set()
                    for show in matched_shows:
                        channel_variants.add(show.lower())

                    channel_filter = {"channelTitle": {"$in": list(channel_variants)}}
                    logging.info(f"[CLIP_FILTER] Channel filter: {list(channel_variants)}")

    # Combine channel filter with people filter using AND
    if channel_filter:
        if final_filter:
            final_filter = {"$and": [final_filter, channel_filter]}
            logging.info("[CLIP_FILTER] Combined people and channel filters with AND")
        else:
            final_filter = channel_filter
            logging.info("[CLIP_FILTER] Using channel filter only (no people filter)")

    date_clause, post_date_range = _build_date_clause(time_filter)
    if date_clause and include_date_clause:
        if final_filter:
            # Combine existing people filter with date filter
            final_filter = {"$and": [final_filter, date_clause]}
            logging.info("Combined people and date filters with AND")
        else:
            final_filter = date_clause
            logging.info("Using date filter only (no people filter)")
    else:
        if date_clause:
            logging.info("Date clause computed but NOT added to Pinecone filter (soft preference mode)")

    # # Combine all filter parts
    # if not filter_parts:
    #     final_filter = None
    #     logging.info("No filter constructed (no names or time constraints)")
    # elif len(filter_parts) == 1:
    #     final_filter = filter_parts[0]
    #     logging.info("Single-clause filter constructed")
    # else:
    #     final_filter = {"$and": filter_parts}
    #     logging.info(f"Multi-clause filter constructed with {len(filter_parts)} parts")

    if final_filter:
        try:
            filter_log_str = json.dumps(final_filter, indent=2)
            logging.info(f"Final filter for Pinecone:\n{filter_log_str}")
        except TypeError:
            logging.error(f"Could not serialize final_filter for logging: {final_filter}")
    else:
        logging.info("No final filter was constructed.")
    logging.info(f"Filter construction duration: {time.time() - filter_build_start:.4f}s")
    return final_filter, post_date_range


# ==============================================================================
# EPISODE SEARCH FILTER BUILDER (Enhanced with channelTitle support)
# ==============================================================================

def build_episode_filter(
    llm_extracted_guests: list,
    llm_extracted_hosts: list,
    show_name: str | None,
    unique_personalities: list,
    unique_authors: list,
    unique_shows: list,
    time_filter: dict | None = None,
    include_date_clause: bool = True,
    strict: bool = True,
):
    """
    Build Pinecone filter for episode search.

    FIXED: Now combines show filter WITH person filter using AND.
    Previously returned early if people_filter existed, missing channelTitle.

    Example: "Invest Like the Best episode with Ari Emanuel"
    OLD: Filter: {"guests": {"$in": ["ari emanuel"]}} (missing show!)
    NEW: Filter: {"$and": [{"guests": ...}, {"channelTitle": ...}]}

    Args:
        llm_extracted_guests: Guest names from intent extraction
        llm_extracted_hosts: Host names from intent extraction
        show_name: Show name from intent extraction (e.g., "the joe rogan experience")
        unique_personalities: Known guest names for fuzzy matching
        unique_authors: Known host names for fuzzy matching
        unique_shows: Known show names for fuzzy matching
        time_filter: Time filter dict with mode, dates, etc.
        include_date_clause: Whether to include date in Pinecone filter
        strict: If True, use AND between all filters. If False, use OR for relaxed recall.

    Returns:
        (pinecone_filter, post_date_range) tuple
    """
    filter_build_start = time.time()
    logging.info(f"[EPISODE_FILTER] Building filter - guests: {llm_extracted_guests}, hosts: {llm_extracted_hosts}, show: {show_name}")

    filter_parts = []

    # =========================================================================
    # STEP 1: Build people filter (guests/hosts)
    # =========================================================================
    people_filter, post_date_range = build_filter(
        llm_extracted_guests,
        llm_extracted_hosts,
        unique_personalities,
        unique_authors,
        time_filter=None,  # Don't add date yet, we'll add it at the end
        include_date_clause=False,
    )

    if people_filter:
        filter_parts.append(people_filter)
        logging.info(f"[EPISODE_FILTER] People filter: {people_filter}")

    # =========================================================================
    # STEP 2: Build channelTitle filter from show_name (ALWAYS, not just fallback)
    # =========================================================================
    if show_name and unique_shows:
        logging.info(f"[EPISODE_FILTER] Building channelTitle filter for show: {show_name}")

        normalized_show = normalize_name(show_name)

        # Create normalized to original mapping
        normalized_to_original = {}
        normalized_choices = []
        for s in unique_shows:
            norm_s = normalize_name(str(s))
            if norm_s and norm_s not in normalized_to_original:
                normalized_to_original[norm_s] = str(s)
                normalized_choices.append(norm_s)

        if normalized_show and normalized_choices:
            raw_matches = process.extract(
                normalized_show,
                normalized_choices,
                scorer=fuzz.WRatio,
                limit=5
            )
            # Filter by score threshold manually for compatibility
            show_matches = [(match, score) for match, score in raw_matches if score >= 60]

            logging.info(f"[EPISODE_FILTER] Show fuzzy matches: {show_matches}")

            if show_matches:
                matched_shows = []
                for norm_match, score in show_matches:
                    original = normalized_to_original.get(norm_match)
                    if original:
                        matched_shows.append(original)

                if matched_shows:
                    # Build channelTitle filter with case variants
                    channel_variants = set()
                    for show in matched_shows:
                        channel_variants.add(show.lower())

                    channel_filter = {"channelTitle": {"$in": list(channel_variants)}}
                    filter_parts.append(channel_filter)
                    logging.info(f"[EPISODE_FILTER] Channel filter: {list(channel_variants)}")

    # =========================================================================
    # STEP 3: Build date filter
    # =========================================================================
    if include_date_clause and time_filter:
        date_clause, post_date_range = _build_date_clause(time_filter)
        if date_clause:
            filter_parts.append(date_clause)
            logging.info(f"[EPISODE_FILTER] Date filter: {date_clause}")
    else:
        _, post_date_range = _build_date_clause(time_filter)

    # =========================================================================
    # STEP 4: Combine all filter parts
    # strict=True: AND between all parts (exact match)
    # strict=False: OR between people/show filters, AND with date (relaxed recall)
    # =========================================================================
    if not filter_parts:
        logging.info(f"[EPISODE_FILTER] No filter constructed")
        logging.info(f"[EPISODE_FILTER] Duration: {time.time() - filter_build_start:.4f}s")
        return None, post_date_range
    elif len(filter_parts) == 1:
        final_filter = filter_parts[0]
    else:
        if strict:
            # Strict mode: ALL conditions must match
            final_filter = {"$and": filter_parts}
        else:
            # Relaxed mode: Separate date filter from entity filters
            # Keep date as AND (must match), but OR between people/show
            logging.info(f"[EPISODE_FILTER] Using RELAXED (OR) mode for entity filters")

            date_filters = []
            entity_filters = []

            for part in filter_parts:
                # Check if this is a date filter (contains pdnumeric)
                part_str = json.dumps(part)
                if 'pdnumeric' in part_str:
                    date_filters.append(part)
                else:
                    entity_filters.append(part)

            # Build final filter: entities with OR, then AND with date
            if entity_filters:
                if len(entity_filters) == 1:
                    entity_combined = entity_filters[0]
                else:
                    entity_combined = {"$or": entity_filters}

                if date_filters:
                    # Combine entity OR with date AND
                    final_filter = {"$and": [entity_combined] + date_filters}
                else:
                    final_filter = entity_combined
            elif date_filters:
                # Only date filters
                final_filter = {"$and": date_filters} if len(date_filters) > 1 else date_filters[0]
            else:
                final_filter = None

    logging.info(f"[EPISODE_FILTER] Final filter (strict={strict}): {json.dumps(final_filter, indent=2)}")
    logging.info(f"[EPISODE_FILTER] Duration: {time.time() - filter_build_start:.4f}s")

    return final_filter, post_date_range