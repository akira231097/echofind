# retrieval/data_fetcher.py

import asyncio
import boto3
import os
import openai
from pinecone import Pinecone
from config import (
    PINECONE_INDEX_NAME,
    PRIMARY_SCORE_THRESHOLD_BASE,
    PRIMARY_SCORE_THRESHOLD_MIN,
    PRIMARY_SCORE_RELATIVE_FACTOR,
    SECONDARY_SCORE_THRESHOLD_BASE,
    SECONDARY_SCORE_THRESHOLD_MIN,
    SECONDARY_SCORE_RELATIVE_FACTOR,
    MAX_CHUNKS_BEFORE_RERANK,
    RDS_HOST,
    RDS_PORT,
    RDS_DATABASE,
    RDS_USERNAME,
    RDS_PASSWORD,
    RDS_TABLE_NAME,
)
import logging
from retrieval.sparse_encoder import get_sparse_embeddings_batch, get_pinecone_sparse_embeddings
from config import (
    PINECONE_SPARSE_INDEX_NAME, 
    HYBRID_ALPHA, 
    USE_PINECONE_SPARSE_MODEL,
    SPARSE_TOP_K
)
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
import json

PD_NUMERIC_FIELD = os.getenv("PUBLISHED_NUMERIC_FIELD", "pdnumeric")


def _parse_iso_date_safe(iso_s: str):
    if not iso_s:
        return None
    try:
        return datetime.fromisoformat(str(iso_s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def _within_pdnumeric_range(meta: dict, rng: dict | None) -> bool:
    # exact copy of your working helper, but using PD_NUMERIC_FIELD if present
    if not rng:
        return True
    start = rng.get("start")
    end = rng.get("end")
    if meta is None:
        return True
    # preferred fast path
    if PD_NUMERIC_FIELD in meta:
        try:
            n = int(meta[PD_NUMERIC_FIELD])
            if start is not None and n < start: return False
            if end   is not None and n > end:   return False
            return True
        except (ValueError, TypeError):
            pass
    # fallback: publishedDate ISO
    iso = meta.get("publishedDate")
    if iso:
        dt = _parse_iso_date_safe(iso)
        if dt:
            n = dt.year*10000 + dt.month*100 + dt.day
            if start is not None and n < start: return False
            if end   is not None and n > end:   return False
            return True
    return True


# AWS credentials are read from the environment or an attached IAM role
# (see .env.example). Never hardcode credentials in source.


# ---- Reciprocal Rank Fusion (RRF) ----
RRF_K = int(os.getenv("RRF_K", "60"))  # SIGIR '09 and Elastic default ~60
DENSE_RRF_WEIGHT  = float(os.getenv("DENSE_RRF_WEIGHT",  "1.0"))
SPARSE_RRF_WEIGHT = float(os.getenv("SPARSE_RRF_WEIGHT", "1.0"))


def _rank_positions(items):
    """Return {id: 1-based-rank} for a ranked list of match dicts."""
    return {m["id"]: i + 1 for i, m in enumerate(items)}

def _coalesce_source_labels(labels: set[str]) -> str:
    """
    Combine multiple retrieval_source labels into one.
    Priority: 'both' > (dense & sparse) > dense > sparse > unknown.
    """
    labels = {str(l).lower() for l in labels if l and str(l).lower() != "unknown"}
    if "both" in labels or (("dense" in labels) and ("sparse" in labels)):
        return "both"
    if "dense" in labels:
        return "dense"
    if "sparse" in labels:
        return "sparse"
    return "unknown"

def _determine_dynamic_threshold(scores, base, minimum, relative_factor, stage_label):
    """Compute a threshold tailored to the score distribution for a stage."""
    if not scores:
        logging.debug(f'[{stage_label}] No scores available; using base threshold {base:.3f}')
        return base
    max_score = max(scores)
    dynamic = max(minimum, max_score * relative_factor)
    threshold = min(base, dynamic) if base is not None else dynamic
    logging.debug(
        '[%s] Thresholds -> max: %.4f | dynamic: %.4f | base: %.4f | chosen: %.4f',
        stage_label,
        max_score,
        dynamic,
        base if base is not None else float('nan'),
        threshold,
    )
    return threshold

def rrf_fuse_lists(lists, weights=None, k: int = RRF_K, top_k: int | None = None):
    """
    Generic Reciprocal Rank Fusion over multiple ranked lists.
    Each list is a list of match dicts with at least 'id' present.
    - Preserves and coalesces 'retrieval_source' across lists.
    - 'weights' (optional) down/up-weights lists (e.g., original vs HyDE).
    """
    if not lists:
        return []

    if weights is None:
        weights = [1.0] * len(lists)
    assert len(weights) == len(lists), "weights length must match number of lists"

    fused_scores = {}
    exemplar = {}
    labels_by_id = {}

    for w, lst in zip(weights, lists):
        if not lst:
            continue
        ranks = _rank_positions(lst)
        for m in lst:
            rid = m["id"]
            # RRF score accumulation
            fused_scores[rid] = fused_scores.get(rid, 0.0) + w * (1.0 / (k + ranks[rid]))
            # keep first occurrence as exemplar
            if rid not in exemplar:
                exemplar[rid] = m
            # collect labels to coalesce later
            label = m.get("retrieval_source", "unknown")
            labels_by_id.setdefault(rid, set()).add(label)

    fused = []
    for rid, score in fused_scores.items():
        base = exemplar[rid].copy()
        base["hybrid_score"] = float(score)
        # FINAL, PRESERVED LABEL:
        base["retrieval_source"] = _coalesce_source_labels(labels_by_id.get(rid, set()))
        fused.append(base)

    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused[:top_k] if top_k else fused

async def concurrent_embedding_generation(openai_client, search_queries: list[str]):
    """Generate embeddings for all queries in a single batched API call."""
    logging.info(f"\n--- Generating embeddings for {len(search_queries)} queries (batched) ---")

    # Single API call with all queries batched together
    response = await asyncio.to_thread(
        openai_client.embeddings.create,
        input=search_queries,  # All queries in one call
        model="text-embedding-3-large"
    )

    valid_query_data = []
    for i, embedding_data in enumerate(response.data):
        if embedding_data.embedding:
            valid_query_data.append({
                "index": i,
                "vector": embedding_data.embedding,
                "query": search_queries[i]
            })
    logging.info(f"Generated embeddings for {len(valid_query_data)} queries.")
    return valid_query_data

async def single_query_two_stage_search(
    index,
    query_item,
    final_filter,
    pinecone_k,
    target_per_query,
    post_date_range=None,
    recall_ratio: float = 0.0,
    allow_relaxed_recall: bool = True,
):
    query_idx, vector = query_item['index'], query_item['vector']
    results, seen_ids = [], set()

    recall_ratio = max(0.0, min(recall_ratio, 0.9))

    if final_filter:
        recall_budget = int(round(target_per_query * recall_ratio))
        n_primary = max(0, target_per_query - recall_budget)
        logging.info(
            f"[Dense Q{query_idx + 1}] Filtered search primary={n_primary}, recall_budget={recall_budget}, allow_relaxed={allow_relaxed_recall}"
        )
    else:
        recall_budget = target_per_query
        n_primary = 0
        logging.info(f"[Dense Q{query_idx + 1}] No filter; allocating all {target_per_query} slots to recall")

    stage_prefix = "dense_filtered" if final_filter else "dense_unfiltered"

    if final_filter and n_primary > 0:
        logging.info(f"[Dense Q{query_idx + 1}] Stage 1 filtered top_k={pinecone_k}")
        s1_res = await asyncio.to_thread(
            index.query,
            vector=vector,
            filter=final_filter,
            top_k=pinecone_k,
            include_metadata=True,
        )
        primary_scores = [m.score for m in s1_res.matches if m.score is not None]
        primary_threshold = _determine_dynamic_threshold(
            primary_scores,
            PRIMARY_SCORE_THRESHOLD_BASE,
            PRIMARY_SCORE_THRESHOLD_MIN,
            PRIMARY_SCORE_RELATIVE_FACTOR,
            f"Dense Primary Q{query_idx + 1}",
        )
        logging.info(
            f"[Dense Q{query_idx + 1}] Stage 1 candidates={len(primary_scores)} threshold={primary_threshold:.4f}"
        )
        for m in s1_res.matches or []:
            if m.score is None or m.score < primary_threshold:
                continue
            candidate = {
                'id': m.id,
                'score': m.score,
                'metadata': dict(m.metadata) if m.metadata else {},
                'retrieval_source': 'dense',
                'retrieval_stage': f"{stage_prefix}_primary",
            }
            if not _within_pdnumeric_range(candidate.get('metadata', {}), post_date_range):
                continue
            if candidate['id'] in seen_ids:
                continue
            results.append(candidate)
            seen_ids.add(candidate['id'])
            if len(results) >= n_primary:
                break

    remaining_needed = target_per_query - len(results)
    filtered_backfill = 0
    if remaining_needed > 0:
        if final_filter:
            if allow_relaxed_recall:
                recall_budget = max(recall_budget, remaining_needed)
            else:
                filtered_backfill = remaining_needed
                logging.info(
                    f"[Dense Q{query_idx + 1}] Strict constraints -> keeping filter for {filtered_backfill} additional slots"
                )
        else:
            recall_budget = remaining_needed

    if filtered_backfill > 0 and final_filter:
        fetch_k = min(max(pinecone_k * 2, target_per_query), 200)
        logging.info(f"[Dense Q{query_idx + 1}] Filtered backfill top_k={fetch_k}")
        s_filter = await asyncio.to_thread(
            index.query,
            vector=vector,
            filter=final_filter,
            top_k=fetch_k,
            include_metadata=True,
        )
        secondary_scores = [m.score for m in s_filter.matches if m.score is not None]
        secondary_threshold = _determine_dynamic_threshold(
            secondary_scores,
            SECONDARY_SCORE_THRESHOLD_BASE,
            SECONDARY_SCORE_THRESHOLD_MIN,
            SECONDARY_SCORE_RELATIVE_FACTOR,
            f"Dense Filtered Backfill Q{query_idx + 1}",
        )
        logging.info(
            f"[Dense Q{query_idx + 1}] Filtered backfill candidates={len(secondary_scores)} threshold={secondary_threshold:.4f}"
        )
        for m in s_filter.matches or []:
            if m.score is None or m.score < secondary_threshold:
                continue
            candidate = {
                'id': m.id,
                'score': m.score,
                'metadata': dict(m.metadata) if m.metadata else {},
                'retrieval_source': 'dense',
                'retrieval_stage': f"{stage_prefix}_secondary",
            }
            if candidate['id'] in seen_ids:
                continue
            if not _within_pdnumeric_range(candidate.get('metadata', {}), post_date_range):
                continue
            results.append(candidate)
            seen_ids.add(candidate['id'])
            if len(results) >= target_per_query:
                break

    if (not final_filter or allow_relaxed_recall) and recall_budget > 0 and len(results) < target_per_query:
        remaining_slots = target_per_query - len(results)
        recall_slots = min(remaining_slots, recall_budget)
        overfetch = 2 if post_date_range else 1
        fetch_k = min(max(pinecone_k * overfetch, target_per_query), 200)
        logging.info(
            f"[Dense Q{query_idx + 1}] Unfiltered recall slots={recall_slots} top_k={fetch_k}"
        )
        s2_res = await asyncio.to_thread(
            index.query,
            vector=vector,
            top_k=fetch_k,
            include_metadata=True,
        )
        recall_scores = [m.score for m in s2_res.matches if m.score is not None]
        recall_threshold = _determine_dynamic_threshold(
            recall_scores,
            SECONDARY_SCORE_THRESHOLD_BASE,
            SECONDARY_SCORE_THRESHOLD_MIN,
            SECONDARY_SCORE_RELATIVE_FACTOR,
            f"Dense Unfiltered Recall Q{query_idx + 1}",
        )
        logging.info(
            f"[Dense Q{query_idx + 1}] Unfiltered recall candidates={len(recall_scores)} threshold={recall_threshold:.4f}"
        )
        stage_name = 'dense_unfiltered_primary' if not final_filter else 'dense_unfiltered_recall'
        for m in s2_res.matches or []:
            if m.score is None or m.score < recall_threshold:
                continue
            candidate = {
                'id': m.id,
                'score': m.score,
                'metadata': dict(m.metadata) if m.metadata else {},
                'retrieval_source': 'dense',
                'retrieval_stage': stage_name,
            }
            if candidate['id'] in seen_ids:
                continue
            if not _within_pdnumeric_range(candidate.get('metadata', {}), post_date_range):
                continue
            results.append(candidate)
            seen_ids.add(candidate['id'])
            if len(results) >= target_per_query:
                break
    elif final_filter and not allow_relaxed_recall and recall_budget > 0:
        logging.info(f"[Dense Q{query_idx + 1}] Skipping unfiltered recall due to strict constraints")

    logging.info(f"[Dense Q{query_idx + 1}] Final result count: {len(results)}")
    return results

async def concurrent_pinecone_search(
    pinecone_client,
    valid_query_data,
    valid_sparse_data,
    final_filter,
    pinecone_k,
    target_per_query,
    use_hybrid=True,
    post_date_range=None,
    recall_ratio: float = 0.0,
    allow_relaxed_recall: bool = True,
):
    per_query_allow_relaxed = allow_relaxed_recall or (final_filter is None)
    if use_hybrid and valid_sparse_data:
        return await concurrent_hybrid_search(
            pinecone_client,
            valid_query_data,
            valid_sparse_data,
            final_filter,
            pinecone_k,
            target_per_query,
            post_date_range,
            recall_ratio,
            allow_relaxed_recall=per_query_allow_relaxed,
        )
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    tasks = [
        single_query_two_stage_search(
            index,
            item,
            final_filter,
            pinecone_k,
            target_per_query,
            post_date_range,
            recall_ratio,
            allow_relaxed_recall=per_query_allow_relaxed,
        )
        for item in valid_query_data
    ]
    return await asyncio.gather(*tasks)

async def concurrent_sparse_embedding_generation(pinecone_client, search_queries: list[str]):
    """Generate sparse embeddings for search queries using pre-trained BM25 model."""
    logging.info(f"\n--- Generating sparse embeddings for {len(search_queries)} queries... ---")
    
    if USE_PINECONE_SPARSE_MODEL:
        # Try Pinecone's sparse model first
        try:
            sparse_vectors = await get_pinecone_sparse_embeddings(pinecone_client, search_queries)
        except Exception as e:
            logging.warning(f"Pinecone sparse model failed, using trained BM25: {e}")
            sparse_vectors = await get_sparse_embeddings_batch(search_queries)
    else:
        # Use the pre-trained BM25 model from S3
        sparse_vectors = await get_sparse_embeddings_batch(search_queries)
    
    valid_sparse_data = []
    for i, sparse_vec in enumerate(sparse_vectors):
        if sparse_vec and sparse_vec.get('indices') is not None and sparse_vec.get('values') is not None:
            # Ensure indices and values are lists (not empty)
            if len(sparse_vec['indices']) > 0:
                valid_sparse_data.append({
                    "index": i,
                    "sparse_vector": sparse_vec,
                    "query": search_queries[i]
                })
            else:
                logging.warning(f"Query '{search_queries[i]}' produced empty sparse vector")
    
    logging.info(f"Generated sparse embeddings for {len(valid_sparse_data)} queries.")
    return valid_sparse_data

async def single_sparse_search(
    sparse_index,
    query_item,
    filter_dict,
    sparse_k,
    target_per_query,
    post_date_range=None,
    recall_ratio: float = 0.0,
    allow_relaxed_recall: bool = True,
):
    qidx = query_item['index']
    svec = query_item['sparse_vector']
    results, seen = [], set()

    recall_ratio = max(0.0, min(recall_ratio, 0.9))

    if filter_dict:
        recall_budget = int(round(target_per_query * recall_ratio))
        n_primary = max(0, target_per_query - recall_budget)
        logging.info(
            f"[Sparse Q{qidx + 1}] Filtered search primary={n_primary}, recall_budget={recall_budget}, allow_relaxed={allow_relaxed_recall}"
        )
    else:
        recall_budget = target_per_query
        n_primary = 0
        logging.info(f"[Sparse Q{qidx + 1}] No filter; allocating all {target_per_query} slots to recall")

    stage_prefix = "sparse_filtered" if filter_dict else "sparse_unfiltered"

    if filter_dict and n_primary > 0:
        logging.info(f"[Sparse Q{qidx + 1}] Stage 1 filtered top_k={sparse_k}")
        r1 = await asyncio.to_thread(
            sparse_index.query,
            sparse_vector=svec,
            filter=filter_dict,
            top_k=sparse_k,
            include_metadata=True,
        )
        primary_scores = [m.score for m in (r1.matches or []) if m.score is not None]
        primary_threshold = _determine_dynamic_threshold(
            primary_scores,
            PRIMARY_SCORE_THRESHOLD_BASE,
            PRIMARY_SCORE_THRESHOLD_MIN,
            PRIMARY_SCORE_RELATIVE_FACTOR,
            f"Sparse Primary Q{qidx + 1}",
        )
        logging.info(
            f"[Sparse Q{qidx + 1}] Stage 1 candidates={len(primary_scores)} threshold={primary_threshold:.4f}"
        )
        for m in r1.matches or []:
            if m.score is None or m.score < primary_threshold:
                continue
            md = dict(m.metadata) if m.metadata else {}
            if not _within_pdnumeric_range(md, post_date_range):
                continue
            if m.id in seen:
                continue
            results.append({
                'id': m.id,
                'score': m.score,
                'metadata': md,
                'retrieval_source': 'sparse',
                'retrieval_stage': f"{stage_prefix}_primary",
            })
            seen.add(m.id)
            if len(results) >= n_primary:
                break

    remaining_needed = target_per_query - len(results)
    filtered_backfill = 0
    if remaining_needed > 0:
        if filter_dict:
            if allow_relaxed_recall:
                recall_budget = max(recall_budget, remaining_needed)
            else:
                filtered_backfill = remaining_needed
                logging.info(
                    f"[Sparse Q{qidx + 1}] Strict constraints -> keeping filter for {filtered_backfill} additional slots"
                )
        else:
            recall_budget = remaining_needed

    if filtered_backfill > 0 and filter_dict:
        fetch_k = min(max(sparse_k * 2, target_per_query), 200)
        logging.info(f"[Sparse Q{qidx + 1}] Filtered backfill top_k={fetch_k}")
        r_filter = await asyncio.to_thread(
            sparse_index.query,
            sparse_vector=svec,
            filter=filter_dict,
            top_k=fetch_k,
            include_metadata=True,
        )
        secondary_scores = [m.score for m in (r_filter.matches or []) if m.score is not None]
        secondary_threshold = _determine_dynamic_threshold(
            secondary_scores,
            SECONDARY_SCORE_THRESHOLD_BASE,
            SECONDARY_SCORE_THRESHOLD_MIN,
            SECONDARY_SCORE_RELATIVE_FACTOR,
            f"Sparse Filtered Backfill Q{qidx + 1}",
        )
        logging.info(
            f"[Sparse Q{qidx + 1}] Filtered backfill candidates={len(secondary_scores)} threshold={secondary_threshold:.4f}"
        )
        for m in r_filter.matches or []:
            if m.score is None or m.score < secondary_threshold:
                continue
            md = dict(m.metadata) if m.metadata else {}
            if not _within_pdnumeric_range(md, post_date_range):
                continue
            if m.id in seen:
                continue
            results.append({
                'id': m.id,
                'score': m.score,
                'metadata': md,
                'retrieval_source': 'sparse',
                'retrieval_stage': f"{stage_prefix}_secondary",
            })
            seen.add(m.id)
            if len(results) >= target_per_query:
                break

    if (not filter_dict or allow_relaxed_recall) and recall_budget > 0 and len(results) < target_per_query:
        remaining_slots = target_per_query - len(results)
        recall_slots = min(remaining_slots, recall_budget)
        overfetch = 2 if post_date_range else 1
        fetch_k = min(max(sparse_k * overfetch, target_per_query), 200)
        logging.info(
            f"[Sparse Q{qidx + 1}] Unfiltered recall slots={recall_slots} top_k={fetch_k}"
        )
        r2 = await asyncio.to_thread(
            sparse_index.query,
            sparse_vector=svec,
            top_k=fetch_k,
            include_metadata=True,
        )
        recall_scores = [m.score for m in (r2.matches or []) if m.score is not None]
        recall_threshold = _determine_dynamic_threshold(
            recall_scores,
            SECONDARY_SCORE_THRESHOLD_BASE,
            SECONDARY_SCORE_THRESHOLD_MIN,
            SECONDARY_SCORE_RELATIVE_FACTOR,
            f"Sparse Unfiltered Recall Q{qidx + 1}",
        )
        logging.info(
            f"[Sparse Q{qidx + 1}] Unfiltered recall candidates={len(recall_scores)} threshold={recall_threshold:.4f}"
        )
        stage_name = 'sparse_unfiltered_primary' if not filter_dict else 'sparse_unfiltered_recall'
        for m in r2.matches or []:
            if m.score is None or m.score < recall_threshold:
                continue
            md = dict(m.metadata) if m.metadata else {}
            if not _within_pdnumeric_range(md, post_date_range):
                continue
            if m.id in seen:
                continue
            results.append({
                'id': m.id,
                'score': m.score,
                'metadata': md,
                'retrieval_source': 'sparse',
                'retrieval_stage': stage_name,
            })
            seen.add(m.id)
            if len(results) >= target_per_query:
                break
    elif filter_dict and not allow_relaxed_recall and recall_budget > 0:
        logging.info(f"[Sparse Q{qidx + 1}] Skipping unfiltered recall due to strict constraints")

    logging.info(f"[Sparse Q{qidx + 1}] Final result count: {len(results)}")
    return results

async def concurrent_hybrid_search(
    pinecone_client,
    valid_query_data,
    valid_sparse_data,
    final_filter,
    pinecone_k,
    target_per_query,
    post_date_range=None,
    recall_ratio: float = 0.0,
    allow_relaxed_recall: bool = True,
):
    """Hybrid search with date filtering"""
    logging.info("\n--- HYBRID search with date filtering ---")
    dense_index = pinecone_client.Index(PINECONE_INDEX_NAME)
    sparse_index = pinecone_client.Index(PINECONE_SPARSE_INDEX_NAME)

    sparse_by_idx = {s["index"]: s for s in valid_sparse_data or []}
    per_query_allow_relaxed = allow_relaxed_recall or (final_filter is None)

    all_tasks, task_meta = [], []
    for d in valid_query_data:
        qidx = d["index"]
        all_tasks.append(
            single_query_two_stage_search(
                dense_index,
                d,
                final_filter,
                pinecone_k,
                target_per_query,
                post_date_range,
                recall_ratio,
                allow_relaxed_recall=per_query_allow_relaxed,
            )
        )
        task_meta.append(("dense", qidx))

        s = sparse_by_idx.get(qidx)
        if s is not None:
            all_tasks.append(
                single_sparse_search(
                    sparse_index,
                    s,
                    final_filter,
                    SPARSE_TOP_K,
                    target_per_query,
                    post_date_range,
                    recall_ratio,
                    allow_relaxed_recall=per_query_allow_relaxed,
                )
            )
            task_meta.append(("sparse", qidx))

    all_results = await asyncio.gather(*all_tasks)

    # Group by query and fuse
    by_query = {}
    for (kind, qidx), results in zip(task_meta, all_results):
        if qidx not in by_query:
            by_query[qidx] = {"dense": [], "sparse": []}
        by_query[qidx][kind] = results or []

    fused_per_query = []
    for qidx in sorted(by_query.keys()):
        dense_results = by_query[qidx]["dense"]
        sparse_results = by_query[qidx]["sparse"]
        fused = rrf_fuse_lists(
            [dense_results, sparse_results],
            weights=[DENSE_RRF_WEIGHT, SPARSE_RRF_WEIGHT],
            k=RRF_K,
            top_k=target_per_query
        )
        fused_per_query.append(fused)

    logging.info(f"Completed hybrid search with date filtering")
    return fused_per_query



def combine_dense_sparse_results(dense_results, sparse_results, alpha_unused, target_k):
    """
    Hybrid fusion using Reciprocal Rank Fusion (RRF).
    NOTE: 'alpha_unused' kept for signature compatibility but ignored.
    """
    return rrf_fuse_lists([dense_results, sparse_results], k=RRF_K, top_k=target_k)

def combine_pinecone_results(nested_results, per_list_weights=None, top_k=None):
    """
    RRF over the list of per-query ranked lists (e.g., original + HyDE variants).
    This replaces de-dupe + sort, giving a principled cross-query fusion.
    """
    logging.info("\n--- RRF-combining results across queries ---")
    # nested_results: List[List[match_dict]]
    fused = rrf_fuse_lists(nested_results, weights=per_list_weights, k=RRF_K, top_k=top_k)
    logging.info(f"Total unique chunks after cross-query RRF: {len(fused)}")
    stage_counts = {}
    for item in fused:
        stage = item.get('retrieval_stage') or item.get('retrieval_source') or 'unknown'
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    logging.info(f"Stage distribution after RRF: {stage_counts}")
    return fused

def get_final_chunk_keys(combined_results, MAX_CHUNKS_BEFORE_RERANK):
    """Replicates the pre-fetch limiting logic from `pinecone_helpers.py`."""
    logging.info(f"\n--- Applying MAX_CHUNKS_BEFORE_RERANK limit of {MAX_CHUNKS_BEFORE_RERANK}... ---")
    # This function in the repo also creates a metadata map. We'll merge that logic
    # into the main data merging function for simplicity in this script.
    final_matches_to_fetch = combined_results[:MAX_CHUNKS_BEFORE_RERANK]
    logging.info(f"Limited to {len(final_matches_to_fetch)} chunks for DB fetch.")
    return final_matches_to_fetch

def batch_get_rds_items(pinecone_matches: list, table_name: str = RDS_TABLE_NAME):
    """
    Fetch items from PostgreSQL RDS instead of DynamoDB
    """
    if not pinecone_matches:
        return []
    
    logging.info(f"\n--- Fetching {len(pinecone_matches)} items from RDS table '{table_name}'... ---")
    
    # Extract chunk IDs from pinecone matches
    chunk_ids = []
    for match in pinecone_matches:
        if 'id' in match:
            chunk_ids.append(match['id'])
    
    if not chunk_ids:
        logging.warning("No valid chunk IDs to fetch")
        return []
    
    try:
        # Connect to RDS
        connection = psycopg2.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            database=RDS_DATABASE,
            user=RDS_USERNAME,
            password=RDS_PASSWORD,
            cursor_factory=RealDictCursor  # Returns dict instead of tuple
        )
        
        with connection.cursor() as cursor:
            # Build query with proper parameterization
            placeholders = ','.join(['%s'] * len(chunk_ids))
            query = f"""
                SELECT
                    "chunkId" as id,
                    "chunkTitle",
                    "chunkDescriptiveTitle",
                    "chunkDescription",
                    "chunkLength",
                    "episodeId",
                    "channelId",
                    "chunkAudioUrl",
                    transcript,
                    "publishedDate",
                    speakers,
                    "podcastTitle",
                    "episodeTitle",
                    guests,
                    "guestsDescription",
                    host,
                    "hostDescription",
                    "startMs",
                    "endMs",
                    topics,
                    sentiment,
                    "additionalData"
                FROM "clips"
                WHERE "chunkId" IN ({placeholders})
                AND "deletedAt" IS NULL
                AND "showOnFeed" = true
            """
            
            cursor.execute(query, chunk_ids)
            results = cursor.fetchall()
            
            # Convert results to list of dicts and handle PostgreSQL arrays
            processed_results = []
            for row in results:
                row_dict = dict(row)

                # Convert PostgreSQL arrays to Python lists
                for key in ['speakers', 'guests', 'guestsDescription', 'topics']:
                    if key in row_dict and row_dict[key] is not None:
                        # PostgreSQL arrays are already converted to Python lists by psycopg2
                        if not isinstance(row_dict[key], list):
                            row_dict[key] = [row_dict[key]]
                    else:
                        row_dict[key] = []

                # Ensure publishedDate is a string in ISO format
                if row_dict.get('publishedDate'):
                    row_dict['publishedDate'] = row_dict['publishedDate'].isoformat()

                # Parse additionalData JSONB field to extract video URL
                if row_dict.get('additionalData'):
                    try:
                        # additionalData comes as a dict from psycopg2 with RealDictCursor
                        if isinstance(row_dict['additionalData'], dict):
                            additional_data = row_dict['additionalData']
                        else:
                            # Fallback if it's a string
                            additional_data = json.loads(row_dict['additionalData'])

                        # Extract video URL
                        row_dict['videoChunkPath'] = additional_data.get('videoChunkPath')
                        row_dict['videoMasterPlaylistPath'] = additional_data.get('videoMasterPlaylistPath')
                    except (json.JSONDecodeError, TypeError) as e:
                        logging.warning(f"Failed to parse additionalData for chunk {row_dict.get('id')}: {e}")
                        row_dict['videoChunkPath'] = None
                        row_dict['videoMasterPlaylistPath'] = None
                else:
                    row_dict['videoChunkPath'] = None
                    row_dict['videoMasterPlaylistPath'] = None

                processed_results.append(row_dict)
            
            logging.info(f"Successfully fetched {len(processed_results)} items from RDS")
            return processed_results
            
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error during fetch: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during RDS fetch: {e}")
        logging.exception("Traceback:")
        return []
    finally:
        if 'connection' in locals() and connection:
            connection.close()

def merge_db_and_pinecone_data(pinecone_matches, db_items):
    """
    Merges Pinecone scores with RDS data
    """
    logging.info("\n--- Merging Pinecone scores with RDS data ---")
    
    # Create mapping of id to db item
    db_map = {item['id']: item for item in db_items}
    merged_data = []

    for match in pinecone_matches:
        if match['id'] in db_map:
            db_item = db_map[match['id']]
            pinecone_meta = match.get('metadata', {})
            
            # Handle the audio URL
            audio_url = db_item.get('chunkAudioUrl')

            merged_chunk = {
                # --- Core Data ---
                'id': match['id'],
                'pinecone_score': match.get('score'),
                'chunk': db_item.get('transcript', ''),  # RDS uses 'transcript' field

                # --- Promoted Metadata Fields ---
                'podcast_title': db_item.get('podcastTitle') or pinecone_meta.get('channelTitle'),
                'episode_title': db_item.get('episodeTitle') or pinecone_meta.get('episodeTitle'),
                'published_date': db_item.get('publishedDate') or pinecone_meta.get('publishedDate'),
                'episodeId': db_item.get('episodeId'),  # Add episodeId from RDS

                # Handle arrays properly
                'guests': db_item.get('guests', []) or pinecone_meta.get('guests'),
                'hosts': db_item.get('host') or pinecone_meta.get('hosts'),  # Note: RDS has 'host' (singular)
                'speakers': db_item.get('speakers', []),

                # --- Chunk timing ---
                'chunkLength': float(db_item.get('chunkLength', 30)),
                'startMs': db_item.get('startMs'),
                'endMs': db_item.get('endMs'),

                # --- Retrieval metadata ---
                'retrieval_source': match.get('retrieval_source', 'unknown'),
                'retrieval_stage': match.get('retrieval_stage'),
                'hybrid_score': match.get('hybrid_score'),

                # --- Audio and video data ---
                'chunkAudioUrl': audio_url,
                'videoChunkPath': db_item.get('videoChunkPath'),  # MP4 video URL
                'videoMasterPlaylistPath': db_item.get('videoMasterPlaylistPath'),  # HLS playlist
                'pinecone_metadata': pinecone_meta,

                # --- Additional RDS fields if needed ---
                'chunkTitle': db_item.get('chunkTitle'),
                'chunkDescription': db_item.get('chunkDescription'),
                'topics': db_item.get('topics', []),
                'sentiment': db_item.get('sentiment'),
                'additionalData': db_item.get('additionalData', {})  # Include the full additionalData object
            }
            merged_data.append(merged_chunk)

    logging.info(f"Successfully merged data for {len(merged_data)} chunks.")
    return merged_data
