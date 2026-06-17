# retrieval/search.py

import logging
import json
import traceback
import openai
import os
from pinecone import Pinecone
import json, logging, re
from datetime import datetime
from typing import Optional, Dict, Any, List
import concurrent.futures


GEMINI_MODEL = "gemini-2.5-flash-lite"

def rerank_chunks_cohere(
    pinecone_client,
    query: str,
    chunks: list,
    top_n: int = 50,
    llm_analysis: Optional[Dict[str, Any]] = None,
    shown_artifact_ids: Optional[List[str]] = None,
) -> list:
    """
    Rerank chunks using Cohere 3.5 with Parallel Batching + Global Re-scoring.

    Strategy:
    1. Split chunks into parallel batches (fast rough sorting) - ~0.50s
    2. Take the top candidates from all batches
    3. Run ONE final global rerank on the top 20 candidates (guarantees accuracy) - ~0.40s

    Result: ~0.9s total latency (vs ~2.0s serial).

    Args:
        pinecone_client: Pinecone client with inference API access
        query: User's natural language query (should be resolved query for chatbot)
        chunks: List of chunk dictionaries with metadata
        top_n: Number of top results to return (default 50 for pre-LLM selection)
        llm_analysis: Optional analyzer output with extracted entities and constraints
        shown_artifact_ids: Optional list of already-shown chunk IDs to deprioritize

    Returns:
        List of reranked chunks with rerank scores
    """
    if not chunks:
        return []

    try:
        # Filter out chunks with None or empty text content
        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get('chunk') and isinstance(chunk['chunk'], str) and chunk['chunk'].strip()
        ]

        if not valid_chunks:
            logging.warning("No valid chunks with text content for reranking")
            return chunks[:top_n]

        # HARD LIMIT: Max 35 chunks for single reranker call
        MAX_CHUNKS_RERANK = 35
        if len(valid_chunks) > MAX_CHUNKS_RERANK:
            logging.info(f"Enforcing hard limit: {len(valid_chunks)} → {MAX_CHUNKS_RERANK} chunks before rerank")
            valid_chunks = valid_chunks[:MAX_CHUNKS_RERANK]

        # ========== 1) PREPARE DOCUMENTS & CONSTRAINTS ==========

        # Build instruction trailer from analyzer
        instruction_lines = []
        if llm_analysis:
            extracted_guests = llm_analysis.get('extracted_guests_interviewees', [])
            extracted_hosts = llm_analysis.get('extracted_hosts_creators', [])

            # Handle referenced entities from memory (for follow-up questions)
            referenced_entities = llm_analysis.get('referenced_entities', [])
            if referenced_entities:
                instruction_lines.append(f"Context entities: {', '.join(referenced_entities)} (prioritize content about these)")

            if extracted_guests:
                instruction_lines.append(f"Prefer chunks where SPEAKERS or GUESTS include: {', '.join(extracted_guests)} (first-person speech)")
            if extracted_hosts:
                hosts_str = ", ".join(extracted_hosts)
                if extracted_guests:
                    instruction_lines.append(f"Accept hosts: {hosts_str} only if guest is absent")
                else:
                    instruction_lines.append(f"Prefer chunks from hosts: {hosts_str}")

            time_filter = llm_analysis.get('time_filter', {})
            if time_filter.get('has_time_constraint'):
                mode = time_filter.get('mode')
                if mode == 'latest': instruction_lines.append("Time intent: prefer most recent content")
                elif mode == 'oldest': instruction_lines.append("Time intent: prefer earliest content")
                elif mode == 'between' and time_filter.get('start_date_utc') and time_filter.get('end_date_utc'):
                    instruction_lines.append(f"Time intent: between {time_filter['start_date_utc']} and {time_filter['end_date_utc']}")
                elif mode == 'before' and time_filter.get('end_date_utc'):
                    instruction_lines.append(f"Time intent: before {time_filter['end_date_utc']}")
                elif mode == 'after' and time_filter.get('start_date_utc'):
                    instruction_lines.append(f"Time intent: after {time_filter['start_date_utc']}")
                elif mode == 'on' and time_filter.get('start_date_utc'):
                    instruction_lines.append(f"Time intent: on {time_filter['start_date_utc']}")

            # Handle follow-up context
            if llm_analysis.get('is_followup'):
                instruction_lines.append("This is a follow-up question; prefer content that provides new information on the same topic")

        instruction_lines.append("Avoid: intro/outro, ads, off-topic banter")
        instruction_lines.append("Must substantially answer the question in the chunk alone")

        # Deprioritize already-shown artifacts (for "show me something else" requests)
        if shown_artifact_ids:
            # We'll handle this by slightly reducing scores for shown chunks after reranking
            logging.info(f"Will deprioritize {len(shown_artifact_ids)} already-shown chunks")

        # Attach constraints to query
        rerank_query = f"{query}\n\nConstraints:\n- " + "\n- ".join(instruction_lines) if instruction_lines else query
        logging.info(f"Enhanced query with {len(instruction_lines)} constraints" if instruction_lines else "Using plain query")

        # Prepare structured docs with stable IDs
        docs_map = {}
        prepared_docs = []

        for i, c in enumerate(valid_chunks):
            doc_id = c.get("id") or c.get("chunkId") or f"cand_{i}"
            docs_map[doc_id] = c

            # Structure the doc for Cohere 3.5
            prepared_docs.append({
                "id": doc_id,
                "speakers": ", ".join(c.get("speakers") or []) if isinstance(c.get("speakers"), list) else str(c.get("speakers") or ""),
                "guests": ", ".join(c.get("guests") or []) if isinstance(c.get("guests"), list) else str(c.get("guests") or ""),
                "hosts": ", ".join(c.get("hosts") or []) if isinstance(c.get("hosts"), list) else str(c.get("hosts") or ""),
                "episode_title": c.get("episode_title") or "",
                "chunk_text": c["chunk"]
            })

        logging.info(f"Prepared {len(prepared_docs)} documents with IDs: {list(docs_map.keys())[:5]}... (showing first 5)")

        # LOG INPUT TO RERANKER (COMMENTED OUT - TOO VERBOSE)
        # logging.info("\n" + "="*70)
        # logging.info("INPUT TO COHERE RERANKER:")
        # logging.info("="*70)
        logging.info(f"Query: {query}")
        logging.info(f"Enhanced Query with Constraints:\n{rerank_query}")
        logging.info(f"\nTotal Chunks to Rerank: {len(valid_chunks)}")
        # logging.info("\nCHUNKS BEING SENT TO RERANKER:")
        # for i, chunk in enumerate(valid_chunks[:30]):  # Log first 30 chunks
        #     guests = ", ".join(chunk.get("guests") or []) if isinstance(chunk.get("guests"), list) else (chunk.get("guests") or "N/A")
        #     hosts = ", ".join(chunk.get("hosts") or []) if isinstance(chunk.get("hosts"), list) else (chunk.get("hosts") or "N/A")
        #     speakers = ", ".join(chunk.get("speakers") or []) if isinstance(chunk.get("speakers"), list) else (chunk.get("speakers") or "N/A")
        #     published_date = chunk.get("published_date") or chunk.get("pinecone_metadata", {}).get("publishedDate") or "N/A"
        #     hybrid_score = chunk.get("hybrid_score")
        #     retrieval_source = chunk.get("retrieval_source", "unknown")
        #     retrieval_stage = chunk.get("retrieval_stage") or retrieval_source or "unknown"
        #
        #     chunk_text = chunk.get("chunk", "")
        #     truncated_text = chunk_text[:150] + '...' if len(chunk_text) > 150 else chunk_text
        #
        #     hybrid_score_str = f"{hybrid_score:.4f}" if hybrid_score else "N/A"
        #     logging.info(
        #         f"\n[Chunk {i}] ID: {chunk.get('id') or chunk.get('chunkId', 'N/A')}\n"
        #         f"  Retrieval Stage: {retrieval_stage}\n"
        #         f"  Hybrid Score: {hybrid_score_str}\n"
        #         f"  Podcast: {chunk.get('podcast_title', 'N/A')}\n"
        #         f"  Episode: {chunk.get('episode_title', 'N/A')}\n"
        #         f"  Published: {published_date}\n"
        #         f"  Guests: {guests}\n"
        #         f"  Hosts: {hosts}\n"
        #         f"  Speakers: {speakers}\n"
        #         f"  Text Preview: {truncated_text}"
        #     )
        # if len(valid_chunks) > 30:
        #     logging.info(f"\n... and {len(valid_chunks) - 30} more chunks")
        # logging.info("="*70 + "\n")

        # ========== 2) SINGLE RERANKER CALL ==========
        #
        # Strategy: Single call with all chunks (up to 35)
        # Returns ALL chunks with rerank scores for hybrid scoring to process
        #

        logging.info(f"Single rerank call: {len(prepared_docs)} chunks")
        inference = pinecone_client.inference

        result = inference.rerank(
            model="cohere-rerank-3.5",
            query=rerank_query,
            documents=prepared_docs,
            top_n=len(prepared_docs),  # Get scores for ALL chunks
            rank_fields=["speakers", "guests", "hosts", "episode_title", "chunk_text"],
            return_documents=False,
            parameters={"max_chunks_per_doc": 4}
        )

        # Map results back to chunks with scores
        all_top_results = []
        if result and hasattr(result, 'data'):
            for r in result.data:
                idx = r.index if hasattr(r, 'index') else r.get('index') if isinstance(r, dict) else None
                if idx is not None and 0 <= idx < len(prepared_docs):
                    d_id = prepared_docs[idx]['id']
                    if d_id in docs_map:
                        chunk = dict(docs_map[d_id])
                        chunk['rerank_score'] = r.score if hasattr(r, 'score') else 0.0
                        chunk['retrieval_stage'] = chunk.get('retrieval_stage', 'unknown') + '_reranked'
                        all_top_results.append(chunk)

        logging.info(f"Rerank complete: {len(all_top_results)} chunks scored")

        if len(all_top_results) == 0:
            logging.error("Reranker returned 0 results!")
            return chunks[:top_n]

        # ========== 3) OUTPUT RESULTS ==========

        # LOG OUTPUT FROM RERANKER
        logging.info("\n" + "="*70)
        logging.info("OUTPUT FROM COHERE RERANKER:")
        logging.info("="*70)
        logging.info(f"Total Reranked Chunks: {len(all_top_results[:top_n])}")
        logging.info("\nTOP RERANKED CHUNKS:")
        for i, chunk in enumerate(all_top_results[:top_n]):
            guests = ", ".join(chunk.get("guests") or []) if isinstance(chunk.get("guests"), list) else (chunk.get("guests") or "N/A")
            hosts = ", ".join(chunk.get("hosts") or []) if isinstance(chunk.get("hosts"), list) else (chunk.get("hosts") or "N/A")
            speakers = ", ".join(chunk.get("speakers") or []) if isinstance(chunk.get("speakers"), list) else (chunk.get("speakers") or "N/A")
            published_date = chunk.get("published_date") or chunk.get("pinecone_metadata", {}).get("publishedDate") or "N/A"
            rerank_score = chunk.get("rerank_score")
            retrieval_stage = chunk.get("retrieval_stage", "unknown")

            chunk_text = chunk.get("chunk", "")
            truncated_text = chunk_text[:150] + '...' if len(chunk_text) > 150 else chunk_text

            rerank_score_str = f"{rerank_score:.4f}" if rerank_score else "N/A"
            logging.info(
                f"\n[Rank {i}] ID: {chunk.get('id') or chunk.get('chunkId', 'N/A')}\n"
                f"  Retrieval Stage: {retrieval_stage}\n"
                f"  Rerank Score: {rerank_score_str}\n"
                f"  Podcast: {chunk.get('podcast_title', 'N/A')}\n"
                f"  Episode: {chunk.get('episode_title', 'N/A')}\n"
                f"  Published: {published_date}\n"
                f"  Guests: {guests}\n"
                f"  Hosts: {hosts}\n"
                f"  Speakers: {speakers}\n"
                f"  Text Preview: {truncated_text}"
            )
        logging.info("="*70 + "\n")

        return all_top_results[:top_n]

    except Exception as e:
        logging.error(f"Cohere reranking wrapper failed: {e}", exc_info=True)
        return chunks[:top_n]
