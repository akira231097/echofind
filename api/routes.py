"""
FastAPI routes for the chatbot API with SSE streaming support.
"""

import json
import logging
import asyncio
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine.schemas import (
    ChatRequest,
    ChatResponse,
    SessionResetRequest,
    SessionResetResponse,
    SessionInfoResponse,
    HealthResponse,
    StreamEvent,
    RecommendationClickRequest,
    RecommendationClickResponse,
    EpisodeRecommendationClickRequest,
    EpisodeRecommendationClickResponse,
)
from engine.memory import memory_store
from engine.recommendations import get_recommendation
from engine.episode_recommendations import get_episode_recommendation

logger = logging.getLogger(__name__)

# Router instance - will be included in main app
router = APIRouter(prefix="/api", tags=["engine"])

# Global agent instance - set by main app
_agent = None


def set_agent(agent):
    """Set the global agent instance. Called during app startup."""
    global _agent
    _agent = agent


def get_agent():
    """Get the global agent instance."""
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot agent not initialized. Server may still be starting up."
        )
    return _agent


def format_agent_response(update_data: dict) -> dict:
    """
    Format response data based on branch type for consistent API output.

    This helper normalizes the response structure across different branches
    (small_talk, episode_search, clip_search) to ensure the frontend receives
    a consistent format regardless of which branch handled the query.

    Args:
        update_data: Raw response data from any branch

    Returns:
        Normalized response dict with branch-specific fields
    """
    branch = update_data.get("branch", "clip_search")

    # Base fields present in all responses
    base_response = {
        "answer": update_data.get("answer", ""),
        "confidence": update_data.get("confidence", 0.0),
        "branch": branch,
    }

    if branch == "small_talk":
        # Small talk returns conversational response only
        return {
            **base_response,
            "response_type": update_data.get("response_type", "greeting"),
            "sources": update_data.get("sources", []),
            # No media fields for small talk
            "video_url": None,
            "video_chunk_path": None,
            "episode_title": None,
            "podcast_title": None,
        }

    elif branch == "episode_search":
        # Episode search returns full episode metadata
        episode = update_data.get("episode", {})
        return {
            **base_response,
            "episode_id": episode.get("episode_id"),
            "episode_title": episode.get("episode_title"),
            "podcast_title": episode.get("podcast_title"),
            "published_date": episode.get("published_date"),
            "episode_description": episode.get("episode_description"),
            "episode_uri": episode.get("episode_uri"),
            "episode_image": episode.get("episode_image"),
            "guests": episode.get("guests", []),
            "hosts": episode.get("hosts", []),
            # No chunk-specific fields for episode search
            "video_url": None,
            "video_chunk_path": None,
        }

    else:  # clip_search (default)
        # Clip search returns chunk/clip media data
        chunk = update_data.get("chunk", {}) or {}
        return {
            **base_response,
            "video_url": chunk.get("chunkAudioUrl"),
            "video_chunk_path": chunk.get("videoChunkPath"),
            "episode_title": chunk.get("episode_title"),
            "podcast_title": chunk.get("podcast_title"),
            "speakers": chunk.get("speakers", []) or [],
            "guests": chunk.get("guests", []) or [],
            "published_date": chunk.get("published_date"),
            "chunk_start_time": chunk.get("startMs") / 1000.0 if chunk.get("startMs") else None,
            "chunk_end_time": chunk.get("endMs") / 1000.0 if chunk.get("endMs") else None,
        }


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        active_sessions=len(memory_store.list_sessions()),
    )


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.

    Processes the question and returns the complete response.
    Use /chat/stream for real-time progress updates.
    """
    agent = get_agent()

    try:
        response = await agent.ask(request.session_id, request.question)
        return response
    except Exception as e:
        logger.error(f"Chat error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks  # PHASE 5: For background memory compression
):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Returns real-time progress updates as the pipeline executes.

    SSE Event Format:
    - event: stage - Pipeline progress update
    - event: complete - Final result
    - event: error - Error occurred

    Data format (JSON):
    {
        "stage": "analyzing|embedding|searching|...|complete",
        "message": "Human-readable status",
        "progress": 0.0-1.0,
        "data": {...}  // Only on complete event
    }
    """
    agent = get_agent()

    async def event_generator():
        try:
            async for update in agent.ask_streaming(request.session_id, request.question):
                # Determine SSE event type based on stage
                if update.stage == "complete":
                    event_type = "complete"
                elif update.stage == "error":
                    event_type = "error"
                elif update.stage == "recommendations":
                    event_type = "recommendations"
                else:
                    event_type = "stage"

                event_data = StreamEvent(
                    event=event_type,
                    stage=update.stage,
                    message=update.message,
                    progress=update.progress,
                    data=update.data,
                )

                # Format as SSE
                yield f"event: {event_type}\n"
                yield f"data: {event_data.model_dump_json()}\n\n"

                # ==============================================================
                # PHASE 5: TRIGGER BACKGROUND COMPRESSION on completion
                # ==============================================================
                if update.stage == "complete":
                    background_tasks.add_task(
                        agent.compress_memory_background,
                        request.session_id
                    )
                    logger.debug(f"[ROUTES] Triggered background compression for {request.session_id}")

                # Small delay to ensure client receives events
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Stream error for session {request.session_id}: {e}", exc_info=True)
            error_event = StreamEvent(
                event="error",
                stage="error",
                message=str(e),
                progress=0.0,
            )
            yield f"event: error\n"
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ============================================================================
# Recommendation Endpoints
# ============================================================================

@router.post("/recommendation/click", response_model=RecommendationClickResponse)
async def click_recommendation(request: RecommendationClickRequest):
    """
    Handle user clicking on a recommendation.

    Returns the pre-computed response and updates conversation memory.
    """
    agent = get_agent()

    try:
        # Get the stored recommendation
        result = await get_recommendation(
            request.session_id,
            request.turn_id,
            request.recommendation_index,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Recommendation not found. It may have expired."
            )

        rec = result["recommendation"]
        chunk = result["chunk"]
        original_question = result.get("original_question", "")
        original_resolved = result.get("resolved_query", "")

        # Update memory with accurate context
        # The user_question should reflect this was a related clip selection
        # from the original question - preserving the conversation flow
        memory = memory_store.get_or_create(request.session_id)

        # Build contextual user question that shows the relationship
        # e.g., "Re: What does Elon think about AI? → [Selected: More on AI risks]"
        if original_question:
            contextual_question = f"Re: {original_question[:60]}{'...' if len(original_question) > 60 else ''} → {rec.prompt}"
        else:
            contextual_question = f"[Selected related clip: {rec.prompt}]"

        # Build resolved query that maintains searchability
        # Combines original intent with the specific recommendation angle
        if original_resolved:
            contextual_resolved = f"{original_resolved} - {rec.prompt}"
        else:
            contextual_resolved = rec.prompt

        memory.add_turn(
            user_question=contextual_question,
            resolved_query=contextual_resolved,
            answer_summary=rec.turn_summary,
            artifact_id=chunk.get('id'),
            artifact_title=chunk.get('episode_title'),
            entities=rec.extracted_entities,
            themes=rec.turn_themes,
        )

        logger.info(f"[RECOMMENDATION] User clicked recommendation {request.recommendation_index} for turn {request.turn_id}")
        logger.info(f"[RECOMMENDATION] Memory updated: {contextual_question[:80]}...")

        return RecommendationClickResponse(
            answer=rec.answer,
            confidence=rec.confidence,
            video_url=chunk.get("chunkAudioUrl"),
            video_chunk_path=chunk.get("videoChunkPath"),
            episode_title=chunk.get("episode_title"),
            podcast_title=chunk.get("podcast_title"),
            speakers=chunk.get("speakers", []) or [],
            guests=chunk.get("guests", []) or [],
            published_date=chunk.get("published_date"),
            chunk_start_time=chunk.get("startMs") / 1000.0 if chunk.get("startMs") else None,
            chunk_end_time=chunk.get("endMs") / 1000.0 if chunk.get("endMs") else None,
            memory_updated=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation click error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episode-recommendation/click", response_model=EpisodeRecommendationClickResponse)
async def click_episode_recommendation(request: EpisodeRecommendationClickRequest):
    """
    Handle user clicking on an episode recommendation.

    Returns the pre-computed response and updates conversation memory.
    """
    agent = get_agent()

    try:
        # Get the stored episode recommendation
        result = await get_episode_recommendation(
            request.session_id,
            request.turn_id,
            request.recommendation_index,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Episode recommendation not found. It may have expired."
            )

        rec = result["recommendation"]
        episode = result["episode"]
        original_question = result.get("original_question", "")
        original_resolved = result.get("resolved_query", "")

        # Update memory with accurate context
        memory = memory_store.get_or_create(request.session_id)

        # Build contextual user question that shows the relationship
        if original_question:
            contextual_question = f"Re: {original_question[:60]}{'...' if len(original_question) > 60 else ''} → {rec.prompt}"
        else:
            contextual_question = f"[Selected related episode: {rec.prompt}]"

        # Build resolved query that maintains searchability
        if original_resolved:
            contextual_resolved = f"{original_resolved} - {rec.prompt}"
        else:
            contextual_resolved = rec.prompt

        memory.add_turn(
            user_question=contextual_question,
            resolved_query=contextual_resolved,
            answer_summary=rec.turn_summary,
            artifact_id=episode.get('episode_id'),
            artifact_title=episode.get('episode_title'),
            entities=rec.extracted_entities,
            themes=rec.turn_themes,
        )

        logger.info(f"[EPISODE_RECOMMENDATION] User clicked recommendation {request.recommendation_index} for turn {request.turn_id}")
        logger.info(f"[EPISODE_RECOMMENDATION] Memory updated: {contextual_question[:80]}...")

        return EpisodeRecommendationClickResponse(
            answer=rec.answer,
            confidence=rec.confidence,
            episode_id=episode.get("episode_id", ""),
            episode_title=episode.get("episode_title", ""),
            podcast_title=episode.get("podcast_title", ""),
            published_date=episode.get("published_date"),
            episode_description=episode.get("description"),
            episode_uri=episode.get("uri"),
            episode_image=episode.get("image"),
            guests=episode.get("guests", []) if isinstance(episode.get("guests"), list) else [],
            hosts=episode.get("hosts", []) if isinstance(episode.get("hosts"), list) else [],
            memory_updated=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Episode recommendation click error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Session Management Endpoints
# ============================================================================

@router.post("/session/reset", response_model=SessionResetResponse)
async def reset_session(
    session_id: Optional[str] = Query(None, description="Session ID to reset (query param)"),
    request: Optional[SessionResetRequest] = Body(None, description="Session ID in JSON body"),
):
    """
    Reset conversation memory for a session.

    Clears all conversation history, entities, and themes.

    Accepts session_id via:
    - Query parameter: POST /session/reset?session_id=xxx
    - JSON body: {"session_id": "xxx"}

    Query parameter takes precedence if both are provided.
    """
    # Resolve session_id: query param > body > error
    resolved_session_id = session_id or (request.session_id if request else None)

    if not resolved_session_id:
        raise HTTPException(
            status_code=422,
            detail="session_id is required. Provide via query parameter or JSON body.",
        )

    # Validate session_id format (basic sanity check)
    resolved_session_id = resolved_session_id.strip()
    if not resolved_session_id or len(resolved_session_id) > 256:
        raise HTTPException(
            status_code=422,
            detail="Invalid session_id: must be non-empty and <= 256 characters.",
        )

    agent = get_agent()
    agent.reset_session(resolved_session_id)

    logger.info(f"Session reset: {resolved_session_id}")

    return SessionResetResponse(
        session_id=resolved_session_id,
        message="Session memory cleared",
    )


@router.get("/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """
    Get information about a session's current state.

    Returns turn count, tracked entities, themes, etc.
    """
    agent = get_agent()
    info = agent.get_session_info(session_id)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    return SessionInfoResponse(**info)


@router.get("/session/{session_id}/memory/debug")
async def get_session_memory_debug(session_id: str):
    """
    DEBUG ENDPOINT: Get full memory state for a session.

    Returns complete memory dump including:
    - All conversation turns (recent + compressed)
    - Entity tracking with relevance scores
    - Search state (last action, route history, topic, entities)
    - What context is rendered for each LLM component

    Use this to inspect exactly what the chatbot "remembers" and
    what context is being fed to each LLM at any point.
    """
    memory = memory_store.get(session_id)

    if not memory:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    return memory.to_dict()


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its memory."""
    memory_store.delete(session_id)
    return {"message": f"Session {session_id} deleted"}


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = memory_store.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.post("/cleanup")
async def cleanup_old_sessions(max_age_hours: int = Query(default=24, ge=1, le=168)):
    """
    Clean up sessions older than max_age_hours.

    Default: 24 hours
    Max: 168 hours (7 days)
    """
    removed = memory_store.cleanup_old_sessions(max_age_hours)
    return {
        "message": f"Cleaned up {removed} old sessions",
        "removed_count": removed,
        "max_age_hours": max_age_hours,
    }
