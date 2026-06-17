"""Pydantic schemas for chatbot API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PipelineStage(str, Enum):
    """Pipeline execution stages for streaming updates."""
    # Router stage (NEW)
    ROUTING = "routing"
    ROUTED = "routed"

    # Existing stages
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    SEARCHING = "searching"
    SEARCHED = "searched"
    FETCHING = "fetching"
    FETCHED = "fetched"
    RERANKING = "reranking"
    RERANKED = "reranked"
    SELECTING = "selecting"
    COMPLETE = "complete"
    RECOMMENDATIONS = "recommendations"  # Sent after complete with recommendations
    EPISODE_RECOMMENDATIONS = "episode_recommendations"  # Sent after episode search with episode recommendations
    ERROR = "error"

    # New stages for agent
    EXPLAINING = "explaining"  # Small talk with grounding
    EPISODE_SEARCHING = "episode_searching"
    EPISODE_SELECTED = "episode_selected"


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier for memory persistence")
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    answer: str = Field(..., description="Natural language answer describing the clip")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    video_url: Optional[str] = Field(None, description="URL to the audio/video chunk")
    video_chunk_path: Optional[str] = Field(None, description="Path to video chunk")
    episode_title: Optional[str] = Field(None, description="Episode title")
    podcast_title: Optional[str] = Field(None, description="Podcast/show title")
    speakers: List[str] = Field(default_factory=list, description="Speakers in the clip")
    guests: List[str] = Field(default_factory=list, description="Guests in the episode")
    hosts: List[str] = Field(default_factory=list, description="Hosts of the podcast")
    published_date: Optional[str] = Field(None, description="Episode publish date")
    chunk_start_time: Optional[float] = Field(None, description="Start time in seconds")
    chunk_end_time: Optional[float] = Field(None, description="End time in seconds")
    chunk_length_ms: Optional[int] = Field(None, description="Chunk length in milliseconds")
    total_time: float = Field(..., description="Total pipeline execution time in seconds")
    is_followup: bool = Field(False, description="Whether this was detected as a follow-up question")
    resolved_query: Optional[str] = Field(None, description="Query after pronoun resolution")


class StreamEvent(BaseModel):
    """SSE event format for streaming updates."""
    event: str = Field(..., description="Event type: 'stage', 'complete', or 'error'")
    stage: Optional[str] = Field(None, description="Current pipeline stage")
    message: Optional[str] = Field(None, description="Human-readable status message")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress 0.0-1.0")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional event data")


class SessionResetRequest(BaseModel):
    """Request to reset a session's memory."""
    session_id: str = Field(..., description="Session to reset")


class SessionResetResponse(BaseModel):
    """Response after resetting a session."""
    session_id: str
    message: str = "Session memory cleared"


class SessionInfoResponse(BaseModel):
    """Information about a session's current state."""
    session_id: str
    turn_count: int
    recent_entities: List[str]
    themes: List[str]
    shown_artifacts_count: int
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    active_sessions: int = 0


class QueryAnalysisResult(BaseModel):
    """Result from query analysis stage."""
    query_title: str = Field(..., description="Formal topic statement")
    resolved_query: str = Field(..., description="Query with pronouns resolved")
    query_complexity: str = Field("simple", description="simple or complex")
    is_followup: bool = Field(False, description="Whether this is a follow-up")
    extracted_guests_interviewees: List[str] = Field(default_factory=list)
    extracted_hosts_creators: List[str] = Field(default_factory=list)
    referenced_entities: List[str] = Field(default_factory=list)
    hyde_documents: List[str] = Field(default_factory=list)
    time_filter: Optional[Dict[str, Any]] = None


class SelectionResult(BaseModel):
    """Result from the selection stage."""
    chosen_index: int = Field(..., description="Index of selected chunk")
    answer: str = Field(..., description="Natural language response")
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    turn_summary: str = Field(..., description="Summary for memory")
    extracted_entities: List[str] = Field(default_factory=list)
    turn_themes: List[str] = Field(default_factory=list)


# ============================================================================
# Recommendation Schemas
# ============================================================================

class RecommendationClickRequest(BaseModel):
    """Request when user clicks a recommendation."""
    session_id: str = Field(..., description="Session identifier")
    turn_id: str = Field(..., description="Turn ID when recommendations were generated")
    recommendation_index: int = Field(..., ge=0, le=2, description="Which recommendation (0-2)")


class RecommendationClickResponse(BaseModel):
    """Response when user clicks a recommendation."""
    answer: str = Field(..., description="Pre-computed answer for this clip")
    confidence: float = Field(..., ge=0.0, le=1.0)
    video_url: Optional[str] = Field(None, description="URL to the audio/video chunk")
    video_chunk_path: Optional[str] = Field(None, description="Path to video chunk")
    episode_title: Optional[str] = Field(None, description="Episode title")
    podcast_title: Optional[str] = Field(None, description="Podcast/show title")
    speakers: List[str] = Field(default_factory=list, description="Speakers in the clip")
    guests: List[str] = Field(default_factory=list, description="Guests in the episode")
    published_date: Optional[str] = Field(None, description="Episode publish date")
    chunk_start_time: Optional[float] = Field(None, description="Start time in seconds")
    chunk_end_time: Optional[float] = Field(None, description="End time in seconds")
    memory_updated: bool = Field(True, description="Whether memory was updated")


# ============================================================================
# Episode Recommendation Schemas
# ============================================================================

class EpisodeRecommendationClickRequest(BaseModel):
    """Request when user clicks an episode recommendation."""
    session_id: str = Field(..., description="Session identifier")
    turn_id: str = Field(..., description="Turn ID when recommendations were generated")
    recommendation_index: int = Field(..., ge=0, le=2, description="Which recommendation (0-2)")


class EpisodeRecommendationClickResponse(BaseModel):
    """Response when user clicks an episode recommendation."""
    answer: str = Field(..., description="Pre-computed answer for this episode")
    confidence: float = Field(..., ge=0.0, le=1.0)
    episode_id: str = Field(..., description="Episode ID")
    episode_title: str = Field(..., description="Episode title")
    podcast_title: str = Field(..., description="Podcast/show title")
    published_date: Optional[str] = Field(None, description="Episode publish date")
    episode_description: Optional[str] = Field(None, description="Episode description")
    episode_uri: Optional[str] = Field(None, description="Episode audio/video URI")
    episode_image: Optional[str] = Field(None, description="Episode thumbnail image")
    guests: List[str] = Field(default_factory=list, description="Guests in the episode")
    hosts: List[str] = Field(default_factory=list, description="Hosts of the episode")
    memory_updated: bool = Field(True, description="Whether memory was updated")


# ==============================================================================
# UNIFIED MEMORY UPDATE SCHEMA (Used by ALL branches)
# ==============================================================================

class BranchMemoryUpdate(BaseModel):
    """
    UNIFIED OUTPUT SCHEMA - Every branch MUST output this after responding.

    This ensures consistent memory updates regardless of which branch handles the query.
    The agent orchestrator uses this to update SearchState.

    Phase 2 Enhancement (Option A): Now includes key_quotes, topics_covered, and
    notable_examples to provide richer context for query analyzers to resolve
    follow-up queries more accurately.
    """

    # What happened this turn
    turn_summary: str = Field(
        ...,
        max_length=500,
        description="Summary of what happened - user query + what was found (max 500 chars)"
    )
    action_type: str = Field(
        ...,
        description="Type of action: 'clip_shown', 'episode_shown', 'explanation', 'greeting', 'error'"
    )
    action_target_id: Optional[str] = Field(
        None,
        description="Chunk ID, Episode ID, or None"
    )
    action_target_title: Optional[str] = Field(
        None,
        description="Human-readable title of shown content"
    )
    published_date: Optional[str] = Field(
        None,
        description="Publication date of the shown episode/clip (YYYY-MM-DD format)"
    )

    # Entities and topics
    entities_mentioned: List[str] = Field(
        default_factory=list,
        description="Key entities mentioned this turn (max 10)"
    )
    topics_discussed: List[str] = Field(
        default_factory=list,
        description="Topics/themes discussed (max 5)"
    )

    # Context hints for next turn
    is_topic_shift: bool = Field(
        False,
        description="True if user started a completely new topic"
    )
    suggested_phase: Optional[str] = Field(
        None,
        description="Suggested conversation phase: 'discovery', 'deep_dive', 'comparison', 'idle'"
    )

    # ===========================================================================
    # OPTION A: ENHANCED CONTEXT FIELDS (Phase 2)
    # ===========================================================================
    # These fields help query analyzers resolve follow-up queries more accurately
    # by providing specific context about what was shown/discussed.

    key_quotes: List[str] = Field(
        default_factory=list,
        description="2-3 memorable quotes from the content that could help resolve future queries"
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="Specific topics/subtopics covered in the content (max 5)"
    )
    notable_examples: List[str] = Field(
        default_factory=list,
        description="Notable examples, stories, guests, or highlights (max 3)"
    )


class RouterOutput(BaseModel):
    """Output schema for the query router."""

    route: str = Field(
        ...,
        description="Chosen route: 'small_talk', 'episode_search', 'clip_search'"
    )
    sub_intent: Optional[str] = Field(
        None,
        description="Sub-intent: greeting|explanation|contextual_knowledge|clarification|off_topic (small_talk)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in routing decision"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this route was chosen"
    )
    resolved_entities: List[str] = Field(
        default_factory=list,
        description="Pronouns resolved to actual entity names from context"
    )
    query_intent: Optional[str] = Field(
        None,
        description="Interpreted intent in 5-10 words"
    )
    key_signals: List[str] = Field(
        default_factory=list,
        description="Key signals in the query that led to this decision"
    )
    fallback_route: Optional[str] = Field(
        None,
        description="Alternative route if primary confidence < 0.7"
    )


class SmallTalkResponse(BaseModel):
    """Response schema for small talk branch."""

    response_text: str = Field(
        ...,
        description="Natural language response to user"
    )
    response_type: str = Field(
        ...,
        description="Type: 'greeting', 'explanation', 'clarification', 'contextual_knowledge', 'off_topic'"
    )
    sources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Sources used if search grounding was applied"
    )
    memory_update: BranchMemoryUpdate = Field(
        ...,
        description="Unified memory update"
    )
    follow_up_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions to keep conversation going"
    )


class EpisodeSearchResponse(BaseModel):
    """Response schema for episode search branch."""

    response_text: str = Field(
        ...,
        description="Natural language response introducing the episode"
    )
    episode_id: str = Field(
        ...,
        description="Selected episode ID"
    )
    episode_title: str = Field(
        ...,
        description="Episode title"
    )
    podcast_title: str = Field(
        ...,
        description="Podcast/show title"
    )
    published_date: Optional[str] = Field(
        None,
        description="Publication date"
    )
    episode_description: Optional[str] = Field(
        None,
        description="Episode description from RDS"
    )
    episode_uri: Optional[str] = Field(
        None,
        description="Episode audio/video URI"
    )
    episode_image: Optional[str] = Field(
        None,
        description="Episode thumbnail image"
    )
    guests: List[str] = Field(
        default_factory=list,
        description="Guests in the episode"
    )
    hosts: List[str] = Field(
        default_factory=list,
        description="Hosts of the episode"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in episode selection"
    )
    memory_update: BranchMemoryUpdate = Field(
        ...,
        description="Unified memory update"
    )
    # Recommendation support - data needed for episode recommendations
    selected_index: int = Field(
        default=0,
        description="Index of selected episode in candidates"
    )
    scored_episodes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="All scored episode candidates (for recommendations)"
    )
    episode_descriptions_data: Optional[Dict[str, Dict]] = Field(
        default=None,
        description="Episode descriptions from RDS (for recommendations)"
    )
