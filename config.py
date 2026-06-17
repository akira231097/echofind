# config.py
#
# Central configuration. All secrets and environment-specific values are read
# from environment variables (optionally via a local .env file) — there are NO
# hardcoded credentials in this file. Copy .env.example to .env and fill it in.

import os

# Load a local .env file if python-dotenv is installed (optional convenience).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- API Keys (required at runtime; no defaults committed) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

# --- AWS Credentials (prefer an attached IAM role; otherwise set via env) ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# --- Relational DB (PostgreSQL / RDS) ---
RDS_HOST = os.getenv("RDS_HOST")                       # e.g. your-db-host.example.com
RDS_PORT = os.getenv("RDS_PORT", 5432)
RDS_DATABASE = os.getenv("RDS_DATABASE", "podcast_content")
RDS_USERNAME = os.getenv("RDS_USERNAME", "db_user")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_TABLE_NAME = os.getenv("RDS_TABLE_NAME", "clips")

# ==============================================================================
# LLM MODEL CONFIGURATION
# ==============================================================================

# --- Base Model (default for general tasks) ---
GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
GEMINI_MODEL = "gemini-3-flash-preview"
# --- Embeddings & Reranking ---
EMBEDDING_MODEL = "text-embedding-3-large"
RERANKING_MODEL = "cohere-rerank-english-v3.0"

# --- Router Models ---
ROUTER_MODEL = GEMINI_MODEL  # Query routing decisions

# --- Small Talk Branch Models ---
SMALL_TALK_MODEL = GEMINI_MODEL_FAST              # Simple responses (greeting, clarification)
SMALL_TALK_WITH_GROUNDING_MODEL = GEMINI_MODEL_FAST  # Responses with Google Search grounding
MEMORY_UPDATE_MODEL = GEMINI_MODEL_FAST           # Memory extraction from responses

# --- Query Analysis Models ---
QUERY_ANALYZER_MODEL = GEMINI_MODEL          # Main query metadata extraction
QUERY_ANALYZER_LITE_MODEL = GEMINI_MODEL_FAST     # HyDE document generation (lightweight)

# --- Clip Selection Models ---
CLIP_SELECTION_MODEL = GEMINI_MODEL          # Selecting best clip from candidates
CLIP_RECOMMENDATION_MODEL = GEMINI_MODEL_FAST     # Generating alternative clip recommendations

# --- Episode Search Models ---
EPISODE_SEARCH_MODEL = GEMINI_MODEL          # Episode selection and response
EPISODE_INTENT_MODEL = GEMINI_MODEL        # Episode search intent extraction
EPISODE_RECOMMENDATION_MODEL = GEMINI_MODEL_FAST  # Alternative episode recommendations

# ==============================================================================
# REASONING EFFORT CONFIGURATION
# ==============================================================================
# Controls how much computational reasoning Gemini applies to each task.
# OpenAI-compatible parameter that maps to Gemini's thinking_level/thinking_budget.
#
# Valid values:
#   - "none"   : Disables thinking entirely (fastest, cheapest - 2.5 models only)
#   - "low"    : Minimal reasoning (good for simple/classification tasks)
#   - "medium" : Moderate reasoning (balanced speed/quality)
#   - "high"   : Maximum reasoning (best quality, slowest)
#
# Cost/latency impact (approximate):
#   none < low < medium < high
#   Disabling reasoning ("none") can reduce costs by up to 96%
# ==============================================================================

# --- Router ---
ROUTER_REASONING_EFFORT = "none"              # Fast classification, no reasoning needed

# --- Small Talk Branch ---
SMALL_TALK_REASONING_EFFORT = "none"          # Simple responses, greetings
SMALL_TALK_GROUNDING_REASONING_EFFORT = "none" # Google Search grounded responses
MEMORY_UPDATE_REASONING_EFFORT = "none"       # Memory extraction (simple)

# --- Query Analysis ---
QUERY_ANALYZER_REASONING_EFFORT = "none"       # Query metadata extraction
QUERY_ANALYZER_HYDE_REASONING_EFFORT = "none" # HyDE document generation (creative)

# --- Clip Selection ---
CLIP_SELECTION_REASONING_EFFORT = "none"       # Selecting best clip from candidates
CLIP_RECOMMENDATION_REASONING_EFFORT = "none" # Alternative clip recommendations

# --- Episode Search ---
EPISODE_INTENT_REASONING_EFFORT = "none"      # Intent extraction (simple)
EPISODE_HYDE_REASONING_EFFORT = "none"        # HyDE for episodes (creative)
EPISODE_SELECTION_REASONING_EFFORT = "none"    # Episode selection
EPISODE_RECOMMENDATION_REASONING_EFFORT = "none"  # Alternative episode recommendations

# --- Reranker Toggle ---
# Set to True to enable Cohere reranking, False to skip and use top-K directly
RERANKER_ENABLED = True  # Toggle: True = rerank, False = skip reranking

# --- Pinecone ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "echofind-dense")

# --- Search Strategy ---
RERANKER_TOP_N = 50
LLM_SELECTOR_INPUT_K = 50
FINAL_TOP_K = 3
RECENT_WINDOW_DAYS_DEFAULT = int(os.getenv('RECENT_WINDOW_DAYS_DEFAULT', '21'))  # 21 days default for "latest"
RECENT_WINDOW_DAYS_MAX = int(os.getenv('RECENT_WINDOW_DAYS_MAX', '365'))           # Max 1 year window
RECENT_BUCKET_WEIGHT = float(os.getenv('RECENT_BUCKET_WEIGHT', '1.25'))            # Weight for recent bucket in RRF (from reference)
RECENT_BACKSTOP_WEIGHT = float(os.getenv('RECENT_BACKSTOP_WEIGHT', '0.3'))         # Weight for fallback "all" bucket (from reference)
RECENT_DEFAULT_LIMIT = int(os.getenv('RECENT_DEFAULT_LIMIT', '50'))                # Default limit for time-sorted results
RECENT_MIN_RESULTS = int(os.getenv('RECENT_MIN_RESULTS', '3'))                     # Minimum results before fallback

# HYDE Embedding Weighting
ORIGINAL_QUERY_WEIGHT = float(os.getenv('ORIGINAL_QUERY_WEIGHT', '1.25'))          # Weight for original user query
HYDE_WEIGHT_MAX = float(os.getenv('HYDE_WEIGHT_MAX', '1.1'))                       # Max weight for highest similarity HyDE
HYDE_WEIGHT_MIN = float(os.getenv('HYDE_WEIGHT_MIN', '0.85'))                      # Min weight for lowest similarity HyDE

# Episode Cap and Bucket Quota
PER_EPISODE_CAP = int(os.getenv('PER_EPISODE_CAP', '6'))                           # Max results from same episode
MIN_PER_BUCKET = int(os.getenv('MIN_PER_BUCKET', '3'))                             # Minimum results per bucket

# --- Recency-First Strategy ---
# NOTE: With hybrid metadata-aware scoring, the relevance floor is less critical
# since we now use weighted fusion of semantic + date + person + show scores.
# These floors are kept as fallback for edge cases.
RELEVANCE_FLOOR = float(os.getenv('RELEVANCE_FLOOR', '0.35'))                      # Allows more candidates through
RELEVANCE_FLOOR_HARD = float(os.getenv('RELEVANCE_FLOOR_HARD', '0.05'))            # Very permissive for pure recency
MIN_RELEVANT_RESULTS = int(os.getenv('MIN_RELEVANT_RESULTS', '3'))                 # Threshold before fallback triggers
RECENCY_BOOST_HARD = float(os.getenv('RECENCY_BOOST_HARD', '1.5'))                 # Boost factor for pure recency queries (STRONGER: newest wins)
RECENCY_BOOST_SOFT = float(os.getenv('RECENCY_BOOST_SOFT', '0.6'))                 # Boost factor for topic+recency queries (moderate: balance topic+date)
RECENCY_FIRST_BUCKET_LIMIT = int(os.getenv('RECENCY_FIRST_BUCKET_LIMIT', '30'))    # How many newest items to fetch
PRIMARY_SCORE_THRESHOLD_BASE = float(os.getenv('PRIMARY_SCORE_THRESHOLD_BASE', '0.25'))
PRIMARY_SCORE_THRESHOLD_MIN = float(os.getenv('PRIMARY_SCORE_THRESHOLD_MIN', '0.18'))
PRIMARY_SCORE_RELATIVE_FACTOR = float(os.getenv('PRIMARY_SCORE_RELATIVE_FACTOR', '0.82'))
SECONDARY_SCORE_THRESHOLD_BASE = float(os.getenv('SECONDARY_SCORE_THRESHOLD_BASE', '0.2'))
SECONDARY_SCORE_THRESHOLD_MIN = float(os.getenv('SECONDARY_SCORE_THRESHOLD_MIN', '0.10'))
SECONDARY_SCORE_RELATIVE_FACTOR = float(os.getenv('SECONDARY_SCORE_RELATIVE_FACTOR', '0.85'))
# Backwards-compatible aliases
PRIMARY_SCORE_THRESHOLD = PRIMARY_SCORE_THRESHOLD_BASE
SECONDARY_SCORE_THRESHOLD = SECONDARY_SCORE_THRESHOLD_BASE
MAX_CHUNKS_BEFORE_RERANK = 35  # Single reranker call -> hybrid scoring -> top 21 to LLM

# --- Fuzzy Matching ---
WRATIO_THRESHOLD = 85
TOKEN_SET_MIN_SCORE = 75
SPECIFICITY_RATIO_THRESHOLD = 70
FUZZY_MATCH_CANDIDATE_LIMIT = 3
PERSONALITY_METADATA_FIELD = "guests"
AUTHOR_METADATA_FIELD = "hosts"

# Sparse Index Configuration
PINECONE_SPARSE_INDEX_NAME = os.getenv("PINECONE_SPARSE_INDEX_NAME", "echofind-sparse")
# Optional explicit host for the sparse index (otherwise resolved via the client).
SPARSE_INDEX_HOST = os.getenv("PINECONE_SPARSE_INDEX_HOST", "")

# Hybrid Search Configuration
HYBRID_ALPHA = 0.7  # Weight for dense search (0.7 dense, 0.3 sparse)
USE_PINECONE_SPARSE_MODEL = False  # Set to True if you have access to pinecone-sparse-english-v0
SPARSE_TOP_K = 30

# --- Agent Pipeline Config ---
PINECONE_K = 100                  # Top K from Pinecone per query
TARGET_CHUNKS_PER_QUERY = 30      # Target chunks per HyDE query
TOP_N_RERANKED = 50               # Keep top N after reranking
LLM_TOP_K = 40                    # Chunks sent to LLM for selection

# ==============================================================================
# CONTEXT WINDOW OPTIMIZATION
# ==============================================================================
# Gemini 2.5 Flash Lite has a 1M token context window - we use it effectively.
# Research shows: documents first, query last improves accuracy; and
# quote extraction before answering improves accuracy substantially.

# Memory context - NO TRUNCATION with 1M token context window
MEMORY_CONTEXT_CHARS = None  # No truncation - full memory context available

# Chunk/transcript limits - NO truncation for full context
CHUNK_TRANSCRIPT_MAX_CHARS = None  # None = no truncation (use full transcript)
EPISODE_DESCRIPTION_MAX_CHARS = int(os.getenv('EPISODE_DESCRIPTION_MAX_CHARS', '2000'))  # Full descriptions

# Entity and theme tracking - tuned for conversation continuity
MAX_TRACKED_ENTITIES = int(os.getenv('MAX_TRACKED_ENTITIES', '15'))
MAX_TURN_SUMMARY_CHARS = int(os.getenv('MAX_TURN_SUMMARY_CHARS', '500'))
MAX_TRACKED_THEMES = int(os.getenv('MAX_TRACKED_THEMES', '8'))

# Episode limits
EPISODE_SELECTION_LIMIT = int(os.getenv('EPISODE_SELECTION_LIMIT', '15'))

# ==============================================================================
# AGENT ROUTER CONFIGURATION
# ==============================================================================

# Confidence threshold - below this, use fallback route
ROUTER_CONFIDENCE_THRESHOLD = 0.70

# Default fallback route when confidence is low
ROUTER_DEFAULT_FALLBACK = "small_talk"
EPISODE_SEARCH_MAX_EPISODES = 15  # Max episodes to consider
EPISODE_SEARCH_HYDE_COUNT = 3     # HyDE documents for episode search
EPISODE_PINECONE_K = 75           # Top K from Pinecone for episode search

# Episode Search Scoring Weights
EPISODE_PURE_RECENCY_WEIGHTS = {
    "semantic": 0.05,
    "recency": 0.75,
    "person": 0.15,
    "show": 0.05,
}

EPISODE_TOPIC_RECENCY_WEIGHTS = {
    "semantic": 0.40,
    "recency": 0.35,
    "person": 0.15,
    "show": 0.10,
}

EPISODE_PERSON_FOCUSED_WEIGHTS = {
    "semantic": 0.30,
    "recency": 0.20,
    "person": 0.35,
    "show": 0.15,
}

EPISODE_STANDARD_WEIGHTS = {
    "semantic": 0.50,
    "recency": 0.20,
    "person": 0.15,
    "show": 0.15,
}

# Chunk limiting
EPISODE_MAX_CHUNKS_PER_EPISODE = 3

# --- Memory Configuration ---
MAX_ROUTE_HISTORY = 5             # Maximum routing decisions to track
MEMORY_MAX_CHARS = None  # No truncation - 1M context window available

# ==============================================================================
# TESTING CONFIGURATION
# ==============================================================================

# Test timeouts (in seconds)
TEST_TIMEOUT_ROUTING = 10         # Max time for routing test
TEST_TIMEOUT_MEMORY = 5           # Max time for memory test
TEST_TIMEOUT_CONCURRENCY = 60     # Max time for concurrency test (10 parallel requests)
