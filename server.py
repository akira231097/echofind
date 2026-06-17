"""
FastAPI application for EchoFind.

This is the main entry point for the API server.
Run with: uvicorn server:app --reload --port 8000
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import openai
from pinecone import Pinecone

from api.routes import router, set_agent
from engine.agent import EchoFindAgent
from engine.small_talk import init_grounding_client, is_grounding_available
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    GEMINI_API_KEY,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_entity_data():
    """Load unique authors, personalities, and shows from JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "data", "entities.sample.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        unique_personalities = data.get("unique_personalities", [])
        unique_authors = data.get("unique_authors", [])
        unique_shows = data.get("unique_shows", [])

        logger.info(f"Loaded {len(unique_personalities)} personalities, {len(unique_authors)} authors, {len(unique_shows)} shows")
        return unique_personalities, unique_authors, unique_shows

    except FileNotFoundError:
        logger.error(f"Entity data file not found: {json_path}")
        return [], [], []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse entity data: {e}")
        return [], [], []


def create_clients():
    """Create API clients for OpenAI, Pinecone, and Gemini."""
    # OpenAI client for embeddings
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Pinecone client for vector search
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

    # Gemini client (OpenAI-compatible endpoint)
    gemini_client = openai.OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    return openai_client, pinecone_client, gemini_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - runs on startup and shutdown."""
    # Startup
    logger.info("Starting EchoFind API...")

    # Load entity data
    unique_personalities, unique_authors, unique_shows = load_entity_data()

    # Create clients
    openai_client, pinecone_client, gemini_client = create_clients()

    # Create and register the chatbot agent
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
            "max_chunks_rerank": 35,  # Single reranker call with all 35 chunks
            "max_chunks_selection": 21,  # Top 21 after hybrid scoring → to LLM
            "reranker_top_n": 35,  # Get rerank scores for all chunks
        }
    )

    set_agent(agent)
    logger.info("Chatbot agent initialized successfully")

    # Initialize grounding for small talk (optional - requires google-genai package)
    grounding_success = init_grounding_client()
    if is_grounding_available():
        logger.info("Google Search grounding ENABLED for small talk explanations")
    else:
        logger.warning("Google Search grounding DISABLED - install 'google-genai' to enable")
        logger.warning("Small talk will use knowledge-based responses without live search")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down EchoFind API...")


# Create FastAPI app
app = FastAPI(
    title="EchoFind API",
    description="AI-powered podcast content retrieval chatbot with memory",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# Root endpoint - serve frontend
@app.get("/")
async def root():
    """Serve the frontend HTML."""
    frontend_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    return FileResponse(
        frontend_path,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
