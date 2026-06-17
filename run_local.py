"""
Local Development Runner

Starts the FastAPI server for local testing.
Run with: python run_local.py
"""

import uvicorn
import os
import sys
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging to show ALL logs (including application logs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to stdout
    ]
)

# Set specific loggers to INFO level
for logger_name in ["engine", "engine.agent", "engine.query_analyzer",
                    "engine.selection", "engine.memory", "retrieval", "root"]:
    logging.getLogger(logger_name).setLevel(logging.INFO)


def main():
    """Run the FastAPI server."""
    print("=" * 60)
    print("  ECHOFIND - Local Development Server")
    print("=" * 60)
    print()
    print("  Frontend:  http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("  Health:    http://localhost:8000/api/health")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled for better logging visibility
        log_level="debug",
        access_log=True,
    )


if __name__ == "__main__":
    main()
