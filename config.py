"""
config.py
Central configuration: API key loading, model names, and pipeline
parameters shared across ingestion, embedding, and the Streamlit app.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR.parent / "lesson 2" / ".env")
load_dotenv(APP_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # Don't crash on import (e.g. during CI or doc-only tasks), but warn loudly.
    print(
        "[config] WARNING: OPENAI_API_KEY not found in environment. "
        "Set it in a .env file (OPENAI_API_KEY=sk-...) before running the app."
    )

# Models
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4  # default number of chunks retrieved per query
MAX_RETRIEVAL_DISTANCE = None  # optional L2 distance cutoff; None = no filter

# Generation
CHAT_TEMPERATURE = 0.2

# Chroma DB persistence path
CHROMA_DB_DIR = str(APP_DIR / "data" / "chroma_db")
