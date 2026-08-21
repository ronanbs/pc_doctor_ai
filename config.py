"""
config.py
Central configuration: API key loading, model names, and pipeline
parameters shared across ingestion, embedding, and the Streamlit app.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file, if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # Don't crash on import (e.g. during CI or doc-only tasks), but warn loudly.
    print(
        "[config] WARNING: OPENAI_API_KEY not found in environment. "
        "Set it in a .env file (OPENAI_API_KEY=sk-...) before running the app."
    )

# Models
CHAT_MODEL = "gpt-4o-mini"          # cheap + fast, good for a course project
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking parameters (character-based, per course requirement)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4  # number of chunks retrieved per query

# Chroma DB persistence path
CHROMA_DB_DIR = "data/chroma_db"
