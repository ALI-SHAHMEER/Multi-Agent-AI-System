"""
config.py  –  Central configuration for the Research Assistant.
Loads environment variables and exposes typed constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ─── LLM ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = "gpt-4o-mini"          # cheap & fast for development
EMBEDDING_MODEL: str = "text-embedding-3-small"

# ─── External APIs ────────────────────────────────────────────────────────────
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# ─── Vector store ─────────────────────────────────────────────────────────────
VECTOR_DB_PATH: str = "./data/vector_db"     # FAISS index persisted here
CHROMA_DB_PATH: str = "./data/chroma_db"     # Chroma DB persisted here
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ─── MCP / data paths ─────────────────────────────────────────────────────────
FILESYSTEM_BASE_PATH: str = os.getenv("FILESYSTEM_BASE_PATH", "./data")
GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

# ─── Memory ───────────────────────────────────────────────────────────────────
MAX_CONVERSATION_TOKENS: int = 2000      # trim buffer beyond this
SUMMARY_MAX_TOKENS: int = 300            # max tokens for summary memory

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is missing. Copy .env.example to .env and fill it in."
    )
