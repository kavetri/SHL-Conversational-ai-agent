"""
config.py — Centralized configuration for the SHL Recommender.

Why a separate config file?
- All settings live in ONE place. If you change an API key or model name,
  you change it here — not scattered across 10 files.
- We load secrets from a .env file so they never get accidentally hardcoded.
"""

import os                          # os = lets Python read environment variables
from dotenv import load_dotenv     # load_dotenv = reads your .env file into os.environ

# Load the .env file so GEMINI_API_KEY becomes available via os.getenv()
load_dotenv()

# ── Gemini settings ──────────────────────────────────────────────────────────

# Your Gemini API key — read from .env, never hardcoded
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Which Gemini model to use.
# gemini-1.5-flash = fast, free, good quality. Perfect for this project.
GEMINI_MODEL: str = "gemini-2.5-flash"

# Temperature controls how "creative" the LLM is.
# 0.0 = very deterministic (same input → same output, almost always)
# 1.0 = very creative/random
# We use 0.1 because we want consistent, factual recommendations — not creativity.
GEMINI_TEMPERATURE: float = 0.1

# ── Retrieval settings ───────────────────────────────────────────────────────

# How many catalog items to retrieve from FAISS for each query.
# We fetch 15 but recommend at most 10 — gives the LLM room to pick the best.
TOP_K_RESULTS: int = 15

# ── File paths ───────────────────────────────────────────────────────────────

# Where the scraped catalog JSON lives
CATALOG_PATH: str = "scraper/catalog.json"

# Where the FAISS index and related files are stored
FAISS_INDEX_PATH: str = "retrieval/faiss_store/index.faiss"
FAISS_METADATA_PATH: str = "retrieval/faiss_store/metadata.json"

# ── Conversation limits ───────────────────────────────────────────────────────

# SHL says max 8 turns (user + assistant combined).
# We set our limit to 7 so the agent wraps up gracefully BEFORE hitting 8.
MAX_TURNS: int = 8
