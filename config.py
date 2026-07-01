"""
config.py
Centralized configuration settings for the SHL Assessment Recommender API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.1

# Retrieval settings
TOP_K_RESULTS: int = 15

# File paths
CATALOG_PATH: str = "scraper/catalog.json"
FAISS_INDEX_PATH: str = "retrieval/faiss_store/index.faiss"
FAISS_METADATA_PATH: str = "retrieval/faiss_store/metadata.json"

# Conversation constraints
MAX_TURNS: int = 8
