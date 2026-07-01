"""
embedder.py — Builds and searches the FAISS vector index.

What this file does:
1. build_index() — Takes catalog.json, converts each assessment to an
   embedding (list of numbers representing meaning), stores in FAISS.
   Run ONCE after scraping.

2. search() — Takes a user query, converts it to an embedding,
   finds the most similar assessments in FAISS. Called on every /chat request.

WHY embeddings instead of keyword search:
- Keyword: "Java developer" won't find "Core Java (Advanced Level)"
- Embedding: "Java developer" WILL find "Core Java (Advanced Level)"
  because they're semantically similar (same meaning cluster in vector space)
"""

import json                           # for reading catalog.json
import os                             # for file path operations
import numpy as np                    # numpy = fast math with arrays
import faiss                          # faiss = vector similarity search library
from sentence_transformers import SentenceTransformer  # converts text → vectors

from config import (
    CATALOG_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    TOP_K_RESULTS,
)

# ── Model Setup ───────────────────────────────────────────────────────────────

# This is the embedding model — it converts text to a 384-dimensional vector.
# "all-MiniLM-L6-v2" is a small, fast, and good model for semantic similarity.
# It runs LOCALLY — no API calls needed, completely free.
# First time it runs, it downloads the model (~80MB). After that it's cached.
MODEL_NAME = "all-MiniLM-L6-v2"
model_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

# We load the model once at module level so it's not reloaded on every search.
# This is called "module-level initialization" — happens once when Python
# first imports this file.
print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_path)
print(f"Embedding model loaded: {MODEL_NAME}")


# ── Text Preparation ──────────────────────────────────────────────────────────

def make_searchable_text(assessment: dict) -> str:
    """
    Combine all fields of an assessment into one searchable text string.
    
    Why? FAISS searches vectors, not structured data.
    We need to encode all useful info into ONE text before embedding.
    
    For example:
        {name: "Core Java Advanced", test_type: "K", description: "Tests Java..."}
    becomes:
        "Core Java Advanced Knowledge Skills Tests Java..."
    
    The embedding model then converts this whole string into a vector.
    """
    # Map type codes to readable words for better semantic matching
    type_labels = {
        "K": "Knowledge Skills",
        "P": "Personality Behavior",
        "A": "Ability Aptitude Cognitive Reasoning",
        "S": "Simulation",
        "B": "Biodata Situational Judgment",
        "C": "Competencies",
        "D": "Development 360",
        "E": "Assessment Exercise",
    }
    
    name = assessment.get("name", "")
    test_type = assessment.get("test_type", "")
    description = assessment.get("description", "")
    
    # Expand type codes to full words
    type_text = " ".join(
        type_labels.get(t.strip(), t.strip())
        for t in test_type.split(",")
    )
    
    # Combine everything into one string
    # We repeat the name twice to give it more "weight" in the embedding
    combined = f"{name} {name} {type_text} {description}"
    
    return combined.strip()


# ── Index Building ────────────────────────────────────────────────────────────

def build_index():
    """
    Read catalog.json, embed every assessment, store in FAISS.
    
    Run this ONCE after scraping:
        python -c "from retrieval.embedder import build_index; build_index()"
    
    What FAISS does:
    - Stores vectors (lists of 384 numbers) for all assessments
    - When you search, it finds the K most similar vectors using cosine similarity
    - This is much faster than comparing every vector manually
    """
    print("Building FAISS index from catalog...")
    
    # Load the scraped catalog
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    
    print(f"  Loaded {len(catalog)} assessments from catalog")
    
    # Create searchable text for each assessment
    texts = [make_searchable_text(item) for item in catalog]
    
    # Convert all texts to embedding vectors
    # This is the heavy step — runs through the neural network for each text
    print("  Generating embeddings (this takes a minute)...")
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,    # shows a progress bar in the terminal
        batch_size=32,             # process 32 at a time for efficiency
        normalize_embeddings=True, # normalize = required for cosine similarity
    )
    
    # embeddings is now a 2D numpy array: shape = (num_assessments, 384)
    # Each row is one assessment's embedding vector
    
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Create a FAISS index
    # IndexFlatIP = "Flat" index with "Inner Product" (dot product) similarity
    # With normalized vectors, inner product = cosine similarity
    dimension = embeddings.shape[1]     # 384 — the size of each vector
    index = faiss.IndexFlatIP(dimension)
    
    # Add all embeddings to the index
    # FAISS expects float32 numpy arrays
    index.add(embeddings.astype(np.float32))
    
    print(f"  FAISS index built with {index.ntotal} vectors")
    
    # Save the FAISS index to disk
    # We save it so we don't have to rebuild every time the server starts
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save the catalog metadata separately
    # FAISS only stores vectors (numbers), not the original text/URLs.
    # We save the catalog as a parallel list so we can look up details by index.
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    
    print(f"  FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"  Metadata saved to: {FAISS_METADATA_PATH}")
    print("Index building complete!")
    
    return index, catalog


# ── Index Loading ─────────────────────────────────────────────────────────────

# Module-level variables to hold the loaded index
# Loaded once when the module is first imported, reused for every search
_faiss_index = None
_catalog_metadata = None


def load_index():
    """
    Load the FAISS index and catalog metadata from disk.
    Called once when the server starts.
    """
    global _faiss_index, _catalog_metadata
    
    if _faiss_index is not None:
        return   # already loaded, skip
    
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run: python scraper/scrape_catalog.py && "
            "python -c \"from retrieval.embedder import build_index; build_index()\""
        )
    
    print("Loading FAISS index from disk...")
    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        _catalog_metadata = json.load(f)
    
    print(f"FAISS index loaded: {_faiss_index.ntotal} assessments indexed")


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Search the catalog for assessments most relevant to the query.
    
    How it works:
    1. Convert query text to an embedding vector (same model as build_index)
    2. Ask FAISS: "what are the K most similar vectors to this one?"
    3. FAISS returns indices of the top K matches
    4. Look up those indices in the catalog metadata to get assessment details
    5. Return the list of matching assessments
    
    Args:
        query: the search query (e.g., "Java developer mid-level backend")
        k: how many results to return (default from config)
    
    Returns:
        list of assessment dicts with name, url, test_type, description
    
    Example:
        results = search("personality test for sales role")
        # Returns: [OPQ32r, OPQ MQ Sales Report, ...]
    """
    # Ensure the index is loaded
    if _faiss_index is None:
        load_index()
    
    # Convert query to embedding
    # [query] = list because encode() expects a list
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,   # must match how we built the index
    )
    
    # Convert to float32 numpy array (FAISS requirement)
    query_vector = query_embedding.astype(np.float32)
    
    # Search! 
    # D = distances (similarity scores), I = indices of nearest neighbors
    D, I = _faiss_index.search(query_vector, k)
    
    # D[0] and I[0] because we searched for 1 query (the [0] picks the first result set)
    results = []
    for distance, idx in zip(D[0], I[0]):
        if idx == -1:
            continue    # FAISS returns -1 for empty slots (shouldn't happen, but safe)
        
        # Look up the assessment by its index in the catalog
        assessment = _catalog_metadata[idx].copy()
        
        # Add the similarity score (useful for debugging)
        assessment["_score"] = float(distance)
        
        results.append(assessment)
    
    return results


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test: run this to verify the index works.
    Usage: python retrieval/embedder.py
    """
    load_index()
    
    test_queries = [
        "Java developer backend senior",
        "personality test leadership executive",
        "customer service contact center entry level",
        "numerical reasoning graduate",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search(query, k=3)
        for r in results:
            print(f"  [{r['test_type']}] {r['name']} (score: {r['_score']:.3f})")
            print(f"       {r['url']}")
