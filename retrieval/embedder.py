"""
embedder.py — Semantic search engine using TF-IDF (Scikit-Learn).

Why TF-IDF instead of PyTorch or Gemini API?
1. PyTorch (sentence-transformers) uses >500MB of RAM, causing Render Free Tier to crash with "Out of Memory".
2. Gemini API has a strict rate limit (100 requests per minute on Free Tier), which gets blocked when we batch-embed the 377 catalog items.
3. TF-IDF (Term Frequency-Inverse Document Frequency):
   - Runs 100% locally.
   - Zero API costs, zero rate limits.
   - Extremely low memory footprint (<10MB RAM).
   - Instant startup (no model loading or downloads).
   - Excellent for keyword-matching assessments like ".NET", "Java", "SQL", "OPQ".
"""

import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CATALOG_PATH,
    TOP_K_RESULTS,
)

# Paths for saving TF-IDF artifacts
# We save under retrieval/faiss_store/ to match configuration paths cleanly
TFIDF_VECTORIZER_PATH = "retrieval/faiss_store/vectorizer.pkl"
TFIDF_MATRIX_PATH = "retrieval/faiss_store/matrix.pkl"
METADATA_PATH = "retrieval/faiss_store/metadata.json"

# ── Text Preparation ──────────────────────────────────────────────────────────

def make_searchable_text(assessment: dict) -> str:
    """Combine fields into one searchable string."""
    type_labels = {
        "K": "Knowledge Skills Technical",
        "P": "Personality Behavior OPQ Work Style",
        "A": "Ability Aptitude Cognitive Reasoning Critical Thinking",
        "S": "Simulation Interactive Roleplay",
        "B": "Biodata Situational Judgment Scenario",
        "C": "Competencies",
        "D": "Development 360",
        "E": "Assessment Exercise",
    }
    
    name = assessment.get("name", "").strip()
    test_type = assessment.get("test_type", "").strip()
    description = assessment.get("description", "").strip()
    
    type_text = " ".join(
        type_labels.get(t.strip(), t.strip())
        for t in test_type.split(",")
    )
    
    # We repeat the name 3 times to give it massive weight in keyword matching
    combined = f"{name} {name} {name} {type_text} {description}"
    return combined.strip()


# ── Index Building ────────────────────────────────────────────────────────────

def build_index():
    """Build TF-IDF matrix and save it to disk."""
    print("Building TF-IDF search index...")
    
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}")
        
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    print(f"  Loaded {len(catalog)} assessments from catalog")
    
    # Prepare text for indexing
    texts = [make_searchable_text(item) for item in catalog]
    
    # Initialize TF-IDF Vectorizer
    # ngram_range=(1, 2) lets it capture both single words ("Java") and phrases ("Spring Boot")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,  # logarithmic scale for term frequency to prevent bias from long texts
    )
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  TF-IDF Matrix shape: {tfidf_matrix.shape}")
    
    # Save artifacts
    os.makedirs(os.path.dirname(TFIDF_VECTORIZER_PATH), exist_ok=True)
    
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
        
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
        
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        
    print(f"  TF-IDF index saved successfully!")
    return vectorizer, tfidf_matrix, catalog


# ── Index Loading ─────────────────────────────────────────────────────────────

_vectorizer = None
_tfidf_matrix = None
_catalog_metadata = None

def load_index():
    """Load the TF-IDF vectorizer and matrix into memory."""
    global _vectorizer, _tfidf_matrix, _catalog_metadata
    
    if _vectorizer is not None:
        return
        
    if not os.path.exists(TFIDF_VECTORIZER_PATH) or not os.path.exists(TFIDF_MATRIX_PATH):
        raise FileNotFoundError("TF-IDF artifacts not found. Run build_index() first.")
        
    print("Loading TF-IDF search index...")
    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)
        
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        _tfidf_matrix = pickle.load(f)
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _catalog_metadata = json.load(f)
        
    print(f"TF-IDF index loaded: {len(_catalog_metadata)} assessments indexed")


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, k: int = TOP_K_RESULTS) -> list[dict]:
    """Search the catalog using cosine similarity on TF-IDF vectors."""
    if _vectorizer is None:
        load_index()
        
    # Transform query to TF-IDF vector
    query_vec = _vectorizer.transform([query])
    
    # Compute cosine similarity between query vector and all catalog vectors
    # similarity shape will be (1, num_assessments)
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    
    # Get top K indices sorted by score descending
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        # Only return items with a non-zero similarity score to filter out irrelevant items
        # If no keywords match, score is 0.0
        if score <= 0.0 and len(results) > 0:
            continue
            
        assessment = _catalog_metadata[idx].copy()
        assessment["_score"] = score
        results.append(assessment)
        
    return results


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_index()
    load_index()
    test_queries = [
        "Java Developer Spring",
        "personality OPQ32",
        "sales supervisor ability",
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        res = search(q, k=3)
        for r in res:
            print(f"  [{r['test_type']}] {r['name']} (score: {r['_score']:.3f})")
