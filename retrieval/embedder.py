"""
embedder.py
Semantic similarity retrieval engine using a locally trained TF-IDF vectorizer.
"""

import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CATALOG_PATH, TOP_K_RESULTS

# Storage paths for model artifacts
TFIDF_VECTORIZER_PATH = "retrieval/faiss_store/vectorizer.pkl"
TFIDF_MATRIX_PATH = "retrieval/faiss_store/matrix.pkl"
METADATA_PATH = "retrieval/faiss_store/metadata.json"

_vectorizer = None
_tfidf_matrix = None
_catalog_metadata = None


def make_searchable_text(assessment: dict) -> str:
    """
    Concatenates catalog assessment metadata fields to form a search document.
    """
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
    
    return f"{name} {name} {name} {type_text} {description}".strip()


def build_index():
    """
    Fits TF-IDF vectorizer and builds retrieval matrix from catalog.json.
    """
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Catalog database not found at {CATALOG_PATH}")
        
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    texts = [make_searchable_text(item) for item in catalog]
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    os.makedirs(os.path.dirname(TFIDF_VECTORIZER_PATH), exist_ok=True)
    
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
        
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
        
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        
    return vectorizer, tfidf_matrix, catalog


def load_index():
    """
    Loads saved vectorizer model and matrix into application context.
    """
    global _vectorizer, _tfidf_matrix, _catalog_metadata
    
    if _vectorizer is not None:
        return
        
    if not os.path.exists(TFIDF_VECTORIZER_PATH) or not os.path.exists(TFIDF_MATRIX_PATH):
        raise FileNotFoundError("Index files not found. Run build_index first.")
        
    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)
        
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        _tfidf_matrix = pickle.load(f)
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _catalog_metadata = json.load(f)


def search(query: str, k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Performs cosine similarity search against index metadata.
    """
    if _vectorizer is None:
        load_index()
        
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score <= 0.0 and len(results) > 0:
            continue
            
        assessment = _catalog_metadata[idx].copy()
        assessment["_score"] = score
        results.append(assessment)
        
    return results
