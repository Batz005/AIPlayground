"""
Retrieval methods: keyword-based and semantic
"""

from typing import List, Tuple
import numpy as np
from retriever.vectorizer import vectorize_string, vectorize_all


# -----------------------------
# Keyword-based retrieval
# -----------------------------
def _keyword_score(query: str, chunk: str) -> int:
    """
    Score = number of query words that appear in chunk.
    """
    query_split = query.lower().split()
    chunk_split = set(chunk.lower().split())
    count = 0
    for q in query_split:
        if q in chunk_split:
            count += 1
    return count


def _retrieve_keyword(query: str, chunks: List[str], top_k: int = 3) -> List[Tuple[str, int]]:
    """
    Return top_k chunks with highest keyword score.
    """
    scored = []
    for chunk in chunks:
        score = _keyword_score(query, chunk)
        if score > 0:
            scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# -----------------------------
# Semantic retrieval
# -----------------------------
def cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def _retrieve_semantic(query: str, chunks: List[str], top_k: int = 3, model_name: str = None) -> List[Tuple[str, float]]:
    """
    Return top_k chunks ranked by semantic similarity (embeddings + cosine similarity).
    """
    query_vec = vectorize_string(query, model_name=model_name)
    chunk_vecs = vectorize_all(chunks, model_name=model_name)

    scored = [(chunk, cosine_similarity(query_vec, vec)) for chunk, vec in zip(chunks, chunk_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# -----------------------------
# Unified retrieval entrypoint
# -----------------------------
def retrieve(
    query: str,
    chunks: List[str],
    top_k: int = 3,
    method: str = "semantic",
    model_name: str = None,
):
    """
    Unified retrieval entrypoint.
    method: "keyword" or "semantic"
    """
    if method == "keyword":
        return _retrieve_keyword(query, chunks, top_k)
    elif method == "semantic":
        return _retrieve_semantic(query, chunks, top_k, model_name=model_name)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")