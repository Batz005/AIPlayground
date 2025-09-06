"""
Retrieval methods: keyword-based and semantic, with optional embedding cache
"""

from typing import List, Tuple
import numpy as np
import pickle
import yaml
from pathlib import Path
from utils.vectorizer import vectorize_string, vectorize_all

# Load config
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Embedding cache file
EMBED_CACHE_FILE = Path(config["data"]["embeddings_cache_path"])
if EMBED_CACHE_FILE.exists():
    with open(EMBED_CACHE_FILE, "rb") as f:
        _embedding_cache = pickle.load(f)
else:
    _embedding_cache = {}

def _save_embedding_cache():
    EMBED_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_CACHE_FILE, "wb") as f:
        pickle.dump(_embedding_cache, f)


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


def _retrieve_keyword(query: str, chunks: List[str], top_k: int) -> List[Tuple[str, int]]:
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
# Semantic retrieval (with caching)
# -----------------------------
def cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def _get_embedding(text: str, model_name: str = None):
    """Get embedding for text from cache or compute if missing."""
    key = (text, model_name)
    if key in _embedding_cache:
        return _embedding_cache[key]
    emb = vectorize_string(text, model_name=model_name)
    _embedding_cache[key] = emb
    _save_embedding_cache()
    return emb


def _retrieve_semantic(query: str, chunks: List[str], top_k: int = 3, model_name: str = None) -> List[Tuple[str, float]]:
    """
    Return top_k chunks ranked by semantic similarity (embeddings + cosine similarity).
    Uses embedding cache for efficiency.
    """
    query_vec = _get_embedding(query, model_name=model_name)
    scored = []
    for chunk in chunks:
        vec = _get_embedding(chunk, model_name=model_name)
        scored.append((chunk, cosine_similarity(query_vec, vec)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# -----------------------------
# Unified retrieval entrypoint
# -----------------------------
def retrieve(
    query: str,
    chunks: List[str],
    top_k: int,
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