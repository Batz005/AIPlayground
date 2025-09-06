import numpy as np
import pickle
from pathlib import Path
from modules.retrievers.base import BaseRetriever
from utils.vectorizer import vectorize_string


class SemanticRetriever(BaseRetriever):
    """
    Retriever that uses embeddings + cosine similarity with optional caching.
    """

    def __init__(self, model_name: str = None, top_k: int = 3, use_cache: bool = True):
        """
        Initialize SemanticRetriever.

        Args:
            model_name (str, optional): Name of the embedding model.
            top_k (int): Number of top results to return by default.
            use_cache (bool): Whether to enable embedding caching.
        """
        super().__init__(name="SemanticRetriever", top_k=top_k)
        self.model_name = model_name or self.config["embedding"]["model_name"]
        self.use_cache = use_cache

        # Embedding cache
        self.cache_file = Path(self.config["data"]["embeddings_cache_path"])
        if self.use_cache and self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                self._embedding_cache = pickle.load(f)
        else:
            self._embedding_cache = {}

    def _save_embedding_cache(self):
        if not self.use_cache:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(self._embedding_cache, f)

    def _cosine_similarity(self, vec_a, vec_b) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

    def _get_embedding(self, text: str):
        """Get embedding from cache if enabled, else compute directly."""
        key = (text, self.model_name)
        if self.use_cache and key in self._embedding_cache:
            return self._embedding_cache[key]

        emb = vectorize_string(text, model_name=self.model_name)
        if self.use_cache:
            self._embedding_cache[key] = emb
            self._save_embedding_cache()
        return emb

    def retrieve(self, query: str, chunks: list, *args, **kwargs):
        """
        Retrieve top-k most relevant chunks based on semantic similarity.

        Args:
            query (str): The query string.
            chunks (list): Candidate chunks of text.

        Returns:
            list[tuple[str, float]]: Top-k (chunk, score) pairs.
        """
        query_vec = self._get_embedding(query)
        scored = []
        for chunk in chunks:
            vec = self._get_embedding(chunk)
            scored.append((chunk, self._cosine_similarity(query_vec, vec)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]
    
