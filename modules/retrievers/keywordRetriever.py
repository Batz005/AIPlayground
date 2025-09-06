from modules.retrievers.base import BaseRetriever

class KeywordRetriever(BaseRetriever):
    """
    Retriever that uses simple keyword overlap between query and chunks.
    """

    def __init__(self, top_k: int = 3):
        """
        Initialize KeywordRetriever.

        Args:
            top_k (int): Number of top results to return by default.
        """
        super().__init__(name="KeywordRetriever", top_k=top_k)

    def _keyword_score(self, query: str, chunk: str) -> int:
        """Score = number of query words that appear in chunk."""
        query_split = query.lower().split()
        chunk_split = set(chunk.lower().split())
        return sum(1 for q in query_split if q in chunk_split)

    def retrieve(self, query: str, chunks: list, *args, **kwargs):
        """
        Retrieve top-k chunks with highest keyword overlap score.

        Args:
            query (str): The query string.
            chunks (list): Candidate chunks of text.

        Returns:
            list[tuple[str, int]]: Top-k (chunk, score) pairs.
        """
        scored = []
        for chunk in chunks:
            score = self._keyword_score(query, chunk)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]
