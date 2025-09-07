from abc import ABC, abstractmethod
from modules.baseModule import BaseModule

class BaseRetriever(BaseModule, ABC):
    """
    Abstract base class for all Retriever modules in AIPlayground.
    Every retriever must inherit from this and implement the `retrieve` method.
    """

    def __init__(self, name: str = None, top_k: int = 3):
        """
        Initialize a retriever module.

        Args:
            name (str, optional): Human-readable identifier for the retriever.
            top_k (int): Number of top results to return by default.
        """
        super().__init__(name)
        self.top_k = top_k
        self.last_results = None

    @abstractmethod
    def retrieve(self, query: str, chunks: list, *args, **kwargs):
        """
        Abstract method that derived retrievers must implement.

        Args:
            query (str): The query string.
            chunks (list): Candidate chunks of text or data to retrieve from.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list: Top-k results ranked according to the retrieval logic.
        """
        raise NotImplementedError("Derived classes must implement `retrieve` method.")

    def run(self, query: str, chunks: list, *args, **kwargs):
        """
        Implements the BaseModule contract. 
        Calls `retrieve` internally so retrievers can also be used as generic modules.
        """
        results = self.retrieve(query, chunks, *args, **kwargs)
        self.last_results = results
        return results