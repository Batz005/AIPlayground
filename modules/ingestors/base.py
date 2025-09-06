from abc import ABC, abstractmethod
from modules.baseModule import BaseModule
from typing import List


class BaseIngestor(BaseModule, ABC):
    """
    Abstract base class for all Ingestor modules.
    Responsible for reading, cleaning, and chunking documents.
    """

    def __init__(self, name: str = None, chunk_size: int = None):
        super().__init__(name)
        self.chunk_size = chunk_size or self.config["retriever"]["chunk_size"]
        self.last_chunks: List[str] = []  # store last ingested chunks

    @abstractmethod
    def ingest(self, folder: str) -> List[str]:
        """
        Abstract method that derived ingestors must implement.
        Should also update self.last_chunks with the produced chunks.
        """
        raise NotImplementedError

    def run(self, folder: str, *args, **kwargs) -> List[str]:
        """Implements BaseModule contract by calling ingest()."""
        chunks = self.ingest(folder)
        self.last_chunks = chunks
        return chunks

    @staticmethod
    def print_ingest_summary(chunks: List[str]) -> None:
        """
        Print a summary of the ingested chunks for debugging.
        Shows total chunks, average length, min/max length.
        """
        lengths = [len(c.split()) for c in chunks]
        if not lengths:
            print("No chunks loaded.")
            return
        print(f"Total chunks: {len(chunks)}")
        print(f"Avg length: {sum(lengths)//len(lengths)} words")
        print(f"Min length: {min(lengths)} words")
        print(f"Max length: {max(lengths)} words")