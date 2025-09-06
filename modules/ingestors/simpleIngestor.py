from modules.ingestors.base import BaseIngestor
from pathlib import Path
from typing import List

class SimpleIngestor(BaseIngestor):
    """
    Ingestor that splits documents into simple fixed-size word chunks.
    """

    def _load_docs(self, folder: str) -> List[str]:
        texts = []
        files = Path(folder).glob("*.md")
        for file in files:
            texts.append(file.read_text(encoding="utf-8"))
        return texts

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        blocks = text.split()
        for i in range(0, len(blocks), self.chunk_size):
            words = blocks[i:i + self.chunk_size]
            chunks.append(" ".join(words))
        return chunks

    def ingest(self, folder: str) -> List[str]:
        docs = self._load_docs(folder)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self._chunk_text(doc))
        return all_chunks
