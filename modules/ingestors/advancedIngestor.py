from modules.ingestors.base import BaseIngestor
from pathlib import Path
from typing import List
import re, nltk

class AdvancedIngestor(BaseIngestor):
    """
    Ingestor that splits text by markdown headings and sentences, then groups into chunks.
    """

    def __init__(self, name: str = None, chunk_size: int = 150, min_size: int = 50):
        super().__init__(name, chunk_size)
        self.min_size = min_size

    def _load_docs(self, folder: str) -> List[str]:
        texts = []
        files = Path(folder).glob("*.md")
        for file in files:
            texts.append(file.read_text(encoding="utf-8"))
        return texts

    def _chunk_text_advanced(self, text: str) -> List[str]:
        chunks = []
        # Split by markdown headings (keep heading with section)
        sections = re.split(r'(?m)(^#+\s.*)', text)
        merged_sections = []
        for i in range(1, len(sections), 2):
            heading = sections[i].strip()
            content = sections[i+1] if i+1 < len(sections) else ""
            merged_sections.append(heading + "\n" + content)
        if sections and sections[0].strip():
            merged_sections.insert(0, sections[0].strip())

        for section in merged_sections if merged_sections else [text]:
            try:
                sentences = nltk.sent_tokenize(section)
            except LookupError:
                nltk.download("punkt", quiet=True)
                sentences = nltk.sent_tokenize(section)

            current_chunk = []
            current_len = 0
            for sent in sentences:
                words = sent.split()
                if current_len + len(words) > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.extend(words)
                current_len += len(words)

            if current_chunk:
                if len(current_chunk) < self.min_size and chunks:
                    chunks[-1] += " " + " ".join(current_chunk)
                else:
                    chunks.append(" ".join(current_chunk))
        return chunks

    def ingest(self, folder: str) -> List[str]:
        docs = self._load_docs(folder)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self._chunk_text_advanced(doc))
        return all_chunks
