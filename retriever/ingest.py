"""
Ingest documents: read, clean, chunk
"""

from pathlib import Path
from typing import List
import yaml
from pathlib import Path
import re
import nltk
import json

# Load config
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_docs(folder: str) -> List[str]:
    """
    Read all .md/.txt files in folder and return list of raw texts.
    """
    texts = []

    files = Path(folder).glob("*.md")
    for file in files:
        texts.append(file.read_text(encoding="utf-8"))
    return texts

def chunk_text(text: str, chunk_size: int = config["retriever"]["chunk_size"]) -> List[str]:
    """
    Split text into chunks of ~chunk_size words.
    """
    chunks = []
    blocks = text.split()
    for i in range(0, len(blocks), chunk_size):
        words = blocks[i:i + chunk_size]
        chunks.append(" ".join(words))
    return chunks


def chunk_text_advanced(text: str, chunk_size: int = 150, min_size: int = 50) -> List[str]:
    """
    Split text into chunks using headings -> sentences.
    - First split on markdown headings.
    - Then split sections into sentences (using nltk).
    - Group sentences until chunk_size words.
    - Merge small remainders (< min_size) into previous chunk.
    """
    chunks = []
    # Split by markdown headings (keep heading with section)
    sections = re.split(r'(?m)(^#+\s.*)', text)
    # Merge back heading + content
    merged_sections = []
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        content = sections[i+1] if i+1 < len(sections) else ""
        merged_sections.append(heading + "\n" + content)

    if sections and sections[0].strip():
        merged_sections.insert(0, sections[0].strip())
    
    for section in merged_sections if merged_sections else [text]:
        # Split into sentences with nltk
        try:
            sentences = nltk.sent_tokenize(section)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(section)
        
        current_chunk = []
        current_len = 0

        for sent in sentences:
            words = sent.split()
            if current_len + len(words) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.extend(words)
            current_len += len(words)

        if current_chunk:
            # Handle leftover: merge if too small
            if len(current_chunk) < min_size and chunks:
                chunks[-1] += " " + " ".join(current_chunk)
            else:
                chunks.append(" ".join(current_chunk))

    return chunks

def ingest(folder: str, chunk_size: int = config["retriever"]["chunk_size"]) -> List[str]:
    """
    Full ingest pipeline: load all docs → chunk → return chunks.
    Uses cache if available.
    """
    cache_file = Path("./cache/chunks.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True) 

    # 1. Try cache
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return chunks

    # 2. No cache → compute fresh
    docs = load_docs(folder)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text_advanced(doc, chunk_size))
    
    # 3. Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    return all_chunks

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