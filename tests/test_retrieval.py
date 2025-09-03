"""
Basic smoke tests
"""

from retriever import chunk_text
from retriever import retrieve

def test_chunk_text():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, chunk_size=3)
    assert all(len(c.split()) <= 3 for c in chunks)

def test_retrieve():
    chunks = ["i love pizza", "dogs are great pets", "i work on ai"]
    query = "pets"
    results = retrieve(query, chunks, top_k=1)
    assert results[0][0] == "dogs are great pets"
    