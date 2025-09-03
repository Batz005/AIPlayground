"""
CLI entrypoint for querying docs
"""

import sys
import yaml
from pathlib import Path
from retriever import ingest, retrieve, print_ingest_summary

# Load config
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py 'your question'")
        return
    
    query = sys.argv[1]
    chunk_size = config["retriever"]["chunk_size"]
    if len(sys.argv) >= 3:
        chunk_size = int(sys.argv[2])
    chunks = ingest("data/docs", chunk_size)
    print_ingest_summary(chunks)  # Show debug summary of chunks
    results = retrieve(query, chunks, top_k=3)

    print("Query:", query)
    print("Results:")
    for text, score in results:
        print(f"[score={score}] {text[:200]}...")

if __name__ == "__main__":
    main()