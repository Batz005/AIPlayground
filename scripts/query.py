"""
CLI entrypoint for querying docs using pipelines
"""

import argparse
import yaml
import json
import time
from pathlib import Path
from modules.pipelines.factory import PipelineFactory

# Load config
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Query the AIPlayground knowledge base")
    parser.add_argument("query", type=str, help="Your query/question")
    parser.add_argument("--pipeline", type=str, default="qa", help="Pipeline to run (default: qa)")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild chunk and embedding cache")
    parser.add_argument("--json", action="store_true", help="Return results as JSON")
    parser.add_argument("--timing", action="store_true", help="Print timing for each step and total time")
    args = parser.parse_args()

    total_start = time.time()

    # Handle cache rebuild
    if args.rebuild_cache:
        Path(config["data"]["chunks_cache_path"]).unlink(missing_ok=True)
        Path(config["data"]["embeddings_cache_path"]).unlink(missing_ok=True)
        print("Cache cleared: chunks + embeddings")

    # Build pipeline
    pipeline = PipelineFactory.create(config, args.pipeline)

    # Run pipeline
    t0 = time.time()
    results = pipeline.run(args.query, folder=config["data"]["docs_path"])
    t1 = time.time()

    # Output results
    print("Query:", args.query)
    print("Results:")
    if args.json:
        print(json.dumps([{"score": score, "text": text} for text, score in results], indent=2, ensure_ascii=False))
    else:
        for text, score in results:
            snippet = " ".join(text.split()[:40])  # ~40 words instead of 200 chars
            print(f"[score={score:.4f}] {snippet}...")

    if args.timing:
        print(f"Pipeline time: {t1 - t0:.2f}s")
        print(f"Total time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()