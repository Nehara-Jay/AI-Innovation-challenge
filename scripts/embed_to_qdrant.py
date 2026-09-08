"""
CLI script to run Voyage AI embedding generation and Qdrant ingestion.
Usage:
    uv run python scripts/embed_to_qdrant.py --qdrant-url http://localhost:6333
    uv run python scripts/embed_to_qdrant.py --batch-size 16 --delay 21
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_pipeline.embed_archive import run_embedding_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate Voyage AI embeddings and ingest into Qdrant vector database."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw extracted JSON file (default: data/extracted_archive.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Voyage AI model name (default: voyage-3 or from .env)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Voyage AI API requests (default: 16)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=21.0,
        help="Delay in seconds between batches to respect 3 RPM / 10k TPM limit (default: 21.0s, set 0 if billed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to embed (useful for test / dry runs)",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection from scratch (default: resume if points exist)",
    )

    args = parser.parse_args()
    input_path = Path(args.input) if args.input else None

    try:
        run_embedding_pipeline(
            input_file=input_path,
            model=args.model,
            batch_size=args.batch_size,
            limit=args.limit,
            recreate_collection=args.recreate,
            qdrant_url=args.qdrant_url,
            delay_between_batches=args.delay,
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
