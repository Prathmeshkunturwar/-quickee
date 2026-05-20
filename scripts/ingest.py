"""Embed the processed catalog and push to ChromaDB.

Usage:
  uv run python scripts/ingest.py                # reset + ingest
  uv run python scripts/ingest.py --no-reset     # append (rarely useful)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import structlog  # noqa: E402

from quickee.config import get_settings  # noqa: E402
from quickee.rag.ingest import ingest_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="Do not drop existing collection first")
    args = parser.parse_args()

    logging.basicConfig(level="INFO")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )

    s = get_settings()
    cat = s.data_processed_dir / "catalog.json"
    if not cat.exists():
        print(f"[err] {cat} missing — run scripts/build_catalog.py first")
        return 1

    t0 = time.time()
    n = ingest_catalog(cat, reset=not args.no_reset)
    print(f"\n[ok] ingested {n} items in {time.time() - t0:.1f}s -> {s.chroma_persist_dir}")
    print(f"[ok] collection: {s.chroma_catalog_collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
