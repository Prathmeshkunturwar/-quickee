"""Merge per-site raw scrapes into data/processed/catalog.json + print spec compliance.

Usage:  uv run python scripts/build_catalog.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import structlog  # noqa: E402

from quickee.config import get_settings  # noqa: E402
from quickee.rag.build import load_raw_items, save_catalog, summarize  # noqa: E402


def main() -> int:
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
    items = load_raw_items(s.data_raw_dir)
    if not items:
        print("[err] no items loaded — run scripts/scrape.py first")
        return 1

    out = s.data_processed_dir / "catalog.json"
    save_catalog(items, out)

    summary = summarize(items)
    print("\n--- catalog summary ---")
    print(json.dumps(summary, indent=2))
    tops = summary["by_category"].get("top", 0)
    bottoms = summary["by_category"].get("bottom", 0)
    ok = "OK" if tops >= 50 and bottoms >= 50 else "WARN"
    print(f"\n[{ok}] spec target >=50/>=50 -- have {tops} tops + {bottoms} bottoms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
