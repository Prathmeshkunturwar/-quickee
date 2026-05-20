"""Run the catalog scrape — Uniqlo + Bewakoof in parallel; saves data/raw/*.json.

Usage:
  uv run python scripts/scrape.py                  # full run
  uv run python scripts/scrape.py --limit 5        # smoke run, 5 per category
  uv run python scripts/scrape.py --site uniqlo    # one site only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import structlog  # noqa: E402

from quickee.config import get_settings  # noqa: E402
from quickee.models import Item  # noqa: E402
from quickee.scraper.base import polite_browser  # noqa: E402
from quickee.scraper.bewakoof import BewakoofScraper  # noqa: E402
from quickee.scraper.uniqlo import UniqloScraper  # noqa: E402


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper())
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), 20)),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )


async def _run_one(scraper, limit: int | None) -> list[Item]:
    async with polite_browser(headless=True) as ctx:
        return await scraper.scrape(ctx, limit_per_category=limit)


def _save(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [it.model_dump() for it in items]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["uniqlo", "bewakoof", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max items per category")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log)

    s = get_settings()
    out_dir = s.data_raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = structlog.get_logger()

    start = time.time()
    if args.site in ("uniqlo", "all"):
        log.info("scrape.start", site="uniqlo")
        items = await _run_one(UniqloScraper(), args.limit)
        _save(items, out_dir / "uniqlo.json")
        log.info("scrape.done", site="uniqlo", n=len(items))

    if args.site in ("bewakoof", "all"):
        log.info("scrape.start", site="bewakoof")
        items = await _run_one(BewakoofScraper(), args.limit)
        _save(items, out_dir / "bewakoof.json")
        log.info("scrape.done", site="bewakoof", n=len(items))

    log.info("scrape.complete", elapsed=f"{time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
