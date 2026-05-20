"""CLI tester: send a prompt straight to the LangGraph agent, print JSON.

Usage:
  uv run python scripts/ask.py "I have navy chinos, what should I wear to a yacht party?"
  uv run python scripts/ask.py "Full outfit under 3000 for a casual office day" --budget 3000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import structlog  # noqa: E402

from quickee.agent.graph import get_graph  # noqa: E402
from quickee.api.main import _build_response_from_state  # noqa: E402
from quickee.cache.semantic_cache import get_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="+", help="Free-text style brief")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper())
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log.upper(), logging.INFO)
        ),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    prompt = " ".join(args.prompt)

    cache = get_cache()
    cache_hit = False
    if not args.no_cache:
        cached = cache.get(prompt)
        if cached is not None:
            cache_hit = True
            cached["cache_hit"] = True
            print("\n=== CACHE HIT -- returning stored response ===\n")
            print(json.dumps(cached, indent=2, ensure_ascii=False))
            return 0

    t0 = time.time()
    graph = get_graph()
    final = graph.invoke({
        "user_prompt": prompt,
        "max_budget_inr": args.budget,
        "trace": [],
        "candidates": {},
        "compose_retry": 0,
        "cache_hit": False,
    })
    elapsed = time.time() - t0

    resp = _build_response_from_state(final, cache_hit=cache_hit)
    if resp.items and not args.no_cache:
        cache.put(prompt, resp.model_dump())

    print(f"\n=== response in {elapsed:.2f}s ===\n")
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
