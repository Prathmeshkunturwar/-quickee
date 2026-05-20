"""Shared building blocks for site-specific scrapers.

Design principles:
- Read structured data (JSON-LD, __NEXT_DATA__) when sites expose it — these
  are stable contracts the site emits for Google's crawler. DOM selectors are
  the fallback, not the default.
- Polite by default: one browser context per site, sequential PDP visits with
  randomized jitter; concurrency lives at the orchestrator (cross-site) level.
- Validate every scraped item with pydantic before yielding. Bad data is
  logged-and-skipped, never silently accepted.
"""
from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from quickee.models import Item

log = structlog.get_logger()

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


@asynccontextmanager
async def polite_browser(headless: bool = True) -> AsyncIterator[BrowserContext]:
    """One Playwright context per scraper run. Closes cleanly on exit."""
    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=headless)
        ctx: BrowserContext = await browser.new_context(
            user_agent=DEFAULT_UA,
            viewport={"width": 1366, "height": 900},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
        )
        try:
            yield ctx
        finally:
            await ctx.close()
            await browser.close()


async def polite_sleep(low: float = 0.6, high: float = 1.4) -> None:
    """Randomized small delay between requests — be a good citizen."""
    await asyncio.sleep(random.uniform(low, high))


async def goto_with_retries(page: Page, url: str, timeout_ms: int = 30_000) -> None:
    """Navigate with exponential backoff on transient failures."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except RetryError as e:
        raise RuntimeError(f"goto failed after retries: {url}") from e


async def extract_jsonld(page: Page) -> list[dict]:
    """Return all JSON-LD blocks parsed as Python objects (skips malformed)."""
    out: list[dict] = []
    blocks = await page.locator('script[type="application/ld+json"]').all_inner_texts()
    for raw in blocks:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            out.extend(o for o in obj if isinstance(o, dict))
        elif isinstance(obj, dict):
            out.append(obj)
    return out


def find_product_node(jsonld_blocks: list[dict]) -> dict | None:
    """Walk JSON-LD graphs and return the first Product-typed node."""

    def walk(o):
        if isinstance(o, dict):
            t = o.get("@type")
            if t == "Product" or (isinstance(t, list) and "Product" in t):
                yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    for block in jsonld_blocks:
        for prod in walk(block):
            return prod
    return None


async def extract_next_data(page: Page) -> dict | None:
    """Return parsed __NEXT_DATA__ JSON if present, else None."""
    loc = page.locator('script#__NEXT_DATA__')
    if await loc.count() == 0:
        return None
    raw = await loc.first.inner_text()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("malformed __NEXT_DATA__", length=len(raw))
        return None


async def autoscroll(page: Page, passes: int = 4, pause_ms: int = 700) -> None:
    """Trigger lazy-load by scrolling the page in steps."""
    for _ in range(passes):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.9)")
        await page.wait_for_timeout(pause_ms)


class BaseScraper(ABC):
    """Site-specific scraper contract."""

    brand: str  # set by subclass, e.g. "Uniqlo"

    @abstractmethod
    async def scrape(self, ctx: BrowserContext, limit_per_category: int | None = None) -> list[Item]:
        """Run the full scrape for this site and return validated Items."""
        ...
