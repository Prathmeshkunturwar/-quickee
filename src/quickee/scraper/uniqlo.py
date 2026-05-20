"""Uniqlo India scraper.

PLP discovery: anchors matching /in/en/products/E*/<sku>?colorDisplayCode=N
PDP extraction: single JSON-LD @graph block per product page — has name, price,
color, material, description, image[].
"""
from __future__ import annotations

from urllib.parse import urljoin

import structlog
from playwright.async_api import BrowserContext, Page

from quickee.models import Category, Item
from quickee.scraper.base import (
    BaseScraper,
    autoscroll,
    extract_jsonld,
    find_product_node,
    goto_with_retries,
    polite_sleep,
)
from quickee.scraper.normalize import (
    infer_subcategory,
    normalize_color,
    parse_price,
    simplify_material,
)

log = structlog.get_logger()

BASE = "https://www.uniqlo.com"

# (PLP URL, broad Category)  — subcategory is inferred per item from name
CATEGORIES: list[tuple[str, Category]] = [
    ("https://www.uniqlo.com/in/en/men/tops", Category.TOP),
    ("https://www.uniqlo.com/in/en/men/bottoms", Category.BOTTOM),
]


class UniqloScraper(BaseScraper):
    brand = "Uniqlo"

    async def scrape(self, ctx: BrowserContext, limit_per_category: int | None = None) -> list[Item]:
        items: list[Item] = []
        for plp_url, category in CATEGORIES:
            log.info("uniqlo.plp.start", url=plp_url, category=category.value)
            page = await ctx.new_page()
            try:
                urls = await self._collect_product_urls(page, plp_url)
            finally:
                await page.close()
            log.info("uniqlo.plp.done", url=plp_url, found=len(urls))
            if limit_per_category:
                urls = urls[:limit_per_category]
            for i, url in enumerate(urls):
                page = await ctx.new_page()
                try:
                    item = await self._extract_item(page, url, category)
                    if item:
                        items.append(item)
                        log.info(
                            "uniqlo.pdp.ok",
                            i=i + 1,
                            n=len(urls),
                            name=item.name[:40],
                            color=item.color,
                            price=item.price_inr,
                        )
                except Exception as e:
                    log.warning("uniqlo.pdp.err", url=url, err=str(e))
                finally:
                    await page.close()
                await polite_sleep()
        return items

    async def _collect_product_urls(self, page: Page, plp_url: str) -> list[str]:
        await goto_with_retries(page, plp_url)
        # Wait for the grid to populate, then scroll to load lazy cards.
        await page.wait_for_timeout(2500)
        await autoscroll(page, passes=4, pause_ms=700)
        # Each card has at least one anchor to /in/en/products/<code>/<sku>?...
        hrefs = await page.locator('a[href*="/in/en/products/"]').evaluate_all(
            "els => els.map(e => e.getAttribute('href'))"
        )
        seen: set[str] = set()
        ordered: list[str] = []
        for href in hrefs:
            if not href:
                continue
            absolute = urljoin(BASE, href.split("#", 1)[0])
            # Trim duplicates that differ only in query-string color codes; keep first occurrence.
            base = absolute.split("?", 1)[0]
            if base in seen:
                continue
            seen.add(base)
            ordered.append(absolute)
        return ordered

    async def _extract_item(self, page: Page, url: str, category: Category) -> Item | None:
        await goto_with_retries(page, url)
        await page.wait_for_timeout(1500)
        blocks = await extract_jsonld(page)
        prod = find_product_node(blocks)
        if not prod:
            log.warning("uniqlo.pdp.no_jsonld", url=url)
            return None

        name = prod.get("name")
        description = prod.get("description") or ""
        if not name:
            return None

        offers = prod.get("offers")
        price = None
        if isinstance(offers, dict):
            price = parse_price(offers.get("price"))
        elif isinstance(offers, list) and offers:
            price = parse_price(offers[0].get("price") if isinstance(offers[0], dict) else None)
        if price is None or price <= 0:
            log.warning("uniqlo.pdp.no_price", url=url, name=name)
            return None

        image = prod.get("image")
        if isinstance(image, list) and image:
            image_url = image[0]
        elif isinstance(image, str):
            image_url = image
        else:
            log.warning("uniqlo.pdp.no_image", url=url, name=name)
            return None

        mpn = prod.get("mpn") or url.rsplit("/", 2)[-2]
        color = normalize_color(prod.get("color"))
        material = simplify_material(prod.get("material"))
        subcategory = infer_subcategory(name, category)

        return Item(
            id=f"uniqlo_{mpn}",
            brand=self.brand,
            name=name,
            description=description.strip(),
            price_inr=price,
            image_url=str(image_url),
            product_url=str(prod.get("url") or url),
            category=category,
            subcategory=subcategory,
            color=color,
            material=material,
        )
