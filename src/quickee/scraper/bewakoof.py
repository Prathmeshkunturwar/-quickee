"""Bewakoof scraper.

PLP discovery: anchors matching /p/<slug>
PDP extraction: __NEXT_DATA__ -> props.pageProps.productDetails has the
complete structured payload (name, price, mrp, description list, colors[],
images.<size>, product_attributes, canonical_url).
"""
from __future__ import annotations

from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext, Page

from quickee.models import Category, Item
from quickee.scraper.base import (
    BaseScraper,
    autoscroll,
    extract_next_data,
    goto_with_retries,
    polite_sleep,
)
from quickee.scraper.normalize import (
    infer_subcategory,
    normalize_color,
    parse_price,
    simplify_material,
)


def _strip_html(s: str) -> str:
    if not s:
        return ""
    return BeautifulSoup(s, "html.parser").get_text(separator=" ", strip=True)

log = structlog.get_logger()

BASE = "https://www.bewakoof.com"

# (PLP URL, broad Category)
CATEGORIES: list[tuple[str, Category]] = [
    ("https://www.bewakoof.com/men-t-shirts", Category.TOP),
    ("https://www.bewakoof.com/men-joggers", Category.BOTTOM),
    ("https://www.bewakoof.com/men-shorts", Category.BOTTOM),
]


class BewakoofScraper(BaseScraper):
    brand = "Bewakoof"

    async def scrape(self, ctx: BrowserContext, limit_per_category: int | None = None) -> list[Item]:
        items: list[Item] = []
        for plp_url, category in CATEGORIES:
            log.info("bewakoof.plp.start", url=plp_url, category=category.value)
            page = await ctx.new_page()
            try:
                urls = await self._collect_product_urls(page, plp_url)
            finally:
                await page.close()
            log.info("bewakoof.plp.done", url=plp_url, found=len(urls))
            if limit_per_category:
                urls = urls[:limit_per_category]
            for i, url in enumerate(urls):
                page = await ctx.new_page()
                try:
                    item = await self._extract_item(page, url, category)
                    if item:
                        items.append(item)
                        log.info(
                            "bewakoof.pdp.ok",
                            i=i + 1,
                            n=len(urls),
                            name=item.name[:40],
                            color=item.color,
                            price=item.price_inr,
                        )
                except Exception as e:
                    log.warning("bewakoof.pdp.err", url=url, err=str(e))
                finally:
                    await page.close()
                await polite_sleep()
        return items

    async def _collect_product_urls(self, page: Page, plp_url: str) -> list[str]:
        await goto_with_retries(page, plp_url)
        await page.wait_for_timeout(2500)
        await autoscroll(page, passes=4, pause_ms=700)
        hrefs = await page.locator('a[href*="/p/"]').evaluate_all(
            "els => els.map(e => e.getAttribute('href'))"
        )
        seen: set[str] = set()
        ordered: list[str] = []
        for href in hrefs:
            if not href:
                continue
            absolute = urljoin(BASE, href.split("#", 1)[0].split("?", 1)[0])
            if absolute in seen:
                continue
            seen.add(absolute)
            ordered.append(absolute)
        return ordered

    async def _extract_item(self, page: Page, url: str, category: Category) -> Item | None:
        await goto_with_retries(page, url)
        await page.wait_for_timeout(1500)
        data = await extract_next_data(page)
        if not data:
            log.warning("bewakoof.pdp.no_next_data", url=url)
            return None

        pd = (
            data.get("props", {})
            .get("pageProps", {})
            .get("productDetails")
        )
        if not isinstance(pd, dict):
            log.warning("bewakoof.pdp.no_product_details", url=url)
            return None

        name = pd.get("name")
        if not name:
            return None

        price = parse_price(pd.get("price"))
        if price is None or price <= 0:
            log.warning("bewakoof.pdp.no_price", url=url, name=name)
            return None

        # Description is a list of {head, line1, line2, ...} dicts with HTML in the line* values.
        desc_raw = pd.get("description")
        parts: list[str] = []
        if isinstance(desc_raw, list):
            for d in desc_raw:
                if isinstance(d, dict):
                    for k, v in d.items():
                        if (k.startswith("line") or k in ("content", "body", "text")) and isinstance(v, str):
                            parts.append(v)
                elif isinstance(d, str):
                    parts.append(d)
        elif isinstance(desc_raw, str):
            parts.append(desc_raw)
        description = _strip_html(" ".join(parts))

        # Image: prefer meta_image, else first images.<size>
        image_url = pd.get("meta_image")
        if not image_url:
            imgs = pd.get("images")
            if isinstance(imgs, dict) and imgs:
                # take first non-empty value
                for v in imgs.values():
                    if v:
                        image_url = v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else None)
                        break
        if not image_url:
            log.warning("bewakoof.pdp.no_image", url=url, name=name)
            return None

        # Color: pd['color'] is {'name': 'Brown 16', 'hexcode': '#...', 'parent_color_name': 'Ginger Root'}.
        # Use parent_color_name (cleaner: 'Ginger Root' -> 'unknown' fallback to name),
        # else color.name, else the product name itself (Bewakoof bakes color into names).
        color_raw = None
        c = pd.get("color")
        if isinstance(c, dict):
            color_raw = c.get("name") or c.get("display_name")
        if not color_raw or normalize_color(color_raw) == "unknown":
            color_raw = name  # 'Men's Brown ...' yields 'brown'
        color = normalize_color(color_raw)

        # Material: try attribute keys first; if empty, scan description prose for keywords.
        material_raw = None
        attrs = pd.get("product_attributes")
        if isinstance(attrs, dict):
            for k in ("fabric_composition", "composition", "fabric", "material"):
                v = attrs.get(k)
                if isinstance(v, dict):
                    v = v.get("value") or v.get("name")
                if v and str(v).strip():
                    material_raw = str(v)
                    break
        if not material_raw and description:
            material_raw = description  # simplify_material does substring matching
        material = simplify_material(material_raw)

        subcategory = infer_subcategory(name, category)
        product_id = pd.get("id") or url.rsplit("/", 1)[-1]
        canonical_url = pd.get("canonical_url") or url

        return Item(
            id=f"bewakoof_{product_id}",
            brand=self.brand,
            name=str(name),
            description=description,
            price_inr=price,
            image_url=str(image_url),
            product_url=str(canonical_url),
            category=category,
            subcategory=subcategory,
            color=color,
            material=material,
        )
