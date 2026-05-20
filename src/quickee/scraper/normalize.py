"""Rule-based normalization of raw scraped fields.

Why rule-based here (and LLM later in the ingest layer)? Because every item
goes through this — cheap, deterministic, and we don't want to burn tokens
on obvious cases like "Off White" → "white". The LLM is reserved for items
where these rules produce 'unknown'.
"""
from __future__ import annotations

import re

from quickee.models import Category, Subcategory


# Canonical color → list of phrases that map to it.
# Order matters: longer phrases first to handle "off white" before "white".
_COLOR_MAP: list[tuple[str, list[str]]] = [
    ("white", ["off white", "off-white", "ecru", "ivory", "cream", "snow white", "natural"]),
    ("black", ["off black", "jet black"]),
    ("navy", ["navy blue", "dark navy", "midnight", "navy"]),
    ("gray", ["heather gray", "heather grey", "charcoal", "smoke gray", "smoke grey", "graphite", "grey"]),
    ("blue", ["denim blue", "sky blue", "light blue", "powder blue", "royal blue", "cobalt"]),
    ("olive", ["olive green", "army green", "military green"]),
    ("beige", ["sand beige", "tan beige"]),
    ("brown", ["dark brown", "tobacco", "espresso", "coffee", "mocha"]),
    ("green", ["forest green", "bottle green", "sage", "mint"]),
    ("red", ["maroon", "burgundy", "wine", "crimson", "scarlet"]),
    ("pink", ["dusty pink", "blush", "rose"]),
    ("yellow", ["mustard", "lemon"]),
    ("orange", ["rust", "coral", "tangerine"]),
    ("purple", ["lavender", "lilac", "violet"]),
    ("khaki", ["dark khaki"]),
    ("white", ["white"]),
    ("black", ["black"]),
    ("blue", ["blue", "indigo"]),
    ("gray", ["gray"]),
    ("green", ["green"]),
    ("red", ["red"]),
    ("pink", ["pink"]),
    ("yellow", ["yellow"]),
    ("orange", ["orange"]),
    ("purple", ["purple"]),
    ("brown", ["brown"]),
    ("beige", ["beige", "tan"]),
    ("olive", ["olive"]),
    ("khaki", ["khaki"]),
]


_KNOWN_COLORS = {
    "white", "black", "navy", "gray", "blue", "olive", "beige", "brown",
    "green", "red", "pink", "yellow", "orange", "purple", "khaki",
}


def normalize_color(raw: str | None) -> str:
    """Map a free-text color label to a canonical 1-word color tag.

    Returns 'unknown' for unrecognized input so callers can fall back to
    other signals (e.g. extracting color from the product name).
    """
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    for canon, phrases in _COLOR_MAP:
        for ph in phrases:
            if ph in s:
                return canon
    # Strict last resort: first word must itself be a known color.
    first = s.split()[0] if s else ""
    return first if first in _KNOWN_COLORS else "unknown"


_MATERIAL_KEYWORDS = [
    ("cotton", ["cotton"]),
    ("linen", ["linen"]),
    ("wool", ["wool"]),
    ("polyester", ["polyester", "polyamide"]),
    ("denim", ["denim"]),
    ("nylon", ["nylon"]),
    ("silk", ["silk"]),
    ("rayon", ["rayon", "viscose"]),
    ("blend", ["blend", "elastane", "spandex", "lycra"]),
]


def simplify_material(raw: str | None) -> str | None:
    """E.g. '61% Cotton, 33% Elastomultiester, 6% Elastane' -> 'cotton blend'."""
    if not raw:
        return None
    s = raw.strip().lower()
    found: list[str] = []
    for canon, keys in _MATERIAL_KEYWORDS:
        if any(k in s for k in keys):
            if canon not in found:
                found.append(canon)
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    primary = found[0]
    if "blend" in found:
        return f"{primary} blend"
    return f"{primary} blend"


# Subcategory inference from product name (case-insensitive substring rules).
# Order matters: "t-shirt" before "shirt".
_TOP_RULES: list[tuple[Subcategory, list[str]]] = [
    (Subcategory.POLO, ["polo"]),
    (Subcategory.HOODIE, ["hoodie", "hooded", "pullover hood"]),
    (Subcategory.SWEATER, ["sweater", "sweatshirt", "knit", "cardigan", "jumper"]),
    (Subcategory.SHIRT, ["dress shirt", "casual shirt", "linen shirt", "oxford", "flannel"]),
    (Subcategory.TSHIRT, ["t-shirt", "tshirt", "tee", "henley"]),
]
_BOTTOM_RULES: list[tuple[Subcategory, list[str]]] = [
    (Subcategory.JEANS, ["jeans", "denim"]),
    (Subcategory.SHORTS, ["shorts"]),
    (Subcategory.JOGGER, ["jogger", "joggers", "track pant", "trackpants", "pyjama"]),
    (Subcategory.CHINOS, ["chinos", "chino", "smart pants"]),
    (Subcategory.PANTS, ["pants", "trousers", "slacks"]),
]


def infer_subcategory(name: str, category: Category) -> Subcategory:
    """Best-guess subcategory from product name; safe defaults if no match."""
    n = name.lower()
    rules = _TOP_RULES if category == Category.TOP else _BOTTOM_RULES
    for sub, keys in rules:
        if any(k in n for k in keys):
            return sub
    return Subcategory.TSHIRT if category == Category.TOP else Subcategory.PANTS


def derive_category_from_name(name: str) -> Category | None:
    """Return the unambiguous Category implied by the product name, else None.

    Used to correct mis-categorization that happens when a PLP cross-promotes
    items (e.g. Uniqlo's /men/tops listing some sweat shorts/pants).
    Conservative: only return a category when the name has an unambiguous
    bottom- or top-marker. Otherwise let the PLP source category stand.
    """
    n = name.lower()
    bottom_markers = ("pants", "shorts", "trousers", "jeans", "chinos", "joggers", "track pant", "pyjama")
    top_markers = ("t-shirt", "tshirt", " tee", "polo", "hoodie", "sweater", "sweatshirt", "shirt", "tank top", "henley")
    has_bottom = any(m in n for m in bottom_markers)
    has_top = any(m in n for m in top_markers)
    if has_bottom and not has_top:
        return Category.BOTTOM
    if has_top and not has_bottom:
        return Category.TOP
    return None  # ambiguous — keep source category


# Strip currency symbols and commas, return float.
_PRICE_RE = re.compile(r"[\d.]+")


def parse_price(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).replace(",", "")
    m = _PRICE_RE.search(s)
    return float(m.group()) if m else None
