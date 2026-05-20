"""Merge per-site raw scrape outputs into one validated catalog.

Pipeline contract:
  data/raw/<site>.json  --(this module)-->  data/processed/catalog.json

The processed catalog is the single source of truth that the ingest layer
embeds. Everything in it is a validated `Item`.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from quickee.models import Category, Item
from quickee.scraper.normalize import (
    derive_category_from_name,
    infer_subcategory,
    normalize_color,
)

log = structlog.get_logger()


def _maybe_correct_category(item: Item) -> Item:
    """Re-derive category from name when name strongly implies a different one."""
    src_cat_value = item.category if isinstance(item.category, str) else item.category.value
    src_cat = Category(src_cat_value)
    implied = derive_category_from_name(item.name)
    if implied is None or implied == src_cat:
        return item
    new_sub = infer_subcategory(item.name, implied)
    log.info("catalog.recategorized", id=item.id, name=item.name[:40], from_=src_cat.value, to=implied.value)
    return item.model_copy(update={"category": implied.value, "subcategory": new_sub.value})


def _maybe_renormalize_color(item: Item) -> Item:
    """Re-apply current color normalization rules.

    If the stored color isn't in our canonical set (e.g. 'aop', 'winter' from
    earlier looser rules) fall back to parsing the product name."""
    re_norm = normalize_color(item.color)
    if re_norm != "unknown":
        if re_norm == item.color:
            return item
        log.info("catalog.recolored", id=item.id, from_=item.color, to=re_norm)
        return item.model_copy(update={"color": re_norm})
    # Try the name as a fallback signal
    from_name = normalize_color(item.name)
    if from_name != "unknown" and from_name != item.color:
        log.info("catalog.recolored_from_name", id=item.id, name=item.name[:40],
                 from_=item.color, to=from_name)
        return item.model_copy(update={"color": from_name})
    return item


def load_raw_items(data_raw_dir: Path) -> list[Item]:
    """Read every *.json file in data/raw/, validate, dedupe by id, and correct categories."""
    files = sorted(p for p in data_raw_dir.glob("*.json") if not p.name.startswith("."))
    seen: dict[str, Item] = {}
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("catalog.bad_json", file=str(fp), err=str(e))
            continue
        if not isinstance(payload, list):
            log.warning("catalog.not_a_list", file=str(fp))
            continue
        ok = 0
        bad = 0
        for raw in payload:
            try:
                item = Item.model_validate(raw)
            except Exception as e:
                bad += 1
                log.warning("catalog.invalid_item", file=fp.name, err=str(e))
                continue
            if item.id in seen:
                continue  # first occurrence wins
            cleaned = _maybe_correct_category(item)
            cleaned = _maybe_renormalize_color(cleaned)
            seen[item.id] = cleaned
            ok += 1
        log.info("catalog.loaded", file=fp.name, ok=ok, bad=bad)
    return list(seen.values())


def save_catalog(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([it.model_dump() for it in items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("catalog.saved", path=str(path), n=len(items))


def summarize(items: list[Item]) -> dict:
    """Aggregate counts to verify we meet the ≥50 tops + ≥50 bottoms spec."""
    from collections import Counter

    by_cat = Counter(it.category for it in items)
    by_brand = Counter(it.brand for it in items)
    by_color = Counter(it.color for it in items)
    by_sub = Counter((it.category, it.subcategory) for it in items)
    return {
        "total": len(items),
        "by_category": dict(by_cat),
        "by_brand": dict(by_brand),
        "by_color": dict(by_color),
        "by_subcategory": {f"{c}:{s}": n for (c, s), n in sorted(by_sub.items())},
        "with_material": sum(1 for it in items if it.material),
        "desc_lengths": {
            "min": min((len(it.description) for it in items), default=0),
            "max": max((len(it.description) for it in items), default=0),
            "median": sorted(len(it.description) for it in items)[len(items) // 2] if items else 0,
        },
    }
