"""Canonical data models used across scraping, RAG, agent, and API layers.

One source of truth for what an Item looks like. If the schema changes here,
every layer that uses it gets a clean type error — that's the point of pydantic.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Category(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


class Subcategory(str, Enum):
    TSHIRT = "tshirt"
    SHIRT = "shirt"
    POLO = "polo"
    SWEATER = "sweater"
    PANTS = "pants"
    SHORTS = "shorts"
    JEANS = "jeans"
    CHINOS = "chinos"


class Item(BaseModel):
    """One catalog item, normalized across all scraped stores."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(..., description="Stable unique id: <brand>_<store_sku>")
    brand: str
    name: str
    description: str
    price_inr: float = Field(..., ge=0)
    image_url: str
    product_url: str
    category: Category
    subcategory: Subcategory
    color: str = Field(..., description="Normalized color tag, e.g. 'navy', 'white'")
    material: str | None = None

    def to_chroma_metadata(self) -> dict:
        """Flat dict for ChromaDB metadata storage (no nested types)."""
        return {
            "id": self.id,
            "brand": self.brand,
            "name": self.name,
            "price_inr": float(self.price_inr),
            "image_url": self.image_url,
            "product_url": self.product_url,
            "category": self.category if isinstance(self.category, str) else self.category.value,
            "subcategory": self.subcategory if isinstance(self.subcategory, str) else self.subcategory.value,
            "color": self.color,
            "material": self.material or "",
        }

    def to_embedding_text(self) -> str:
        """Concatenated text the embedding model sees."""
        parts = [
            self.name,
            f"Category: {self.category}.",
            f"Subcategory: {self.subcategory}.",
            f"Color: {self.color}.",
        ]
        if self.material:
            parts.append(f"Material: {self.material}.")
        parts.append(self.description)
        return " ".join(parts)


# --- API contract ---


class StyleRequest(BaseModel):
    prompt: str = Field(..., min_length=4, max_length=600)
    max_budget_inr: float | None = Field(None, ge=0, description="Optional total budget cap")


class RecommendedItem(BaseModel):
    id: str
    brand: str
    name: str
    price_inr: float
    image_url: str
    product_url: str
    slot: Literal["top", "bottom"]
    color: str


class AgentStep(BaseModel):
    node: str
    summary: str
    duration_ms: int


class StyleResponse(BaseModel):
    items: list[RecommendedItem]
    total_price_inr: float
    stylist_note: str
    cache_hit: bool = False
    agent_trace: list[AgentStep] = Field(default_factory=list)
