"""Catalog retriever: metadata pre-filter + semantic ANN search.

This is the heart of the RAG layer. The agent calls `retrieve()` once per
"slot" (top, bottom) with constraints like color/price/subcategory, and gets
back ranked Item candidates with similarity scores.

Why pre-filter + ANN (not the other way around):
  - If we did ANN first and filtered after, we might get 10 results back and
    then drop 9 because they're the wrong color/category — leaving us with 1
    real candidate.
  - Chroma's `where` filter runs against the HNSW index's payload BEFORE the
    distance computation, so the top-k we get is k actual matches that pass
    the filter. Same pattern as Qdrant/Pinecone production retrievers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from quickee.config import get_settings
from quickee.models import Category, Item, Subcategory
from quickee.rag.embeddings import Embedder
from quickee.rag.ingest import get_chroma_client

log = structlog.get_logger()


@dataclass
class Hit:
    """One retrieval result: the item + its similarity score (0..1)."""
    item: Item
    score: float

    @classmethod
    def from_chroma(cls, metadata: dict[str, Any], distance: float) -> "Hit":
        # Chroma returns cosine distance in [0..2]; similarity = 1 - distance/2 keeps it [0..1].
        # We just clamp & invert: similarity ≈ 1 - distance for cosine space.
        sim = max(0.0, 1.0 - float(distance))
        return cls(item=Item.model_validate(metadata), score=sim)


class Retriever:
    """Stateful wrapper around the Chroma collection + Embedder."""

    def __init__(self) -> None:
        s = get_settings()
        self._client = get_chroma_client()
        self._coll = self._client.get_collection(name=s.chroma_catalog_collection)
        self._embedder = Embedder()

    def _build_where(
        self,
        category: Category | None,
        subcategory: Subcategory | None,
        color: str | None,
        max_price_inr: float | None,
        excluded_ids: list[str] | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if category is not None:
            clauses.append({"category": category.value if hasattr(category, "value") else category})
        if subcategory is not None:
            clauses.append({"subcategory": subcategory.value if hasattr(subcategory, "value") else subcategory})
        if color:
            clauses.append({"color": color.lower()})
        if max_price_inr is not None:
            clauses.append({"price_inr": {"$lte": float(max_price_inr)}})
        if excluded_ids:
            clauses.append({"id": {"$nin": excluded_ids}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def retrieve(
        self,
        query: str,
        *,
        category: Category | None = None,
        subcategory: Subcategory | None = None,
        color: str | None = None,
        max_price_inr: float | None = None,
        excluded_ids: list[str] | None = None,
        k: int = 5,
    ) -> list[Hit]:
        """Return top-k items matching the semantic query under the filters."""
        vec = self._embedder.embed_query(query)
        where = self._build_where(category, subcategory, color, max_price_inr, excluded_ids)
        log.info("rag.query", q=query[:60], k=k, where=where)
        result = self._coll.query(
            query_embeddings=[vec],
            n_results=k,
            where=where,
        )
        hits: list[Hit] = []
        if not result["ids"] or not result["ids"][0]:
            return hits
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            try:
                hits.append(Hit.from_chroma(meta, dist))
            except Exception as e:
                log.warning("rag.bad_hit", err=str(e), meta=meta)
        return hits
