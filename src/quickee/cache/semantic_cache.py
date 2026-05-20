"""Semantic prompt cache — the "frugal mindset" deliverable.

How it works:
  1) On every request, embed the user prompt.
  2) Query the cache collection in ChromaDB for the nearest stored prompt.
  3) If similarity >= SEMANTIC_CACHE_THRESHOLD, return the stored response —
     no LLM calls, no agent run, no embedding of catalog items. Free.
  4) On miss, the API handler stores the (prompt_vector, response_json) pair
     so the next near-duplicate is a hit.

Why a semantic cache vs exact-string cache:
  - "summer yacht party top with navy chinos" and "what should I wear with
    navy chinos to a yacht in summer" are the SAME query in intent. Exact
    match would miss; cosine ~0.94 catches it.

Interview talking points:
  - Lives in the SAME Chroma instance as the catalog → no extra infra.
  - Threshold is tunable per environment (we expose it via env var).
  - Stores the full final response so a hit returns in <50ms (just one vector
    lookup, no LLM, no compose).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from quickee.config import get_settings
from quickee.rag.embeddings import Embedder

log = structlog.get_logger()


class SemanticCache:
    def __init__(self) -> None:
        s = get_settings()
        s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(s.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._coll = self._client.get_or_create_collection(
            name=s.chroma_cache_collection,
            metadata={"hnsw:space": "cosine", "kind": "prompt_cache"},
        )
        self._embedder = Embedder()
        self._threshold = s.semantic_cache_threshold

    def get(self, prompt: str) -> dict[str, Any] | None:
        """Return cached response dict if a near-duplicate prompt is stored."""
        if self._coll.count() == 0:
            return None
        vec = self._embedder.embed_query(prompt)
        result = self._coll.query(query_embeddings=[vec], n_results=1)
        ids = result.get("ids", [[]])[0]
        if not ids:
            return None
        distance = result["distances"][0][0]
        sim = max(0.0, 1.0 - float(distance))
        if sim < self._threshold:
            log.info("cache.miss", sim=round(sim, 3), threshold=self._threshold)
            return None
        meta = result["metadatas"][0][0]
        raw_response = meta.get("response_json")
        if not raw_response:
            return None
        log.info("cache.hit", sim=round(sim, 3), cached_at=meta.get("cached_at"))
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            log.warning("cache.bad_payload")
            return None

    def put(self, prompt: str, response: dict[str, Any]) -> None:
        """Store the (prompt, response) pair. Vector is computed on the fly."""
        vec = self._embedder.embed_query(prompt)
        entry_id = f"q_{uuid.uuid4().hex[:12]}"
        self._coll.add(
            ids=[entry_id],
            embeddings=[vec],
            metadatas=[
                {
                    "prompt": prompt[:500],
                    "response_json": json.dumps(response, ensure_ascii=False),
                    "cached_at": int(time.time()),
                }
            ],
        )
        log.info("cache.stored", id=entry_id)


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
