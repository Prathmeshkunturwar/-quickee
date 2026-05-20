"""Embed the processed catalog and push to ChromaDB with rich metadata.

Why the metadata is rich: ChromaDB supports pre-filtering with a `where`
clause that runs BEFORE the ANN search. So storing price/category/color/etc.
in metadata lets the retriever say "only navy tops under 2000 INR, ranked by
semantic similarity to this query" in one call — no two-stage filtering.
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from quickee.config import get_settings
from quickee.models import Item
from quickee.rag.embeddings import Embedder

log = structlog.get_logger()


def get_chroma_client() -> chromadb.PersistentClient:
    s = get_settings()
    s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(s.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_catalog_collection(client: chromadb.PersistentClient):
    s = get_settings()
    return client.get_or_create_collection(
        name=s.chroma_catalog_collection,
        metadata={
            "embed_model": s.gemini_embed_model,
            "embed_dim": s.gemini_embed_dim,
            "hnsw:space": "cosine",
        },
    )


def ingest_catalog(catalog_path: Path, *, reset: bool = True) -> int:
    """Read processed catalog, embed every item, upsert into Chroma.

    Returns the count of items ingested.
    """
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = [Item.model_validate(p) for p in payload]
    if not items:
        log.warning("ingest.empty_catalog", path=str(catalog_path))
        return 0

    s = get_settings()
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(s.chroma_catalog_collection)
            log.info("ingest.collection_reset", name=s.chroma_catalog_collection)
        except Exception:
            pass

    coll = get_or_create_catalog_collection(client)

    embedder = Embedder()
    texts = [it.to_embedding_text() for it in items]
    log.info("ingest.embedding", n=len(texts), model=s.gemini_embed_model, dim=s.gemini_embed_dim)
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == len(items), "embedding count mismatch"

    ids = [it.id for it in items]
    metadatas = [it.to_chroma_metadata() for it in items]
    coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
    log.info("ingest.added", n=len(items), collection=s.chroma_catalog_collection)
    return len(items)
