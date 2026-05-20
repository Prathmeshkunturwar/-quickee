"""Gemini embedding wrapper — true batch via google.genai SDK.

Why google.genai directly (not langchain-google-genai's wrapper):
  - LangChain's `embed_documents` calls the API once per text, so 170 items =
    170 requests. Free tier caps at 100 req/min, so we'd 429 on the second
    batch.
  - google.genai's `client.models.embed_content(contents=[t1, t2, ...])`
    sends the entire batch as ONE request. 170 items / 50-per-batch = 4
    requests total. Stays well under quotas and is faster (less network RTT).

Tenacity wraps both single + batch with generous backoff for transient 429s.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import cast

import structlog
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from quickee.config import get_settings

log = structlog.get_logger()


class Embedder:
    """Thin wrapper around Gemini embeddings with TRUE batch + retry."""

    def __init__(self, batch_size: int = 20, inter_batch_pause: float = 8.0):
        """Defaults tuned for Gemini's free-tier embed quota (100 RPM / 30K TPM).
        Smaller batches + ~8s pacing keeps us under both ceilings even if the
        catalog grows."""
        s = get_settings()
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.gemini_embed_model
        self._config = types.EmbedContentConfig(output_dimensionality=s.gemini_embed_dim)
        self.dim = s.gemini_embed_dim
        self.batch_size = batch_size
        self._pause = inter_batch_pause

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2, max=40),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embed_query(self, text: str) -> list[float]:
        """Single-text embedding (used for user prompts + cache lookups)."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._config,
        )
        return cast(list[float], list(result.embeddings[0].values))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2, max=40),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=batch,  # type: ignore[arg-type]  # SDK accepts list of strings
            config=self._config,
        )
        return [list(e.values) for e in result.embeddings]

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Batch-embed a corpus. Returns one vector per input, in order."""
        items = list(texts)
        out: list[list[float]] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            vecs = self._embed_batch(batch)
            out.extend(vecs)
            log.info("embed.batch", from_=i, to=i + len(batch), of=len(items))
            if i + self.batch_size < len(items):
                time.sleep(self._pause)
        return out
