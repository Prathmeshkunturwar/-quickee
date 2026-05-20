"""FastAPI surface: POST /api/v1/style-me, plus /health and /docs.

Lifecycle:
  startup -> pre-warm the LangGraph compile + ChromaDB clients + Embedder so
             the first request isn't a cold-start.
  request -> 1. lookup in SemanticCache; if hit, return it directly.
             2. otherwise run the agent graph.
             3. cache the response for future near-duplicate prompts.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from quickee.agent.graph import get_graph
from quickee.agent.state import GraphState
from quickee.cache.semantic_cache import get_cache
from quickee.config import get_settings
from quickee.models import AgentStep, RecommendedItem, StyleRequest, StyleResponse
from quickee.rag.retriever import Retriever

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm so the first /style-me request is fast."""
    s = get_settings()
    logging.basicConfig(level=s.log_level.upper())
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, s.log_level.upper(), logging.INFO)
        ),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )
    log.info("startup.prewarm")
    # Force compile + connect
    _ = get_graph()
    _ = Retriever()
    _ = get_cache()
    log.info("startup.ready")
    yield
    log.info("shutdown")


app = FastAPI(
    title="Quickeee Luxury Stylist Concierge",
    version="0.1.0",
    description=(
        "Agentic fashion concierge. POST a free-text style brief, get a complete "
        "outfit recommendation grounded in scraped real-time inventory."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


def _build_response_from_state(final: GraphState, cache_hit: bool = False) -> StyleResponse:
    chosen = final.get("chosen_items") or []
    total = sum(it.price_inr for it in chosen) if chosen else 0.0
    note = final.get("stylist_note") or "We could not assemble a recommendation for this brief."
    trace = [
        AgentStep(node=t["node"], summary=t["summary"], duration_ms=t["duration_ms"])
        for t in (final.get("trace") or [])
    ]
    return StyleResponse(
        items=chosen,
        total_price_inr=total,
        stylist_note=note,
        cache_hit=cache_hit,
        agent_trace=trace,
    )


@app.post("/api/v1/style-me", response_model=StyleResponse)
def style_me(req: StyleRequest) -> StyleResponse:
    log.info("api.style_me", prompt=req.prompt[:80], budget=req.max_budget_inr)
    cache = get_cache()

    # 1) Semantic cache
    cached = cache.get(req.prompt)
    if cached is not None:
        # Re-validate against our schema (defensive) and stamp cache_hit=True.
        try:
            resp = StyleResponse.model_validate(cached)
            resp = resp.model_copy(update={"cache_hit": True})
            return resp
        except Exception as e:
            log.warning("api.cache_payload_invalid", err=str(e))

    # 2) Run the agent graph
    initial: GraphState = {
        "user_prompt": req.prompt,
        "max_budget_inr": req.max_budget_inr,
        "trace": [],
        "candidates": {},
        "compose_retry": 0,
        "cache_hit": False,
    }
    graph = get_graph()
    try:
        final: GraphState = graph.invoke(initial)  # type: ignore[assignment]
    except Exception as e:
        log.exception("api.agent.crashed")
        raise HTTPException(status_code=500, detail=f"agent failed: {e}") from e

    resp = _build_response_from_state(final, cache_hit=False)
    if not resp.items:
        # Don't cache empty responses
        raise HTTPException(status_code=502, detail="no items could be recommended for this brief")

    # 3) Store in cache
    try:
        cache.put(req.prompt, resp.model_dump())
    except Exception as e:
        log.warning("api.cache_put_failed", err=str(e))

    return resp
