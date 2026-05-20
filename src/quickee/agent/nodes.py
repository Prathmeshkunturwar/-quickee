"""LangGraph node functions — each is a pure (state) -> partial-state-update.

Design notes:
- Every node returns ONLY the fields it touched. LangGraph merges into state.
- Every node also appends a trace entry — the demo screen recording reads
  these to show the agent's thought process step by step.
- The compose node uses Gemini's structured output to get back a validated
  ComposedOutfit (no JSON-parsing-the-LLM hacks).
"""
from __future__ import annotations

import json
import time
from typing import Any

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI

from quickee.agent.prompts import (
    COMPOSE_SYSTEM,
    INTENT_SYSTEM,
    compose_user_prompt,
    intent_user_prompt,
)
from quickee.agent.state import (
    ComposedOutfit,
    GraphState,
    ParsedIntent,
)
from quickee.config import get_settings
from quickee.models import Category, RecommendedItem
from quickee.rag.retriever import Hit, Retriever

log = structlog.get_logger()


# --- shared LLM client (one instance per process) ---

_chat: ChatGoogleGenerativeAI | None = None


def _get_chat(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    global _chat
    if _chat is None:
        s = get_settings()
        _chat = ChatGoogleGenerativeAI(
            model=s.gemini_chat_model,
            google_api_key=s.gemini_api_key,
            temperature=temperature,
        )
    return _chat


_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _trace(node: str, summary: str, t0: float) -> dict[str, Any]:
    return {
        "node": node,
        "summary": summary,
        "duration_ms": int((time.time() - t0) * 1000),
    }


# --- nodes ---


def parse_intent_node(state: GraphState) -> dict[str, Any]:
    t0 = time.time()
    user_text = state["user_prompt"]
    log.info("agent.parse_intent.start", prompt=user_text[:80])

    llm = _get_chat()
    structured = llm.with_structured_output(ParsedIntent)
    intent: ParsedIntent = structured.invoke(
        [
            ("system", INTENT_SYSTEM),
            ("user", intent_user_prompt(user_text)),
        ]
    )  # type: ignore[assignment]

    log.info(
        "agent.parse_intent.done",
        occasion=intent.occasion,
        slots=intent.slots_to_recommend,
        owned=[o.description for o in intent.owned_items],
    )
    return {
        "intent": intent,
        "trace": [_trace("parse_intent", f"occasion={intent.occasion!r} slots={intent.slots_to_recommend}", t0)],
    }


def _build_slot_query(slot: str, intent: ParsedIntent) -> str:
    """Compose a semantic query for one slot from the parsed intent."""
    parts: list[str] = []
    parts.append(f"a {slot}")
    if intent.occasion:
        parts.append(f"for {intent.occasion}")
    if intent.season:
        parts.append(f"in {intent.season}")
    # Mention paired owned items so semantic search nudges toward matching styles
    paired = [o for o in intent.owned_items if o.slot != slot]
    if paired:
        descs = ", ".join(f"{p.color} {p.description}" for p in paired)
        parts.append(f"to pair with {descs}")
    if intent.style_keywords:
        parts.append("style: " + ", ".join(intent.style_keywords))
    return " ".join(parts)


def retrieve_slots_node(state: GraphState) -> dict[str, Any]:
    t0 = time.time()
    intent: ParsedIntent = state["intent"]
    s = get_settings()  # noqa: F841
    retriever = _get_retriever()
    candidates: dict[str, list[Hit]] = {}

    # Per-slot price ceiling: if a total budget is set, allocate generously
    # (e.g. budget 5000 -> each slot can cost up to 4000 to leave room for the other)
    budget = state.get("max_budget_inr")
    per_slot_ceiling = None
    if budget is not None and len(intent.slots_to_recommend) > 0:
        per_slot_ceiling = float(budget) * 0.85  # loose; final budget check is in validate

    for slot in intent.slots_to_recommend:
        cat = Category.TOP if slot == "top" else Category.BOTTOM
        color = intent.color_hints.get(slot)
        q = _build_slot_query(slot, intent)
        hits = retriever.retrieve(
            q,
            category=cat,
            color=color,
            max_price_inr=per_slot_ceiling,
            k=5,
        )
        # Fallback: if a color hint filter is too restrictive and yields <2 results, retry without color
        if color and len(hits) < 2:
            log.info("agent.retrieve.relax_color", slot=slot, color=color)
            hits = retriever.retrieve(q, category=cat, max_price_inr=per_slot_ceiling, k=5)
        candidates[slot] = hits
        log.info("agent.retrieve.slot", slot=slot, q=q, n=len(hits),
                 top=[(h.item.name[:30], h.item.color, round(h.score, 3)) for h in hits[:3]])

    summary = ", ".join(f"{s}={len(h)}" for s, h in candidates.items())
    return {
        "candidates": candidates,
        "trace": [_trace("retrieve_slots", f"hits: {summary}", t0)],
    }


def compose_outfit_node(state: GraphState) -> dict[str, Any]:
    t0 = time.time()
    intent: ParsedIntent = state["intent"]
    candidates: dict[str, list[Hit]] = state["candidates"]

    # Early-fail when retrieval yielded nothing for required slots — saves an
    # LLM call AND prevents the model from hallucinating item ids.
    empty_slots = [s for s in intent.slots_to_recommend if not candidates.get(s)]
    if empty_slots:
        log.warning("agent.compose.no_candidates", missing=empty_slots)
        from quickee.agent.state import ComposedOutfit as _CO
        return {
            "composed": _CO(choices=[], stylist_note=f"No catalog matches for slot(s): {empty_slots}"),
            "chosen_items": [],
            "stylist_note": (
                "We couldn't find inventory matching this brief in our current catalog. "
                "Try widening the budget or occasion, or simplify the description."
            ),
            "trace": [_trace("compose_outfit", f"no candidates for {empty_slots} — skipping LLM", t0)],
        }

    # Build the lightweight candidate list — only fields the LLM needs to pick.
    candidates_brief = {
        slot: [
            {
                "id": h.item.id,
                "name": h.item.name,
                "brand": h.item.brand,
                "color": h.item.color,
                "price_inr": h.item.price_inr,
            }
            for h in hits
        ]
        for slot, hits in candidates.items()
    }

    llm = _get_chat(temperature=0.4)  # warmer for the stylist note
    structured = llm.with_structured_output(ComposedOutfit)
    composed: ComposedOutfit = structured.invoke(
        [
            ("system", COMPOSE_SYSTEM),
            (
                "user",
                compose_user_prompt(
                    state["user_prompt"],
                    intent.model_dump_json(indent=2),
                    candidates_brief,
                    state.get("max_budget_inr"),
                ),
            ),
        ]
    )  # type: ignore[assignment]

    # Materialize chosen_items from the candidate Hits using ids returned.
    chosen: list[RecommendedItem] = []
    for choice in composed.choices:
        hits = candidates.get(choice.slot, [])
        matched = next((h for h in hits if h.item.id == choice.item_id), None)
        if matched is None:
            log.warning("agent.compose.id_mismatch", slot=choice.slot, id=choice.item_id)
            continue
        it = matched.item
        chosen.append(
            RecommendedItem(
                id=it.id,
                brand=it.brand,
                name=it.name,
                price_inr=it.price_inr,
                image_url=it.image_url,
                product_url=it.product_url,
                slot=choice.slot,  # type: ignore[arg-type]
                color=it.color,
            )
        )

    log.info("agent.compose.done", picks=[(c.slot, c.item_id) for c in composed.choices])
    return {
        "composed": composed,
        "chosen_items": chosen,
        "stylist_note": composed.stylist_note,
        "trace": [_trace("compose_outfit", f"picked {len(chosen)} items", t0)],
    }


def validate_node(state: GraphState) -> dict[str, Any]:
    t0 = time.time()
    errs: list[str] = []
    chosen = state.get("chosen_items", [])
    intent: ParsedIntent = state["intent"]

    if not chosen:
        errs.append("compose_outfit produced 0 valid picks")
    # Slot coverage
    chosen_slots = {it.slot for it in chosen}
    missing = [s for s in intent.slots_to_recommend if s not in chosen_slots]
    if missing:
        errs.append(f"missing slots in compose output: {missing}")

    # Budget check
    budget = state.get("max_budget_inr")
    total = sum(it.price_inr for it in chosen)
    if budget is not None and total > budget:
        errs.append(f"total price {total:.0f} exceeds budget {budget:.0f}")

    validated = len(errs) == 0
    log.info("agent.validate", ok=validated, total_inr=total, errors=errs)
    return {
        "validated": validated,
        "validation_errors": errs,
        "trace": [_trace("validate", "ok" if validated else f"failed: {errs}", t0)],
    }


def respond_node(state: GraphState) -> dict[str, Any]:
    t0 = time.time()
    return {"trace": [_trace("respond", "finalized response", t0)]}


# --- conditional edge router ---


def after_validate_router(state: GraphState) -> str:
    """Loop back to compose once if validation failed, else respond."""
    if state.get("validated"):
        return "respond"
    retries = state.get("compose_retry", 0)
    if retries >= 1:
        # Give up and respond with whatever we have (still better than 500ing).
        log.warning("agent.validate.giving_up", errs=state.get("validation_errors"))
        return "respond"
    return "compose_retry"
