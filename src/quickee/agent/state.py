"""LangGraph state + the structured-output schemas Gemini fills in.

Why a TypedDict for state (vs Pydantic): LangGraph reducers operate on plain
dicts; using BaseModel everywhere would force constant `.model_dump()` calls.
TypedDict is the idiomatic LangGraph choice. We still use Pydantic for the
LLM's structured outputs (parse_intent, compose_outfit) because we want
schema-validated JSON coming back from Gemini.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from quickee.models import RecommendedItem
from quickee.rag.retriever import Hit


# --- Structured outputs Gemini produces (validated by .with_structured_output) ---


class OwnedItem(BaseModel):
    """Something the user explicitly says they already have/own."""
    slot: Literal["top", "bottom"]
    color: str = Field(..., description="Lowercased canonical color, e.g. 'navy'")
    description: str = Field(..., description="Short phrase, e.g. 'navy chinos'")


class ParsedIntent(BaseModel):
    """Everything the LLM extracts from the user's free-text prompt."""
    occasion: str = Field(..., description="Single phrase, e.g. 'summer yacht party', 'office', 'wedding'")
    season: str | None = Field(None, description="'summer' | 'winter' | 'monsoon' | None if unspecified")
    owned_items: list[OwnedItem] = Field(default_factory=list)
    slots_to_recommend: list[Literal["top", "bottom"]] = Field(
        ...,
        description=(
            "Which slots the user needs us to recommend. "
            "If user mentions owning a top, slots_to_recommend should be ['bottom']. "
            "If user mentions owning a bottom, ['top']. "
            "If user mentions neither, ['top', 'bottom'] for a full outfit."
        ),
    )
    style_keywords: list[str] = Field(
        default_factory=list,
        description="Extra descriptive keywords to bias retrieval (e.g. 'breezy', 'linen', 'smart-casual')",
    )
    color_hints: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-slot color hint, e.g. {'top': 'white'} when user implies pairing logic",
    )
    subcategory_hints: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-slot subcategory hint when the user names a specific garment. "
            "Allowed top values: tshirt, shirt, polo, sweater, hoodie. "
            "Allowed bottom values: pants, shorts, jeans, chinos, jogger. "
            "Examples: 'formal shirt and trousers' -> {'top':'shirt','bottom':'pants'}. "
            "'I want a polo' -> {'top':'polo'}. "
            "OMIT when the user only says generic words like 'top' or 'outfit'."
        ),
    )
    notes_for_stylist: str = Field(
        default="",
        description="Anything else the LLM thought relevant for the human-readable stylist note later",
    )


class OutfitChoice(BaseModel):
    """One slot's chosen catalog item (by id) inside the compose step."""
    slot: Literal["top", "bottom"]
    item_id: str = Field(..., description="MUST match exactly one of the id's shown in candidates")
    why: str = Field(..., description="One sentence on why this item fits the brief")


class ComposedOutfit(BaseModel):
    """The compose step's full output: chosen items + the user-facing note."""
    choices: list[OutfitChoice]
    stylist_note: str = Field(..., description="2-3 sentences, luxurious tone, addresses occasion + pairing")


# --- LangGraph state ---


def _merge_traces(left: list, right: list) -> list:
    """Reducer: concatenate trace entries from each node."""
    return (left or []) + (right or [])


def _merge_candidates(left: dict, right: dict) -> dict:
    """Reducer: shallow-merge slot -> hits dicts."""
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    # inputs
    user_prompt: str
    max_budget_inr: float | None

    # parse_intent output
    intent: ParsedIntent

    # retrieve_slots output  (slot -> list of Hit)
    candidates: Annotated[dict[str, list[Hit]], _merge_candidates]

    # compose_outfit output
    composed: ComposedOutfit
    chosen_items: list[RecommendedItem]
    stylist_note: str

    # validate output
    validated: bool
    validation_errors: list[str]
    compose_retry: int  # bounded retry counter

    # observability
    trace: Annotated[list[dict], _merge_traces]
    cache_hit: bool

    # fatal error short-circuit
    error: str | None
