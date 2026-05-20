"""Compile the LangGraph state machine.

Flow:
  START -> parse_intent -> retrieve_slots -> compose_outfit -> validate
        (validate.ok ?) -> respond -> END
        (validate.fail, retry<1?) -> compose_outfit -> validate -> ...

This is the literal answer to "show me your agentic workflow" — every edge is
explicit, every node logs a trace entry. The Mermaid diagram in
ARCHITECTURE.md is a 1:1 of this code.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from quickee.agent.nodes import (
    after_validate_router,
    compose_outfit_node,
    parse_intent_node,
    respond_node,
    retrieve_slots_node,
    validate_node,
)
from quickee.agent.state import GraphState


def _compose_with_retry_bump(state: GraphState) -> dict:
    """Wrapper around compose_outfit_node that increments the retry counter
    when entered via the validate->compose loop-back edge."""
    out = compose_outfit_node(state)
    out["compose_retry"] = state.get("compose_retry", 0) + 1
    return out


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("parse_intent", parse_intent_node)
    g.add_node("retrieve_slots", retrieve_slots_node)
    g.add_node("compose_outfit", compose_outfit_node)
    g.add_node("compose_retry", _compose_with_retry_bump)
    g.add_node("validate", validate_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "parse_intent")
    g.add_edge("parse_intent", "retrieve_slots")
    g.add_edge("retrieve_slots", "compose_outfit")
    g.add_edge("compose_outfit", "validate")
    g.add_edge("compose_retry", "validate")
    g.add_conditional_edges(
        "validate",
        after_validate_router,
        {
            "respond": "respond",
            "compose_retry": "compose_retry",
        },
    )
    g.add_edge("respond", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Cached compiled graph — built once per process."""
    return build_graph()
