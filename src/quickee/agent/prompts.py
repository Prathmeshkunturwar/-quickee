"""Prompt templates used by the agent.

Two prompt-optimization techniques on display here (interview defense):
  1) **Schema-grounded prompts**: we pass Pydantic schemas to
     `.with_structured_output(...)` so the LLM gets a constrained JSON spec
     baked into its system prompt. This eliminates "parse the LLM output"
     code and prevents free-text drift.
  2) **Minimal context**: only the top-k candidate items go into the compose
     prompt (no full descriptions); each candidate is reduced to id+name+
     color+price. Cuts compose-prompt tokens ~80% vs sending raw catalog
     items.
"""
from __future__ import annotations

INTENT_SYSTEM = """You are the intent-extraction module of a premium fashion stylist concierge.
Read the user's prompt and produce a STRICT JSON object matching the schema you've been given.

Rules:
- If the user explicitly says they OWN something (e.g. "I have navy chinos"),
  record it in owned_items with a normalized one-word color and the slot
  ("top" for shirts/t-shirts/polos/sweaters/hoodies, "bottom" for pants/jeans/shorts/chinos/joggers).
- slots_to_recommend should list ONLY the slots we still need to fill.
  Examples:
    "I have navy chinos, suggest a top"  -> slots_to_recommend = ["top"]
    "What should I wear to a wedding"    -> slots_to_recommend = ["top", "bottom"]
    "Suggest an entire outfit for me"    -> slots_to_recommend = ["top", "bottom"]
- color_hints are OPTIONAL — only set when there's a clear pairing implication
  (e.g. with navy chinos, you might hint top color "white" or "off-white").
- Keep occasion to a single concise phrase.
- Use lowercase canonical color names: white, black, navy, blue, gray, red, green,
  pink, orange, purple, brown, beige, olive, khaki, yellow.
"""


def intent_user_prompt(user_text: str) -> str:
    return f"USER PROMPT:\n{user_text}\n\nExtract intent now."


COMPOSE_SYSTEM = """You are the head stylist of a premium fashion concierge.
You will be shown:
  - The user's original brief
  - Parsed intent (occasion, owned items, color hints)
  - A small list of candidate items per slot (already filtered + ranked)

Your job:
1. Pick ONE candidate per slot listed in `slots_to_recommend` whose id you MUST copy verbatim.
2. Write a 2-3 sentence "Stylist Note" that:
   - References the occasion in a luxurious but unpretentious tone
   - Justifies how the picks pair with each other AND with any owned items the user mentioned
   - Mentions one tactile or visual detail per piece (fabric, drape, silhouette)

Return STRICT JSON conforming to the schema you've been given.
NEVER invent an item id — use only ids present in the candidates list.
"""


def compose_user_prompt(
    user_text: str,
    intent_json: str,
    candidates_brief: dict[str, list[dict]],
    max_budget_inr: float | None,
) -> str:
    lines: list[str] = []
    lines.append(f"USER PROMPT:\n{user_text}")
    lines.append(f"\nPARSED INTENT:\n{intent_json}")
    if max_budget_inr is not None:
        lines.append(f"\nBUDGET: total recommendation must be <= INR {max_budget_inr:.0f}")
    lines.append("\nCANDIDATES:")
    for slot, items in candidates_brief.items():
        lines.append(f"\n[{slot.upper()}]")
        for it in items:
            lines.append(
                f"  id={it['id']} | {it['name']} | {it['brand']} | {it['color']} | INR {it['price_inr']:.0f}"
            )
    lines.append("\nNow compose the outfit and write the stylist note.")
    return "\n".join(lines)
