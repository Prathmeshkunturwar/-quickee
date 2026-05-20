"""One-shot sanity check that the Gemini key works for chat AND embeddings.

Run with:  uv run python scripts/smoke_gemini.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # noqa: E402

from quickee.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    print(f"[ok] loaded config; chat={s.gemini_chat_model} embed={s.gemini_embed_model} target_dim={s.gemini_embed_dim}")

    chat = ChatGoogleGenerativeAI(
        model=s.gemini_chat_model,
        google_api_key=s.gemini_api_key,
        temperature=0,
    )
    resp = chat.invoke("In <=10 words, what is a navy chino?")
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    print(f"[ok] chat reply: {text!r}")

    embedder = GoogleGenerativeAIEmbeddings(
        model=s.gemini_embed_model,
        google_api_key=s.gemini_api_key,
        output_dimensionality=s.gemini_embed_dim,
    )
    vec = embedder.embed_query("navy chinos summer outfit")
    print(f"[ok] embedding dim: {len(vec)} (target {s.gemini_embed_dim})")
    assert len(vec) == s.gemini_embed_dim, f"unexpected embedding dim {len(vec)}"
    print("[ok] Gemini key + chat + embeddings all working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
