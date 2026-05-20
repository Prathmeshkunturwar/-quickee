"""Quickeee Stylist Concierge — Streamlit front-end.

Calls the FastAPI backend at /api/v1/style-me. Renders recommendations as
image cards + a styled stylist note + a collapsible agent trace.

Run:
    uv run streamlit run streamlit_app.py
"""
from __future__ import annotations

import time

import httpx
import streamlit as st


# --- page setup ---------------------------------------------------------------

st.set_page_config(
    page_title="Quickeee Stylist Concierge",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle custom styling — luxury-leaning minimal palette
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
        .stylist-note {
            background: linear-gradient(180deg, #f8f6f1 0%, #f1ede4 100%);
            border-left: 3px solid #8a7a4d;
            padding: 1.2rem 1.5rem;
            border-radius: 4px;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.05rem;
            line-height: 1.6;
            color: #2a2a2a;
            font-style: italic;
        }
        .item-card {
            background: #ffffff;
            border: 1px solid #e8e3d8;
            border-radius: 6px;
            padding: 1rem;
            height: 100%;
        }
        .slot-label {
            display: inline-block;
            background: #1f1f1f;
            color: #fff;
            padding: 0.15rem 0.6rem;
            font-size: 0.7rem;
            letter-spacing: 0.06rem;
            text-transform: uppercase;
            border-radius: 2px;
            margin-bottom: 0.5rem;
        }
        .brand-line { color: #8a7a4d; font-size: 0.8rem; letter-spacing: 0.05rem; text-transform: uppercase; }
        .price-line { font-size: 1.3rem; font-weight: 600; color: #1f1f1f; margin-top: 0.4rem; }
        .cache-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            font-size: 0.72rem;
            letter-spacing: 0.05rem;
            text-transform: uppercase;
            border-radius: 12px;
            font-weight: 600;
        }
        .cache-hit { background: #e6f4ea; color: #1f6f37; border: 1px solid #b6d8c1; }
        .cache-miss { background: #f3eee0; color: #6f5b1f; border: 1px solid #d8cfb6; }
        .step-row { font-family: "JetBrains Mono", "Consolas", monospace; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Quickeee")
    st.caption("Luxury Stylist Concierge")
    st.divider()

    api_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
    st.caption("Run `uv run uvicorn quickee.api.main:app --port 8000` in another terminal.")

    st.divider()

    use_budget = st.toggle("Set a budget", value=False)
    budget = st.number_input(
        "Budget (INR)",
        min_value=500,
        max_value=20000,
        value=4500,
        step=500,
        disabled=not use_budget,
    )

    st.divider()
    st.markdown("**Try a sample prompt:**")
    samples = [
        "I have dark navy chinos. What t-shirt should I wear for a summer yacht party?",
        "Suggest a full smart-casual outfit for a wedding cocktail evening in Mumbai. I prefer earthy tones.",
        "Need a comfortable gym outfit for a hot Bangalore morning run.",
        "What jogger and tee combo works for a casual weekend brunch in Goa?",
        "I want a crisp white shirt and a versatile bottom for office Fridays.",
    ]
    for i, s in enumerate(samples):
        if st.button(s, key=f"sample_{i}", use_container_width=True):
            st.session_state["prompt_text"] = s

    st.divider()
    show_trace = st.toggle("Show agent trace", value=True)


# --- main column --------------------------------------------------------------

st.markdown("## Quickeee Stylist Concierge")
st.markdown(
    "<p style='color:#666; margin-top:-0.6rem;'>"
    "Tell us the occasion. We'll style the look — grounded in real-time inventory from Uniqlo &amp; Bewakoof."
    "</p>",
    unsafe_allow_html=True,
)

# Persistent prompt across reruns (so sample buttons work)
if "prompt_text" not in st.session_state:
    st.session_state["prompt_text"] = ""

prompt = st.text_area(
    "Your brief",
    value=st.session_state["prompt_text"],
    placeholder="e.g. I have dark navy chinos — what should I wear to a summer yacht party?",
    height=110,
    key="prompt_text",
)

col_a, col_b = st.columns([1, 5])
with col_a:
    submit = st.button("Style me", type="primary", use_container_width=True)


# --- backend call -------------------------------------------------------------

def call_backend(api_url: str, prompt: str, max_budget_inr: float | None) -> tuple[dict | None, str | None, float]:
    payload: dict = {"prompt": prompt}
    if max_budget_inr is not None:
        payload["max_budget_inr"] = float(max_budget_inr)
    t0 = time.time()
    try:
        r = httpx.post(f"{api_url.rstrip('/')}/api/v1/style-me", json=payload, timeout=60.0)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text}", elapsed
        return r.json(), None, elapsed
    except httpx.ConnectError:
        return None, f"Couldn't reach {api_url}. Is the FastAPI server running?", time.time() - t0
    except Exception as e:
        return None, f"Request failed: {e}", time.time() - t0


# --- render response ----------------------------------------------------------

def render_response(data: dict, elapsed: float) -> None:
    cache_hit = bool(data.get("cache_hit"))
    badge_class = "cache-hit" if cache_hit else "cache-miss"
    badge_text = "Cache hit — instant + no LLM tokens" if cache_hit else "Fresh agent run"
    st.markdown(
        f"<div style='margin-top:1rem;'>"
        f"<span class='cache-badge {badge_class}'>{badge_text}</span> "
        f"<span style='color:#888; font-size:0.85rem; margin-left:0.6rem;'>"
        f"served in {elapsed:.2f}s</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    items = data.get("items", [])
    if not items:
        st.warning("No items could be recommended for this brief — try widening the budget or simplifying the prompt.")
        return

    # Item cards
    st.markdown("### Recommended pieces")
    cols = st.columns(max(1, len(items)))
    for col, it in zip(cols, items):
        with col:
            st.markdown("<div class='item-card'>", unsafe_allow_html=True)
            try:
                st.image(it["image_url"], use_container_width=True)
            except Exception:
                st.caption(f"(image unavailable: {it['image_url']})")
            st.markdown(
                f"<div class='slot-label'>{it['slot']}</div>"
                f"<div class='brand-line'>{it['brand']} &middot; {it['color']}</div>"
                f"<div style='font-weight:500; margin-top:0.25rem; line-height:1.3;'>{it['name']}</div>"
                f"<div class='price-line'>INR {int(it['price_inr']):,}</div>"
                f"<a href='{it['product_url']}' target='_blank' style='font-size:0.85rem;'>View product &rarr;</a>"
                "</div>",
                unsafe_allow_html=True,
            )

    # Total + stylist note
    total = data.get("total_price_inr", 0)
    col_tot, col_note = st.columns([1, 2])
    with col_tot:
        st.metric("Outfit total", f"INR {int(total):,}")
    with col_note:
        st.markdown("**Stylist Note**")
        st.markdown(
            f"<div class='stylist-note'>{data.get('stylist_note', '').strip()}</div>",
            unsafe_allow_html=True,
        )

    # Agent trace
    if show_trace and data.get("agent_trace"):
        with st.expander("Agent trace (LangGraph nodes)", expanded=False):
            for step in data["agent_trace"]:
                st.markdown(
                    f"<div class='step-row'>"
                    f"<b>{step['node']:<18s}</b> &nbsp; "
                    f"<span style='color:#888;'>{step['duration_ms']:>5d} ms</span> &nbsp; "
                    f"<span>{step['summary']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# --- handle submit ------------------------------------------------------------

if submit:
    if not prompt or len(prompt.strip()) < 4:
        st.error("Please enter a brief of at least 4 characters.")
    else:
        with st.spinner("Curating your outfit…"):
            data, err, elapsed = call_backend(
                api_url,
                prompt.strip(),
                budget if use_budget else None,
            )
        if err:
            st.error(err)
        else:
            render_response(data, elapsed)
