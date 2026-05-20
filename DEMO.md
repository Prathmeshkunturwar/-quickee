# Demo Recording Script

Five-minute screen recording showing the system end-to-end. Record with OBS, Windows Game Bar (`Win + G`), or any screen recorder.

> **Two paths**:
> - **Path A — Streamlit UI (recommended for the recording).** Looks polished, easy to drive.
> - **Path B — Swagger UI / `curl`.** Lower-level; only useful if you want to demo the raw JSON API.

The two demos call the same FastAPI backend; the Streamlit UI just renders the response prettily.

## Pre-flight (do BEFORE recording)

```powershell
cd D:\quickee
uv sync                                       # ensure deps installed
uv run python scripts/build_catalog.py        # confirm "OK 68 tops + 102 bottoms"
uv run python scripts/ingest.py               # confirm "[ok] ingested 170 items"

# (optional) clear the semantic-cache collection so the first call is a fresh run
Remove-Item D:\quickee\chroma_db\*cache* -Recurse -Force -ErrorAction SilentlyContinue
```

Open **two PowerShell windows** in `D:\quickee`:
- **Window A (backend)**: `uv run uvicorn quickee.api.main:app --port 8000`
- **Window B (frontend)**: `uv run streamlit run streamlit_app.py`

Streamlit opens at http://localhost:8501; FastAPI's Swagger is at http://localhost:8000/docs.

---

## Path A — Streamlit UI demo (recommended, ~4 min)

### Scene 1 — Project tour (30s)

Open `D:\quickee` in your editor. Tour the tree quickly:
- `README.md`, `ARCHITECTURE.md` (highlight the Mermaid flow diagram)
- `src/quickee/`: `scraper/`, `rag/`, `agent/`, `cache/`, `api/`
- `data/processed/catalog.json` — scroll to show real items

> *Voice-over*: "Backend that scrapes Uniqlo + Bewakoof, indexes 170 items in ChromaDB with rich metadata, and exposes an agentic LangGraph workflow via FastAPI. Streamlit UI calls that same endpoint."

### Scene 2 — Open the Streamlit UI (15s)

Open http://localhost:8501. Show the layout:
- Hero + "Style me" button
- Sidebar with sample prompts, budget toggle, "Show agent trace" toggle
- "Your brief" textarea

> *Voice-over*: "Same API the interview spec asked for — now with a usable face."

### Scene 3 — First prompt (60s)

Click the first sample prompt button in the sidebar (or type):
> *I have dark navy chinos. What t-shirt should I wear for a summer yacht party?*

Click **Style me**. While the spinner runs (~15s), **switch to Window A (uvicorn terminal)** so the viewer can see the agent trace lines stream:

```
agent.parse_intent.done    occasion='summer yacht party' slots=['top'] owned=['navy chinos']
agent.retrieve.slot        slot=top  n=5
agent.compose.done         picks=[('top', 'uniqlo_E483924-000')]
agent.validate             ok=True
cache.stored
```

Switch back to Streamlit. The result renders:
- Cache-miss badge ("Fresh agent run")
- One item card with image, brand, color, price, product link
- Stylist Note in a serif quote block
- Expand the **Agent trace** panel — point at the LangGraph nodes

> *Voice-over*: "Five explicit nodes — parse_intent, retrieve_slots, compose_outfit, validate, respond. Every step trace-logged. That's the agentic workflow."

### Scene 4 — Multi-slot prompt with budget (45s)

Click the second sample prompt:
> *Suggest a full smart-casual outfit for a wedding cocktail evening in Mumbai. I prefer earthy tones.*

Toggle **"Set a budget"** ON in the sidebar, enter `4500`. Click **Style me**.

Show the response: **two** item cards (top + bottom), color-themed, total under ₹4,500, stylist note that references "earthy tones" + fabric + occasion.

> *Voice-over*: "Two-slot recommendation. Budget enforced. The agent factored earthy tones into both the retrieval query and the stylist note."

### Scene 5 — Frugal mindset: semantic cache (45s)

Click **Style me** again on the **same prompt** (no changes). Response returns in **<1s**, badge says **"Cache hit — instant + no LLM tokens"**.

Now reword the prompt to:
> *What complete smart-casual look do you suggest for a Mumbai wedding cocktail night? Earthy palette please.*

Click **Style me**. Still a cache hit, still <1s.

> *Voice-over*: "Semantic cache. The wording changed completely but the intent is the same — cosine similarity above 0.93 catches it. Zero tokens spent on this request."

### Scene 6 — Wrap (15s)

Switch to your editor, open `ARCHITECTURE.md`, scroll to the Mermaid diagram.

> *Voice-over*: "Stack: Playwright scrapers reading structured JSON-LD and Next.js data, 170 items embedded with Gemini at 768 dims, ChromaDB pre-filtered ANN, LangGraph state machine with one retry edge, semantic cache, FastAPI."

---

## Path B — Swagger / curl backup demo (~2 min)

Use this only if Streamlit fails on camera, or if you specifically want to demo the raw JSON.

1. Open http://localhost:8000/docs in browser.
2. Expand `POST /api/v1/style-me` → **Try it out** → paste body → **Execute**.
3. Repeat for each prompt in the cheat-sheet.

---

## Test-prompt cheat-sheet

```text
1. "I have dark navy chinos. What t-shirt should I wear for a summer yacht party?"
2. "Suggest a full smart-casual outfit for a wedding cocktail evening in Mumbai. I prefer earthy tones." (budget 4500)
3. (repeat 2 verbatim — cache hit)
4. "What complete smart-casual look do you suggest for a Mumbai wedding cocktail night? Earthy palette please." (budget 4500 — semantic cache hit)
5. "Need a comfortable gym outfit for a hot Bangalore morning run." (budget 3000 — fresh agent, athletic items)
```

## If something fails on camera

- **Streamlit shows "Couldn't reach 127.0.0.1:8000"** → the FastAPI window isn't running. Start it in Window A and retry.
- **500 from API** → catalog might be stale. Stop the server, `Remove-Item .\chroma_db\ -Recurse -Force`, run `scripts/ingest.py`, restart server.
- **Gemini 429 quota** → free tier is 100 embed-requests/minute. Wait 60s.
- **Empty response with "no items found"** → broaden the prompt or remove the budget; the cosmos of catalog items couldn't satisfy the brief.
