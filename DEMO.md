# Demo Recording Script

Five-minute screen recording showing the system end-to-end. Record with OBS, Windows Game Bar (`Win + G`), or any screen recorder.

## Pre-flight (do BEFORE recording)

```powershell
# 1. Make sure data is built
cd D:\quickee
uv run python scripts/build_catalog.py   # confirm "OK 68 tops + 102 bottoms"
uv run python scripts/ingest.py           # confirm "[ok] ingested 170 items"

# 2. Clear the demo cache so the first call is a fresh agent run.
# (Optional — only if you want to demonstrate fresh agent timing.)
Remove-Item D:\quickee\chroma_db\*cache* -Recurse -Force -ErrorAction SilentlyContinue
```

## Recording

### Scene 1 — Project tour (30s)

Open `D:\quickee` in your editor. Tour the tree:
- `README.md` and `ARCHITECTURE.md` (open both briefly)
- `src/quickee/` — point out `scraper/`, `rag/`, `agent/`, `cache/`, `api/`
- `data/processed/catalog.json` — open, scroll to show 170 items with all fields

> *Voice-over*: "Backend that scrapes Uniqlo + Bewakoof, indexes 170 items in ChromaDB with rich metadata, and exposes an agentic LangGraph workflow via FastAPI."

### Scene 2 — Start the API (15s)

```powershell
cd D:\quickee
uv run uvicorn quickee.api.main:app --host 127.0.0.1 --port 8000
```

Wait for `startup.ready` log. Open `http://127.0.0.1:8000/docs` in browser.

> *Voice-over*: "Single endpoint, `POST /api/v1/style-me`, with auto-generated OpenAPI docs."

### Scene 3 — Complex prompt (Swagger UI) (60s)

In Swagger UI: click `POST /api/v1/style-me` → "Try it out".

**Paste prompt 1**:
```json
{
  "prompt": "I have dark navy chinos. What t-shirt should I wear for a summer yacht party?"
}
```

Click Execute. While agent runs, **switch back to the terminal** showing uvicorn logs. Narrate as the trace prints:
- `parse_intent` — Gemini extracts occasion, owned items, slots
- `retrieve_slots` — vector search filtered by category=top
- `compose_outfit` — Gemini picks an item and writes the Stylist Note
- `validate` — schema + slot coverage passes
- `respond`

Switch back to Swagger; show the JSON response, **highlight the `stylist_note`** and `agent_trace`.

> *Voice-over*: "Every step is logged. The `agent_trace` in the response is the literal sequence of LangGraph nodes — no black-box magic."

### Scene 4 — Multi-slot complex prompt (45s)

**Prompt 2** (a full outfit with budget):
```json
{
  "prompt": "Suggest a full smart-casual outfit for a wedding cocktail evening in Mumbai. I prefer earthy tones.",
  "max_budget_inr": 4500
}
```

Show the result: top + bottom recommendation, total under budget, stylist note that references occasion + tones + fabric details.

> *Voice-over*: "Two-slot recommendation, color-themed, under budget. The agent factored 'earthy tones' into both the retrieval query and the stylist note."

### Scene 5 — Frugal mindset / semantic cache (45s)

**Prompt 3** — same as Prompt 2 (verbatim):

Show response comes back in **<1 second**, `cache_hit: true`. Terminal log shows `cache.hit` line.

**Prompt 4** — reworded paraphrase of Prompt 2:
```json
{
  "prompt": "What complete smart-casual look do you suggest for a Mumbai wedding cocktail night? Earthy palette please.",
  "max_budget_inr": 4500
}
```

Same response, still `cache_hit: true`, still <1s.

> *Voice-over*: "Semantic cache. Different words, same intent — we recognize cosine similarity above 0.93 and skip the LLM entirely. Zero tokens spent on this request."

### Scene 6 — Wrap (15s)

Switch to the terminal. Show ARCHITECTURE.md Mermaid block (or render at https://mermaid.live).

> *Voice-over*: "Stack: Uniqlo + Bewakoof scrapers via Playwright + structured JSON-LD/Next.js data. 170 items embedded with Gemini gemini-embedding-001 @ 768 dims. LangGraph state machine with explicit parse/retrieve/compose/validate nodes. Semantic cache on the same Chroma instance. Single FastAPI endpoint."

## Test prompt cheat-sheet

```text
1. "I have dark navy chinos. What t-shirt should I wear for a summer yacht party?"
2. "Suggest a full smart-casual outfit for a wedding cocktail evening in Mumbai. I prefer earthy tones." (budget 4500)
3. (repeat 2 verbatim — cache hit)
4. "What complete smart-casual look do you suggest for a Mumbai wedding cocktail night? Earthy palette please." (budget 4500 — semantic cache hit)
5. "Need a comfortable gym outfit for a hot Bangalore morning run." (budget 3000 — fresh agent, athletic items)
```

## If something fails on camera

- **500 error**: `Remove-Item .\chroma_db\` and re-run `scripts/ingest.py`. Most likely cause is stale partial state.
- **Quota error from Gemini**: free tier is 100 embed RPM. If you spam, you'll get 429. Wait 60s.
- **Empty response**: re-check that `data/processed/catalog.json` has items (`Get-Content catalog.json | Measure-Object`)
