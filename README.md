# Quickeee — Luxury Stylist Concierge

An agentic backend that scrapes premium fashion inventory, indexes it in a vector store, and uses a multi-step LangGraph agent to return outfit recommendations via a single FastAPI endpoint.

> Take-home for Quickeee's Gen AI / Data Engineer role.

## What it does

```
POST /api/v1/style-me
{ "prompt": "I have dark navy chinos, what t-shirt should I wear for a summer yacht party?" }
   │
   ▼
parse_intent → retrieve (RAG) → compose_outfit → validate → respond
   │
   ▼
{ "items": [...], "total_price_inr": 4298, "stylist_note": "..." }
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Google Gemini Flash | Generous free tier, fast, native structured-output JSON |
| Embeddings | Gemini `gemini-embedding-001` @ 768 dims (Matryoshka) | Same provider = single key; shrunk dims for cheap storage |
| Vector DB | ChromaDB (local persistent) | Zero infra; pre-filters on metadata before ANN |
| Agent | LangGraph | Explicit state machine; every step is debuggable |
| Scraping | Playwright + structured JSON-LD / `__NEXT_DATA__` | Real browser bypasses anti-bot; structured data > CSS selectors |
| API | FastAPI + Uvicorn | Async, auto OpenAPI docs at `/docs` |
| Package mgr | uv | Reproducible installs in seconds |

## Data sources

Scraped (catalog totals: **170 items** = 68 tops + 102 bottoms):
- **Uniqlo India** — `/in/en/men/tops`, `/in/en/men/bottoms`
- **Bewakoof** — `/men-t-shirts`, `/men-joggers`, `/men-shorts`

Original spec named H&M as an example. H&M's Akamai Bot Manager returns
HTTP 403 to every headless Chrome request on both India and US storefronts;
the pivot to Uniqlo + Bewakoof is documented in [ARCHITECTURE.md](./ARCHITECTURE.md#32-scraping-pivot-hm--uniqlo--bewakoof).

## Setup

### 1. Prerequisites
- Python 3.12+
- `uv` package manager — install: `pip install uv` (or see [uv docs](https://docs.astral.sh/uv/))

### 2. Install
```bash
git clone https://github.com/<your-user>/quickee.git
cd quickee
uv sync
uv run playwright install chromium
```

### 3. Configure
```bash
cp .env.example .env
# edit .env and paste your GEMINI_API_KEY
# get a key (free): https://aistudio.google.com/
```

### 4. Smoke-test the key
```bash
uv run python scripts/smoke_gemini.py
```

### 5. Build the catalog (one-time)
```bash
uv run python scripts/scrape.py       # → data/raw/*.json
uv run python scripts/ingest.py       # → chroma_db/
```

### 6. Run the API
```bash
uv run uvicorn quickee.api.main:app --reload
# open http://localhost:8000/docs
```

## Folder layout

```
quickee/
├── src/quickee/
│   ├── config.py       # pydantic-settings env loader
│   ├── models.py       # canonical Item + API contracts
│   ├── scraper/        # Uniqlo + Bewakoof Playwright scrapers
│   ├── rag/            # embeddings + ChromaDB ingest + retriever
│   ├── agent/          # LangGraph state, nodes, prompts
│   ├── cache/          # semantic prompt cache
│   └── api/            # FastAPI app + /api/v1/style-me route
├── scripts/            # CLI entry points (scrape, ingest, smoke tests)
├── data/raw/           # raw scraped JSON (gitignored)
├── data/processed/     # normalized + enriched catalog (gitignored)
├── chroma_db/          # vector store (gitignored, rebuildable)
└── ARCHITECTURE.md     # design rationale + Mermaid flowchart
```

## Docs
- [ARCHITECTURE.md](./ARCHITECTURE.md) — design choices, schema, prompt strategies, Mermaid flowchart
- [DEMO.md](./DEMO.md) — step-by-step script for the screen recording
