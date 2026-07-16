# Lead-Gen Agent

An LLM agent that finds and qualifies sales leads for a **digital marketing / web-development agency**, then drafts grounded outreach — with a human approval step before anything gets sent.

**scrape → qualify → *pause for human approval* → draft outreach email**

Runs on **two interchangeable backends** (a free local model or Claude) behind a shared interface, is exposed as a **containerized API**, and ships with an **eval harness + cost/latency observability** — not just a working pipeline, but one that's instrumented and checked.

> Working **prototype** built as an applied-LLM/agentic-engineering learning project, not a production product. It's meant to demonstrate the engineering end-to-end.

<!-- Screenshot or short GIF of the Gradio approve/reject flow goes here — biggest thing a reader looks at. -->
<!-- ![Dashboard demo](docs/demo.gif) -->

## What it actually does

1. **Scrapes** each prospect's site (`requests` + BeautifulSoup), flagging JS-rendered/empty pages as `review` instead of scoring them cold on bad data.
2. **Qualifies** the lead via LLM into a `hot` / `warm` / `cold` tier and score — with tier *always* derived deterministically from the score in code, never trusted from the model's own self-reported field.
3. **Pauses** hot/warm leads before drafting, via a LangGraph checkpoint — a human reviews and Approves or Rejects each one in the dashboard before any email gets written.
4. **Drafts** a RAG-grounded outreach email on approval, citing real services/case studies from a local knowledge base.

Known competitors are excluded before step 2 even runs — a small hardcoded domain list, not an LLM judgment call, after a real competitor repeatedly (and inconsistently) scored `hot` in testing.

## Backends

| Backend | Model | Cost | Use it for |
|---|---|---|---|
| **Open-source** | Ollama (e.g. `llama3.2`) | Free, local | Development and bulk runs |
| **Anthropic** | Claude (e.g. `claude-haiku-4-5`) | ~$0.002/lead qualified | Higher-quality drafts on your shortlist |

Scraper, prompts, schema, and pipeline are shared; only the LLM client differs, selectable via env var or CLI flag.

## Beyond the pipeline

- **Human-in-the-loop via LangGraph** (`graph.py`) — a `MemorySaver` checkpoint with `interrupt_before=["draft"]` genuinely pauses mid-execution and resumes on command (`invoke(None, config)`), rather than just reordering an existing for-loop. Wired into the Gradio dashboard as live Approve/Reject buttons per lead.
- **MCP server** (FastMCP) exposing `scrape_website` / `save_lead` / `list_leads` tools, a `leads://all` resource, and a `qualify_lead` prompt — tested end-to-end in Claude Desktop over HTTP transport.
- **FastAPI endpoint** (`api.py`) wrapping qualification as a stateless `POST /qualify`, containerized with Docker (`Dockerfile`) — backend selectable via env var so the same image runs Anthropic or Ollama without a rebuild.
- **Observability** (`llm_base.py`) — every real LLM call is timed and logged to `outputs/llm_calls.jsonl` (latency, input/output tokens, computed USD cost, pipeline stage), through one shared choke point rather than scattered logging.
- **Eval harness** (`eval.py` + `eval_set.csv`) — a small hand-labeled set of prospects, including a deliberately adversarial case, checked against actual pipeline output. Already caught two real issues: an inconsistent competitor-qualification bug (now fixed deterministically) and a scraper failing silently on a bot-walled site.

## Project structure

Flat, no subpackages, deliberately (avoids the import-shadowing issues that come from nested `llm/`-style folders):

```
app.py                 Gradio dashboard - graph-based, with approve/reject
api.py                 FastAPI /qualify endpoint
graph.py               LangGraph state machine + human-in-the-loop pause/resume
run_graph.py            CLI entry point for the graph pipeline
eval.py / eval_set.csv  Evaluation harness + hand-labeled test cases
llm_base.py             Shared LLM interface + observability logging
anthropic_backend.py    Claude backend
ollama_backend.py       Local/free backend
pipeline.py             scrape -> qualify -> draft orchestration
scraper.py              Website fetch + clean + empty/blocked-page detection
schemas.py              Lead-assessment shape + deterministic tier derivation
prompts.py              Analysis + outreach prompt builders
config.py               Agency details, model names, competitor list, cost knobs
Dockerfile / .dockerignore / requirements.txt
prospects.csv           Sample input
```

## Setup & run

```bash
pip install -r requirements.txt
```

**Dashboard (recommended):**
```bash
python app.py   # http://127.0.0.1:7860
```

**CLI, with human-in-the-loop:**
```bash
python run_graph.py --limit 5 --threshold 70
```

**API, containerized:**
```bash
docker build -t leadgen-api .
docker run -p 8000:8000 --env-file .env leadgen-api   # docs at /docs
```

**Eval:**
```bash
python eval.py
```

Anthropic reads `ANTHROPIC_API_KEY` via `python-dotenv` from a `.env` file (gitignored — never bake keys into the Docker image; pass with `--env-file` at run time).

## Configure

`config.py`: `COMPANY_NAME` / `COMPANY_PITCH` / `SENDER_NAME` (ground outreach in your agency), `ANTHROPIC_MODEL` / `OLLAMA_MODEL`, `COMPETITOR_DOMAINS`, and the cost/quality knobs (`SCRAPE_CHAR_LIMIT`, `*_MAX_TOKENS`, `DRAFT_THRESHOLD`).

## How scoring works

Two scores running in opposite directions: `lead_score` (higher = better prospect for *you*) and `website_quality_score` (lower = more opportunity — a weak existing site usually means more work to pitch). Only `lead_score` drives the tier, computed in code from a fixed threshold table in `schemas.py`, never read from the model's own `tier` field:

| `lead_score` | Tier | Action |
|---|---|---|
| 70–100 | hot | Reach out — pauses for approval before drafting |
| 45–69 | warm | Worth a look — also pauses for approval |
| 0–44 | cold | Skip |
| — | review | Page empty/JS-rendered/blocked — couldn't be judged, check manually |

## RAG grounding

Outreach emails cite **real** services/case studies instead of generic claims. `knowledge/*.md` files are chunked, embedded locally (`nomic-embed-text` via Ollama, no PyTorch), cached, and the top-matching chunks are retrieved and injected at draft time. Degrades gracefully to ungrounded drafting if embeddings are unavailable. Toggle via `USE_RAG` in `config.py`.

## Limitations

- **Scraper can't run JavaScript or pass bot walls.** `requests`+BeautifulSoup handles static sites well; JS-heavy sites come back empty (`review` tier) and sites behind a JS challenge (Cloudflare, etc.) fail outright with a 403 — confirmed directly, not assumed, via a standalone scraper smoke test. Clean fix is a Playwright-backed scraper (roadmap).
- **`MemorySaver` checkpointing is in-process only** — pending human-approval state doesn't survive an app restart. `SqliteSaver` (drop-in swap, same checkpointer interface) is the fix, not yet done.
- **Eval set is small (5 cases)** — real signal, not yet statistically robust.
- Serial processing (no concurrency yet), no CI/CD.

## Roadmap

- Playwright-backed scraper for JS-rendered/bot-walled sites
- `SqliteSaver` for durable human-in-the-loop state across restarts
- Concurrent prospect processing
- GitHub Actions CI (lint + eval-as-regression-test on every push)
- Expanded eval set; track pass rate across repeated runs, not single pass/fail, given real LLM output variance
- Multi-agent refactor (scanner / analyst / messenger) with memory to skip already-seen leads