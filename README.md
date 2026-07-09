# Lead-Gen Agent

An LLM agent that finds and qualifies sales leads for a **digital marketing / web-development agency**.

Give it a list of prospect websites and it runs a four-step pipeline on each one:

**scrape → analyze → score & qualify → draft a personalized outreach email**

It runs on **two interchangeable backends** — a free local model or Claude — behind a single shared interface, so the same pipeline code works with either.

> This is a working **prototype** built as an applied-LLM learning project, not a production product. It's meant to demonstrate the engineering end-to-end.

<!-- Add a screenshot or short demo GIF of the dashboard here — it's the single biggest thing a reader looks at. -->
<!-- ![Dashboard demo](docs/demo.gif) -->

## Backends

| Backend | Model | Cost | Use it for |
|---------|-------|------|------------|
| **Open-source** | Ollama (e.g. `llama3.2`) | Free, local | Development and bulk runs |
| **Anthropic** | Claude (e.g. `claude-haiku-4-5`) | Pay per token | Higher-quality drafts on your shortlist |

The scraper, prompts, schema, and pipeline are **shared**; only the LLM module and the run script differ. Develop for free on Ollama, then send only your best leads through Claude.

## Features

- **Two interchangeable backends** behind one `LLMClient` interface (`llm_base.py`) — switch with a dropdown or a different run script.
- **Cross-provider structured output** — reliable JSON from both backends (assistant-prefill for Claude, JSON mode for Ollama) with a brace-matching fallback parser that tolerates malformed responses.
- **Live Gradio dashboard** (`app.py`) that streams results in as each site is processed, with a ranked table, per-lead email viewer, and CSV download.
- **Fails loud, not silent** — JavaScript-rendered sites return near-empty HTML to a basic scraper. Instead of scoring them "cold," the agent flags them `review` so bad data never pollutes the results.
- **Cost controls** — truncated scrapes, token caps, a `--limit` flag, and a draft threshold so emails are only written for qualifying leads.

## Project structure

All files live in one folder (no subpackages, deliberately):

```
app.py                 Gradio dashboard (recommended entry point)
run_ollama.py          CLI entry point — open-source backend
run_anthropic.py       CLI entry point — Anthropic backend
ollama_backend.py      Ollama LLM client
anthropic_backend.py   Anthropic LLM client
llm_base.py            Shared LLM interface + JSON parsing
pipeline.py            scrape -> analyze -> qualify -> draft orchestration
scraper.py             Website fetch + clean (BeautifulSoup) + empty-page detection
prompts.py             Analysis + outreach prompt builders
schemas.py             Lead-assessment shape + score/tier normalization
config.py              Your agency details, model names, cost knobs
prospects.csv          Sample input (edit the url column)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `anthropic`, `openai`, `requests`, `beautifulsoup4`, `gradio`.
(Do **not** `pip install llm` — that's an unrelated PyPI package and will shadow local imports.)

## Run

### Dashboard (recommended)
```bash
python app.py
```
Open the local URL it prints (usually `http://127.0.0.1:7860`). Paste URLs, pick a backend, hit **Find Leads**.

### CLI — open-source / free
```bash
ollama pull llama3.2
python run_ollama.py
```

### CLI — Anthropic
```bash
# PowerShell:  $env:ANTHROPIC_API_KEY="sk-ant-..."
# bash/zsh:    export ANTHROPIC_API_KEY=sk-ant-...
python run_anthropic.py --limit 5 --threshold 70
```

The Anthropic backend reads `ANTHROPIC_API_KEY` from the environment of whatever terminal launches it. To avoid setting it every session, copy `.env.example` to `.env`, add your key, add `python-dotenv` to `requirements.txt`, and put `from dotenv import load_dotenv; load_dotenv()` at the top of your entry point. Keep `.env` in `.gitignore`.

## Configure

Edit `config.py`:
- `COMPANY_NAME`, `COMPANY_PITCH`, `SENDER_NAME` — so outreach is grounded in *your* agency.
- `ANTHROPIC_MODEL` / `OLLAMA_MODEL` — swap models.
- `SCRAPE_CHAR_LIMIT`, `*_MAX_TOKENS`, `DRAFT_THRESHOLD` — the cost/quality knobs.

## Retrieval-augmented grounding (RAG)

Outreach emails are grounded in the agency's own knowledge base so they cite **real** services and case-study results instead of generic claims.

How it works: markdown files in `knowledge/` are chunked, embedded once (local `nomic-embed-text` via Ollama — no PyTorch), and cached to `rag_index.json`. When drafting for a qualified lead, the agent retrieves the top chunks most relevant to that lead's identified problems (cosine similarity) and injects them into the draft prompt, with instructions to cite only real results.

- `embedder.py` — swappable embedding client (mirrors the LLM-backend pattern).
- `rag.py` — chunk → embed → cache → retrieve, plus `context_for()` (lazy, process-wide index; degrades to ungrounded drafting if embeddings are unavailable).
- `knowledge/*.md` — services, case studies, and past winning outreach (edit for your agency).
- Toggle with `USE_RAG` in `config.py`.

Verify it before relying on it:
```bash
ollama pull nomic-embed-text
python test_embeddings.py   # confirms embeddings work on your machine
python test_rag.py          # confirms retrieval returns relevant chunks
```

## How scoring works

Each prospect gets **two scores that run in opposite directions**:

- **`lead_score` (0–100, higher = better)** — how strong a prospect this is *for you*. Drives the tier and the decision to draft an email.
- **`website_quality_score` (0–100, lower = more opportunity)** — how good their *current* site is. A weak site usually means more for the agency to fix, pushing `lead_score` up. It's a diagnostic, not the ranking field.

`lead_score` maps to a tier:

| `lead_score` | Tier | Meaning | Action |
|---|---|---|---|
| 70–100 | **hot** | Strong fit, clear gaps to fix | Reach out first |
| 45–69 | **warm** | Plausible, weaker signal | Worth a look |
| 0–44 | **cold** | Poor fit or too little to work with | Skip |
| — | **review** | Page empty / JS-rendered — couldn't be judged | Check manually |

Thresholds live in `schemas.py` and are meant to be tuned once you've eyeballed a real batch. Note `DRAFT_THRESHOLD` defaults to 60, which sits inside the warm band — set it to 70 if you only ever want to email hot leads.

## Output

- `outputs/results.csv` — every prospect, ranked by lead score.
- `outputs/outreach_emails.md` — drafted emails for qualifying leads.

## Limitations

The default scraper is `requests` + BeautifulSoup: fast and free, but it can't run JavaScript and has no anti-bot defenses. It handles static small-business sites well; JS-heavy (React/Wix/Squarespace) sites come back empty and get flagged `review`, and sites behind Cloudflare/CAPTCHAs will fail. The clean upgrade is a swappable scraper interface with a Playwright (free, JS-capable) or Firecrawl (paid, LLM-ready) backend.

## Design notes

A few decisions worth calling out, since they're the point of the project:

- **One brain, two front doors.** `app.py` and the CLI scripts both import the same `pipeline` functions — no logic is duplicated in the UI. Fix a bug in the pipeline and every entry point gets it.
- **Provider abstraction over provider lock-in.** Both backends satisfy the same interface, which is what makes "develop free locally, pay only for the shortlist" possible without branching the pipeline.
- **Defensive parsing as a first-class concern.** Local models produce messy JSON; the fallback parser treats that as expected, not exceptional.

## Roadmap

- Swappable scraper backend (Playwright / Firecrawl) for JS-rendered sites.
- CSV upload in the dashboard instead of pasting URLs.
- A prospect-discovery step so the agent finds leads instead of being handed them.
- Multi-agent refactor (scanner / analyst / messenger) with memory to skip already-seen leads.