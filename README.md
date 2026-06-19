# Lead-Gen Agent (v1, flat layout)

LLM agent that finds & qualifies sales leads for a digital marketing / web-dev agency.
Per prospect website: scrape -> analyze -> score/qualify -> draft a personalized email.

Two interchangeable backends (same shared logic):
- Open-source: Ollama (local, free)   -> run_ollama.py + ollama_backend.py
- Anthropic:   Claude (pay per token)  -> run_anthropic.py + anthropic_backend.py

## Files (all in ONE folder - no subfolders)
- run_ollama.py / run_anthropic.py  -> entry points
- ollama_backend.py / anthropic_backend.py -> the two LLM versions
- llm_base.py    -> shared LLM interface + JSON parsing
- pipeline.py    -> scrape->analyze->qualify->draft orchestration
- scraper.py / prompts.py / schemas.py / config.py -> shared building blocks
- prospects.csv  -> sample input (edit the url column)

## Setup
    pip install -r requirements.txt          # anthropic, openai, requests, beautifulsoup4
    # DO NOT run `pip install llm` - that's an unrelated package and not needed.

## Run (open-source / free)
    ollama pull llama3.2
    python run_ollama.py

## Run (Anthropic)
    set ANTHROPIC_API_KEY=sk-ant-...          # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
    python run_anthropic.py --limit 5 --threshold 70

## Configure
Edit config.py: COMPANY_NAME / COMPANY_PITCH (so outreach is grounded in your agency),
model names, and cost knobs (SCRAPE_CHAR_LIMIT, *_MAX_TOKENS, DRAFT_THRESHOLD).

## Output
outputs/results.csv (ranked leads) + outputs/outreach_emails.md (drafts).
