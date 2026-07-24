"""
Diagnose why the Anthropic backend returns 0 scores.

The pipeline swallows exceptions and turns them into a score-0 result, so a
failure looks like a bad score. This script bypasses that and shows you the
RAW response, plus exactly where it breaks.

    python diag_anthropic.py                      # uses config model
    python diag_anthropic.py --model claude-haiku-4-5
    python diag_anthropic.py --url https://example.com
"""

import argparse
import json

import config
import prompts
from anthropic_backend import AnthropicClient, _first_text
from llm_base import extract_json
from scraper import Website


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://example.com")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    model = args.model or config.ANTHROPIC_MODEL
    max_tokens = args.max_tokens or config.ANALYSIS_MAX_TOKENS
    print(f"model      = {model}")
    print(f"max_tokens = {max_tokens}\n")

    # 1. scrape
    site = Website(args.url)
    if not site.ok:
        print(f"SCRAPE FAILED: {site.error}")
        return
    print(f"scraped {len(site.text)} chars from {args.url}\n")

    # 2. raw API call (no try/except hiding anything)
    client = AnthropicClient(model=model)
    user = prompts.analysis_user_prompt(site.title, site.describe(config.SCRAPE_CHAR_LIMIT))

    try:
        resp = client.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=prompts.ANALYSIS_SYSTEM,
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": "{"},
            ],
        )
    except Exception as e:
        print("API CALL FAILED:")
        print(f"  {type(e).__name__}: {e}")
        print("\n-> If this is a 404, the model string is wrong.")
        print("   Valid examples: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-8")
        return

    print(f"stop_reason = {resp.stop_reason}")
    print(f"usage       = in {resp.usage.input_tokens}, out {resp.usage.output_tokens}\n")

    raw = "{" + _first_text(resp)
    print("--- RAW RESPONSE ---")
    print(raw)
    print("--- END RAW ---\n")

    if resp.stop_reason == "max_tokens":
        print("!! TRUNCATED: the model hit max_tokens mid-JSON, so parsing fails.")
        print(f"   Fix: raise ANALYSIS_MAX_TOKENS in config.py (currently {max_tokens}).")
        return

    # 3. parse
    try:
        parsed = extract_json(raw)
    except Exception as e:
        print(f"JSON PARSE FAILED: {type(e).__name__}: {e}")
        return

    print("PARSED OK:")
    print(json.dumps(parsed, indent=2)[:800])

    ls = parsed.get("lead_score")
    print(f"\nlead_score returned by model: {ls!r}")
    if ls in (None, 0, "0"):
        print("-> The model genuinely returned 0/absent. Not a parsing bug;")
        print("   check that the scraped text actually has content to judge.")
    else:
        print("-> Scoring works here. If the app still shows 0, the failure is")
        print("   elsewhere in the pipeline (check the summary column for ERROR:).")


if __name__ == "__main__":
    main()