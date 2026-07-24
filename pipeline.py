"""
The lead-gen pipeline. Backend-agnostic: hand it any LLMClient and it runs the
same four steps for every prospect.

  scrape  ->  analyze (JSON)  ->  qualify (threshold)  ->  draft email

Outputs:
  outputs/results.csv          one row per prospect, with scores
  outputs/outreach_emails.md   drafted emails for qualifying leads
"""

import csv
import os
import time

import config
import prompts
import rag
import schemas
from llm_base import LLMClient
from scraper import Website

# ANSI colors so you can watch it work (course style)
CYAN, GREEN, YELLOW, RED, RESET = "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[0m"


def read_prospects(path: str):
    """Read prospects.csv -> list of {name, url}. 'name' column is optional."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            url = (r.get("url") or "").strip()
            if not url:
                continue
            rows.append({"name": (r.get("name") or "").strip(), "url": url})
    return rows


def assess_prospect(client: LLMClient, name: str, url: str) -> dict:
    """Scrape + analyze a single prospect. Returns a normalized assessment dict."""
    site = Website(url)
    if not site.ok:
        return {**schemas.normalize({}, name or url, url),
                "summary": f"SCRAPE FAILED: {site.error}", "tier": "cold", "lead_score": 0}

    if site.looks_empty:
        # Fetch worked but there's no readable text - almost always a JS-rendered
        # site. Don't waste an LLM call; flag it so you can check it by hand.
        return {**schemas.normalize({}, name or site.title or url, url),
                "summary": "EMPTY/JS-RENDERED: little or no text scraped - review manually",
                "tier": "review", "lead_score": 0}

    fallback_name = name or site.title or url
    try:
        raw = client.complete_json(
            system=prompts.ANALYSIS_SYSTEM,
            user=prompts.analysis_user_prompt(fallback_name, site.describe(config.SCRAPE_CHAR_LIMIT)),
            max_tokens=config.ANALYSIS_MAX_TOKENS,
        )
    except Exception as e:
        # Callers turn exceptions into a score-0 result, which makes a broken
        # API key or a JSON parse failure look like a legitimate low score.
        # Print it here so the real cause is always visible in the console.
        print(f"\n  !! ANALYSIS FAILED for {url}")
        print(f"     {type(e).__name__}: {e}\n")
        raise

    assessment = schemas.normalize(raw, fallback_name, url)
    if assessment["lead_score"] == 0:
        # Parsed fine but scored 0 - usually means the model returned different
        # field names than we asked for. Show what actually came back.
        print(f"  ?? lead_score=0 for {url}; model returned keys: {sorted(raw.keys())}")
    return assessment


def draft_email(client: LLMClient, assessment: dict) -> str:
    context = rag.context_for(assessment)  # "" if RAG off/unavailable
    return client.complete(
        system=prompts.DRAFT_SYSTEM,
        user=prompts.draft_user_prompt(assessment, context=context),
        max_tokens=config.DRAFT_MAX_TOKENS,
    )


def run(client: LLMClient, prospects, threshold: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    results, emails = [], []

    print(f"{CYAN}Running lead-gen with backend: {client.name}{RESET}\n")
    for i, p in enumerate(prospects, 1):
        print(f"{CYAN}[{i}/{len(prospects)}] {p['url']}{RESET}")
        try:
            a = assess_prospect(client, p["name"], p["url"])
        except Exception as e:
            print(f"  {RED}analysis error: {e}{RESET}")
            continue

        color = (GREEN if a["tier"] == "hot" else YELLOW if a["tier"] == "warm"
                 else CYAN if a["tier"] == "review" else RED)
        print(f"  {color}{a['tier'].upper():4} lead_score={a['lead_score']:3} "
              f"site_quality={a['website_quality_score']:3}{RESET}  {a['summary'][:80]}")

        drafted = ""
        if a["lead_score"] >= threshold and not a["summary"].startswith("SCRAPE FAILED"):
            try:
                drafted = draft_email(client, a)
                emails.append((a, drafted))
                print(f"  {GREEN}-> outreach email drafted{RESET}")
            except Exception as e:
                print(f"  {RED}draft error: {e}{RESET}")

        results.append(a)
        time.sleep(config.SCRAPE_DELAY)

    _write_csv(results, os.path.join(out_dir, "results.csv"))
    _write_emails(emails, os.path.join(out_dir, "outreach_emails.md"))

    hot = sum(1 for r in results if r["tier"] == "hot")
    review = sum(1 for r in results if r["tier"] == "review")
    print(f"\n{GREEN}Done. {len(results)} prospects, {hot} hot, "
          f"{len(emails)} emails drafted, {review} need manual review.{RESET}")
    print(f"  {out_dir}/results.csv\n  {out_dir}/outreach_emails.md")
    return results


def _write_csv(results, path):
    cols = ["company_name", "url", "tier", "lead_score",
            "website_quality_score", "summary", "observations", "opportunities"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(results, key=lambda x: x["lead_score"], reverse=True):
            w.writerow([
                r["company_name"], r["url"], r["tier"], r["lead_score"],
                r["website_quality_score"], r["summary"],
                " | ".join(r["observations"]), " | ".join(r["opportunities"]),
            ])


def _write_emails(emails, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Drafted outreach emails\n\n")
        for a, body in sorted(emails, key=lambda x: x[0]["lead_score"], reverse=True):
            f.write(f"## {a['company_name']}  ({a['tier']}, score {a['lead_score']})\n")
            f.write(f"<{a['url']}>\n\n")
            f.write(body.strip() + "\n\n---\n\n")