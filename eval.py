"""
Evaluation harness for the lead-gen qualification step.

Runs assess_prospect() against a small hand-labeled set of prospects
(eval_set.csv) and checks whether the pipeline's tier decision matches
what a human decided it SHOULD be. This checks the qualification LOGIC,
not the scraper - it deliberately includes one adversarial case (a known
competitor) to test whether the model's own stated reasoning actually
translates into the correct tier output.

Run:
    python eval.py
    python eval.py --backend ollama
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient

load_dotenv(override=True)

CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[96m",
    "\033[92m",
    "\033[93m",
    "\033[91m",
    "\033[0m",
)

EVAL_SET_PATH = "eval_set.csv"
LOG_PATH = Path("outputs/llm_calls.jsonl")


def load_eval_set(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "url": r["url"].strip(),
                    "name": r.get("name", "").strip(),
                    "expected_tier": r["expected_tier"].strip().lower(),
                    "notes": r.get("notes", "").strip(),
                }
            )
    return rows


def run_eval(client, cases):
    results = []
    for i, case in enumerate(cases, 1):
        print(f"{CYAN}[{i}/{len(cases)}] {case['url']}{RESET}")
        try:
            a = pipeline.assess_prospect(
                client, case["name"], case["url"], stage="eval"
            )
            actual_tier = a["tier"]
            match = actual_tier == case["expected_tier"]
            error_text = None
        except Exception as e:
            a = None
            actual_tier = "ERROR"
            match = False
            error_text = str(e)
            print(f"  {RED}exception: {error_text}{RESET}")

        color = GREEN if match else RED
        label = "PASS" if match else "FAIL"
        score_str = f"  score={a['lead_score']}" if a else ""
        print(
            f"  {color}{label}{RESET}  expected={case['expected_tier']:5} "
            f"actual={actual_tier:5}{score_str}"
        )
        if a and a.get("summary"):
            print(f"  summary: {a['summary'][:120]}")
        if case["notes"]:
            print(f"  {YELLOW}note: {case['notes']}{RESET}")

        results.append(
            {
                **case,
                "actual_tier": actual_tier,
                "match": match,
                "lead_score": a["lead_score"] if a else None,
                "error": error_text,
            }
        )
    return results


def summarize(results, run_started_at):
    total = len(results)
    passed = sum(1 for r in results if r["match"])
    accuracy = passed / total if total else 0.0

    print(f"\n{CYAN}{'=' * 50}{RESET}")
    acc_color = GREEN if accuracy == 1.0 else YELLOW
    print(f"{acc_color}Accuracy: {passed}/{total} ({accuracy:.0%}){RESET}")

    # Pull cost/latency for just this run from the shared observability log
    # (stage="eval") rather than measuring it separately - one source of
    # truth for what any given run of the pipeline actually cost.
    if LOG_PATH.exists():
        eval_calls = []
        with open(LOG_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("stage") == "eval" and rec["timestamp"] >= run_started_at:
                    eval_calls.append(rec)
        if eval_calls:
            total_cost = sum(c["cost_usd"] for c in eval_calls)
            avg_latency = sum(c["latency_ms"] for c in eval_calls) / len(eval_calls)
            print(
                f"Eval run cost: ${total_cost:.4f}  ·  "
                f"avg latency: {avg_latency:.0f}ms  ·  {len(eval_calls)} LLM calls"
            )

    fails = [r for r in results if not r["match"]]
    if fails:
        print(f"\n{RED}Mismatches:{RESET}")
        for r in fails:
            note = f" ({r['notes']})" if r["notes"] else ""
            err = f"  [exception: {r['error']}]" if r.get("error") else ""
            print(
                f"  {r['url']}: expected {r['expected_tier']}, got {r['actual_tier']}{note}{err}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "anthropic"], default="anthropic")
    ap.add_argument("--eval-set", default=EVAL_SET_PATH)
    args = ap.parse_args()

    client = AnthropicClient() if args.backend == "anthropic" else OllamaClient()
    cases = load_eval_set(args.eval_set)
    run_started_at = datetime.now(timezone.utc).isoformat()

    print(f"{CYAN}Running eval · backend={client.name} · {len(cases)} cases{RESET}\n")
    results = run_eval(client, cases)
    summarize(results, run_started_at)


if __name__ == "__main__":
    main()
