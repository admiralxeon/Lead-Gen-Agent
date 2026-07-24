"""
Run the pipeline with observability turned on (eval Part A).

Identical to run_ollama.py / run_anthropic.py, except every LLM call is traced.
At the end you get a summary of tokens, estimated cost, latency and errors, and
the per-call records are appended to traces.jsonl.

    python run_traced.py --limit 5
    python run_traced.py --backend anthropic --limit 3

Why this matters: you can't reason about "is the cheap model good enough?"
without measuring what each one actually costs and how slow it is. This is the
foundation the accuracy evals build on.
"""

import argparse

import config
import pipeline
from anthropic_backend import AnthropicClient
from observability import InstrumentedClient, RunTracker
from ollama_backend import OllamaClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/prospects.csv")
    ap.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=int, default=config.DRAFT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--traces", default="traces.jsonl")
    args = ap.parse_args()

    base = (AnthropicClient(model=args.model) if args.backend == "anthropic"
            else OllamaClient(model=args.model))

    tracker = RunTracker()
    client = InstrumentedClient(base, tracker)   # same interface, now traced

    prospects = pipeline.read_prospects(args.input)
    if args.limit:
        prospects = prospects[: args.limit]

    pipeline.run(client, prospects, args.threshold, "outputs")

    tracker.report()
    path = tracker.save(args.traces)
    print(f"\ntraces appended to {path}")

    if tracker.total_cost == 0:
        print("(cost $0.00 - local model, so you're only paying in time)")


if __name__ == "__main__":
    main()