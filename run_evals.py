"""
Run the label-free evals (Part B).

    # how stable is one model on the same prospect?
    python run_evals.py consistency --url https://example.com --runs 5

    # do the free local model and Claude agree?
    python run_evals.py agreement --limit 5

Both are instrumented, so you also see what the evaluation itself cost.
Note that evals are token-hungry by nature: consistency with --runs 5 means 5x
the calls for ONE prospect, and agreement runs every prospect through both
backends. Start small, and prefer Ollama while you're iterating.
"""

import argparse

import config
import evals
import pipeline
from anthropic_backend import AnthropicClient
from observability import InstrumentedClient, RunTracker
from ollama_backend import OllamaClient


def make(backend, model=None):
    return (AnthropicClient(model=model) if backend == "anthropic"
            else OllamaClient(model=model))


def cmd_consistency(args):
    tracker = RunTracker()
    client = InstrumentedClient(make(args.backend, args.model), tracker)

    urls = [args.url] if args.url else [p["url"] for p in
                                        pipeline.read_prospects(args.input)[: args.limit or 2]]

    print(f"Consistency: {len(urls)} prospect(s) x {args.runs} runs "
          f"on {args.backend}")
    results = [evals.consistency_eval(client, u, n=args.runs) for u in urls]

    evals.print_consistency(results)
    tracker.report()
    print(f"\nsaved -> {evals.save(results, 'eval_consistency.json')}")


def cmd_agreement(args):
    tracker = RunTracker()
    a = InstrumentedClient(make("ollama", args.model_a), tracker)
    b = InstrumentedClient(make("anthropic", args.model_b), tracker)

    urls = [p["url"] for p in pipeline.read_prospects(args.input)]
    if args.limit:
        urls = urls[: args.limit]

    print(f"Agreement: {len(urls)} prospects through ollama and anthropic")
    res = evals.agreement_eval(a, b, urls, label_a="ollama", label_b="claude")

    evals.print_agreement(res, "ollama", "claude")
    tracker.report()
    print(f"\nsaved -> {evals.save(res, 'eval_agreement.json')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("consistency", help="same prospect, repeated runs")
    c.add_argument("--url", default=None)
    c.add_argument("--input", default="prospects.csv")
    c.add_argument("--runs", type=int, default=5)
    c.add_argument("--limit", type=int, default=2)
    c.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    c.add_argument("--model", default=None)
    c.set_defaults(func=cmd_consistency)

    g = sub.add_parser("agreement", help="ollama vs anthropic on the same prospects")
    g.add_argument("--input", default="prospects.csv")
    g.add_argument("--limit", type=int, default=5)
    g.add_argument("--model-a", default=None, help="ollama model")
    g.add_argument("--model-b", default=None, help="anthropic model")
    g.set_defaults(func=cmd_agreement)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()