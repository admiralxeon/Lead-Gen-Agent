"""
Run the lead-gen agent on the ANTHROPIC backend.

    export ANTHROPIC_API_KEY=sk-ant-...
    python run_anthropic.py --input prospects.csv --limit 5

Token-saving tips:
  --limit N         process only the first N prospects while testing
  --threshold 70    only draft emails for strong leads (fewer LLM calls)
"""

import argparse

import config
import pipeline
from anthropic_backend import AnthropicClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/prospects.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--limit", type=int, default=None, help="max prospects to process")
    ap.add_argument("--threshold", type=int, default=config.DRAFT_THRESHOLD)
    ap.add_argument("--model", default=config.ANTHROPIC_MODEL)
    args = ap.parse_args()

    prospects = pipeline.read_prospects(args.input)
    if args.limit:
        prospects = prospects[: args.limit]

    client = AnthropicClient(model=args.model)
    pipeline.run(client, prospects, args.threshold, args.out_dir)


if __name__ == "__main__":
    main()
