"""
Run the lead-gen agent on the OPEN-SOURCE (Ollama) backend. Free, local, no key.

    ollama pull llama3.2          # one time
    python run_ollama.py --input prospects.csv

Because it's local you can process your whole list without watching a budget.
If JSON parsing is flaky on a small model, try a stronger one:
    python run_ollama.py --model qwen2.5
"""

import argparse

import config
import pipeline
from ollama_backend import OllamaClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/prospects.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--limit", type=int, default=None, help="max prospects to process")
    ap.add_argument("--threshold", type=int, default=config.DRAFT_THRESHOLD)
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    args = ap.parse_args()

    prospects = pipeline.read_prospects(args.input)
    if args.limit:
        prospects = prospects[: args.limit]

    client = OllamaClient(model=args.model)
    pipeline.run(client, prospects, args.threshold, args.out_dir)


if __name__ == "__main__":
    main()
