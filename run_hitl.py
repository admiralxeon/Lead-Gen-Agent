"""
Human-in-the-loop runner (LangGraph Step 5).

The agent qualifies each prospect, then PAUSES and asks you whether to draft an
outreach email. Approve and it drafts; reject and it skips - and either way it
resumes from exactly where it stopped, without re-scraping or re-running the
qualification LLM call.

    python run_hitl.py                         # Ollama, in-memory state
    python run_hitl.py --backend anthropic
    python run_hitl.py --persist               # SQLite: pauses survive a restart

--persist writes to checkpoints.sqlite, which is what makes this more than a
blocking prompt: the process can exit while a lead is awaiting review, and a
later run can resume that same thread.
"""

import argparse
import os
import uuid

import config
import graph as graph_mod
import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient

CYAN, GREEN, YELLOW, RED, BOLD, RESET = (
    "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m")


def show_pending(p):
    print(f"\n{BOLD}--- REVIEW REQUIRED ---{RESET}")
    print(f"  Company : {p.get('company')}")
    print(f"  URL     : {p.get('url')}")
    print(f"  Tier    : {p.get('tier')}  (score {p.get('lead_score')})")
    print(f"  Summary : {p.get('summary')}")
    for o in (p.get("observations") or [])[:3]:
        print(f"    - {o}")


def ask():
    while True:
        ans = input(f"{BOLD}Draft an email for this lead? [y/n]: {RESET}").strip().lower()
        if ans in ("y", "yes"):
            return "approve"
        if ans in ("n", "no"):
            return "reject"
        print("  please answer y or n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/prospects.csv")
    ap.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=int, default=config.DRAFT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--persist", action="store_true",
                    help="store paused state in SQLite so it survives a restart")
    args = ap.parse_args()

    client = (AnthropicClient(model=args.model) if args.backend == "anthropic"
              else OllamaClient(model=args.model))

    checkpointer = None
    cm = None
    if args.persist:
        from langgraph.checkpoint.sqlite import SqliteSaver
        cm = SqliteSaver.from_conn_string("checkpoints.sqlite")
        checkpointer = cm.__enter__()

    try:
        app = graph_mod.build_graph(client, require_approval=True,
                                    checkpointer=checkpointer)

        prospects = pipeline.read_prospects(args.input)
        if args.limit:
            prospects = prospects[: args.limit]

        print(f"{CYAN}Human-in-the-loop run · backend={args.backend} · "
              f"{len(prospects)} prospects{RESET}")

        results, emails = [], []
        for i, p in enumerate(prospects, 1):
            print(f"\n{CYAN}[{i}/{len(prospects)}] {p['url']}{RESET}")
            thread_id = f"{uuid.uuid4().hex[:8]}"

            state = graph_mod.run_one(app, p["url"], p["name"],
                                      args.threshold, thread_id=thread_id)

            pending = graph_mod.pending_review(state)
            if pending:
                show_pending(pending)
                decision = ask()
                state = graph_mod.resume_one(app, thread_id, decision)

            a = state["assessment"]
            status = state.get("status")
            color = (GREEN if status == "drafted" else
                     YELLOW if status == "rejected" else RED)
            print(f"  {color}{status.upper()}{RESET}  "
                  f"{a['tier']} · score {a['lead_score']}")

            results.append(a)
            if state.get("email"):
                emails.append((a, state["email"]))

        os.makedirs("outputs", exist_ok=True)
        pipeline._write_csv(results, "outputs/results.csv")
        pipeline._write_emails(emails, "outputs/outreach_emails.md")

        print(f"\n{GREEN}Done. {len(results)} prospects · "
              f"{len(emails)} approved and drafted.{RESET}")
        print("  outputs/results.csv\n  outputs/outreach_emails.md")
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()