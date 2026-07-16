import argparse

from langgraph.checkpoint.memory import MemorySaver

import config
import graph as graph_mod
import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient

CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[96m",
    "\033[92m",
    "\033[93m",
    "\033[91m",
    "\033[0m",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="prospects.csv")
    ap.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=int, default=config.DRAFT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--show-graph",
        action="store_true",
        help="print the graph structure (Mermaid) and exit",
    )
    args = ap.parse_args()

    client = (
        AnthropicClient(model=args.model)
        if args.backend == "anthropic"
        else OllamaClient(model=args.model)
    )
    checkpointer = MemorySaver()
    app = graph_mod.build_graph(client, checkpointer=checkpointer)

    if args.show_graph:
        print("\n--- Graph structure (paste into https://mermaid.live) ---\n")
        print(app.get_graph().draw_mermaid())
        return

    prospects = pipeline.read_prospects(args.input)
    if args.limit:
        prospects = prospects[: args.limit]

    print(
        f"{CYAN}LangGraph pipeline · backend={args.backend} · "
        f"{len(prospects)} prospects{RESET}\n"
    )

    results, emails, pending = [], [], []
    for i, p in enumerate(prospects, 1):
        thread_id = p["url"]
        print(f"{CYAN}[{i}/{len(prospects)}] {p['url']}{RESET}")
        state = graph_mod.run_one(
            app, p["url"], p["name"], args.threshold, thread_id=thread_id
        )

        a = state["assessment"]
        color = (
            GREEN
            if a["tier"] == "hot"
            else YELLOW
            if a["tier"] == "warm"
            else CYAN
            if a["tier"] == "review"
            else RED
        )

        # Check whether this thread paused before "draft" (interrupt_before)
        # or ran straight through to END (e.g. it routed to "skip").
        snapshot = app.get_state({"configurable": {"thread_id": thread_id}})
        paused = bool(snapshot.next)

        status_label = "paused (awaiting approval)" if paused else state["status"]
        print(
            f"  {color}{a['tier'].upper():6} score={a['lead_score']:3}{RESET}  "
            f"-> {status_label}"
        )

        if paused:
            pending.append((p, thread_id))
        else:
            results.append(a)
            if state.get("email"):
                emails.append((a, state["email"]))

    # --- Resume phase: continue every paused thread past the draft node. ---
    # For now this auto-approves everything pending, just to prove the
    # pause -> resume mechanism works end to end. Real approve/reject
    # decisions come in 5d (wired into the Gradio dashboard).
    if pending:
        print(f"\n{CYAN}Resuming {len(pending)} paused prospect(s)...{RESET}")
        for p, thread_id in pending:
            resume_config = {"configurable": {"thread_id": thread_id}}
            final_state = app.invoke(
                None, resume_config
            )  # None = continue, don't restart

            a = final_state["assessment"]
            results.append(a)
            if final_state.get("email"):
                emails.append((a, final_state["email"]))
            print(f"  {p['url']} -> {final_state['status']}")

    state_snapshot = app.get_state(
        {"configurable": {"thread_id": prospects[-1]["url"]}}
    )
    print(f"\n{CYAN}Checkpoint check for last prospect (post-resume):{RESET}")
    print(f"  values keys: {list(state_snapshot.values.keys())}")
    print(f"  next node: {state_snapshot.next}")  # expect () now — fully complete

    pipeline._write_csv(results, "outputs/results.csv")
    pipeline._write_emails(emails, "outputs/outreach_emails.md")

    hot = sum(1 for r in results if r["tier"] == "hot")
    print(
        f"\n{GREEN}Done. {len(results)} prospects, {hot} hot, "
        f"{len(emails)} emails drafted.{RESET}"
    )
    print("  outputs/results.csv\n  outputs/outreach_emails.md")


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)
    main()
