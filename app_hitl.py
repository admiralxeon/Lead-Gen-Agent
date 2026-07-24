"""
Human-in-the-loop dashboard (LangGraph Step 5d).

The agent qualifies prospects one at a time and PAUSES when a lead is strong
enough to email. You get a review card with Approve / Reject buttons; clicking
one resumes the paused graph from exactly where it stopped.

    python app_hitl.py

Why this file is separate from app.py: the interaction model is different.
app.py streams every prospect straight through; this one is stop-and-wait.
Both use the same pipeline underneath, and app.py is untouched.

The Gradio detail worth understanding: handlers are request/response, so a
paused graph has to survive BETWEEN clicks. Everything about the run - the
compiled graph, the remaining queue, the paused thread_id, results so far -
lives in a single `gr.State` dict that each handler receives and returns.
"""

import os

import gradio as gr

import config
import graph as graph_mod
import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient

OLLAMA_LABEL = "Ollama (free, local)"
ANTHROPIC_LABEL = "Anthropic (Claude)"
HEADERS = ["Company", "Tier", "Lead score", "Site quality", "URL", "Status"]


def rows_from(results):
    """results: list of (assessment, status) -> table rows, ranked."""
    ranked = sorted(results, key=lambda r: r[0]["lead_score"], reverse=True)
    return [[a["company_name"], a["tier"], a["lead_score"],
             a["website_quality_score"], a["url"], status]
            for a, status in ranked]


def review_markdown(p):
    """Render the interrupt payload as a review card."""
    obs = "\n".join(f"- {o}" for o in (p.get("observations") or [])) or "- (none)"
    return (
        f"### Review required: {p.get('company')}\n"
        f"**{p.get('tier', '').upper()}** · lead score **{p.get('lead_score')}** · "
        f"[{p.get('url')}]({p.get('url')})\n\n"
        f"{p.get('summary', '')}\n\n"
        f"**What the agent found:**\n{obs}\n\n"
        f"Draft an outreach email for this lead?"
    )


def _blank_session():
    return {"graph": None, "queue": [], "thread_id": None,
            "results": [], "emails": {}, "threshold": config.DRAFT_THRESHOLD}


def _outputs(session, status, card=None, show_buttons=False, csv=None):
    """Every handler returns this same tuple, in the order of `outputs=`."""
    choices = list(session["emails"].keys())
    return (
        status,
        gr.update(value=card or "", visible=bool(card)),
        gr.update(visible=show_buttons),
        gr.update(visible=show_buttons),
        rows_from(session["results"]),
        gr.update(choices=choices, value=choices[0] if choices else None),
        csv,
        session,
    )


def _advance(session):
    """Process prospects until one PAUSES for review, or the queue empties."""
    while session["queue"]:
        url, name = session["queue"].pop(0)
        thread_id = f"lead-{len(session['results'])}-{abs(hash(url)) % 10000}"

        try:
            state = graph_mod.run_one(session["graph"], url, name,
                                      session["threshold"], thread_id=thread_id)
        except Exception as e:
            session["results"].append(
                ({**pipeline.schemas.normalize({}, url, url),
                  "summary": f"ERROR: {e}"}, "error"))
            continue

        pending = graph_mod.pending_review(state)
        if pending:
            session["thread_id"] = thread_id
            remaining = len(session["queue"])
            return _outputs(session,
                            f"Paused for your review · {remaining} prospect(s) left",
                            card=review_markdown(pending), show_buttons=True)

        # finished without needing a human (skipped / review tier / error)
        session["results"].append((state["assessment"], state.get("status", "skipped")))

    return _finish(session)


def _finish(session):
    """Queue empty - write outputs and report."""
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "results.csv")
    pipeline._write_csv([a for a, _ in session["results"]], csv_path)

    drafted = len(session["emails"])
    total = len(session["results"])
    return _outputs(session,
                    f"Done. {total} prospects · {drafted} approved and drafted.",
                    card=None, show_buttons=False, csv=csv_path)


def start(urls_text, backend_label, model, threshold, limit):
    """Build the graph and begin the review session."""
    session = _blank_session()
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if limit and int(limit) > 0:
        urls = urls[: int(limit)]
    if not urls:
        return _outputs(session, "Paste at least one URL above.")

    model = (model or "").strip() or None
    try:
        client = (AnthropicClient(model=model) if backend_label == ANTHROPIC_LABEL
                  else OllamaClient(model=model))
        # require_approval=True -> the graph pauses before drafting
        session["graph"] = graph_mod.build_graph(client, require_approval=True)
    except Exception as e:
        return _outputs(session, f"Could not start backend: {e}")

    session["queue"] = [(u, "") for u in urls]
    session["threshold"] = int(threshold)
    return _advance(session)


def decide(session, decision):
    """Resume the paused graph with the human's decision, then keep going."""
    if not session or not session.get("thread_id"):
        return _outputs(session or _blank_session(), "Nothing is awaiting review.")

    state = graph_mod.resume_one(session["graph"], session["thread_id"], decision)
    a = state["assessment"]
    session["results"].append((a, state.get("status", "?")))
    if state.get("email"):
        session["emails"][a["company_name"]] = state["email"]
    session["thread_id"] = None
    return _advance(session)


def show_email(company, session):
    emails = (session or {}).get("emails", {})
    if not company or company not in emails:
        return "_Approve a lead to see its drafted email._"
    return f"```\n{emails[company]}\n```"


with gr.Blocks(title="Lead-Gen Agent · Review") as demo:
    gr.Markdown(f"# Lead-Gen Agent — human in the loop\n"
                f"The agent qualifies each prospect and pauses for your approval "
                f"before drafting outreach for **{config.COMPANY_NAME}**.")

    session_state = gr.State(_blank_session())

    with gr.Row():
        with gr.Column(scale=2):
            urls_in = gr.Textbox(lines=8, label="Prospect URLs (one per line)",
                                 placeholder="https://a-local-bakery.com")
        with gr.Column(scale=1):
            backend_in = gr.Dropdown([OLLAMA_LABEL, ANTHROPIC_LABEL],
                                     value=OLLAMA_LABEL, label="Backend")
            model_in = gr.Textbox(label="Model (blank = config default)", value="")
            threshold_in = gr.Slider(0, 100, value=config.DRAFT_THRESHOLD, step=5,
                                     label="Ask for approval if lead score ≥")
            limit_in = gr.Number(value=0, label="Limit (0 = all)", precision=0)

    start_btn = gr.Button("Start Review Session", variant="primary")
    status_out = gr.Markdown()

    review_card = gr.Markdown(visible=False)
    with gr.Row():
        approve_btn = gr.Button("Approve — draft the email", variant="primary", visible=False)
        reject_btn = gr.Button("Reject — skip this lead", visible=False)

    results_out = gr.Dataframe(headers=HEADERS, label="Processed leads", wrap=True)
    csv_out = gr.File(label="Download results.csv")

    email_pick = gr.Dropdown(label="View drafted email for:", choices=[])
    email_out = gr.Markdown()

    OUTPUTS = [status_out, review_card, approve_btn, reject_btn,
               results_out, email_pick, csv_out, session_state]

    start_btn.click(start,
                    inputs=[urls_in, backend_in, model_in, threshold_in, limit_in],
                    outputs=OUTPUTS)
    approve_btn.click(lambda s: decide(s, "approve"), inputs=[session_state], outputs=OUTPUTS)
    reject_btn.click(lambda s: decide(s, "reject"), inputs=[session_state], outputs=OUTPUTS)
    email_pick.change(show_email, inputs=[email_pick, session_state], outputs=email_out)


if __name__ == "__main__":
    demo.launch()