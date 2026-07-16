"""
Gradio dashboard for the lead-gen agent.

Routes prospects through the LangGraph graph (graph.py) instead of calling
pipeline.assess_prospect / draft_email directly. This gets the dashboard the
human-in-the-loop pause: hot/warm leads stop before drafting and wait in the
"Pending review" queue below until you Approve or Reject them.

Run it with:

    python app.py

then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import os

import gradio as gr
from langgraph.checkpoint.memory import MemorySaver

import config
import graph as graph_mod
import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient
from dotenv import load_dotenv

load_dotenv(override=True)

OLLAMA_LABEL = "Ollama (free, local)"
ANTHROPIC_LABEL = "Anthropic (Claude)"

HEADERS = ["Company", "Tier", "Lead score", "Site quality", "URL", "Summary"]


def make_client(backend_label: str, model: str):
    """Map the dropdown choice to one of our two interchangeable backends."""
    model = model.strip() or None
    if backend_label == ANTHROPIC_LABEL:
        return AnthropicClient(model=model)
    return OllamaClient(model=model)


def rows_from(assessments):
    """Turn assessment dicts into table rows, ranked by lead score."""
    ranked = sorted(assessments, key=lambda a: a["lead_score"], reverse=True)
    return [
        [
            a["company_name"],
            a["tier"],
            a["lead_score"],
            a["website_quality_score"],
            a["url"],
            a["summary"],
        ]
        for a in ranked
    ]


def find_leads(urls_text, backend_label, model, threshold, limit):
    """
    Generator: yields progressive UI updates as each prospect is run through
    the graph. Leads that route toward "draft" pause there (interrupt_before)
    and land in the pending-review queue instead of getting an email
    immediately; leads that route to "skip" complete normally in this pass.

    Yields a tuple matching the `outputs=` list on the button click:
        (status, results_table, email_dropdown, csv_file, emails_state,
         pending_dropdown, pending_state, graph_state, assessments_state)
    """
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if limit and int(limit) > 0:
        urls = urls[: int(limit)]

    empty_tail = (gr.update(choices=[]), {}, gr.update(choices=[]), {}, None, [])

    if not urls:
        yield "Paste at least one URL above.", [], *empty_tail
        return

    try:
        client = make_client(backend_label, model)
    except Exception as e:
        yield f"Could not start backend: {e}", [], *empty_tail
        return

    # Fresh graph + checkpointer per run. Threads (keyed by URL) live for the
    # lifetime of this graph object, which we hand back via graph_state so the
    # Approve/Reject handlers can resume the exact same paused threads later.
    checkpointer = MemorySaver()
    app_graph = graph_mod.build_graph(client, checkpointer=checkpointer)

    assessments, emails, pending = [], {}, {}

    for i, url in enumerate(urls, 1):
        status = f"Processing {i}/{len(urls)}: {url}"
        yield (
            status,
            rows_from(assessments),
            gr.update(),
            None,
            emails,
            gr.update(),
            pending,
            app_graph,
            assessments,
        )

        state = graph_mod.run_one(app_graph, url, "", threshold, thread_id=url)
        a = state["assessment"]
        assessments.append(a)

        snapshot = app_graph.get_state({"configurable": {"thread_id": url}})
        if snapshot.next:  # paused before "draft" -> needs a human decision
            pending[a["company_name"]] = {"thread_id": url, "assessment": a}
        elif state.get("email"):  # routed straight to draft+END in one call
            emails[a["company_name"]] = state["email"]

        yield (
            status,
            rows_from(assessments),
            gr.update(),
            None,
            emails,
            gr.update(choices=list(pending.keys())),
            pending,
            app_graph,
            assessments,
        )

    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "results.csv")
    pipeline._write_csv(assessments, csv_path)

    hot = sum(1 for a in assessments if a["tier"] == "hot")
    review = sum(1 for a in assessments if a["tier"] == "review")
    summary = (
        f"Done. {len(assessments)} prospects · {hot} hot · "
        f"{len(emails)} emails drafted · {len(pending)} awaiting your "
        f"approval below · {review} need manual review."
    )

    email_choices = list(emails.keys())
    yield (
        summary,
        rows_from(assessments),
        gr.update(
            choices=email_choices, value=email_choices[0] if email_choices else None
        ),
        csv_path,
        emails,
        gr.update(choices=list(pending.keys()), value=None),
        pending,
        app_graph,
        assessments,
    )


def show_email(company, emails):
    if not company or company not in emails:
        return "_Select a lead above to see its drafted email._"
    return f"```\n{emails[company]}\n```"


def _resolve_pending(company, pending, app_graph):
    """Shared guard for approve/reject: bail out cleanly if there's nothing
    to act on (e.g. the dropdown is empty or the graph hasn't been built)."""
    if not company or company not in pending or app_graph is None:
        return None
    return pending[company]["thread_id"]


def approve_lead(company, pending, app_graph, emails, assessments):
    thread_id = _resolve_pending(company, pending, app_graph)
    if thread_id is None:
        return (
            gr.update(),
            rows_from(assessments),
            gr.update(),
            emails,
            gr.update(),
            pending,
        )

    config_dict = {"configurable": {"thread_id": thread_id}}
    final_state = app_graph.invoke(None, config_dict)  # None = continue, don't restart

    if final_state.get("email"):
        emails[company] = final_state["email"]

    pending = {k: v for k, v in pending.items() if k != company}
    status = f"Approved '{company}' — email drafted."
    email_choices = list(emails.keys())

    return (
        status,
        rows_from(assessments),
        gr.update(choices=email_choices, value=company),
        emails,
        gr.update(choices=list(pending.keys()), value=None),
        pending,
    )


def reject_lead(company, pending, app_graph, assessments):
    thread_id = _resolve_pending(company, pending, app_graph)
    if thread_id is None:
        return gr.update(), rows_from(assessments), gr.update(), pending

    config_dict = {"configurable": {"thread_id": thread_id}}
    # Simulate skip_node's output so "rejected" means exactly what "skipped"
    # already means everywhere else in the graph — one definition, not two.
    app_graph.update_state(
        config_dict, {"status": "skipped", "email": None}, as_node="skip"
    )
    app_graph.invoke(None, config_dict)  # follow skip -> END

    pending = {k: v for k, v in pending.items() if k != company}
    status = f"Rejected '{company}' — no email drafted."

    return (
        status,
        rows_from(assessments),
        gr.update(choices=list(pending.keys()), value=None),
        pending,
    )


with gr.Blocks(title="Lead-Gen Agent") as demo:
    gr.Markdown(
        f"# Lead-Gen Agent\nFinds & qualifies leads for **{config.COMPANY_NAME}**."
    )

    emails_state = gr.State({})
    pending_state = gr.State({})
    graph_state = gr.State(None)
    assessments_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            urls_in = gr.Textbox(
                lines=8,
                label="Prospect URLs (one per line)",
                placeholder="https://a-local-bakery.com\nhttps://a-plumber.com",
            )
        with gr.Column(scale=1):
            backend_in = gr.Dropdown(
                [OLLAMA_LABEL, ANTHROPIC_LABEL],
                value=OLLAMA_LABEL,
                label="Backend",
            )
            model_in = gr.Textbox(label="Model (blank = config default)", value="")
            threshold_in = gr.Slider(
                0,
                100,
                value=config.DRAFT_THRESHOLD,
                step=5,
                label="Draft email if lead score ≥",
            )
            limit_in = gr.Number(value=0, label="Limit (0 = all)", precision=0)

    run_btn = gr.Button("Find Leads", variant="primary")
    status_out = gr.Markdown()
    results_out = gr.Dataframe(headers=HEADERS, label="Leads (ranked)", wrap=True)
    csv_out = gr.File(label="Download results.csv")

    with gr.Row():
        email_pick = gr.Dropdown(label="View drafted email for:", choices=[])
    email_out = gr.Markdown()

    gr.Markdown(
        "## Pending review\nHot/warm leads pause here before an email is drafted."
    )
    with gr.Row():
        pending_pick = gr.Dropdown(label="Awaiting your decision:", choices=[])
        approve_btn = gr.Button("Approve → draft email")
        reject_btn = gr.Button("Reject → skip")

    run_btn.click(
        find_leads,
        inputs=[urls_in, backend_in, model_in, threshold_in, limit_in],
        outputs=[
            status_out,
            results_out,
            email_pick,
            csv_out,
            emails_state,
            pending_pick,
            pending_state,
            graph_state,
            assessments_state,
        ],
    )
    email_pick.change(show_email, inputs=[email_pick, emails_state], outputs=email_out)

    approve_btn.click(
        approve_lead,
        inputs=[
            pending_pick,
            pending_state,
            graph_state,
            emails_state,
            assessments_state,
        ],
        outputs=[
            status_out,
            results_out,
            email_pick,
            emails_state,
            pending_pick,
            pending_state,
        ],
    )
    reject_btn.click(
        reject_lead,
        inputs=[pending_pick, pending_state, graph_state, assessments_state],
        outputs=[status_out, results_out, pending_pick, pending_state],
    )


if __name__ == "__main__":
    demo.launch()
