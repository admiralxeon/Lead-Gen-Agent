"""
Gradio dashboard for the lead-gen agent.

This does NOT reimplement anything - it wraps the existing pipeline functions
(assess_prospect, draft_email) in a UI. Run it with:

    python app.py

then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import os

import gradio as gr

import config
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
        [a["company_name"], a["tier"], a["lead_score"],
         a["website_quality_score"], a["url"], a["summary"]]
        for a in ranked
    ]


def find_leads(urls_text, backend_label, model, threshold, limit):
    """
    Generator: yields progressive UI updates as each prospect is processed,
    so the table fills in live instead of freezing until the end.

    Yields a tuple matching the `outputs=` list on the button click:
        (status, results_table, email_dropdown, csv_file, emails_state)
    """
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if limit and int(limit) > 0:
        urls = urls[: int(limit)]

    if not urls:
        yield "Paste at least one URL above.", [], gr.update(choices=[]), None, {}
        return

    try:
        client = make_client(backend_label, model)
    except Exception as e:
        yield f"Could not start backend: {e}", [], gr.update(choices=[]), None, {}
        return

    assessments, emails = [], {}

    for i, url in enumerate(urls, 1):
        status = f"Processing {i}/{len(urls)}: {url}"
        yield status, rows_from(assessments), gr.update(), None, {}

        try:
            a = pipeline.assess_prospect(client, "", url)
        except Exception as e:
            a = {**pipeline.schemas.normalize({}, url, url),
                 "summary": f"ERROR: {e}", "tier": "cold", "lead_score": 0}
        assessments.append(a)

        # Draft an email only for qualifying leads (saves tokens)
        qualifies = (a["lead_score"] >= threshold
                     and a["tier"] != "review"
                     and not a["summary"].startswith(("SCRAPE FAILED", "ERROR")))
        if qualifies:
            try:
                emails[a["company_name"]] = pipeline.draft_email(client, a)
            except Exception as e:
                emails[a["company_name"]] = f"(email draft failed: {e})"

        yield status, rows_from(assessments), gr.update(), None, {}

    # Final pass: write a downloadable CSV and populate the email dropdown
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "results.csv")
    pipeline._write_csv(assessments, csv_path)

    hot = sum(1 for a in assessments if a["tier"] == "hot")
    review = sum(1 for a in assessments if a["tier"] == "review")
    summary = (f"Done. {len(assessments)} prospects · {hot} hot · "
               f"{len(emails)} emails drafted · {review} need manual review.")

    choices = list(emails.keys())
    yield (summary,
           rows_from(assessments),
           gr.update(choices=choices, value=choices[0] if choices else None),
           csv_path,
           emails)


def show_email(company, emails):
    if not company or company not in emails:
        return "_Select a lead above to see its drafted email._"
    return f"```\n{emails[company]}\n```"


with gr.Blocks(title="Lead-Gen Agent") as demo:
    gr.Markdown(f"# Lead-Gen Agent\nFinds & qualifies leads for **{config.COMPANY_NAME}**.")

    emails_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            urls_in = gr.Textbox(
                lines=8, label="Prospect URLs (one per line)",
                placeholder="https://a-local-bakery.com\nhttps://a-plumber.com",
            )
        with gr.Column(scale=1):
            backend_in = gr.Dropdown(
                [OLLAMA_LABEL, ANTHROPIC_LABEL], value=OLLAMA_LABEL,
                label="Backend",
            )
            model_in = gr.Textbox(label="Model (blank = config default)", value="")
            threshold_in = gr.Slider(
                0, 100, value=config.DRAFT_THRESHOLD, step=5,
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

    run_btn.click(
        find_leads,
        inputs=[urls_in, backend_in, model_in, threshold_in, limit_in],
        outputs=[status_out, results_out, email_pick, csv_out, emails_state],
    )
    email_pick.change(show_email, inputs=[email_pick, emails_state], outputs=email_out)


if __name__ == "__main__":
    demo.launch()
