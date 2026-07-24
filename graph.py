r"""
LangGraph version of the lead-gen pipeline.

Same work as pipeline.py, restructured as a STATE MACHINE instead of a for-loop.
This file is additive: pipeline.py, the CLI runners, and the Gradio app all keep
working exactly as before.

------------------------------------------------------------------------------
STEP 1 - STATE
------------------------------------------------------------------------------
In a for-loop, intermediate values live in local variables. In a graph, they
live in an explicit STATE object that every node receives and updates.

LeadState below is that object. Each node takes the state, does one job, and
returns a dict of ONLY the keys it changed - LangGraph merges that back in.
That "return only what you changed" contract is the core mental model.

------------------------------------------------------------------------------
STEP 2 - NODES
------------------------------------------------------------------------------
A node is just a function: state -> partial state update. Our nodes are thin
wrappers around the functions you ALREADY have (assess_prospect, draft_email),
so no business logic is rewritten - only the control flow changes.

------------------------------------------------------------------------------
STEP 3 - EDGES
------------------------------------------------------------------------------
Edges connect nodes. The interesting one is the CONDITIONAL edge, which replaces
this line from pipeline.py:

    if a["lead_score"] >= threshold and ...:
        draft_email(...)

In the graph, that `if` becomes a routing function (should_draft) that decides
which node runs next. The decision becomes part of the architecture rather than
being buried inside a loop body.

    START -> qualify -> [should_draft?] -> draft -> END
                              \-------------------> END
"""

from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import config
import pipeline


# ---------------------------------------------------------------- STEP 1: STATE
class LeadState(TypedDict, total=False):
    """Everything that flows through the graph for ONE prospect.

    `total=False` means keys may be absent early on (e.g. `email` doesn't exist
    until the draft node runs).
    """
    # inputs
    url: str
    name: str
    threshold: int

    # produced by the qualify node
    assessment: dict

    # produced by the draft node
    email: Optional[str]

    # set by the human reviewer (STEP 5)
    decision: str   # "approve" | "reject"

    # bookkeeping
    status: str   # "qualified" | "drafted" | "skipped" | "rejected" | "error"


# ---------------------------------------------------------------- STEP 2: NODES
def qualify_node(state: LeadState, client) -> dict:
    """Scrape + analyze one prospect. Wraps the existing assess_prospect()."""
    try:
        assessment = pipeline.assess_prospect(client, state.get("name", ""), state["url"])
        return {"assessment": assessment, "status": "qualified"}
    except Exception as e:
        return {
            "assessment": {
                **pipeline.schemas.normalize({}, state["url"], state["url"]),
                "summary": f"ERROR: {e}",
                "tier": "cold",
                "lead_score": 0,
            },
            "status": "error",
        }


def draft_node(state: LeadState, client) -> dict:
    """Draft a grounded outreach email. Wraps the existing draft_email()
    (which already pulls RAG context internally)."""
    try:
        email = pipeline.draft_email(client, state["assessment"])
        return {"email": email, "status": "drafted"}
    except Exception as e:
        return {"email": f"(draft failed: {e})", "status": "error"}


def skip_node(state: LeadState) -> dict:
    """Terminal path for leads that don't qualify. Exists so 'skipped' is an
    explicit outcome in the graph rather than an implicit fall-through."""
    return {"status": "skipped", "email": None}


# ------------------------------------------------ STEP 5: HUMAN-IN-THE-LOOP
def human_review_node(state: LeadState) -> dict:
    """Pause the graph and wait for a human decision before drafting.

    THIS is the thing a for-loop can't do cleanly. interrupt() halts execution
    right here and hands the payload back to whoever invoked the graph. The
    run does NOT block a thread - the state is checkpointed and the process can
    exit. Later, resuming with Command(resume="approve"/"reject") re-enters
    this node, and interrupt() returns that value instead of pausing again.

    Requires a checkpointer, because the paused state has to live somewhere.
    """
    a = state.get("assessment") or {}
    decision = interrupt({
        "question": "Draft an outreach email for this lead?",
        "company": a.get("company_name"),
        "url": a.get("url"),
        "tier": a.get("tier"),
        "lead_score": a.get("lead_score"),
        "summary": a.get("summary"),
        "observations": a.get("observations", []),
    })
    # Anything other than an explicit approval is treated as a rejection.
    decision = str(decision).strip().lower()
    return {"decision": "approve" if decision in ("approve", "yes", "y", "true") else "reject"}


def after_review(state: LeadState) -> str:
    """Route on the human's decision."""
    return "draft" if state.get("decision") == "approve" else "reject"


def reject_node(state: LeadState) -> dict:
    """Terminal path for leads a human declined."""
    return {"status": "rejected", "email": None}


# ------------------------------------------------------- STEP 3: CONDITIONAL EDGE
def should_draft(state: LeadState) -> str:
    """The routing function - this IS the `if` from pipeline.py, promoted to a
    first-class part of the graph. Returns the NAME of the next node.
    """
    a = state.get("assessment") or {}
    threshold = state.get("threshold", config.DRAFT_THRESHOLD)

    if state.get("status") == "error":
        return "skip"
    if a.get("tier") == "review":          # empty / JS-rendered page
        return "skip"
    if a.get("lead_score", 0) < threshold: # not a strong enough lead
        return "skip"
    return "draft"


# ------------------------------------------------------------------ BUILD GRAPH
def build_graph(client, require_approval=False, checkpointer=None):
    """Wire nodes + edges into a runnable graph.

    `client` is one of your existing LLM backends (Ollama or Anthropic) - it's
    closed over by the nodes, so the graph stays backend-agnostic just like the
    rest of the project.

    require_approval=False (default):
        START -> qualify -> [should_draft?] -> draft -> END
                                  \\----------> skip  -> END

    require_approval=True (STEP 5):
        START -> qualify -> [should_draft?] -> human_review --(interrupt)--
                                  \\----------> skip -> END
        ...resume--> [after_review?] -> draft  -> END
                            \\--------> reject -> END

    A checkpointer is REQUIRED for approval mode (the paused state has to be
    stored somewhere), so we default to an in-memory one.
    """
    g = StateGraph(LeadState)

    # nodes (lambdas inject the client without making it part of the state)
    g.add_node("qualify", lambda s: qualify_node(s, client))
    g.add_node("draft", lambda s: draft_node(s, client))
    g.add_node("skip", skip_node)

    g.add_edge(START, "qualify")

    if require_approval:
        g.add_node("human_review", human_review_node)
        g.add_node("reject", reject_node)
        # qualifying leads go to a human instead of straight to drafting
        g.add_conditional_edges(
            "qualify", should_draft,
            {"draft": "human_review", "skip": "skip"},
        )
        g.add_conditional_edges(
            "human_review", after_review,
            {"draft": "draft", "reject": "reject"},
        )
        g.add_edge("reject", END)
        if checkpointer is None:
            checkpointer = MemorySaver()
    else:
        g.add_conditional_edges(
            "qualify", should_draft,
            {"draft": "draft", "skip": "skip"},  # routing map: return value -> node
        )

    g.add_edge("draft", END)
    g.add_edge("skip", END)

    return g.compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> dict:
    """A checkpointed run is identified by a thread_id. Every resume for the
    same prospect must reuse the SAME id, or LangGraph won't find its state."""
    return {"configurable": {"thread_id": thread_id}}


def run_one(graph, url: str, name: str = "", threshold: int = None,
            thread_id: str = None) -> LeadState:
    """Run a single prospect through the graph.

    With approval enabled, the returned state contains an "__interrupt__" key
    instead of a finished result - that means the graph is PAUSED. Use
    pending_review() to read it, then resume_one() to continue.
    """
    payload = {
        "url": url,
        "name": name,
        "threshold": threshold if threshold is not None else config.DRAFT_THRESHOLD,
    }
    cfg = thread_config(thread_id) if thread_id else None
    return graph.invoke(payload, config=cfg) if cfg else graph.invoke(payload)


def pending_review(state) -> dict:
    """Return the interrupt payload if the graph paused, else None."""
    interrupts = (state or {}).get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def resume_one(graph, thread_id: str, decision: str) -> LeadState:
    """Resume a paused graph with the human's decision ("approve"/"reject").

    Command(resume=...) makes the earlier interrupt() call RETURN that value,
    so execution continues from exactly where it stopped - no re-scraping and
    no re-running the qualification LLM call.
    """
    return graph.invoke(Command(resume=decision), config=thread_config(thread_id))