from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
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

    # bookkeeping
    status: str  # "qualified" | "drafted" | "skipped" | "error"


# ---------------------------------------------------------------- STEP 2: NODES
def qualify_node(state: LeadState, client) -> dict:
    """Scrape + analyze one prospect. Wraps the existing assess_prospect()."""
    try:
        assessment = pipeline.assess_prospect(
            client, state.get("name", ""), state["url"]
        )
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


# ------------------------------------------------------- STEP 3: CONDITIONAL EDGE
def should_draft(state: LeadState) -> str:
    """The routing function - this IS the `if` from pipeline.py, promoted to a
    first-class part of the graph. Returns the NAME of the next node.
    """
    a = state.get("assessment") or {}
    threshold = state.get("threshold", config.DRAFT_THRESHOLD)

    if state.get("status") == "error":
        return "skip"
    if a.get("tier") == "review":  # empty / JS-rendered page
        return "skip"
    if a.get("lead_score", 0) < threshold:  # not a strong enough lead
        return "skip"
    return "draft"


# ------------------------------------------------------------------ BUILD GRAPH
def build_graph(client, checkpointer=None):
    """Wire nodes + edges into a runnable graph.

    `client` is one of your existing LLM backends (Ollama or Anthropic) - it's
    closed over by the nodes, so the graph stays backend-agnostic just like the
    rest of the project.
    """
    g = StateGraph(LeadState)

    # nodes (lambdas inject the client without making it part of the state)
    g.add_node("qualify", lambda s: qualify_node(s, client))
    g.add_node("draft", lambda s: draft_node(s, client))
    g.add_node("skip", skip_node)

    # edges
    g.add_edge(START, "qualify")
    g.add_conditional_edges(
        "qualify",
        should_draft,
        {"draft": "draft", "skip": "skip"},  # routing map: return value -> node
    )
    g.add_edge("draft", END)
    g.add_edge("skip", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["draft"])


def run_one(
    graph, url: str, name: str = "", threshold: int = None, thread_id: str = None
) -> LeadState:
    """Run a single prospect through the graph.

    `thread_id` identifies this prospect's persisted state to the checkpointer.
    Required whenever the graph was compiled with a checkpointer — without it,
    LangGraph has no key to save/load state against. Defaults to the URL, since
    each prospect is naturally its own independent thread.
    """
    config_dict = {"configurable": {"thread_id": thread_id or url}}
    return graph.invoke(
        {
            "url": url,
            "name": name,
            "threshold": threshold if threshold is not None else config.DRAFT_THRESHOLD,
        },
        config_dict,
    )
