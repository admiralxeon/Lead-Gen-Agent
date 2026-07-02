"""
Lead-Gen MCP Server
===================
Exposes the lead-generation agent's capabilities over the Model Context Protocol.

Three MCP server primitives:
  - TOOLS     (actions): scrape_website, save_lead
  - RESOURCE  (read-only data): leads://all
  - PROMPT    (reusable template): qualify_lead

Run over stdio (default, for Claude Desktop):
    python leadgen_mcp_server.py

Run over HTTP (for easy local testing, no subprocess spawning):
    python leadgen_mcp_server.py --http
    -> serves at http://127.0.0.1:8000/mcp
"""

from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup
from pathlib import Path
import sqlite3
import requests
import sys

DB_PATH = Path(__file__).parent / "leads.db"

mcp = FastMCP("leadgen")


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS leads ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT, url TEXT, "
        "tier TEXT, notes TEXT)"
    )
    return conn


@mcp.tool()
def scrape_website(url: str) -> dict:
    """Fetch a prospect's website and return cleaned text for qualification.

    Flags JS-rendered / empty pages ('looks_empty') so they get routed to a
    'review' tier instead of being miscounted as cold leads.
    """
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "leadgen-bot/1.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        looks_empty = len(text) < 200
        return {
            "url": url,
            "title": (
                soup.title.string.strip() if soup.title and soup.title.string else ""
            ),
            "text": text[:4000],
            "looks_empty": looks_empty,
            "tier_hint": "review" if looks_empty else "unscored",
        }
    except Exception as e:
        return {"url": url, "error": str(e), "looks_empty": True, "tier_hint": "review"}


@mcp.tool()
def save_lead(company: str, url: str, tier: str, notes: str = "") -> str:
    """Persist a qualified lead. tier is one of: hot, warm, cold, review."""
    conn = _db()
    conn.execute(
        "INSERT INTO leads (company, url, tier, notes) VALUES (?, ?, ?, ?)",
        (company, url, tier, notes),
    )
    conn.commit()
    conn.close()
    return f"Saved '{company}' ({url}) as {tier}."


@mcp.tool()
def list_leads() -> str:
    """List every saved lead. A model-callable tool version of the leads://all resource,
    so a host can retrieve saved leads autonomously (resources are user-attached, not
    model-invoked)."""
    return all_leads()


@mcp.resource("leads://all")
def all_leads() -> str:
    """Read-only view of every saved lead (the 'read' side of MCP)."""
    conn = _db()
    rows = conn.execute(
        "SELECT company, url, tier, notes FROM leads ORDER BY id DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return "No leads saved yet."
    return "\n".join(f"{c} | {u} | {t} | {n}" for c, u, t, n in rows)


@mcp.prompt()
def qualify_lead(company: str, website_text: str) -> str:
    """A reusable qualification prompt the host can summon by name."""
    return (
        f"You are a B2B lead-qualification analyst for a digital marketing and web "
        f"development agency.\n\n"
        f"Company: {company}\n"
        f"Website text:\n{website_text}\n\n"
        f"Decide a tier (hot / warm / cold) based on whether they likely need web/marketing "
        f"services, and give a one-sentence reason. Return JSON: "
        f'{{"tier": "...", "reason": "..."}}'
    )


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")  # http://127.0.0.1:8000/mcp
    else:
        mcp.run()  # stdio (default)
