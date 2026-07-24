"""
Prospect website scraper.

Mirrors the Week 1 Day 1 pattern: fetch a page, strip the noise, return
clean-ish text. Kept deliberately simple and provider-agnostic.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LeadGenBot/1.0; +https://example.com/bot)"
    )
}

# If a fetch SUCCEEDS but yields less text than this, the page is almost
# certainly JavaScript-rendered (or a near-empty shell). We flag it for manual
# review instead of letting the LLM score an empty page as a "cold" lead.
MIN_CONTENT_CHARS = 200


class Website:
    """A scraped prospect website."""

    def __init__(self, url: str):
        self.url = url
        self.title = ""
        self.text = ""
        self.error = None
        self._load()

    def _load(self):
        try:
            resp = requests.get(self.url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:  # network / DNS / timeout / HTTP errors
            self.error = str(e)
            return

        soup = BeautifulSoup(resp.content, "html.parser")
        self.title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

        # Remove things that are noise for analysis
        for tag in soup(["script", "style", "img", "input", "svg", "noscript"]):
            tag.decompose()

        body = soup.body
        self.text = body.get_text(separator="\n", strip=True) if body else ""

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def looks_empty(self) -> bool:
        """True when the fetch worked but there's basically no text to analyze
        (the tell-tale sign of a JS-rendered single-page app)."""
        return self.ok and len(self.text.strip()) < MIN_CONTENT_CHARS

    def describe(self, char_limit: int) -> str:
        """Compact representation fed to the LLM."""
        snippet = self.text[:char_limit]
        return f"URL: {self.url}\nPage title: {self.title}\n\nPage text:\n{snippet}"