"""
Anthropic backend  ===  THE ANTHROPIC VERSION  ===

Uses the Anthropic SDK. Two course-specific details applied here:
  - system prompt is a top-level `system=` param (never inside messages)
  - structured JSON via the assistant-prefill trick: we seed the assistant
    turn with "{" so the model is forced to continue raw JSON (Anthropic has
    no response_format=json_object mode).
"""

import os

from anthropic import Anthropic

import config
from llm_base import LLMClient, extract_json


def _first_text(response) -> str:
    """Return the first text block (skips any thinking blocks)."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str = None):
        self.model = model or config.ANTHROPIC_MODEL
        # Fail loudly and usefully if the key isn't loaded, instead of letting
        # every call throw an auth error that the pipeline swallows as score 0.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                "  - Create a .env file next to config.py containing:\n"
                "      ANTHROPIC_API_KEY=sk-ant-...\n"
                "  - and install the loader:  pip install python-dotenv\n"
                "  (config.py calls load_dotenv() on import)"
            )
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        # Populated after every call so an instrumentation wrapper can read
        # real token counts instead of estimating them.
        self.last_usage = None

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._record(resp)
        return _first_text(resp).strip()

    def complete_json(self, system: str, user: str, max_tokens: int) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": "{"},  # prefill -> forces JSON
            ],
        )
        self._record(resp)
        text = "{" + _first_text(resp)
        return extract_json(text)

    def _record(self, resp):
        u = getattr(resp, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "stop_reason": getattr(resp, "stop_reason", None),
        }