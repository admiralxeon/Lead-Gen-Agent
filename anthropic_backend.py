"""
Anthropic backend  ===  THE ANTHROPIC VERSION  ===

Uses the Anthropic SDK. Two course-specific details applied here:
  - system prompt is a top-level `system=` param (never inside messages)
  - structured JSON via the assistant-prefill trick: we seed the assistant
    turn with "{" so the model is forced to continue raw JSON (Anthropic has
    no response_format=json_object mode).
"""

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
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
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
        text = "{" + _first_text(resp)
        return extract_json(text)
