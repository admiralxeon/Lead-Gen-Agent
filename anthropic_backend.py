"""
Anthropic backend  ===  THE ANTHROPIC VERSION  ===

Uses the Anthropic SDK. Two course-specific details applied here:
  - system prompt is a top-level `system=` param (never inside messages)
  - structured JSON via the assistant-prefill trick: we seed the assistant
    turn with "{" so the model is forced to continue raw JSON (Anthropic has
    no response_format=json_object mode).

Every call is timed and logged via LLMClient._log() using real usage counts
from resp.usage (Anthropic always returns this) - see llm_base.py.
"""

import time

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

    def complete(
        self, system: str, user: str, max_tokens: int, stage: str = "unknown"
    ) -> str:
        start = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            self._log(
                stage=stage,
                latency_ms=(time.perf_counter() - start) * 1000,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                status="success",
            )
            return _first_text(resp).strip()
        except Exception as e:
            self._log(
                stage=stage,
                latency_ms=(time.perf_counter() - start) * 1000,
                input_tokens=0,
                output_tokens=0,
                status="error",
                error=str(e),
            )
            raise

    def complete_json(
        self, system: str, user: str, max_tokens: int, stage: str = "unknown"
    ) -> dict:
        start = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "{"},  # prefill -> forces JSON
                ],
            )
        except Exception as e:
            self._log(
                stage=stage,
                latency_ms=(time.perf_counter() - start) * 1000,
                input_tokens=0,
                output_tokens=0,
                status="error",
                error=str(e),
            )
            raise

        # The API call itself succeeded - log it as such regardless of what
        # happens next. A JSON parse failure below is a separate, client-side
        # concern: real tokens were spent either way, and re-logging this
        # same call as "error" would double-count it in the observability data.
        self._log(
            stage=stage,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            status="success",
        )
        text = "{" + _first_text(resp)
        return extract_json(text)
