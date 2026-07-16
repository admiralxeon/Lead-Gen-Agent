"""
Open-source backend  ===  THE OPEN-SOURCE VERSION  ===

Talks to a local Ollama server through its OpenAI-COMPATIBLE endpoint, so we
use the OpenAI SDK (not the Anthropic one). Free to run, no API key, no token
budget to worry about.

Prereqs:
  1. Install Ollama:  https://ollama.com
  2. Pull a model:    ollama pull llama3.2     (or qwen2.5 for cleaner JSON)
  3. Ollama serves automatically on http://localhost:11434

Every call is timed and logged via LLMClient._log(). Unlike Anthropic,
Ollama's OpenAI-compatible endpoint doesn't reliably return usage counts
depending on version, so token counts are read defensively and default to 0
rather than assuming the field exists. cost_usd will correctly come out to
$0.00 either way, since PRICING has no entry for local models.
"""

import time

from openai import OpenAI

import config
from llm_base import LLMClient, extract_json


def _usage_tokens(resp):
    """Best-effort extraction; Ollama doesn't always populate usage."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return getattr(usage, "prompt_tokens", 0) or 0, getattr(
        usage, "completion_tokens", 0
    ) or 0


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.OLLAMA_MODEL
        # api_key is required by the SDK but ignored by Ollama
        self.client = OpenAI(
            base_url=base_url or config.OLLAMA_BASE_URL,
            api_key="ollama",
        )

    def complete(
        self, system: str, user: str, max_tokens: int, stage: str = "unknown"
    ) -> str:
        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
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

        input_tokens, output_tokens = _usage_tokens(resp)
        self._log(
            stage=stage,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="success",
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_json(
        self, system: str, user: str, max_tokens: int, stage: str = "unknown"
    ) -> dict:
        start = time.perf_counter()
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            # Ask Ollama for JSON mode if the build supports it; fall back gracefully.
            try:
                resp = self.client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            except Exception:
                resp = self.client.chat.completions.create(**kwargs)
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

        # The API call succeeded - log it as such. A JSON parse failure below
        # is separate and client-side; the call itself still happened.
        input_tokens, output_tokens = _usage_tokens(resp)
        self._log(
            stage=stage,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="success",
        )
        return extract_json(resp.choices[0].message.content or "")
