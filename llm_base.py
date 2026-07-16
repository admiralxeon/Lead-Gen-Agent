"""
Shared LLM interface.

Both backends implement two methods:
  - complete(system, user, max_tokens, stage) -> str          (free text, e.g. the email)
  - complete_json(system, user, max_tokens, stage) -> dict     (structured assessment)

extract_json() makes the JSON parsing resilient regardless of which model
produced it (handles markdown fences, leading prose, and prefill).

`stage` is a free-text label ("qualify" / "draft" / "unknown") identifying
which pipeline step made the call — it exists purely for observability, so
logged calls can be grouped by where in the pipeline they happened. It's
optional and defaults to "unknown" so existing call sites don't break.

_log() is the single choke point every real API call passes through before
returning to its caller. It writes one JSON line per call to
outputs/llm_calls.jsonl — latency, token counts, and computed cost. This is
deliberately NOT a decorator: token counts have to be pulled out of each
SDK's response object, which only the backend itself has access to, so each
backend calls _log() explicitly right after it gets a response.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("outputs/llm_calls.jsonl")

# USD per 1M tokens. Verified against Anthropic's pricing page (Jul 2026).
# Ollama has no entry -> _cost_usd() falls through to 0.0, since it's local
# and free. Update this table if pricing changes or you switch models -
# it's intentionally a small explicit table, not fetched at runtime.
PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Sonnet 5 is at introductory pricing ($2/$10) through Aug 31, 2026,
    # then reverts to standard ($3/$15). Update if you switch to it.
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    for key, rates in PRICING.items():
        if key in (model or ""):
            return (
                input_tokens * rates["input"] + output_tokens * rates["output"]
            ) / 1_000_000
    return 0.0  # unknown/local model (e.g. Ollama) -> not priced


def extract_json(text: str) -> dict:
    """Pull the first balanced {...} object out of a model response."""
    if not text:
        raise ValueError("Empty response, no JSON to parse")

    # Strip common markdown fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first balanced brace block
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in response: {text[:200]!r}")


class LLMClient(ABC):
    name = "base"

    @abstractmethod
    def complete(
        self, system: str, user: str, max_tokens: int, stage: str = "unknown"
    ) -> str: ...

    @abstractmethod
    def complete_json(
        self, system: str, user: str, max_tokens: int, stage: str = "unknown"
    ) -> dict: ...

    def _log(
        self,
        *,
        stage: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        status: str,
        error: str = None,
    ) -> None:
        """Append one observability record. Never raises - a logging
        failure should never take down a real pipeline run."""
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": self.name,
                "model": getattr(self, "model", "unknown"),
                "stage": stage,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(
                    _cost_usd(getattr(self, "model", ""), input_tokens, output_tokens),
                    6,
                ),
                "status": status,
                "error": error,
            }
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # observability must never break the actual pipeline
