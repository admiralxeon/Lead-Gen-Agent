"""
Shared LLM interface.

Both backends implement two methods:
  - complete(system, user, max_tokens) -> str          (free text, e.g. the email)
  - complete_json(system, user, max_tokens) -> dict     (structured assessment)

extract_json() makes the JSON parsing resilient regardless of which model
produced it (handles markdown fences, leading prose, and prefill).
"""

import json
import re
from abc import ABC, abstractmethod


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
    def complete(self, system: str, user: str, max_tokens: int) -> str:
        ...

    @abstractmethod
    def complete_json(self, system: str, user: str, max_tokens: int) -> dict:
        ...
