"""
Open-source backend  ===  THE OPEN-SOURCE VERSION  ===

Talks to a local Ollama server through its OpenAI-COMPATIBLE endpoint, so we
use the OpenAI SDK (not the Anthropic one). Free to run, no API key, no token
budget to worry about.

Prereqs:
  1. Install Ollama:  https://ollama.com
  2. Pull a model:    ollama pull llama3.2     (or qwen2.5 for cleaner JSON)
  3. Ollama serves automatically on http://localhost:11434
"""

from openai import OpenAI

import config
from llm_base import LLMClient, extract_json


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.OLLAMA_MODEL
        # api_key is required by the SDK but ignored by Ollama
        self.client = OpenAI(
            base_url=base_url or config.OLLAMA_BASE_URL,
            api_key="ollama",
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_json(self, system: str, user: str, max_tokens: int) -> dict:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # Ask Ollama for JSON mode if the build supports it; fall back gracefully.
        try:
            resp = self.client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
        except Exception:
            resp = self.client.chat.completions.create(**kwargs)
        return extract_json(resp.choices[0].message.content or "")
