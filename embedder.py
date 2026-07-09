"""
Embedding backend for the RAG layer.

Mirrors the LLM-backend pattern: embeddings come from a swappable provider,
so the retrieval layer never hard-codes where vectors come from.

Default is Ollama's local `nomic-embed-text` - free, runs locally, and needs
NO PyTorch, which sidesteps the sentence-transformers/torch wheel problem on
Python 3.14 / Windows. A hosted embedder (e.g. Voyage) can be added later as
another backend without touching the retrieval code.
"""

import math

from openai import OpenAI

import config


class Embedder:
    """Turns text into vectors via Ollama's OpenAI-compatible endpoint."""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.EMBED_MODEL
        # api_key is required by the SDK but ignored by Ollama
        self.client = OpenAI(
            base_url=base_url or config.OLLAMA_BASE_URL,
            api_key="ollama",
        )

    def embed(self, texts):
        """Embed a string or list of strings -> list of vectors (list[float]).

        Order of the returned vectors matches the order of `texts`.
        """
        if isinstance(texts, str):
            texts = [texts]
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # Sort by index to be safe, then return just the vectors in input order.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]

    def embed_one(self, text: str):
        return self.embed([text])[0]


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two equal-length vectors (1.0 = identical)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
