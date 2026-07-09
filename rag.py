"""
Retrieval layer for the RAG upgrade.

Pipeline: load knowledge/*.md -> chunk -> embed each chunk once -> cache the
vectors to disk -> retrieve the top-k most relevant chunks for a query via
cosine similarity.

This is the "vector store by hand" version - no numpy, no vector DB - so the
mechanics are fully visible. Step 3 swaps the storage/search for Chroma without
changing how the pipeline calls it.
"""

import glob
import hashlib
import json
import os

import config
from embedder import Embedder, cosine_similarity


def load_documents(knowledge_dir):
    """Read every .md file in the knowledge dir -> list of (source, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(knowledge_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text, source):
    """Split on blank lines into paragraph-ish chunks.

    Chunking is a real RAG lever: chunks too big => noisy retrieval; too small
    => lost context. Paragraph-level is a sane default for short knowledge docs.
    Headings and tiny fragments (< 40 chars) are skipped.
    """
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) >= 40:
            chunks.append({"text": para, "source": source})
    return chunks


def _fingerprint(chunks):
    """Hash of all chunk text, so the cache is rebuilt only when content changes.

    Directly targets the 'stale state' trap: change a knowledge file and the
    fingerprint changes, forcing a re-embed instead of serving old vectors.
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()


class RagIndex:
    def __init__(self, knowledge_dir=None, cache_path=None, embedder=None):
        self.knowledge_dir = knowledge_dir or config.KNOWLEDGE_DIR
        self.cache_path = cache_path or config.RAG_INDEX_PATH
        self.embedder = embedder or Embedder()
        self.chunks = []  # list of {text, source}
        self.vectors = []  # list of list[float], aligned 1:1 with self.chunks

    def build(self, force=False):
        """Load -> chunk -> embed -> cache. Reuses the cache if content is unchanged."""
        docs = load_documents(self.knowledge_dir)
        chunks = []
        for source, text in docs:
            chunks.extend(chunk_text(text, source))
        if not chunks:
            raise ValueError(f"No usable content in {self.knowledge_dir}/*.md")

        fp = _fingerprint(chunks)
        if not force and self._load_cache(fp):
            print(f"RAG index: loaded {len(self.chunks)} chunks from cache.")
            return self

        print(f"RAG index: embedding {len(chunks)} chunks ...")
        self.vectors = self.embedder.embed([c["text"] for c in chunks])
        self.chunks = chunks
        self._save_cache(fp)
        print(f"RAG index: built and cached {len(chunks)} chunks.")
        return self

    def retrieve(self, query, k=None):
        """Return the top-k chunks most similar to the query, each with a score."""
        k = k or config.RAG_TOP_K
        if not self.vectors:
            raise RuntimeError("Index not built - call build() first.")
        qv = self.embedder.embed_one(query)
        scored = [
            {**chunk, "score": cosine_similarity(qv, vec)}
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:k]

    def _save_cache(self, fingerprint):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "chunks": self.chunks,
                    "vectors": self.vectors,
                },
                f,
            )

    def _load_cache(self, fingerprint):
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        if data.get("fingerprint") != fingerprint:
            return False  # knowledge changed -> force rebuild
        self.chunks = data["chunks"]
        self.vectors = data["vectors"]
        return True
