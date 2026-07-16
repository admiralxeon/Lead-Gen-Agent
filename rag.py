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


_INDEX = None
_INDEX_TRIED = False


def _make_store():
    """Pick the vector store backend from config.

    Both stores expose build()/retrieve(), so nothing downstream changes.
    Chroma is imported lazily so the project still runs if chromadb isn't
    installed - we fall back to the simple store instead of crashing.
    """
    if getattr(config, "VECTOR_STORE", "simple").lower() == "chroma":
        try:
            from chroma_store import ChromaStore

            return ChromaStore()
        except ImportError:
            print(
                "[RAG] chromadb not installed - falling back to the simple store. "
                "(pip install chromadb)"
            )
    return RagIndex()


def get_index():
    global _INDEX, _INDEX_TRIED
    if _INDEX is not None:
        return _INDEX
    if _INDEX_TRIED:
        return None  # already failed once; don't retry every prospect
    _INDEX_TRIED = True
    try:
        _INDEX = _make_store().build()
        return _INDEX
    except Exception as e:
        print(f"[RAG disabled] could not build knowledge index: {e}")
        print(
            "[RAG disabled] drafting without grounding. "
            "Is Ollama running with nomic-embed-text pulled?"
        )
        return None


def context_for(assessment, k=None):
    """Retrieve knowledge-base context relevant to a lead's problems.

    Returns a formatted context string, or "" if RAG is off/unavailable.
    """
    if not config.USE_RAG:
        return ""
    index = get_index()
    if index is None:
        return ""
    query = " ".join(
        assessment.get("observations", []) + assessment.get("opportunities", [])
    ).strip()
    if not query:
        query = assessment.get("summary", "")
    if not query:
        return ""
    hits = index.retrieve(query, k=k)
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
