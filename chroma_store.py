import chromadb
import config
from embedder import Embedder
from rag import _fingerprint, chunk_text, load_documents


class ChromaStore:
    def __init__(
        self, knowledge_dir=None, persist_dir=None, collection_name=None, embedder=None
    ):
        self.knowledge_dir = knowledge_dir or config.KNOWLEDGE_DIR
        self.persist_dir = persist_dir or config.CHROMA_DIR
        self.collection_name = collection_name or config.CHROMA_COLLECTION
        self.embedder = embedder or Embedder()
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = None

    def build(self, force=False):
        """Load -> chunk -> embed -> upsert into Chroma.

        Re-embeds only when the knowledge content changes. The fingerprint is
        stored on the collection's metadata, so a stale collection is detected
        and rebuilt rather than silently served.
        """
        docs = load_documents(self.knowledge_dir)
        chunks = []
        for source, text in docs:
            chunks.extend(chunk_text(text, source))
        if not chunks:
            raise ValueError(f"No usable content in {self.knowledge_dir}/*.md")

        fp = _fingerprint(chunks)

        # Reuse the existing collection if the content hasn't changed.
        if not force:
            try:
                existing = self.client.get_collection(self.collection_name)
                if (existing.metadata or {}).get(
                    "fingerprint"
                ) == fp and existing.count() > 0:
                    self.collection = existing
                    print(
                        f"Chroma: loaded {existing.count()} chunks from '{self.collection_name}'."
                    )
                    return self
            except Exception:
                pass  # not found / unreadable -> fall through and rebuild

        # Content changed (or forced): rebuild the collection from scratch.
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"fingerprint": fp, "hnsw:space": "cosine"},
        )

        print(f"Chroma: embedding {len(chunks)} chunks ...")
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)  # our embedder, not Chroma's default

        self.collection.add(
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            documents=texts,
            embeddings=vectors,
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        print(
            f"Chroma: built and persisted {len(chunks)} chunks in '{self.persist_dir}'."
        )
        return self

    def retrieve(self, query, k=None):
        """Top-k most similar chunks. Same return shape as RagIndex.retrieve()."""
        k = k or config.RAG_TOP_K
        if self.collection is None:
            raise RuntimeError("Store not built - call build() first.")

        qv = self.embedder.embed_one(query)
        res = self.collection.query(
            query_embeddings=[qv],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Chroma returns cosine DISTANCE; convert to similarity so the score
            # means the same thing as it does in the simple store (1.0 = best).
            hits.append(
                {
                    "text": doc,
                    "source": (meta or {}).get("source", "?"),
                    "score": 1.0 - float(dist),
                }
            )
        return hits
