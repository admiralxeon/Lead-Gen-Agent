"""
Step 0 smoke test for RAG.

Confirms that local embeddings actually work on THIS machine (Python 3.14 /
Windows) and that similarity is meaningful, BEFORE we build the retrieval
layer on top. If this passes, the risky part is behind us.

Prereq:  ollama pull nomic-embed-text
Run:     python test_embeddings.py
"""

from embedder import Embedder, cosine_similarity
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    emb = Embedder()
    print(f"Embedding model: {emb.model}\n")

    sentences = [
        "We build fast, mobile-friendly websites for small businesses.",  # A
        "Our team designs responsive sites for local shops.",  # B  ~ related to A
        "The restaurant serves wood-fired pizza on weekends.",  # C  unrelated
    ]

    try:
        vectors = emb.embed(sentences)
    except Exception as e:
        print(f"FAILED to get embeddings: {e}\n")
        print("Checklist:")
        print("  - Is Ollama running? (it serves on http://localhost:11434)")
        print("  - Did you run:  ollama pull nomic-embed-text")
        return

    dim = len(vectors[0])
    print(f"Got {len(vectors)} vectors, dimension {dim}")
    if dim == 0:
        print("Empty vectors returned - something's wrong with the model.")
        return

    sim_related = cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = cosine_similarity(vectors[0], vectors[2])
    print(f"\nsimilarity(A, B)  [related]   = {sim_related:.3f}")
    print(f"similarity(A, C)  [unrelated] = {sim_unrelated:.3f}")

    if sim_related > sim_unrelated:
        print("\nPASS - related text is closer than unrelated text.")
        print("Embeddings work. We can build the retrieval layer next.")
    else:
        print("\nWARNING - similarity ordering looks off. Try a different")
        print("embedding model (e.g. `ollama pull mxbai-embed-large`).")


if __name__ == "__main__":
    main()
