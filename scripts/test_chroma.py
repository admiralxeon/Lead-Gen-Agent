"""
Step 3 smoke test: Chroma vector store.

Runs the SAME queries through both stores - the hand-rolled cosine search and
Chroma - and prints them side by side. They should return the same chunks:
that's the point. Chroma isn't smarter, it just handles persistence, indexing
and metadata for you instead of you doing it by hand.

Prereq:  pip install chromadb
         ollama pull nomic-embed-text   (and Ollama running)
Run:     python test_chroma.py
"""

from chroma_store import ChromaStore
from rag import RagIndex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERIES = [
    "restaurant that only takes phone orders, no online ordering",
    "plumber does not show up in Google local search",
    "law firm site is hard to use on mobile phones",
]


def show(name, store):
    print(f"\n=== {name} ===")
    for q in QUERIES:
        print(f"\nQUERY: {q}")
        for hit in store.retrieve(q, k=2):
            snippet = hit["text"][:70].replace("\n", " ")
            print(f"  [{hit['score']:.3f}] ({hit['source']}) {snippet}...")


def main():
    try:
        chroma = ChromaStore().build()
    except ImportError:
        print("chromadb isn't installed. Run:  pip install chromadb")
        return
    except Exception as e:
        print(f"Could not build Chroma store: {e}")
        print("Is Ollama running with nomic-embed-text pulled?")
        return

    simple = RagIndex().build()

    show("CHROMA", chroma)
    show("SIMPLE (hand-rolled)", simple)

    print("\nIf both stores surface the same chunks, the swap is clean and the")
    print("pipeline can use either one via VECTOR_STORE in config.py.")


if __name__ == "__main__":
    main()
