"""
Smoke test for the retrieval layer (Step 1).

Builds the index from knowledge/*.md, then runs a few realistic prospect-problem
queries and prints the top chunks retrieved. You're checking that each query
pulls back the RELEVANT case study / service, not random chunks.

Prereq:  ollama pull nomic-embed-text   (and Ollama running)
Run:     python test_rag.py
"""

from rag import RagIndex


def main():
    try:
        idx = RagIndex().build()
    except Exception as e:
        print(f"Could not build index: {e}")
        print("Is Ollama running and is `nomic-embed-text` pulled?")
        return

    queries = [
        "restaurant that only takes phone orders, no online ordering",
        "plumber does not show up in Google local search",
        "law firm site looks broken and hard to use on mobile phones",
    ]

    for q in queries:
        print(f"\nQUERY: {q}")
        for hit in idx.retrieve(q, k=2):
            snippet = hit["text"][:90].replace("\n", " ")
            print(f"  [{hit['score']:.3f}] ({hit['source']}) {snippet}...")


if __name__ == "__main__":
    main()
