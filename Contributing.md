# Contributing

Thanks for your interest! Quick context: this is a **personal learning project**. I'm using it to build hands-on experience with LLM engineering, and a lot of its value to me is that I've written and can explain every part of it. That shapes what I can merge — please read this before writing code.

## Welcome

- Bug reports (reproducible ones are gold)
- Small, focused bug fixes
- Tests — the project is under-tested
- Docs: setup on macOS/Linux, fixing anything inaccurate
- A **Playwright scraper backend** for JavaScript-rendered sites (real gap)
- Example knowledge-base files for the RAG layer
- Packaging / CI / linting config

## I'd rather build myself

The core is what I'm learning from, so I'd prefer to write it even if that's slower:

- Pipeline and orchestration (`pipeline.py`, `graph.py`)
- LLM backends and prompt design
- The RAG layer (embedding, retrieval, vector store)
- Lead scoring and tiering logic
- The MCP server

Got an idea in one of these? **Open an issue and let's talk.** I'm happy to discuss approach — I just may not merge the implementation, and I'd rather say that before you spend a weekend on it.

## Process

1. **Open an issue first** for anything beyond a trivial fix.
2. Fork, branch, and keep the PR to one logical change.
3. Make sure it runs. Say what you tested, on which OS and Python version.
4. **Don't add dependencies** without raising it first — the dependency tree is deliberately small and some packages don't build on Python 3.14.
5. Match the existing style: flat file layout, plain functions, comments explaining *why*.

## Won't be merged

- Anything committing secrets, `.env`, or generated files (`rag_index.json`, `chroma_db/`, `outputs/`)
- Large refactors or framework swaps not agreed in an issue
- Unrelated changes bundled together
- Code I can't understand well enough to maintain

## Credit

Merged contributions get credited in the README. Say so in the PR if you'd rather not be.

Questions? Open an issue — including if you think a boundary above is wrong.


