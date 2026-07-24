"""
Central configuration for the lead-generation agent.

Edit COMPANY_* to describe YOUR agency. The model + budget knobs below
are the main levers for controlling cost / quality.
"""

# Load .env FIRST, before anything reads an API key. python-dotenv does not
# load automatically - something has to call load_dotenv(). Doing it here means
# every entry point (CLI, Gradio, graph, diagnostics) gets it, because they all
# import config.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # not installed -> fall back to real env vars
    pass

import os

# --- Your agency (used in prompts so outreach is grounded in who you are) ---
COMPANY_NAME = "Nyxen Digital"
COMPANY_PITCH = (
    "We build modern, fast, mobile-friendly websites and run digital "
    "marketing (SEO, paid ads, content) for small and mid-sized businesses."
)
SENDER_NAME = "Sanjit"
SENDER_ROLE = "Growth Partner"

# --- Anthropic backend ---
# Current cost-efficient model. Use the dated pin "claude-haiku-4-5-20251001"
# if you want a fixed snapshot. For higher-quality drafts, "claude-sonnet-4-6".
ANTHROPIC_MODEL = "claude-haiku-4-5"

# --- Open-source (Ollama) backend ---
# llama3.2 is fine; qwen2.5 tends to produce cleaner JSON if you have it.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
# Overridable because inside a container "localhost" is the CONTAINER, not your
# machine. Docker Desktop exposes the host as host.docker.internal:
#   OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# --- Embeddings (for RAG) ---
# Local, free, no PyTorch (avoids the torch/sentence-transformers wheel problem
# on Python 3.14). Pull once with:  ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

# --- RAG / retrieval ---
USE_RAG = True                    # ground outreach emails in the knowledge base
KNOWLEDGE_DIR = "knowledge"       # folder of .md files to ground outreach in
RAG_INDEX_PATH = "rag_index.json" # cached embeddings (rebuilt when content changes)
RAG_TOP_K = 3                     # how many chunks to retrieve per query

# Vector store backend: "simple" = hand-rolled cosine search over cached vectors
# (no extra deps); "chroma" = Chroma vector DB (pip install chromadb).
# Both satisfy the same build()/retrieve() interface, so the pipeline is unaware.
VECTOR_STORE = "chroma"
CHROMA_DIR = "chroma_db"          # on-disk persistence for the Chroma collection
CHROMA_COLLECTION = "leadgen_knowledge"

# --- Token / cost control (matters most for the Anthropic backend) ---
SCRAPE_CHAR_LIMIT = 6000   # truncate scraped site text before sending to the LLM
ANALYSIS_MAX_TOKENS = 1500  # must fit the WHOLE JSON: truncation = parse error
DRAFT_MAX_TOKENS = 400     # outreach email is short on purpose

# A lead only gets an outreach email drafted if it scores >= this.
# Raising it saves tokens (fewer draft calls) and keeps you focused on hot leads.
DRAFT_THRESHOLD = 60

# Politeness between scrapes (seconds)
SCRAPE_DELAY = 1.0