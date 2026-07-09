"""
Central configuration for the lead-generation agent.

Edit COMPANY_* to describe YOUR agency. The model + budget knobs below
are the main levers for controlling cost / quality.
"""

# --- Your agency (used in prompts so outreach is grounded in who you are) ---
COMPANY_NAME = "BrightPixel Studio"
COMPANY_PITCH = (
    "We build modern, fast, mobile-friendly websites and run digital "
    "marketing (SEO, paid ads, content) for small and mid-sized businesses."
)
SENDER_NAME = "Alex"
SENDER_ROLE = "Growth Partner"

# --- Anthropic backend ---
# Current cost-efficient model. Use the dated pin "claude-haiku-4-5-20251001"
# if you want a fixed snapshot. For higher-quality drafts, "claude-sonnet-4-6".
ANTHROPIC_MODEL = "claude-haiku-4-5"

# --- Open-source (Ollama) backend ---
# llama3.2 is fine; qwen2.5 tends to produce cleaner JSON if you have it.
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"  # OpenAI-compatible endpoint


USE_RAG = False

# --- Embeddings (for RAG) ---
# Local, free, no PyTorch (avoids the torch/sentence-transformers wheel problem
# on Python 3.14). Pull once with:  ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

# --- RAG / retrieval ---
KNOWLEDGE_DIR = "knowledge"  # folder of .md files to ground outreach in
RAG_INDEX_PATH = "rag_index.json"  # cached embeddings (rebuilt when content changes)
RAG_TOP_K = 3  # how many chunks to retrieve per query

# --- Token / cost control (matters most for the Anthropic backend) ---
SCRAPE_CHAR_LIMIT = 6000  # truncate scraped site text before sending to the LLM
ANALYSIS_MAX_TOKENS = 700  # JSON assessment is small; keep this tight
DRAFT_MAX_TOKENS = 400  # outreach email is short on purpose

# A lead only gets an outreach email drafted if it scores >= this.
# Raising it saves tokens (fewer draft calls) and keeps you focused on hot leads.
DRAFT_THRESHOLD = 60

# Politeness between scrapes (seconds)
SCRAPE_DELAY = 1.0
