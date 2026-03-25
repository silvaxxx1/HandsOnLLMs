"""
config.py — single source of truth for all LiteRAG settings.
Change values here; nothing else needs to be touched.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

PDF_PATH          = os.path.join(ASSETS_DIR, "raw.pdf")
CHUNKS_CSV        = os.path.join(ASSETS_DIR, "chunks.csv")
EMBEDDINGS_PKL    = os.path.join(ASSETS_DIR, "embeddings.npy")
LLM_PATH          = os.path.join(BASE_DIR, "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")
# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 10    # sentences per chunk
                      # ↑ larger  → more context, less precise retrieval
                      # ↓ smaller → more precise, may lose context

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"   # 384-dim, fast, CPU-friendly

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K         = 5     # chunks returned per query

# ── LLM ───────────────────────────────────────────────────────────────────────
N_CTX         = 2048  # context window (tokens)
N_THREADS     = 8     # set to your CPU core count
TEMPERATURE   = 0.2   # 0 = deterministic, 1 = creative
MAX_TOKENS    = 512
