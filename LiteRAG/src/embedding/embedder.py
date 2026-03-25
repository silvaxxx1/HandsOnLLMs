"""
embedding/embedder.py — Stage 3: convert text chunks into dense vectors.

WHY EMBEDDINGS?
  Computers cannot compare text directly. A SentenceTransformer maps
  each chunk to a point in 384-dimensional space where similar meaning
  sits close together — regardless of exact wording.

  "The cat sat on the mat" ≈ "A feline rested on the rug"  (high cosine sim)

CRITICAL RULE: the query at retrieval time MUST use the same model.
  Query and chunks must live in the same vector space or similarity
  scores are meaningless.
"""

import functools
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, EMBEDDINGS_PKL


@functools.lru_cache(maxsize=1)
def load_embed_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Load and cache the embedding model (loaded once, reused everywhere).
    lru_cache(maxsize=1) keeps exactly the last-loaded model in memory.
    """
    logging.info(f"Loading embedding model: '{model_name}' ...")
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[dict],
                 model_name: str = EMBED_MODEL) -> np.ndarray:
    """
    Embed all chunk texts and return a float32 array of shape (N, 384).
    embeddings[i] is the vector for chunks[i] — indices must stay in sync.
    """
    model = load_embed_model(model_name)
    texts = [c["text"] for c in chunks]

    logging.info(f"Embedding {len(texts)} chunks ...")
    vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    logging.info(f"Embeddings shape: {vecs.shape}")
    return vecs.astype(np.float32)


def save_embeddings(vecs: np.ndarray, path: str = EMBEDDINGS_PKL) -> None:
    """
    Save embeddings with np.save → .npy file.
    Using .npy (not .pkl) is explicit about the format and avoids
    the pickle security surface. Load with np.load(path).
    """
    np.save(path, vecs)
    logging.info(f"Embeddings saved to '{path}'.")


def load_embeddings(path: str = EMBEDDINGS_PKL) -> np.ndarray:
    """Load embeddings from a .npy file."""
    if not __import__("os").path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: '{path}'")
    return np.load(path)
