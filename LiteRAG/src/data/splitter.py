"""
data/splitter.py — Stage 2: split pages into sentence-based chunks.

WHY SENTENCES?
  Character splits break mid-thought. Word splits lose sentence-level
  semantics. Sentence splits preserve meaning — each chunk is coherent
  on its own, which makes embedding and retrieval much more reliable.

WHY A FIXED WINDOW SIZE?
  CHUNK_SIZE is the single most important RAG hyperparameter.
  Too small → chunks lack context.  Too large → retrieval is imprecise.
  10 sentences is a solid starting point for most documents.
"""

import re
import logging
import pandas as pd
from spacy.lang.en import English
from config import CHUNK_SIZE, CHUNKS_CSV


def _sentencize(text: str, nlp) -> list[str]:
    """Return a list of non-empty sentence strings from raw text."""
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Split each page into non-overlapping windows of `chunk_size` sentences.

    Returns a flat list of chunk dicts:
        { "id": int, "page": int, "text": str }

    NOTE: overlap (step = chunk_size // 2) would improve boundary recall
    at the cost of ~2× more chunks — a valid production trade-off.
    """
    nlp = English()
    nlp.add_pipe("sentencizer")   # lightweight — no full NLP model needed

    chunks, idx = [], 0

    for page in pages:
        sentences = _sentencize(page["text"], nlp)

        for i in range(0, len(sentences), chunk_size):
            window = sentences[i : i + chunk_size]
            chunks.append({
                "id":   idx,
                "page": page["page"],
                "text": " ".join(window),
            })
            idx += 1

    logging.info(f"Created {len(chunks)} chunks (window={chunk_size} sentences).")
    return chunks


def save_chunks(chunks: list[dict], path: str = CHUNKS_CSV) -> None:
    """Persist chunks to CSV for inspection and pipeline caching."""
    pd.DataFrame(chunks).to_csv(path, index=False)
    logging.info(f"Chunks saved to '{path}'.")


def load_chunks(path: str = CHUNKS_CSV) -> list[dict]:
    """Reload chunks from a previously saved CSV."""
    return pd.read_csv(path).to_dict(orient="records")
