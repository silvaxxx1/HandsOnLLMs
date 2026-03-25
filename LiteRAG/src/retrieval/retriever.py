"""
retrieval/retriever.py — Stage 4: find the chunks most relevant to a query.

HOW IT WORKS:
  1. Embed the query with the same model used for chunks.
  2. Compute cosine similarity between query vector and all chunk vectors.
  3. Return the top-K highest-scoring chunks as evidence for the LLM.

WHY COSINE SIMILARITY (not Euclidean distance)?
  Sentence embeddings vary in magnitude with sentence length. Cosine
  similarity normalises for magnitude — it measures the *angle* between
  vectors, which is a purer proxy for semantic direction (meaning).

  Score = 1.0 → same direction   (very relevant)
  Score = 0.0 → perpendicular    (unrelated)
  Score = -1.0 → opposite        (antonyms)

DEVICE NOTE:
  SentenceTransformer auto-selects GPU if available, so the query
  tensor may land on cuda:0. We move chunk tensors to the same device
  via .to(query.device) — never hardcode 'cpu' or 'cuda'.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from config import TOP_K
from embedding.embedder import load_embed_model


def retrieve(query: str,
             vecs: np.ndarray,
             chunks: list[dict],
             k: int = TOP_K,
             model_name: str = None) -> list[dict]:
    """
    Embed `query` and return the top-k most similar chunks.

    Args:
        query      : the user's question
        vecs       : (N, D) float32 array of pre-computed chunk embeddings
        chunks     : list of chunk dicts — must be index-aligned with vecs
        k          : number of chunks to return
        model_name : override the default embedding model

    Returns:
        list of chunk dicts, each with an extra "score" key, sorted
        highest → lowest similarity.
    """
    model = load_embed_model(model_name) if model_name else load_embed_model()

    # embed query — lands on GPU automatically if one is available
    q_vec = model.encode(query, convert_to_tensor=True)           # (D,)
    c_vec = torch.tensor(vecs).to(q_vec.device)                   # (N, D) — same device

    # single matrix operation: no Python loop, handles 100k+ chunks easily
    scores = torch.nn.functional.cosine_similarity(
        q_vec.unsqueeze(0),   # (1, D) → broadcast over all chunk rows
        c_vec,                # (N, D)
    )

    top_scores, top_idx = torch.topk(scores, k=min(k, len(chunks)))

    results = []
    for score, i in zip(top_scores.tolist(), top_idx.tolist()):
        results.append({**chunks[i], "score": round(score, 4)})

    return results   # already sorted highest → lowest by torch.topk
