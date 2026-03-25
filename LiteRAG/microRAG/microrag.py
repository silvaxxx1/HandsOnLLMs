"""
microrag.py — Retrieval-Augmented Generation, distilled to its essence.

Inspired by Andrej Karpathy's microGPT:
  "This is the full algorithmic content of what is needed.
   Everything else is just efficiency."

This file is the complete RAG algorithm — ~130 lines, zero hidden
abstractions, every step visible and commented. Read it top to bottom
and you will understand exactly how RAG works.

  pip install pymupdf spacy sentence-transformers torch llama-cpp-python
  python -m spacy download en_core_web_sm   # (only if sentencizer fails)

The algorithm, in one breath:
  Split document into chunks → embed chunks into vectors → embed query
  into same vector space → find closest chunks by cosine similarity →
  inject them into a prompt → generate an answer with a local LLM.
"""

import os, re, torch, numpy as np
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Every knob in one place. Change these and nothing else.

PDF_PATH    = "/home/silva/SILVA.AI/Projects/Hands_on_LLM/LiteRAG/assets/raw.pdf"          # point to your PDF
CHUNK_SIZE  = 10     # sentences per chunk — the most important RAG hyperparameter
                     # too small → chunks lose context. too large → retrieval is imprecise.
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim vectors. fast. strong. CPU + GPU.
TOP_K       = 5      # chunks handed to the LLM. more = richer context, longer prompt.
LLM_PATH    = "/home/silva/SILVA.AI/Projects/Hands_on_LLM/LiteRAG/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
N_CTX       = 2048   # LLM context window in tokens
N_THREADS   = 8      # set to your CPU core count


# ─── STAGE 1 · LOAD ───────────────────────────────────────────────────────────
# RAG starts with your document. We extract raw text page by page using
# PyMuPDF. Page numbers are kept so every answer can be traced to a source.

def load(path: str) -> list[dict]:
    import fitz
    pages = []
    for i, page in enumerate(fitz.open(path), 1):
        text = page.get_text()
        if text.strip():                              # skip blank / image-only pages
            pages.append({
                "page": i,
                "text": re.sub(r"\s+", " ", text).strip(),   # collapse whitespace
            })
    print(f"[load]  {len(pages)} pages")
    return pages


# ─── STAGE 2 · CHUNK ──────────────────────────────────────────────────────────
# We cannot feed an entire book to the LLM — its context window is finite.
# We also cannot feed one word — too little meaning. The answer: sentence windows.
#
# WHY SENTENCES, not characters or words?
#   Sentence boundaries are natural meaning boundaries. A 10-sentence chunk
#   is coherent on its own. A 500-character chunk may start mid-thought.
#
# Overlap (step = size // 2) would improve recall at boundaries at the cost
# of ~2× more chunks — a valid production trade-off not needed here.

def chunk(pages: list[dict], size: int = CHUNK_SIZE) -> list[dict]:
    nlp = English()
    nlp.add_pipe("sentencizer")   # lightweight — no full NLP model needed
    chunks, idx = [], 0

    for page in pages:
        sents = [s.text.strip() for s in nlp(page["text"]).sents if s.text.strip()]
        for i in range(0, len(sents), size):
            chunks.append({
                "id":   idx,
                "page": page["page"],
                "text": " ".join(sents[i : i + size]),
            })
            idx += 1

    print(f"[chunk] {len(chunks)} chunks  (size={size} sentences)")
    return chunks


# ─── STAGE 3 · EMBED ──────────────────────────────────────────────────────────
# Text → numbers. A SentenceTransformer maps each chunk to a point in
# 384-dimensional space where similar meaning lives geometrically close.
#
# "The cat sat on the mat" ≈ "A feline rested on the rug"  (high cosine sim)
#
# CRITICAL: query and chunks MUST use the SAME model — same vector space.

def embed(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vecs  = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"[embed] shape={vecs.shape}  ({vecs.shape[1]}-dim vectors)")
    return vecs.astype(np.float32)   # shape: (num_chunks, 384)


# ─── STAGE 4 · RETRIEVE ───────────────────────────────────────────────────────
# Embed the question → measure angle to every chunk → return top-K closest.
#
# COSINE SIMILARITY measures the angle between two vectors (not magnitude).
#   Score 1.0  → same direction   (highly relevant)
#   Score 0.0  → perpendicular    (unrelated)
#   Score -1.0 → opposite         (antonyms)
#
# WHY COSINE, not Euclidean? Embeddings vary in magnitude by sentence length.
# Cosine normalises for that — purely measures semantic direction.
#
# DEVICE: SentenceTransformer auto-selects GPU if available. We move
# chunk tensors to match query's device — never hardcode cpu/cuda.

def retrieve(query: str, model: SentenceTransformer,
             vecs: np.ndarray, chunks: list[dict],
             k: int = TOP_K) -> list[dict]:

    q_vec = model.encode(query, convert_to_tensor=True)    # (384,) — GPU if available
    c_vec = torch.tensor(vecs).to(q_vec.device)            # (N, 384) — follow query

    # one matrix op — no Python loop — fast even at 100k+ chunks
    scores = torch.nn.functional.cosine_similarity(
        q_vec.unsqueeze(0),   # (1, 384) broadcast
        c_vec,                # (N, 384)
    )
    top_scores, top_idx = torch.topk(scores, k=min(k, len(chunks)))

    return [{**chunks[i], "score": round(s, 4)}
            for s, i in zip(top_scores.tolist(), top_idx.tolist())]


# ─── STAGE 5 · PROMPT ─────────────────────────────────────────────────────────
# THE "augmented" step. Retrieved chunks become explicit context in the prompt.
# Telling the LLM to answer ONLY from context keeps it grounded — no hallucination.
#
# Chat template tags (<|system|> etc.) are TinyLlama-specific. Other models
# use different formats — always match the template to the model.

def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[Context {i+1} — Page {h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return (
        f"<|system|>\nYou are a helpful assistant. "
        f"Answer using ONLY the context below. "
        f"If the answer is not there, say \"I don't know.\"\n</s>\n"
        f"<|user|>\n{context}\n\nQuestion: {query}\n</s>\n"
        f"<|assistant|>"
    )


# ─── STAGE 6 · GENERATE ───────────────────────────────────────────────────────
# TinyLlama via llama.cpp — local, offline, no GPU required.
# Autoregressive generation: predict one token at a time, each conditioned
# on everything before it, until a stop token or max_tokens is reached.
#
# temperature: 0 = deterministic / factual.  1 = creative / unpredictable.
# 0.2 is the sweet spot for grounded document Q&A.

def generate(llm: Llama, prompt: str) -> str:
    out = llm(prompt, max_tokens=512, temperature=0.2,
              stop=["</s>", "<|user|>"], echo=False)
    return out["choices"][0]["text"].strip()


# ─── WIRE IT TOGETHER ─────────────────────────────────────────────────────────
# setup()  → stages 1–3 + LLM load. Expensive. Run once.
# ask()    → stages 4–6. Fast. Run per query.

def setup():
    print("\n── microrag · setup ──────────────────────────────────────────")
    pages  = load(PDF_PATH)
    chunks = chunk(pages, CHUNK_SIZE)
    model  = SentenceTransformer(EMBED_MODEL)
    vecs   = embed(chunks, model)
    llm    = Llama(model_path=LLM_PATH, n_ctx=N_CTX,
                   n_threads=N_THREADS, use_mlock=True, verbose=False)
    print("── ready ─────────────────────────────────────────────────────\n")
    return model, vecs, chunks, llm


def ask(query: str, model, vecs, chunks, llm) -> str:
    hits   = retrieve(query, model, vecs, chunks)
    prompt = build_prompt(query, hits)
    answer = generate(llm, prompt)

    # printing hits is crucial — it shows exactly what evidence the LLM saw
    print(f"\n  query  : {query}")
    for h in hits:
        print(f"  hit    : page={h['page']}  score={h['score']}  "
              f"{h['text'][:70]}…")
    print(f"  answer : {answer}\n")
    return answer


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, vecs, chunks, llm = setup()

    while True:
        q = input("question > ").strip()
        if not q: continue
        if q.lower() in {"quit", "exit", "q"}: break
        ask(q, model, vecs, chunks, llm)
