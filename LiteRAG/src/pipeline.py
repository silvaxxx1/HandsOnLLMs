"""
pipeline.py — LiteRAG orchestrator.

Wires all six stages into two clean functions:
  setup() → run once to build the knowledge base
  ask()   → run for every user query

The expensive work (chunking, embedding, loading the LLM) happens in
setup(). After that, each question only pays the cost of retrieval +
generation — both of which are fast.

Usage (CLI):
    python -m src.pipeline

Usage (import):
    from src.pipeline import setup, ask
    state = setup()
    answer = ask("What is backpropagation?", *state)
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from config  import PDF_PATH, CHUNK_SIZE, CHUNKS_CSV, EMBEDDINGS_PKL
from data.loader  import load_pdf_pages
from data.splitter     import chunk_pages, save_chunks
from embedding.embedder import load_embed_model, embed_chunks, save_embeddings, load_embeddings
from retrieval.retriever import retrieve
from inference.prompt    import build_prompt
from inference.generator import load_llm, generate

import os


def setup(pdf_path: str = PDF_PATH) -> tuple:
    """
    Stages 1–3 + LLM load.  Run once per session.

    Returns (embed_model, vecs, chunks, llm) — pass this tuple to ask().
    Embeddings and chunks are cached to disk so re-runs skip recomputation.
    """
    print("\n── LiteRAG · setup ───────────────────────────────────────────")

    # ── Stage 1 · Load ────────────────────────────────────────────────────────
    pages = load_pdf_pages(pdf_path)

    # ── Stage 2 · Chunk ───────────────────────────────────────────────────────
    # Use cached chunks if available — chunking a large PDF takes a few seconds
    if os.path.exists(CHUNKS_CSV):
        from data.splitter import load_chunks
        logging.info("Loading cached chunks ...")
        chunks = load_chunks(CHUNKS_CSV)
    else:
        chunks = chunk_pages(pages, CHUNK_SIZE)
        save_chunks(chunks, CHUNKS_CSV)

    # ── Stage 3 · Embed ───────────────────────────────────────────────────────
    # Use cached embeddings if available — embedding 2000 chunks takes ~10s
    embed_model = load_embed_model()
    if os.path.exists(EMBEDDINGS_PKL):
        logging.info("Loading cached embeddings ...")
        vecs = load_embeddings(EMBEDDINGS_PKL)
    else:
        vecs = embed_chunks(chunks)
        save_embeddings(vecs, EMBEDDINGS_PKL)

    # ── Stage 6 · Load LLM ────────────────────────────────────────────────────
    llm = load_llm()

    print("── ready ─────────────────────────────────────────────────────\n")
    return embed_model, vecs, chunks, llm


def ask(query: str, embed_model, vecs, chunks, llm,
        show_hits: bool = True) -> str:
    """
    Stages 4–6 for a single query.

    Args:
        query      : the user's question
        show_hits  : if True, print the retrieved chunks before the answer
                     — extremely useful for debugging retrieval quality

    Returns:
        the generated answer string
    """
    # ── Stage 4 · Retrieve ────────────────────────────────────────────────────
    hits = retrieve(query, vecs, chunks)

    if show_hits:
        print(f"\n  [retrieve] top {len(hits)} chunks:")
        for h in hits:
            print(f"    page={h['page']}  score={h['score']}  "
                  f"{h['text'][:70]}…")

    # ── Stage 5 · Prompt ──────────────────────────────────────────────────────
    prompt = build_prompt(query, hits)

    # ── Stage 6 · Generate ────────────────────────────────────────────────────
    answer = generate(llm, prompt)
    print(f"\n  [answer] {answer}\n")
    return answer


# ── Interactive Q&A loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    state = setup()

    print("Type a question and press Enter.  Type 'quit' to exit.\n")
    while True:
        q = input("question > ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        ask(q, *state)
