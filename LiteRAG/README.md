# LiteRAG

**Local Retrieval-Augmented Generation, built from scratch.**

LiteRAG comes in two flavours that share the same algorithm:

| | **microRAG** | **LiteRAG** |
|---|---|---|
| **File** | `microRAG/microrag.py` | `src/pipeline.py` (modular) |
| **Lines of code** | ~130 | ~500 |
| **Learning curve** | Gentle — see everything at once | Moderate — navigate multiple files |
| **Caching** | No — recomputes everything each run | Yes — chunks & embeddings persisted |
| **Best for** | Teaching, understanding the algorithm | Building, extending, production use |
| **Run with** | `python microRAG/microrag.py` | `python -m src.pipeline` |

> *"This is the full algorithmic content of what is needed. Everything else is just efficiency."*
> — Andrej Karpathy, on microGPT

---

## What is RAG?

Traditional LLMs are frozen at training time — they know nothing about your documents, your company's data, or anything that happened after their cutoff date.

**RAG solves this** by giving the LLM a cheat sheet before it answers:

```
Your PDF  →  split into chunks  →  embed as vectors  →  store
                                                            ↓
  Answer  ←  LLM generates  ←  build prompt  ←  retrieve top-K chunks  ←  Question
```

No fine-tuning. No cloud. No API key. Just your document and a local model.

---

## The Pipeline — 6 Stages

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 1. LOAD  │───▶│ 2. CHUNK │───▶│ 3. EMBED │  (run once — cached to disk)
│  PDF     │    │  text    │    │  vectors │
└──────────┘    └──────────┘    └──────────┘
                                      │
                              ┌───────┘  Question
                              ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 6.GENERA-│◀───│ 5.PROMPT │◀───│ 4.RETRIE-│  (run per query — fast)
│   TE     │    │  build   │    │   VE     │
└──────────┘    └──────────┘    └──────────┘
      │
      ▼
   Answer
```

**Stage 1 — Load:** Extract text from each PDF page (PyMuPDF). Keep page numbers for citations.

**Stage 2 — Chunk:** Split pages into 10-sentence windows (spaCy sentencizer). `CHUNK_SIZE` is the single most important RAG hyperparameter — too small loses context, too large hurts retrieval precision.

**Stage 3 — Embed:** Map each chunk to a 384-dimensional vector (SentenceTransformer `all-MiniLM-L6-v2`). Similar meaning → geometrically close vectors.

**Stage 4 — Retrieve:** Embed the query with the *same model*, compute cosine similarity against all chunk vectors in one matrix operation, return top-K hits.

**Stage 5 — Prompt:** Inject retrieved chunks as explicit context. The LLM is told to answer *only* from this context — this is what prevents hallucination.

**Stage 6 — Generate:** TinyLlama reads the prompt and generates an answer token by token, locally, via llama.cpp. No internet. No GPU required.

---

## Project Structure

```
LiteRAG/
│
├── assets/                     # Data storage (auto-created)
│   ├── raw.pdf                # Your PDF document
│   ├── chunks.csv             # Cached chunks (auto-generated)
│   └── embeddings.npy         # Cached vectors (auto-generated)
│
├── microRAG/
│   └── microrag.py            # ← start here. the full algorithm in one file.
│
├── src/
│   ├── config.py              # all settings in one place
│   ├── pipeline.py            # orchestrator — wires all stages
│   │
│   ├── data/
│   │   ├── loader.py          # stage 1 — PDF load & page extraction
│   │   └── splitter.py        # stage 2 — sentence-based chunking
│   │
│   ├── embedding/
│   │   └── embedder.py        # stage 3 — embed chunks, cache to .npy
│   │
│   ├── retrieval/
│   │   └── retriever.py       # stage 4 — cosine similarity search
│   │
│   └── inference/
│       ├── prompt.py          # stage 5 — grounded prompt construction
│       └── generator.py       # stage 6 — llama.cpp generation
│
└── tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf   # local LLM (you provide)
```

**Read order if you're learning:** `microrag.py` → `config.py` → `loader.py` → `splitter.py` → `embedder.py` → `retriever.py` → `prompt.py` → `generator.py` → `pipeline.py`

---

## Quick Start

### Install dependencies

```bash
pip install pymupdf spacy sentence-transformers torch llama-cpp-python pandas requests tqdm
```

### Get the model

Download `tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf` from HuggingFace and place it in the project root:

```bash
# with huggingface-hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf \
    --local-dir ./
```

### Add your PDF

Place your PDF at `assets/raw.pdf` in the project root:

```bash
mkdir -p assets
cp /path/to/your/document.pdf assets/raw.pdf
```

### Run microRAG (single file, great for learning)

```bash
python microRAG/microrag.py
```

### Run LiteRAG (modular, great for extending)

```bash
python -m src.pipeline
```

### Quick start script (optional)

For an automated setup, run the provided quick start script:

```bash
chmod +x quick_start.sh
./quick_start.sh
```

Both approaches give you an interactive question prompt:

```
── LiteRAG · setup ───────────────────────────────────────────
[load]  800 pages
[chunk] 1970 chunks  (size=10 sentences)
Batches: 100%|████████████| 62/62 [00:10<00:00]
[embed] shape=(1970, 384)  (384-dim vectors)
── ready ─────────────────────────────────────────────────────

question > what is backpropagation?

  [retrieve] top 5 chunks:
    page=204  score=0.7312  Backpropagation is an algorithm for computing...
    ...

  [answer] Backpropagation is a method for computing gradients...
```

---

## Configuration

All knobs live in `src/config.py` (modular) or the `CONFIG` block at the top of `microrag.py` (single file):

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `10` | Sentences per chunk. Most impactful RAG parameter. |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Swap for a larger model if you have GPU. |
| `TOP_K` | `5` | Chunks retrieved per question. More = richer context. |
| `N_CTX` | `2048` | LLM context window. Increase if answers get cut off. |
| `TEMPERATURE` | `0.2` | Near 0 = factual. Near 1 = creative. |

---

## Lecture Guide

`microRAG/microrag.py` is designed to be **coded live in a 1–2 hour session** with software engineering audiences.

### Suggested flow (90 min)

| Time | Topic | Code |
|---|---|---|
| 0–15 min | What is RAG? Why does it exist? | No code — whiteboard the pipeline diagram |
| 15–35 min | Load & chunk | Write `load()` and `chunk()` live |
| 35–55 min | Embed | Write `embed()`, explain vector space visually |
| 55–75 min | Retrieve | Write `retrieve()`, explain cosine similarity |
| 75–85 min | Prompt + Generate | Write `build_prompt()` and `generate()` |
| 85–90 min | Wire it up | Write `setup()` + `ask()`, run live demo |

### Key teaching moments

- **After chunk():** ask *"what happens if CHUNK_SIZE = 1? or 100?"* — let students reason about the trade-off.
- **After embed():** visualise that two semantically similar sentences have vectors close in space even with no shared words.
- **After retrieve():** show the printed `hit` lines — students see exactly what the LLM is about to read. This demystifies the "black box" feeling.
- **After generate():** ask a question that's NOT in the document — the model should say "I don't know." — this shows grounding working.

### From microRAG to LiteRAG

After the session, open `src/` and show students: *"this is the same 6 functions — now split into modules with caching, logging, and clean separation of concerns."* The architecture click lands much harder after they've written it from scratch.

---

## Why Build Your Own RAG?

- **Full control** — every stage is explicit and swappable
- **No vendor lock-in** — no LangChain, no LlamaIndex, no cloud
- **CPU-first** — runs on a laptop with no GPU
- **Teachable** — the entire algorithm fits in 130 lines

---

## Roadmap

- [ ] FAISS index for sub-linear retrieval at scale (10k+ chunks)
- [ ] Chunk overlap for better boundary recall
- [ ] Gradio UI for browser-based Q&A
- [ ] Multi-document ingestion
- [ ] Semantic / topic-based chunking

---

## Acknowledgements

Inspired by [Andrej Karpathy's microGPT](https://github.com/karpathy/microGPT) philosophy of distilling algorithms to their essential content, and the broader open-source community behind SentenceTransformers, llama.cpp, PyMuPDF, and FAISS.

---

## License

MIT
```
