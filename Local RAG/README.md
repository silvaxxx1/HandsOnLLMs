# RAG-From-Scratch

A simple **Retrieval-Augmented Generation (RAG)** system built from the ground up — no heavy frameworks, just clean Python code.

---

## Project Overview

This project demonstrates how to build a local RAG pipeline step-by-step, starting from:

* Loading and chunking documents (e.g., books)
* Generating embeddings for text chunks
* Indexing chunks using FAISS for fast similarity search
* Querying with an LLM to generate answers grounded in retrieved content
* Interactive Gradio UI for easy usage

---

## Why Build Your Own RAG?

* Understand every component inside out
* Customize to your specific use case
* Run locally on modest hardware
* Avoid vendor lock-in and cloud costs

---

## Directory Structure

```plaintext
rag-from-scratch/
├── app/
│   ├── embedder.py         # Embedding code  
│   ├── retriever.py        # FAISS index and search  
│   ├── reader.py           # Load and split text  
│   ├── llm.py              # LLM query wrapper  
│   └── rag_pipeline.py     # Orchestrates all components  
├── chunks/                 # Processed text chunks  
├── data/                   # Place your books here  
├── embeddings/             # Stores vector indexes  
├── models/                 # Downloaded ML models  
├── ui/
│   └── gradio_app.py       # Web UI  
├── requirements.txt        # Dependencies  
├── setup.sh                # Setup script  
└── README.md               # This file  
```

---

## Getting Started

1. **Clone the repo**

   ```bash
   git clone <repo_url>
   cd rag-from-scratch
   ```

2. **Install dependencies**

   ```bash
   bash setup.sh
   ```

3. **Add your book/document**
   Place your `.txt` file(s) inside the `data/` folder.

4. **Create chunks & embeddings**
   Run the reader and embedder scripts (or the full pipeline) to process your data.

5. **Start the UI**

   ```bash
   python ui/gradio_app.py
   ```

   Open the displayed URL in your browser.

---

## Notes

* Designed to run on modest hardware (e.g., laptops with 16GB RAM + decent GPU)
* Use smaller LLM models for local inference (e.g., `distilgpt2` or similar)
* Extensible — swap in your own embedding model, vector store, or LLM

---

## Next Steps

* Improve chunking strategy (e.g., semantic chunking)
* Add support for multiple documents
* Experiment with larger LLMs or quantized models
* Add caching and persistence

---
