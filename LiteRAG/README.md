# LiteRAG : Local RAG from Scratch

A lightweight, modular **Retrieval-Augmented Generation (RAG)** pipeline designed to run efficiently on local machines — especially CPU-bound environments — without relying on heavy frameworks or cloud services.

This project builds a functional backbone for RAG by combining:

- Document loading and chunking
- Vector embedding with customizable models
- FAISS-based similarity search
- Local LLM inference powered by [TinyLlama](https://github.com/johnsmith0031/tiny-llama) using `llama.cpp`
- Optional interactive Gradio UI (coming soon)

---

## Why Build Your Own Local RAG?

- **Full control:** Understand and customize every stage of the pipeline  
- **Resource efficient:** Optimized for CPU and modest hardware (e.g., laptops, low-end servers)  
- **No vendor lock-in:** Run fully offline without cloud costs or dependencies  
- **Extensible:** Swap embedding models, chunking strategies, or LLMs as needed  

---

## Project Structure

```

Local-RAG/
├── app.py                     # Gradio web UI (optional)
├── README.md                  # This file
└── src/
├── assets/                # Raw PDFs, chunk CSVs, embeddings
├── data/                  # Document loading & chunking logic
├── embedding/             # Embedding models and storage
├── inference/             # LLM loading and query handling
├── rag\_pipline.py         # Modular CLI pipeline orchestrator
├── retrieval/             # FAISS index and similarity search
└── tinyllama-1.1b-chat-v1.0.Q5\_K\_M.gguf  # TinyLlama quantized model for local inference

````

+------------+      +------------+      +--------------+      +-----------+
|  Upload &  | ---> |  Data      | ---> |  Embedding   | ---> |  Inference |
|  Load PDF  |      |  Pipeline  |      |  Pipeline    |      |  (TinyLlama)|
+------------+      +------------+      +--------------+      +-----------+
                                       |                           |
                                       v                           v
                                 +--------------+          +----------------+
                                 |  Chunked     |          |  Answers to    |
                                 |  Text        |          |  User Queries  |
                                 +--------------+          +----------------+




###  Add your documents

Place your PDF files inside `src/assets/` or provide a direct URL when running the pipeline.

###  Run the pipeline via CLI

You can run each stage individually or run the full pipeline sequentially.

#### Example CLI commands:

* **Run only the data pipeline (download, chunk):**

  ```bash
  python src/rag_pipline.py data \
    --url http://example.com/file.pdf \
    --save-path ./src/assets/raw.pdf \
    --chunk-size 512 \
    --show-stats
  ```

* **Run only embedding:**

  ```bash
  python src/rag_pipline.py embed \
    --model-key sentence-transformers/all-MiniLM-L6-v2
  ```

* **Run only inference:**

  ```bash
  python src/rag_pipline.py inf --device cpu
  ```

* **Run the full pipeline:**

  ```bash
  python src/rag_pipline.py all \
    --url http://example.com/file.pdf \
    --save-path ./src/assets/raw.pdf \
    --chunk-size 512 \
    --show-stats \
    --model-key sentence-transformers/all-MiniLM-L6-v2 \
    --device cpu
  ```

---

## About TinyLlama and llama.cpp

This project leverages [TinyLlama](https://github.com/johnsmith0031/tiny-llama), a lightweight quantized LLM model, loaded and queried locally through the [llama.cpp](https://github.com/ggerganov/llama.cpp) runtime. This combination enables fast, offline inference without a GPU — perfect for CPU-limited environments.

---

## Configuration & Optimization

* **Chunk Size:** Adjust `--chunk-size` to balance chunk granularity and embedding/inference speed. Smaller chunks can improve retrieval relevance but increase processing time.
* **Embedding Models:** Easily swap embedding models via `--model-key`. Use smaller, faster models for CPU environments or larger models if GPU available.
* **Inference Device:** Use `--device cpu` to run on CPU or `--device cuda` if GPU and CUDA available.
* **Chunk Stats:** Enable `--show-stats` to visualize chunk length distributions for data diagnostics.
* **Caching:** Embeddings and chunks are saved in `src/assets/` to avoid redundant computation.

---

## Interactive Web UI (Coming Soon)

A user-friendly Gradio interface will allow uploading PDFs and querying directly from the browser with real-time response from the local RAG pipeline.

---

## Future Improvements

* Support multi-document ingestion and indexing
* Enhanced chunking (semantic or topic-based)
* More advanced caching and persistence layers
* Additional embedding and LLM model support
* Performance profiling and further optimization for very low-resource devices

---

## License

MIT License

---

## Acknowledgments

Inspired by open-source RAG research and implementations, thanks to the open-source community for embedding models, FAISS, llama.cpp, and TinyLlama.

---

**Build your own RAG pipeline tailored for local environments — fully transparent, modular, and efficient.**
Happy exploring! 🚀

```
