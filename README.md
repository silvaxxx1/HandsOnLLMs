Thanks! Based on your updated directory structure and your vision, here's a **refined, modular, and scalable top-level `README.md`** for the full `Hands_on_LLM` repository. This version clearly separates projects and integrates your new **"Local RAG"** as the **second project**, following the **Text Classification** one. It also references notebooks and subdirectories cleanly.

---

````markdown
# 🧠 Hands-On LLMs: Fine-Tuning, Prompt Engineering, RAG & Agents

Welcome to **HandsOnLLM**, a modular, real-world project suite for mastering **Large Language Models (LLMs)** through practical use cases. Each subproject demonstrates a different aspect of working with LLMs — from prompt engineering and fine-tuning to lightweight local RAG and agent workflows.

---

## 📦 Repository Structure

```text
Hands_on_LLM/
│
├── Text Classification/     # Project 1 - Classify text with fine-tuning & prompting
├── Local RAG/               # Project 2 - Build a local Retrieval-Augmented Generator using TinyLlama + llama.cpp
├── notebooks/               # Exploratory notebooks for each module
├── requirements.txt         # Dependencies for all projects
├── pyproject.toml           # Project metadata (optional)
├── main.py                  # Entry point for consolidated runs (WIP)
├── README.md                # You're here!
````

---

## 🔧 Tech Stack & Core Libraries

| Category                   | Tools & Frameworks                               |
| -------------------------- | ------------------------------------------------ |
| LLMs & Fine-Tuning         | `transformers`, `trl`, `peft`, `bitsandbytes`    |
| Embeddings & Search        | `sentence-transformers`, `faiss`, `annoy`        |
| Prompt Engineering         | `langchain`, `setfit`                            |
| Local LLMs                 | `llama.cpp`, `llama-cpp-python`, `TinyLlama`     |
| Evaluation & Visualization | `sklearn`, `evaluate`, `matplotlib`, `nltk`      |
| Notebook & Utilities       | `jupyterlab`, `pandas`, `datasets`, `ipywidgets` |

---

## 📁 Projects Overview

### 🟩 1. Text Classification

📂 `Text Classification/`

* **Use Cases**: Zero-shot, few-shot, fine-tuned classification
* **Pipelines**:

  * Preprocessing & Feature Extraction
  * Sentence Embedding
  * Fine-tuning transformer models
  * Prompt-based zero-shot prediction
* 📒 Notebooks:

  * `01_eda.ipynb`, `04_finetune.ipynb`, `05_prompt.ipynb`
* ✅ Output: `finetuned/` model ready for inference

---

### 🟨 2. Local RAG with TinyLlama + llama.cpp

📂 `Local RAG/`

* **Goal**: Deploy a **lightweight, fast** Retrieval-Augmented Generation (RAG) pipeline that works **entirely offline**, even on **CPU-only systems**
* **Key Components**:

  * PDF parsing → chunking → embedding (MiniLM, MPNet, Paraphrase-MiniLM)
  * Local similarity search (FAISS)
  * Contextual prompt construction
  * Inference using **TinyLlama-1.1B** via **llama.cpp**
* 📦 Assets:

  * Precomputed embeddings: `.csv` / `.pkl`
  * Local quantized `.gguf` LLM
* 📒 Notebook: `notebooks/local_Rag.ipynb`, `simple-local-rag.ipynb`

#### 🔁 RAG Pipeline (Simplified Diagram)

```text
+-------------+    +--------------+    +----------------+    +---------------+
|  PDF Input  | →  |   Chunking   | →  |  Vector Search  | → |   TinyLlama    |
+-------------+    +--------------+    +----------------+    +---------------+
       ↑                                                        ↓
    User Query ←----------- Contextual Prompt Injection --------+
```

---

## 🔬 Notebooks Summary

| Notebook                    | Description                                     |
| --------------------------- | ----------------------------------------------- |
| `Prompt_Engineering.ipynb`  | Prompt tuning, templates, & few-shot examples   |
| `Text_Classification.ipynb` | Classification workflows (zero-shot, fine-tune) |
| `semantic_search.ipynb`     | Document similarity search using embeddings     |
| `simple-local-rag.ipynb`    | End-to-end RAG pipeline using llama.cpp         |

---

## 🚀 Quick Start

### Set up Environment (with `uv`)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 🎯 Goals & Philosophy

* 💡 **Real-World Scenarios**: Practical over theoretical
* 🔁 **Reusable Components**: Modular design for pipelines and prompts
* 🧪 **Testable Code**: Designed for clarity and debugging
* 🔌 **Low-Resource Friendly**: Run on CPU-only or minimal RAM
* 🔍 **Explorability First**: Every step has an accompanying notebook

---

## 🤝 Contributions

We welcome pull requests, suggestions, and new modules!
Create an issue or fork the repo and open a PR.

---

## 📄 License

MIT License © 2025 **Ex Machina**

```

