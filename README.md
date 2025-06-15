# 🧠 Hands-On LLMs: Fine-Tuning, Prompt Engineering, RAG & Agents

Welcome to a practical, modular project repository focused on **building real-world applications using Large Language Models (LLMs)**. This repo offers hands-on workflows and example pipelines leveraging **industry-standard tools and frameworks** such as Hugging Face, LangChain, and more.

---

## 🔧 Tech Stack & Key Libraries

This project uses a carefully curated set of production-ready libraries:

| Purpose                      | Tools / Packages                                          |
|-----------------------------|----------------------------------------------------------|
| LLM Inference & Fine-Tuning | `transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate` |
| Embeddings & Retrieval       | `sentence-transformers`, `faiss-cpu`, `bertopic`, `annoy`, `mteb` |
| Prompt Engineering           | `langchain`, `langchain-community`, `setfit`             |
| Agents & Tool Use            | `langchain`, `duckduckgo-search`, custom toolkits         |
| Evaluation                  | `evaluate`, `seqeval`, `scikit-learn`, `nltk`, `matplotlib` |
| Cloud APIs & Integration     | `openai`, `cohere`                                        |
| Environment & Notebook       | `jupyterlab`, `ipywidgets`                                |
| Data Handling               | `pandas`, `numpy`, `datasets`                             |

---

## ⚙️ Getting Started

### Step 1: Create and activate your virtual environment

Using [`uv`](https://github.com/ultraviolet-dev/uv):

```bash
uv venv myenv
source myenv/bin/activate
uv pip install -r requirements.txt
````

Alternatively, with Conda:

```bash
conda env create -f environment.yml
conda activate llm-hands-on
```

### Step 2: Launch JupyterLab

```bash
jupyter lab
```

### Step 3: (Optional) Rebuild `llama_cpp_python` with BLAS support for faster inference

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON" uv pip install --force-reinstall llama_cpp_python==0.2.78
```

---

## 💻 Included Use Cases & Notebooks

| # | Topic                                | Notebook                      |
| - | ------------------------------------ | ----------------------------- |
| 1 | Prompt Engineering                   | `01_prompt_engineering.ipynb` |
| 2 | Supervised Fine-Tuning               | `02_finetuning.ipynb`         |
| 3 | Retrieval-Augmented Generation (RAG) | `03_rag_pipeline.ipynb`       |
| 4 | Agents & Tool Integration            | `04_agents_tool_use.ipynb`    |

---

## 🧠 Project Goals

* Build reusable, modular, and testable components
* Demonstrate working with both open-source and commercial LLMs
* Provide end-to-end workflows from data processing to model deployment
* Illustrate best practices in production-grade LLM applications

---

## 📦 Dependencies

Install dependencies via:

* `requirements.txt` — For `uv` or `pip` installs
* `environment.yml` — For Conda environments
* `requirements.lock.txt` — Locked dependency versions for reproducibility

---

## 🤝 Contributions

Contributions, issues, and pull requests are welcome! Please fork the repo and open a PR.

---

## 📄 License

MIT License © 2025 Ex Machina

```

