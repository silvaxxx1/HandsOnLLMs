## 📘 `README.md` – Hands-on LLMs with Industry-Standard Tools

```markdown
# 🧠 Hands-On LLMs: Fine-Tuning, Prompt Engineering, RAG & Agents

Welcome to a practical, modular, and cutting-edge project repository for building **real-world applications with Large Language Models (LLMs)**. This repo showcases hands-on workflows and architectures using **industry-standard tools and frameworks** for:

- ✅ Fine-tuning LLMs
- ✍️ Prompt engineering
- 🔎 Retrieval-Augmented Generation (RAG)
- 🤖 Agentic workflows and tool use

---

## 🔧 Tech Stack & Key Libraries

This project uses a curated set of production-grade libraries:

| Purpose                      | Tools/Packages |
|-----------------------------|----------------|
| LLM Inference & Fine-Tuning | `transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate` |
| Embeddings & Retrieval      | `sentence-transformers`, `faiss-cpu`, `bertopic`, `annoy`, `mteb` |
| Prompt Engineering          | `langchain`, `langchain-community`, `langchain-openai`, `setfit` |
| Agents & Tool Use           | `langchain`, `duckduckgo-search`, custom toolkits |
| Evaluation                  | `evaluate`, `seqeval`, `scikit-learn`, `nltk`, `matplotlib` |
| Cloud APIs & Integration    | `openai`, `cohere` |
| Environment                 | `jupyterlab`, `ipywidgets` |
| Data Handling               | `pandas`, `numpy`, `datasets` |

---



---

## ⚙️ Getting Started

### 🔨 Step 1: Create a virtual environment using `uv`

```bash
uv venv myenv
source myenv/bin/activate
uv pip install -r requirements.txt
````

> Optional: If using `conda`, you can run:
>
> ```bash
> conda env create -f environment.yml
> conda activate llm-hands-on
> ```

### 💡 Step 2: Launch JupyterLab

```bash
jupyter lab
```

### 🔁 Step 3: Rebuild `llama_cpp_python` with BLAS support (optional)

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON" uv pip install --force-reinstall llama_cpp_python==0.2.78
```

---

## 💻 Use Cases Covered

| # | Topic                     | Notebook                      |
| - | ------------------------- | ----------------------------- |
| 1 | Prompt Engineering        | `01_prompt_engineering.ipynb` |
| 2 | Supervised Fine-Tuning    | `02_finetuning.ipynb`         |
| 3 | Retrieval-Augmented Gen   | `03_rag_pipeline.ipynb`       |
| 4 | Agents & Tool Integration | `04_agents_tool_use.ipynb`    |

---

## 🧠 Goals of the Project

* Build reusable, testable, modular components
* Demonstrate open-source and commercial LLMs
* Cover end-to-end workflows from data to deployment
* Follow best practices used in real production systems

---

## 📦 Dependencies

Install dependencies from:

* `requirements.txt` — For `uv` or `pip`
* `environment.yml` — For `conda`
* `requirements.lock.txt` — For locked version installs

---

## 🤝 Contributions

Contributions, issues, and pull requests are welcome! Please fork the repo and open a PR.

---

## 📄 License

MIT License © 2025 Ex Machina

```
