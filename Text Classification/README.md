## 🧠 Unified LLM Text Classification Pipeline

A modular, end-to-end project for text classification using Large Language Models (LLMs) — supporting **fine-tuning**, **feature extraction**, **zero-shot**, and **few-shot prompting** from a single CLI interface using `uv`.

---

## ✨ Features

✅ Fine-tuning with Hugging Face Transformers

✅ Feature extraction using LLM embeddings + classical ML

✅ Embedding-based classification with SentenceTransformers

✅ Zero-shot prompting with OpenAI Chat Models (GPT-3.5, GPT-4)

✅ Unified command-line interface via `uv`

✅ Built-in evaluation metrics + confusion matrix

✅ **Interactive web demo via Gradio app**

---

## 📂 Project Structure

```
text_classification_llm/
├── config.py                    # Global config
├── run_pipeline.py             # Main entry point (CLI with uv)
├── src/
│   ├── train_finetune.py       # Fine-tune transformer models
│   ├── train_features.py       # Train ML classifier on hidden states
│   ├── embedding_pipeline.py   # SentenceTransformer embeddings + ML
│   ├── zero_shot.py            # Zero-shot via OpenAI GPT
│   ├── eval.py                 # Evaluation utilities
│   └── __init__.py
├── app/
│   └── gradio_app.py           # Interactive web UI for emotion classification
├── outputs/                    # Logs, predictions, confusion matrix
├── models/                     # Fine-tuned model checkpoints
├── pyproject.toml              # uv-based dependency management
└── README.md
```

---

## 📊 Classification Modes

| Mode              | Description                            | Script                  |
| ----------------- | -------------------------------------- | ----------------------- |
| ✅ Fine-Tuning     | Train transformer on labeled data      | `train_finetune.py`     |
| ✅ Feature-Based   | Extract BERT features → ML classifier  | `train_features.py`     |
| ✅ Embedding-Based | SentenceTransformers → classifier      | `embedding_pipeline.py` |
| ✅ Zero-Shot       | Few-shot GPT-style prompt → prediction | `zero_shot.py`          |

---

## ▶️ Run the Pipeline with `uv`

```bash
# Fine-tune BERT on emotion dataset
uv run run_pipeline.py --mode finetune

# Use BERT hidden states + logistic regression
uv run run_pipeline.py --mode feature

# Use SentenceTransformer embeddings + ML classifier
uv run run_pipeline.py --mode embedding

# Run zero-shot classification with OpenAI GPT (requires API key)
uv run run_pipeline.py --mode zero --openai_api_key sk-...
```

---

## 🖥️ Run the Interactive Gradio Web App

Try the emotion classifier live in your browser! The app classifies input text into emotions like joy, sadness, anger, and more — displaying colored badges with confidence scores.

```bash
python app/gradio_app.py
```

* Visit the local URL printed (e.g., `http://localhost:7860/`) or the public link (if `share=True` is enabled).
* Enter any text and see instant emotion predictions with friendly colored pills.
* Perfect for quick demos or user testing without any coding.

---

## ⚙️ CLI Options

You can override config values at runtime:

```bash
uv run run_pipeline.py --mode finetune --epochs 5 --batch_size 16 --model_name distilbert-base-uncased
```

For zero-shot:

```bash
uv run run_pipeline.py --mode zero --openai_api_key sk-... --model_name gpt-4
```

---

## 📈 Evaluation

All training/inference modes output:

* Accuracy, F1, Precision, Recall
* Per-class metrics
* Confusion matrix plot saved to `outputs/`

---

## 🧪 Default Dataset

We use [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) by default:

```
["sadness", "joy", "love", "anger", "fear", "surprise"]
```

To use a custom dataset, change `config.py` or plug in your own Hugging Face dataset.

---

## 🧠 Few-Shot Prompt Format (Zero-Shot)

Prompt format used with GPT models:

```
Classify the following sentence into one of: [joy, sadness, anger, fear, love, surprise]

Example: I'm feeling great today!
Label: joy

Example: Why is everything so hard today?
Label: sadness

Example: I just got promoted!
Label:
```

Predicted class is parsed from GPT response.

---

## 📦 Setup with `uv`

Install `uv` if you haven’t:

```bash
curl -Ls https://astro.build/install | bash
```

Then install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or use `pyproject.toml` with:

```bash
uv pip install -e .
```

---

## ❤️ Built With

* Hugging Face Transformers
* OpenAI API (GPT models)
* Scikit-learn
* Gradio interactive app
* `uv` for blazing-fast dependency management

---

## 📄 License

MIT License
