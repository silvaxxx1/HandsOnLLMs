# 🧠 Unified LLM Text Classification Pipeline

A modular, end-to-end project for text classification using Large Language Models (LLMs) — supporting **fine-tuning**, **feature extraction**, **zero-shot**, and **few-shot prompting** in a single unified pipeline.

---

## ✨ Features

✅ Fine-tuning with Hugging Face Transformers  
✅ Feature extraction using LLM embeddings + classical ML  
✅ Zero-shot classification via NLI models  
✅ Few-shot prompting via OpenAI / instruct models  
✅ Unified command-line interface  
✅ Ready for deployment with Gradio

---

## 📂 Project Structure

```

text\_classification\_llm/
├── config.py
├── run\_pipeline.py            # Entry point for all modes
├── src/
│   ├── data.py                # Load and tokenize datasets
│   ├── features.py            # Extract LLM embeddings
│   ├── train\_features.py      # Train ML on embeddings
│   ├── train\_finetune.py      # Fine-tune transformer models
│   ├── zero\_shot.py           # Zero-shot via BART/Roberta NLI
│   ├── few\_shot.py            # Few-shot prompting via GPT
│   ├── evaluate.py            # Evaluation helpers
├── models/                    # Saved models
├── outputs/                   # Logs, plots, reports
├── app/
│   └── gradio\_app.py          # Web UI (optional)
├── requirements.txt
└── README.md

````

---

## 📊 Supported Classification Modes

| Mode             | Description                                         | File             |
|------------------|-----------------------------------------------------|------------------|
| ✅ Fine-Tuning     | Train a transformer model on labeled data          | `train_finetune.py` |
| ✅ Feature-Based   | Extract embeddings → ML classifier                 | `train_features.py` |
| ✅ Zero-Shot       | Use NLI model to classify without training         | `zero_shot.py`      |
| ✅ Few-Shot        | Prompt LLM with examples → predict class           | `few_shot.py`       |

---


## ▶️ Run Any Mode

```bash
# Fine-tune BERT on dataset
python run_pipeline.py --mode finetune

# Use embeddings + Logistic Regression
python run_pipeline.py --mode feature

# Use zero-shot classification (no training)
python run_pipeline.py --mode zero

# Use few-shot prompting (OpenAI GPT)
python run_pipeline.py --mode fewshot
```

---

## 📈 Evaluation

* Accuracy, F1-score (feature & finetune modes)
* Per-class metrics
* Confusion matrix in `outputs/`

---

## 🧪 Example Dataset

Uses [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) by default.

Classes:

```
["sadness", "joy", "love", "anger", "fear", "surprise"]
```

You can change dataset in `config.py` or plug in your own.

---

## 🌐 Optional: Run Web App

```bash
python app/gradio_app.py
```

Then visit `http://localhost:7860/` to test your model live.

---

## 🧠 Few-Shot Prompting Logic

Uses OpenAI Chat models (GPT-3.5, GPT-4) to classify text by example:

**Example Prompt:**

```
Classify the following sentence into one of: [joy, sadness, anger, fear, love, surprise]

Example: I'm feeling great today!
Label: joy

Example: Why is everything so hard today?
Label: sadness

Example: I just got promoted!
Label:
```

---


## 🧠 Credits

Built with ❤️ using Hugging Face Transformers, OpenAI, and Scikit-Learn.

---

## 📄 License

MIT License.
