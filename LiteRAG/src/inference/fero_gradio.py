import gradio as gr
import torch
import pandas as pd
import os
from src.config import (
    BASE_DIR,
    CHUNKS_CSV_PATH,
    DEFAULT_EMBEDDING_MODEL_KEY,
    DEFAULT_GENERATION_MODEL_KEY,
    get_embeddings_pickle_path,
    SUPPORTED_GENERATION_MODELS,
)
from src.embedding.load_embed_model import load_embedding_model
from src.inference.prompt_builder import build_prompt
from llama_cpp import Llama

# Quiet context manager to suppress llama.cpp logs
class SuppressOutput:
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def load_embeddings(path: str) -> torch.Tensor:
    import numpy as np
    return torch.from_numpy(np.load(path, mmap_mode="r"))


def retrieve_contexts(query, embed_model, embeddings, metadata, device, top_k=5):
    query_embedding = embed_model.encode(query, convert_to_tensor=True).to(device)
    query_embedding = torch.nn.functional.normalize(query_embedding, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    scores = embeddings @ query_embedding
    top_indices = torch.topk(scores, top_k).indices.tolist()
    return [metadata[i].get("chunk_text", "") for i in top_indices]


def load_llama_cpp_model(model_path: str, n_threads=4, n_ctx=2048):
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)
    with SuppressOutput():
        model = Llama(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx)
    return model


def generate(llm, prompt, max_tokens=150, temperature=0.7):
    with SuppressOutput():
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stop=["\n\n"]
        )
    return output["choices"][0]["text"].strip()


def clean_answer(text):
    lines = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
    cleaned = " ".join(lines)
    if cleaned.endswith(('.', '!', '?')):
        return cleaned
    return cleaned.rsplit('.', 1)[0] + '.' if '.' in cleaned else cleaned


# Setup models once
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = load_embedding_model(DEFAULT_EMBEDDING_MODEL_KEY)
embeddings = load_embeddings(get_embeddings_pickle_path(DEFAULT_EMBEDDING_MODEL_KEY)).to(device)
df = pd.read_csv(CHUNKS_CSV_PATH)
metadata = df.to_dict(orient="records")
model_cfg = SUPPORTED_GENERATION_MODELS[DEFAULT_GENERATION_MODEL_KEY]
llama_model = load_llama_cpp_model(
    model_path=model_cfg["path"],
    n_threads=model_cfg.get("threads", 4),
    n_ctx=model_cfg.get("ctx", 2048)
)

# Simple in-memory cache dictionary
cache = {}

def answer_question(query):
    query = query.strip()
    if not query:
        return "Please ask something to start our lovely chat! 🌸"
    if query in cache:
        return cache[query]  # Return cached answer immediately

    chunks = retrieve_contexts(query, embed_model, embeddings, metadata, device)
    prompt = build_prompt(chunks, query, style="qa")
    raw_answer = generate(llama_model, prompt)
    answer = clean_answer(raw_answer)

    cache[query] = answer  # Cache the answer for future requests
    return answer


title = "🌸 Gentle Q&A with Fero 🌸"
description = "Ask anything about my beloved Fero — her story, her magic, and all the little things that make her truly special. 💖"

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask your question here:", placeholder="Type your question..."),
    outputs=gr.Textbox(label="AI Thinking:", lines=8),
    title=title,
    description=description,
    theme="compact",
)

if __name__ == "__main__":
    iface.launch(share=True)
