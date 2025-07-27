import os
import sys
import time
import torch
import pandas as pd
from config import (
    BASE_DIR,
    CHUNKS_CSV_PATH,
    DEFAULT_EMBEDDING_MODEL_KEY,
    DEFAULT_GENERATION_MODEL_KEY,
    get_embeddings_pickle_path,
)
from embedding.load_embed_model import load_embedding_model
from .prompt_builder import build_prompt
from retrieval.search import compute_similarity
from llama_cpp import Llama  # llama.cpp python bindings

# Helper context manager to suppress stdout/stderr
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def load_embeddings(path: str) -> torch.Tensor:
    import numpy as np
    embeddings = np.load(path, mmap_mode="r")  # memory efficient loading
    return torch.from_numpy(embeddings)


def retrieve_contexts(query, embed_model, embeddings, metadata, device, top_k=5):
    start = time.time()
    query_embedding = embed_model.encode(query, convert_to_tensor=True).to(device)
    # Normalize for cosine similarity
    query_embedding = torch.nn.functional.normalize(query_embedding, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    scores = embeddings @ query_embedding
    top_scores, top_indices = torch.topk(scores, top_k)
    selected_contexts = [metadata[i].get("chunk_text", "") for i in top_indices.tolist()]
    end = time.time()
    print(f"[Timing] Context retrieval took {end - start:.3f} seconds.")
    return selected_contexts


def load_llama_cpp_model(model_path: str, n_threads: int = 4, n_ctx: int = 2048):
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)
    print(f"[LLaMA.cpp] Loading model from: {model_path} with context size: {n_ctx}")
    with SuppressOutput():
        llama_model = Llama(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx)
    print("[LLaMA.cpp] Model loaded successfully")
    return llama_model


def generate_with_llama_cpp(llama_model, prompt, max_tokens=128, temperature=0.7):
    print("[LLaMA.cpp] Generating text...")
    with SuppressOutput():
        response = llama_model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stop=["\n\n"]
        )
    return response["choices"][0]["text"].strip()


def main():
    device = "cpu"
    print(f"[System] Using device: {device}")

    # Load embedding model
    embed_model_key = DEFAULT_EMBEDDING_MODEL_KEY
    print(f"[Embedding] Loading model: {embed_model_key}")
    start = time.time()
    embed_model = load_embedding_model(embed_model_key)
    end = time.time()
    print(f"[Timing] Embedding model loaded in {end - start:.3f} seconds.")

    embeddings_path = get_embeddings_pickle_path(embed_model_key)
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

    start = time.time()
    embeddings = load_embeddings(embeddings_path).to(device)
    end = time.time()
    print(f"[Timing] Loaded embeddings in {end - start:.3f} seconds.")

    if not os.path.exists(CHUNKS_CSV_PATH):
        raise FileNotFoundError(f"Chunks CSV not found at {CHUNKS_CSV_PATH}")

    df = pd.read_csv(CHUNKS_CSV_PATH)
    metadata = df.to_dict(orient="records")

    from config import SUPPORTED_GENERATION_MODELS, DEFAULT_GENERATION_MODEL_KEY
    gen_model_cfg = SUPPORTED_GENERATION_MODELS.get(DEFAULT_GENERATION_MODEL_KEY)
    llama_model_path = gen_model_cfg.get("path") if gen_model_cfg else None

    if llama_model_path is None:
        raise ValueError("No path found for the default generation model in config.")

    ctx_size = gen_model_cfg.get("ctx", 2048)
    threads = gen_model_cfg.get("threads", 4)

    # Suppress llama.cpp verbose logs during model load
    llama_model = load_llama_cpp_model(llama_model_path, n_threads=threads, n_ctx=ctx_size)

    print("\n💬 Enter your questions (type 'exit' or 'quit' to end):\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("\n👋 Exiting. Have a great day!")
            break

        print("\n🔍 Retrieving relevant context...")
        contexts = retrieve_contexts(query, embed_model, embeddings, metadata, device)
        prompt = build_prompt(contexts, query, style="qa")

        start = time.time()
        # Suppress llama.cpp verbose logs during generation
        answer = generate_with_llama_cpp(llama_model, prompt)
        end = time.time()

        print("\n[Answer]:")
        print(answer)
        print(f"[Timing] Query response generation took {end - start:.3f} seconds.")
        print("-" * 80)


if __name__ == "__main__":
    main()
