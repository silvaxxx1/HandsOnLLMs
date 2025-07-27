import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from config import (
    BASE_DIR,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_GENERATION_MODELS,
    DEFAULT_EMBEDDING_MODEL_KEY,
    DEFAULT_GENERATION_MODEL_KEY,
)

def load_embedding_model(model_key=None):
    model_key = model_key or DEFAULT_EMBEDDING_MODEL_KEY
    model_name = SUPPORTED_EMBEDDING_MODELS.get(model_key)

    if model_name is None:
        raise ValueError(f"Embedding model '{model_key}' not supported.")
    
    print(f"[Embedding] Loading model: {model_name}")
    return SentenceTransformer(model_name)


def load_generation_model(model_key=None, device=None):
    model_key = model_key or DEFAULT_GENERATION_MODEL_KEY
    model_config = SUPPORTED_GENERATION_MODELS.get(model_key)

    if model_config is None:
        raise ValueError(f"Generation model '{model_key}' not supported.")

    if model_config.get("engine") == "llamacpp":
        # Resolve absolute path relative to BASE_DIR
        model_path = model_config["path"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
        # Pass context size (n_ctx) and threads from config to the llama.cpp model loader
        return load_llamacpp_generation_model(
            model_path=model_path,
            n_ctx=model_config.get("ctx", 2048),    # Fix here: pass n_ctx explicitly
            n_threads=model_config.get("threads", 4)
        )

    model_name = model_config["name"]

    print(f"[Generation] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )

    return generator


def load_llamacpp_generation_model(model_path, n_ctx=2048, n_threads=4):
    from llama_cpp import Llama
    print(f"[llama.cpp] Loading GGUF model from: {model_path} with context size: {n_ctx}")
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,           # Pass context window size here
        n_threads=n_threads,
        use_mlock=True,
        verbose=False
    )
