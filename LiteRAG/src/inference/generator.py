"""
inference/generator.py — Stage 6: generate an answer with a local LLM.

WHY llama.cpp + GGUF?
  TinyLlama quantised to Q5_K_M runs at ~4 tokens/sec on a modern CPU.
  No GPU required. No internet. No API key. Fully offline.
  GGUF is llama.cpp's native format — optimised for CPU inference.

TEMPERATURE:
  Controls how the model samples the next token:
    0.0 → always pick highest-probability token (deterministic, factual)
    0.2 → slight randomness — our default for document Q&A
    1.0 → sample proportionally (creative, less focused)
"""

import logging
from llama_cpp import Llama
from config import LLM_PATH, N_CTX, N_THREADS, TEMPERATURE, MAX_TOKENS


def load_llm(model_path: str = LLM_PATH) -> Llama:
    """
    Load TinyLlama from a local GGUF file via llama.cpp.
    This is the heaviest setup step — do it once and reuse the object.
    """
    logging.info(f"Loading LLM from '{model_path}' ...")
    return Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        use_mlock=True,   # lock model weights in RAM — prevents swapping
        verbose=False,    # set True to see llama.cpp token-level debug output
    )


def generate(llm: Llama, prompt: str) -> str:
    """
    Run autoregressive generation: the model predicts one token at a time,
    each conditioned on the full prompt + all tokens generated so far,
    until it hits a stop token or MAX_TOKENS is reached.
    """
    out = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["</s>", "<|user|>"],   # TinyLlama's end-of-turn tokens
        echo=False,                  # don't repeat the prompt in the output
    )
    return out["choices"][0]["text"].strip()
