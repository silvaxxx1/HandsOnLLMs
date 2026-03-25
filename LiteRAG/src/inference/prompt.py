"""
inference/prompt.py — Stage 5: build the prompt that grounds the LLM.

This is the "augmented" in Retrieval-Augmented Generation.
We inject retrieved chunks as explicit context so the LLM answers
from YOUR document — not from its training data (hallucination).

PROMPT TEMPLATE NOTE:
  <|system|> / <|user|> / <|assistant|> are TinyLlama's chat tags.
  Other models use different formats:
    Mistral  → [INST] ... [/INST]
    Qwen     → <|im_start|> ... <|im_end|>
    LLaMA-3  → <|begin_of_text|><|start_header_id|> ...
  Always match the template to the model or generation quality drops.
"""

from config import TOP_K


def build_prompt(query: str, hits: list[dict]) -> str:
    """
    Construct a grounded prompt from retrieved chunks and the user query.

    Each hit is labelled with its page number so the LLM (and the user)
    can trace every claim back to a specific page of the source document.
    """
    context = "\n\n".join(
        f"[Context {i+1} — Page {h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )

    return (
        f"<|system|>\n"
        f"You are a helpful assistant. "
        f"Answer using ONLY the context provided below. "
        f"If the answer is not present, say \"I don't know.\"\n"
        f"</s>\n"
        f"<|user|>\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"</s>\n"
        f"<|assistant|>"
    )
