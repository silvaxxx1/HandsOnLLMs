"""
ultra-rag.py — RAG from first principles. Nothing hidden.
The only dependencies: numpy (vectors) and transformers (embeddings + LLM).
Everything else is pure Python.
"""

import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import re

# ─── CONFIG ──────────────────────────────────────────────────────────────
# Everything configurable. Nothing hidden.
FILE_PATH = "document.txt"           # plain text only - no PDF parsing
CHUNK_SIZE = 500                     # characters per chunk
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/DialoGPT-medium"  # small, runs on CPU
TOP_K = 3

# ─── STAGE 1 · LOAD ──────────────────────────────────────────────────────
# Pure Python text loading. No dependencies.
def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ─── STAGE 2 · CHUNK ────────────────────────────────────────────────────
# Simple character-based chunking. No NLP libraries.
def chunk_text(text, size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ─── STAGE 3 · EMBED ────────────────────────────────────────────────────
# Uses transformers, but we show the math explicitly.
def get_embeddings(texts, model, tokenizer):
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Get embeddings (average of token embeddings)
    with torch.no_grad():
        outputs = model(**inputs)
        # Average token embeddings → sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    
    # L2 normalize (so cosine = dot product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# ─── STAGE 4 · RETRIEVE ──────────────────────────────────────────────────
# The core algorithm - fully visible math.
def retrieve(query, chunk_embeddings, chunks, k=TOP_K):
    # Manual cosine similarity calculation
    q_embed = get_embeddings([query], embed_model, tokenizer)[0]
    
    # SIMPLE MATH: cosine = dot product of normalized vectors
    scores = np.dot(chunk_embeddings, q_embed)  # (N,) array
    
    # Get top k indices
    top_indices = np.argsort(scores)[-k:][::-1]
    
    return [(chunks[i], scores[i]) for i in top_indices]

# ─── STAGE 5 · PROMPT ────────────────────────────────────────────────────
# No template - just string concatenation. See everything.
def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([f"[Context {i+1}]\n{chunk}" 
                           for i, (chunk, _) in enumerate(retrieved_chunks)])
    return f"""Answer using ONLY the context below.
If you don't know, say "I don't know."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

# ─── STAGE 6 · GENERATE ──────────────────────────────────────────────────
# Minimum viable LLM call. Shows every step.
def generate(prompt, llm_model, llm_tokenizer):
    inputs = llm_tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,  # deterministic
            pad_token_id=llm_tokenizer.eos_token_id
        )
    
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading models...")
    
    # Load embedding model
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL)
    
    # Load LLM
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    
    print("Processing document...")
    text = load_text(FILE_PATH)
    chunks = chunk_text(text)
    
    print(f"Created {len(chunks)} chunks")
    chunk_embeddings = get_embeddings(chunks, embed_model, tokenizer)
    
    # Query loop
    while True:
        query = input("\nQuestion (or 'quit'): ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        
        # Retrieve
        retrieved = retrieve(query, chunk_embeddings, chunks)
        
        # Show what was retrieved (critical for understanding)
        print("\n--- RETRIEVED CHUNKS ---")
        for i, (chunk, score) in enumerate(retrieved, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   {chunk[:100]}...")
        
        # Generate
        prompt = build_prompt(query, retrieved)
        answer = generate(prompt, llm_model, llm_tokenizer)
        
        print("\n--- ANSWER ---")
        print(answer)
