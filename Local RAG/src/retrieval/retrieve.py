import os
import torch
import pandas as pd
from config import (
    CHUNKS_CSV_PATH,
    DEFAULT_EMBEDDING_MODEL_KEY,
    get_embeddings_pickle_path,
)
from embedding.load_embed_model import load_embedding_model
from .search import compute_similarity
from .display import print_top_results

def load_embeddings(path: str) -> torch.Tensor:
    import numpy as np
    embeddings = np.load(path)
    return torch.tensor(embeddings)

def main():
    model_key = DEFAULT_EMBEDDING_MODEL_KEY
    print(f"Using embedding model key: {model_key}")

    embeddings_path = get_embeddings_pickle_path(model_key)
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    embeddings = load_embeddings(embeddings_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_embedding_model(model_key)
    model.to(device)
    embeddings = embeddings.to(device)

    df = pd.read_csv(CHUNKS_CSV_PATH)
    chunks_metadata = df.to_dict(orient="records")

    print("\nEnter your queries below (type 'exit' or 'quit' to stop):\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting retrieval.")
            break

        query_embedding = model.encode(query, convert_to_tensor=True).to(device)
        scores = compute_similarity(query_embedding, embeddings, metric="cosine")

        top_k = 5
        top_scores, top_indices = torch.topk(scores, top_k)
        print_top_results(query, top_scores.tolist(), top_indices.tolist(), chunks_metadata, threshold=0.3)

if __name__ == "__main__":
    main()
