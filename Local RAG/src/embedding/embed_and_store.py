import os
import pandas as pd
import numpy as np
from config import (
    CHUNKS_CSV_PATH,
    get_embeddings_csv_path,
    get_embeddings_pickle_path,
    DEFAULT_EMBEDDING_MODEL_KEY,
)
from .load_embed_model import load_embedding_model


def embed_texts(texts: list[str], model) -> np.ndarray:
    print(f"🔗 Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def save_embeddings_csv(embeddings: np.ndarray, df: pd.DataFrame, path: str):
    """Save embeddings as strings in CSV (for inspection and portability)"""
    embedding_strs = [",".join(map(str, emb)) for emb in embeddings]
    df_copy = df.copy()
    df_copy["embedding"] = embedding_strs

    print(f"💾 Saving embeddings to CSV: {path}")
    df_copy.to_csv(path, index=False)
    print("✅ Embeddings saved to CSV.")


def save_embeddings_pickle(embeddings: np.ndarray, path: str):
    """Save raw embeddings as .pkl (for fast loading into vector DB)"""
    print(f"💾 Saving raw embeddings to pickle: {path}")
    with open(path, "wb") as f:
        np.save(f, embeddings)
    print("✅ Embeddings saved to pickle.")


def main(model_key: str = DEFAULT_EMBEDDING_MODEL_KEY):
    embeddings_csv_path = get_embeddings_csv_path(model_key)
    embeddings_pickle_path = get_embeddings_pickle_path(model_key)

    if os.path.exists(embeddings_csv_path):
        existing_df = pd.read_csv(embeddings_csv_path)
        if "embedding" in existing_df.columns and existing_df["embedding"].notna().all():
            print(f"✅ Embeddings for model '{model_key}' already exist. Skipping embedding step.")
            return
        else:
            print(f"⚠️ Embeddings CSV for model '{model_key}' exists but incomplete. Re-embedding...")

    if not os.path.exists(CHUNKS_CSV_PATH):
        raise FileNotFoundError(f"❌ Cannot find chunks CSV at: {CHUNKS_CSV_PATH}")

    df = pd.read_csv(CHUNKS_CSV_PATH)
    if "chunk_text" not in df.columns:
        raise KeyError("❌ 'chunk_text' column is missing in chunks CSV.")

    texts = df["chunk_text"].dropna().tolist()
    if not texts:
        raise ValueError("❌ No valid text chunks found for embedding.")

    model = load_embedding_model(model_key)
    embeddings = embed_texts(texts, model)

    save_embeddings_csv(embeddings, df, embeddings_csv_path)
    save_embeddings_pickle(embeddings, embeddings_pickle_path)


if __name__ == "__main__":
    # Change the model_key here to switch models
    main(model_key=DEFAULT_EMBEDDING_MODEL_KEY)
