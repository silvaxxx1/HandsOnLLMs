import os
import pandas as pd
import numpy as np
from config import CHUNKS_CSV_PATH, get_embeddings_csv_path, get_embeddings_pickle_path

def load_chunks_metadata():
    """
    Load chunk metadata CSV (text chunks + page numbers, etc.)
    """
    if not os.path.exists(CHUNKS_CSV_PATH):
        raise FileNotFoundError(f"Chunks CSV not found at {CHUNKS_CSV_PATH}")
    df = pd.read_csv(CHUNKS_CSV_PATH)
    return df

def load_embeddings_pickle(model_key: str):
    """
    Load raw embeddings from pickle file for the given model_key.
    """
    path = get_embeddings_pickle_path(model_key)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings pickle file not found at {path}")
    embeddings = np.load(path)
    return embeddings

def load_embeddings_csv(model_key: str):
    """
    Optionally, load embeddings from CSV if needed (embeddings stored as strings).
    """
    path = get_embeddings_csv_path(model_key)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings CSV file not found at {path}")
    df = pd.read_csv(path)
    # Convert embedding strings back to list/array
    embeddings = df["embedding"].apply(lambda x: np.array(list(map(float, x.split(","))))).tolist()
    return np.array(embeddings)
