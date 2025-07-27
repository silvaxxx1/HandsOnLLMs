# retrieval/embed_query.py

import torch
from typing import Union
from .load_embed_model import load_embedding_model

def embed_query(query: Union[str, list[str]], model_key: str = None, device: str = "cpu") -> torch.Tensor:
    """
    Embed a single query string or list of queries and return a tensor.
    """
    model = load_embedding_model(model_key) if model_key else load_embedding_model()
    
    # Encode query with tensor output on requested device
    embeddings = model.encode(query, convert_to_tensor=True, device=device)
    
    return embeddings
