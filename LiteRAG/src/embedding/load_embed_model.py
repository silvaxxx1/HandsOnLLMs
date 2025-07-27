from sentence_transformers import SentenceTransformer
import functools
from config import SUPPORTED_EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL_KEY

@functools.lru_cache(maxsize=1)
def load_embedding_model(model_key: str = DEFAULT_EMBEDDING_MODEL_KEY) -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer embedding model based on model_key.
    """
    model_name = SUPPORTED_EMBEDDING_MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Model key '{model_key}' not found in config.SUPPORTED_EMBEDDING_MODELS.")
    
    print(f"🔍 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("✅ Model loaded successfully.")
    return model
