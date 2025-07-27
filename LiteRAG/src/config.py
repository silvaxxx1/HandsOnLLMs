import os

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDING_DIR = os.path.join(BASE_DIR, "embedding")
RETRIEVAL_DIR = os.path.join(BASE_DIR, "retrieval")
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Create directories if they don't exist
for directory in [DATA_DIR, EMBEDDING_DIR, RETRIEVAL_DIR, INFERENCE_DIR, ASSETS_DIR]:
    os.makedirs(directory, exist_ok=True)

# === Output File Paths ===
CHUNKS_CSV_PATH = os.path.join(ASSETS_DIR, "chunks_output.csv")

def get_embeddings_csv_path(model_key: str) -> str:
    return os.path.join(ASSETS_DIR, f"embeddings_{model_key}.csv")

def get_embeddings_pickle_path(model_key: str) -> str:
    return os.path.join(ASSETS_DIR, f"embeddings_{model_key}.pkl")

# === Embedding Models ===
SUPPORTED_EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "paraphrase_minilm": "paraphrase-MiniLM-L12-v2",
    "paraphrase_multilingual_minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

# Default embedding model key
DEFAULT_EMBEDDING_MODEL_KEY = "minilm"

# === Generation Models (Local Friendly) ===
SUPPORTED_GENERATION_MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local": True,
        "quantized": True,
        "max_tokens": 2048
    },

   "tinyllama_llamacpp": {
    "engine": "llamacpp",
    "path": os.path.join(BASE_DIR, "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"),  # or use a relative path "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    "ctx": 2048,
    "threads": 8,
    },
    
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "local": True,
        "quantized": True,
        "max_tokens": 4096
    }

}

DEFAULT_GENERATION_MODEL_KEY = "tinyllama_llamacpp"

# === Chunking Parameters ===
CHUNK_SIZE = 10           # Number of sentences per chunk
CHUNK_OVERLAP = 0         # Sentence overlap between chunks

# === Retrieval Parameters ===
TOP_K = 5                 # How many top chunks to return per query

# === Logging ===
LOGGING_LEVEL = "INFO"

# === Debug Mode ===
DEBUG = False


