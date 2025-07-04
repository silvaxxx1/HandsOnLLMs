# config.py

CONFIG = {
    # Dataset
    "dataset_name": "emotion",
    "text_column": "text",
    "label_column": "label",
    "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],

    # Feature extraction (hidden states from HF model)
    "feature": {
        "model": "distilbert-base-uncased",
        "save_path": "outputs/features/",
        "device": "cpu"
    },

    # Embedding with Sentence-Transformer (replaces few-shot)
    "embedding": {
        "model": "all-MiniLM-L6-v2",
        "save_path": "outputs/embeddings/",
        "device": "cpu"
    },

    # Fine-tuning Hugging Face model
    "finetune": {
        "model": "distilbert-base-uncased",
        "output_dir": "models/finetuned/",
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "device": "cpu"
    },

    "zero_shot": {
    "model": "gpt-3.5-turbo",  # or "gpt-4"
    "api_key_env": "OPENAI_API_KEY"
},
    "paths": {
        "models_dir": "models/",
        "outputs_dir": "outputs/",
        "logs_dir": "outputs/logs/"
    }
}
