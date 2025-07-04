from datasets import load_dataset
from transformers import AutoTokenizer
from config import CONFIG

def load_data(split="train"):
    """
    Load the emotion dataset and return texts + labels
    """
    dataset = load_dataset(CONFIG["dataset_name"])
    texts = dataset[split][CONFIG["text_column"]]
    labels = dataset[split][CONFIG["label_column"]]
    return texts, labels

def get_dataset_splits():
    """
    Return full dataset splits (train, validation, test)
    """
    dataset = load_dataset(CONFIG["dataset_name"])
    return dataset["train"], dataset["validation"], dataset["test"]

def tokenize_data(texts, model_name, max_length=128):
    """
    Tokenize input texts using Hugging Face tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
