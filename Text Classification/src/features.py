import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def extract_embedding(text, tokenizer, model, device, max_length=128):
    """
    Extract embedding vector from a single text input.
    Tries pooler_output > CLS token > mean pooling.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.squeeze()
        elif outputs.last_hidden_state.shape[1] > 0:
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        else:
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.detach().cpu().numpy()


def batch_extract_embeddings(texts, tokenizer, model, device, max_length=128, batch_size=32):
    """
    Batch extract embeddings for a list of texts.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_embs = outputs.pooler_output
            elif outputs.last_hidden_state.shape[1] > 0:
                batch_embs = outputs.last_hidden_state[:, 0, :]
            else:
                batch_embs = outputs.last_hidden_state.mean(dim=1)

            batch_embs = batch_embs.cpu().numpy()
            embeddings.append(batch_embs)

    return np.vstack(embeddings)


def extract_sentence_transformer_embeddings(texts, model_name="all-MiniLM-L6-v2", device="cpu", batch_size=32):
    """
    Extract embeddings using sentence-transformers library with batch support.
    """
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings
