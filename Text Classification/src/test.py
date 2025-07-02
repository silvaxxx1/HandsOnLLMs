import torch
from transformers import AutoTokenizer, AutoModel

def test_model_load():
    model_name = "distilbert-base-uncased"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # inference mode

    text = "This is a test sentence for feature extraction."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output.squeeze()
        else:
            emb = outputs.last_hidden_state.mean(dim=1).squeeze()

    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding snippet: {emb[:5]}")

if __name__ == "__main__":
    test_model_load()
