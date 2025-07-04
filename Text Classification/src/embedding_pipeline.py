import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from config import CONFIG
from .features import extract_sentence_transformer_embeddings
from .eval import evaluate  


def run_embedding_pipeline(config):
    model_name = config["embedding"]["model"]
    device = config["embedding"]["device"]
    save_path = config["embedding"]["save_path"]
    batch_size = config["embedding"].get("batch_size", 32)

    os.makedirs(save_path, exist_ok=True)

    print("[INFO] Loading dataset...")
    dataset = load_dataset(config["dataset_name"])
    train_texts = dataset["train"][config["text_column"]]
    test_texts = dataset["test"][config["text_column"]]
    y_train = dataset["train"][config["label_column"]]
    y_test = dataset["test"][config["label_column"]]

    print(f"[INFO] Extracting embeddings with {model_name} on {device}...")
    X_train = extract_sentence_transformer_embeddings(train_texts, model_name=model_name, device=device, batch_size=batch_size)
    X_test = extract_sentence_transformer_embeddings(test_texts, model_name=model_name, device=device, batch_size=batch_size)

    # Save embeddings
    np.save(os.path.join(save_path, "X_train.npy"), X_train)
    np.save(os.path.join(save_path, "X_test.npy"), X_test)

    # Train classical classifier
    print("[INFO] Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating classifier...")
    y_pred = clf.predict(X_test)

    evaluate(
        y_true=y_test,
        y_pred=y_pred,
        labels=config["labels"],
        out_dir=config["paths"]["outputs_dir"],
        name="embedding_logreg"
    )

    # Save model
    model_path = os.path.join(save_path, "embedding_logreg_model.joblib")
    joblib.dump(clf, model_path)
    print(f"[INFO] Saved model to {model_path}")
