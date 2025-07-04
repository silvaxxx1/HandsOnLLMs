import os
import joblib
from sklearn.linear_model import LogisticRegression
from .data import load_data
from .features import batch_extract_embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from config import CONFIG
from .eval import evaluate  


def run_feature_pipeline(config):
    device = config["feature"]["device"]
    model_name = config["feature"]["model"]
    save_path = config["feature"]["save_path"]

    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)

    print("[INFO] Loading data...")
    train_texts, y_train = load_data("train")
    test_texts, y_test = load_data("test")

    print(f"[INFO] Loading model {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    print("[INFO] Extracting features for train set...")
    X_train = batch_extract_embeddings(train_texts, tokenizer, model, device)

    print("[INFO] Extracting features for test set...")
    X_test = batch_extract_embeddings(test_texts, tokenizer, model, device)

    # Save extracted features for reproducibility
    train_feat_path = os.path.join(save_path, "train_features.npy")
    test_feat_path = os.path.join(save_path, "test_features.npy")
    print(f"[INFO] Saving features to {save_path}")
    np.save(train_feat_path, X_train)
    np.save(test_feat_path, X_test)

    print("[INFO] Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating classifier...")
    y_pred = clf.predict(X_test)

    # Use the unified evaluation function here
    evaluate(
        y_true=y_test,
        y_pred=y_pred,
        labels=config["labels"],
        out_dir=config["paths"]["outputs_dir"],
        name="feature_logreg"
    )

    # Save trained model
    model_save_path = os.path.join(save_path, "logreg_model.joblib")
    joblib.dump(clf, model_save_path)
    print(f"[INFO] Saved trained classifier to {model_save_path}")
