import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
from config import CONFIG
from .eval import evaluate
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(examples[CONFIG["text_column"]], truncation=True, padding="max_length", max_length=max_length)

def run_finetune(config):
    model_name = config["finetune"]["model"]
    output_dir = config["finetune"]["output_dir"]
    num_epochs = config["finetune"]["num_epochs"]
    batch_size = config["finetune"]["batch_size"]
    learning_rate = config["finetune"].get("learning_rate", 2e-5)
    device = config["finetune"]["device"]

    max_length = config.get("max_length", 128)  # optionally pass max_length via config if needed

    print("[INFO] Loading dataset...")
    dataset = load_dataset(config["dataset_name"])
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["validation"],
        "test": dataset["test"]
    })

    print(f"[INFO] Loading tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(config["labels"])
    ).to(device)

    print("[INFO] Tokenizing dataset...")
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        # Remove evaluation_strategy (no eval during training)
        # Remove or disable load_best_model_at_end
        load_best_model_at_end=False,  # disable this
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        seed=42,
        report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Evaluating on test set...")
    predictions = trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Use evaluate helper to print and save report and confusion matrix
    evaluate(
        y_true=labels,
        y_pred=preds,
        labels=config["labels"],
        out_dir=config["paths"]["outputs_dir"],
        name="finetune_bert"
    )

    print(f"[INFO] Saving model to {output_dir}...")
    trainer.save_model(output_dir)
