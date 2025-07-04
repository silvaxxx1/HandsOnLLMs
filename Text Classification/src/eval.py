import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(y_true, y_pred, labels, out_dir="outputs", name="default"):
    os.makedirs(out_dir, exist_ok=True)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=labels))

    # Save JSON
    report_path = os.path.join(out_dir, f"{name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_path = os.path.join(out_dir, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"[INFO] Saved report to: {report_path}")
    print(f"[INFO] Saved confusion matrix to: {cm_path}")

    return report
