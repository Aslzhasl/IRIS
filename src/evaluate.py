import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "iris_recognition_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return its classification metrics."""
    y_pred = model.predict(X_test)

    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def save_results(results, output_path):
    """Save a list of metric dictionaries to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir):
    """Plot and save a confusion matrix heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
