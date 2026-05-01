from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.classical_models import (
    save_model,
    train_knn,
    train_random_forest,
    train_svm,
)
from src.cnn_model import evaluate_cnn, train_cnn
from src.ensemble import train_voting_ensemble
from src.evaluate import evaluate_model, plot_confusion_matrix, save_results
from src.features import (
    extract_hog_features,
    extract_lbp_features,
    extract_orb_features,
)
from src.preprocessing import load_dataset


RANDOM_STATE = 42
DATA_DIR = Path("data/raw/iris")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "results.csv"


def _cnn_metrics(y_true, y_pred, model_name):
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _evaluate_classical_model(model, X_test, y_test, class_names, model_name):
    results = evaluate_model(model, X_test, y_test, model_name)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, class_names, model_name, RESULTS_DIR)
    return results


def main():
    try:
        torch.manual_seed(RANDOM_STATE)

        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading dataset from {DATA_DIR}...")
        X_images, y_labels, class_names = load_dataset(DATA_DIR)

        if len(X_images) == 0:
            raise ValueError(f"No valid images found in {DATA_DIR}")

        if len(class_names) < 2:
            raise ValueError("At least two class folders are required for training")

        print(f"Loaded {len(X_images)} images from {len(class_names)} classes.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_images,
            y_labels,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_labels,
        )

        results = []

        print("Extracting LBP features...")
        X_train_lbp = extract_lbp_features(X_train)
        X_test_lbp = extract_lbp_features(X_test)

        print("Training SVM on LBP features...")
        svm_model = train_svm(X_train_lbp, y_train)
        save_model(svm_model, MODELS_DIR / "svm_lbp.joblib")
        results.append(
            _evaluate_classical_model(
                svm_model,
                X_test_lbp,
                y_test,
                class_names,
                "SVM_LBP",
            )
        )

        print("Training voting ensemble on LBP features...")
        ensemble_model = train_voting_ensemble(X_train_lbp, y_train)
        save_model(ensemble_model, MODELS_DIR / "voting_ensemble_lbp.joblib")
        results.append(
            _evaluate_classical_model(
                ensemble_model,
                X_test_lbp,
                y_test,
                class_names,
                "Voting_Ensemble_LBP",
            )
        )

        print("Extracting HOG features...")
        X_train_hog = extract_hog_features(X_train)
        X_test_hog = extract_hog_features(X_test)

        print("Training RandomForest on HOG features...")
        random_forest_model = train_random_forest(X_train_hog, y_train)
        save_model(random_forest_model, MODELS_DIR / "random_forest_hog.joblib")
        results.append(
            _evaluate_classical_model(
                random_forest_model,
                X_test_hog,
                y_test,
                class_names,
                "RandomForest_HOG",
            )
        )

        print("Extracting ORB features...")
        X_train_orb = extract_orb_features(X_train)
        X_test_orb = extract_orb_features(X_test)

        print("Training KNN on ORB features...")
        knn_model = train_knn(X_train_orb, y_train)
        save_model(knn_model, MODELS_DIR / "knn_orb.joblib")
        results.append(
            _evaluate_classical_model(
                knn_model,
                X_test_orb,
                y_test,
                class_names,
                "KNN_ORB",
            )
        )

        print("Splitting CNN train/validation data...")
        X_cnn_train, X_cnn_val, y_cnn_train, y_cnn_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        print("Training CNN on CPU...")
        cnn_model = train_cnn(
            X_cnn_train,
            y_cnn_train,
            X_cnn_val,
            y_cnn_val,
            num_classes=len(class_names),
            model_path=MODELS_DIR / "cnn_model.pth",
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
        )

        print("Evaluating CNN...")
        cnn_y_true, cnn_y_pred = evaluate_cnn(cnn_model, X_test, y_test, batch_size=32)
        results.append(_cnn_metrics(cnn_y_true, cnn_y_pred, "CNN"))
        plot_confusion_matrix(cnn_y_true, cnn_y_pred, class_names, "CNN", RESULTS_DIR)

        save_results(results, RESULTS_PATH)

        comparison = pd.DataFrame(results)
        print("\nFinal comparison table:")
        print(comparison.to_string(index=False))
        print(f"\nSaved results to {RESULTS_PATH}")
        print(f"Saved models to {MODELS_DIR}")
        print(f"Saved confusion matrices to {RESULTS_DIR}")

    except Exception as error:
        print(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()
