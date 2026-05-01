from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_svm(X_train, y_train):
    """Train and return an SVM classifier with an RBF kernel."""
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train and return a random forest classifier."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    """Train and return a k-nearest neighbors classifier."""
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    """Save a trained model to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
