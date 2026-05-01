from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_voting_ensemble(X_train, y_train):
    """Train and return a soft-voting ensemble classifier."""
    ensemble = VotingClassifier(
        estimators=[
            (
                "svm",
                SVC(kernel="rbf", probability=True, random_state=42),
            ),
            (
                "random_forest",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=3),
            ),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    return ensemble
