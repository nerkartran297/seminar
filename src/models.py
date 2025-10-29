from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn_classifier(
    n_neighbors: int = 15,
    weights: str = "distance",
    p: int = 2,
) -> Pipeline:
    """Create a KNN classifier pipeline with scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)),
        ]
    )


def build_knn_regressor(
    n_neighbors: int = 15,
    weights: str = "distance",
    p: int = 2,
) -> Pipeline:
    """Create a KNN regressor pipeline with scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)),
        ]
    )


