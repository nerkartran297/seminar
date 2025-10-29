from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_COLORS = ["#e45756", "#4c78a8", "#72b7b2"]
CLASS_MARKERS = ["o", "s", "^"]


@dataclass
class NeighborInfo:
    index: int
    distance: float
    label: str | float
    weight: float


def build_knn_pipeline_classifier(n_neighbors: int, weights: str, p: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)),
        ]
    )


def build_knn_pipeline_regressor(n_neighbors: int, weights: str, p: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)),
        ]
    )


def plot_decision_boundary_classification(
    X: np.ndarray,
    y: pd.Series,
    feature_names: Tuple[str, str],
    model: Pipeline,
    highlight_point: Tuple[float, float] | None = None,
    neighbors: List[NeighborInfo] | None = None,
    title: str | None = None,
    figsize: Tuple[int, int] = (7, 6),
):
    # mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict class labels on the grid and map to numeric indices for contourf
    classes = list(y.cat.categories if hasattr(y, "cat") else np.unique(y))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    Z_labels = model.predict(grid)
    try:
        Z_numeric = np.vectorize(lambda v: class_to_idx.get(v, np.nan))(Z_labels)
    except Exception:
        # Fallback: numeric labels already
        Z_numeric = Z_labels
    Z = np.asarray(Z_numeric, dtype=float).reshape(xx.shape)
    cmap_light = ListedColormap(CLASS_COLORS[: len(classes)])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_light)

    for i, cls in enumerate(classes):
        mask = (y == cls).values if isinstance(y, pd.Series) else y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=CLASS_COLORS[i],
            marker=CLASS_MARKERS[i % len(CLASS_MARKERS)],
            edgecolor="k",
            s=35,
            label=str(cls),
        )

    if highlight_point is not None:
        ax.scatter(
            [highlight_point[0]],
            [highlight_point[1]],
            c="black",
            marker="*",
            s=180,
            label="query",
        )
    if neighbors:
        for nb in neighbors:
            # draw a faint line from query to neighbor
            ax.plot(
                [highlight_point[0], X[nb.index, 0]],
                [highlight_point[1], X[nb.index, 1]],
                color="#555555",
                linewidth=1,
                alpha=0.4,
            )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(title="Class")
    ax.set_title(title or "KNN decision boundary")
    fig.tight_layout()
    return fig, ax


def compute_neighbors(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    query: np.ndarray,
    mode: str = "classify",
) -> Tuple[List[NeighborInfo], float | str, float | None]:
    scaler = model.named_steps["scaler"]
    knn = model.named_steps["knn"]  # type: ignore[assignment]

    Xs = scaler.transform(X_train)
    qs = scaler.transform(query.reshape(1, -1))
    distances, indices = knn.kneighbors(qs, n_neighbors=knn.n_neighbors)
    distances = distances[0]
    indices = indices[0]

    eps = 1e-8
    if getattr(knn, "weights") == "distance":
        weights = 1.0 / (distances + eps)
    else:
        weights = np.ones_like(distances)

    neighbors: List[NeighborInfo] = []
    for d, idx, w in zip(distances, indices, weights):
        neighbors.append(
            NeighborInfo(index=int(idx), distance=float(d), label=y_train[idx], weight=float(w))
        )

    if mode == "classify":
        # Manual weighted vote
        labels = np.array([n.label for n in neighbors])
        unique = pd.unique(labels)
        votes = {u: 0.0 for u in unique}
        for n in neighbors:
            votes[n.label] += n.weight
        manual_pred = max(votes.items(), key=lambda kv: kv[1])[0]
        proba = None
        if hasattr(knn, "predict_proba"):
            proba = float(np.max(knn.predict_proba(qs)))
        return neighbors, str(manual_pred), proba
    else:
        # Weighted average
        num = sum(n.label * n.weight for n in neighbors)  # type: ignore[operator]
        den = sum(n.weight for n in neighbors)
        manual_pred = float(num / den)
        return neighbors, manual_pred, None


def plot_neighbors_overlay(
    X: np.ndarray,
    y: pd.Series | np.ndarray,
    feature_names: Tuple[str, str],
    query_point: Tuple[float, float],
    neighbors: List[NeighborInfo],
    title: str | None = None,
    figsize: Tuple[int, int] = (7, 6),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    classes = list(y.cat.categories if hasattr(y, "cat") else np.unique(y))
    for i, cls in enumerate(classes):
        mask = (y == cls).values if isinstance(y, pd.Series) else y == cls
        ax.scatter(
            X[mask, 0], X[mask, 1], c=CLASS_COLORS[i], marker=CLASS_MARKERS[i % 3], edgecolor="k", s=35, label=str(cls)
        )
    ax.scatter([query_point[0]], [query_point[1]], c="black", marker="*", s=180, label="query")
    for nb in neighbors:
        ax.plot(
            [query_point[0], X[nb.index, 0]],
            [query_point[1], X[nb.index, 1]],
            color="#444444",
            linewidth=1,
            alpha=0.5,
        )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()
    ax.set_title(title or "KNN neighbors")
    fig.tight_layout()
    return fig, ax


