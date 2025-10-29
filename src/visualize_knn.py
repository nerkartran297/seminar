from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_winequality_red, make_quality_labels
from src.plotting import (
    build_knn_pipeline_classifier,
    build_knn_pipeline_regressor,
    compute_neighbors,
    plot_decision_boundary_classification,
    plot_neighbors_overlay,
)


FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate explanatory KNN figures")
    p.add_argument("--mode", choices=["classify", "regress"], default="classify")
    p.add_argument("--feat_x", default="alcohol", choices=FEATURES)
    p.add_argument("--feat_y", default="volatile acidity", choices=FEATURES)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--weights", choices=["uniform", "distance"], default="distance")
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--out_dir", default="reports/figures")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_winequality_red(args.data_dir)

    X2 = df[[args.feat_x, args.feat_y]].copy()
    y_raw = df["quality"]
    if args.mode == "classify":
        y = make_quality_labels(y_raw)
    else:
        y = y_raw.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X2.values, y.values if isinstance(y, pd.Series) else y, test_size=0.2, random_state=args.seed,
        stratify=y if isinstance(y, pd.Series) and str(y.dtype)=="category" else None
    )

    if args.mode == "classify":
        model = build_knn_pipeline_classifier(args.neighbors, args.weights, args.p)
    else:
        model = build_knn_pipeline_regressor(args.neighbors, args.weights, args.p)

    model.fit(X_train, y_train)

    # choose a query point from test set
    qx = X_test[0]

    # compute neighbors in scaled space
    neighbors, manual_pred, proba = compute_neighbors(
        model=model, X_train=X_train, y_train=y_train, query=qx, mode=args.mode
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "classify":
        fig1, _ = plot_decision_boundary_classification(
            X=X_train,
            y=pd.Series(y_train).astype("category"),
            feature_names=(args.feat_x, args.feat_y),
            model=model,
            highlight_point=(qx[0], qx[1]),
            neighbors=neighbors,
            title=f"KNN decision boundary — {args.feat_x} vs {args.feat_y} (k={args.neighbors}, {args.weights})",
        )
        fig2, _ = plot_neighbors_overlay(
            X=X_train,
            y=pd.Series(y_train).astype("category"),
            feature_names=(args.feat_x, args.feat_y),
            query_point=(qx[0], qx[1]),
            neighbors=neighbors,
            title=f"Neighbors around query (manual={manual_pred}, proba={proba if proba is not None else 'n/a'})",
        )
        fig1.savefig(out_dir / "decision_boundary_classification.png", dpi=180)
        fig2.savefig(out_dir / "neighbors_classification.png", dpi=180)
        plt.close(fig1)
        plt.close(fig2)
    else:
        # For regression, reuse overlay plot (colors not meaningful); just plot points
        fig2, _ = plot_neighbors_overlay(
            X=X_train,
            y=np.array([0] * len(X_train)),
            feature_names=(args.feat_x, args.feat_y),
            query_point=(qx[0], qx[1]),
            neighbors=neighbors,
            title=f"Neighbors (manual y={manual_pred:.2f})",
        )
        fig2.savefig(out_dir / "neighbors_regression.png", dpi=180)
        plt.close(fig2)

    # Write a small txt explanation
    with (out_dir / "explanation.txt").open("w", encoding="utf-8") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Features: {args.feat_x}, {args.feat_y}\n")
        f.write(f"k={args.neighbors}, weights={args.weights}, p={args.p}\n")
        f.write(f"Manual prediction: {manual_pred}\n")
        if proba is not None:
            f.write(f"Max predicted probability: {proba:.4f}\n")
        f.write("Neighbors (idx, distance, label, weight):\n")
        for nb in neighbors:
            f.write(f"  {nb.index}, {nb.distance:.4f}, {nb.label}, {nb.weight:.4f}\n")

    # Console-friendly talking points
    print("\n==== Cách trình bày nhanh ====")
    if args.mode == "classify":
        print("- Đây là ranh giới quyết định KNN trên 2 đặc trưng đã chọn.")
        print("- Dấu * là mẫu query; các đường nối đến k láng giềng gần nhất.")
        print("- Nếu weights=distance: láng giềng gần hơn cho trọng số cao hơn (1/(d+ε)).")
        print("- Manual prediction là kết quả 'giải tay' bằng phiếu bầu/trọng số từ các láng giềng.")
    else:
        print("- Với hồi quy: không có ranh giới; ta xem k láng giềng và trung bình có trọng số.")
        print("- Manual prediction là trung bình có trọng số khoảng cách của nhãn láng giềng.")
    print("- Bạn có thể nói: 'K nhỏ nhạy; K lớn mượt; mình chọn K qua thử nghiệm/grid search'.")

    print(f"Saved figures and explanation to {out_dir}")


if __name__ == "__main__":
    main()


