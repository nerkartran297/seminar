from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_winequality_red, make_quality_labels
from src.models import build_knn_classifier, build_knn_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wine Quality Prediction with KNN")
    parser.add_argument(
        "--mode",
        choices=["classify", "regress"],
        default="classify",
        help="Run classification or regression",
    )
    parser.add_argument("--data_dir", default="data", help="Directory containing csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--neighbors",
        default="3,5,7,9,11,13,15,17,19,21",
        help="Comma-separated k values for grid search",
    )
    parser.add_argument(
        "--save_model",
        default=None,
        help="Path to save best model (e.g., models/best_knn.joblib)",
    )
    parser.add_argument(
        "--no_explain",
        action="store_true",
        help="Disable explanatory summary text in the console",
    )
    return parser.parse_args()


def _parse_neighbors(arg: str) -> List[int]:
    try:
        return [int(x.strip()) for x in arg.split(",") if x.strip()]
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--neighbors must be a comma-separated list of integers") from exc


def _train_classification(df: pd.DataFrame, args: argparse.Namespace) -> None:
    X = df.drop(columns=["quality"])  # features
    y = make_quality_labels(df["quality"])  # 3-class labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = build_knn_classifier()
    param_grid = {
        "knn__n_neighbors": _parse_neighbors(args.neighbors),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=args.cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("CV best accuracy:", f"{grid.best_score_:.4f}")

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print("Test accuracy:", f"{metrics.accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:\n", metrics.classification_report(y_test, y_pred))
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

    if not args.no_explain:
        print("\n==== Giải thích nhanh (đọc khi thuyết trình) ====")
        print("- Bài toán classification: chuyển quality thành 3 lớp (poor/average/good).")
        print("- KNN dùng nguyên tắc 'gần thì giống': mẫu mới được gán nhãn theo đa số láng giềng.")
        print("- Tiền xử lý: chuẩn hoá bằng StandardScaler để khoảng cách giữa đặc trưng công bằng.")
        print(
            f"- Mình dùng Cross-Validation (cv={args.cv}) để chọn tham số tốt: {grid.best_params_}."
        )
        print(
            "- Cách giải tay: lấy k mẫu gần nhất, nếu weights=distance thì bầu có trọng số 1/(d+ε)."
        )
        print(
            "- Đọc kết quả: accuracy test ở trên là tỷ lệ đúng; xem thêm precision/recall/F1 và ma trận nhầm lẫn."
        )
        print("- Ý nghĩa k: k nhỏ nhạy (dễ overfit); k lớn mượt hơn nhưng có thể mất chi tiết (underfit).\n")

    if args.save_model:
        from joblib import dump

        out_path = Path(args.save_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump(best, out_path.as_posix())
        print(f"Saved model to {out_path}")


def _train_regression(df: pd.DataFrame, args: argparse.Namespace) -> None:
    X = df.drop(columns=["quality"])  # features
    y = df["quality"].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = build_knn_regressor()
    param_grid = {
        "knn__n_neighbors": _parse_neighbors(args.neighbors),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=args.cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("CV best (neg RMSE):", f"{grid.best_score_:.4f}")

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    print("Test RMSE:", f"{rmse:.4f}")
    print("Test MAE:", f"{mae:.4f}")
    print("Test R2:", f"{r2:.4f}")

    if not args.no_explain:
        print("\n==== Giải thích nhanh (đọc khi thuyết trình) ====")
        print("- Bài toán regression: dự đoán điểm quality. KNN dự đoán bằng trung bình (hoặc trung bình có trọng số theo khoảng cách).")
        print("- Tiền xử lý: chuẩn hoá bằng StandardScaler vì KNN dựa vào khoảng cách.")
        print(
            f"- Dùng Cross-Validation (cv={args.cv}) để chọn tham số tốt: {grid.best_params_}."
        )
        print("- Đọc kết quả: RMSE, MAE càng nhỏ càng tốt; R2 gần 1 là tốt (0 là baseline trung bình).\n")

    if args.save_model:
        from joblib import dump

        out_path = Path(args.save_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump(best, out_path.as_posix())
        print(f"Saved model to {out_path}")


def main() -> None:
    args = parse_args()
    df = load_winequality_red(data_dir=args.data_dir)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    if args.mode == "classify":
        _train_classification(df, args)
    else:
        _train_regression(df, args)


if __name__ == "__main__":
    main()


