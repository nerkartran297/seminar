## Hướng dẫn train KNN (Dự án Wine Quality)

Tài liệu này giải thích chi tiết cách KNN được huấn luyện trong dự án, kèm trích dẫn mã nguồn để bạn có thể đọc trực tiếp khi thuyết trình.

### 1) Tải dữ liệu và tạo nhãn

- Dữ liệu đọc từ `data/winequality-red.csv` (nếu chưa có, sẽ tự tải từ UCI).
- Với phân loại (classification), ta ánh xạ cột `quality` (số) thành 3 nhãn: `poor` (≤4), `average` (5–6), `good` (≥7).

```55:73:src/data.py
def make_quality_labels(quality: pd.Series, scheme: str = "3class") -> pd.Series:
    """Map numeric quality scores to categorical labels.

    scheme="3class": poor (<=4), average (5-6), good (>=7)
    Returns a pandas Series of dtype category with ordered classes.
    """
    if scheme != "3class":
        raise ValueError("Only '3class' scheme is supported currently.")

    labels = (
        quality
        .apply(lambda q: "poor" if q <= 4 else ("average" if q <= 6 else "good"))
        .astype("category")
    )
    labels = labels.cat.set_categories(["poor", "average", "good"], ordered=True)
    return labels
```

Ghi chú: Nếu muốn dùng cách cắt nhãn khác (ví dụ: `low ≤5`, `medium = 6`, `high ≥7`), có thể sửa nhanh logic trong hàm trên.

### 2) Pipeline mô hình

KNN dựa trên khoảng cách, nên cần chuẩn hoá đặc trưng trước (StandardScaler), sau đó mới áp dụng KNN.

```8:19:src/models.py
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
```

```22:33:src/models.py
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
```

Tham số quan trọng:

- `n_neighbors` (k): số láng giềng dùng để dự đoán.
- `weights`: `uniform` (mọi láng giềng như nhau) hoặc `distance` (gần hơn nặng hơn, w≈1/(d+ε)).
- `p`: metric Minkowski; `p=2` là Euclidean, `p=1` là Manhattan.

### 3) Quy trình train — Classification

```59:90:src/train.py
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
```

Giải thích:

- Tách train/test có `stratify=y` để giữ cân bằng lớp.
- Tìm tham số tốt bằng Cross-Validation với lưới `k`, `weights`, `p`.
- Đánh giá cuối trên test set: accuracy, precision/recall/F1, ma trận nhầm lẫn.

### 4) Quy trình train — Regression

```117:151:src/train.py
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
```

Giải thích:

- Hồi quy dự đoán giá trị số `quality`. Chỉ số: RMSE, MAE (càng nhỏ càng tốt), R² (gần 1 càng tốt).

### 5) Tham số dòng lệnh và tái lập kết quả

Script expose các tham số quan trọng (chế độ, số folds CV, lưới k, weights, p, tỉ lệ test, random_state, đường dẫn lưu model).

```22:49:src/train.py
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
```

Khuyến nghị chạy:

- Classification: `python src/train.py --mode classify`
- Regression: `python src/train.py --mode regress`
- Thử danh sách k khác: `--neighbors 3,5,7,9,11,13,15`
- Lưu model: `--save_model models/best_knn_cls.joblib`

### 6) Vì sao cần scale? Chọn k/weights/p thế nào?

- Scale (StandardScaler) để mọi đặc trưng đóng góp công bằng vào khoảng cách.
- k nhỏ ⇒ mô hình nhạy, dễ overfit; k lớn ⇒ mượt hơn, có thể underfit.
- weights=distance cho láng giềng gần ảnh hưởng lớn hơn.
- p=2 (Euclidean) thường là mặc định tốt; p=1 (Manhattan) có thể ổn hơn nếu đặc trưng có đuôi dài theo từng trục.

### 7) Thuật toán tìm láng giềng (brute/kd-tree/ball-tree)

- Scikit-learn mặc định `algorithm='auto'`. Với dữ liệu nhỏ, ~11 chiều, thường rơi vào brute force.
- Có thể kiểm tra sau khi fit: `best.named_steps['knn']._fit_method` (`'brute' | 'kd_tree' | 'ball_tree'`).

### 8) Tóm tắt “script đọc khi thuyết trình”

- “KNN dùng nguyên tắc gần thì giống: dự đoán dựa vào k láng giềng gần nhất. Nếu dùng `weights=distance`, điểm gần hơn được trọng số lớn hơn.”
- “Mình chuẩn hoá dữ liệu trước (StandardScaler) để đảm bảo khoảng cách có ý nghĩa.”
- “Mình dùng GridSearchCV để chọn tham số tốt (k, weights, p) qua cross-validation, sau đó đánh giá trên tập test.”
- “Classification đọc Accuracy, Precision/Recall/F1, ma trận nhầm lẫn; Regression đọc RMSE/MAE/R².”
- “k nhỏ nhạy/overfit; k lớn mượt/underfit. p=2 Euclidean, p=1 Manhattan.”

## KNN Training Guide (Wine Quality Project)

This note explains how the KNN models are trained in this repo, with exact code references.

### 1) Data loading and labels

- The dataset is loaded from `data/winequality-red.csv` (auto-download from UCI if missing).
- Classification uses a 3-class mapping of the numeric `quality` target.

```55:73:src/data.py
def make_quality_labels(quality: pd.Series, scheme: str = "3class") -> pd.Series:
    """Map numeric quality scores to categorical labels.

    scheme="3class": poor (<=4), average (5-6), good (>=7)
    Returns a pandas Series of dtype category with ordered classes.
    """
    if scheme != "3class":
        raise ValueError("Only '3class' scheme is supported currently.")

    labels = (
        quality
        .apply(lambda q: "poor" if q <= 4 else ("average" if q <= 6 else "good"))
        .astype("category")
    )
    labels = labels.cat.set_categories(["poor", "average", "good"], ordered=True)
    return labels
```

### 2) Model pipelines

We always scale features before KNN (distance-based), then apply KNN.

```8:19:src/models.py
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
```

```22:33:src/models.py
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
```

Key hyperparameters:

- `n_neighbors` (k): number of neighbors.
- `weights`: `uniform` (equal vote) or `distance` (closer ⇒ higher weight).
- `p`: Minkowski metric; `p=2` Euclidean, `p=1` Manhattan.

### 3) Training flow — Classification

```59:90:src/train.py
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
```

Notes:

- Split keeps class balance via `stratify=y`.
- Hyperparameter search tries multiple `k`, `weights`, `p` using cross-validation.
- Final evaluation on hold-out test set prints accuracy, report, and confusion matrix.

### 4) Training flow — Regression

```117:151:src/train.py
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
```

Notes:

- Regression predicts the numeric `quality`. Metrics: RMSE, MAE (lower better), R² (closer to 1 better).

### 5) CLI arguments and reproducibility

The script exposes key knobs via CLI (mode, cv, k list, weights, p, test split, random_state, saving).

```22:49:src/train.py
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
```

Recommended usage:

- Classification: `python src/train.py --mode classify`
- Regression: `python src/train.py --mode regress`
- Try different k list: `--neighbors 3,5,7,9,11,13,15`
- Save model: `--save_model models/best_knn_cls.joblib`

### 6) Why scaling? Why k, weights, p?

- Scaling (StandardScaler) ensures features contribute fairly to distance.
- k small ⇒ sensitive/overfit; k large ⇒ smoother/underfit risk.
- weights=distance gives more influence to closer points.
- p=2 (Euclidean) is common default; p=1 (Manhattan) can be robust to axis-wise tails.

### 7) What algorithm (brute/kd-tree/ball-tree)?

- We rely on scikit-learn’s default `algorithm='auto'`. For small, ~11-dim data, it often uses brute force.
- You can inspect used method via `best.named_steps['knn']._fit_method` after training.
