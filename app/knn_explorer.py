from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Ensure project root is on sys.path when running from app/ directory
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


st.set_page_config(page_title="Wine KNN Explorer", layout="wide")
st.title("Wine Quality — KNN Explorer")
st.write("Tương tác để hiểu KNN: chọn 2 đặc trưng, cấu hình k/weights và xem lân cận cùng ranh giới quyết định.")

mode = st.sidebar.selectbox("Chế độ", ["classify", "regress"], index=0)
feat_x = st.sidebar.selectbox("Feature X", FEATURES, index=FEATURES.index("alcohol"))
feat_y = st.sidebar.selectbox("Feature Y", FEATURES, index=FEATURES.index("volatile acidity"))
k = st.sidebar.slider("k (neighbors)", min_value=1, max_value=51, value=15, step=2)
weights = st.sidebar.selectbox("weights", ["uniform", "distance"], index=1)
p = st.sidebar.selectbox("metric p", [1, 2], index=1)
seed = st.sidebar.number_input("Seed", min_value=0, value=42)

df = load_winequality_red("data")
X2 = df[[feat_x, feat_y]].copy()
if mode == "classify":
    y = make_quality_labels(df["quality"]).astype("category")
else:
    y = df["quality"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X2.values,
    y.values if isinstance(y, pd.Series) else y,
    test_size=0.2,
    random_state=seed,
    stratify=y if isinstance(y, pd.Series) and str(y.dtype) == "category" else None,
)

if mode == "classify":
    model = build_knn_pipeline_classifier(k, weights, p)
else:
    model = build_knn_pipeline_regressor(k, weights, p)

model.fit(X_train, y_train)

# Allow custom query point
st.sidebar.subheader("Chọn mẫu để dự đoán")
use_custom = st.sidebar.checkbox("Dùng điểm tự nhập", value=False)
if use_custom:
    x_min, x_max = float(df[feat_x].min()), float(df[feat_x].max())
    y_min, y_max = float(df[feat_y].min()), float(df[feat_y].max())
    default_x = float(df[feat_x].median())
    default_y = float(df[feat_y].median())
    val_x = st.sidebar.slider(feat_x, x_min, x_max, default_x)
    val_y = st.sidebar.slider(feat_y, y_min, y_max, default_y)
    qx = np.array([val_x, val_y], dtype=float)
else:
    if len(X_test) > 0:
        idx_in_test = st.sidebar.slider("Index trong test set", 0, len(X_test) - 1, 0)
    else:
        idx_in_test = 0
    qx = X_test[idx_in_test]

neighbors, manual_pred, proba = compute_neighbors(
    model=model, X_train=X_train, y_train=y_train, query=qx, mode=mode
)
model_pred = model.predict(qx.reshape(1, -1))[0]
model_proba = None
if mode == "classify" and hasattr(model.named_steps["knn"], "predict_proba"):
    try:
        model_proba = float(np.max(model.named_steps["knn"].predict_proba(model.named_steps["scaler"].transform(qx.reshape(1, -1)))))
    except Exception:
        model_proba = None

col1, col2 = st.columns(2)

with col1:
    if mode == "classify":
        fig, _ = plot_decision_boundary_classification(
            X=X_train,
            y=pd.Series(y_train).astype("category"),
            feature_names=(feat_x, feat_y),
            model=model,
            highlight_point=(qx[0], qx[1]),
            neighbors=neighbors,
            title=f"Decision boundary — {feat_x} vs {feat_y}",
        )
        st.pyplot(fig)
    else:
        fig, _ = plot_neighbors_overlay(
            X=X_train,
            y=np.array([0] * len(X_train)),
            feature_names=(feat_x, feat_y),
            query_point=(qx[0], qx[1]),
            neighbors=neighbors,
            title="Neighbors (regression)",
        )
        st.pyplot(fig)

with col2:
    st.subheader("Neighbor details")
    df_nb = pd.DataFrame(
        {
            "index": [n.index for n in neighbors],
            "distance": [n.distance for n in neighbors],
            "label": [n.label for n in neighbors],
            "weight": [n.weight for n in neighbors],
        }
    )
    st.dataframe(df_nb, use_container_width=True)

    if mode == "classify":
        st.write(f"Manual vote (giải tay): {manual_pred}")
        st.write(f"Model predict: {model_pred}")
        st.write(f"Model prob (max): {model_proba if model_proba is not None else 'n/a'}")
    else:
        st.write(f"Manual weighted average (giải tay): {manual_pred:.3f}")
        st.write(f"Model predict: {float(model_pred):.3f}")

st.markdown("---")
with st.expander("Giải thích / Script đọc khi trình bày", expanded=True):
    if mode == "classify":
        st.markdown(
            f"""
            - Bài toán phân loại 3 lớp từ cột quality (poor/average/good).
            - KNN dựa trên nguyên tắc 'gần thì giống': mẫu mới lấy phiếu bầu từ {k} láng giềng.
            - weights = `{weights}`: nếu là distance, láng giềng gần hơn được trọng số lớn hơn (1/(d+ε)).
            - Hình trái: ranh giới quyết định trên 2 đặc trưng `{feat_x}` và `{feat_y}`.
            - Dấu sao là mẫu test đang chọn; các đường là nối đến k láng giềng.
            - Bảng phải: khoảng cách và trọng số từng láng giềng; "manual vote" là cách giải tay.
            - Ý nghĩa tham số:
              - k nhỏ ⇒ nhạy, dễ overfit; k lớn ⇒ mượt nhưng có thể mất chi tiết.
              - p=1 (Manhattan), p=2 (Euclidean).
            """
        )
    else:
        st.markdown(
            f"""
            - Bài toán hồi quy: dự đoán điểm quality (3–8).
            - KNN dự đoán bằng trung bình (hoặc trung bình có trọng số theo khoảng cách) của {k} láng giềng.
            - Hình hiển thị láng giềng xung quanh mẫu test đang chọn trên 2 đặc trưng `{feat_x}` và `{feat_y}`.
            - Bảng phải: khoảng cách và trọng số; "manual weighted average" là dự đoán giải tay.
            - p=1 (Manhattan), p=2 (Euclidean).
            """
        )


