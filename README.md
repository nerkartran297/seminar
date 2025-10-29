# Wine Quality Prediction (Red Wine)

Dự án nhỏ dùng Python để dự đoán/chấm điểm chất lượng rượu vang đỏ dựa trên dữ liệu UCI. Hỗ trợ cả hai bài toán:

- Phân loại (classification): phân nhóm chất lượng kém / trung bình / tốt
- Hồi quy (regression): dự đoán điểm chất lượng (số nguyên)

## Dữ liệu

- Nguồn: UCI Machine Learning Repository (Wine Quality - Red)
- File: `winequality-red.csv` (phân cách bằng dấu chấm phẩy `;`)
- Script sẽ tự tải file nếu chưa có, hoặc bạn có thể đặt file vào thư mục `data/` trước.

## Cài đặt môi trường

```powershell
# Tạo môi trường ảo (PowerShell - Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
```

## Chạy mô hình

Script dòng lệnh nằm ở `src/train.py`.

### 1) Phân loại (3 lớp: poor/average/good)

```powershell
python src/train.py --mode classify
```

Tùy chọn hữu ích:

- `--data_dir data` (mặc định `data`)
- `--test_size 0.2` (mặc định 0.2)
- `--cv 5` (mặc định 5)
- `--neighbors 3,5,7,9,11,13,15,17,19,21` (grid tìm kiếm k)
- `--save_model models/best_knn_cls.joblib` (nếu muốn lưu mô hình)

### 2) Hồi quy (dự đoán điểm chất lượng)

```powershell
python src/train.py --mode regress
```

Tùy chọn tương tự phần phân loại; đánh giá gồm RMSE, MAE, R2.

## Ghi chú tiền xử lý

- Chuẩn hoá đặc trưng bằng StandardScaler trước khi đưa vào KNN
- Với phân loại: ánh xạ nhãn thành 3 lớp theo quy tắc mặc định:
  - `quality <= 4` → `poor`
  - `quality in {5, 6}` → `average`
  - `quality >= 7` → `good`

## Cấu trúc dự án

```
./
├─ data/                       # Nơi chứa csv (tự tạo khi chạy)
├─ models/                     # Nơi lưu mô hình (nếu --save_model)
├─ reports/figures/            # Nơi lưu hình minh hoạ KNN
├─ app/
│  └─ knn_explorer.py          # Ứng dụng Streamlit tương tác
├─ src/
│  ├─ __init__.py
│  ├─ data.py                  # Tải/đọc dữ liệu
│  ├─ models.py                # Pipelines KNN
│  ├─ plotting.py              # Hàm vẽ ranh giới/neighbor
│  ├─ visualize_knn.py         # CLI sinh hình giải thích KNN
│  └─ train.py                 # CLI train/evaluate
├─ requirements.txt
└─ README.md
```

## Bản quyền dữ liệu

- Dữ liệu thuộc UCI Machine Learning Repository. Vui lòng tham khảo giấy phép/điều khoản sử dụng từ UCI.

## Minh hoạ KNN (hình ảnh)

Sinh hình tự động (2 đặc trưng bất kỳ) để giải thích KNN:

```powershell
# Classification
python src/visualize_knn.py --mode classify --feat_x alcohol --feat_y "volatile acidity" --neighbors 15 --weights distance

# Regression
python src/visualize_knn.py --mode regress --feat_x alcohol --feat_y "volatile acidity" --neighbors 15 --weights distance
```

Ảnh và file giải thích sẽ lưu tại `reports/figures/`.

## Ứng dụng web (Streamlit)

```powershell
streamlit run app/knn_explorer.py
```

Chọn hai đặc trưng, điều chỉnh `k`, `weights`, chọn 1 mẫu test để xem lân cận và ranh giới quyết định (classification).
