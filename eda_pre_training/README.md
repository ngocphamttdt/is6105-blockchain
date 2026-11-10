# eda_pre_training — EDA dữ liệu trước khi training (theo `nhanxet.md`)

**Kết quả: [`reports/eda_pre_training.md`](reports/eda_pre_training.md)** (đọc bảng Tóm tắt đầu file trước).

Bổ sung cho hai phần đã có, không làm lại:

- `baseline_check/`: EDA nhãn, tự tương quan return, baseline OHLCV-only, xem dự báo đúng/sai (góp ý 1–3).
- `tram_features_backtest/notebooks/eda.ipynb`: chuỗi giá, bản đồ ô trống, tương quan sơ bộ.

| Phần | Nội dung                                                                                                                                | Góp ý |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 0     | sha256, ngày liên tục, logic OHLC, giá trị "đứng", quy ước ngày CoinMetrics và Binance                                        | 7       |
| 1     | Tỷ lệ tăng train so với test theo fold, giá trị tham chiếu của metric, ví dụ tính tay DA, log-loss, Brier, AUC                | 1, 2    |
| 2     | On-chain thô: xu hướng, mùa vụ theo thứ trong tuần, tính dừng (ADF, KPSS) của từng biến đổi                                | 4       |
| 3     | Ma trận 43 đặc trưng đúng như pipeline: đuôi dày, trùng thông tin, R² của On-chain theo OHLCV, trôi phân phối theo fold | 4       |
| 4     | Flows: kiểm ΔSply = Netflow, bằng chứng sửa lịch sử, bước nhảy                                                                 | 5       |
| 5     | Regime: độ dài đợt, ý nghĩa Bull/Bear/Sideways theo giá, so với một luật đơn giản                                          | 6       |

**Không có thống kê nào giữa đặc trưng On-chain và nhãn** (data/processed/README.md, lưu ý 6).

## Chạy lại

Từ thư mục gốc repo (khoảng 20 giây):

```
pip install pandas numpy scipy scikit-learn statsmodels matplotlib tabulate pyyaml
python eda_pre_training/src/eda_pre_training.py
```

Đặc trưng được dựng bằng `ngoc_pipeline/src/features.py::build_dataset` và fold bằng `ngoc_pipeline/src/walkforward.py::make_folds`,
nên khớp đúng dữ liệu mô hình sẽ thấy. Script đọc `configs/features_onchain.yaml` (phương án Flows hiện tại).

## Đầu ra

```
reports/
├── eda_pre_training.md        báo cáo đầy đủ
├── uprate_by_fold.csv         tỷ lệ tăng train/test của 82 fold
├── feature_distribution.csv   skew, kurtosis, cực trị của 43 đặc trưng
├── onchain_r2_on_ohlcv.csv    R² của từng đặc trưng On-chain theo 10 đặc trưng OHLCV
├── feature_drift.csv          % ngày test ngoài [P1, P99] của train
└── figures/eda_*.png
```
