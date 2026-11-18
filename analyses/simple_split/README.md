# simple_split — 5 baseline trên cùng một cách chia train / validation / test

Theo ý 1 của thầy: chạy baseline cơ bản trước, xem trực tiếp dự báo đúng/sai, bảo đảm kết quả có ý nghĩa rồi mới
mở rộng. Mọi mô hình chạy **cùng** dữ liệu (data freeze v1.0), **cùng** 10 đặc trưng OHLCV, **cùng** cách chia:

| Đoạn | Từ | Đến | Dùng để |
|---|---|---|---|
| train | 2018-01-01 | 2021-12-30 | học |
| validation | 2022-01-01 | 2023-12-30 | chọn tham số (log-loss) |
| test | 2024-01-01 | 2026-08-30 | chấm đúng 1 lần, sau khi học lại trên train + validation |

Thiết kế: `docs/superpowers/specs/2026-11-13-simple-split-design.md`.

| Mô hình | Trước tune | Grid sau tune | Seed |
|---|---|---|---|
| B0: Majority, Persistence, Buy-and-hold, Đảo chiều | – | – | – |
| LR (Vi) | C = 1, L2, 10 đặc trưng | 300 tổ hợp của `lr_tuning.py` | tất định |
| SVM RBF (Trâm) | C = 1, gamma = "scale" | C × gamma 5 × 6 | tất định |
| RF (Ngân) | sklearn mặc định, 500 cây | 24 tổ hợp `b2t.yaml` | trung bình 42, 43, 44 |
| LSTM (Ngọc) | lookback 30, hidden 32, dropout 0, lr 1e-3 | 16 cấu hình `lstm_tuning.py` | trung bình 42, 43, 44 |

## Chạy

Từ thư mục gốc repo:

```
python analyses/simple_split/src/simple_split.py --test     # test
python analyses/simple_split/src/simple_split.py --quick    # chạy thử nhanh, không ghi báo cáo
python analyses/simple_split/src/simple_split.py            # chạy đủ
```

## Đầu ra (`reports/`)

- `simple_split.md`: bảng so sánh, bảng tuning từng mô hình, phần xem dự báo từng mô hình.
- `test_metrics.csv`, `tuning_<mô hình>.csv`, `predictions_test.csv` (p̂ từng ngày của mọi mô hình).

Lưu ý: script nạp `torch` trước sklearn/scipy; đổi thứ tự này làm LSTM bị segfault trên macOS.
