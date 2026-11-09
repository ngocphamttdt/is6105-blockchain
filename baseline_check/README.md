# baseline_check — Kiểm tra data, EDA nhãn và baseline cơ bản (góp ý 1 của thầy)

Phụ trách: Vi, Trâm. Góp ý của thầy:

> Kết quả baseline hiện chưa đủ chắc chắn để rút ra kết luận. Kiểm tra data, làm EDA và chạy một baseline
> cơ bản trước; xem trực tiếp một số dự báo đúng/sai và bảo đảm kết quả có ý nghĩa rồi mới mở rộng.

**Kết quả: `reports/baseline_check.md`** (đọc phần Tóm tắt đầu file trước).

## Làm gì

| Phần | Nội dung |
|---|---|
| 1. Dữ liệu và nhãn | Gộp lại nến 1h và so với `master_daily.csv`; kiểm nhãn bằng ví dụ ngày 10–14/03/2020; ngày thiếu nến; ngày giá bằng nhau |
| 2. EDA nhãn | Tỷ lệ tăng, tự tương quan return và dấu return, persistence theo giai đoạn và theo năm |
| 3. Baseline cơ bản | Chia theo thời gian train 2018–2021 / validation 2022–2023 / test 2024–08/2026, gap 1 ngày. Majority, luôn tăng, persistence, đảo chiều, LR 1 đặc trưng, LR 10 đặc trưng. C của LR tune trên validation theo log-loss (ghi đủ bảng trước/sau tuning), test chạy 1 lần. Kiểm metric bằng tay; phép thử xáo nhãn; phép thử rò rỉ cố ý |
| 4. Xem dự báo | Ma trận nhầm lẫn, 10 ngày ngẫu nhiên, 8 ngày sai tự tin nhất, 10 ngày biến động mạnh nhất, độ chính xác theo mức biến động, calibration. Dự báo từng ngày: `reports/predictions_test.csv` |
| 5. Walk-forward | Cùng lịch `ngoc_pipeline` (82 fold) cho các luật B0 và LR, để đối chiếu với báo cáo giữa kỳ |

Chỉ dùng OHLCV và data freeze v1.0. Không dùng On-chain, không dùng `ngoc_pipeline` (hai code độc lập; Majority và Persistence khớp đúng Bảng 8.1).

## Chạy lại

Cần `tram_features_backtest/src/features_ohlcv.py` bản đọc data freeze (PR #4). Từ thư mục gốc repo:

```
pip install pandas numpy scipy scikit-learn matplotlib tabulate
python baseline_check/src/baseline_check.py
```

Chạy khoảng 10 giây, ghi lại toàn bộ `reports/`.

## Đầu ra

```
baseline_check/
├── README.md
├── src/baseline_check.py
└── reports/
    ├── baseline_check.md          báo cáo đầy đủ
    ├── test_metrics.csv           bảng metric trên test 2024–08/2026
    ├── walkforward_metrics.csv    bảng metric walk-forward 2020–08/2026
    ├── predictions_test.csv       dự báo từng ngày của LR 10 đặc trưng trên test
    └── figures/bc_0*.png
```
