# tram_baselines — Chạy và tuning B0 (Majority, Persistence, Buy-and-hold) và B3 (SVM RBF)

Phụ trách: Trâm. Theo phân công của Ngọc sau góp ý của thầy:

> Ghi rõ phần tuning nào đã thực sự chạy, chọn tham số thế nào và kết quả trước/sau tuning. Tuning bằng
> train/validation, không chọn model hoặc tham số theo test; hoàn thiện từng baseline trước khi so sánh.
> Sau khi baseline OHLCV-only đã ổn định, mới bổ sung On-chain từng bước, giữ cùng model và protocol.

**Kết quả: `reports/baselines_report.md`** (đọc phần Tóm tắt đầu file trước).

## Làm gì

| Mô hình | Tên file | Tuning |
|---|---|---|
| B0 Majority (730 ngày) | `b0maj` | không có tham số |
| B0 Majority, tune cửa sổ | `b0majlb` | L ∈ {30, 90, 180, 365, 730} ngày |
| B0 Persistence | `b0pers` | không có tham số |
| B0 Đảo chiều | `b0rev` | chiều của luật, chọn trên train |
| B0 Buy-and-hold (luôn đoán tăng) | `b0bh` | không có tham số |
| B3 SVM mặc định | `b3def` | không tune (C = 1, gamma = "scale") — **trước tuning** |
| B3 SVM grid 3×3 | `b3g9` | grid của `ngoc_pipeline/configs/models/b3.yaml` — tái tạo đúng `b3` của Ngọc |
| B3 SVM grid 5×6 | `b3g30` | C ∈ {0.01, 0.1, 1, 10, 100} × gamma ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3} — **sau tuning** |

- Dữ liệu, đặc trưng, 82 fold walk-forward, chuẩn hoá và `CalibratedSVC` lấy nguyên từ `ngoc_pipeline`, nên kết quả so trực tiếp được với bảng E1.
- Tune bằng inner `TimeSeriesSplit(5, gap 1)` **trong cửa sổ train 730 ngày**, tiêu chí log-loss, tune lại mỗi 6 fold (14 lần). Ngày test không bao giờ được dùng để chọn.
- Mỗi lần tune ghi lại điểm validation của **mọi** tổ hợp và của tham số mặc định: `results/tables/tuning_log.csv`.
- Chạy OHLCV-only trước, rồi OHLCV + On-chain với cùng mô hình, cùng grid.
- Thử thêm: SVM chỉ với 1 đặc trưng `logret_lag1`.

## Chạy lại

Cần `pandas numpy scipy scikit-learn pyyaml matplotlib`. Từ thư mục gốc repo:

```
python tram_baselines/src/run_baselines.py            # chạy mô hình (~3 phút) rồi viết báo cáo
python tram_baselines/src/run_baselines.py --report   # chỉ dựng lại bảng và báo cáo từ dự báo đã lưu
```

## Đầu ra

```
tram_baselines/
├── README.md
├── src/run_baselines.py
├── results/
│   ├── predictions/{model}_{config}.csv   dự báo từng ngày, cùng định dạng ngoc_pipeline
│   └── tables/
│       ├── tuning_log.csv                 mọi tổ hợp × mọi lần tune: val log-loss, val DA, chọn hay không
│       ├── test_metrics.csv               DA, CI, p, log-loss, Brier, AUC, % đoán tăng
│       ├── vs_reversal.csv                SVM so với luật đảo chiều (McNemar)
│       ├── delta_onchain.csv              Δ khi thêm On-chain
│       ├── by_year.csv                    DA theo năm
│       ├── svm_diagnostics.csv            phân bố p̂, fold đoán một hướng, tương quan với tỷ lệ tăng train
│       ├── backtest.csv                   long/flat, phí 0,1%
│       └── f1_*.csv                       dự báo SVM 1 đặc trưng
└── reports/
    ├── baselines_report.md
    └── figures/ (heatmap grid SVM, DA theo năm)
```

Muốn đưa dự báo vào bảng chung của nhóm thì chép `results/predictions/*.csv` sang `ngoc_pipeline/results/predictions/` rồi chạy `python ngoc_pipeline/src/run_all.py --stage tables`.
