# tram_naive_folds — Baseline ngây thơ luôn tăng / luôn giảm trên từng fold

Phụ trách: Trâm. **Kết quả: `reports/naive_folds.md`** (đọc phần Tóm tắt trước).

Câu hỏi: nếu lúc nào cũng đoán tăng thì đúng bao nhiêu? Một đoạn có 60% ngày tăng thì "luôn tăng" đúng 60%.
Script đặt **luôn tăng** và **luôn giảm** cạnh các baseline đã tuning, trên dữ liệu chia nhỏ:

| Dữ liệu | Chia nhỏ | Mô hình so sánh |
|---|---|---|
| Walk-forward 2020 – 08/2026 (`ngoc_pipeline`) | 82 fold × 30 ngày | B0 Majority, Persistence, Đảo chiều; B1 LR (E1), B1 tune đầy đủ (Vi), B2 `b2t`, B3 SVM, B4 XGBoost, B5 LSTM; cả OHLCV-only và + On-chain |
| `simple_split` test 2024 – 08/2026 | 33 khối 30 ngày | Majority, Persistence, Đảo chiều, LR / SVM / RF / LSTM sau tune |

Tính: DA toàn kỳ và McNemar so với luôn tăng; số fold mô hình thắng luôn tăng, luôn giảm, cả hai; phân phối tỷ lệ tăng giữa
các fold so với nhị thức; theo năm; mô hình có đi theo xu hướng của fold không.

Chỉ đọc file dự báo đã có, không chạy lại mô hình, chạy vài giây:

```
python tram_naive_folds/src/naive_folds.py
```

Đầu ra: `results/` (bảng theo fold, tóm tắt, theo năm), `reports/naive_folds.md`, `reports/figures/`.
