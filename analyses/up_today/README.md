# analyses/up_today — Thêm đặc trưng up_today, chạy lại B1 và B3

Phụ trách: Trâm. **Kết quả: `reports/up_today_report.md`** (đọc phần Tóm tắt trước).

## Thay đổi

`features/src/features_ohlcv.py`:

- Thêm cột **`up_today`** = 1 nếu giá đóng cửa ngày t cao hơn ngày t−1, ngược lại 0 (ngày đầu chuỗi là NaN).
- `FEATURES` = 10 đặc trưng ban đầu + `up_today` (đặt cuối để `lrfs` vẫn bắt đầu từ `logret_lag1`).
- `FEATURES_10` giữ bộ 10 đặc trưng cũ. `baseline_check` chuyển sang dùng `FEATURES_10` để báo cáo cũ tái tạo được (đã chạy lại, kết quả không đổi).
- `data/features_ohlcv.csv` tạo lại với 11 cột.

**Vì sao:** quy luật duy nhất có ý nghĩa mà `baseline_check` tìm được là hướng giá đảo chiều sau 1 ngày, nằm ở *dấu* return. 10 đặc trưng cũ chỉ đo *độ lớn*.

**Ảnh hưởng:** `pipeline/src/features.py` lấy mọi cột của `build_ohlcv_features`, nên lần chạy E1 tiếp theo sẽ dùng 11 đặc trưng OHLCV cho mọi mô hình. Dự báo đang có trong `pipeline/results/` không tự đổi.

## So sánh đã chạy

`src/compare_up_today.py`: `lr`, `lrfs`, `b3` theo đúng config trong `pipeline/configs/models/`, 82 fold, tune bằng inner CV trong cửa sổ train. Mỗi mô hình chạy 4 lần: {10, 11 đặc trưng} × {OHLCV-only, OHLCV + On-chain}. So với luật đảo chiều `b0rev`.

```
python analyses/up_today/src/compare_up_today.py            # chạy mô hình rồi viết báo cáo (lrfs chậm, ~15–20 phút)
python analyses/up_today/src/compare_up_today.py --report   # chỉ dựng lại báo cáo
```

## Đầu ra

```
analyses/up_today/
├── src/compare_up_today.py
├── results/
│   ├── predictions/{model}__{f10|f11}_{config}.csv, b0rev.csv
│   └── tables/ metrics.csv, delta_up_today.csv, delta_onchain.csv, by_year_ohlcv.csv,
│               fit_log.csv (tham số chọn mỗi fold, cột lrfs giữ lại)
└── reports/up_today_report.md, figures/da_by_year.png
```
