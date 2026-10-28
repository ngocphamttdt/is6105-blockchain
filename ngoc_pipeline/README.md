# ngoc_pipeline: walk-forward, đặc trưng On-chain, mô hình, chạy thí nghiệm

Code và kết quả cho task **1.15, 1.16, 2.8, 2.11, 3.8** (Ngọc). Đây là pipeline đo:

```
Δ = M(OHLCV + On-chain) − M(OHLCV-only)
```

Hai vế dùng cùng mô hình, cùng 82 fold walk-forward và cùng ngân sách tuning; chỉ khác bộ đặc trưng.

## Trạng thái task

| Task | Nội dung | Trạng thái | File |
|---|---|---|---|
| 1.15 | Walk-forward chống rò rỉ: train 730 ngày, gap 1 ngày, test 30 ngày, 82 fold (01/2020 – 08/2026) | ✅ Xong | `src/walkforward.py` |
| 1.16 | Biến đổi On-chain: 3 đặc trưng mỗi chỉ số; phương án Flows A có 11 chỉ số, 33 đặc trưng | ✅ Xong | `src/features.py` |
| 2.8 | Interface mô hình và pipeline chạy bằng một lệnh | ✅ Xong | `src/models.py`, `src/run_all.py` |
| 2.11, 3.8 | B0 (`b0pers`, `b0maj`) và B3 (`b3`, SVM RBF), đủ 82 fold × 2 cấu hình | ✅ Xong | `src/models.py`, `results/` |
| 2.9, 2.10 | E0 sanity check: xáo nhãn, chia 80/20 | ⏳ Chưa có | (`--stage e0`) |

## Cấu trúc

```
ngoc_pipeline/
├── README.md
├── src/                      mỗi file ứng với một bước; chạy file nào cũng test trước
│   ├── data.py               (0) đọc data freeze, kiểm sha256, tên cột, chọn cột theo dictionary A/B
│   ├── features.py           (1) nhãn y, đặc trưng OHLCV (tạm) và On-chain
│   ├── walkforward.py        (2) chia fold, scaler + tune trong train, chạy mô hình qua mọi fold
│   ├── models.py             (3) interface mô hình; B0, B3
│   └── run_all.py            (4) chạy thí nghiệm: --stage data | e1
├── configs/
│   ├── walkforward.yaml      tham số protocol
│   ├── features_onchain.yaml flows_set: A, shift_days: 1, ffill_limit: 3, signed_cols
│   └── models/{lr,b0maj,b3}.yaml   tham số cố định và grid để tune
└── results/
    ├── predictions/{model}_{config}.csv
    ├── tables/e1_delta_draft.csv
    └── figures/walkforward.png
```

## Dữ liệu đầu vào

Chỉ đọc `data/processed/` (data freeze v1.0 của Khoa và Vi), không đọc `crawl_data/`. Không có code nào ghi vào thư mục này.

| File | Dùng cho |
|---|---|
| `master_daily.csv` | Giá ngày (`ohlcv_*`) cho nhãn và đặc trưng OHLCV; chỉ số On-chain (`cm_*`, `bc_*`, `puell_multiple`, `nvt_ratio_bc`) |
| `data_dictionary_A.csv` | Danh sách chỉ số On-chain đưa vào mô hình: `status = retained`, `role` là `candidate` hoặc `backup` |
| `MANIFEST.json` | sha256 của mọi file. Mỗi lần đọc đều so; lệch thì dừng |

## Chạy

Cần Python 3.10+ với `pandas numpy scikit-learn pyyaml matplotlib`. Chạy từ thư mục gốc repo:

```bash
python ngoc_pipeline/src/run_all.py --test                                   # chỉ chạy test (~3 giây)
python ngoc_pipeline/src/run_all.py --stage data                             # kiểm freeze, đặc trưng, fold
python ngoc_pipeline/src/run_all.py --stage e1 --models lr b0pers b0maj b3   # E1 đủ 82 fold (~20 giây)
python ngoc_pipeline/src/run_all.py --stage e1 --models b3 --every 10        # chạy thử 1/10 số fold -> every10/
python ngoc_pipeline/src/walkforward.py                                      # vẽ lại results/figures/walkforward.png
```

- Mỗi lần chạy stage đều chạy test của mọi file trước.
- Có thể chạy riêng một file, ví dụ `python ngoc_pipeline/src/features.py --test`.

`--stage data` phải in:

```
Data freeze v1.0: sha256 khớp 4 file; flows set A; 7 OHLCV + 33 On-chain; 82 fold, train fold 0 từ 2017-12-31; NaN từ ngày đó: 0
```

Mỗi người có thể chạy mô hình của mình riêng. Bảng Δ luôn tính lại từ mọi file trong `results/predictions/`, nên không mất dòng của mô hình chạy trước.

## Quy tắc chống rò rỉ, và test nào kiểm

| Rủi ro | Cách chặn | Test |
|---|---|---|
| Đặc trưng dùng dữ liệu tương lai | Chỉ dùng `diff` và `rolling` lùi. `ffill` chỉ lấy quá khứ, tối đa 3 ngày | `features.py` (2), (6): sửa dữ liệu sau mốc cut thì đặc trưng đến cut không đổi |
| On-chain công bố trễ | `shift_days: 1`: dòng t mang On-chain của ngày t−1. OHLCV vẫn dùng đến hết ngày t | `features.py` (3) |
| Nhãn ngày train cuối chứa giá ngày test đầu | Gap 1 ngày giữa train và test, cả trong inner CV | `walkforward.py` (1), (4) |
| Scaler hoặc tuning thấy test | StandardScaler nằm trong Pipeline; GridSearchCV nhận Pipeline | `walkforward.py` (3): nhân dữ liệu test lên 1.000 lần, dự báo vẫn khớp pipeline chỉ fit trên train |
| Persistence nhìn ngày mai | p̂_t = 1[close_t > close_{t−1}] | `models.py` (3) |
| Dùng nhầm phiên bản dữ liệu | So sha256 với `MANIFEST.json` | `data.py` (1), (3) |

Đã thử cố tình cài 4 lỗi vào một bản sao, và `--test` bắt được cả 4:
- z-score dùng `rolling(center=True)`, tức nhìn tương lai;
- bỏ gap giữa train và test;
- persistence dùng nhãn của chính ngày t;
- scaler fit trên cả tập test.

## Đặc trưng

| Nhóm | Đặc trưng |
|---|---|
| OHLCV (**tạm**, 7 cột) | `ret_1` … `ret_5` (`ret_1` = log return của chính ngày t), `hl_range` = (H−L)/C, `dlog_volume`. Sẽ thay bằng `features_ohlcv.py` của Trâm (thêm RSI-14, RV 7/30 ngày) |
| On-chain, cột dương | `{col}_logdiff` = log x_t − log x_{t−1}, `{col}_ma7_30` = MA7 / MA30, `{col}_z90` = z-score 90 ngày của log x |
| On-chain, cột có dấu (`cm_NetflowExNtv`) | `{col}_diff` = x_t − x_{t−1}, `{col}_ma7m30` = MA7 − MA30, `{col}_z90` = z-score 90 ngày của x. Netflow âm ở khoảng một nửa số ngày nên không lấy log được |

Số đặc trưng On-chain theo nhóm (phương án A, dùng cho ablation E2 qua `feature_groups`): Activity 9, Flows 6, Mining 9, Valuation 9.

## Mô hình

| Tên | Mô hình | Ghi chú |
|---|---|---|
| `lr` | B1 Logistic Regression, grid C | Bản cho pipeline mỏng; Trâm sở hữu B1 |
| `b0maj` | B0 majority class: p̂ = tỷ lệ ngày tăng trong train | Mức sàn cho log-loss và Brier |
| `b0pers` | B0 persistence: ngày mai cùng hướng hôm nay | Mức sàn cho DA. p̂ chỉ là 0/1, nên log-loss không dùng được |
| `b3` | B3 SVM RBF, grid C × gamma | Xác suất hiệu chỉnh sigmoid trên điểm ngoài mẫu theo thời gian (TimeSeriesSplit, gap 1); SVM cuối fit trên toàn cửa sổ train |

B0 không dùng đặc trưng, nên hai cấu hình cho cùng dự báo (Δ = 0).

Thêm mô hình mới:
1. Gọi `register("rf", RandomForestClassifier)` trong `src/models.py`.
2. Tạo `configs/models/rf.yaml` gồm `class`, `params`, `param_grid`.
3. Chạy `--stage e1 --models rf`.

## Kết quả E1 nháp (28/10/2026)

Phương án A, 2.434 ngày test. DA và AUC: cao hơn là tốt hơn. Log-loss và Brier: thấp hơn là tốt hơn.

| Mô hình | DA: OHLCV → +On-chain | ΔDA | Log-loss: OHLCV → +On-chain | Δlog-loss |
|---|---|---|---|---|
| `b0maj` | 0,5008 | 0 | 0,6938 | 0 |
| `b0pers` | 0,4651 | 0 | không dùng | 0 |
| `lr` | 0,5164 → 0,5119 | −0,0045 | 0,6941 → 0,6997 | +0,0056 |
| `b3` | 0,5094 → 0,5037 | −0,0058 | 0,7023 → 0,6961 | −0,0062 |

- Mọi mô hình chỉ đoán đúng 50–52% số ngày.
- Với 2.434 ngày, sai số chuẩn của DA khoảng 1 điểm % (ước lượng thô), nên mọi Δ ở trên đều nằm trong mức nhiễu.
- `b0pers` dưới 50%: trong giai đoạn 2020–2026, hướng giá ngày mai hay ngược với hôm nay.
- `b3` cấu hình OHLCV-only có log-loss tệ hơn mức sàn `b0maj`. Có những ngày mô hình quá tự tin (p̂ tới 0,998); cần xem lại theo từng fold.

**Chưa dùng các con số này để kết luận RQ1:** chưa có kiểm định (Vi, 2.3), OHLCV vẫn là bản tạm, và mới có 2 mô hình có học.

## Điểm cần nhóm chốt

1. **Phương án Flows (họp 1.19):** đang dùng A (`cm_NetflowExNtv` + `cm_SplyExNtv`). Nếu chốt B, đổi `flows_set: B` trong `configs/features_onchain.yaml` rồi chạy lại các cấu hình OHLCV + On-chain.
2. **Đặc trưng OHLCV:** khi chuyển sang code của Trâm, `features_ohlcv.py` cần đọc `data/processed/` (hiện đang đọc `crawl_data/`). Tên `logret_lag1` của Trâm cùng nghĩa với `ret_1` ở đây.
3. **Cửa sổ train 1.095 ngày (task 4.5):** fold đầu cần train từ 31/12/2016, nhưng OHLCV chỉ có từ 17/08/2017. Đề xuất: với cửa sổ này, bắt đầu test từ khoảng 09/2020, rồi so độ nhạy trên phần ngày test chung của cả ba cửa sổ.
4. **Fold cuối chỉ có 4 ngày test** (27–30/08/2026). Ghi chú trong mục Method.
