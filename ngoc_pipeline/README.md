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
| (thử) | B2 (`b2`, Random Forest) và B4 (`b4`, XGBoost), đủ 82 fold × 2 cấu hình. Bản chạy thử để có bảng Δ sớm; bản chính thức vẫn do Trâm (B2) và Khoa (B4) làm | ✅ Chạy thử | `src/models.py`, `configs/models/b2.yaml`, `b4.yaml` |
| 2.9, 2.10 | E0 sanity check: xáo nhãn, chia 80/20 | ⏳ Chưa có | (`--stage e0`) |

## Cấu trúc

```
ngoc_pipeline/
├── README.md
├── src/                      mỗi file ứng với một bước; chạy file nào cũng test trước
│   ├── data.py               (0) đọc data freeze, kiểm sha256, tên cột, chọn cột theo dictionary A/B
│   ├── features.py           (1) nhãn y, đặc trưng OHLCV (lấy từ tram_features_backtest/) và On-chain
│   ├── walkforward.py        (2) chia fold, scaler + tune trong train, chạy mô hình qua mọi fold
│   ├── models.py             (3) interface mô hình; B0, B3; bản thử B2, B4
│   └── run_all.py            (4) chạy thí nghiệm: --stage data | e1
├── configs/
│   ├── walkforward.yaml      tham số protocol
│   ├── features_onchain.yaml flows_set: A, shift_days: 1, ffill_limit: 3, signed_cols
│   └── models/{lr,b0maj,b2,b3,b4}.yaml   tham số cố định và grid để tune
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
| `ohlcv_1h_clean.csv` | Giá 1h, chỉ để tính `rv_7d`, `rv_30d` |
| `data_dictionary_A.csv` | Danh sách chỉ số On-chain đưa vào mô hình: `status = retained`, `role` là `candidate` hoặc `backup` |
| `MANIFEST.json` | sha256 của mọi file. Mỗi lần đọc đều so; lệch thì dừng |

## Chạy

Cần Python 3.10+ với `pandas numpy scikit-learn pyyaml matplotlib`, và thư mục `tram_features_backtest/src/` (đặc trưng OHLCV). `b4` cần thêm `xgboost`; trên macOS phải `brew install libomp` trước. Chạy từ thư mục gốc repo:

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
Data freeze v1.0: sha256 khớp 4 file; flows set A; 10 OHLCV + 33 On-chain; 82 fold, train fold 0 từ 2017-12-31; NaN từ ngày đó: 0
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
| OHLCV (10 cột, `build_ohlcv_features` trong `tram_features_backtest/src/features_ohlcv.py`, task 1.13) | `logret_lag1` … `logret_lag5` (`logret_lag1` = log return của chính ngày t), `hl_range` = (H−L)/C, `dlog_volume`, `rsi_14`, `rv_7d`, `rv_30d` (từ return 1h). Công thức chi tiết: README của `tram_features_backtest/` |
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
| `b2` | B2 Random Forest, grid số cây × độ sâu × `min_samples_leaf` (12 tổ hợp) | Bản chạy thử; Trâm sở hữu B2. Chạy đủ 82 fold mất khoảng 7 phút |
| `b4` | B4 XGBoost, grid learning rate × độ sâu | Bản chạy thử; Khoa sở hữu B4. Số cây chọn bằng early stopping trên 20% cuối cửa sổ train (bỏ 1 ngày gap), rồi fit lại trên toàn cửa sổ |

B0 không dùng đặc trưng, nên hai cấu hình cho cùng dự báo (Δ = 0).

Thêm mô hình mới:
1. Gọi `register("rf", RandomForestClassifier)` trong `src/models.py`.
2. Tạo `configs/models/rf.yaml` gồm `class`, `params`, `param_grid`.
3. Chạy `--stage e1 --models rf`.

## Kết quả E1 nháp (29/10/2026)

Phương án A, 10 đặc trưng OHLCV của Trâm, 2.434 ngày test. DA và AUC: cao hơn là tốt hơn. Log-loss và Brier: thấp hơn là tốt hơn.

| Mô hình | DA: OHLCV → +On-chain | ΔDA | Log-loss: OHLCV → +On-chain | Δlog-loss |
|---|---|---|---|---|
| `b0maj` | 0,5008 | 0 | 0,6938 | 0 |
| `b0pers` | 0,4651 | 0 | không dùng | 0 |
| `lr` | 0,5111 → 0,5189 | +0,0078 | 0,6953 → 0,7006 | +0,0054 |
| `b2` (thử) | 0,5123 → 0,5140 | +0,0016 | 0,6961 → 0,6957 | −0,0005 |
| `b3` | 0,5062 → 0,5029 | −0,0033 | 0,7009 → 0,6959 | −0,0050 |
| `b4` (thử) | 0,5045 → 0,4897 | −0,0148 | 0,6973 → 0,6958 | −0,0015 |

- Mọi mô hình chỉ đoán đúng 49–52% số ngày.
- Không mô hình có học nào có log-loss thấp hơn mức sàn `b0maj` (0,6938), ở cả hai cấu hình. Xác suất của mô hình chưa tốt hơn việc luôn đoán tỷ lệ ngày tăng của tập train.
- Với 2.434 ngày, sai số chuẩn của DA khoảng 1 điểm % (ước lượng thô), nên các Δ ở trên nằm trong mức nhiễu. Riêng `b4` có ΔDA −1,5 điểm, hơi vượt mức đó, nhưng Δlog-loss của nó lại âm (On-chain tốt hơn chút), nên chưa đủ để nói On-chain làm `b4` tệ đi. Cần kiểm định của Vi.
- Δ đổi dấu giữa các mô hình: ΔDA dương ở `lr`, `b2`, âm ở `b3`, `b4`. Không có xu hướng chung nào.
- Δ rất nhạy với bộ OHLCV: với 7 đặc trưng tạm (bản 28/10), ΔDA của `lr` là −0,0045; đổi sang 10 đặc trưng thì thành +0,0078. Δlog-loss của `lr` vẫn dương (On-chain làm xác suất tệ hơn), nên DA và log-loss không cùng chiều.
- `b0pers` dưới 50%: trong giai đoạn 2020–2026, hướng giá ngày mai hay ngược với hôm nay.
- `b3` cấu hình OHLCV-only có log-loss tệ hơn mức sàn `b0maj`. Có những ngày mô hình quá tự tin (p̂ tới 0,987); cần xem lại theo từng fold.
- Tỷ lệ ngày tăng đổi mạnh theo năm, từ khoảng 46% (2022) đến 58% (2020) ([`eda_05_label_balance.png`](../tram_features_backtest/reports/figures/eda_05_label_balance.png)). Vì vậy DA của từng fold nên so với tỷ lệ tăng của chính giai đoạn đó, không so với 0,5.

Backtest nhanh bằng `tram_features_backtest/src/backtest.py` trên các dự báo này (long nếu p̂ > 0,5, phí 0,1%/lượt, 2020-01-02 → 2026-08-31). Bảng đầy đủ và equity curve là task 3.7 của Trâm.

| Mô hình | Lợi nhuận tích luỹ: OHLCV → +On-chain | Sharpe | % ngày long |
|---|---|---|---|
| buy-and-hold | +990% | 0,90 | 100% |
| `b0maj` | +72% | 0,43 | 63% |
| `b0pers` | −71% | −0,23 | 51% |
| `lr` | +19% → +245% | 0,31 → 0,63 | 56% → 55% |
| `b2` (thử) | +241% → +14% | 0,62 → 0,29 | 55% → 54% |
| `b3` | +286% → +63% | 0,65 → 0,41 | 73% → 65% |
| `b4` (thử) | −43% → −65% | 0,11 → −0,05 | 63% → 58% |

- Không mô hình nào thắng buy-and-hold. `b4` còn lỗ ở cả hai cấu hình.
- Xếp hạng theo lợi nhuận khác xếp hạng theo DA, vì lời lỗ phụ thuộc vào việc đoán đúng những ngày giá biến động mạnh. Ví dụ `b2` có DA gần như không đổi giữa hai cấu hình (0,512 → 0,514) nhưng lợi nhuận giảm từ +241% còn +14%. Δ theo lợi nhuận cũng đổi dấu giữa các mô hình, nên chưa nói được gì về On-chain.
- `b3` OHLCV-only ở trạng thái long 73% số ngày, trong khi chỉ 51% số ngày test là ngày tăng: mô hình lệch hẳn về phía "tăng".
- `b0pers` cho −71% ở cả `backtest.py` của Trâm và kết quả ở đây, dù hai bên viết code độc lập. Code của Trâm còn kiểm `y_true` = 1[close_{t+1} > close_t] trước khi chạy. Hai điều này xác nhận quy ước ngày của dự báo là đúng.

**Chưa dùng các con số này để kết luận RQ1:** chưa có kiểm định (Vi, 2.3), `b2` và `b4` mới là bản chạy thử, và còn thiếu B5 (LSTM, Ngân).

## Điểm cần nhóm chốt

1. **Phương án Flows (họp 1.19):** đang dùng A (`cm_NetflowExNtv` + `cm_SplyExNtv`). Nếu chốt B, đổi `flows_set: B` trong `configs/features_onchain.yaml` rồi chạy lại các cấu hình OHLCV + On-chain.
2. **Cửa sổ train 1.095 ngày (task 4.5):** fold đầu cần train từ 31/12/2016, nhưng OHLCV chỉ có từ 17/08/2017. Đề xuất: với cửa sổ này, bắt đầu test từ khoảng 09/2020, rồi so độ nhạy trên phần ngày test chung của cả ba cửa sổ.
3. **Fold cuối chỉ có 4 ngày test** (27–30/08/2026). Ghi chú trong mục Method.
4. **Hai điểm Trâm nêu, ảnh hưởng trực tiếp tới đặc trưng của pipeline** (chi tiết ở mục "Điểm cần nhóm chốt" trong README của `tram_features_backtest/`):
   - Ngày thiếu nến 2018-02-08 và 2018-02-09 (`dlog_volume` rất lớn) nằm trong cửa sổ train của các fold đầu. Có lọc ngày `ohlcv_n_bars_1h < 24` không?
   - Đặc trưng trùng thông tin ([`eda_04_feature_corr.png`](../tram_features_backtest/reports/figures/eda_04_feature_corr.png)): `cm_CapMVRVCur` gần như trùng return của ngày t (ρ = 0,98), `nvt_ratio_bc` với `bc_est_tx_vol_usd` (ρ = −1,00), `cm_HashRate` với `puell_multiple` (ρ = 0,95). Cần lưu ý khi diễn giải Δ của nhóm Valuation ở E2.
