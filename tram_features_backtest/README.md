# tram_features_backtest — Backtest, đặc trưng OHLCV, EDA

Code, dữ liệu và hình cho task **1.12 – 1.14** (Trâm). Từ 29/10/2026 chỉ đọc **data freeze v1.0** (`data/processed/`).

## Trạng thái task

| Task | Nội dung | Trạng thái | Kết quả chính | File |
|---|---|---|---|---|
| 1.12 | Module backtest | ✅ Xong | Long/flat từ p̂, phí 0,1%/lượt, lợi nhuận tích luỹ, Sharpe (√365), MDD, so với buy-and-hold. 5 test qua. Đọc thẳng dự báo của `ngoc_pipeline` và kiểm quy ước ngày bằng `y_true` | `src/backtest.py`, `reports/backtest_sanity.md`, `reports/backtest_e1_draft.md` |
| 1.13 | Đặc trưng OHLCV | ✅ Xong | 10 đặc trưng, 3.302 ngày; từ 2017-12-31 (ngày train đầu của fold 0) không còn NaN. Có hàm `build_ohlcv_features(master)` cắm thẳng vào pipeline của Ngọc | `src/features_ohlcv.py`, `data/features_ohlcv.csv` |
| 1.14 | Khung EDA | ✅ Xong | Notebook chạy trên data freeze, nhận tham số `FLOWS_SET` = A hoặc B; 5 hình trong `reports/figures/` | `notebooks/eda.ipynb` |

## Cấu trúc

```
tram_features_backtest/
├── README.md
├── src/
│   ├── features_ohlcv.py     (1) task 1.13 – chạy trước
│   └── backtest.py           (2) task 1.12
├── data/
│   └── features_ohlcv.csv    đầu ra của (1)
├── notebooks/
│   └── eda.ipynb             (3) task 1.14 – cần data/features_ohlcv.csv
└── reports/
    ├── backtest_sanity.md                  luôn long, B0 persistence trên giá thật
    ├── backtest_e1_draft.md / .csv         backtest dự báo E1 nháp của Ngọc
    ├── integration_check_ngoc_pipeline.md  thử ghép đặc trưng vào pipeline Ngọc (bản sao, không chính thức)
    └── figures/eda_0*.png
```

## Dữ liệu đầu vào

Chỉ đọc `data/processed/` (data freeze v1.0 của Khoa và Vi). Mỗi lần chạy đều so sha256 với `MANIFEST.json`; lệch thì dừng.

| File | Dùng cho |
|---|---|
| `data/processed/master_daily.csv` | Giá ngày (cột `ohlcv_*`); On-chain cho EDA |
| `data/processed/ohlcv_1h_clean.csv` | Realized volatility từ return 1h |
| `data/processed/data_dictionary_{A,B}.csv` | EDA: danh sách chỉ số On-chain theo phương án Flows |
| `ngoc_pipeline/results/predictions/*.csv` | Backtest dự báo E1 |

So với bản `crawl_data/` dùng trước đây: giá trị giống hệt; freeze chỉ thêm cột `cm_NetflowExNtv`.

## `data/features_ohlcv.csv`

Dòng ngày t chỉ dùng dữ liệu đến hết ngày t (UTC).

| Cột | Công thức |
|---|---|
| `date` | Ngày UTC |
| `logret_lag1` … `logret_lag5` | `logret_lag1` = ln(C_t / C_{t−1}) là return của chính ngày t (cùng nghĩa `ret_1` của Ngọc); `logret_lagk` = return của ngày t−k+1 |
| `hl_range` | (H_t − L_t) / C_t |
| `dlog_volume` | ln V_t − ln V_{t−1} (bỏ "+1" của bản trước để khớp `features.py` của Ngọc; lệch tối đa 0,0026) |
| `rsi_14` | RSI 14 ngày kiểu Wilder, 0–100 |
| `rv_7d`, `rv_30d` | √(tổng bình phương log return 1h trong 7 / 30 ngày gần nhất), không năm hoá |

## Dùng trong pipeline của Ngọc

```python
# ngoc_pipeline/src/features.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tram_features_backtest/src"))
from features_ohlcv import build_ohlcv_features      # thay hàm tạm cùng tên
```

`build_ohlcv_features(master)` nhận đúng `load_master()` của Ngọc, trả cột `date` (datetime64[ns]) + 10 đặc trưng, cùng số dòng và thứ tự. 7 cột chung khớp bản tạm của Ngọc đến sai số 2e-15. Đã thử trên bản sao: `--stage data` báo `10 OHLCV + 33 On-chain … NaN từ ngày đó: 0`, E1 chạy đủ 82 fold (xem `reports/integration_check_ngoc_pipeline.md`).

## `src/backtest.py`

- `run_backtest(close, p_hat)`: p̂ ở dòng ngày t là dự báo cho ngày t+1. Vị thế ngày t+1 = 1 nếu p̂ > 0,5. Lợi nhuận = vị thế × return − 0,1% × |đổi vị thế|.
- `summarize(bt)`: bảng `cum_return`, `sharpe`, `mdd`, `n_trades`, `pct_days_long` cho chiến lược và buy-and-hold.
- `backtest_predictions(pred_dir)`: backtest mọi file `{model}_{config}.csv` của `ngoc_pipeline`. Trước khi chạy, kiểm `y_true` = 1[close_{t+1} > close_t] để chắc quy ước ngày khớp.

Kết quả 01/2020 – 08/2026: buy-and-hold +990%, Sharpe 0,90, MDD −77%. B0 persistence −71%, khớp với `b0pers` của Ngọc (hai code độc lập). Chưa mô hình nào trong E1 nháp thắng buy-and-hold (`reports/backtest_e1_draft.md`).

## Chạy lại

Từ thư mục gốc repo (cần pandas, numpy, matplotlib, tabulate, jupyter):

```
python tram_features_backtest/src/features_ohlcv.py
python tram_features_backtest/src/backtest.py
```

Mỗi file chạy test trước rồi mới chạy trên dữ liệu thật. Thêm `--test` để chỉ chạy test.
Mở `notebooks/eda.ipynb` bằng Jupyter hoặc VS Code từ thư mục `notebooks/` rồi chọn Run All. Nhóm chốt Flows B thì đổi `FLOWS_SET = "B"` ở ô đầu.

## Điểm cần nhóm chốt

1. ~~Quy ước "trễ 1 ngày"~~: đã khớp. `logret_lag1` = `ret_1` của Ngọc = return của chính ngày t.
2. ~~Độ trễ công bố On-chain~~: đã xử lý trong pipeline Ngọc (`shift_days: 1`).
3. **Ngày thiếu nến:** `dlog_volume` rất lớn ở 2018-02-08 và 2018-02-09 vì Binance ngừng giao dịch. Có giữ những ngày `ohlcv_n_bars_1h < 24` không? Hai ngày này nằm trong cửa sổ train của các fold đầu.
4. **Bộ OHLCV cho E1 chính thức:** Δ nháp của `lr` đổi dấu (−0,45 → +0,78 điểm DA) chỉ vì đổi 7 đặc trưng tạm sang 10 đặc trưng. Nên chốt bộ OHLCV trước khi chạy E1 chính thức.
5. **Tương quan mạnh phát hiện trong EDA** (Spearman trên log-diff, 2018–08/2026):
   - `cm_CapMVRVCur` với `logret_lag1`: ρ = 0,98. MVRV thay đổi gần như theo giá, nên nhóm Valuation có thể chỉ lặp lại thông tin giá. Điều này ảnh hưởng trực tiếp tới Δ của RQ1, RQ2.
   - `bc_est_tx_vol_usd` với `nvt_ratio_bc`: ρ = −1,00. NVT được tính bằng cách chia cho chính volume đó.
   - `cm_HashRate` với `puell_multiple`: ρ = 0,95.
