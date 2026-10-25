# tram_features_backtest — Backtest, đặc trưng OHLCV, EDA

Code, dữ liệu và hình cho task **1.12 – 1.14** (Trâm).

## Trạng thái task

| Task | Nội dung | Trạng thái | Kết quả chính | File |
|---|---|---|---|---|
| 1.12 | Module backtest | ✅ Xong | Long/flat từ p̂, phí 0,1%/lượt, lợi nhuận tích luỹ, Sharpe (√365), MDD, so với buy-and-hold. 5 test qua; test "luôn long = buy-and-hold" khớp trên giá thật 2020–2026 | `src/backtest.py`, `reports/backtest_sanity.md` |
| 1.13 | Đặc trưng OHLCV | ✅ Xong | 10 đặc trưng, 3.302 ngày; từ 2018-01-01 không còn NaN. Có test chống nhìn tương lai | `src/features_ohlcv.py`, `data/features_ohlcv.csv` |
| 1.14 | Khung EDA | ✅ Xong | Notebook chạy được trên `master_daily.csv`; 5 hình trong `reports/figures/` | `notebooks/eda.ipynb` |

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
    ├── backtest_sanity.md
    └── figures/eda_0*.png
```

## Dữ liệu đầu vào

| File | Người làm | Dùng cho |
|---|---|---|
| `crawl_data/khoa-btc-onchain-ohlcv-daily-2016-2026/master_daily.csv` | Khoa (1.7) | Giá ngày (cột `ohlcv_*`), On-chain cho EDA |
| `crawl_data/khoa-btc-onchain-ohlcv-daily-2016-2026/data_dictionary.csv` | Khoa (1.7) | Nhóm của từng chỉ số On-chain |
| `crawl_data/vi_ohlcv_flows/data/clean/ohlcv_1h_clean.csv` | Vi (1.9) | Realized volatility từ return 1h |

Giá đóng cửa ngày trong hai file của Khoa và Vi khớp nhau trên toàn bộ 3.287 ngày trùng nhau.

## `data/features_ohlcv.csv`

Dòng ngày t chỉ dùng dữ liệu đến hết ngày t (UTC).

| Cột | Công thức |
|---|---|
| `date` | Ngày UTC |
| `logret_lag1` … `logret_lag5` | `logret_lag1` = ln(C_t / C_{t−1}) là return của chính ngày t; `logret_lagk` = return của ngày t−k+1 |
| `hl_range` | (H_t − L_t) / C_t |
| `dlog_volume` | ln(V_t + 1) − ln(V_{t−1} + 1) |
| `rsi_14` | RSI 14 ngày kiểu Wilder, 0–100 |
| `rv_7d`, `rv_30d` | √(tổng bình phương log return 1h trong 7 / 30 ngày gần nhất), không năm hoá |

## `src/backtest.py`

- `run_backtest(close, p_hat)`: p̂ ở dòng ngày t là dự báo cho ngày t+1. Vị thế ngày t+1 = 1 nếu p̂ > 0,5. Lợi nhuận = vị thế × return − 0,1% × |đổi vị thế|.
- `summarize(bt)`: bảng `cum_return`, `sharpe`, `mdd`, `n_trades`, `pct_days_long` cho chiến lược và buy-and-hold.
- Ví dụ dùng sau khi có dự báo walk-forward:

```python
from backtest import run_backtest, summarize
bt = run_backtest(close, preds["p_hat"])   # close, p_hat đều index theo ngày
print(summarize(bt))
```

Kết quả chạy thử 01/2020 – 08/2026: buy-and-hold +991%, Sharpe 0,90, MDD −77%; B0 persistence −71%.

## Chạy lại

Từ thư mục gốc repo (cần pandas, numpy, matplotlib, tabulate, jupyter):

```
python tram_features_backtest/src/features_ohlcv.py
python tram_features_backtest/src/backtest.py
```

Mỗi file chạy test trước rồi mới chạy trên dữ liệu thật. Thêm `--test` để chỉ chạy test.
Mở `notebooks/eda.ipynb` bằng Jupyter hoặc VS Code từ thư mục `notebooks/` rồi chọn Run All.

## Điểm cần nhóm chốt

1. **Quy ước "trễ 1 ngày":** `logret_lag1` đang là return của chính ngày t. Cần thống nhất với Ngọc (1.15, 1.16).
2. **Độ trễ công bố On-chain:** README của Khoa ghi nên dùng On-chain đến ngày t−1. OHLCV thì dùng được đến hết ngày t.
3. **Ngày thiếu nến:** `dlog_volume` rất lớn ở 2018-02-08 và 2018-02-09 vì Binance ngừng giao dịch. Có giữ những ngày `ohlcv_n_bars_1h < 24` không?
4. **Tương quan mạnh phát hiện trong EDA** (Spearman trên log-diff, 2018–08/2026):
   - `cm_CapMVRVCur` với `logret_lag1`: ρ = 0,98. MVRV thay đổi gần như theo giá, nên nhóm Valuation có thể chỉ lặp lại thông tin giá. Điều này ảnh hưởng trực tiếp tới Δ của RQ1, RQ2.
   - `bc_est_tx_vol_usd` với `nvt_ratio_bc`: ρ = −1,00. NVT được tính bằng cách chia cho chính volume đó.
   - `cm_HashRate` với `puell_multiple`: ρ = 0,95.
