# Backtest trên dự báo E1 nháp của Ngọc

Tạo bởi `src/backtest.py` (hàm `report_predictions`) từ `ngoc_pipeline/results/predictions/`. Long nếu p̂ > 0,5, phí 0,1%/lượt, Sharpe năm hoá √365. Ngày giữ vị thế: 2020-01-02 → 2026-08-31 (2434 ngày).

Chưa dùng để kết luận RQ1: dự báo E1 vẫn là bản nháp (OHLCV tạm 7 đặc trưng, chưa kiểm định).

| model        | config        |      da |   cum_return |   sharpe |    mdd |   n_trades |   pct_days_long |
|:-------------|:--------------|--------:|-------------:|---------:|-------:|-----------:|----------------:|
| b0maj        | ohlcv         |   0.501 |        0.716 |    0.426 | -0.824 |          7 |           0.63  |
| b0maj        | ohlcv_onchain |   0.501 |        0.716 |    0.426 | -0.824 |          7 |           0.63  |
| b0pers       | ohlcv         |   0.465 |       -0.713 |   -0.225 | -0.794 |       1302 |           0.509 |
| b0pers       | ohlcv_onchain |   0.465 |       -0.713 |   -0.225 | -0.794 |       1302 |           0.509 |
| b3           | ohlcv         |   0.509 |        2.092 |    0.591 | -0.719 |        192 |           0.678 |
| b3           | ohlcv_onchain |   0.504 |        0.402 |    0.366 | -0.801 |        311 |           0.641 |
| lr           | ohlcv         |   0.516 |        0.438 |    0.37  | -0.854 |        887 |           0.548 |
| lr           | ohlcv_onchain |   0.512 |        2.276 |    0.615 | -0.689 |        885 |           0.535 |
| buy_and_hold | -             | nan     |        9.901 |    0.901 | -0.766 |          1 |           1     |
