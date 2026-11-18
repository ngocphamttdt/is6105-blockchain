# Backtest sanity check trên giá thật (01/2020 – 08/2026)

Tạo bởi `src/backtest.py` (hàm `demo_real_data`). Phí 0,1%/lượt, Sharpe năm hoá √365.

## always_long

|              |   cum_return |   sharpe |    mdd |   n_trades |   pct_days_long |
|:-------------|-------------:|---------:|-------:|-----------:|----------------:|
| strategy     |         9.91 |    0.901 | -0.766 |          1 |               1 |
| buy_and_hold |         9.91 |    0.901 | -0.766 |          1 |               1 |

## B0_persistence

|              |   cum_return |   sharpe |    mdd |   n_trades |   pct_days_long |
|:-------------|-------------:|---------:|-------:|-----------:|----------------:|
| strategy     |       -0.713 |   -0.225 | -0.794 |       1302 |           0.509 |
| buy_and_hold |        9.91  |    0.901 | -0.766 |          1 |           1     |
