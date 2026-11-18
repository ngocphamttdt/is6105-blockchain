# Backtest trên dự báo E1 nháp của Ngọc

Tạo bởi `src/backtest.py` (hàm `report_predictions`) từ `pipeline/results/predictions/`. Long nếu p̂ > 0,5, phí 0,1%/lượt, Sharpe năm hoá √365. Ngày giữ vị thế: 2020-01-02 → 2026-08-31 (2434 ngày).

Chưa dùng để kết luận RQ1: dự báo E1 vẫn là bản nháp (OHLCV tạm 7 đặc trưng, chưa kiểm định).

| model        | config        |      da |   cum_return |   sharpe |    mdd |   n_trades |   pct_days_long |
|:-------------|:--------------|--------:|-------------:|---------:|-------:|-----------:|----------------:|
| b0maj        | ohlcv         |   0.501 |        0.716 |    0.426 | -0.824 |          7 |           0.63  |
| b0maj        | ohlcv_onchain |   0.501 |        0.716 |    0.426 | -0.824 |          7 |           0.63  |
| b0pers       | ohlcv         |   0.465 |       -0.713 |   -0.225 | -0.794 |       1302 |           0.509 |
| b0pers       | ohlcv_onchain |   0.465 |       -0.713 |   -0.225 | -0.794 |       1302 |           0.509 |
| b2           | ohlcv         |   0.512 |        2.409 |    0.62  | -0.802 |        857 |           0.547 |
| b2           | ohlcv_onchain |   0.514 |        0.137 |    0.294 | -0.764 |        771 |           0.541 |
| b3           | ohlcv         |   0.506 |        2.864 |    0.648 | -0.724 |        264 |           0.728 |
| b3           | ohlcv_onchain |   0.503 |        0.629 |    0.41  | -0.802 |        327 |           0.647 |
| b4           | ohlcv         |   0.505 |       -0.429 |    0.107 | -0.864 |        509 |           0.627 |
| b4           | ohlcv_onchain |   0.49  |       -0.65  |   -0.052 | -0.93  |        559 |           0.58  |
| b5           | ohlcv         |   0.507 |        0.994 |    0.455 | -0.852 |        363 |           0.575 |
| b5           | ohlcv_onchain |   0.506 |        1.919 |    0.608 | -0.564 |        288 |           0.334 |
| lr           | ohlcv         |   0.511 |        0.192 |    0.314 | -0.866 |        843 |           0.564 |
| lr           | ohlcv_onchain |   0.519 |        2.447 |    0.629 | -0.654 |        857 |           0.547 |
| buy_and_hold | -             | nan     |        9.901 |    0.901 | -0.766 |          1 |           1     |
