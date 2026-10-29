# Thử ghép đặc trưng OHLCV của Trâm vào pipeline của Ngọc (29/10/2026)

**Không phải kết quả chính thức.** Chạy trên một bản sao của repo; không sửa file nào của `ngoc_pipeline/`.
Mục đích: kiểm tra `build_ohlcv_features` của Trâm cắm vào được pipeline, và xem Δ nháp thay đổi thế nào.

## Cách làm

Trong bản sao, `build_ohlcv_features` của `ngoc_pipeline/src/features.py` được thay bằng hàm cùng tên trong
`tram_features_backtest/src/features_ohlcv.py`. Mọi thứ khác giữ nguyên: data freeze v1.0, flows set A,
82 fold, cùng grid, cùng mô hình `lr b0pers b0maj b3`.

`--stage data` in ra:

```
Data freeze v1.0: sha256 khớp 4 file; flows set A; 10 OHLCV + 33 On-chain; 82 fold, train fold 0 từ 2017-12-31; NaN từ ngày đó: 0
```

7 đặc trưng chung (`ret_1..5` = `logret_lag1..5`, `hl_range`, `dlog_volume`) khớp với bản tạm của Ngọc
đến sai số 2e-15. Bản của Trâm thêm `rsi_14`, `rv_7d`, `rv_30d`.

## Δ nháp: 7 đặc trưng tạm so với 10 đặc trưng của Trâm

DA, AUC: cao hơn là tốt hơn. Log-loss: thấp hơn là tốt hơn. 2.434 ngày test.

| Mô hình   | Metric   |   OHLCV-only (7 tạm) |   OHLCV-only (10 của Trâm) |   Δ (7 tạm) |   Δ (10 của Trâm) |
|:----------|:---------|---------------------:|---------------------------:|------------:|------------------:|
| lr        | da       |               0.5164 |                     0.5111 |     -0.0045 |            0.0078 |
| lr        | log_loss |               0.6941 |                     0.6953 |      0.0056 |            0.0054 |
| lr        | auc      |               0.5123 |                     0.5098 |      0.0028 |            0.0034 |
| b3        | da       |               0.5094 |                     0.5062 |     -0.0058 |           -0.0033 |
| b3        | log_loss |               0.7023 |                     0.7009 |     -0.0062 |           -0.005  |
| b3        | auc      |               0.5059 |                     0.5027 |     -0.006  |           -0.0045 |

- `lr`: ΔDA đổi dấu, từ −0,45 điểm thành +0,78 điểm, chỉ vì đổi bộ đặc trưng OHLCV. Cả hai đều nhỏ hơn
  sai số chuẩn của DA (khoảng 1 điểm), nên đây là nhiễu, không phải bằng chứng On-chain có ích hay vô ích.
- Bài học cho RQ1: Δ rất nhạy với baseline OHLCV, nên cần chốt bộ OHLCV trước khi chạy E1 chính thức và kiểm định.

## Backtest các dự báo thử (long nếu p̂ > 0,5, phí 0,1%/lượt)

| model        | config        |      da |   cum_return |   sharpe |    mdd |   n_trades |
|:-------------|:--------------|--------:|-------------:|---------:|-------:|-----------:|
| b0maj        | ohlcv         |   0.501 |        0.716 |    0.426 | -0.824 |          7 |
| b0maj        | ohlcv_onchain |   0.501 |        0.716 |    0.426 | -0.824 |          7 |
| b0pers       | ohlcv         |   0.465 |       -0.713 |   -0.225 | -0.794 |       1302 |
| b0pers       | ohlcv_onchain |   0.465 |       -0.713 |   -0.225 | -0.794 |       1302 |
| b3           | ohlcv         |   0.506 |        2.864 |    0.648 | -0.724 |        264 |
| b3           | ohlcv_onchain |   0.503 |        0.629 |    0.41  | -0.802 |        327 |
| lr           | ohlcv         |   0.511 |        0.192 |    0.314 | -0.866 |        843 |
| lr           | ohlcv_onchain |   0.519 |        2.447 |    0.629 | -0.654 |        857 |
| buy_and_hold | -             | nan     |        9.901 |    0.901 | -0.766 |          1 |

Không mô hình nào thắng buy-and-hold (+990%). Lợi nhuận backtest xếp hạng khác với DA
(ví dụ `b3` OHLCV-only có DA thấp hơn `lr` nhưng lời nhiều hơn), vì lời lỗ phụ thuộc vào việc
đoán đúng những ngày giá biến động mạnh.
