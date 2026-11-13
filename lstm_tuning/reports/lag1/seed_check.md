# Kiểm tra theo random seed: B5 LSTM, chỉ logret_lag1

Tạo bởi `lstm_tuning/src/seed_check.py`. Giữ nguyên dữ liệu, cách chia và hai cấu hình, chỉ đổi seed (10 seed: 0–9). Mỗi seed là **một mô hình riêng**, không lấy trung bình (báo cáo chính `lstm_tuning.md` dùng p̂ trung bình 3 seed 42–44).

- Trước tune: lookback=30, hidden=32, dropout=0.0, lr=0.001
- Sau tune (chọn trên validation ở `lstm_tuning.py`): lookback=10, hidden=8, dropout=0.2, lr=0.001
- Mốc không phụ thuộc seed: LR 1 đặc trưng (WF 52.22%, test 52.42%), luật đảo chiều (WF 53.25%, test 50.98%), đoán 50/50 (log-loss 0,6931). LR dùng solver tất định nên chạy lại luôn ra cùng kết quả.
- Vì sao 10 seed: số seed cần để phát hiện chênh lệch d (so ghép cặp) ≈ (2σ/d)². Với σ ≈ 0,6–0,8 điểm DA, phát hiện chênh lệch 0,5 điểm cần khoảng 10 seed; sai số của DA trung bình 10 seed ≈ 0,2 điểm, đủ nhỏ so với khoảng cách tới LR (1,2 điểm). Seed được cố định trước (0, 1, …), không chọn seed theo kết quả.

## Phân bố theo seed

|                           |   trước tune |   sau tune |
|:--------------------------|-------------:|-----------:|
| ('wf_DA', 'mean')         |      50.5012 |    51.0312 |
| ('wf_DA', 'std')          |       0.5211 |     0.593  |
| ('wf_DA', 'min')          |      49.9589 |    49.8357 |
| ('wf_DA', 'max')          |      51.3147 |    51.8077 |
| ('wf_log_loss', 'mean')   |       0.6943 |     0.6938 |
| ('wf_log_loss', 'std')    |       0.0007 |     0.0005 |
| ('wf_log_loss', 'min')    |       0.6933 |     0.693  |
| ('wf_log_loss', 'max')    |       0.6953 |     0.6945 |
| ('test_DA', 'mean')       |      51.0175 |    50.8941 |
| ('test_DA', 'std')        |       0.7303 |     0.5146 |
| ('test_DA', 'min')        |      50.2569 |    50.3597 |
| ('test_DA', 'max')        |      52.2097 |    52.2097 |
| ('test_log_loss', 'mean') |       0.6927 |     0.6932 |
| ('test_log_loss', 'std')  |       0.0002 |     0.0003 |
| ('test_log_loss', 'min')  |       0.6924 |     0.6926 |
| ('test_log_loss', 'max')  |       0.693  |     0.6935 |

## Sau tune có thật sự tốt hơn trước tune? (ghép cặp theo seed)

| metric                    |      TB |   độ lệch chuẩn |   số seed sau tune tốt hơn |   p (t-test ghép cặp) |
|:--------------------------|--------:|----------------:|---------------------------:|----------------------:|
| ΔDA walk-forward (điểm %) |  0.53   |          0.8138 |                          9 |                0.0695 |
| Δlog-loss walk-forward    | -0.0006 |          0.0009 |                          7 |                0.0637 |
| ΔDA test (điểm %)         | -0.1233 |          1.0056 |                          5 |                0.7071 |

## Có seed nào vượt các mốc không?

| cấu hình   |   seed có WF DA > LR 1 đặc trưng (52.22%) |   seed có WF DA > luật đảo chiều (53.25%) |   seed có WF log-loss < 0,6931 |
|:-----------|------------------------------------------:|------------------------------------------:|-------------------------------:|
| trước tune |                                         0 |                                         0 |                              0 |
| sau tune   |                                         0 |                                         0 |                              2 |

## Từng seed

|   seed | cấu hình   |   test_DA |   test_log_loss |   wf_DA |   wf_log_loss |   wf_pct_pred_up |
|-------:|:-----------|----------:|----------------:|--------:|--------------:|-----------------:|
|      0 | trước tune |     51.9  |          0.6927 |   50.33 |        0.6935 |            50.45 |
|      0 | sau tune   |     50.77 |          0.6933 |   51.4  |        0.6939 |            41.82 |
|      1 | trước tune |     51.49 |          0.6928 |   50.08 |        0.6942 |            71.73 |
|      1 | sau tune   |     50.77 |          0.6926 |   50.82 |        0.6941 |            40.76 |
|      2 | trước tune |     50.77 |          0.6927 |   50.29 |        0.6942 |            72.19 |
|      2 | sau tune   |     50.87 |          0.6934 |   51.44 |        0.6938 |            72.6  |
|      3 | trước tune |     50.36 |          0.6924 |   51.31 |        0.6933 |            44.78 |
|      3 | sau tune   |     50.87 |          0.6935 |   51.36 |        0.694  |            43.1  |
|      4 | trước tune |     52.21 |          0.6925 |   51.27 |        0.6944 |            68.57 |
|      4 | sau tune   |     50.77 |          0.6931 |   49.84 |        0.6945 |            41.17 |
|      5 | trước tune |     51.7  |          0.6926 |   51.11 |        0.6938 |            40.39 |
|      5 | sau tune   |     50.46 |          0.6935 |   51.52 |        0.693  |            68.73 |
|      6 | trước tune |     50.57 |          0.6926 |   49.96 |        0.6953 |            38    |
|      6 | sau tune   |     52.21 |          0.6934 |   50.41 |        0.6938 |            73.29 |
|      7 | trước tune |     50.36 |          0.693  |   50.25 |        0.695  |            70.01 |
|      7 | sau tune   |     51.18 |          0.6933 |   51.81 |        0.6931 |            71.49 |
|      8 | trước tune |     50.26 |          0.6927 |   50.33 |        0.6948 |            41.99 |
|      8 | sau tune   |     50.67 |          0.6931 |   50.9  |        0.6935 |            71.73 |
|      9 | trước tune |     50.57 |          0.6926 |   50.08 |        0.6947 |            67.38 |
|      9 | sau tune   |     50.36 |          0.6929 |   50.82 |        0.6939 |            41.66 |
