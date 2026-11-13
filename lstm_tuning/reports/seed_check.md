# Kiểm tra theo random seed: B5 LSTM, 10 đặc trưng OHLCV

Tạo bởi `lstm_tuning/src/seed_check.py`. Giữ nguyên dữ liệu, cách chia và hai cấu hình, chỉ đổi seed (10 seed: 0–9). Mỗi seed là **một mô hình riêng**, không lấy trung bình (báo cáo chính `lstm_tuning.md` dùng p̂ trung bình 3 seed 42–44).

- Trước tune: lookback=30, hidden=32, dropout=0.0, lr=0.001
- Sau tune (chọn trên validation ở `lstm_tuning.py`): lookback=10, hidden=32, dropout=0.2, lr=0.0001
- Mốc không phụ thuộc seed: LR 1 đặc trưng (WF 52.22%, test 52.42%), luật đảo chiều (WF 53.25%, test 50.98%), đoán 50/50 (log-loss 0,6931). LR dùng solver tất định nên chạy lại luôn ra cùng kết quả.
- Vì sao 10 seed: số seed cần để phát hiện chênh lệch d (so ghép cặp) ≈ (2σ/d)². Với σ ≈ 0,6–0,8 điểm DA, phát hiện chênh lệch 0,5 điểm cần khoảng 10 seed; sai số của DA trung bình 10 seed ≈ 0,2 điểm, đủ nhỏ so với khoảng cách tới LR (1,2 điểm). Seed được cố định trước (0, 1, …), không chọn seed theo kết quả.

## Phân bố theo seed

|                           |   trước tune |   sau tune |
|:--------------------------|-------------:|-----------:|
| ('wf_DA', 'mean')         |      49.9055 |    50.5218 |
| ('wf_DA', 'std')          |       0.5244 |     0.7344 |
| ('wf_DA', 'min')          |      48.8085 |    49.3837 |
| ('wf_DA', 'max')          |      50.5752 |    51.3969 |
| ('wf_log_loss', 'mean')   |       0.6967 |     0.6938 |
| ('wf_log_loss', 'std')    |       0.0016 |     0.0006 |
| ('wf_log_loss', 'min')    |       0.6944 |     0.6929 |
| ('wf_log_loss', 'max')    |       0.6991 |     0.6949 |
| ('test_DA', 'mean')       |      50.74   |    50.6064 |
| ('test_DA', 'std')        |       0.909  |     1.0268 |
| ('test_DA', 'min')        |      48.9209 |    48.407  |
| ('test_DA', 'max')        |      52.1069 |    51.7986 |
| ('test_log_loss', 'mean') |       0.6925 |     0.6927 |
| ('test_log_loss', 'std')  |       0.0004 |     0.0004 |
| ('test_log_loss', 'min')  |       0.6919 |     0.692  |
| ('test_log_loss', 'max')  |       0.6932 |     0.6932 |

## Sau tune có thật sự tốt hơn trước tune? (ghép cặp theo seed)

| metric                    |      TB |   độ lệch chuẩn |   số seed sau tune tốt hơn |   p (t-test ghép cặp) |
|:--------------------------|--------:|----------------:|---------------------------:|----------------------:|
| ΔDA walk-forward (điểm %) |  0.6163 |          0.5576 |                          9 |                0.0068 |
| Δlog-loss walk-forward    | -0.0029 |          0.0016 |                         10 |                0.0003 |
| ΔDA test (điểm %)         | -0.1336 |          1.4575 |                          5 |                0.7785 |

## Có seed nào vượt các mốc không?

| cấu hình   |   seed có WF DA > LR 1 đặc trưng (52.22%) |   seed có WF DA > luật đảo chiều (53.25%) |   seed có WF log-loss < 0,6931 |
|:-----------|------------------------------------------:|------------------------------------------:|-------------------------------:|
| trước tune |                                         0 |                                         0 |                              0 |
| sau tune   |                                         0 |                                         0 |                              1 |

## Từng seed

|   seed | cấu hình   |   test_DA |   test_log_loss |   wf_DA |   wf_log_loss |   wf_pct_pred_up |
|-------:|:-----------|----------:|----------------:|--------:|--------------:|-----------------:|
|      0 | trước tune |     50.46 |          0.6928 |   50.25 |        0.6945 |            54.48 |
|      0 | sau tune   |     50.77 |          0.6932 |   50.78 |        0.6929 |            49.67 |
|      1 | trước tune |     50.05 |          0.6929 |   49.92 |        0.697  |            69.19 |
|      1 | sau tune   |     51.7  |          0.6925 |   51.4  |        0.6942 |            80.12 |
|      2 | trước tune |     50.15 |          0.6926 |   50.37 |        0.6977 |            70.95 |
|      2 | sau tune   |     50.87 |          0.6925 |   51.07 |        0.6946 |            78.88 |
|      3 | trước tune |     50.57 |          0.6928 |   49.42 |        0.6944 |            47.58 |
|      3 | sau tune   |     51.8  |          0.693  |   50.04 |        0.6933 |            33.81 |
|      4 | trước tune |     48.92 |          0.6932 |   49.67 |        0.6967 |            60.07 |
|      4 | sau tune   |     49.95 |          0.6927 |   50.08 |        0.6938 |            62.53 |
|      5 | trước tune |     51.39 |          0.6919 |   49.96 |        0.6957 |            40.55 |
|      5 | sau tune   |     49.74 |          0.6926 |   49.38 |        0.6949 |            37.43 |
|      6 | trước tune |     51.49 |          0.6926 |   49.75 |        0.6968 |            53    |
|      6 | sau tune   |     48.41 |          0.6932 |   49.96 |        0.6937 |            47.29 |
|      7 | trước tune |     51.28 |          0.692  |   50.33 |        0.6989 |            52.51 |
|      7 | sau tune   |     50.46 |          0.6924 |   51.4  |        0.6931 |            41.17 |
|      8 | trước tune |     50.98 |          0.6926 |   48.81 |        0.6964 |            53.86 |
|      8 | sau tune   |     50.98 |          0.6924 |   49.88 |        0.6937 |            52.88 |
|      9 | trước tune |     52.11 |          0.692  |   50.58 |        0.6991 |            73.79 |
|      9 | sau tune   |     51.39 |          0.692  |   51.23 |        0.6937 |            80.85 |
