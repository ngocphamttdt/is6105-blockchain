# Tuning B5 LSTM (góp ý 3 của thầy): 10 đặc trưng OHLCV

Tạo bởi `analyses/lstm_tuning/src/lstm_tuning.py`. Đặc trưng: **10 đặc trưng OHLCV**, data freeze v1.0. Dữ liệu, cách chia, metric và luật B0 lấy từ `baseline_check` nên số liệu so được trực tiếp với bảng LR ở đó.

## Thiết lập

- Kiến trúc: LSTM 1 lớp → dropout → Linear, BCE, Adam, batch 32, tối đa 50 epoch, patience 5; p̂ trung bình 3 seed (42, 43, 44).
- Số seed: kết quả chính lấy p̂ trung bình 3 seed để giảm dao động do khởi tạo ngẫu nhiên; độ nhạy theo seed được kiểm riêng bằng `seed_check.py` (10 seed, mỗi seed một mô hình, `seed_check.md`). Seed được cố định trước khi chạy, không chọn seed theo kết quả.
- Mỗi lần học: scaler fit trên ngày train; early stopping trên 20% cuối phần train, **bỏ 1 ngày gap** giữa phần fit và phần early stopping; sau đó học lại từ đầu trên 100% phần train với số epoch tốt nhất. Batch xáo trộn trong phần train.
- Khác `model_lstm.py` (B5 trong `pipeline`): code đó không có gap ở early stopping, không xáo batch, không dropout và không học lại trên 100% train. Vì vậy cột "trước tune" ở đây được chạy lại bằng code này, không lấy con số 50,74% của báo cáo giữa kỳ.
- Grid (chốt trước khi chạy): lookback [10, 30] × hidden [8, 32] × dropout [0.0, 0.2] × lr [0.001, 0.0001] = 16 cấu hình. Trước tune = cấu hình B5 hiện tại (lookback=30, hidden=32, dropout=0.0, lr=0.001).
- Chọn theo log-loss validation (p̂ trung bình các seed), làm tròn 4 chữ số; hoà thì chọn mô hình đơn giản hơn (hidden nhỏ, lookback ngắn).

## Bước A. Grid: train 2018–2021 → validation 2022–2023 (chưa đụng test)

**Cấu hình được chọn: lookback=10, hidden=32, dropout=0.2, lr=0.0001.** Mốc: đoán 50/50 mọi ngày có log-loss 0,6931.

|   lookback |   hidden |   dropout |     lr | trước tune   |   train_log_loss |   val_log_loss |   val_log_loss_sd_seed |   train_DA |   val_DA |   val_pct_pred_up | best_epoch_seeds   |
|-----------:|---------:|----------:|-------:|:-------------|-----------------:|---------------:|-----------------------:|-----------:|---------:|------------------:|:-------------------|
|         10 |       32 |       0   | 0.0001 | False        |           0.6921 |         0.6936 |                 0.002  |       51.6 |     47.7 |              88.5 | 3/2/1              |
|         10 |       32 |       0.2 | 0.0001 | False        |           0.6921 |         0.6936 |                 0.002  |       52   |     47.5 |              89.3 | 4/2/1              |
|         30 |       32 |       0   | 0.0001 | False        |           0.6921 |         0.6936 |                 0.002  |       51.6 |     47.7 |              88.5 | 3/2/1              |
|         30 |       32 |       0.2 | 0.0001 | False        |           0.692  |         0.6936 |                 0.0019 |       51.4 |     47.7 |              88.8 | 4/2/2              |
|         10 |       32 |       0.2 | 0.001  | False        |           0.6886 |         0.6942 |                 0.0007 |       54.4 |     48.6 |              87.7 | 1/1/6              |
|         30 |       32 |       0.2 | 0.001  | False        |           0.6885 |         0.6943 |                 0.0007 |       54.4 |     48.6 |              87.1 | 1/1/6              |
|         10 |       32 |       0   | 0.001  | False        |           0.6881 |         0.6944 |                 0.0005 |       54.7 |     49.2 |              86.4 | 1/1/6              |
|         30 |       32 |       0   | 0.001  | True         |           0.688  |         0.6944 |                 0.0005 |       54.7 |     49.2 |              86.1 | 1/1/6              |
|         10 |        8 |       0.2 | 0.001  | False        |           0.6835 |         0.6945 |                 0.0004 |       56.4 |     48.6 |              80.5 | 1/14/28            |
|         30 |        8 |       0.2 | 0.001  | False        |           0.6834 |         0.6946 |                 0.0005 |       56.4 |     48.4 |              80.7 | 1/14/28            |
|         10 |        8 |       0   | 0.001  | False        |           0.6843 |         0.6955 |                 0.001  |       56   |     49.9 |              82.7 | 1/14/18            |
|         30 |        8 |       0   | 0.001  | False        |           0.6842 |         0.6956 |                 0.0012 |       55.9 |     49.9 |              82.4 | 1/14/18            |
|         10 |        8 |       0   | 0.0001 | False        |           0.6905 |         0.6959 |                 0.0016 |       52.1 |     48.4 |              80.4 | 2/50/39            |
|         10 |        8 |       0.2 | 0.0001 | False        |           0.6905 |         0.6959 |                 0.0019 |       52   |     48   |              83   | 2/50/48            |
|         30 |        8 |       0   | 0.0001 | False        |           0.6905 |         0.696  |                 0.0016 |       52.5 |     48.3 |              80.5 | 2/50/39            |
|         30 |        8 |       0.2 | 0.0001 | False        |           0.6905 |         0.696  |                 0.0018 |       51.8 |     48   |              83   | 2/50/48            |

Cột: `val_log_loss_sd_seed` = độ lệch chuẩn val log-loss giữa các seed (cho biết chênh lệch giữa hai cấu hình có lớn hơn dao động do seed không); `best_epoch_seeds` = số epoch early stopping chọn ở từng seed.

## Bước A. Kết quả test 2024 – 08/2026 (chạy 1 lần)

Học lại trên train + validation. Trên train + validation: persistence đúng 45.2% → luật đảo chiều.

|                                                                    |   n |   DA (%) | CI 95% DA    |   p (DA = 50%) |   bal_acc |    MCC |   AUC |   log_loss |   brier |   pct_pred_up |
|:-------------------------------------------------------------------|----:|---------:|:-------------|---------------:|----------:|-------:|------:|-----------:|--------:|--------------:|
| B0 Majority (p̂ = tỷ lệ tăng của train)                             | 973 |    50.67 | [47.5; 53.9] |          0.7   |     0.5   |  0     | 0.5   |     0.6931 |   0.25  |         1     |
| B0 Đảo chiều (mai ngược chiều hôm nay)                             | 973 |    50.98 | [47.8; 54.2] |          0.564 |     0.51  |  0.02  | 0.51  |     2.2627 |   0.481 |         0.493 |
| B5 LSTM trước tune (lookback=30, hidden=32, dropout=0.0, lr=0.001) | 973 |    50.98 | [47.8; 54.2] |          0.564 |     0.506 |  0.014 | 0.511 |     0.6926 |   0.25  |         0.785 |
| B5 LSTM sau tune (lookback=10, hidden=32, dropout=0.2, lr=0.0001)  | 973 |    49.33 | [46.1; 52.5] |          0.7   |     0.492 | -0.016 | 0.51  |     0.6927 |   0.25  |         0.6   |

## Bước B. Walk-forward 82 fold (2020 – 08/2026), trước và sau tune

Cùng lịch `pipeline` (train 730, gap 1, test 30, bước 30); tham số giữ cố định theo Bước A ở mọi fold.

|                    |    n |   DA (%) | CI 95% DA    |   p (DA = 50%) |   bal_acc |    MCC |   AUC |   log_loss |   brier |   pct_pred_up |
|:-------------------|-----:|---------:|:-------------|---------------:|----------:|-------:|------:|-----------:|--------:|--------------:|
| Majority           | 2434 |    50.08 | [48.1; 52.1] |          0.952 |     0.498 | -0.003 | 0.501 |     0.6938 |   0.25  |         0.63  |
| Đảo chiều          | 2434 |    53.25 | [51.2; 55.2] |          0.001 |     0.533 |  0.065 | 0.533 |     2.1585 |   0.458 |         0.494 |
| B5 LSTM trước tune | 2434 |    50    | [48.0; 52.0] |          1     |     0.498 | -0.004 | 0.503 |     0.6947 |   0.251 |         0.596 |
| B5 LSTM sau tune   | 2434 |    50.41 | [48.4; 52.4] |          0.7   |     0.503 |  0.006 | 0.507 |     0.6933 |   0.25  |         0.55  |

Số epoch early stopping chọn (trung bình qua fold × seed): B5 LSTM trước tune: 4.1 (min 1, max 26); B5 LSTM sau tune: 17.1 (min 1, max 50).

### Theo năm (DA %, walk-forward)

|   date |   số ngày |   Đảo chiều |   B5 LSTM trước tune |   B5 LSTM sau tune |
|-------:|----------:|------------:|---------------------:|-------------------:|
|   2020 |       366 |        58.5 |                 56.3 |               57.1 |
|   2021 |       365 |        56.4 |                 48.5 |               49.6 |
|   2022 |       365 |        51   |                 49.3 |               49.9 |
|   2023 |       365 |        54.8 |                 49   |               49.6 |
|   2024 |       366 |        54.1 |                 50.5 |               50.8 |
|   2025 |       365 |        51   |                 47.7 |               46   |
|   2026 |       242 |        43.8 |                 47.9 |               49.6 |

Cột Đảo chiều ở đây là luật B0 của walk-forward: chiều được chọn lại trên từng cửa sổ train (một số fold chọn persistence), nên năm 2026 khác con số 46,3% của bảng EDA trong `baseline_check` (luật đảo chiều thuần). DA gộp 53,25% thì khớp.
