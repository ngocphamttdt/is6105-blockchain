# Tuning B5 LSTM (góp ý 3 của thầy): chỉ logret_lag1

Tạo bởi `analyses/lstm_tuning/src/lstm_tuning.py`. Đặc trưng: **chỉ logret_lag1**, data freeze v1.0. Dữ liệu, cách chia, metric và luật B0 lấy từ `baseline_check` nên số liệu so được trực tiếp với bảng LR ở đó.

## Thiết lập

- Kiến trúc: LSTM 1 lớp → dropout → Linear, BCE, Adam, batch 32, tối đa 50 epoch, patience 5; p̂ trung bình 3 seed (42, 43, 44).
- Số seed: kết quả chính lấy p̂ trung bình 3 seed để giảm dao động do khởi tạo ngẫu nhiên; độ nhạy theo seed được kiểm riêng bằng `seed_check.py` (10 seed, mỗi seed một mô hình, `seed_check.md`). Seed được cố định trước khi chạy, không chọn seed theo kết quả.
- Mỗi lần học: scaler fit trên ngày train; early stopping trên 20% cuối phần train, **bỏ 1 ngày gap** giữa phần fit và phần early stopping; sau đó học lại từ đầu trên 100% phần train với số epoch tốt nhất. Batch xáo trộn trong phần train.
- Khác `model_lstm.py` (B5 trong `pipeline`): code đó không có gap ở early stopping, không xáo batch, không dropout và không học lại trên 100% train. Vì vậy cột "trước tune" ở đây được chạy lại bằng code này, không lấy con số 50,74% của báo cáo giữa kỳ.
- Grid (chốt trước khi chạy): lookback [10, 30] × hidden [8, 32] × dropout [0.0, 0.2] × lr [0.001, 0.0001] = 16 cấu hình. Trước tune = cấu hình B5 hiện tại (lookback=30, hidden=32, dropout=0.0, lr=0.001).
- Chọn theo log-loss validation (p̂ trung bình các seed), làm tròn 4 chữ số; hoà thì chọn mô hình đơn giản hơn (hidden nhỏ, lookback ngắn).

## Bước A. Grid: train 2018–2021 → validation 2022–2023 (chưa đụng test)

**Cấu hình được chọn: lookback=10, hidden=8, dropout=0.2, lr=0.001.** Mốc: đoán 50/50 mọi ngày có log-loss 0,6931.

|   lookback |   hidden |   dropout |     lr | trước tune   |   train_log_loss |   val_log_loss |   val_log_loss_sd_seed |   train_DA |   val_DA |   val_pct_pred_up | best_epoch_seeds   |
|-----------:|---------:|----------:|-------:|:-------------|-----------------:|---------------:|-----------------------:|-----------:|---------:|------------------:|:-------------------|
|         10 |        8 |       0.2 | 0.001  | False        |           0.6886 |         0.6928 |                 0.0006 |       53.2 |     50.3 |              86.7 | 21/19/2            |
|         10 |       32 |       0.2 | 0.001  | False        |           0.6886 |         0.6928 |                 0.0004 |       53.5 |     50.5 |              85.7 | 8/1/8              |
|         30 |        8 |       0   | 0.001  | False        |           0.6881 |         0.6928 |                 0.0003 |       52.7 |     50.6 |              85.6 | 18/19/2            |
|         30 |        8 |       0.2 | 0.001  | False        |           0.6886 |         0.6928 |                 0.0006 |       53.3 |     50.5 |              86.8 | 21/19/2            |
|         30 |       32 |       0.2 | 0.001  | False        |           0.6887 |         0.6928 |                 0.0004 |       53.5 |     50.3 |              85.9 | 8/1/8              |
|         10 |        8 |       0   | 0.001  | False        |           0.6881 |         0.6929 |                 0.0004 |       52.7 |     50.6 |              85.9 | 18/20/2            |
|         10 |       32 |       0   | 0.001  | False        |           0.6884 |         0.6931 |                 0.0004 |       53.4 |     50.5 |              87.7 | 8/1/6              |
|         10 |       32 |       0   | 0.0001 | False        |           0.6924 |         0.6931 |                 0.0003 |       53.7 |     48.6 |              94.2 | 3/8/1              |
|         10 |       32 |       0.2 | 0.0001 | False        |           0.6924 |         0.6931 |                 0.0003 |       53.5 |     48.4 |              94.1 | 3/9/1              |
|         30 |       32 |       0   | 0.001  | True         |           0.6884 |         0.6931 |                 0.0004 |       53.4 |     50.5 |              87.7 | 8/1/6              |
|         30 |       32 |       0   | 0.0001 | False        |           0.6924 |         0.6932 |                 0.0003 |       53.2 |     48   |              95.1 | 3/8/1              |
|         30 |       32 |       0.2 | 0.0001 | False        |           0.6924 |         0.6932 |                 0.0003 |       53.2 |     48   |              95.1 | 3/9/1              |
|         10 |        8 |       0   | 0.0001 | False        |           0.6911 |         0.6938 |                 0.001  |       52.5 |     48   |             100   | 21/48/19           |
|         30 |        8 |       0   | 0.0001 | False        |           0.6911 |         0.6938 |                 0.001  |       52.5 |     48   |             100   | 21/48/19           |
|         10 |        8 |       0.2 | 0.0001 | False        |           0.6911 |         0.6939 |                 0.0011 |       52.5 |     48   |             100   | 25/48/21           |
|         30 |        8 |       0.2 | 0.0001 | False        |           0.6911 |         0.6939 |                 0.0011 |       52.5 |     48   |             100   | 25/48/21           |

Cột: `val_log_loss_sd_seed` = độ lệch chuẩn val log-loss giữa các seed (cho biết chênh lệch giữa hai cấu hình có lớn hơn dao động do seed không); `best_epoch_seeds` = số epoch early stopping chọn ở từng seed.

## Bước A. Kết quả test 2024 – 08/2026 (chạy 1 lần)

Học lại trên train + validation. Trên train + validation: persistence đúng 45.2% → luật đảo chiều.

|                                                                    |   n |   DA (%) | CI 95% DA    |   p (DA = 50%) |   bal_acc |   MCC |   AUC |   log_loss |   brier |   pct_pred_up |
|:-------------------------------------------------------------------|----:|---------:|:-------------|---------------:|----------:|------:|------:|-----------:|--------:|--------------:|
| B0 Majority (p̂ = tỷ lệ tăng của train)                             | 973 |    50.67 | [47.5; 53.9] |          0.7   |     0.5   | 0     | 0.5   |     0.6931 |   0.25  |         1     |
| B0 Đảo chiều (mai ngược chiều hôm nay)                             | 973 |    50.98 | [47.8; 54.2] |          0.564 |     0.51  | 0.02  | 0.51  |     2.2627 |   0.481 |         0.493 |
| B5 LSTM trước tune (lookback=30, hidden=32, dropout=0.0, lr=0.001) | 973 |    52.11 | [48.9; 55.3] |          0.2   |     0.519 | 0.04  | 0.514 |     0.6926 |   0.25  |         0.64  |
| B5 LSTM sau tune (lookback=10, hidden=8, dropout=0.2, lr=0.001)    | 973 |    50.77 | [47.6; 54.0] |          0.654 |     0.507 | 0.014 | 0.517 |     0.6932 |   0.25  |         0.543 |

## Bước B. Walk-forward 82 fold (2020 – 08/2026), trước và sau tune

Cùng lịch `pipeline` (train 730, gap 1, test 30, bước 30); tham số giữ cố định theo Bước A ở mọi fold.

|                    |    n |   DA (%) | CI 95% DA    |   p (DA = 50%) |   bal_acc |    MCC |   AUC |   log_loss |   brier |   pct_pred_up |
|:-------------------|-----:|---------:|:-------------|---------------:|----------:|-------:|------:|-----------:|--------:|--------------:|
| Majority           | 2434 |    50.08 | [48.1; 52.1] |          0.952 |     0.498 | -0.003 | 0.501 |     0.6938 |   0.25  |         0.63  |
| Đảo chiều          | 2434 |    53.25 | [51.2; 55.2] |          0.001 |     0.533 |  0.065 | 0.533 |     2.1585 |   0.458 |         0.494 |
| B5 LSTM trước tune | 2434 |    50.99 | [49.0; 53.0] |          0.341 |     0.51  |  0.021 | 0.512 |     0.6937 |   0.25  |         0.471 |
| B5 LSTM sau tune   | 2434 |    51.11 | [49.1; 53.1] |          0.283 |     0.511 |  0.021 | 0.513 |     0.6933 |   0.25  |         0.53  |

Số epoch early stopping chọn (trung bình qua fold × seed): B5 LSTM trước tune: 8.4 (min 1, max 48); B5 LSTM sau tune: 12.8 (min 1, max 50).

### Theo năm (DA %, walk-forward)

|   date |   số ngày |   Đảo chiều |   B5 LSTM trước tune |   B5 LSTM sau tune |
|-------:|----------:|------------:|---------------------:|-------------------:|
|   2020 |       366 |        58.5 |                 56.3 |               56   |
|   2021 |       365 |        56.4 |                 50.7 |               52.1 |
|   2022 |       365 |        51   |                 51   |               52.1 |
|   2023 |       365 |        54.8 |                 49.6 |               50.4 |
|   2024 |       366 |        54.1 |                 48.6 |               48.9 |
|   2025 |       365 |        51   |                 51   |               48.5 |
|   2026 |       242 |        43.8 |                 49.2 |               49.2 |

Cột Đảo chiều ở đây là luật B0 của walk-forward: chiều được chọn lại trên từng cửa sổ train (một số fold chọn persistence), nên năm 2026 khác con số 46,3% của bảng EDA trong `baseline_check` (luật đảo chiều thuần). DA gộp 53,25% thì khớp.
