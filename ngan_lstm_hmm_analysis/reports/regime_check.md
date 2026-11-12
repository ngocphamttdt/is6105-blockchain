# Regime: tái lập và ý nghĩa (góp ý 6)

Tạo bởi `ngan_lstm_hmm_analysis/run_regime.py` trên data freeze v1.0. Giai đoạn: 2.435 ngày test của walk-forward.

## 1. Tái lập regime_map.csv

Chạy lại `regime_hmm.run_hmm_walk_forward` (code của Ngân, không sửa) với đầu vào dựng từ data freeze: `log_return` = log return giá đóng cửa ngày, `realized_volatility` = căn tổng bình phương log return 1h trong ngày UTC. Kết quả khớp **100%** nhãn và fold của `regime_map.csv` trên 2,435 ngày. Thử các cách tính khác của realized_volatility (độ lệch chuẩn 7/14/30 ngày, |return|, RV 7 ngày) chỉ khớp 40–48%, nên đây đúng là cách Ngân đã dùng.

## 2. Ý nghĩa Bull / Sideways / Bear

**HMM (2 đặc trưng: return và biến động ngày):**

| regime   |   n_days |   n_runs |   mean_run_days |   share_runs_1_day_% |   std_daily_ret_% |   ret_30d_before_% |   ret_30d_after_% |
|:---------|---------:|---------:|----------------:|---------------------:|------------------:|-------------------:|------------------:|
| Bull     |     1250 |      417 |            3    |                39.81 |              2.94 |               5.39 |              4.19 |
| Sideways |      903 |      318 |            2.84 |                39.62 |              1.61 |               4.77 |              4.16 |
| Bear     |      282 |      149 |            1.89 |                55.03 |              6.5  |               0    |              6.14 |

**Theo xu hướng** (close so với MA200, return 90 ngày ±10%, làm mượt 21 ngày):

| regime   |   n_days |   n_runs |   mean_run_days |   share_runs_1_day_% |   std_daily_ret_% |   ret_30d_before_% |   ret_30d_after_% |
|:---------|---------:|---------:|----------------:|---------------------:|------------------:|-------------------:|------------------:|
| Bull     |     1032 |        8 |          129    |                    0 |              3.54 |              11.97 |              7.65 |
| Sideways |      632 |       20 |           31.6  |                    5 |              2.55 |               1.96 |             -1.71 |
| Bear     |      771 |       13 |           59.31 |                    0 |              3.25 |              -3.3  |              5.07 |

HMM tách theo **biến động**: Bear là ngày biến động mạnh (độ lệch chuẩn return cao nhất), đổi nhãn sau vài ngày, và return 30 ngày **sau** ngày Bear lại dương. Regime theo xu hướng có đợt dài hàng chục ngày và return 30 ngày **trước** đúng dấu (Bull dương, Bear âm), tức đúng nghĩa thị trường tăng/giảm.

Bảng chéo (% theo hàng HMM):

| regime   |   Bull |   Sideways |   Bear |
|:---------|-------:|-----------:|-------:|
| Bull     |     44 |         23 |     33 |
| Sideways |     40 |         30 |     30 |
| Bear     |     41 |         25 |     34 |

## 3. Kiểm tra trên các giai đoạn đã biết

Mỗi giai đoạn: % số ngày được gán đúng nhãn kỳ vọng, và nhãn chiếm nhiều nhất.

| giai đoạn             | từ         | đến        | kỳ vọng   |   HMM: % đúng kỳ vọng | HMM: nhãn chiếm nhiều nhất   |   Xu hướng: % đúng kỳ vọng | Xu hướng: nhãn chiếm nhiều nhất   |
|:----------------------|:-----------|:-----------|:----------|----------------------:|:-----------------------------|---------------------------:|:----------------------------------|
| Sập COVID             | 2020-03-01 | 2020-03-31 | Bear      |                    68 | Bear                         |                         42 | Bull                              |
| Tăng mạnh             | 2020-10-01 | 2021-04-14 | Bull      |                    68 | Bull                         |                        100 | Bull                              |
| Điều chỉnh 05–07/2021 | 2021-05-15 | 2021-07-20 | Bear      |                    21 | Bull                         |                         81 | Bear                              |
| Đỉnh 11/2021          | 2021-10-15 | 2021-11-15 | Bull      |                    69 | Bull                         |                        100 | Bull                              |
| LUNA, giảm 2022       | 2022-05-01 | 2022-07-31 | Bear      |                    26 | Bull                         |                         85 | Bear                              |
| FTX                   | 2022-11-01 | 2022-12-31 | Bear      |                     5 | Sideways                     |                        100 | Bear                              |
| Hồi phục 2023         | 2023-01-15 | 2023-04-15 | Bull      |                    52 | Bull                         |                         84 | Bull                              |
| Trước halving 2024    | 2024-02-01 | 2024-03-31 | Bull      |                    53 | Bull                         |                        100 | Bull                              |
| Đi ngang giữa 2024    | 2024-05-01 | 2024-09-30 | Sideways  |                    29 | Bull                         |                         53 | Sideways                          |

Trung bình % đúng kỳ vọng trên 9 giai đoạn: HMM 43%, theo xu hướng 83%.

**Hạn chế của regime theo xu hướng:** MA200 và return 90 ngày phản ứng chậm, nên cú sập đột ngột (03/2020) vẫn bị gán Bull trong những ngày đầu. Regime này phù hợp để hỏi "On-chain có giúp khác nhau giữa thị trường tăng, giảm, đi ngang không" (RQ3), không phải để bắt điểm đảo chiều.
