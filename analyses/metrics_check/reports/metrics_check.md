# Kiểm tra metric (góp ý 2 của thầy)

Tạo bởi `analyses/metrics_check/src/metrics_check.py`. Kiểm trên file dự báo chính thức của `pipeline` và code tạo bảng `run_all.metrics`. Mọi phép kiểm là `assert`; báo cáo này chỉ được ghi khi tất cả đều đúng.

## Tóm tắt

| # | Câu hỏi | Kết quả |
|---|---|---|
| 1 | Nhãn trong file dự báo là hướng giá ngày mai? | Đúng ở mọi dòng của 18 file, 2434 ngày; cùng ngày test giữa các file |
| 2 | Code tạo bảng tính đúng định nghĩa? | Đúng: tính tay trên 8 ngày khớp sklearn; trên đủ 2.434 ngày khớp run_all.metrics và e1_delta_draft.csv |
| 3 | Δ có đúng dấu, đúng cách tính? | Đúng: tính lại từ hai file khớp bảng; ΔDA = (số ngày chỉ +On-chain đúng − số ngày chỉ OHLCV đúng)/N |
| 4 | Ô nào không nên đọc? | Log-loss, Brier của luật (log-loss của `b0pers` từng in 19,28) và AUC của Majority: đã đổi thành ô trống |
| 5 | Backtest có lệch ngày? | Không: p̂_t → vị thế t+1 → return t+1, và dấu return t+1 khớp y_t ở mọi ngày |
| 6 | Các bảng có nhất quán? | Có: theo năm, so với luật đảo chiều và bảng Δ khớp nhau; hai code độc lập cho cùng DA của 3 luật |

**Cách đọc bảng kết quả từ nay:** DA và log-loss là hai metric chính, luôn in kèm mốc (DA 50%, luôn đoán tăng 50,90%, luật đảo chiều 53,25%; log-loss 0,6931). Δ log-loss âm mới là On-chain giúp. Không đọc log-loss, Brier của luật và AUC của Majority.

## Phần 1. Nhãn trong file dự báo có đúng là hướng giá ngày mai?

Kiểm **mọi dòng của 18 file** trong `pipeline/results/predictions/`: cùng 2434 ngày test (2020-01-01 → 2026-08-30), và `y_true` của dòng ngày t bằng đúng 1[close_{t+1} > close_t] tính lại từ `master_daily.csv`. Không có ngày nào giá đóng cửa bằng nhau. Ví dụ:

| date       |   close_t |   close_t+1 |   y_true trong file | tăng?   |
|:-----------|----------:|------------:|--------------------:|:--------|
| 2020-03-11 |   7934.52 |      4800   |                   0 | giảm    |
| 2020-03-12 |   4800    |      5578.6 |                   1 | tăng    |
| 2021-11-09 |  66947.7  |     64882.4 |                   0 | giảm    |
| 2021-11-10 |  64882.4  |     64774.3 |                   0 | giảm    |

Đọc dòng 2020-03-11: dòng này dự báo cho ngày 12/03 (ngày sập), giá đi từ 7.934,52 xuống 4.800, nên y = 0. Một mô hình dự báo đúng cú sập phải có p̂ < 0,5 ở dòng **11/03**, không phải dòng 12/03.

## Phần 2. Định nghĩa và cách đọc từng metric

Ngày test có y = 1 (giá ngày mai tăng) chiếm 50.90%. Mọi metric dùng ngưỡng 0,5 (`run_all.metrics`): p̂ > 0,5 là đoán tăng; p̂ đúng bằng 0,5 tính là đoán giảm.

| metric        | công thức                                                  | đo gì trong bài toán hướng giá                                | mốc đoán bừa                       | tốt khi   | Δ > 0 nghĩa là      |
|:--------------|:-----------------------------------------------------------|:--------------------------------------------------------------|:-----------------------------------|:----------|:--------------------|
| DA (accuracy) | tỷ lệ ngày 1[p̂ > 0,5] = y                                  | đoán đúng hướng bao nhiêu % số ngày                           | ≈ 50%; luôn đoán tăng: 50.90%      | cao       | On-chain giúp       |
| Log-loss      | trung bình −log p̂(nhãn thật)                               | xác suất có đáng tin không; phạt nặng khi tự tin mà sai       | p̂ = 0,5: 0.6931; p̂ = 0.509: 0.6930 | thấp      | On-chain làm XẤU đi |
| Brier         | trung bình (p̂ − y)²                                        | như log-loss nhưng phạt nhẹ hơn khi sai tự tin                | p̂ = 0,5: 0,2500                    | thấp      | On-chain làm XẤU đi |
| AUC           | tỷ lệ cặp (ngày tăng, ngày giảm) mà ngày tăng có p̂ cao hơn | xếp hạng ngày tăng trên ngày giảm, không phụ thuộc ngưỡng 0,5 | 0,5                                | cao       | On-chain giúp       |

**Hai điểm dễ đọc nhầm:**

- DA và AUC: càng cao càng tốt, Δ dương là On-chain giúp. Log-loss và Brier: càng thấp càng tốt, **Δ âm** mới là On-chain giúp. Bảng `e1_delta_draft.csv` để chung các metric trong một cột `delta`, nên phải đọc dấu theo từng dòng.
- DA chỉ xem p̂ nằm phía nào của 0,5; log-loss xem p̂ cách 0,5 bao xa. Một mô hình có thể DA trên 50% nhưng log-loss tệ hơn đoán 50/50 (Phần 3).

## Phần 3. Tính tay trên ngày thật

File `lr_ohlcv.csv` (B1 Logistic Regression, chỉ OHLCV). 8 ngày: 2 ngày cho mỗi tổ hợp (giảm/tăng) × (sai/đúng).

| date       |   y |   p_hat |   dự báo (p̂ > 0,5) |   đúng |   −log p(y) |   (p̂ − y)² |
|:-----------|----:|--------:|-------------------:|-------:|------------:|-----------:|
| 2020-01-01 |   0 |  0.4889 |                  0 |      1 |      0.6711 |     0.239  |
| 2020-01-02 |   1 |  0.5211 |                  1 |      1 |      0.6519 |     0.2294 |
| 2020-01-03 |   1 |  0.4766 |                  0 |      0 |      0.7411 |     0.274  |
| 2020-01-04 |   1 |  0.5226 |                  1 |      1 |      0.649  |     0.2279 |
| 2020-01-06 |   1 |  0.484  |                  0 |      0 |      0.7258 |     0.2663 |
| 2020-01-07 |   0 |  0.4978 |                  0 |      1 |      0.6887 |     0.2478 |
| 2020-01-08 |   0 |  0.5554 |                  1 |      0 |      0.8105 |     0.3084 |
| 2020-01-15 |   0 |  0.5511 |                  1 |      0 |      0.801  |     0.3037 |

Trên 8 ngày này: DA = 0.5000; log-loss = 0.7174; Brier = 0.2621; AUC = 0.2500 (4 trong 16 cặp (ngày tăng, ngày giảm) có ngày tăng được p̂ cao hơn). sklearn cho cùng 4 con số.

Trên đủ 2434 ngày, công thức tay, hàm `run_all.metrics` và bảng `e1_delta_draft.csv` khớp nhau (lệch ≤ 1e-12):

| metric   |   tính tay (công thức) |   run_all.metrics |   e1_delta_draft.csv |
|:---------|-----------------------:|------------------:|---------------------:|
| da       |               0.511093 |          0.511093 |             0.511093 |
| log_loss |               0.695255 |          0.695255 |             0.695255 |
| brier    |               0.25106  |          0.25106  |             0.25106  |
| auc      |               0.509827 |          0.509827 |             0.509827 |

**Ví dụ cho điểm dễ nhầm ở Phần 2:** B1 có DA 51.11% (trên 50%) nhưng log-loss 0.6953 > 0.6931: xác suất của B1 tệ hơn việc luôn nói 50/50.

## Phần 4. Δ được tính đúng không?

Δ = metric(OHLCV + On-chain) − metric(chỉ OHLCV), cùng mô hình, cùng 2.434 ngày. Tính lại từ hai file dự báo khớp cột `delta` của `e1_delta_draft.csv` cho mọi mô hình và metric. ΔDA còn tách được thành số ngày chỉ một bên đúng: ΔDA = (chỉ +On-chain đúng − chỉ OHLCV đúng) / N. Đây cũng là hai số mà kiểm định McNemar dùng.

| model   |   ΔDA (điểm %) |   chỉ OHLCV đúng |   chỉ +On-chain đúng |   (b − a)/N × 100 |   Δ log-loss |
|:--------|---------------:|-----------------:|---------------------:|------------------:|-------------:|
| b0maj   |         0      |                0 |                    0 |            0      |       0      |
| b0pers  |         0      |                0 |                    0 |            0      |     nan      |
| b0rev   |         0      |                0 |                    0 |            0      |     nan      |
| b2      |         0.1643 |              309 |                  313 |            0.1643 |      -0.0005 |
| b3      |        -0.3287 |              255 |                  247 |           -0.3287 |      -0.005  |
| b4      |        -1.479  |              308 |                  272 |           -1.479  |      -0.0015 |
| b5      |        -0.1643 |              431 |                  427 |           -0.1643 |       0.0009 |
| lr      |         0.7806 |              327 |                  346 |            0.7806 |       0.0054 |
| lrfs    |         0.4519 |              297 |                  308 |            0.4519 |       0.0036 |

Đọc dòng `lr`: thêm On-chain làm đúng thêm ở một số ngày nhưng sai ở gần chừng ấy ngày khác; ΔDA chỉ là hiệu của hai con số lớn gần bằng nhau, nên dễ đổi dấu khi đổi mô hình hoặc giai đoạn.

## Phần 5. Ô nào trong bảng không có nghĩa

**Log-loss và Brier của luật** (`b0pers`, `b0rev`). Luật cho p̂ chỉ là 0 hoặc 1. Khi đó:
- log-loss = (tỷ lệ ngày sai) × (−log eps) = 0.4675 × 36.04 = 16.85. Con số chỉ đo tỷ lệ ngày sai nhân một hằng số do sklearn chọn; trước khi sửa, bảng in 19,28 cho `b0pers` (và 16,85 cho `b0rev` khi mới thêm), dễ bị đọc như mô hình cực tệ.
- Brier = tỷ lệ ngày sai = 0.4675, tức 1 − DA, không thêm thông tin.
- AUC của luật bằng đúng balanced accuracy (0.5326), vẫn đọc được.

**AUC của `b0maj`.** p̂ của Majority không đổi trong mỗi fold (1 giá trị/fold), chỉ khác giữa các fold. AUC khi đó xếp hạng các fold theo tỷ lệ tăng của cửa sổ train, không phải xếp hạng các ngày.

**Đã sửa** `run_all.metrics`: các ô này ghi NaN (ô trống) thay vì một con số. Bảng hiện tại:

| model   | metric   |    ohlcv |
|:--------|:---------|---------:|
| b0maj   | da       |   0.5008 |
| b0maj   | log_loss |   0.6938 |
| b0maj   | brier    |   0.2503 |
| b0maj   | auc      | nan      |
| b0pers  | da       |   0.4651 |
| b0pers  | log_loss | nan      |
| b0pers  | brier    | nan      |
| b0pers  | auc      |   0.4649 |
| b0rev   | da       |   0.5325 |
| b0rev   | log_loss | nan      |
| b0rev   | brier    | nan      |
| b0rev   | auc      |   0.5326 |

## Phần 6. Backtest dùng p̂ đúng ngày không

`features/src/backtest.py`: p̂ ở dòng ngày t quyết định vị thế của ngày t+1, lời lỗ là return của ngày t+1. Kiểm trên đủ giai đoạn test: (a) vị thế ngày t+1 = 1 đúng khi p̂_t > 0,5; (b) return ngày t+1 > 0 đúng khi y_t = 1. Nên backtest đo đúng cùng bài toán với DA, chỉ khác là mỗi ngày có trọng số bằng độ lớn return.

| dòng dự báo t   |    p̂_t |   y_t | ngày giữ vị thế   |   vị thế t+1 |   return t+1 (%) |
|:----------------|-------:|------:|:------------------|-------------:|-----------------:|
| 2020-01-01      | 0.4889 |     0 | 2020-01-02        |            0 |          -3.2654 |
| 2020-01-02      | 0.5211 |     1 | 2020-01-03        |            1 |           5.4445 |
| 2020-01-03      | 0.4766 |     1 | 2020-01-04        |            0 |           0.1246 |
| 2020-01-04      | 0.5226 |     1 | 2020-01-05        |            1 |           0.0631 |

## Phần 7. Các bảng có cho cùng con số không

- `e1_by_year.csv`: với mọi mô hình và cấu hình, số ngày các năm cộng lại đủ 2.434 và DA gộp theo trọng số số ngày bằng đúng DA trong `e1_delta_draft.csv`.
- `e1_vs_reversal.csv`: DA của mô hình và của luật đảo chiều khớp `run_all.metrics`.
- Hai code viết độc lập (`pipeline` và `baseline_check`) cho cùng DA của ba luật:

| luật        |   pipeline |   baseline_check (code riêng) |
|:------------|----------------:|------------------------------:|
| Majority    |           50.08 |                         50.08 |
| Persistence |           46.51 |                         46.51 |
| Đảo chiều   |           53.25 |                         53.25 |
