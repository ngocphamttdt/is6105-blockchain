# Báo cáo tái tạo Hafid et al. 2024 (arXiv:2410.06935)

Mọi số kết quả trong báo cáo này được sinh tự động từ `reports/results.csv` bởi `src/run.py`; số nến thiếu lấy từ `data/manifest.json`.

## 1. Đã chạy gì

- **A**: BTCUSDT nến 15 phút (2021-02-01 → 2022-02-01 UTC), nhãn của bài: y_t = 1 nếu SMA10_t ≥ SMA60_t (SMA của Close, gồm cả nến t).
- **B**: cùng dữ liệu 15 phút, nhãn dự báo thật: y_t = 1 nếu C_{t+1} > C_t.
- **C**: OHLCV ngày của nhóm (`data/processed/master_daily.csv`, chỉ 5 cột OHLCV, 2018-01-01 → 2026-08-30), cùng 17 đặc trưng tính trên nến ngày, nhãn ngày mai 1[C_{t+1} > C_t].
- Mỗi thí nghiệm chạy với 2 chế độ chuẩn hoá (`full`: scaler và chi2 fit trên toàn bộ dòng như bài; `train`: chỉ fit trên 80% đầu) và 2 mô hình (XGBoost, LR) = 12 cấu hình. Chia 80/20 theo thời gian, không xáo trộn; mô hình luôn fit trên train, đánh giá trên test.
- LR (max_iter = 100, giá trị của bài) hội tụ trong giới hạn này ở cả 6 cấu hình: n_iter_ từ 20 đến 79 (< 100); không có ConvergenceWarning trên dữ liệu thật. n_iter_ theo cấu hình: A/full = 79, A/train = 68, B/full = 20, B/train = 20, C/full = 20, C/train = 20.
- Majority baseline là tỷ lệ lớp nhiều nhất trong tập TEST (hơi có lợi cho baseline so với dùng tỷ lệ của train).

## 2. Bảng kết quả

| TN | Chuẩn hoá | Mô hình | Accuracy | Precision | Recall | F1 | AUC | Majority baseline | n_test | 8 đặc trưng được chọn |
|---|---|---|---|---|---|---|---|---|---|---|
| A | full | XGBoost | 0,9313 | 0,9060 | 0,9481 | 0,9266 | 0,9845 | 0,5426 | 6955 | RSI14 RSI30 MOM30 MACD %K30 %K200 %D30 %D200 |
| A | full | LR | 0,9261 | 0,9027 | 0,9396 | 0,9208 | 0,9810 | 0,5426 | 6955 | RSI14 RSI30 MOM30 MACD %K30 %K200 %D30 %D200 |
| A | train | XGBoost | 0,9310 | 0,9059 | 0,9475 | 0,9262 | 0,9845 | 0,5426 | 6955 | RSI14 RSI30 MOM30 MACD %K30 %K200 %D30 %D200 |
| A | train | LR | 0,9260 | 0,9020 | 0,9403 | 0,9207 | 0,9811 | 0,5426 | 6955 | RSI14 RSI30 MOM30 MACD %K30 %K200 %D30 %D200 |
| B | full | XGBoost | 0,5168 | 0,5038 | 0,5667 | 0,5334 | 0,5257 | 0,5126 | 6955 | Close RSI14 RSI30 %K10 %K30 %K200 %D30 %D200 |
| B | full | LR | 0,5410 | 0,5264 | 0,5820 | 0,5528 | 0,5580 | 0,5126 | 6955 | Close RSI14 RSI30 %K10 %K30 %K200 %D30 %D200 |
| B | train | XGBoost | 0,5168 | 0,5038 | 0,5667 | 0,5334 | 0,5257 | 0,5126 | 6955 | Close RSI14 RSI30 %K10 %K30 %K200 %D30 %D200 |
| B | train | LR | 0,5410 | 0,5264 | 0,5820 | 0,5528 | 0,5580 | 0,5126 | 6955 | Close RSI14 RSI30 %K10 %K30 %K200 %D30 %D200 |
| C | full | XGBoost | 0,4857 | 0,4840 | 0,6177 | 0,5427 | 0,4846 | 0,5059 | 593 | Close RSI200 EMA10 EMA30 EMA200 %K10 %K30 %D200 |
| C | full | LR | 0,4992 | 0,4959 | 0,8328 | 0,6217 | 0,5167 | 0,5059 | 593 | Close RSI200 EMA10 EMA30 EMA200 %K10 %K30 %D200 |
| C | train | XGBoost | 0,4857 | 0,4840 | 0,6177 | 0,5427 | 0,4846 | 0,5059 | 593 | Close RSI200 EMA10 EMA30 EMA200 %K10 %K30 %D200 |
| C | train | LR | 0,5008 | 0,4969 | 0,8328 | 0,6224 | 0,5167 | 0,5059 | 593 | Close RSI200 EMA10 EMA30 EMA200 %K10 %K30 %D200 |

## 3. Giai đoạn train/test (sau khi bỏ dòng warm-up)

| TN | n_train | Train đầu | Train cuối | n_test | Test đầu | Test cuối |
|---|---|---|---|---|---|---|
| A | 27819 | 2021-02-03 02:15:00 | 2021-11-20 13:00:00 | 6955 | 2021-11-20 13:15:00 | 2022-01-31 23:45:00 |
| B | 27818 | 2021-02-03 02:15:00 | 2021-11-20 12:45:00 | 6955 | 2021-11-20 13:00:00 | 2022-01-31 23:30:00 |
| C | 2369 | 2018-07-21 00:00:00 | 2025-01-13 00:00:00 | 593 | 2025-01-14 00:00:00 | 2026-08-29 00:00:00 |

## 4. So sánh A với số của bài

| Mô hình | Bài báo | A (full) | Hiệu (full − bài) | A (train) | Hiệu (train − bài) |
|---|---|---|---|---|---|
| XGBoost | 0,9240 | 0,9313 | +0,0073 | 0,9310 | +0,0070 |
| LR | 0,9101 | 0,9261 | +0,0160 | 0,9260 | +0,0159 |

Các tham số không được chỉnh để khớp số của bài. Bài không công bố các dòng dữ liệu chính xác, ranh giới chia train/test và mã nguồn, nên không thể xác nhận khớp chính xác.

## 5. Nhận xét chính

Chênh lệch accuracy so với majority baseline (chuẩn hoá train-only). SE = sqrt(0,25/n_test); gap = accuracy − majority baseline; z = gap/SE.

| TN | Mô hình | n_test | Accuracy | Baseline | gap | SE | gap/SE | AUC | Trong ±2 SE? |
|---|---|---|---|---|---|---|---|---|---|
| A | XGBoost | 6955 | 0,9310 | 0,5426 | +0,3884 | 0,0060 | +64,8 | 0,9845 | không |
| A | LR | 6955 | 0,9260 | 0,5426 | +0,3833 | 0,0060 | +63,9 | 0,9811 | không |
| B | XGBoost | 6955 | 0,5168 | 0,5126 | +0,0042 | 0,0060 | +0,7 | 0,5257 | có |
| B | LR | 6955 | 0,5410 | 0,5126 | +0,0285 | 0,0060 | +4,7 | 0,5580 | không |
| C | XGBoost | 593 | 0,4857 | 0,5059 | −0,0202 | 0,0205 | −1,0 | 0,4846 | có |
| C | LR | 593 | 0,5008 | 0,5059 | −0,0051 | 0,0205 | −0,2 | 0,5167 | có |

SE giả định các quan sát độc lập. Với A, nhãn bền vững theo thời gian (SMA10 so với SMA60 đổi dấu hiếm) nên giả định này sai nặng và SE của A rất lạc quan. Với B, nhãn là hướng của một nến, các nhãn không chồng lấn nên giả định độc lập chỉ vi phạm nhẹ (tự tương quan còn lại của lợi suất 15 phút).

**B.** LR vượt baseline +0,0285 (+4,7 SE, AUC 0,5580), còn XGBoost trên cùng 8 đặc trưng chỉ +0,0042 (+0,7 SE), nằm trong nhiễu. Lưu ý cho B: một lần chia duy nhất; một cửa sổ test duy nhất từ 2021-11-20 13:00:00 đến 2022-01-31 23:30:00; so sánh 4 cấu hình dự báo (2 mô hình × 2 chế độ chuẩn hoá) nên có vấn đề so sánh nhiều lần; không có chi phí giao dịch hay phân tích lợi suất (accuracy không bằng lợi nhuận).

**C.** n_test = 593, SE = 0,0205; hiệu ứng nhỏ nhất phát hiện được với power 80% (hai phía, 5%) ≈ 2,8·SE = 0,0575. Vì vậy kết quả của C không có thông tin: không phải bằng chứng là không có lợi thế dự báo.

**A.** Nhãn của A là hàm tất định của Close đến hết nến t (SMA10 so với SMA60), và các đặc trưng được chọn (RSI14, RSI30, MOM30, MACD, %K30, %K200, %D30, %D200) là các đại lượng đà giá tính từ cùng chuỗi Close; A không phải dự báo giá tương lai.

**Chế độ `full` và `train` cho kết quả gần như trùng nhau.** MinMax sau StandardScaler triệt tiêu bước Standard, nên đầu vào hiệu dụng là (x − min_fit)/(max_fit − min_fit); `scale_on` chỉ có tác dụng khi min/max của một cột được chọn rơi vào giai đoạn test. XGBoost bất biến với phép biến đổi affine tăng theo từng cột, nên XGBoost giống hệt nhau giữa hai chế độ ở B và C. Ở B, cực trị của các cột được chọn nằm trong train, trừ %D30 (thay đổi rất nhỏ) mà LR-L1 gán trọng số gần 0, nên dự đoán không đổi. Ở C, LR khác nhau vì Close và các EMA lập đỉnh mới trong 2025–26 (giai đoạn test). Chế độ `full` là GIẢ ĐỊNH của nhóm về bài: PDF chỉ nói "We scale the data by using the StandardScaler" và chia 80% train / 20% test, không nói scaler được fit trên những dòng nào.

## 6. Lựa chọn của nhóm khi bài không nêu

- RSI kiểu Wilder (EWM alpha = 1/n)
- MACD = EMA12 − EMA26
- %D = SMA 3 của %K cùng cửa sổ
- MOM_n = C_t − C_{t−n}
- PROC_9 = 100·(C_t − C_{t−9})/C_{t−9}
- chi2 chạy trên dữ liệu đã chuẩn hoá Standard rồi MinMax về [0, 1] (chi2 cần giá trị không âm)
- RSI và EMA khởi tạo bằng quan sát đầu tiên (pandas ewm, adjust=False), không phải SMA; với RSI200 giá trị khởi tạo vẫn còn ảnh hưởng lên các dòng train đầu
- bỏ các dòng warm-up có NaN; chọn 8 đặc trưng bằng SelectKBest(chi2, k=8); thêm %D30, %D200 vì Eq. 6 của bài chọn chúng
- XGBoost: n_estimators 400, learning_rate 0,1, max_depth 4, min_child_weight 3, subsample 0,8, colsample_bytree 1,0, gamma 0,1, reg_alpha 0,5, reg_lambda 1, random_state 0; LR: penalty l1, C 0,1, solver saga, max_iter 100, random_state 0

## 7. Dữ liệu thiếu

Dữ liệu 15 phút có 34975 nến; so với 365 × 96 = 35040 nến, thiếu 65 nến (không nội suy). Features dùng cửa sổ theo số dòng nên các chỗ thiếu làm cửa sổ trải dài hơn thời gian danh nghĩa.
