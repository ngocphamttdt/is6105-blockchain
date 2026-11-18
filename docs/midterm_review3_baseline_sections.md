# Bản nháp mục 5 và mục 8 (báo cáo giữa kỳ)

Sinh ngày 13/11/2026 bởi `python analyses/simple_split/src/report_sections.py` từ kết quả trong `analyses/simple_split/reports/`. Chạy lại lệnh này sau mỗi lần chạy `simple_split.py`. Mọi con số lấy từ file kết quả; không có số gõ tay.

## 5. Baselines

Mọi baseline chỉ dùng 10 đặc trưng OHLCV (mục 4.1), cùng dữ liệu data freeze v1.0 và cùng một cách chia theo thời gian. Ngày cuối của train và validation bị bỏ (gap 1 ngày) vì nhãn của ngày đó dùng giá của đoạn sau.

| Đoạn | Từ | Đến | Số ngày | Dùng để |
|---|---|---|---|---|
| train | 2018-01-01 | 2021-12-30 | 1460 | học |
| val | 2022-01-01 | 2023-12-30 | 729 | chọn tham số |
| test | 2024-01-01 | 2026-08-30 | 973 | chấm 1 lần |

Bảng 5.1. Cách chia train / validation / test

| ID | Mô hình | Trước tune | Grid sau tune | Số tổ hợp | Seed |
|---|---|---|---|---|---|
| B0 | Majority; Persistence; Buy-and-hold; Đảo chiều | không có tham số | – | – | – |
| B1 | Logistic Regression | set = ohlcv10; C = 1; l1_ratio = 0; class_weight = None; clip = False | set ∈ {sign; lag1; lag1+sign; ohlcv10; ohlcv10+sign} × C ∈ {0,0001; 0,0003; 0,001; 0,003; 0,01; 0,03; 0,1; 0,3; 1; 10} × l1_ratio ∈ {0; 0,5; 1} × class_weight ∈ {None; balanced} | 300 | không (tất định) |
| B2 | Random Forest | mặc định sklearn; n_estimators = 500 | min_samples_leaf ∈ {50; 100; 150; 200} × max_features ∈ {sqrt; 0,33; 0,5} × max_samples ∈ {None; 0,5} | 24 | 42, 43, 44 (trung bình p̂) |
| B3 | SVM kernel RBF | C = 1; gamma = scale | C ∈ {0,01; 0,1; 1; 10; 100} × gamma ∈ {0,001; 0,003; 0,01; 0,03; 0,1; 0,3} | 30 | không (tất định) |
| B5 | LSTM 1 lớp | lookback = 30; hidden = 32; dropout = 0; lr = 0,001 | lookback ∈ {10; 30} × hidden ∈ {8; 32} × dropout ∈ {0; 0,2} × lr ∈ {0,001; 0,0001} | 16 | 42, 43, 44 (trung bình p̂) |

Bảng 5.2. Baseline, tham số trước tune và grid sau tune

- B0: Majority đoán theo tỷ lệ ngày tăng của train + validation; Persistence: mai cùng hướng hôm nay; Buy-and-hold: luôn đoán tăng; Đảo chiều: mai ngược hướng hôm nay (chiều luật chọn trên train + validation).
- LR: tập đặc trưng `sign` = 1[return hôm nay > 0], `lag1` = return hôm nay, `ohlcv10` = 10 đặc trưng; bản sau tune có winsorize 1%/99% trước StandardScaler. SVM: xác suất hiệu chỉnh sigmoid. RF: 500 cây, max_depth không giới hạn. LSTM: early stopping (patience 5) trên 20% cuối phần đang học, có gap 1 ngày, rồi học lại trên toàn bộ phần đang học.
- Chọn tham số: log-loss nhỏ nhất trên validation (làm tròn 4 chữ số; hoà thì chọn tổ hợp đứng trước, tức đơn giản hơn). Sau đó học lại trên train + validation với tham số đã chọn và chấm test đúng 1 lần. StandardScaler fit trên phần đang học.
- B4 XGBoost: chưa chạy ở cách chia này.
- Kết quả có On-chain tạm hoãn cho đến khi baseline OHLCV ổn định (góp ý 1, 4).

## 8. Kết quả

Mọi số liệu tính trên đoạn test 2024-01-01 → 2026-08-30 (973 ngày; tỷ lệ ngày tăng 50,7%), sau khi học lại trên train + validation. Tạo bởi `analyses/simple_split/src/simple_split.py`.

### 8.1. So sánh trên test, trước và sau tuning

| Mô hình | DA (%) | CI 95% DA | p (DA = 50%) | Log-loss | % ngày đoán tăng | McNemar p so với Đảo chiều |
|---|---|---|---|---|---|---|
| Majority | 50,67 | [47,5; 53,9] | 0,700 | 0,6931 | 100,0 | 0,928 |
| Buy-and-hold | 50,67 | [47,5; 53,9] | 0,700 | – | 100,0 | 0,928 |
| Persistence | 49,02 | [45,8; 52,2] | 0,564 | – | 50,7 | 0,564 |
| Đảo chiều | 50,98 | [47,8; 54,2] | 0,564 | – | 49,3 | – |
| LR trước tune | 52,62 | [49,4; 55,8] | 0,109 | 0,6911 | 56,9 | 0,307 |
| LR sau tune | 52,11 | [48,9; 55,3] | 0,200 | 0,6923 | 35,0 | 0,406 |
| SVM RBF trước tune | 50,46 | [47,3; 53,6] | 0,798 | 0,6936 | 40,4 | 0,827 |
| SVM RBF sau tune | 51,08 | [47,9; 54,3] | 0,521 | 0,7148 | 23,9 | 1,000 |
| RF trước tune | 51,08 | [47,9; 54,3] | 0,521 | 0,7118 | 54,2 | 1,000 |
| RF sau tune | 50,67 | [47,5; 53,9] | 0,700 | 0,6949 | 57,7 | 0,881 |
| LSTM trước tune | 50,98 | [47,8; 54,2] | 0,564 | 0,6926 | 78,5 | 1,000 |
| LSTM sau tune | 49,13 | [45,9; 52,3] | 0,608 | 0,6927 | 64,3 | 0,358 |

Bảng 8.1. 973 ngày test. Log-loss của Persistence, Buy-and-hold, Đảo chiều không tính vì p̂ chỉ là 0/1. Đoán 50/50 mọi ngày có log-loss 0,6931. Với 973 ngày, sai số chuẩn của DA khoảng 1,6 điểm.

### 8.2. Tuning trên validation (chưa dùng test)

| Mô hình | Tham số được chọn | Val log-loss | Val DA (%) | Số tổ hợp có val log-loss < 0,6931 |
|---|---|---|---|---|
| LR | set = ohlcv10; C = 0,1; l1_ratio = 1; class_weight = balanced | 0,6908 | 52,5 | 75/300 |
| SVM RBF | C = 0,01; gamma = 0,3 | 0,6959 | 48,3 | 0/30 |
| RF | min_samples_leaf = 100; max_features = sqrt; max_samples = 0,5 | 0,6926 | 52,8 | 12/24 |
| LSTM | lookback = 10; hidden = 32; dropout = 0; lr = 0,0001 | 0,6936 | 47,7 | 0/16 |

Bảng 8.2. Bảng đầy đủ mọi tổ hợp: `analyses/simple_split/reports/tuning_<mô hình>.csv`.

### 8.3. Xem trực tiếp dự báo (bản sau tune)

| Mô hình | Thật giảm, đoán giảm | Thật giảm, đoán tăng | Thật tăng, đoán giảm | Thật tăng, đoán tăng | Cùng hướng Đảo chiều (%) | Đúng / 10 ngày biến động mạnh nhất |
|---|---|---|---|---|---|---|
| LR | 323 | 157 | 309 | 184 | 85,1 | 4/10 |
| SVM RBF | 372 | 108 | 368 | 125 | 56,1 | 6/10 |
| RF | 206 | 274 | 206 | 287 | 81,8 | 6/10 |
| LSTM | 166 | 314 | 181 | 312 | 64,9 | 7/10 |

Bảng 8.3. Ma trận nhầm lẫn, mức trùng hướng với luật Đảo chiều, và số ngày đoán đúng trong 10 ngày có return ngày mai lớn nhất (theo trị tuyệt đối). Danh sách từng ngày: `analyses/simple_split/reports/simple_split.md`, mục 3.

### 8.4. Nhận xét

- 12/12 dòng có CI 95% của DA chứa 50%.
- DA cao nhất: LR trước tune 52,62% (p = 0,109); luật Đảo chiều 50,98%.
- DA sau tune thấp hơn trước tune ở 3/4 mô hình: LR 52,62 → 52,11; RF 51,08 → 50,67; LSTM 50,98 → 49,13.
- Trên test, bản sau tune có log-loss thấp hơn 0,6931: LR 0,6923; LSTM 0,6927.
- Trên validation, số tổ hợp có log-loss thấp hơn 0,6931: LR 75/300; SVM RBF 0/30; RF 12/24; LSTM 0/16.
- So với luật Đảo chiều, McNemar p nhỏ nhất là 0,307 (LR trước tune).

## Các chỗ cần sửa ở phần còn lại của báo cáo

- **Mục 1:** Giữ bài toán Δ On-chain và RQ1–RQ3 làm mục tiêu đề tài; thêm một câu: giữa kỳ tập trung hoàn thiện baseline OHLCV theo góp ý của giảng viên, phần On-chain làm sau khi baseline ổn định.
- **Mục 4.2:** Thêm cách chia cố định train 2018–2021 / validation 2022–2023 / test 2024 – 08/2026 dùng cho baseline giữa kỳ (mục 5). Giữ đoạn walk-forward và ghi là protocol cho các thí nghiệm sau.
- **Mục 4.3–4.4:** Regime HMM, kiểm định Δ, Diebold–Mariano, Holm, placebo: chuyển thành kế hoạch (chưa dùng ở giữa kỳ). Giữ McNemar và CI của DA vì mục 8 có dùng.
- **Bảng 4.1:** Dòng MCC: đổi "Chưa triển khai" thành đã có (simple_split và baseline_check có tính MCC). Dòng F1 macro: vẫn chưa có trong simple_split, giữ "chưa triển khai".
- **Mục 7:** E0: (a) xong; (b) xáo nhãn đã chạy trong baseline_check (cho LR); (c) chia cố định đã chạy (mục 8). E1, E2, E3, E4: đổi trạng thái thành "tạm hoãn đến khi baseline OHLCV ổn định".
- **Mục 8:** Thay toàn bộ 8.1–8.4 cũ (2.434 ngày, có On-chain, backtest, regime) bằng mục 8 mới ở trên.
- **Mục 9:** Bỏ câu "B1, B2, B4 hiện là bản chạy thử" và "B5 không tune bằng grid" (đã tune, mục 5). Thêm: test chỉ 973 ngày nên CI của DA rộng; B4 XGBoost chưa chạy ở cách chia này; kết quả mới có một lần chia, chưa đối chiếu walk-forward. Giữ các hạn chế về dữ liệu; câu về Flows và regime chuyển sang phần kế hoạch.
- **Mục 10:** Thay kết luận "On-chain không cải thiện" bằng kết luận rút từ mục 8.4 (chỉ về baseline OHLCV). Hướng tiếp theo: chạy B4, đối chiếu walk-forward, rồi mới thêm On-chain theo từng nhóm.
- **Phụ lục B:** Bỏ (kết quả E3 theo regime).
- **Phụ lục C:** Thay grid cũ bằng Bảng 5.2.
