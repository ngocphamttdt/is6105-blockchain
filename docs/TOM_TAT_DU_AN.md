# Tóm tắt dự án C1: Dự báo hướng giá Bitcoin bằng dữ liệu On-chain theo chế độ thị trường

**Môn:** IS6105 Công nghệ Blockchain · **Nhóm 1:** BLOCK&ROLL (Ngọc, Vi, Khoa, Ngân, Trâm) · **GVHD:** Trần Hưng Nghiệp
**Nguồn tóm tắt:** báo cáo giữa kỳ `Blockchain-Group01-C1-OnchainBitcoinForecast-Midterm1.pdf`, `PROJECT_FLOW.md` và các file kết quả trong `ngoc_pipeline/results/`.

---

## 1. Dự án làm gì?

**Vấn đề:** nhiều bài báo nói dữ liệu On-chain giúp đoán hướng giá BTC ngày mai đúng tới 82–83%. Nhưng các bài đó thường:

1. không so với **cùng mô hình nhưng chỉ dùng dữ liệu giá (OHLCV)**,
2. chỉ chia train/test **một lần cố định** (ví dụ 80/20),
3. không xét **trạng thái thị trường** (tăng / giảm / đi ngang).

Vì vậy chưa biết On-chain thật sự **thêm được bao nhiêu** so với giá.

**Bài toán:** với mỗi ngày t, dự báo xác suất giá ngày t+1 tăng:

```
y = 1 nếu Close(t+1) > Close(t), ngược lại y = 0
Mô hình ước lượng p̂ = P(y = 1 | đặc trưng đến ngày t)
```

**Đại lượng chính cần đo không phải "mô hình đúng bao nhiêu %"** mà là **giá trị bổ sung Δ** của On-chain (xem mục 6.3).

| Câu hỏi | Giả thuyết | Thí nghiệm | Kết quả giữa kỳ |
|---|---|---|---|
| RQ1: OHLCV + On-chain có tốt hơn OHLCV-only, có ý nghĩa thống kê không? | H1: Δ > 0 nhưng nhỏ | E0, E1 | **Không chấp nhận H1** |
| RQ2: mỗi nhóm On-chain (Activity, Flows, Mining, Valuation) đóng góp bao nhiêu? Ablation có khớp SHAP không? | H2: đóng góp không đều | E2 | Chưa chạy (tuần 7) |
| RQ3: Δ có khác nhau giữa Bull, Bear, Sideways? | H3: có, Δ ở Bull > Bear | E3 | **Không chấp nhận H3** |
| (E4) RAM-Net: học trọng số mẫu theo regime | Chỉ làm nếu E3 dương tính | E4 | Không thực hiện |

---

## 2. Dữ liệu

### 2.1 Nguồn

| Thành phần | Nguồn | Phạm vi | Ghi chú |
|---|---|---|---|
| Giá BTC/USDT (OHLCV) | Binance Data Vision, nến 1h gộp thành ngày UTC | 17/08/2017 – 31/08/2026 | 30 ngày có dưới 24 nến 1h (bảo trì, sự cố) |
| Giá theo giờ | Binance Data Vision | 78.761 nến | Chỉ dùng tính realized volatility |
| 11 chỉ số On-chain | CoinMetrics Community (CC BY-NC 4.0), Blockchain.com; Puell, NVT, Netflow tự tính | 01/01/2016 – 31/08/2026 | Trước 2018 là giai đoạn "warm-up" cho MA365, z-score 90 ngày |
| Bảng chính `master_daily.csv` | Ghép theo ngày UTC | 3.896 ngày × 24 cột | Giai đoạn nghiên cứu 01/2018 – 08/2026 (3.165 ngày) |

**Data freeze v1.0 (28/10/2026):** mọi thí nghiệm chỉ đọc `data/processed/`. Mỗi lần chạy, code so sha256 với `MANIFEST.json`, lệch là dừng. Data freeze không chuẩn hoá gì; mọi scaler/z-score fit trong cửa sổ train của từng fold.

### 2.2 Cách chọn chỉ số On-chain

Danh sách chốt **trước khi huấn luyện**, chỉ dựa trên related work và khả năng lấy dữ liệu, **không xem tương quan với nhãn** (tránh data snooping). Quy tắc QA cố định:

- bỏ chỉ số thiếu hơn 5% số ngày từ 2018;
- trong cùng nhóm, hai chuỗi (đã sai phân) có |r| > 0,9 thì giữ một → loại `TxTfrCnt` (trùng `TxCnt`), `BlkCnt` (trùng `HashRate`);
- SOPR không có ở bản miễn phí → thay bằng Puell Multiple và NVT.

### 2.3 Mô tả dữ liệu theo nhóm (2018 – 08/2026)

| Nhóm | Chỉ số | Ý nghĩa | Trung vị (min – max) |
|---|---|---|---|
| Activity | `cm_AdrActCnt` | Số địa chỉ hoạt động/ngày | 798 nghìn (415 nghìn – 1,37 triệu) |
| | `cm_TxCnt` | Số giao dịch/ngày | 320 nghìn (124 nghìn – 928 nghìn) |
| | `bc_est_tx_vol_usd` | Giá trị giao dịch ước lượng (USD) | 2,77 tỷ (220 triệu – 58,4 tỷ) |
| Flows | `cm_NetflowExNtv` | BTC vào sàn − ra sàn | 136 BTC (−52.318 – 88.036); âm ở 48,2% số ngày |
| | `cm_SplyExNtv` | Lượng BTC đang nằm trên sàn | 2,89 triệu (1,95 – 3,18 triệu) BTC |
| Mining | `cm_HashRate` | Hash rate trung bình (TH/s) | 203 triệu (13,7 triệu – 1,31 tỷ) |
| | `cm_FeeTotNtv` | Tổng phí giao dịch (BTC) | 20,8 (1,62 – 1.206) |
| | `bc_difficulty` | Độ khó đào | 28,2×10¹² (1,92 – 156×10¹²) |
| Valuation | `cm_CapMVRVCur` | MVRV | 1,67 (0,69 – 3,96) |
| | `puell_multiple` | Puell Multiple | 1,01 (0,30 – 4,36) |
| | `nvt_ratio_bc` | NVT | 166 (25,2 – 1.144) |

**Phương án Flows:** vì `Netflow = In − Out`, dùng cả ba sẽ đa cộng tuyến hoàn hảo. Phương án **A** (đang dùng): Netflow + Sply. Phương án **B**: In + Out + Sply. Nhóm chưa chốt (task 1.19).

### 2.4 Phát hiện từ EDA

- Tỷ lệ ngày tăng theo năm dao động **45,8% (2022) – 57,7% (2020)** → DA của từng giai đoạn phải so với tỷ lệ tăng của chính giai đoạn đó, không so với 50%.
- HashRate và difficulty tăng gần như đơn điệu → chỉ dùng dạng đã biến đổi (log-diff, z-score), không dùng mức thô.
- Các cặp trùng thông tin: **MVRV ≈ return ngày t** (ρ = 0,98), **NVT ≈ −giá trị giao dịch** (ρ = −1,00, vì NVT chia cho chính nó), **HashRate ≈ Puell** (ρ = 0,95).
- `rv_7d`, `rv_30d`, `hl_range` tương quan mạnh với nhau.

---

## 3. Kỹ thuật thực hiện

### 3.1 Luồng tổng thể

```
Binance · CoinMetrics · Blockchain.com
  → crawl_data/ (crawl, làm sạch, QA)            [Khoa, Vi]
  → data/processed/ (data freeze v1.0)            [Khoa, Vi]
  → tram_features_backtest/ (10 đặc trưng OHLCV, EDA, backtest)   [Trâm]
  → ngoc_pipeline/ (nhãn, 33 đặc trưng On-chain, walk-forward, mô hình) [Ngọc]
  → ngan_lstm_hmm_analysis/ (HMM gán regime, LSTM)                [Ngân]
  → results/predictions/{model}_{config}.csv (date, fold, y_true, p_hat, regime)
  → bảng Δ, Δ theo regime, backtest, kiểm định thống kê            [Vi]
```

### 3.2 Đặc trưng

**OHLCV (10 đặc trưng)**, dòng ngày t chỉ dùng dữ liệu đến hết ngày t:

| Đặc trưng | Công thức |
|---|---|
| `logret_lag1..5` | Log return của ngày t, t−1, …, t−4 |
| `hl_range` | (High − Low) / Close |
| `dlog_volume` | ln V(t) − ln V(t−1) |
| `rsi_14` | RSI 14 ngày kiểu Wilder |
| `rv_7d`, `rv_30d` | Realized volatility từ return 1h trong 7 / 30 ngày |

**On-chain (33 = 11 chỉ số × 3 biến đổi)**, sau đó **dịch thêm 1 ngày**:

| Loại cột | 3 đặc trưng |
|---|---|
| Cột dương (đa số) | `logdiff` = log x(t) − log x(t−1); `ma7_30` = MA7 / MA30; `z90` = z-score 90 ngày của log x |
| Cột có dấu (Netflow) | `diff` = x(t) − x(t−1); `ma7m30` = MA7 − MA30; `z90` = z-score 90 ngày của x |

Ngày nguồn không công bố được điền bằng giá trị trước đó, tối đa 3 ngày.

**Vì sao dịch 1 ngày?** CoinMetrics công bố số liệu ngày d khoảng 2–9,5 giờ sau 00:00 UTC ngày d+1. Lúc cuối ngày t (khi cần dự báo t+1), OHLCV đã có đến ngày t nhưng On-chain chỉ chắc chắn có đến ngày t−1. Không dịch thì mô hình "nhìn trộm" dữ liệu chưa công bố.

### 3.3 Walk-forward

Thay vì chia 80/20 một lần, dữ liệu được chia thành nhiều cửa sổ trượt theo thời gian. Mô hình luôn học từ quá khứ rồi dự báo tương lai gần, giống cách dùng thực tế.

| Tham số | Giá trị |
|---|---|
| Cửa sổ train | 730 ngày, trượt |
| Gap train–test | 1 ngày (nhãn ngày train cuối dùng giá ngày test đầu) |
| Cửa sổ test | 30 ngày, bước 30 ngày |
| Giai đoạn test | 01/01/2020 – 30/08/2026: **82 fold, 2.434 ngày** |
| Tuning | GridSearchCV trên TimeSeriesSplit(5, gap 1), tiêu chí log-loss, tune lại mỗi 6 fold, cùng grid cho hai cấu hình |
| Chuẩn hoá | StandardScaler trong sklearn Pipeline, fit lại mỗi fold |

### 3.4 Mô hình

| Mã | Mô hình | Thiết lập |
|---|---|---|
| B0 Majority | p̂ = tỷ lệ ngày tăng trong train | Mức sàn |
| B0 Persistence | Ngày mai cùng hướng hôm nay | Mức sàn |
| B1 | Logistic Regression | Grid C |
| B2 | Random Forest | Grid số cây, độ sâu, min_samples_leaf |
| B3 | SVM RBF | Grid C, gamma; xác suất hiệu chỉnh sigmoid |
| B4 | XGBoost | Grid learning rate, độ sâu; early stopping trên 20% cuối cửa sổ train |
| B5 | LSTM | 1 lớp, hidden 32, cửa sổ 30 ngày; p̂ = trung bình 3 seed (42, 43, 44) |

Mỗi mô hình chạy 2 lần: **OHLCV-only** và **OHLCV + On-chain**. B0 không dùng đặc trưng nên Δ luôn = 0.

### 3.5 Chống rò rỉ dữ liệu (leakage)

Mỗi lần chạy, bộ test tự động kiểm tra:
- sửa dữ liệu tương lai thì đặc trưng quá khứ không đổi;
- On-chain đã dịch 1 ngày;
- có gap giữa train và test, kể cả trong inner CV;
- scaler không thấy test (nhân dữ liệu test lên 1.000 lần, dự báo không đổi);
- dữ liệu đúng phiên bản freeze.

Nhóm cố ý cài 4 lỗi rò rỉ vào một bản sao, bộ test bắt được cả 4. Kết quả tái lập giống hệt trên Python 3.11 và 3.14.

---

## 4. Kết quả thử nghiệm

Tất cả tính trên 2.434 ngày test ngoài mẫu, tỷ lệ ngày tăng 50,9%, phương án Flows A.

### 4.1 E0 – Mức sàn

- **Majority:** DA 50,08%, log-loss 0,6938.
- **Persistence:** DA 46,51% (MCC −0,07) → giai đoạn 2020–2026, hướng giá ngày mai hay **ngược** hôm nay.

### 4.2 E1 – On-chain có giúp không? (RQ1)

| Mô hình | DA (%) OHLCV → +OC | F1 macro | AUC | Log-loss | % ngày đoán tăng |
|---|---|---|---|---|---|
| B0 Majority | 50,08 | 0,491 | 0,501 | 0,6938 | 63,0 |
| B0 Persistence | 46,51 | 0,465 | 0,465 | – | 50,9 |
| B1 Logistic Regression | 51,11 → 51,89 | 0,509 → 0,517 | 0,510 → 0,513 | 0,6953 → 0,7006 | 56,4 → 54,7 |
| B2 Random Forest | 51,23 → 51,40 | 0,511 → 0,513 | 0,514 → 0,515 | 0,6961 → 0,6957 | 54,7 → 54,1 |
| B3 SVM RBF | 50,62 → 50,29 | 0,477 → 0,491 | 0,503 → 0,498 | 0,7009 → 0,6959 | 72,8 → 64,7 |
| B4 XGBoost | 50,45 → 48,97 | 0,495 → 0,486 | 0,502 → 0,499 | 0,6973 → 0,6958 | 62,7 → 58,0 |
| B5 LSTM (3 seed) | 50,74 → 50,58 | 0,504 → 0,493 | 0,500 → 0,509 | 0,6950 → 0,6959 | 57,5 → 33,4 |

**Bảng Δ và kiểm định:**

| Mô hình | ΔDA (điểm %) [CI 95%] | McNemar n01/n10, p (Holm) | Δlog-loss [CI 95%] | DM stat, p (Holm) |
|---|---|---|---|---|
| B1 LR | +0,78 [−1,27; +2,92] | 327/346, 0,49 (1,00) | +0,0054 [+0,0012; +0,0102] | −2,23, 0,026 (0,13) |
| B2 RF | +0,16 [−1,81; +2,01] | 309/313, 0,90 (1,00) | −0,0005 [−0,0034; +0,0025] | +0,29, 0,77 (1,00) |
| B3 SVM | −0,33 [−2,05; +1,40] | 255/247, 0,75 (1,00) | −0,0050 [−0,0117; +0,0010] | +1,65, 0,098 (0,39) |
| B4 XGB | −1,48 [−3,33; +0,37] | 308/272, 0,15 (0,73) | −0,0015 [−0,0052; +0,0024] | +0,77, 0,44 (1,00) |
| B5 LSTM | −0,16 [−2,38; +2,10] | 431/427, 0,92 (1,00) | +0,0009 [−0,0025; +0,0046] | −0,49, 0,62 (1,00) |

n01 / n10: số ngày chỉ OHLCV-only đoán đúng / chỉ bản có On-chain đoán đúng. DM > 0 nghĩa là On-chain tốt hơn.

**Nhận xét:**
- Mọi mô hình chỉ đúng **49–52%** số ngày, gần như tung đồng xu.
- **Không mô hình nào thắng Majority về log-loss** (0,6938): xác suất mô hình đưa ra chưa tốt hơn việc luôn đoán bằng tỷ lệ tăng của train.
- ΔDA **đổi dấu** giữa các mô hình (−1,48 đến +0,78 điểm). **Mọi CI 95% của ΔDA đều chứa 0**, không cặp nào có p McNemar < 0,05.
- Theo quy tắc ghi trước, mô hình tốt nhất (log-loss thấp nhất ở cấu hình đầy đủ) là **B2 Random Forest** (0,6957). ΔDA của B2 chỉ +0,16 điểm, p = 0,90 → **không chấp nhận H1**.
- Riêng B1, thêm On-chain làm log-loss **tệ hơn** (CI không chứa 0, DM p = 0,026), nhưng sau hiệu chỉnh Holm p = 0,13.
- Δ rất nhạy với bộ đặc trưng OHLCV: đổi từ 7 đặc trưng tạm sang 10 đặc trưng chính thức, ΔDA của B1 đổi dấu từ −0,45 thành +0,78 điểm.

### 4.3 Backtest long/flat

Chiến lược: giữ BTC ngày t+1 nếu p̂ > 0,5, ngược lại cầm tiền; phí 0,1% mỗi lần đổi vị thế; 02/01/2020 – 31/08/2026.

| Mô hình | Lợi nhuận tích luỹ | Sharpe | MDD | Số lần đổi vị thế |
|---|---|---|---|---|
| Buy-and-hold | +990% | 0,90 | −76,6% | 1 |
| B0 Majority | +72% | 0,43 | −82,4% | 7 |
| B0 Persistence | −71% | −0,22 | −79,4% | 1.302 |
| B1 LR | +19% → +245% | 0,31 → 0,63 | −86,6% → −65,4% | 843 → 857 |
| B2 RF | +241% → +14% | 0,62 → 0,29 | −80,2% → −76,4% | 857 → 771 |
| B3 SVM | +286% → +63% | 0,65 → 0,41 | −72,4% → −80,2% | 264 → 327 |
| B4 XGB | −43% → −65% | 0,11 → −0,05 | −86,4% → −93,0% | 509 → 559 |
| B5 LSTM | +99% → +192% | 0,46 → 0,61 | −85,2% → −56,4% | 363 → 288 |

**Không chiến lược nào thắng buy-and-hold.** Lợi nhuận đổi chiều thất thường giữa hai cấu hình dù DA gần như bằng nhau, vì lời lỗ phụ thuộc vào vài ngày biến động mạnh. Ví dụ B5 + On-chain chỉ giữ BTC 33% số ngày mà lời +192%, trong khi DA không đổi → nhiều khả năng là may mắn về thời điểm. Vì vậy backtest **không** được dùng làm bằng chứng chính cho RQ1.

### 4.4 E3 – Δ theo regime (RQ3)

| Regime | Số ngày test | Số đoạn | Độ dài đoạn TB | Tỷ lệ ngày tăng |
|---|---|---|---|---|
| Bull | 1.250 | 417 | 3,0 ngày | 51,6% |
| Sideways | 902 | 318 | 2,8 ngày | 49,7% |
| Bear | 282 | 149 | 1,9 ngày | 51,8% |

| Mô hình | ΔDA Bull | ΔDA Sideways | ΔDA Bear |
|---|---|---|---|
| B1 LR | +0,08 | +1,66 | +1,06 |
| B2 RF | −1,20 | +1,66 | +1,42 |
| B3 SVM | −0,64 | −0,89 | +2,84 |
| B4 XGB | −1,12 | −2,88 | +1,42 |
| B5 LSTM | −2,08 | +1,55 | +2,84 |

- Trong 15 cặp mô hình × regime, **mọi CI 95% của ΔDA đều chứa 0**.
- Placebo (xáo thứ tự các đoạn regime 1.000 lần): Δ_Bull − Δ_Bear nằm trong khoảng 95% của phân phối giả ở cả 5 mô hình, p (Holm) ≥ 0,91.
- Chiều chênh lệch còn **ngược** H3 (Bear nhỉnh hơn Bull) → **không chấp nhận H3**, nên **E4 không thực hiện**.

### 4.5 Kết luận giữa kỳ

Trên 82 fold walk-forward (2020 – 08/2026), thêm 11 chỉ số On-chain (33 đặc trưng) **không cải thiện** dự báo hướng giá BTC ngày kế tiếp ở bất kỳ mô hình nào trong 5 mô hình có học. Mọi Δ nằm trong vùng nhiễu, và Δ không khác nhau có ý nghĩa giữa các regime. Kết quả này khác hẳn con số 82–83% của các bài báo trước, và khớp với phỏng đoán ban đầu của nhóm.

### 4.6 Hạn chế và việc còn lại

- **Flows không point-in-time:** CoinMetrics sửa lại lịch sử Flows (dữ liệu "flash", phụ thuộc nhãn ví sàn), nên kết quả có Flows có thể lạc quan hơn thực tế.
- **Regime đổi gần như mỗi ngày** (đoạn trung bình 1,9–3,0 ngày): HMM trên return và volatility theo ngày tách các ngày dao động hơn là các pha thị trường kéo dài.
- **Regime chưa tái lập được từ repo:** `regime_map.csv` là file xuất sẵn, thử dựng lại chỉ khớp 39–59% số ngày.
- Chưa chạy E2 (ablation + SHAP), độ nhạy cửa sổ train, phương án Flows B; B1, B2, B4 hiện là bản chạy thử.
- Hướng tiếp theo: HMM theo tuần để có pha dài hơn; thử mục tiêu khác như dự báo biến động hoặc horizon dài hơn.

---

## 5. Thuật ngữ: giá trị On-chain

**On-chain** là dữ liệu đọc trực tiếp từ blockchain Bitcoin (giao dịch, địa chỉ, block, phí, phần thưởng đào), khác với dữ liệu **off-chain** như giá và volume trên sàn (OHLCV). Ý tưởng: hành vi của người dùng và thợ đào trên mạng lưới có thể báo trước biến động giá.

| Chỉ số | Cách tính | Ý nghĩa trực giác |
|---|---|---|
| **Active addresses** (`cm_AdrActCnt`) | Số địa chỉ khác nhau xuất hiện trong giao dịch thành công trong ngày | Mạng lưới đông người dùng hơn hay ít hơn |
| **Transaction count** (`cm_TxCnt`) | Số giao dịch trong ngày | Mức độ sử dụng mạng |
| **Transaction volume USD** (`bc_est_tx_vol_usd`) | Tổng giá trị USD được chuyển trên chuỗi (Blockchain.com ước lượng) | Lượng tiền thật sự di chuyển |
| **Exchange Netflow** (`cm_NetflowExNtv`) | `FlowIn − FlowOut`: BTC chuyển vào ví sàn trừ BTC rút ra | Dương: người ta đưa BTC lên sàn, có thể để bán (áp lực bán). Âm: rút về ví riêng để giữ |
| **Supply on exchanges** (`cm_SplyExNtv`) | Tổng BTC nằm trong các ví được gắn nhãn là của sàn | Lượng BTC "sẵn sàng bán" |
| **Hash rate** (`cm_HashRate`) | Tổng sức tính toán của mạng (TH/s) | Mức đầu tư và niềm tin của thợ đào |
| **Total fees** (`cm_FeeTotNtv`) | Tổng phí giao dịch trong ngày (BTC) | Mạng có bị tắc nghẽn, nhu cầu chuyển tiền cao không |
| **Difficulty** (`bc_difficulty`) | Độ khó đào, tự điều chỉnh mỗi 2.016 block để giữ ~10 phút/block | Đi theo hash rate, tăng gần như đơn điệu |
| **MVRV** (`cm_CapMVRVCur`) | `Market Cap / Realized Cap`. Market Cap = giá hiện tại × lượng cung. Realized Cap = mỗi đồng BTC định giá theo giá lúc nó di chuyển lần cuối | > 1: trung bình người nắm giữ đang lời; quá cao thường là vùng đỉnh, < 1 thường là vùng đáy |
| **Puell Multiple** (`puell_multiple`) | `Giá trị BTC phát hành trong ngày (USD) / Trung bình 365 ngày của giá trị đó` | Thu nhập thợ đào so với mức bình thường; cao → thợ đào dễ bán chốt lời |
| **NVT** (`nvt_ratio_bc`) | `Market Cap / Transaction volume USD` (cùng ngày, không làm mượt) | Giống P/E của cổ phiếu: vốn hoá cao so với lượng sử dụng thực → có thể bị định giá quá cao |

Lưu ý từ EDA: MVRV, Puell, NVT đều chứa giá trong công thức, nên nhóm Valuation có thể chỉ lặp lại thông tin giá (MVRV ρ = 0,98 với return ngày t).

---

## 6. Thuật ngữ: Regime và Delta

### 6.1 Regime (chế độ thị trường)

**Regime** là trạng thái chung của thị trường trong một khoảng thời gian: **Bull** (tăng), **Bear** (giảm), **Sideways** (đi ngang). Giả định là quan hệ giữa On-chain và giá có thể khác nhau tuỳ regime (bài [4] cho thấy tín hiệu On-chain hiệu quả ở chiều tăng nhưng không ở chiều giảm).

### 6.2 Cách gán regime bằng HMM

**HMM (Hidden Markov Model)** giả định thị trường luôn ở một trong K trạng thái ẩn (ở đây K = 3). Mỗi trạng thái sinh ra quan sát theo một phân phối riêng, và thị trường chuyển giữa các trạng thái theo xác suất chuyển.

Cách làm trong dự án:
1. **Quan sát mỗi ngày:** `[log return, realized volatility]`.
2. **Mô hình:** GaussianHMM 3 trạng thái, covariance dạng diag, 5 lần khởi tạo.
3. **Fit lại trong mỗi fold** walk-forward, chỉ trên dữ liệu train (đã chuẩn hoá).
4. **Đặt tên trạng thái:** trạng thái có return trung bình trên train thấp nhất là **Bear**, cao nhất là **Bull**, còn lại là **Sideways**.
5. **Gán regime cho ngày test bằng forward filtering:** chỉ dùng quan sát đến ngày đó, không dùng tương lai (khác với Viterbi/smoothing dùng cả chuỗi).
6. Kết quả lưu ở `regime_map.csv`, được ghép vào cột `regime` của mọi file dự báo theo `date` + `fold`.

### 6.3 Delta (Δ): giá trị bổ sung của On-chain

**Công thức:**

```
Δ = M( f_OHLCV+OnChain ) − M( f_OHLCV-only )
```

trong đó `M` là một metric (DA, log-loss, Brier, AUC…). Hai vế dùng **cùng mô hình, cùng các fold walk-forward, cùng grid tuning**; chỉ khác bộ đặc trưng.

**Ví dụ tính với B1 Logistic Regression:**

```
DA(OHLCV-only)      = 51,11%
DA(OHLCV+On-chain)  = 51,89%
ΔDA = 51,89 − 51,11 = +0,78 điểm %

Log-loss(OHLCV-only)     = 0,6953
Log-loss(OHLCV+On-chain) = 0,7006
Δlog-loss = 0,7006 − 0,6953 = +0,0054   (log-loss càng thấp càng tốt → On-chain làm TỆ hơn)
```

**Đọc dấu của Δ:**

| Metric | Δ > 0 nghĩa là |
|---|---|
| DA, F1, MCC, AUC (cao là tốt) | On-chain giúp |
| Log-loss, Brier (thấp là tốt) | On-chain làm tệ hơn |

Δ còn được tính theo từng regime (Δ_Bull, Δ_Bear, Δ_Sideways, cho RQ3) và theo từng nhóm chỉ số (Δ_Activity, Δ_Flows…, cho RQ2).

**Vì sao dùng Δ thay vì nhìn độ chính xác của từng mô hình?**

1. **Tách riêng phần đóng góp của On-chain.** Một con số "DA 83%" trộn lẫn nhiều thứ: mô hình mạnh hay yếu, cách chia train/test, giai đoạn test dễ hay khó, có rò rỉ hay không. Khi giữ nguyên mọi thứ và chỉ đổi bộ đặc trưng, phần chênh lệch còn lại chỉ có thể do On-chain.
2. **Lấp đúng khoảng trống của related work.** Các bài báo trước không so với mô hình cùng kiến trúc chỉ dùng OHLCV, nên không biết 82–83% là nhờ On-chain hay nhờ giá.
3. **Trung hoà mức "dễ/khó" của giai đoạn test.** Tỷ lệ ngày tăng thay đổi từ 45,8% đến 57,7% theo năm. DA 55% có thể cao ở năm này nhưng thấp ở năm khác. Hai vế của Δ chấm trên cùng những ngày test nên yếu tố đó tự triệt tiêu.
4. **Cho phép kiểm định cặp (paired test).** Vì hai cấu hình dự báo cùng những ngày, có thể so trực tiếp từng ngày (McNemar, Diebold–Mariano, bootstrap cặp). Kiểm định cặp nhạy hơn nhiều so với so hai con số độc lập.
5. **So sánh được giữa các mô hình và regime.** Δ là đơn vị chung để hỏi "On-chain giúp LR nhiều hơn XGBoost không?" hay "giúp ở Bull nhiều hơn Bear không?".

**Mức nhiễu cần nhớ:** với n = 2.434 ngày, sai số chuẩn của DA ≈ √(0,25 / 2.434) ≈ 1 điểm %. Với Bear (282 ngày) là khoảng 3 điểm %. Δ nhỏ hơn mức này rất dễ chỉ là ngẫu nhiên, vì vậy Δ luôn phải đi kèm CI và kiểm định.

---

## 7. Các chỉ số đánh giá

### 7.1 Chỉ số phân loại và xác suất

| Chỉ số | Công thức / cách tính | Tốt khi | Mốc tham chiếu |
|---|---|---|---|
| **DA** (Directional Accuracy) | Tỷ lệ ngày đoán đúng hướng (p̂ > 0,5 → đoán tăng) | Cao | 50% là tung đồng xu; nên so với tỷ lệ ngày tăng của giai đoạn |
| **F1 macro** | Trung bình F1 của lớp "tăng" và lớp "giảm"; F1 = 2·Precision·Recall / (Precision + Recall) | Cao | Phạt mô hình chỉ đoán một phía (ví dụ luôn đoán tăng) |
| **MCC** (Matthews Correlation Coefficient) | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)), từ −1 đến 1 | Cao | 0 là ngẫu nhiên; < 0 là ngược (Persistence = −0,07) |
| **AUC** (ROC-AUC) | Xác suất một ngày tăng ngẫu nhiên được cho p̂ cao hơn một ngày giảm ngẫu nhiên | Cao | 0,5 là không phân biệt được |
| **Log-loss** | −trung bình[ y·ln p̂ + (1−y)·ln(1−p̂) ] | Thấp | ln 2 ≈ 0,6931 khi luôn đoán p̂ = 0,5; Majority = 0,6938. Phạt nặng khi tự tin mà sai |
| **Brier score** | trung bình (p̂ − y)² | Thấp | 0,25 khi luôn đoán p̂ = 0,5 |

DA, F1, MCC đánh giá **quyết định** (đúng hay sai). AUC, log-loss, Brier đánh giá **chất lượng xác suất**. Log-loss là tiêu chí chính để tune và chọn mô hình tốt nhất.

### 7.2 Chỉ số backtest (chiến lược long/flat)

| Chỉ số | Cách tính | Ý nghĩa |
|---|---|---|
| **Lợi nhuận tích luỹ** | ∏(1 + r_t) − 1, với r_t là lợi nhuận ngày khi đang giữ BTC (trừ phí 0,1% mỗi lần đổi vị thế) | Tổng lời/lỗ cả giai đoạn |
| **Sharpe ratio** | trung bình(r) / độ lệch chuẩn(r) × √365 | Lợi nhuận trên mỗi đơn vị rủi ro, quy về năm (crypto giao dịch 365 ngày) |
| **MDD** (Maximum Drawdown) | Mức sụt lớn nhất từ đỉnh vốn xuống đáy sau đó | Rủi ro tệ nhất phải chịu |
| **Số lần đổi vị thế** | Số lần chuyển giữa giữ BTC và cầm tiền | Càng nhiều càng tốn phí |

### 7.3 Kiểm định thống kê cho Δ

| Phương pháp | Dùng cho | Ý tưởng |
|---|---|---|
| **Circular block bootstrap** (khối 15 ngày, 2.000 mẫu) | CI 95% của Δ | Lấy mẫu lại theo khối liên tiếp để giữ phụ thuộc thời gian; nếu CI chứa 0 thì Δ chưa khác 0 có ý nghĩa |
| **McNemar chính xác** | ΔDA | Chỉ xét những ngày hai cấu hình đoán khác nhau: n01 (chỉ OHLCV đúng) và n10 (chỉ On-chain đúng). Nếu On-chain không giúp, hai số này xấp xỉ nhau |
| **Diebold–Mariano** (phương sai Newey–West, độ trễ 8) | Δlog-loss | So chuỗi chênh lệch loss từng ngày giữa hai cấu hình, có hiệu chỉnh tự tương quan |
| **Hiệu chỉnh Holm** | Nhiều phép so sánh | Chạy 5 mô hình thì có nhiều cơ hội "trúng" p < 0,05 do may mắn; Holm nâng ngưỡng để kiểm soát sai lầm này |
| **Placebo regime** (1.000 lần, giữ độ dài đoạn) | Δ_Bull − Δ_Bear | Xáo thứ tự các đoạn regime; nếu chênh lệch thật nằm trong phân phối giả thì nó không đặc biệt gì so với regime ngẫu nhiên |

**Quy tắc kết luận ghi trước (pre-registered):** chọn mô hình tốt nhất theo log-loss của cấu hình OHLCV + On-chain **trước khi xem Δ**; chấp nhận H1 nếu ΔDA > 0 với p < 0,05 ở mô hình đó. Ghi trước để tránh việc chọn mô hình "đẹp" sau khi đã thấy kết quả.
