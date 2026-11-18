# Thực nghiệm lại bài Hafid et al. 2024 (arXiv:2410.06935) – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái tạo thí nghiệm của Hafid et al. 2024 (XGBoost/LR trên chỉ báo kỹ thuật từ OHLCV BTC 15 phút, báo cáo 92,4%) trên dữ liệu raw tải từ Binance, rồi chạy lại cùng đặc trưng và mô hình với nhãn dự báo thật để xem DA còn bao nhiêu.

**Architecture:** Thư mục `analyses/paper_replication/hafid2024/`: `download.py` tải nến 15 phút BTCUSDT từ Binance Data Vision (kiểm checksum) và bản PDF của bài; `paper.py` chứa chỉ báo, nhãn, mô hình đúng như bài; `run.py` chạy 3 thí nghiệm × 2 cách chuẩn hoá và ghi báo cáo. Mọi tham số bài không nêu được ghi rõ là lựa chọn của nhóm.

**Tech Stack:** Python, pandas, numpy, scikit-learn, xgboost (cài thêm), pypdf. Test bằng `_tests()` + `--test` theo quy ước repo.

**Spec:** bài báo arXiv:2410.06935v2 (tóm tắt thiết lập trong hội thoại 14/11/2026) và yêu cầu của Ngọc: "tạo thư mục, download raw data, viết chạy thử xem sao".

## Global Constraints

- Thư mục: `analyses/paper_replication/hafid2024/` (Ngọc gõ `exeriment_paper`, dùng `analyses/paper_replication` cho đúng chính tả).
- Dữ liệu bài: BTCUSDT nến 15 phút, 2021-02-01 00:00 → 2022-02-01 00:00 UTC, nguồn `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/` (12 file tháng 2021-02 … 2022-01), mỗi file kiểm `.CHECKSUM` sha256.
- Đặc trưng đầu vào (Eq. 4 của bài): Close, Volume, RSI14, RSI30, RSI200, MOM10, MOM30, MACD, PROC9, EMA10, EMA30, EMA200, %K10, %K30, %K200; thêm %D30, %D200 vì Eq. 6 chọn chúng. Chọn 8 đặc trưng bằng `SelectKBest(chi2, k=8)`.
- Nhãn của bài: `y_t = 1` nếu `MA10_t ≥ MA60_t` (trung bình giản đơn của Close, gồm cả nến t), ngược lại 0.
- Chia 80/20 theo thời gian, không xáo trộn.
- XGBoost: n_estimators 400, learning_rate 0,1, max_depth 4, min_child_weight 3, subsample 0,8, colsample_bytree 1,0, gamma 0,1, reg_alpha 0,5, reg_lambda 1, objective binary:logistic, random_state 0.
- LR: penalty l1, C 0,1, solver saga, max_iter 100.
- Lựa chọn của nhóm khi bài không nêu (ghi trong báo cáo): RSI kiểu Wilder; MACD = EMA12 − EMA26; %D = SMA 3 của %K cùng cửa sổ; MOM_n = C_t − C_{t−n}; PROC_9 = 100·(C_t − C_{t−9})/C_{t−9}; chi2 chạy trên dữ liệu đã MinMax về [0, 1] (chi2 cần giá trị không âm); bỏ các dòng warm-up có NaN.
- Không commit (nhánh `feature/midterm02`).

## Review Focus

- **Nhãn hoặc đặc trưng nhìn tương lai ngoài ý muốn:** đặc trưng dòng t chỉ dùng dữ liệu đến hết nến t; test đổi giá sau t không làm đổi đặc trưng dòng t.
- **Nhãn "dự báo thật" lệch một nến:** nhãn B ở dòng t phải bằng 1[C_{t+1} > C_t]; test so tay vài dòng.
- **Thiếu nến trong dữ liệu Binance:** báo số nến thiếu so với 96 nến/ngày; không nội suy.
- **Fit scaler / chi2 trên toàn bộ dữ liệu khi lẽ ra chỉ trên train:** chạy cả hai chế độ, test chế độ "train-only" không đọc dòng test khi fit.
- **Tái tạo con số của bài:** thí nghiệm A phải cho accuracy cao (gần 0,9); nếu không, báo cáo nêu rõ chênh lệch thay vì chỉnh tham số cho khớp.

---

### Task 1: Tải dữ liệu raw và bài báo

**Files:** Create `analyses/paper_replication/hafid2024/src/download.py`, `analyses/paper_replication/hafid2024/README.md`

**Interfaces:**
- Produces: `data/raw/BTCUSDT-15m-YYYY-MM.zip` (12 file), `data/btcusdt_15m.csv` (cột `open_time` UTC, `open, high, low, close, volume`), `data/manifest.json` (sha256, số dòng, số nến thiếu), `paper/2410.06935v2.pdf`. Hàm `load_15m() -> pd.DataFrame` (index thời gian UTC).

- [ ] **Step 1:** Viết `_tests()`: sau khi tải, `load_15m()` có index tăng dần, không trùng, đầu `2021-02-01 00:00`, cuối `2022-01-31 23:45`; số dòng = 365·96 − số nến thiếu ghi trong manifest; mọi zip khớp sha256.
- [ ] **Step 2:** `python analyses/paper_replication/hafid2024/src/download.py --test` → FAIL (chưa có dữ liệu).
- [ ] **Step 3:** Cài tải zip theo tháng + `.CHECKSUM` (dùng `urllib`, cùng cách `data/raw/vi_ohlcv_flows/src/crawl_binance.py`), xử lý timestamp ms/µs, ghép CSV, ghi manifest; tải `https://arxiv.org/pdf/2410.06935v2` vào `paper/`.
- [ ] **Step 4:** Chạy tải, rồi `--test` → PASS.

### Task 2: Chỉ báo, nhãn, mô hình theo bài

**Files:** Create `analyses/paper_replication/hafid2024/src/paper.py`

**Interfaces:**
- Consumes: `load_15m()`.
- Produces: `features(df) -> pd.DataFrame` (17 cột như Global Constraints), `label_paper(close) -> pd.Series` (MA10 ≥ MA60), `label_next(close) -> pd.Series` (1[C_{t+1} > C_t], NaN ở dòng cuối), `make_xgb()`, `make_lr()`, `split_80_20(n) -> (train_idx, test_idx)`, `fit_eval(X, y, model, scale_on="full"|"train") -> dict` (accuracy, precision, recall, F1, AUC, confusion, % nhãn 1 ở test, 8 đặc trưng được chọn).

- [ ] **Step 1:** Viết `_tests()`: RSI, EMA, %K, MOM, PROC trên chuỗi đồ chơi khớp tính tay; đổi giá sau dòng t không làm đổi `features` và `label_paper` ở dòng t; `label_next` dòng t bằng 1[C_{t+1} > C_t]; `split_80_20(100)` cho train 0–79, test 80–99; với `scale_on="train"`, nhân dữ liệu test lên 1000 lần không làm đổi tham số scaler/chi2.
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài các hàm theo đúng tham số ở Global Constraints. Pipeline: StandardScaler → (MinMax cho chi2) SelectKBest(chi2, 8) → mô hình; `scale_on="full"` fit scaler + chi2 trên toàn bộ dòng (như bài), `"train"` chỉ trên train.
- [ ] **Step 4:** `--test` → PASS.

### Task 3: Chạy 3 thí nghiệm và báo cáo

**Files:** Create `analyses/paper_replication/hafid2024/src/run.py`

**Interfaces:**
- Consumes: Task 1, Task 2; `data/processed/master_daily.csv` (OHLCV ngày của nhóm, qua `baseline_check.load_data` hoặc đọc trực tiếp).
- Produces: `reports/report.md`, `reports/results.csv`.

- [ ] **Step 1:** Viết `_tests()`: `results.csv` có đủ 3 thí nghiệm × 2 chế độ chuẩn hoá × 2 mô hình = 12 dòng, mỗi dòng có majority baseline (tỷ lệ lớp nhiều nhất ở test).
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài 3 thí nghiệm: **A** dữ liệu 15 phút, nhãn của bài (`label_paper`); **B** dữ liệu 15 phút, nhãn nến kế tiếp (`label_next`); **C** OHLCV ngày của nhóm 2018-01-01 → 2026-08-30, cùng 17 đặc trưng tính trên nến ngày, nhãn ngày mai (`label_next`). Mỗi thí nghiệm × {full, train} × {XGBoost, LR}. Báo cáo: bảng kết quả, so với số của bài (XGBoost 0,9240 / LR 0,9101), majority baseline, 8 đặc trưng được chọn, danh sách lựa chọn của nhóm, nhận xét có số đi kèm.
- [ ] **Step 4:** `--test` PASS; chạy đủ; đọc báo cáo, đối chiếu một dòng với `results.csv`.
- [ ] **Step 5:** Không commit; báo kết quả cho Ngọc.
