# simple_split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một script chạy B0, LR, SVM RBF, RF, LSTM trên cùng cách chia train/val/test, tune trên validation, test 1 lần, xem trực tiếp dự báo, ra một báo cáo so sánh.

**Architecture:** `simple_split/src/simple_split.py` dùng lại `baseline_check` (dữ liệu, chia, metric, luật B0), estimator của từng người (`lr_tuning.ElasticNetLR`, `models.CalibratedSVC`, `RandomForestClassifier`, `lstm_tuning.fit_predict`). Một hàm tune chung nhận "hàm học → p̂" cho mỗi tổ hợp, nên mọi mô hình đi qua đúng một vòng chọn tham số.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, scipy, torch (LSTM), tabulate. Test theo quy ước của repo: hàm `_tests()` gọi bằng `--test` (repo không dùng pytest).

**Spec:** `docs/superpowers/specs/2026-11-13-simple-split-design.md`

## Global Constraints

- Đặc trưng: `FEATURES_10` import tường minh từ `features_ohlcv`; không dùng `FEATURES`.
- Chia: `baseline_check.split(baseline_check.load_data())` (train 2018-01-01→2021-12-30, val 2022-01-01→2023-12-30, test 2024-01-01→2026-08-30).
- Chọn tham số: min log-loss validation, làm tròn 4 chữ số, hoà thì tổ hợp đứng trước trong grid (grid xếp từ đơn giản đến phức tạp).
- Mô hình cuối học trên train + val; test chỉ được đọc ở bước chấm cuối.
- Seed: RF và LSTM dùng `SEEDS = [42, 43, 44]`, p̂ trung bình 3 seed ở mọi lần học. B0, LR, SVM không seed.
- Grid: LR = `lr_tuning.GRID` × `lr_tuning.feature_sets(FEATURES_10)` + winsorize `CLIP_Q`; SVM = `{"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]}`; RF = `b2t.yaml` param_grid, 500 cây, max_depth None; LSTM = `lstm_tuning.GRID`.
- "Trước tune": LR `ElasticNetLR(C=1, l1_ratio=0)` trên 10 đặc trưng, không winsorize; SVM `CalibratedSVC(C=1, gamma="scale")`; RF `RandomForestClassifier(n_estimators=500)`; LSTM `lstm_tuning.DEFAULT`.
- Không sửa file của người khác; chỉ tạo `simple_split/`.

## Review Focus

- **Test bị đọc khi tune:** hàm tune không được nhận ngày test. Test: gọi tune với dữ liệu test bị thay bằng NaN vẫn chạy và chọn cùng tham số.
- **Tập đặc trưng 11 cột lọt vào:** mọi mô hình phải thấy đúng 10 cột (LR thêm cột dấu nếu tập được chọn có). Test: assert số cột đầu vào của LR/SVM/RF/LSTM ⊆ `FEATURES_10 + ["up_lag1"]`.
- **Cột dấu của LR nhìn tương lai:** `up_lag1` phải = 1[logret_lag1 > 0] của chính ngày t. Test: so khớp từng dòng.
- **Dự báo lệch ngày giữa các mô hình:** `predictions_test.csv` mọi cột cùng 973 ngày, không NaN. Test: assert trong `--quick`.
- **Seed thay đổi kết quả khi chạy lại:** cùng seed phải ra cùng p̂. Test: chạy RF hai lần với `[42]`, p̂ giống hệt.

---

### Task 1: Khung, chia dữ liệu, B0 và vòng tune chung

**Files:**
- Create: `simple_split/src/simple_split.py`, `simple_split/README.md`

**Interfaces:**
- Produces:
  - `load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]`: df (đã thêm `up_lag1`) và parts {"train","val","test"}.
  - `tune(name: str, fit_fn: Callable[[dict, pd.DataFrame, pd.DataFrame], np.ndarray], grid: list[dict], parts) -> tuple[pd.DataFrame, dict]`: với mỗi tổ hợp gọi `fit_fn(params, learn_df, target_df)` trả p̂ cho target; học trên train, chấm train và val; trả bảng (mọi cột params + `train_log_loss, val_log_loss, train_DA, val_DA, chosen`) và tham số chọn.
  - `final(fit_fn, params, parts) -> np.ndarray`: học trên train+val, trả p̂ test.
  - `rules(parts) -> dict[str, np.ndarray]`: 4 luật B0 (từ `bc.rule_predictions`, thêm "Buy-and-hold" = luôn tăng).

- [ ] **Step 1: Viết `_tests()` cho Task 1**: `up_lag1` khớp `(logret_lag1 > 0)` mọi dòng; `len(parts["test"]) == 973`; `tune` trên một `fit_fn` giả (trả p̂ = params["p"]) với grid `[{"p":0.5},{"p":0.4}]` chọn đúng tổ hợp có val log-loss thấp hơn; gọi `tune` sau khi gán `parts["test"]["y"] = np.nan` vẫn ra cùng lựa chọn; DA của luật đảo chiều trên test = 0.5098 (khớp `baseline_check`, 50,98%).
- [ ] **Step 2:** `python simple_split/src/simple_split.py --test` → FAIL (chưa có hàm).
- [ ] **Step 3:** Cài các hàm ở Interfaces; CLI `--test`, `--quick` (grid rút gọn: tổ hợp đầu và cuối của mỗi grid, 1 seed), mặc định chạy đủ.
- [ ] **Step 4:** `--test` → "Tất cả test simple_split đều PASS".

### Task 2: LR, SVM, RF

**Files:** Modify `simple_split/src/simple_split.py`

**Interfaces:**
- Consumes: `tune`, `final`, `load` (Task 1).
- Produces: `MODELS: dict[str, dict]` với mỗi khoá `"LR"`, `"SVM RBF"`, `"RF"` có `{"before": dict, "grid": list[dict], "fit": Callable}`; `fit` đúng chữ ký `fit_fn` của Task 1. RF `fit` lặp `SEEDS` (đặt `random_state`) và lấy trung bình p̂.

- [ ] **Step 1: Thêm vào `_tests()`**: `fit` của LR với tập `"sign"` chỉ nhận cột `["up_lag1"]`, của SVM/RF nhận đúng `FEATURES_10`; RF chạy 2 lần với `SEEDS=[42]` ra p̂ giống hệt; số tổ hợp grid: LR 300, SVM 30, RF 24.
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài `MODELS` cho LR (pipeline = `walkforward.make_pipeline(ElasticNetLR(...), CLIP_Q)` khi sau tune, `make_pipeline(..., None)` khi trước tune), SVM (`make_pipeline(CalibratedSVC(...))`), RF (`make_pipeline(RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=s, **params))`).
- [ ] **Step 4:** `--test` PASS; `--quick` chạy hết không lỗi.

### Task 3: LSTM

**Files:** Modify `simple_split/src/simple_split.py`

**Interfaces:**
- Consumes: `lstm_tuning.load_full`, `lstm_tuning.fit_predict(feats, train_dates, target_sets, cfg, seed)` (đặt `lstm_tuning.COLS = FEATURES_10` trước khi gọi), `tune`, `final`.
- Produces: khoá `"LSTM"` trong `MODELS`, `fit` trung bình p̂ qua `SEEDS`.

- [ ] **Step 1: Thêm vào `_tests()`**: sau khi khởi tạo, `lstm_tuning.COLS == FEATURES_10`; `fit` của LSTM với `{"lookback":10,"hidden":8,"dropout":0.0,"lr":1e-3}` trả đúng số p̂ bằng số dòng target, không NaN.
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài `fit` LSTM: `feats = load_full()` (cache một lần), gán `feats["_y"]`, gọi `fit_predict(feats, learn_df.index, {"t": target_df.index}, cfg, s)`.
- [ ] **Step 4:** `--test` PASS.

### Task 4: Xem dự báo, báo cáo, chạy đủ

**Files:** Modify `simple_split/src/simple_split.py`, `simple_split/README.md`

**Interfaces:**
- Consumes: mọi thứ ở Task 1–3, `bc.metrics`, `bc.fmt_metrics`.
- Produces: `inspect(name: str, p: np.ndarray, test: pd.DataFrame, rev: np.ndarray) -> list[str]` (các dòng markdown); `report(...)` ghi `simple_split/reports/{simple_split.md, test_metrics.csv, tuning_<model>.csv, predictions_test.csv}`.

- [ ] **Step 1: Thêm vào `_tests()`**: `inspect` trên p̂ giả trả đủ 6 mục (ma trận nhầm lẫn, 10 ngày ngẫu nhiên với seed cố định, 8 ngày sai tự tin nhất, 10 ngày biến động mạnh nhất, DA theo 4 nhóm biến động, % đồng ý với luật đảo chiều); bảng so sánh có thêm cột McNemar p so với luật đảo chiều.
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài `inspect` và `report`. Báo cáo có: bảng so sánh (4 luật B0 + trước/sau tune của 4 mô hình), bảng tuning từng mô hình (sắp theo val log-loss, đánh dấu tổ hợp chọn), phần xem dự báo của bản sau tune mỗi mô hình, dòng lưu ý sai số ±1,6 điểm với 973 ngày. README ghi lệnh chạy và đầu ra.
- [ ] **Step 4:** `--test` PASS, `--quick` PASS và assert `predictions_test.csv` đủ 973 dòng, không NaN.
- [ ] **Step 5:** Chạy đủ `python simple_split/src/simple_split.py` ở nền; kiểm tra trong báo cáo: DA luật đảo chiều = 50,98%, Majority = 50,67% (khớp `baseline_check`).
- [ ] **Step 6:** Không commit (người dùng đang ở nhánh `feature/midterm02`); báo kết quả và hỏi.
