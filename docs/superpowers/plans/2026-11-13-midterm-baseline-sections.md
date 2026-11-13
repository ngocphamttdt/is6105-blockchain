# Viết lại mục 5 và mục 8 báo cáo giữa kỳ – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bản nháp mục 5 (Baselines) và mục 8 (Kết quả trước/sau tuning) cho báo cáo giữa kỳ, chỉ trình bày đúng những gì `simple_split` đã chạy, kèm danh sách chỗ phải sửa ở các mục khác.

**Architecture:** Một script `simple_split/src/report_sections.py` sinh bản nháp markdown **trực tiếp từ file kết quả** (`simple_split/reports/*.csv`) và từ cấu hình trong `simple_split.py` (`MODELS`, `SEEDS`, cách chia). Không gõ tay con số nào; test so từng số trong bản nháp với CSV. Phần sửa ở các mục khác là văn bản tay, chỉ mô tả việc cần sửa, không thêm kết quả.

**Tech Stack:** Python, pandas, tabulate; test bằng `_tests()` + `--test` theo quy ước repo.

**Spec:** yêu cầu của Ngọc trong hội thoại 13/11/2026 (chỉ OHLCV, cùng train/val/test, bỏ kết quả On-chain, "code chạy gì trình bày nấy, không viết hoa mỹ") và `docs/superpowers/specs/2026-11-13-simple-split-design.md` (thiết lập đã chạy).

## Global Constraints

- Nguồn số liệu duy nhất: `simple_split/reports/test_metrics.csv`, `tuning_<model>.csv`, `predictions_test.csv`, và hằng số trong `simple_split.py`. Không dùng số walk-forward, không dùng số On-chain.
- Chỉ các mô hình đã chạy: Majority, Persistence, Buy-and-hold, Đảo chiều, LR, SVM RBF, RF, LSTM. **Không có XGBoost** (không chạy trong `simple_split`); ghi rõ B4 chưa chạy ở cách chia này.
- Văn phong: câu ngắn, mô tả đúng kết quả; không dùng từ đánh giá không có số đi kèm ("vượt trội", "hiệu quả", "đáng kể"). Mọi so sánh "hơn/kém" phải kèm CI hoặc p.
- Số thập phân kiểu Việt (dấu phẩy), DA 2 chữ số, log-loss 4 chữ số, p 3 chữ số.
- Không sửa file PDF/DOCX; đầu ra là markdown để Ngọc chép vào báo cáo.

## Review Focus

- **Số trong bản nháp lệch CSV** (làm tròn sai, lấy nhầm dòng trước/sau tune): test so mọi DA, log-loss, p trong bảng với CSV.
- **Lọt chữ On-chain hoặc XGBoost vào mục 8** như một kết quả: test grep mục 8 không chứa "On-chain", "Δ", "XGBoost" ngoài câu ghi chú cố định.
- **Tham số "được chọn" không khớp file tuning**: test so tham số in ra với dòng `chosen == True` của `tuning_<model>.csv`.
- **Grid ghi sai số tổ hợp**: test so số tổ hợp in ở mục 5 với `len(MODELS[m]["grid"])` (300, 30, 24, 16).
- **Chạy lại `simple_split` làm bản nháp cũ đi**: bản nháp ghi ngày sinh và lệnh tạo lại; test chạy được khi CSV vừa được ghi lại.

---

### Task 1: Script sinh mục 5 (thiết lập baseline)

**Files:**
- Create: `simple_split/src/report_sections.py`

**Interfaces:**
- Consumes: `simple_split.MODELS`, `simple_split.SEEDS`, `simple_split.load()` (cho ngày đầu/cuối và số ngày của train/val/test).
- Produces: `section5() -> str` (markdown).

- [ ] **Step 1: Viết `_tests()`**: `section5()` chứa `"973"`, `"1460"`, `"729"`; chứa chuỗi số tổ hợp `"300"`, `"30"`, `"24"`, `"16"` ở đúng dòng LR, SVM RBF, RF, LSTM; chứa `"42, 43, 44"`; chứa `"B4 XGBoost: chưa chạy"`; không chứa `"On-chain"` ngoài câu ghi chú cố định `NOTE_ONCHAIN`.
- [ ] **Step 2:** `python simple_split/src/report_sections.py --test` → FAIL.
- [ ] **Step 3:** Cài `section5()`. Nội dung cố định theo thứ tự: (a) bảng cách chia (từ `load()`); (b) bảng mô hình: ID (B0, B1 LR, B2 RF, B3 SVM RBF, B5 LSTM), "trước tune" (từ `MODELS[m]["before"]`), grid (tên tham số và giá trị, từ `MODELS[m]["grid"]`), số tổ hợp, seed; (c) quy tắc chọn: log-loss validation, làm tròn 4 chữ số, hoà thì tổ hợp đứng trước; học lại train + val; test 1 lần; (d) dòng `B4 XGBoost: chưa chạy ở cách chia này`; (e) `NOTE_ONCHAIN = "Kết quả có On-chain tạm hoãn cho đến khi baseline OHLCV ổn định (góp ý 1, 4)."`
- [ ] **Step 4:** `--test` PASS.

### Task 2: Script sinh mục 8 (kết quả trước/sau tuning)

**Files:**
- Modify: `simple_split/src/report_sections.py`

**Interfaces:**
- Consumes: CSV ở `simple_split/reports/`, `simple_split.rules`, `simple_split.load`.
- Produces: `section8() -> str`; `main()` ghi `docs/midterm_review3_baseline_sections.md` = tiêu đề + ngày + lệnh tạo lại + `section5()` + `section8()` + `edits()` (Task 3).

- [ ] **Step 1: Thêm vào `_tests()`**: mọi số DA và log-loss trong bảng 8.1 bằng `test_metrics.csv` sau làm tròn; tham số "được chọn" của mỗi mô hình bằng dòng `chosen` của `tuning_<model>.csv`; số tổ hợp có val log-loss < 0,6931 khớp tính lại từ CSV (LR 75, SVM 0, RF 12, LSTM 0 ở lần chạy hiện tại, nhưng test tính lại, không hard-code); mục 8 không chứa `"On-chain"`, `"Δ"`, `"XGBoost"`.
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài `section8()` gồm: 8.1 bảng so sánh trên test (DA, CI 95%, p so với 50%, log-loss, % đoán tăng, McNemar p so với đảo chiều) cho 4 luật + trước/sau tune 4 mô hình; 8.2 bảng tuning tóm tắt mỗi mô hình (tham số chọn, val log-loss, val DA, số tổ hợp tốt hơn 0,6931 / tổng); 8.3 xem dự báo: với mỗi mô hình sau tune một dòng số liệu (% đoán tăng, % cùng hướng luật đảo chiều, đúng x/10 ngày biến động mạnh nhất, ma trận nhầm lẫn 2×2) lấy từ `predictions_test.csv`; 8.4 nhận xét: tối đa 5 gạch đầu dòng, mỗi dòng có số (ví dụ "Mọi CI 95% của DA chứa 50%"), sinh từ dữ liệu bằng điều kiện (CI chứa 0,5? sau tune DA thấp hơn trước tune?), không câu nào không có số.
- [ ] **Step 4:** `--test` PASS; chạy `python simple_split/src/report_sections.py` → file nháp được ghi.

### Task 3: Danh sách sửa các mục khác + kiểm tra cuối

**Files:**
- Modify: `simple_split/src/report_sections.py` (hàm `edits() -> str`, văn bản cố định)

**Interfaces:**
- Produces: `edits()`; bản nháp đầy đủ `docs/midterm_review3_baseline_sections.md`.

- [ ] **Step 1: Thêm vào `_tests()`**: `edits()` có đủ các mục `1`, `4.2`, `4.3–4.4`, `7`, `9`, `10`, `Phụ lục B`, `Phụ lục C`, `Bảng 4.1`; không chứa con số kết quả nào ngoài số đã có trong CSV (kiểm bằng regex: mọi số thập phân dạng `dd,dd` trong `edits()` phải xuất hiện trong `section8()`).
- [ ] **Step 2:** `--test` → FAIL.
- [ ] **Step 3:** Cài `edits()`: mỗi mục một dòng "hiện viết gì → sửa thành gì", đúng nội dung đã nêu với Ngọc (mục 1 thêm câu phạm vi giữa kỳ; 4.2 thêm cách chia cố định; 4.3–4.4 chuyển thành kế hoạch; 7: E0(b),(c) xong, E1–E3 tạm hoãn; 9 bỏ câu "bản chạy thử", "B5 không tune"; 10 thay kết luận On-chain bằng kết luận baseline; bỏ Phụ lục B; Phụ lục C thay bằng grid ở mục 5; Bảng 4.1: MCC đã có).
- [ ] **Step 4:** `--test` PASS; chạy `main()`; đọc lại bản nháp, đối chiếu bảng 8.1 với `test_metrics.csv` bằng mắt một lần.
- [ ] **Step 5:** Không commit (nhánh `feature/midterm02`); báo Ngọc đường dẫn bản nháp.
