# Tái tạo Hafid et al. 2024 (arXiv:2410.06935)

## Dữ liệu (Task 1)
- Nguồn: Binance Data Vision, BTCUSDT nến 15 phút, 12 file tháng 2021-02 .. 2022-01 (`data/raw/`, kiểm `.CHECKSUM` sha256).
- `data/btcusdt_15m.csv`: cột `open_time` (UTC), open, high, low, close, volume. `data/manifest.json`: sha256, số dòng, số nến thiếu.
- Bài báo: `paper/2410.06935v2.pdf`.
- Nến thiếu: không nội suy; chỉ đếm so với 365 x 96 = 35040 nến.
- `load_15m()` trong `src/download.py`: index `open_time` kiểu UTC naive (không gắn tz), float, tăng dần, không trùng.

## Chạy
    python3 experiment_paper/hafid2024/src/download.py          # tải + ghép + manifest
    python3 experiment_paper/hafid2024/src/download.py --test   # test

## Thí nghiệm (Task 3)
    python3 experiment_paper/hafid2024/src/paper.py --test   # test chỉ báo/nhãn/mô hình
    python3 experiment_paper/hafid2024/src/run.py            # chạy A/B/C x {full,train} x {XGBoost,LR} (~5 giây)
    python3 experiment_paper/hafid2024/src/run.py --test     # kiểm tra results.csv và report.md
- Đầu ra: `reports/results.csv` (12 dòng, gồm majority baseline, n_test, giai đoạn train/test, 8 đặc trưng), `reports/report.md` (tiếng Việt, sinh tự động từ results.csv).
- A: 15 phút + nhãn bài; B: 15 phút + nhãn nến kế tiếp; C: OHLCV ngày của nhóm (`data/processed/master_daily.csv`, 2018-01-01 → 2026-08-30) + nhãn ngày mai.
