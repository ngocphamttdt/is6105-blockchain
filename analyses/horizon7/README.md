# horizon7 — Dự báo hướng giá 7 ngày tới: On-chain có giúp ích không? (task 6, Vi)

Thí nghiệm giống E1 bỏ Flows (`pipeline/results/no_flows`) nhưng nhãn là hướng giá 7 ngày tới,
`y_t = 1[close_{t+7} > close_t]`. Gap train/test và gap của inner CV tăng từ 1 lên 7 ngày; mọi thứ khác giữ nguyên
(đặc trưng, lịch fold, grid, tiêu chí tune). Báo cáo: [`reports/horizon7.md`](reports/horizon7.md).

**Kết quả:** On-chain không cải thiện ở tầm 7 ngày. ΔDA của cả 5 mô hình có CI chứa 0 (4/5 âm), log-loss
tăng ở cả 5 mô hình, và Δ không nhất quán theo năm.

## Chạy (từ gốc repo)

```bash
python analyses/horizon7/src/horizon7.py --test                       # test nhãn, gap, luật (~5 giây)
python analyses/horizon7/src/horizon7.py --stage run [--models ...]   # walk-forward, mặc định b0maj b0pers b0rev b1 b1t b2 b2t b3t
python analyses/horizon7/src/horizon7.py --stage tables               # bảng + hình
python analyses/horizon7/src/horizon7.py --stage forecast             # dự báo 7 ngày sau ngày cuối của freeze
```

## Đầu ra

| File | Nội dung |
|---|---|
| `results/predictions/{model}_{config}.csv` | p̂ từng ngày, cùng định dạng với pipeline (`date, fold, y_true, p_hat, regime`) |
| `results/tune_log/` | Tham số được chọn ở mỗi lần tune |
| `results/tables/metrics.csv` | DA, log-loss, Brier, AUC của mỗi mô hình × cấu hình |
| `results/tables/delta_onchain.csv` | Δ On-chain: CI block bootstrap (khối 30), DM (lag 14), Holm, mẫu không chồng lấn |
| `results/tables/delta_by_year.csv`, `consistency_by_year.csv` | Δ từng năm có CI; đếm số năm Δ > 0 |
| `results/tables/delta_by_regime.csv` | Δ theo regime HMM |
| `results/tables/vs_baselines.csv` | Mô hình so với Majority, Persistence 7d, Đảo chiều 7d |
| `results/tables/h1_vs_h7.csv` | Δ ở tầm 1 ngày (no_flows) và 7 ngày, cùng cách tính |
| `results/tables/forecast_next7.csv` | Dự báo cho 25–31/08/2026 (chưa chấm được) |
| `reports/figures/` | `delta_h1_vs_h7.png`, `delta_da_by_year_h7.png` |
