# tram_metrics — Metric chính / phụ, log-loss trước và sau tune, OHLCV so với + On-chain (góp ý 3)

Phụ trách: Trâm. **Kết quả: [`reports/metrics_report.md`](reports/metrics_report.md)**. Mục 1 của báo cáo là đoạn văn "Metric" để đưa thẳng vào bài.

- Mô hình đã chốt (góp ý 1): LR (`b1` grid gốc → `b1t` tune lại) và RF (`b2` → `b2t`).
- Bộ đặc trưng chính: On-chain bỏ Flows (góp ý 4), dự báo ở `pipeline/results/no_flows/predictions/`.
- Đối chiếu: bộ còn Flows, ở `pipeline/results/rerun_2026-11-19/predictions/`.
- Kiểm định dùng đúng `pipeline/src/stats.py`: CI 95% block bootstrap 15 ngày, McNemar cho ΔDA, Diebold–Mariano cho Δ log-loss, Holm.

Chỉ đọc file dự báo có sẵn, không chạy lại mô hình (vài giây; cần pandas, numpy, scipy, matplotlib):

```
python analyses/tram_metrics/src/metrics_report.py
```

| File trong `results/` | Nội dung |
|---|---|
| `metrics_no_flows.csv` | DA, log-loss, Brier, % đoán tăng của từng mô hình × cấu hình × giai đoạn tune |
| `tuning_before_after_no_flows.csv` | Δ = tune lại − grid gốc, kèm CI, McNemar, DM, Holm |
| `delta_onchain_no_flows.csv` | Δ = (+On-chain) − OHLCV-only của mô hình đã tune lại |
| `vs_benchmarks_no_flows.csv` | So với luôn tăng, luật đảo chiều, B0 Majority |
| `metrics_with_flows.csv`, `delta_onchain_with_flows.csv` | Như trên, cho bộ còn Flows |
