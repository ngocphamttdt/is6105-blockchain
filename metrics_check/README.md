# metrics_check: kiểm tra metric (góp ý 2 của thầy)

> Kiểm tra lại định nghĩa, cách tính và ý nghĩa của các metrics. Đối chiếu bằng một vài ví dụ cụ thể, bảo đảm các bảng đang đo đúng bài toán dự báo hướng giá.

**Kết quả: [`reports/metrics_check.md`](reports/metrics_check.md)** (đọc bảng Tóm tắt đầu file trước).

Bước này đi sau `baseline_check/` (góp ý 1). Khác với ví dụ tính tay ở `baseline_check` và `eda_pre_training` (kiểm code riêng hoặc dữ liệu giả), ở đây kiểm **file dự báo chính thức** trong `ngoc_pipeline/results/predictions/` và **đúng hàm tạo bảng** `run_all.metrics`. Mọi phép kiểm là `assert`: sai thì script dừng, không ghi báo cáo.

| Phần | Kiểm gì |
|---|---|
| 1 | `y_true` của mọi dòng, mọi file có đúng là 1[close_{t+1} > close_t] |
| 2 | Định nghĩa, mốc đoán bừa, chiều tốt, chiều của Δ cho DA, log-loss, Brier, AUC |
| 3 | Tính tay trên 8 ngày thật; trên đủ 2.434 ngày khớp `run_all.metrics` và `e1_delta_draft.csv` |
| 4 | Δ tính lại từ hai file khớp bảng; ΔDA tách thành số ngày chỉ một bên đúng |
| 5 | Ô không có nghĩa: log-loss, Brier của luật; AUC của Majority (đã sửa thành ô trống) |
| 6 | Backtest dùng p̂ ngày t cho vị thế ngày t+1 |
| 7 | `e1_by_year`, `e1_vs_reversal`, `e1_delta_draft` và `baseline_check` cho cùng con số |

## Chạy lại

Từ thư mục gốc repo, sau khi đã có predictions và bảng (`python ngoc_pipeline/src/run_all.py --stage tables`):

```
python metrics_check/src/metrics_check.py
```
