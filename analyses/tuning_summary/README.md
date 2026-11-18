# tuning_summary — Trước và sau tuning của mọi baseline (B0–B5)

**Kết quả: [`reports/tuning_summary.md`](reports/tuning_summary.md)** · bảng đầy đủ: `results/tuning_summary.csv`

Gom vào một bảng mọi lần tuning mà nhóm đã chạy. Các lần tuning này đang nằm rải ở nhiều nơi (`pipeline/results/tuning/`, `b1t/`, `gate/`, `analyses/b2_tuning`, `analyses/baselines`, `analyses/lstm_tuning`), mỗi nơi một cách đặt tên.

Script **chỉ đọc các file dự báo đã có** và không chạy lại mô hình nào. Mọi metric được tính lại theo cùng một cách trên cùng 2.434 ngày test walk-forward, nên các dòng so được trực tiếp với nhau.

## Ba mức tune

Trùng với trường `tuning` trong `pipeline/configs/models/*.yaml`:

| Mức | Nghĩa | Nguồn dự báo |
|---|---|---|
| chưa tune (`none`) | Luật B0, tham số mặc định của sklearn, hoặc tham số cố định (B5) | `pipeline/results/predictions/b0*`, `b5*`; `pipeline/results/tuning/default/` |
| grid gốc (`original_grid`) | Tune bằng grid gốc (Phụ lục C) | Dự báo E1: `pipeline/results/predictions/` |
| tune lại (`retuned`) | Grid thiết kế lại hoặc mở rộng | `b1t`, `b2t`, grid mở rộng của `gate.py`, grid 5×6 của Trâm, `lstm_tuning` của Ngọc |

## Các cột chính

- `da`, `log_loss`, `brier`, `auc`, `pct_pred_up`: tính bằng `run_all.metrics`, cùng hàm với bảng E1. Log-loss và Brier của luật B0 để trống vì p̂ chỉ là 0/1, AUC của `b0maj` cũng để trống.
- `so_với`, `d_da`, `d_logloss` (kèm `_lo`/`_hi`), `p_mcnemar`, `p_dm`: so dòng hiện tại với bản **chưa tune cùng code**. CI 95% bằng block bootstrap 15 ngày (`stats.paired_row`).
- `d_da_vs_b0rev`: ΔDA so với luật đảo chiều, mốc mà chưa mô hình có học nào vượt được.
- `nguồn`, `ghi_chú`: file dự báo của từng dòng và điểm khác biệt về cách chạy.

## Cần lưu ý khi đọc

- **B4 không tái lập được.** Bước tuning (`experiments.py --stage tuning`) chạy lại cùng grid gốc nhưng cho dự báo lệch 312–359 ngày quyết định so với dự báo E1, nhiều khả năng do khác máy hoặc phiên bản `xgboost`. Bảng dùng dự báo E1.
- **Grid mở rộng của `gate.py` chỉ có OHLCV**, và có thêm winsorize 1%/99%. Δ so với bản mặc định vì thế gồm cả tác động của winsorize.
- **B5 có hai bộ code.** `b5` là dự báo E1 (`regime/model_lstm.py`, không tune). `b5 lstm_tuning trước/sau tune` là code riêng của Ngọc, chỉ chạy OHLCV, và được so với nhau chứ không so với `b5`.

## Chạy lại

```bash
python analyses/tuning_summary/src/tuning_summary.py          # ~1 phút, ghi results/ và reports/
python analyses/tuning_summary/src/tuning_summary.py --test   # chỉ chạy test
```

Cần chạy lại khi có thêm hoặc có thay đổi file dự báo của một bản tuning. Muốn thêm một bản tune mới thì thêm một dòng vào `rows_spec()`.
