# Tuning B2 Random Forest (`b2t`)

**Cập nhật:** 13/11/2026 · Phương án Flows A · 82 fold walk-forward · 2.434 ngày test (01/01/2020 → 30/08/2026)

Thư mục này tune lại siêu tham số cho B2 Random Forest. Mọi thứ vẫn theo đúng protocol của `pipeline`: cùng fold, cùng `TimeSeriesSplit(5, gap 1)`, tiêu chí log-loss, tune lại mỗi 6 fold, và cùng một grid cho cả hai cấu hình. Bản đã tuning được thêm vào pipeline thành mô hình mới **`b2t`**. `b2.yaml` gốc vẫn giữ nguyên, nên kết quả hiện có không thay đổi.

## 1. Vấn đề của grid cũ

Grid cũ (`configs/models/b2.yaml`, Phụ lục C) có `n_estimators` {200, 500}, `max_depth` {3, 6, None} và `min_samples_leaf` {20, 50}. Nó có ba điểm yếu:

1. **`min_samples_leaf` chạm biên.** Ở OHLCV, cả 14/14 lần tune đều chọn 50, là giá trị lớn nhất của grid. Ở cấu hình đầy đủ là 13/14 lần. Nghĩa là grid chưa phủ tới vùng tối ưu.
2. **Thiếu `max_features`**, tham số quan trọng nhất của RF, nên luôn dùng mặc định `sqrt`.
3. **Tune `n_estimators` là thừa.** Với RF, thêm cây chỉ làm giảm phương sai chứ không gây overfit, nên tune tham số này chỉ tốn thời gian.

## 2. Cách thiết kế grid mới (không chạm dữ liệu test)

Grid mới chỉ được chọn bằng inner CV trên **cửa sổ train của fold 0 (31/12/2017 → 30/12/2019)**, toàn bộ trước ngày test đầu tiên. Bước này dùng `src/explore_pre2020.py`, cho 150 tổ hợp × 2 cấu hình, kết quả nằm ở `results/explore_pre2020_grid.csv`.

Những gì rút ra từ dữ liệu trước 2020:
- Log-loss giảm đều khi lá lớn dần: OHLCV tốt nhất ở 150, On-chain tốt nhất ở 100.
- Khi lá đã lớn, `max_depth` 3, 6 hay None cho log-loss giống hệt nhau, nên bỏ khỏi grid.
- Với On-chain, `max_features` = 0,33 tốt hơn `sqrt`.
- `max_samples` = 0,5 (bagging nửa cửa sổ) giúp ở OHLCV.
- Đã thử đặt `min_samples_leaf` theo tỷ lệ cửa sổ (`results/explore_pre2020_leaf_fraction.csv`), nhưng kết quả kém hơn dạng số tuyệt đối nên không dùng.

**Grid `b2t`** (`configs/models/b2t.yaml`), gồm 24 tổ hợp:

| Tham số | Giá trị |
|---|---|
| `min_samples_leaf` | 50, 100, 150, 200 |
| `max_features` | sqrt, 0,33, 0,5 |
| `max_samples` | None, 0,5 |
| Cố định | `n_estimators` 500, `max_depth` None, `random_state` 0 |

Trong walk-forward, tham số được chọn nằm rải khắp lòng grid và không còn dồn về biên. Chi tiết từng lần tune: `results/chosen_params_per_retune.csv`.

## 3. Kết quả trên 2.434 ngày test

Bảng: `results/b2_metrics_orig_vs_tuned.csv`. Dòng "gốc" dùng đúng file predictions `b2_*.csv` đang có trong repo. Log-loss được tune trực tiếp, thấp hơn là tốt hơn. Mức sàn `b0maj` có log-loss 0,6938.

| Cấu hình | Grid | DA | Log-loss | Brier | AUC | % ngày dự báo tăng | Backtest long/flat |
|---|---|---|---|---|---|---|---|
| OHLCV | gốc | 0,5123 | 0,6961 | 0,2514 | 0,514 | 54,7% | +241%, Sharpe 0,62 |
| OHLCV | **b2t** | 0,5115 | **0,6947** | **0,2507** | **0,517** | 54,5% | +229%, Sharpe 0,61 |
| OHLCV + On-chain | gốc | 0,5140 | 0,6957 | 0,2512 | 0,515 | 54,1% | +14%, Sharpe 0,29 |
| OHLCV + On-chain | **b2t** | 0,4922 | **0,6951** | **0,2509** | 0,506 | 49,2% | −61%, Sharpe −0,05 |

Chênh lệch b2t − gốc (`results/b2_tuned_minus_orig.csv`; CI 95% block bootstrap 15 ngày, 2.000 mẫu):

| Cấu hình | Δlog-loss [CI 95%] | ΔDA [CI 95%] | McNemar |
|---|---|---|---|
| OHLCV | −0,0014 [−0,0029; +0,0000] | −0,08 điểm [−1,44; +1,31] | p = 0,95 |
| OHLCV + On-chain | −0,0006 [−0,0025; +0,0012] | −2,18 điểm [−3,53; −0,78] | p = 0,004 |

Δ On-chain trong từng grid (`results/b2_delta_onchain.csv`):

| Grid | ΔDA [CI 95%] | Δlog-loss [CI 95%] | McNemar (n01 / n10) |
|---|---|---|---|
| gốc | +0,16 điểm [−1,77; +2,01] | −0,0005 [−0,0036; +0,0024] | 313 / 309, p = 0,90 |
| b2t | −1,93 điểm [−3,70; −0,21] | +0,0004 [−0,0027; +0,0029] | 271 / 318, p = 0,058 |

## 4. Nhận xét

1. **Tuning đạt mục tiêu của nó.** Log-loss là tiêu chí tune, và nó giảm ở cả hai cấu hình. Ở OHLCV mức giảm (−0,0014) có CI chạm 0; Brier và AUC cũng tốt lên.
2. **Vẫn chưa thắng mức sàn `b0maj`** (0,6938). Lá lớn kéo p̂ về gần tỷ lệ ngày tăng của train: độ lệch chuẩn của p̂ giảm từ 0,05 xuống 0,04. Điều này cho thấy RF gần như không tìm được tín hiệu hướng giá ngày t+1, khớp với nhận xét 7.4 trong `PROJECT_FLOW.md`.
3. **DA không cải thiện, và giảm ở cấu hình On-chain.** Khi p̂ dồn sát 0,5, chỉ cần dịch rất nhỏ cũng làm quyết định "> 0,5" đổi chiều. Bản b2t + On-chain dự báo tăng 49% số ngày, ít hơn tỷ lệ ngày tăng thực tế (~51%), nên DA và backtest đều tệ hơn. Đây là hệ quả của việc tối ưu log-loss chứ không phải tối ưu DA.
4. **Ảnh hưởng tới H1 (cần nhóm quyết định).** Với b2t, log-loss của cấu hình đầy đủ là 0,6951, thấp hơn mọi mô hình có học hiện có (`b4` 0,6958, `b2` 0,6957). Theo quy tắc ghi sẵn trong proposal, **b2t sẽ thành mô hình tốt nhất**. ΔDA của nó âm (−1,93 điểm, CI bootstrap không chứa 0 nhưng McNemar p = 0,058), nên kết luận vẫn là **không chấp nhận H1**, chỉ là chiều Δ đảo từ "≈ 0" sang "hơi âm".
5. **Về tính công bằng.** Grid mới được thiết kế chỉ từ dữ liệu trước 2020 và dùng chung cho hai cấu hình. Tuy vậy, quyết định tune lại được đưa ra sau khi đã thấy kết quả E1. Đề xuất trình bày `b2t` như **phân tích độ nhạy theo grid** (Δ của RF thay đổi thế nào khi đổi grid), còn bảng chính giữ grid đã ghi trong Phụ lục C. Cách còn lại là cập nhật Phụ lục C và ghi rõ lý do.

## 5. Chạy lại

Mô hình `b2t` đã có trong pipeline:

```bash
python pipeline/src/run_all.py --stage e1 --models b2t          # khoảng 55 phút trên 2 lõi CPU
python pipeline/src/run_all.py --stage tables                   # bảng E1 đã có dòng b2t
```

Các script khảo sát và so sánh (chạy từ thư mục gốc repo):

```bash
python analyses/b2_tuning/src/explore_pre2020.py . analyses/b2_tuning/results/explore_pre2020_grid.csv            # khảo sát trước 2020
python analyses/b2_tuning/src/rf_runner.py --repo . --grid pipeline/configs/models/b2.yaml  --out runs/orig
python analyses/b2_tuning/src/rf_runner.py --repo . --grid pipeline/configs/models/b2t.yaml --out runs/tuned_ohlcv --configs ohlcv
python analyses/b2_tuning/src/rf_runner.py --repo . --grid pipeline/configs/models/b2t.yaml --out runs/tuned_onchain --configs ohlcv_onchain
python analyses/b2_tuning/src/compare.py . runs analyses/b2_tuning/results                                         # bảng so sánh
```

`rf_runner.py` gọi đúng `walkforward.run_walkforward` của repo và chỉ ghi thêm tham số được chọn ở mỗi lần tune. Kết quả của nó khớp với `run_all.py --stage e1 --models b2t`.

## 6. File thay đổi

| File | Thay đổi |
|---|---|
| `pipeline/configs/models/b2t.yaml` | Mới: grid đã tuning |
| `pipeline/src/models.py` | Thêm `b2t` vào docstring và test đọc config `b2t` |
| `pipeline/results/predictions/b2t_{ohlcv,ohlcv_onchain}.csv` | Mới: predictions 82 fold, đã ghép regime |
| `pipeline/results/tables/e1_*.csv` | Tính lại, thêm dòng `b2t` (các dòng khác giữ nguyên) |
| `analyses/b2_tuning/` | Mới: script, kết quả khảo sát, bảng so sánh, README này |
