# Thiết kế: `analyses/simple_split/` – chạy 5 baseline trên cùng một cách chia train / validation / test

Ngày: 13/11/2026 · Kích thước: thay đổi nhỏ (một script mới, dùng lại code có sẵn)

## Mục đích

Theo ý 1 của thầy: chạy baseline cơ bản trước, xem trực tiếp dự báo đúng/sai, bảo đảm kết quả có ý nghĩa rồi mới
mở rộng. Ngọc muốn mọi baseline chạy **cùng một cách đơn giản** để so sánh dễ. Walk-forward 82 fold hiện có
giữ nguyên, dùng để đối chiếu sau.

## Mục tiêu

- 5 nhóm mô hình (B0, LR, SVM RBF, RF, LSTM) chạy trên **cùng** cách chia, cùng dữ liệu, cùng đặc trưng, cùng metric.
- Mỗi mô hình: tune trên validation, test chạy **đúng 1 lần**, có bảng trước/sau tuning.
- Mỗi mô hình có phần **xem trực tiếp dự báo**.
- Một bảng so sánh chung, có mốc luật đảo chiều.

**Không làm:** không thay walk-forward; không thêm On-chain; không sửa code của người khác.

**Thất bại nếu:** có mô hình nào nhìn test khi chọn tham số, hoặc hai mô hình chạy khác cách chia hay khác đặc trưng.

## Ràng buộc chung (mọi mô hình)

| Hạng mục | Chốt |
|---|---|
| Dữ liệu | Data freeze v1.0 (`baseline_check.load_data`, có kiểm sha256) |
| Đặc trưng | `FEATURES_10` (10 đặc trưng OHLCV), import tường minh, không dùng `FEATURES` (11 cột) |
| Cách chia | `baseline_check.split`: train 2018-01-01 → 2021-12-30 · val 2022-01-01 → 2023-12-30 · test 2024-01-01 → 2026-08-30; ngày cuối train và val bị bỏ (gap 1 ngày) |
| Chuẩn hoá | Fit chỉ trên phần dữ liệu đang học (train khi tune; train + val khi học lại) |
| Chọn tham số | Log-loss trên validation 2022–2023; hoà (làm tròn 4 chữ số) thì chọn tổ hợp đơn giản hơn |
| Mô hình cuối | Học lại trên train + val với tham số đã chọn, chấm test 1 lần |
| Seed | Chỉ RF và LSTM có ngẫu nhiên: danh sách chung `[42, 43, 44]`, mỗi tổ hợp lấy trung bình p̂ của 3 seed, cả khi tune lẫn khi chấm test. Không chọn seed theo kết quả |
| Metric | `baseline_check.metrics` (DA, CI, p so với 50%, log-loss, Brier, AUC, % đoán tăng) + McNemar so với luật đảo chiều |

## Từng mô hình (grid lấy đúng của người phụ trách)

| Mô hình | "Trước tune" (đặt trước) | Grid "sau tune" | Nguồn |
|---|---|---|---|
| B0: Majority, Persistence, Buy-and-hold, Đảo chiều | – (không tham số; chiều đảo chọn trên train) | – | `baseline_check.rule_predictions` |
| LR | L2, C = 1, 10 đặc trưng | 5 tập đặc trưng × C (10 giá trị) × l1_ratio {0; 0,5; 1} × class_weight {None, balanced} = 300; winsorize 1%/99% + scaler | Vi: `ElasticNetLR`, `feature_sets`, `GRID`, `CLIP_Q` trong `b1t_tuning.py` |
| SVM RBF | C = 1, gamma = "scale" | C {0,01; 0,1; 1; 10; 100} × gamma {0,001; 0,003; 0,01; 0,03; 0,1; 0,3} = 30 | Trâm: `CalibratedSVC` (`models.py`), `GRID_WIDE` |
| RF | sklearn mặc định + `n_estimators` 500 | min_samples_leaf {50, 100, 150, 200} × max_features {sqrt; 0,33; 0,5} × max_samples {None; 0,5} = 24; 500 cây | Ngân: `configs/models/b2t.yaml` |
| LSTM | lookback 30, hidden 32, dropout 0, lr 1e-3 | lookback {10, 30} × hidden {8, 32} × dropout {0; 0,2} × lr {1e-3; 1e-4} = 16 | Ngọc: `lstm_tuning.fit_predict` (early stopping trong 20% cuối phần đang học, có gap) |

## Đầu ra (`analyses/simple_split/reports/`)

- `simple_split.md`:
  1. **Bảng so sánh chung** trên test (trước và sau tune của mọi mô hình + 4 luật B0).
  2. **Bảng tuning từng mô hình**: mọi tổ hợp với train/val log-loss và DA, đánh dấu tổ hợp được chọn.
  3. **Xem dự báo từng mô hình** (bản sau tune): ma trận nhầm lẫn; 10 ngày ngẫu nhiên; 8 ngày sai tự tin nhất;
     10 ngày biến động mạnh nhất; DA theo mức biến động; mức đồng ý với luật đảo chiều.
  4. Lưu ý về độ tin cậy: 973 ngày test → sai số DA khoảng ±1,6 điểm.
- `test_metrics.csv`, `tuning_<model>.csv`, `predictions_test.csv` (p̂ từng ngày của mọi mô hình).

Chạy: `python analyses/simple_split/src/simple_split.py` (ước tính dưới 15 phút trên CPU), `--quick` để chạy thử.

## Các cách đã cân nhắc

1. **Một script chung cho cả 5 mô hình** ← đã chọn. Bảo đảm cùng cách chia và metric; grid vẫn của từng người.
2. Mỗi người tự thêm cách A vào thư mục mình, dùng chung một file tiện ích. Đúng phân công nhưng dễ lệch chi tiết,
   phải chờ đủ người.
3. Chỉ lấy số test, không có phần xem dự báo. Loại vì thiếu đúng phần thầy yêu cầu.

## Quyết định của Claude (cần Ngọc kiểm tra)

- "Trước tune" của RF và SVM là tham số mặc định của sklearn (RF thêm 500 cây cho ổn định).
- LR "trước tune" là C = 1, L2, 10 đặc trưng, giống `baseline_check`.
- Tune bằng một validation (2022–2023) thay vì inner CV, vì đây là cách chia đơn giản; LR của Vi vốn dùng inner CV
  trong walk-forward, ở đây dùng cùng grid nhưng chọn trên validation.
- Không chạy lại phần kiểm tra 10 seed ở bước này.
