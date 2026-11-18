# Config mô hình

Mỗi file `{mã}.yaml` khai báo một mô hình. **Tên file chính là mã mô hình**: `--models b2t` đọc `b2t.yaml`, và dự báo được ghi ra `results/predictions/b2t_{config}.csv`.

## Quy ước đặt tên: mã nào đã tune

| Mức tune | Trường `tuning` | Cách nhận ra từ tên | Ý nghĩa |
|---|---|---|---|
| Không tune | `none` | | Luật hoặc tham số cố định, không có grid |
| Grid gốc | `original_grid` | không có hậu tố | Tune bằng grid đã ghi trong proposal / Phụ lục C |
| Đã tune lại | `retuned` | **hậu tố `t`** | Grid được thiết kế lại hoặc mở rộng, chỉ dựa trên train/validation |

Bản tune lại **không thay** bản gốc mà nằm cạnh nó (`b2` và `b2t`, `lr` và `lrt`). Nhờ vậy các kết quả cũ vẫn tái lập được. Test trong `models.py` kiểm quy ước này: mã kết thúc bằng `t` khi và chỉ khi `tuning: retuned`.

Mã mô hình **không chứa dấu `_`**, vì file dự báo `{mã}_{config}.csv` được tách theo dấu `_` đầu tiên.

## Danh sách

| Baseline | Không tune | Grid gốc | Đã tune lại |
|---|---|---|---|
| **B0** | `b0maj` Majority<br>`b0pers` Persistence<br>`b0rev` Đảo chiều | | |
| **B1** Logistic Regression | | `lr` C ∈ {0,01; 0,1; 1; 10}<br>`lrfs` + chọn đặc trưng tiến dần | `lrt` 5 tập đặc trưng × 10 mức C × L1/L2/elastic-net × class_weight (300 tổ hợp) |
| **B2** Random Forest | | `b2` số cây × độ sâu × `min_samples_leaf` | `b2t` `min_samples_leaf` × `max_features` × `max_samples` |
| **B3** SVM RBF | | `b3` C × gamma | |
| **B4** XGBoost | | `b4` learning_rate × max_depth | |
| **B5** LSTM | `b5` tham số cố định, 3 seed | | |

"Không tune" ở B0 nghĩa là không có grid. `b0rev` vẫn chọn chiều của luật trên cửa sổ train, nhưng đó là một phần của luật chứ không phải tuning. B5 chỉ dùng early stopping trên 20% cuối cửa sổ train.

## Nội dung một file

```yaml
baseline: B2               # thuộc baseline nào (B0–B5)
tuning: retuned            # none | original_grid | retuned
class: b2                  # lớp mô hình: khoá trong REGISTRY của src/models.py
params: {...}              # tham số cố định
param_grid: {...}          # grid để tune: inner TimeSeriesSplit(5, gap 1), log-loss, tune lại mỗi 6 fold
```

Ba loại không chạy qua `GridSearchCV` chung của `walkforward.py`:

| `class` | Mã | Chạy bằng |
|---|---|---|
| `rule` | `b0pers`, `b0rev` | `run_rule()` trong `src/models.py`. `rule:` là tên luật, `params:` là tham số của luật |
| `lrt` | `lrt` | `src/lr_tuning.py`, vì ngoài tham số mô hình còn tune cả tập đặc trưng (`feature_sets`) và có 2 tiêu chí chọn (`criteria`) |
| (không có) | `b5` | `run_all.py`: chạy mỗi seed trong `seeds` một lần rồi lấy trung bình p̂ |

## Thêm mô hình hoặc grid mới

Tạo file mới thay vì sửa file cũ. Sửa grid trong file cũ sẽ làm kết quả cũ không còn tái lập được. Bản tune lại của một mô hình đặt tên là mã cũ + `t`, và ghi `tuning: retuned`.
