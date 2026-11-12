# Module LSTM và HMM

## 1. Mục đích

Thư mục này chứa các thành phần phục vụ hai phần của đồ án:

-   **LSTM**: mô hình học sâu dự báo hướng biến động giá Bitcoin ngày kế
    tiếp.
-   **HMM (Hidden Markov Model)**: phân loại trạng thái thị trường thành
    `Bear`, `Sideways` và `Bull` theo phương pháp walk-forward.
-   **Bản đồ regime**: kết quả phân loại regime theo ngày và fold, dùng
    để tra cứu hoặc ghép với kết quả dự báo của mô hình khác.
-   **Biểu đồ timeline**: hình minh họa diễn biến regime trong giai đoạn
    đánh giá.

Các file này bổ sung cho pipeline hiện có của nhóm. Trước khi dùng làm
kết quả cuối cùng, cần bảo đảm chúng được chạy với cùng dữ liệu, target,
lịch fold và quy ước feature chính thức của repository.

## 2. Danh sách file

  ----------------------------------------------------------------------------------------------
  File                            Vai trò           Đầu vào chính           Đầu ra
  ------------------------------- ----------------- ----------------------- --------------------
  `model_lstm.py`                 Định nghĩa và     DataFrame có ngày,      `date`, `fold`,
                                  chạy LSTM cho một target, các feature;    `y_true`, `p_hat`,
                                  fold hoặc nhiều   đối tượng fold của      `regime`
                                  folds             `walkforward.py`        

  `regime_hmm.py`                 Huấn luyện HMM và Dữ liệu ngày có `date`, Regime theo
                                  phân loại regime  `log_return`,           ngày/fold, xác suất
                                  theo walk-forward `realized_volatility`   state và bảng thống
                                                                            kê fold

  `regime_map.csv`                Bản đồ regime đã  Kết quả từ quy trình    2.435 dòng, 3 cột:
                                  xuất sẵn          HMM                     `date`, `fold`,
                                                                            `regime`

  `bitcoin_regime_timeline.png`   Hình timeline     `regime_map.csv` hoặc   Hình phân bố
                                  trực quan hóa     kết quả regime tương    Bear/Sideways/Bull
                                  regime            ứng                     theo thời gian
  ----------------------------------------------------------------------------------------------

## 3. `model_lstm.py` --- Mô hình LSTM

### Chức năng

`model_lstm.py` định nghĩa `LSTMClassifier` và các hàm:

-   `run_lstm_fold(...)`: huấn luyện và dự báo trên một fold.
-   `run_lstm(...)`: lặp qua danh sách folds, ghép kết quả thành một
    bảng.

### Cấu hình mặc định

-   Lookback: 30 ngày.
-   Hidden size: 32.
-   Số LSTM layers: 1.
-   Số epoch tối đa: 50.
-   Early stopping patience: 5 epoch.
-   Validation: 20% cuối của tập train theo thứ tự thời gian.
-   Loss: `BCEWithLogitsLoss`.
-   Optimizer: Adam, learning rate `0.001`.
-   Random seed mặc định: 42.
-   Scaler: `StandardScaler`, fit trên feature thuộc cửa sổ train.

Mô hình tạo chuỗi đầu vào 30 ngày và dùng target ở ngày kết thúc chuỗi
để dự báo hướng giá ngày kế tiếp. `p_hat` là xác suất dự báo lớp tăng do
sigmoid chuyển từ logit sang xác suất.

### Phụ thuộc

`model_lstm.py` import các hằng số `DATE`, `TARGET`, `PREDICTION_COLS`
từ `data.py` và hàm `fold_indices` từ `walkforward.py`. Vì vậy, file này
cần được đặt trong cùng môi trường/module path với các file pipeline đó.

### Điều kiện dữ liệu

-   Dữ liệu phải được sắp xếp theo ngày và không có ngày trùng.
-   Code hiện tại yêu cầu ngày liên tục, không tự điền ngày bị thiếu.
-   Các cột trong `feature_cols` phải tồn tại trong DataFrame.
-   Target cần tuân theo quy ước của pipeline nhóm: `y_t = 1` khi giá
    đóng cửa ngày `t+1` cao hơn ngày `t`.

### Ví dụ gọi hàm

Ví dụ dưới đây minh họa cách gọi sau khi đã tạo `lstm_data`, `folds` và
`feature_cols` theo đúng pipeline của nhóm:

``` python
from model_lstm import run_lstm_fold

pred_fold0, best_val_loss = run_lstm_fold(
    data=lstm_data,
    fold=folds[0],
    feature_cols=feature_cols,
    lookback=30,
    hidden_size=32,
    num_layers=1,
    max_epochs=50,
    patience=5,
    seed=42,
)

pred_fold0.to_csv("lstm_ohlcv_fold0.csv", index=False)
```

Đây là ví dụ sử dụng module, không phải script chạy độc lập:
`lstm_data`, `folds` và `feature_cols` phải được tạo trước.

## 4. `regime_hmm.py` --- Phân loại trạng thái thị trường

### Chức năng

Module có ba thành phần chính:

-   `forward_filter(...)`: tính xác suất trạng thái theo thời gian bằng
    forward filtering, không dùng các quan sát tương lai để cập nhật
    trạng thái của ngày trước.
-   `run_hmm_fold(...)`: fit HMM trên tập train, ánh xạ ba state thành
    `Bear`, `Sideways`, `Bull`, rồi phân loại tập test.
-   `run_hmm_walk_forward(...)`: chạy nhiều fold liên tiếp và có thể lưu
    kết quả cùng bảng tổng kết.

### Đặc trưng đầu vào

Mặc định module yêu cầu:

-   `date`
-   `log_return`
-   `realized_volatility`

Có thể có thêm `ohlcv_close` để đính kèm trong kết quả.

Trong mỗi fold, `StandardScaler` chỉ fit trên train. HMM có ba state,
covariance dạng `diag`, tối đa 200 iterations; mặc định thử 5 lần khởi
tạo và chọn mô hình có log-likelihood train cao nhất trong các lần hợp
lệ.

Ba state được ánh xạ dựa trên lợi suất trung bình của các quan sát
train: thấp nhất thành `Bear`, ở giữa thành `Sideways`, cao nhất thành
`Bull`.

### Lưu ý

-   Module dùng **số dòng (row counts)** để chia cửa sổ, không tự chèn
    ngày bị thiếu.
-   Train/test cần được sắp xếp theo thời gian và các feature phải không
    có NaN hoặc vô cực.
-   `regime` là nhãn trạng thái thị trường, không phải xác suất dự báo
    giá tăng/giảm.
-   `regime_hmm.py` dùng thư viện `hmmlearn`; môi trường chạy cần cài
    thư viện này cùng với NumPy, pandas và scikit-learn.

### Ví dụ chạy walk-forward

``` python
from regime_hmm import run_hmm_walk_forward

regime_results, fold_summary, fold_metadata = run_hmm_walk_forward(
    data=hmm_data,
    first_test_date="2020-01-01",
    train_rows=730,
    test_rows=30,
    step_rows=30,
    gap_rows=1,
    feature_cols=("log_return", "realized_volatility"),
    n_components=3,
    n_init=5,
    random_state=42,
    out_dir="results/regime",
)
```

`hmm_data` phải được chuẩn bị trước và có các cột đầu vào được mô tả ở
trên.

## 5. `regime_map.csv` --- Bản đồ regime theo ngày

CSV hiện có các cột:

  Cột        Ý nghĩa
  ---------- -------------------------------------
  `date`     Ngày quan sát
  `fold`     Mã fold walk-forward, bắt đầu từ 0
  `regime`   Nhãn `Bear`, `Sideways` hoặc `Bull`

Bản CSV đi kèm có **2.435 dòng**, từ **2020-01-01 đến 2026-08-31**, với
`fold` từ **0 đến 81**. Các fold 0--80 có 30 ngày; fold cuối có 5 ngày.

### Ghép regime vào dự báo LSTM

Nên ghép bằng cả `date` và `fold` để tránh ghép nhầm ngày ở các fold
khác nhau:

``` python
import pandas as pd

pred = pd.read_csv("lstm_predictions.csv")
regime = pd.read_csv("regime_map.csv")

pred["date"] = pd.to_datetime(pred["date"], utc=True).dt.normalize()
regime["date"] = pd.to_datetime(regime["date"], utc=True).dt.normalize()

pred_with_regime = pred.drop(columns=["regime"], errors="ignore").merge(
    regime[["date", "fold", "regime"]],
    on=["date", "fold"],
    how="left",
    validate="one_to_one",
)

pred_with_regime.to_csv("lstm_predictions_with_regime.csv", index=False)
```

Sau khi ghép, cần kiểm tra các giá trị `regime` bị thiếu. Không tự gán
nhãn cho những ngày không có kết quả HMM.

## 6. `bitcoin_regime_timeline.png` --- Biểu đồ timeline

Biểu đồ thể hiện nhãn regime theo thời gian trong giai đoạn 2020--2026:

-   **Bear**: trạng thái gắn với mức lợi suất trung bình thấp nhất trong
    ba state train của từng fold.
-   **Sideways**: state có lợi suất trung bình ở giữa.
-   **Bull**: state có lợi suất trung bình cao nhất.

Biểu đồ giúp quan sát sự thay đổi regime và những giai đoạn trạng thái
chuyển đổi thường xuyên. Đây là hình trực quan hóa kết quả phân loại,
không phải bằng chứng độc lập rằng HMM dự báo chính xác hướng giá.

## 7. Cài đặt môi trường

Các thư viện chính được sử dụng trong những module này:

``` bash
pip install numpy pandas scikit-learn torch hmmlearn
```

Nếu repository đã có `requirements.txt` hoặc môi trường chuẩn của nhóm,
ưu tiên sử dụng cấu hình phụ thuộc trong repository để tránh xung đột
phiên bản.

## 8. Quy trình đề xuất

1.  Dùng `data.py` để nạp dữ liệu theo quy ước của nhóm.
2.  Dùng `features.py` để tạo target và đặc trưng chính thức.
3.  Dùng `walkforward.py` cùng cấu hình walk-forward của nhóm để tạo
    folds.
4.  Chạy `model_lstm.py` để sinh xác suất `p_hat` cho từng fold.
5.  Chạy `regime_hmm.py` để sinh nhãn regime cho cùng lịch đánh giá.
6.  Ghép LSTM và regime bằng `date` + `fold`.
7.  Kiểm tra số dòng, ngày, fold, giá trị thiếu và lưu kết quả vào thư
    mục results theo quy ước chung.

## 9. Trạng thái và giới hạn hiện tại

**Cập nhật theo góp ý 6 (Vi): `run_regime.py`.**

-   `regime_map.csv` giờ **tái lập được**. `run_regime.py` dựng đầu vào từ data freeze (`log_return` = log return giá đóng cửa ngày; `realized_volatility` = căn tổng bình phương log return 1h trong ngày UTC), gọi `regime_hmm.run_hmm_walk_forward` không sửa gì, và kiểm tra khớp 100% nhãn và fold với `regime_map.csv`.
-   Thêm regime theo xu hướng (`regime_trend_map.csv`, cùng định dạng): Bull nếu close > MA200 và return 90 ngày > +10%; Bear nếu close < MA200 và return 90 ngày < −10%; còn lại là Sideways; làm mượt bằng nhãn nhiều nhất trong 21 ngày. Chỉ dùng quá khứ.
-   `reports/regime_check.md` so hai cách gán: độ dài đợt, return trước/sau, và 9 giai đoạn đã biết của BTC. HMM đúng nhãn kỳ vọng 43%, regime theo xu hướng đúng 83%. HMM tách theo mức biến động, không theo xu hướng.

```
python ngan_lstm_hmm_analysis/run_regime.py --test
python ngan_lstm_hmm_analysis/run_regime.py      # khoảng 1 phút
```

Các ghi chú gốc:

-   `model_lstm.py` là module mô hình; cần được gọi bằng dữ liệu, danh
    sách fold và danh sách feature đã chuẩn bị đúng.
-   `regime_hmm.py` có thể chạy độc lập với pipeline dự báo nếu cung cấp
    `hmm_data` đúng định dạng.
-   `regime_map.csv` và biểu đồ PNG là các artifact kết quả đã xuất,
    không tự cập nhật khi dữ liệu hoặc mô hình thay đổi.
-   Các file này chưa tự động được nối vào `run_all.py` chỉ bằng cách
    thêm vào repository. Cần tích hợp lời gọi và đường dẫn output vào
    pipeline chính nếu nhóm muốn chạy tất cả bằng một lệnh.
-   Trước khi dùng làm kết quả cuối cùng, nhóm nên kiểm tra target,
    feature, lịch fold, gap và tính nhất quán của dữ liệu giữa các
    module.

## 10. Quy ước output dự báo

Output dự báo dùng chung nên có các cột theo đúng thứ tự:

``` text
date, fold, y_true, p_hat, regime
```

Trong đó:

-   `date`: ngày dự báo.
-   `fold`: fold walk-forward.
-   `y_true`: nhãn thực tế của hướng giá.
-   `p_hat`: xác suất LSTM dự báo giá tăng.
-   `regime`: nhãn trạng thái thị trường do HMM cung cấp.

Hai nhiệm vụ cần được phân biệt rõ: **LSTM dự báo hướng giá; HMM mô tả
trạng thái thị trường**. Việc so sánh hiệu quả dự báo vẫn cần đánh giá
trên các metric của mô hình dự báo, cả tổng thể và theo từng regime.
