# Dữ liệu đóng băng v1.0 (data freeze)

Đây là **dữ liệu chính thức** của đề tài. Mọi đặc trưng, mô hình và thí nghiệm (E0–E4) **chỉ đọc từ thư mục này**, không đọc từ `crawl_data/`.

- Phiên bản: **v1.0** (28/10/2026). Lịch sử thay đổi: [`CHANGELOG.md`](../../CHANGELOG.md).
- Tạo lại: `/opt/miniconda3/bin/python src/data_freeze.py` (chạy test trước, rồi ghi file).
- Thấy lỗi dữ liệu: báo Khoa hoặc Vi để ra bản v1.1. **Không sửa file trong thư mục này và không tự sửa dữ liệu trong code của mình.**

## Các file

| File | Là gì | Kích thước |
|---|---|---|
| `master_daily.csv` | **Bảng chính.** Mỗi dòng là một ngày UTC, gồm giá BTC (OHLCV) và các chỉ số On-chain | 3.896 dòng × 24 cột, 2016-01-01 → 2026-08-31 |
| `ohlcv_1h_clean.csv` | Giá BTC theo **từng giờ**. Chỉ dùng để tính realized volatility (`rv_7d`, `rv_30d`) | 78.761 dòng × 9 cột, 2017-09-01 → 2026-08-31 |
| `data_dictionary_A.csv` | Mô tả từng cột của master: nguồn, đơn vị, nhóm, **có đưa vào mô hình không**. Theo **phương án Flows A** | 29 dòng |
| `data_dictionary_B.csv` | Giống file trên, theo **phương án Flows B** | 29 dòng |
| `MANIFEST.json` | Mã sha256, số dòng, số cột của từng file, để kiểm tra mình đang dùng đúng phiên bản | |

Đây là **một bộ dữ liệu**, không phải ba hay bốn bộ:
- `master_daily.csv` và `ohlcv_1h_clean.csv` **dùng chung** cho cả hai phương án A và B.
- Hai file dictionary chỉ khác nhau ở **4 dòng của nhóm Flows** (cột `role` và `reason`).

## Mọi người dùng file nào?

Cả nhóm dùng **chung** `master_daily.csv` (bảng chính) và `ohlcv_1h_clean.csv` (giá theo giờ). Điểm duy nhất phải chọn là **file dictionary**, tức danh sách cột On-chain được đưa vào mô hình:

- **Chưa chốt:** tạm dùng **`data_dictionary_A.csv`** (phương án đang được đề xuất). Viết code nhận `"A"` hoặc `"B"` làm tham số.
- **Nhóm chốt A:** không phải sửa gì.
- **Nhóm chốt B:** đổi tham số thành `"B"` rồi chạy lại.

Chỉ phần **On-chain** bị ảnh hưởng bởi lựa chọn A/B:

| Ai / phần nào | Có bị ảnh hưởng không |
|---|---|
| Đặc trưng OHLCV, backtest, EDA giá (Trâm) | **Không**, chỉ dùng giá |
| HMM gán regime (Ngân) | **Không**, chỉ dùng return và volatility từ giá |
| Mô hình cấu hình **OHLCV-only** | **Không**, không dùng On-chain |
| Biến đổi On-chain (Ngọc, 1.16) | **Có**: với A, Netflow âm nên không dùng được log-diff và MA7/MA30 |
| Mô hình cấu hình **OHLCV + On-chain**, ablation và SHAP (Khoa, E2) | **Có**: số cột Flows khác nhau (2 hoặc 3) |

Nên chốt **trước khi chạy đủ các fold của E1** (tuần 3, từ 31/10), để không phải chạy lại cấu hình OHLCV + On-chain.

## Hai phương án Flows (A và B), chờ nhóm chốt ở họp 1.19

Nhóm Flows có ba chỉ số về dòng BTC vào và ra khỏi các sàn giao dịch, cộng với một chỉ số tính ra từ hai chỉ số đầu:

| Cột trong master | Nghĩa | Ví dụ 01/01/2016 |
|---|---|---|
| `cm_FlowInExNtv` | BTC chuyển **vào** sàn trong ngày | 32.032,87 |
| `cm_FlowOutExNtv` | BTC rút **ra** khỏi sàn trong ngày | 33.163,55 |
| `cm_NetflowExNtv` | **Dòng ròng = vào − ra** (tính trong data freeze, không lấy từ API) | −1.130,68 |
| `cm_SplyExNtv` | Tổng BTC đang nằm trên các sàn cuối ngày | 1.471.426,77 |

**Vì sao phải chọn:** `Netflow = In − Out`, nên nếu đưa cả In, Out và Netflow vào cùng một mô hình thì một cột là thừa hoàn toàn (đa cộng tuyến hoàn hảo). Hệ số của Logistic Regression khi đó không xác định được, và kết quả ablation (RQ2) bị méo. Vì vậy chỉ dùng **một trong hai** phương án:

| | **Phương án A** | **Phương án B** |
|---|---|---|
| Đặc trưng Flows | `cm_NetflowExNtv` + `cm_SplyExNtv` | `cm_FlowInExNtv` + `cm_FlowOutExNtv` + `cm_SplyExNtv` |
| Dictionary | `data_dictionary_A.csv` | `data_dictionary_B.csv` |
| Số chỉ số Flows | 2 | 3 |
| Tổng chỉ số On-chain | 11 | 12 |
| Ưu điểm | Gọn. Một ý dễ giải thích: "BTC vào sàn nhiều hơn ra thì có áp lực bán". Flows không có nhiều cột hơn các nhóm khác | Giữ cả quy mô dòng tiền: vào 100 ra 90 khác vào 10.000 ra 9.990. **Mọi cột đều dương**, nên dùng chung một công thức biến đổi |
| Nhược điểm | **Netflow âm ở 48,2% số ngày** (từ 2018, nhỏ nhất −52.318, lớn nhất +88.036 BTC), nên **không tính được log-diff và MA7/MA30**. Chỉ dùng được z-score 90 ngày hoặc sai phân thường (x_t − x_{t−1}) | Thêm một chỉ số. In và Out tương quan khá cao với nhau (0,85 trên chuỗi đã sai phân) |

Báo cáo QA (`crawl_data/vi_ohlcv_flows/reports/qa_report_ohlcv_flows.md`) đề xuất **A**. Khi nhóm chốt:
1. Ghi quyết định vào mục "Đề xuất chốt 1.19" của báo cáo QA.
2. Ra bản v1.1: giữ đúng một dictionary, đổi tên thành `data_dictionary.csv`, điền `flows_set_chosen` trong `MANIFEST.json`, ghi vào `CHANGELOG.md`.

Trong lúc chờ, **mọi code nên nhận tên phương án ("A" hoặc "B") làm tham số** và đọc `data_dictionary_{A|B}.csv` tương ứng, để đổi phương án mà không phải sửa code.

## Chọn cột đặc trưng

Cột được đưa vào mô hình là cột có trong dictionary với `status = retained` và `role` là `candidate` hoặc `backup`:

```python
import pandas as pd

flows_set = "A"   # hoặc "B"
d = pd.read_csv(f"data/processed/data_dictionary_{flows_set}.csv")
onchain = d.loc[(d.status == "retained") & d.role.isin(["candidate", "backup"]), "column"].tolist()
master = pd.read_csv("data/processed/master_daily.csv", parse_dates=["date"]).set_index("date")
```

| Nhóm | Phương án A | Phương án B |
|---|---|---|
| Activity | `cm_AdrActCnt`, `cm_TxCnt`, `bc_est_tx_vol_usd` | giống A |
| Flows | `cm_NetflowExNtv`, `cm_SplyExNtv` | `cm_FlowInExNtv`, `cm_FlowOutExNtv`, `cm_SplyExNtv` |
| Mining | `cm_HashRate`, `cm_FeeTotNtv`, `bc_difficulty` | giống A |
| Valuation | `cm_CapMVRVCur`, `puell_multiple`, `nvt_ratio_bc` | giống A |

Các giá trị khác của `role`:
- `role = input`: cột có trong master nhưng **không phải đặc trưng**. Gồm `cm_IssTotUSD`, `cm_CapMrktCurUSD`, `cm_IssTotNtv`, `cm_PriceUSD` (dùng để tính Puell, NVT hoặc đối chiếu), cộng với các cột Flows không thuộc phương án đang dùng.
- `role = key`: `date`.
- Các cột `ohlcv_*`: dùng để tạo nhãn y và đặc trưng OHLCV (xem `tram_features_backtest/`), không đưa thẳng vào mô hình.
- Dòng `status = dropped` không có trong master. Cột `reason` giải thích lý do bị loại.

## Lưu ý bắt buộc khi dùng

1. **Độ trễ công bố On-chain.** CoinMetrics công bố dữ liệu ngày d khoảng 2–9,5 giờ sau 00:00 UTC ngày d+1. Khi dự báo hướng giá ngày t+1:
   - OHLCV dùng được đến hết ngày t;
   - **On-chain chỉ dùng đến ngày t−1.**
2. **Không chuẩn hoá ở đây.** Dữ liệu chưa log, chưa z-score, chưa tính đặc trưng. Mọi biến đổi phải fit trong cửa sổ train của từng fold, nếu không thông tin tương lai sẽ lọt vào train.
3. **Ô trống là thiếu thật, không điền:**

   | Cột | Số ngày trống | Lý do |
   |---|---|---|
   | `ohlcv_*` | 594 | Trước 2017-08-17, Binance chưa có BTCUSDT |
   | `puell_multiple` | 364 | Cần 365 ngày lịch sử |
   | `bc_est_tx_vol_usd`, `nvt_ratio_bc` | 8 | Blockchain.com thiếu dữ liệu |
   | `bc_difficulty` | 3 | Blockchain.com thiếu dữ liệu |

   Giai đoạn mô hình dùng là từ 2018-01-01, nên chỉ còn vài ngày trống của Blockchain.com.
4. **Ngày thiếu nến.** 29 ngày từ 2017-09 có `ohlcv_n_bars_1h < 24` do Binance bảo trì; nặng nhất là 2018-02-08 chỉ có 1 nến. Trong `ohlcv_1h_clean.csv`, 43 nến lệch +28 phút sau sự cố 09–11/02/2018 đã được làm tròn về đầu giờ và đánh dấu `flag_ts_shifted = True`. Giờ thiếu không được điền, nên khi tính return 1h cần kiểm tra hai nến liên tiếp cách nhau đúng 1 giờ.
5. **Flows là số liệu `flash`.** Giá trị Flows phụ thuộc danh sách ví sàn do CoinMetrics gắn nhãn, và có thể bị sửa lại về sau. Dữ liệu lịch sử vì vậy không phản ánh đúng thông tin có tại thời điểm đó. Điều này cần ghi vào Limitations.
6. **Không tính tương quan giữa chỉ số On-chain và nhãn** trước khi chạy thí nghiệm (chống data snooping).

## Nguồn và giấy phép

| Nguồn | Cột | Giấy phép |
|---|---|---|
| Binance Data Vision (data.binance.vision), nến 1h BTCUSDT | `ohlcv_*`, toàn bộ `ohlcv_1h_clean.csv` | Dữ liệu công khai của Binance |
| CoinMetrics Community API | `cm_*` | **CC BY-NC 4.0**, chỉ dùng phi thương mại, phải ghi nguồn |
| Blockchain.com Charts API | `bc_*` | Không tìm thấy giấy phép dữ liệu |
| Tự tính | `puell_multiple`, `nvt_ratio_bc`, `cm_NetflowExNtv` | Theo nguồn gốc |

Chi tiết cách thu thập và làm sạch:
- `crawl_data/khoa-btc-onchain-ohlcv-daily-2016-2026/README.md` (Khoa);
- `crawl_data/vi_ohlcv_flows/README.md` (Vi).

## Kiểm tra đúng phiên bản

So mã sha256 của file với `MANIFEST.json`:

```
shasum -a 256 data/processed/*.csv
```

| File | sha256 (v1.0) |
|---|---|
| `master_daily.csv` | `bae8d2ea2e59558dd0af26ad8037087d9fb04d18e84849893bf33f72a0bb06a3` |
| `ohlcv_1h_clean.csv` | `149c78ecb277134e344d069a18adc75feeed85a0dc06bfab34418ba4a57105ae` |
| `data_dictionary_A.csv` | `6d086400c066fde88559f849754fcbd0836d979b65a73e156de323f80fe5e7ea` |
| `data_dictionary_B.csv` | `c109cadf81d70b5188562b7ec75e0a849c659c9bfb141e057ab54e9b446c0bb5` |
