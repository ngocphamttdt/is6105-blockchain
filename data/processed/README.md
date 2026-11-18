# Dữ liệu đóng băng v1.0 (data freeze)

Đây là **dữ liệu chính thức** của đề tài. Mọi đặc trưng, mô hình và thí nghiệm (E0–E4) **chỉ đọc từ thư mục này**, không đọc từ `data/raw/`.

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

Báo cáo QA (`data/raw/vi_ohlcv_flows/reports/qa_report_ohlcv_flows.md`) đề xuất **A**. Khi nhóm chốt:
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
- Các cột `ohlcv_*`: dùng để tạo nhãn y và đặc trưng OHLCV (xem `features/`), không đưa thẳng vào mô hình.
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

   Giai đoạn mô hình dùng là từ 2018-01-01, nên chỉ còn vài ngày trống của Blockchain.com. Xem thêm mục [Câu hỏi thường gặp](#câu-hỏi-thường-gặp).
4. **Ngày thiếu nến.** 30 ngày có `ohlcv_n_bars_1h < 24`: ngày niêm yết 2017-08-17 và 29 ngày Binance bảo trì hoặc gặp sự cố. Nặng nhất là 2018-02-08, chỉ có 1 nến. Trong `ohlcv_1h_clean.csv`, 43 nến lệch +28 phút sau sự cố 09–11/02/2018 đã được làm tròn về đầu giờ và đánh dấu `flag_ts_shifted = True`. Giờ thiếu không được điền, nên khi tính return 1h cần kiểm tra hai nến liên tiếp cách nhau đúng 1 giờ. Nguyên nhân và ảnh hưởng: xem [Câu hỏi thường gặp](#câu-hỏi-thường-gặp).
5. **Flows là số liệu `flash`.** Giá trị Flows phụ thuộc danh sách ví sàn do CoinMetrics gắn nhãn, và có thể bị sửa lại về sau. Dữ liệu lịch sử vì vậy không phản ánh đúng thông tin có tại thời điểm đó. Điều này cần ghi vào Limitations.
6. **Không tính tương quan giữa chỉ số On-chain và nhãn** trước khi chạy thí nghiệm (chống data snooping).

## Câu hỏi thường gặp

Các con số dưới đây được kiểm tra trên file v1.0 ngày 29/10/2026.

### Vì sao 594 dòng đầu (2016-01-01 → 2017-08-16) trống các cột `ohlcv_*`?

Binance chỉ có cặp BTCUSDT từ **2017-08-17**, nên trước đó không có giá. Các cột On-chain thì đủ từ 2016-01-01. Sau ngày 2017-08-17, các cột `ohlcv_*` không thiếu ngày nào.

### Proposal dùng dữ liệu từ ngày nào? Sao master lại bắt đầu từ 2016?

- **Proposal (mục 3. Data):** giai đoạn nghiên cứu là **01/01/2018 – 31/08/2026**, bao trọn hai kỳ halving (05/2020, 04/2024).
- **Phần trước 2018 là warm-up**, theo kế hoạch phân công:
  - CoinMetrics tải từ 01/2016 "để đủ warm-up cho MA365 của Puell" (task 1.3);
  - Binance tải từ 09/2017 "để đủ warm-up cho z-score 90 ngày và MA30" (task 1.8).
- **Walk-forward khớp với mốc 2018.** Test từ 01/2020, train 730 ngày, nên fold đầu train từ khoảng 2017-12-31.
- **Ngoại lệ:** khi kiểm tra độ nhạy với cửa sổ 1.095 ngày, fold đầu cần dữ liệu từ khoảng 01/2017, sớm hơn ngày có giá Binance. Các fold đó thực tế được train trên ít ngày hơn 1.095. Cần ghi điều này trong báo cáo.

### Vì sao `puell_multiple` trống 364 ngày đầu?

Công thức là:

```
Puell(t) = IssTotUSD(t) / trung bình 365 ngày gần nhất của IssTotUSD
```

Một ngày chỉ có Puell khi đã có đủ 365 ngày lịch sử. Dữ liệu bắt đầu 2016-01-01, nên ngày đầu có Puell là 2016-12-30. Giá trị trong file khớp đúng công thức này.

### Các ô trống này có ảnh hưởng đến mô hình không?

**Không.** Mô hình chỉ dùng dữ liệu từ khoảng 2017-12-31. Từ ngày đó:
- 10 đặc trưng OHLCV (`features_ohlcv.csv`) không có ô trống nào; ngày đầu đủ cả 10 đặc trưng là 2017-09-30;
- nhãn y không thiếu ngày nào.

**Điểm cần lưu ý:** trong giai đoạn mô hình dùng vẫn còn vài ngày trống của Blockchain.com.

| Cột | Số ngày trống | Có đưa vào mô hình không |
|---|---|---|
| `nvt_ratio_bc` | 8 | **Có** (`candidate`, Valuation) |
| `bc_est_tx_vol_usd` | 8 | Không (`backup`) |
| `bc_difficulty` | 3 | Không (`backup`) |

Các ngày trống:
- `nvt_ratio_bc` và `bc_est_tx_vol_usd` trống cùng 8 ngày: 2021-12-11, 2021-12-21, 2021-12-24, 2023-07-14, 2023-11-14, 2024-06-30, 2024-11-18 và 2025-06-24;
- `bc_difficulty` trống 3 ngày, 2025-11-13 → 2025-11-15.

- **Cách xử lý:** imputer nằm trong pipeline, chỉ fit trên train, nên vẫn đúng quy tắc không backfill.
- **Lưu ý cho `features_onchain.py` (1.16):** một ngày trống có thể lan sang ngày sau qua log-diff, và tới 30 ngày sau qua MA7/MA30, tùy cách đặt `min_periods`.

### Có nên xoá các dòng 2016 cho gọn không?

**Không nên xoá khỏi file.** Nếu cần, hãy lọc trong code:

```python
master = master[master.index >= "2017-08-17"]   # hoặc master[master.ohlcv_close.notna()]
```

- **Lọc trong code, sau khi đọc file:** kết quả không đổi. Đã kiểm tra: biến đổi On-chain (log-diff, MA7/MA30, z-score 90 ngày) của cả 9 chỉ số đưa vào mô hình giống hệt từng giá trị từ 2017-08-17, dù có hay không có các dòng 2016.
- **Xoá khỏi file:**
  - sha256 không còn khớp `MANIFEST.json`;
  - `src/data_freeze.py` sẽ sinh lại các dòng 2016 ở lần chạy sau;
  - phải ra bản v1.1.
- **Bỏ 2016 rồi tính lại Puell:** Puell chỉ có từ 2017-12-31, z-score 90 ngày của Puell chỉ có từ 2018-03-30. Các fold đầu sẽ thiếu khoảng 3 tháng đặc trưng Puell.

### Vì sao có ngày không đủ 24 nến 1 giờ?

Giờ nào Binance ngừng giao dịch thì giờ đó **không có lệnh khớp nên không có nến**. Binance Data Vision không ghi nến rỗng. Đây là trạng thái thật của sàn, không phải lỗi crawl hay làm sạch, nên không điền.

Giờ thiếu (UTC) chia thành ba kiểu:

| Kiểu | Ngày | Giờ thiếu (UTC) |
|---|---|---|
| Ngày niêm yết | 2017-08-17 | Trước khi BTCUSDT mở giao dịch (20 nến) |
| Bảo trì định kỳ, **bắt đầu đúng 02:00 UTC** (10:00 giờ Singapore) | 2018-06-26 | 02h → 11h |
| | 2018-11-14 | 02h → 08h |
| | 2019-03-12 | 02h → 07h |
| | 2019-08-15 | 02h → 09h |
| | 2019-11-13, 2019-11-25, 2020-04-25, 2021-04-20 | 02h → 03h |
| | 2020-06-28 | 02h → 04h |
| | 2021-08-13 | 02h → 05h |
| | 2020-02-09, 2020-12-25, 2021-03-06 | 02h |
| Sự cố hoặc bảo trì đột xuất | 2018-02-08 01h → 2018-02-09 08h | **32 giờ liền**, sự cố lớn nhất |
| | 2017-09-06 | 16h → 22h |
| | 2018-01-04 | 04h |
| | 2018-06-27 | 13h |
| | 2018-07-04 | 01h → 07h |
| | 2018-10-19 | 06h → 08h |
| | 2019-05-15 | 03h → 12h |
| | 2020-02-19 | 12h → 16h |
| | 2020-03-04 | 10h |
| | 2020-11-30 | 06h |
| | 2020-12-21 | 15h → 17h |
| | 2021-02-11 | 04h |
| | 2021-04-25 | 05h → 07h |
| | 2021-09-29 | 07h → 08h |
| | 2023-03-24 | 13h |

Nguyên nhân của từng sự cố chưa được đối chiếu với thông báo của Binance (binance.com/en/support/announcement). Ví dụ, 2018-07-04 nghi là sự cố giá Syscoin, 2023-03-24 nghi là lỗi lệnh trailing stop.

### Thiếu nến có ảnh hưởng đến đặc trưng và mô hình không?

**Chỉ đáng kể ở 2018-02-08 và 2018-02-09.** Bảng dưới đo z-score của đặc trưng so với toàn giai đoạn 2018+, trên các ngày thiếu nến và ngày ngay sau đó.

| Ngày | Số nến | z của `dlog_volume` | Mức ảnh hưởng |
|---|---|---|---|
| 2018-02-08 | 1 | −7,9 | Lớn: volume gần như bằng 0 |
| 2018-02-09 | 15 | +5,7 | Lớn: volume "tăng vọt" so với ngày trước |
| 15 ngày trong giai đoạn test (2020–2023) | 19–23 | \|z\| ≤ 1,6 | Không đáng kể |
| Các ngày khác của 2018–2019 | 14–23 | \|z\| ≤ 3,0 | Nhỏ |

Ngày 2018-11-14 có z = 3,0, nhưng ngày đó thị trường sập mạnh nên volume cao là thật.

- **`dlog_volume`:** bị ảnh hưởng nhiều nhất, vì volume ngày là tổng các nến.
- **`hl_range`, `rv_7d`, `rv_30d`:** gần như không lệch.
- **Nhãn y và log return:** không sai, vì giá đóng cửa vẫn là giá giao dịch thật cuối ngày.
- **Mô hình:** hai ngày 2018-02-08 và 2018-02-09 chỉ nằm trong cửa sổ train của các fold đầu, tức 2 trong khoảng 730 mẫu mỗi fold, và không nằm trong giai đoạn test.

Nếu muốn chắc chắn hơn, mà không sửa dữ liệu, có hai cách:
1. Đặt `dlog_volume` thành trống cho ngày có `ohlcv_n_bars_1h < 20` và ngày ngay sau, rồi để imputer xử lý.
2. Dùng volume trung bình mỗi giờ thay cho tổng volume ngày.

## Nguồn và giấy phép

| Nguồn | Cột | Giấy phép |
|---|---|---|
| Binance Data Vision (data.binance.vision), nến 1h BTCUSDT | `ohlcv_*`, toàn bộ `ohlcv_1h_clean.csv` | Dữ liệu công khai của Binance |
| CoinMetrics Community API | `cm_*` | **CC BY-NC 4.0**, chỉ dùng phi thương mại, phải ghi nguồn |
| Blockchain.com Charts API | `bc_*` | Không tìm thấy giấy phép dữ liệu |
| Tự tính | `puell_multiple`, `nvt_ratio_bc`, `cm_NetflowExNtv` | Theo nguồn gốc |

Chi tiết cách thu thập và làm sạch:
- `data/raw/khoa-btc-onchain-ohlcv-daily-2016-2026/README.md` (Khoa);
- `data/raw/vi_ohlcv_flows/README.md` (Vi).

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
