# vi_ohlcv_flows — Dữ liệu OHLCV & Flows

Toàn bộ code, dữ liệu và báo cáo cho task **1.8 – 1.11** trong `Task_Planning_Group1_BlocknRoll.docx`.
Thư mục độc lập với dữ liệu cũ trong `data/` (lấy qua REST API 1d, từ 2018).

## Trạng thái task (chạy thật ngày 23–24/10/2026)

| Task | Nội dung | Trạng thái | Kết quả chính | File |
|---|---|---|---|---|
| 1.8 | Tải Binance 1h BTCUSDT | ✅ Xong | 108 file zip 09/2017 → 08/2026, tất cả khớp `.CHECKSUM`; 20 file từ 2025 dùng timestamp microsecond (đã tự nhận diện) | `src/crawl_binance.py`, `data/raw/binance_1h/`, `reports/binance_manifest.json` |
| 1.9 | Làm sạch OHLCV daily | ✅ Xong | 3.287 ngày, không ngày nào mất hẳn; thiếu 127/78.888 giờ; 29 ngày < 24 nến (`flag_incomplete`); khớp nến 1d của REST API trừ 09–10/02/2018 | `src/clean_ohlcv.py`, `data/clean/ohlcv_daily.csv`, `data/clean/ohlcv_1h_clean.csv`, `reports/ohlcv_clean_log.json` |
| 1.10 | Kiểm chứng Flows | ✅ Xong — đề xuất **GIỮ** | 3 metric miễn phí từ 2016, thiếu 0% từ 2018, khớp 100% file GitHub; 23 ngày SplyExNtv nhảy bậc | `src/flows_check.py`, `data/clean/flows_daily.csv`, `reports/flows_check.md`, `reports/figures/flows_series.png` |
| 1.11 | QA Flows + OHLCV, Constraints | ✅ Xong (chờ nhóm chốt) | Không chỉ số nào bị loại theo quy tắc; cần chốt vì Netflow = In − Out | `src/qa_ohlcv_flows.py`, `reports/qa_report_ohlcv_flows.md` |

## Cấu trúc

```
vi_ohlcv_flows/
├── README.md                      ← bạn đang đọc
├── src/                           ← chạy theo thứ tự 1 → 4
│   ├── crawl_binance.py           (1) task 1.8
│   ├── clean_ohlcv.py             (2) task 1.9
│   ├── flows_check.py             (3) task 1.10
│   └── qa_ohlcv_flows.py          (4) task 1.11
├── data/
│   ├── raw/                       ← bất biến, không sửa
│   │   ├── binance_1h/            108 × BTCUSDT-1h-YYYY-MM.zip + .CHECKSUM
│   │   ├── coinmetrics_flows/     JSON gốc từ API (có crawled_at_utc)
│   │   └── coinmetrics_github_btc.csv   file đối chiếu
│   └── clean/                     ← dữ liệu sạch dùng cho đặc trưng, pipeline, regime và merge
│       ├── ohlcv_1h_clean.csv     index datetime UTC; cột flag_ts_shifted
│       ├── ohlcv_daily.csv        index date UTC; n_candles, flag_incomplete, flag_imputed
│       └── flows_daily.csv        FlowInExNtv, FlowOutExNtv, SplyExNtv, NetflowExNtv (2016-01-01 → 2026-08-31)
└── reports/
    ├── flows_check.md             ← đọc trước khi họp go/no-go (task 1.19)
    ├── qa_report_ohlcv_flows.md   ← QA + nháp đoạn Constraints
    ├── binance_manifest.json      sha256, số dòng, đơn vị timestamp từng file
    ├── ohlcv_clean_log.json       giờ thiếu, ngày thiếu nến, nhảy giá, đối chiếu REST API
    └── figures/flows_series.png
```

## Nguồn dữ liệu

| # | Nguồn | Địa chỉ | Lấy gì | Dùng để làm gì |
|---|---|---|---|---|
| 1 | **Binance Data Vision**, kho dữ liệu lịch sử công khai của sàn Binance | `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/` | 108 file zip nến 1h của cặp BTCUSDT (09/2017 → 08/2026), kèm 108 file `.CHECKSUM` (mã sha256 do Binance công bố) | Dữ liệu giá chính (task 1.8, 1.9) |
| 2 | **CoinMetrics Community API** (gói miễn phí, không cần API key) | `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics` | 3 chỉ số on-chain theo ngày của BTC (01/2016 → 08/2026): `FlowInExNtv`, `FlowOutExNtv`, `SplyExNtv`; gọi riêng từng chỉ số | Dữ liệu nhóm Flows (task 1.10) |
| 3 | **CoinMetrics trên GitHub**, file dữ liệu community công khai | `https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv` | Toàn bộ file CSV của BTC | Chỉ dùng để **đối chiếu** với nguồn 2; khớp 100% trên 3.796 ngày trùng nhau |

Ngoài ra, dữ liệu ngày còn được so với file cũ `data/btc_spot_1d_20180101_20261021.csv` (lấy từ Binance REST API `api.binance.com/api/v3/klines`). File này đã có sẵn trong dự án và không được crawl lại; nó chỉ dùng để kiểm tra chéo.

Bản gốc của cả 3 nguồn được giữ nguyên trong `data/raw/`. Thời điểm tải ghi trong `reports/binance_manifest.json` và trong trường `crawled_at_utc` của từng file JSON.

## Nến là gì

"Nến" (candlestick, nến Nhật) tóm tắt toàn bộ giao dịch trong một khoảng thời gian cố định bằng 4 mức giá và khối lượng, viết tắt là **OHLCV**:

| Thành phần | Ý nghĩa |
|---|---|
| **Open** | Giá khớp lệnh đầu tiên của khoảng thời gian |
| **High** | Giá cao nhất trong khoảng đó |
| **Low** | Giá thấp nhất trong khoảng đó |
| **Close** | Giá khớp lệnh cuối cùng |
| **Volume** | Tổng khối lượng giao dịch |

Trên biểu đồ, mỗi khoảng thời gian được vẽ thành một cây nến:

```
      │   ← bấc trên: lên tới High
    ┌─┴─┐
    │   │ ← thân nến: từ Open đến Close
    │   │   (xanh nếu Close > Open, đỏ nếu Close < Open)
    └─┬─┘
      │   ← bấc dưới: xuống tới Low
```

- **Nến 1h** tóm tắt 1 giờ giao dịch; một ngày đủ dữ liệu có 24 nến 1h.
- **Nến 1d** tóm tắt cả ngày. Trong thư mục này, nến 1d được gộp từ 24 nến 1h: open lấy từ nến đầu, high là max, low là min, close lấy từ nến cuối, volume là tổng.
- Ngày có **ít hơn 24 nến** là ngày có những giờ Binance ngừng giao dịch (bảo trì, sự cố). Nến ngày đó không phản ánh đầy đủ cả ngày và được đánh dấu bằng cột `flag_incomplete`.

## Mô tả dữ liệu sau khi làm sạch

Ba file trong `data/clean/`, đều là CSV có dòng tiêu đề, **không có giá trị rỗng**. Thời gian theo UTC. Số cột dưới đây đã tính cả cột thời gian ở đầu.

| File | Số dòng | Số cột | Khoảng thời gian | Một dòng là |
|---|---|---|---|---|
| `ohlcv_1h_clean.csv` | 78.761 | 9 | 2017-09-01 00:00 → 2026-08-31 23:00 | 1 giờ giao dịch BTCUSDT trên Binance |
| `ohlcv_daily.csv` | 3.287 | 11 | 2017-09-01 → 2026-08-31 | 1 ngày (gộp từ các nến 1h) |
| `flows_daily.csv` | 3.896 | 5 | 2016-01-01 → 2026-08-31 | 1 ngày dòng BTC ra/vào sàn |

### `ohlcv_1h_clean.csv`

| Cột | Kiểu | Đơn vị | Ý nghĩa |
|---|---|---|---|
| `datetime` | thời gian | UTC | Giờ mở của nến, ví dụ `2017-09-01 00:00:00+00:00` là nến từ 00:00 đến 00:59 |
| `open` | số thực | USDT | Giá khớp đầu tiên trong giờ |
| `high` | số thực | USDT | Giá cao nhất trong giờ |
| `low` | số thực | USDT | Giá thấp nhất trong giờ |
| `close` | số thực | USDT | Giá khớp cuối cùng trong giờ |
| `volume` | số thực | BTC | Khối lượng BTC được mua bán trong giờ |
| `quote_volume` | số thực | USDT | Giá trị giao dịch trong giờ, tính bằng USDT |
| `n_trades` | số nguyên | lệnh | Số lệnh khớp trong giờ |
| `flag_ts_shifted` | True/False | – | `True` nếu nến gốc bị lệch +28 phút sau sự cố Binance 09–11/02/2018 và đã được làm tròn về đầu giờ (43 dòng) |

Lưới giờ thiếu 127/78.888 giờ, do sàn bảo trì. Các giờ thiếu **không được điền thêm**, nên khi tính return 1h cần kiểm tra hai nến liên tiếp có cách nhau đúng 1 giờ không.

### `ohlcv_daily.csv`

| Cột | Kiểu | Đơn vị | Ý nghĩa |
|---|---|---|---|
| `date` | ngày | UTC | Ngày giao dịch, từ 00:00 đến 23:59 UTC |
| `open` | số thực | USDT | Giá mở của nến 1h đầu tiên trong ngày |
| `high` | số thực | USDT | Giá cao nhất trong các nến 1h của ngày |
| `low` | số thực | USDT | Giá thấp nhất trong các nến 1h của ngày |
| `close` | số thực | USDT | Giá đóng của nến 1h cuối cùng trong ngày; dùng để tạo nhãn hướng giá |
| `volume` | số thực | BTC | Tổng khối lượng BTC trong ngày |
| `quote_volume` | số thực | USDT | Tổng giá trị giao dịch trong ngày |
| `n_trades` | số nguyên | lệnh | Tổng số lệnh khớp trong ngày |
| `n_candles` | số nguyên | nến | Số nến 1h có dữ liệu trong ngày (đủ là 24) |
| `flag_imputed` | True/False | – | `True` nếu cả ngày không có nến nào và giá được lấy từ giá đóng cửa hôm trước. Hiện có **0 ngày** |
| `flag_incomplete` | True/False | – | `True` nếu ngày có ít hơn 24 nến. Hiện có **29 ngày**, nặng nhất là 2018-02-08 chỉ có 1 nến |

### `flows_daily.csv`

Nguồn: CoinMetrics Community API. Hậu tố `Ntv` nghĩa là đơn vị gốc của tài sản, tức **BTC**.

| Cột | Kiểu | Đơn vị | Ý nghĩa |
|---|---|---|---|
| `date` | ngày | UTC | Ngày quan sát |
| `FlowInExNtv` | số thực | BTC | Lượng BTC chuyển **vào** các ví được CoinMetrics gắn nhãn là ví sàn trong ngày |
| `FlowOutExNtv` | số thực | BTC | Lượng BTC chuyển **ra** khỏi các ví sàn trong ngày |
| `SplyExNtv` | số thực | BTC | Tổng lượng BTC đang nằm trong các ví sàn vào cuối ngày |
| `NetflowExNtv` | số thực | BTC | Dòng ròng = `FlowInExNtv − FlowOutExNtv`. Dương: BTC chảy vào sàn nhiều hơn (thường gắn với áp lực bán); âm: BTC rút khỏi sàn nhiều hơn. Cột này được tính thêm, không lấy từ API |

## Chạy lại

Từ thư mục gốc của repo (cần pandas, pyarrow, matplotlib):

```
python -B vi_ohlcv_flows/src/crawl_binance.py     # bỏ qua file đã có và đúng checksum
python -B vi_ohlcv_flows/src/clean_ohlcv.py
python -B vi_ohlcv_flows/src/flows_check.py
python -B vi_ohlcv_flows/src/qa_ohlcv_flows.py
```

Hoặc trong Claude Code: gọi agent **`ohlcv-flows-crawler`** (`.claude/agents/ohlcv-flows-crawler.md`).

## Lưu ý cho người dùng dữ liệu

- Chưa chuẩn hoá, chưa có đặc trưng: mọi biến đổi làm trong pipeline, fit theo từng fold.
- **2018-02-08 chỉ có 1 nến** (Binance dừng giao dịch); 43 nến sau đó lệch +28 phút, đã làm tròn giờ (`flag_ts_shifted`).
- Realized volatility nên tính từ `ohlcv_1h_clean.csv` (đúng đặc tả task 1.13), không từ return ngày.
- Flows phụ thuộc nhãn ví sàn của CoinMetrics, giá trị đang ở trạng thái `flash`. **Không dùng cả In, Out và Netflow cùng lúc** (đa cộng tuyến hoàn hảo); đề xuất Netflow + SplyExNtv.

## Việc còn lại

- [ ] Họp go/no-go Flows (task 1.19) và chốt bộ chỉ số Flows.
- [ ] Merge `data/clean/*` vào `master_daily.parquet` (task 1.7, data freeze). Bộ này **chưa** phải data freeze.
- [ ] Sau data freeze, mọi sửa đổi ra phiên bản mới kèm `CHANGELOG.md`.
