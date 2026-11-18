# QA nhóm OHLCV và Flows (task 1.11)

Giai đoạn QA: 2018-01-01 → 2026-08-31. Quy tắc: thiếu > 5% thì loại; chuỗi có xu hướng thì sai phân; trong cùng nhóm |tương quan| > 0.9 (trên chuỗi đã sai phân) thì giữ một. Không tính tương quan giữa chỉ số và nhãn.

## 1. OHLCV (Binance BTCUSDT, nến 1h gộp thành 1d theo UTC)

- Nguồn: 108 file zip theo tháng, tất cả khớp `.CHECKSUM` (xem `binance_manifest.json`). File từ 01/2025 dùng timestamp microsecond, đã tự nhận diện.
- Nến 1h: thiếu 127/78888 giờ (0.16%); trùng: 0; nến vi phạm high/low: 0; giờ có volume = 0: 4.
- 43 nến sau sự cố Binance 08–09/02/2018 bị lệch +28m14s (2018-02-09 09:28 → 2018-02-11 03:28); đã làm tròn xuống giờ, đánh dấu cột `flag_ts_shifted`.
- Daily trong giai đoạn QA: 3165 ngày, thiếu hoàn toàn: 0, có < 24 nến: 28 (cột `flag_incomplete`). Ngày tệ nhất: 2018-02-08 chỉ có 1 nến (sàn dừng giao dịch), OHLC ngày đó không đại diện cho cả ngày.
- Nhảy giá > 10%/1h: 14 giờ, đều trùng sự kiện thị trường đã biết (ví dụ 12–13/03/2020); giữ nguyên, không coi là lỗi.
- Đối chiếu với file 1d lấy từ REST API trước đây: 3137 ngày đủ nến, chỉ 1 ngày lệch giá đóng cửa (09–10/02/2018, do nến bị lệch giờ ở trên).
- Kết luận: **giữ** OHLCV. Mô hình nên dùng `flag_incomplete` để cân nhắc loại các ngày < 24 nến khỏi tính realized volatility nếu cần.

## 2. Flows (CoinMetrics Community)

| Chỉ số | Thiếu | Tăng đơn điệu | Biến đổi trước khi tính tương quan |
|---|---|---|---|
| FlowInExNtv | 0.00% | không | logdiff |
| FlowOutExNtv | 0.00% | không | logdiff |
| NetflowExNtv | 0.00% | không | diff |
| SplyExNtv | 0.00% | không | logdiff |

Ma trận tương quan (trên chuỗi đã sai phân):

| | FlowInExNtv | FlowOutExNtv | NetflowExNtv | SplyExNtv |
|---|---|---|---|---|
| FlowInExNtv | 1.000 | 0.851 | 0.244 | 0.198 |
| FlowOutExNtv | 0.851 | 1.000 | -0.235 | -0.119 |
| NetflowExNtv | 0.244 | -0.235 | 1.000 | 0.658 |
| SplyExNtv | 0.198 | -0.119 | 0.658 | 1.000 |

- **Giữ:** NetflowExNtv, SplyExNtv, FlowInExNtv, FlowOutExNtv
- **Loại:** không có
- **Cần nhóm chốt:** quy tắc tương quan từng cặp không bắt được đa cộng tuyến hoàn hảo NetflowExNtv = FlowInExNtv − FlowOutExNtv. Không nên đưa cả ba vào cùng một mô hình; đề xuất giữ NetflowExNtv + SplyExNtv (hoặc FlowIn + FlowOut + SplyExNtv). Về lý thuyết Δ SplyExNtv ≈ Netflow, nhưng tương quan chỉ 0.66, thêm dấu hiệu danh sách ví gắn nhãn thay đổi theo thời gian.
- Thứ tự ưu tiên khi hai chỉ số trùng thông tin: NetflowExNtv > SplyExNtv > FlowInExNtv > FlowOutExNtv (đặt trước khi xem kết quả).
- Lưu ý: bước nhảy của SplyExNtv (xem `flows_check.md`) gồm cả sự kiện thật (FTX 11/2022, LUNA 05/2022, 12/03/2020) lẫn khả năng thay đổi danh sách ví gắn nhãn; không tự sửa dữ liệu, ghi vào Limitations.

## 3. Nháp đoạn Constraints

> **SOPR.** Chỉ số SOPR miễn phí (Glassnode, CryptoQuant gói free) chỉ cung cấp khoảng 4 năm gần nhất, không phủ giai đoạn 2018–2022 vốn cần cho cửa sổ train 730 ngày của các fold đầu; CoinMetrics Community không có SOPR. Nhóm thay SOPR bằng hai chỉ số định giá tự tính từ CoinMetrics: Puell Multiple và NVT.
>
> **Whale Transfers.** Không có nguồn miễn phí cung cấp sẵn chuỗi giao dịch cá voi theo ngày. Tự tính từ BigQuery `crypto_bitcoin` bị nhiễu do output tiền thừa (change output) không phân biệt được với chuyển khoản thật nếu không có heuristic gom địa chỉ, và tốn quota truy vấn. Nhóm bỏ chỉ số này.
>
> **Exchange Flows.** FlowInExNtv, FlowOutExNtv, SplyExNtv phụ thuộc danh sách ví sàn do CoinMetrics gắn nhãn; khi danh sách thay đổi, chuỗi có thể nhảy bậc mà không phản ánh dòng tiền thật. Giá trị trên API đang ở trạng thái `flash` (chưa qua bước reviewed) và có thể được CoinMetrics sửa lại về sau; nhóm cố định bản tải trong data freeze.
