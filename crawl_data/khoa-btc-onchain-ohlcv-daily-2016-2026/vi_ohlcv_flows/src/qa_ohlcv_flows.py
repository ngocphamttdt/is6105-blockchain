"""Task 1.11 (Vi): QA nhóm Flows và OHLCV theo quy tắc trong proposal, kèm nháp đoạn Constraints.

Chạy sau clean_ohlcv.py và flows_check.py: /opt/miniconda3/bin/python src/qa_ohlcv_flows.py
Quy tắc (giống task 1.6 của Khoa):
  1. Loại chỉ số thiếu > 5% số ngày (tính từ 01/2018).
  2. Chuỗi tăng đơn điệu / có xu hướng thì dùng sai phân.
  3. Trong cùng nhóm, |tương quan| > 0,9 trên chuỗi đã sai phân thì chỉ giữ một.
Chỉ tính tương quan giữa các chỉ số với nhau, KHÔNG tính với nhãn (chống data snooping).
"""
import json

import numpy as np
import pandas as pd

from crawl_binance import OUT

CLEAN = OUT / "data" / "clean"
REPORTS = OUT / "reports"
QA_FROM, QA_TO = "2018-01-01", "2026-08-31"
MAX_MISSING, MAX_CORR = 0.05, 0.9
# NetflowExNtv có thể âm nên sai phân thường; các chuỗi còn lại dương nên log-diff
FLOWS = {"FlowInExNtv": "logdiff", "FlowOutExNtv": "logdiff", "NetflowExNtv": "diff", "SplyExNtv": "logdiff"}
# thứ tự ưu tiên giữ khi hai chỉ số tương quan cao: netflow là chỉ số chính trong related work
PRIORITY = ["NetflowExNtv", "SplyExNtv", "FlowInExNtv", "FlowOutExNtv"]


def transform(s, how):
    return np.log(s).diff() if how == "logdiff" else s.diff()


def qa_flows():
    f = pd.read_csv(CLEAN / "flows_daily.csv", index_col="date", parse_dates=True).loc[QA_FROM:QA_TO]
    rows, tr = [], {}
    for m, how in FLOWS.items():
        s = f[m]
        miss = s.isna().mean()
        mono = s.is_monotonic_increasing
        tr[m] = transform(s, how)
        rows.append({"metric": m, "missing": miss, "monotonic_increasing": mono, "transform": how,
                     "keep_missing": miss <= MAX_MISSING})
    tbl = pd.DataFrame(rows).set_index("metric")
    corr = pd.DataFrame(tr).corr()

    kept = []
    dropped = {}
    for m in PRIORITY:
        if not tbl.loc[m, "keep_missing"]:
            dropped[m] = f"thiếu {tbl.loc[m, 'missing']:.1%} > {MAX_MISSING:.0%}"
            continue
        clash = [k for k in kept if abs(corr.loc[m, k]) > MAX_CORR]
        if clash:
            dropped[m] = f"|tương quan| với {clash[0]} = {abs(corr.loc[m, clash[0]]):.3f} > {MAX_CORR}"
        else:
            kept.append(m)
    return tbl, corr, kept, dropped


def main():
    log = json.loads((REPORTS / "ohlcv_clean_log.json").read_text())
    d = pd.read_csv(CLEAN / "ohlcv_daily.csv", index_col="date", parse_dates=True).loc[QA_FROM:QA_TO]
    tbl, corr, kept, dropped = qa_flows()

    L = ["# QA nhóm OHLCV và Flows (task 1.11)", "",
         f"Giai đoạn QA: {QA_FROM} → {QA_TO}. Quy tắc: thiếu > {MAX_MISSING:.0%} thì loại; chuỗi có xu hướng thì sai phân; "
         f"trong cùng nhóm |tương quan| > {MAX_CORR} (trên chuỗi đã sai phân) thì giữ một. "
         "Không tính tương quan giữa chỉ số và nhãn.", "",
         "## 1. OHLCV (Binance BTCUSDT, nến 1h gộp thành 1d theo UTC)", "",
         f"- Nguồn: {len(json.loads((REPORTS / 'binance_manifest.json').read_text())['files'])} file zip theo tháng, "
         "tất cả khớp `.CHECKSUM` (xem `binance_manifest.json`). File từ 01/2025 dùng timestamp microsecond, đã tự nhận diện.",
         f"- Nến 1h: thiếu {log['hours_missing']}/{log['hours_expected']} giờ "
         f"({log['hours_missing'] / log['hours_expected']:.2%}); trùng: {log['duplicates_removed']}; "
         f"nến vi phạm high/low: {log['bad_ohlc_rows']}; giờ có volume = 0: {len(log['zero_volume_hours'])}.",
         f"- {log['ts_shifted_rows']} nến sau sự cố Binance 08–09/02/2018 bị lệch +28m14s "
         f"({log['ts_shifted_range'][0][:16]} → {log['ts_shifted_range'][1][:16]}); đã làm tròn xuống giờ, "
         "đánh dấu cột `flag_ts_shifted`.",
         f"- Daily trong giai đoạn QA: {len(d)} ngày, thiếu hoàn toàn: {int(d['flag_imputed'].sum())}, "
         f"có < 24 nến: {int(d['flag_incomplete'].sum())} (cột `flag_incomplete`). "
         "Ngày tệ nhất: 2018-02-08 chỉ có 1 nến (sàn dừng giao dịch), OHLC ngày đó không đại diện cho cả ngày.",
         f"- Nhảy giá > 10%/1h: {len(log['price_jumps_1h'])} giờ, đều trùng sự kiện thị trường đã biết "
         "(ví dụ 12–13/03/2020); giữ nguyên, không coi là lỗi.",
         f"- Đối chiếu với file 1d lấy từ REST API trước đây: {log['compare_rest_api_1d']['complete_days_compared']} ngày đủ nến, "
         f"chỉ {log['compare_rest_api_1d']['days_rel_diff_close_gt_1e-6']} ngày lệch giá đóng cửa "
         "(09–10/02/2018, do nến bị lệch giờ ở trên).",
         "- Kết luận: **giữ** OHLCV. Mô hình nên dùng `flag_incomplete` để cân nhắc loại các ngày < 24 nến khỏi tính "
         "realized volatility nếu cần.", "",
         "## 2. Flows (CoinMetrics Community)", "",
         "| Chỉ số | Thiếu | Tăng đơn điệu | Biến đổi trước khi tính tương quan |", "|---|---|---|---|"]
    for m, r in tbl.iterrows():
        L.append(f"| {m} | {r['missing']:.2%} | {'có' if r['monotonic_increasing'] else 'không'} | {r['transform']} |")
    L += ["", "Ma trận tương quan (trên chuỗi đã sai phân):", "",
          "| | " + " | ".join(corr.columns) + " |", "|---" * (len(corr.columns) + 1) + "|"]
    for m, r in corr.iterrows():
        L.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    L += ["", f"- **Giữ:** {', '.join(kept)}"]
    L += [f"- **Loại:** {m} — {why}" for m, why in dropped.items()] or ["- **Loại:** không có"]
    L += ["- **Cần nhóm chốt:** quy tắc tương quan từng cặp không bắt được đa cộng tuyến hoàn hảo "
          "NetflowExNtv = FlowInExNtv − FlowOutExNtv. Không nên đưa cả ba vào cùng một mô hình; đề xuất giữ "
          "NetflowExNtv + SplyExNtv (hoặc FlowIn + FlowOut + SplyExNtv). Về lý thuyết Δ SplyExNtv ≈ Netflow, "
          f"nhưng tương quan chỉ {corr.loc['NetflowExNtv', 'SplyExNtv']:.2f}, thêm dấu hiệu danh sách ví gắn nhãn thay đổi theo thời gian.",
          "- Thứ tự ưu tiên khi hai chỉ số trùng thông tin: " + " > ".join(PRIORITY) + " (đặt trước khi xem kết quả).",
          "- Lưu ý: bước nhảy của SplyExNtv (xem `flows_check.md`) gồm cả sự kiện thật (FTX 11/2022, LUNA 05/2022, "
          "12/03/2020) lẫn khả năng thay đổi danh sách ví gắn nhãn; không tự sửa dữ liệu, ghi vào Limitations.", "",
          "## 3. Nháp đoạn Constraints", "",
          "> **SOPR.** Chỉ số SOPR miễn phí (Glassnode, CryptoQuant gói free) chỉ cung cấp khoảng 4 năm gần nhất, không phủ "
          "giai đoạn 2018–2022 vốn cần cho cửa sổ train 730 ngày của các fold đầu; CoinMetrics Community không có SOPR. "
          "Nhóm thay SOPR bằng hai chỉ số định giá tự tính từ CoinMetrics: Puell Multiple và NVT.",
          ">",
          "> **Whale Transfers.** Không có nguồn miễn phí cung cấp sẵn chuỗi giao dịch cá voi theo ngày. Tự tính từ BigQuery "
          "`crypto_bitcoin` bị nhiễu do output tiền thừa (change output) không phân biệt được với chuyển khoản thật nếu "
          "không có heuristic gom địa chỉ, và tốn quota truy vấn. Nhóm bỏ chỉ số này.",
          ">",
          "> **Exchange Flows.** FlowInExNtv, FlowOutExNtv, SplyExNtv phụ thuộc danh sách ví sàn do CoinMetrics gắn nhãn; "
          "khi danh sách thay đổi, chuỗi có thể nhảy bậc mà không phản ánh dòng tiền thật. Giá trị trên API đang ở trạng "
          "thái `flash` (chưa qua bước reviewed) và có thể được CoinMetrics sửa lại về sau; nhóm cố định bản tải trong "
          "data freeze.", ""]
    (REPORTS / "qa_report_ohlcv_flows.md").write_text("\n".join(L))
    print("\n".join(L[L.index("## 2. Flows (CoinMetrics Community)"):L.index("## 3. Nháp đoạn Constraints")]))


if __name__ == "__main__":
    main()
