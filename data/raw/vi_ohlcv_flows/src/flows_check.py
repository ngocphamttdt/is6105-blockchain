"""Task 1.10 (Vi): crawl và kiểm chứng nhóm Flows (CoinMetrics Community).

Chạy: /opt/miniconda3/bin/python src/flows_check.py
- Gọi API từng metric một (metric ngoài gói community sẽ làm cả request lỗi), nghỉ giữa các request
  để không vượt 10 request / 6 giây. Lưu JSON gốc kèm ngày tải vào data/raw/coinmetrics_flows/.
- So giá trị API với file CSV community trên GitHub (coinmetrics/data).
- Tìm bước nhảy đột ngột (thường do thêm sàn mới vào danh sách ví gắn nhãn), vẽ chuỗi.
- Xuất data/clean/flows_daily.csv và reports/flows_check.md kèm đề xuất go/no-go (nhóm quyết định ở task 1.19).
"""
import io
import json
import time
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crawl_binance import OUT, http_get

RAW = OUT / "data" / "raw" / "coinmetrics_flows"
CLEAN = OUT / "data" / "clean"
REPORTS = OUT / "reports"
FIGS = REPORTS / "figures"
API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
GITHUB_CSV = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
METRICS = ["FlowInExNtv", "FlowOutExNtv", "SplyExNtv"]
START, END = "2016-01-01", "2026-08-31"
CHECK_FROM = "2018-01-01"  # train của fold đầu bắt đầu 01/2018
MAX_MISSING = 0.05  # quy tắc QA trong proposal
JUMP_Z = 8  # robust z-score của sai phân SplyExNtv


def crawl_metric(metric):
    params = {"assets": "btc", "metrics": metric, "frequency": "1d",
              "start_time": START, "end_time": END, "page_size": 10000}
    url = API + "?" + urllib.parse.urlencode(params)
    rows = []
    while url:
        out = json.loads(http_get(url))
        rows.extend(out["data"])
        url = out.get("next_page_url")
        time.sleep(0.7)
    return rows


def to_series(rows, metric):
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
    s = pd.Series(pd.to_numeric(df[metric]).values, index=idx, name=metric)
    status = df.get(f"{metric}-status", pd.Series(dtype=str)).value_counts().to_dict()
    return s.sort_index(), status


def robust_jumps(s):
    d = s.diff().dropna()
    mad = (d - d.median()).abs().median() * 1.4826
    z = (d - d.median()) / mad
    return d[z.abs() > JUMP_Z], z


def main():
    for p in (RAW, CLEAN, FIGS):
        p.mkdir(parents=True, exist_ok=True)
    crawled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    series, status, errors = {}, {}, {}
    for m in METRICS:
        try:
            rows = crawl_metric(m)
        except urllib.error.HTTPError as e:
            errors[m] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
            print(f"{m}: LỖI {errors[m]}")
            continue
        (RAW / f"{m}_{START}_{END}.json").write_text(
            json.dumps({"crawled_at_utc": crawled_at, "params": {"start": START, "end": END}, "data": rows}))
        series[m], status[m] = to_series(rows, m)
        print(f"{m}: {len(rows)} dòng, {series[m].index.min().date()} -> {series[m].index.max().date()}")

    flows = pd.DataFrame(series)
    flows.index.name = "date"
    grid = pd.date_range(CHECK_FROM, END, freq="D")
    flows = flows.reindex(pd.date_range(flows.index.min(), END, freq="D").rename("date"))
    if {"FlowInExNtv", "FlowOutExNtv"} <= set(flows):
        flows["NetflowExNtv"] = flows["FlowInExNtv"] - flows["FlowOutExNtv"]
    flows.to_csv(CLEAN / "flows_daily.csv")

    # so với file GitHub
    gh_path = OUT / "data" / "raw" / "coinmetrics_github_btc.csv"
    gh_bytes = http_get(GITHUB_CSV, timeout=120)
    gh_path.write_bytes(gh_bytes)
    gh = pd.read_csv(io.BytesIO(gh_bytes), low_memory=False)
    gh.index = pd.to_datetime(gh["time"]).dt.normalize()

    rows = []
    for m in METRICS:
        if m not in flows:
            rows.append({"metric": m, "api": "FAIL", "note": errors.get(m)})
            continue
        s = flows[m]
        valid = s.dropna()
        miss = s.reindex(grid).isna().mean()
        r = {"metric": m, "api": "OK", "first": valid.index.min().date(), "last": valid.index.max().date(),
             "missing_from_2018": miss, "status": status[m]}
        if m in gh:
            g = pd.to_numeric(gh[m], errors="coerce")
            j = pd.concat([s, g.rename("gh")], axis=1, join="inner").dropna()
            rel = (j[m] - j["gh"]).abs() / j["gh"].abs().clip(lower=1e-9)
            r.update(gh_overlap=len(j), gh_last=g.dropna().index.max().date(),
                     gh_max_rel_diff=float(rel.max()), gh_days_rel_diff_gt_1pct=int((rel > 0.01).sum()))
        else:
            r.update(gh_overlap=0, gh_last=None, gh_max_rel_diff=np.nan, gh_days_rel_diff_gt_1pct=None)
        rows.append(r)
    summary = pd.DataFrame(rows)

    jumps, _ = robust_jumps(flows["SplyExNtv"].dropna()) if "SplyExNtv" in flows else (pd.Series(dtype=float), None)
    sply = flows.get("SplyExNtv")
    jump_tbl = pd.DataFrame({"delta_btc": jumps,
                             "pct_of_exchange_supply": jumps / sply.shift(1).reindex(jumps.index) * 100})

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, m in zip(axes, ["SplyExNtv", "FlowInExNtv", "FlowOutExNtv"]):
        if m in flows:
            ax.plot(flows.index, flows[m], lw=0.7)
            if m == "SplyExNtv":
                ax.scatter(jumps.index, sply.reindex(jumps.index), color="red", s=12, zorder=3, label="bước nhảy")
                ax.legend()
        ax.set_ylabel(m)
        ax.axvline(pd.Timestamp(CHECK_FROM), color="gray", ls="--", lw=0.8)
    axes[0].set_title("CoinMetrics Community – nhóm Flows (BTC)")
    fig.tight_layout()
    fig.savefig(FIGS / "flows_series.png", dpi=120)

    ok = summary[summary["api"] == "OK"]
    go = (len(ok) == len(METRICS)
          and (ok["missing_from_2018"] <= MAX_MISSING).all()
          and (pd.to_datetime(ok["last"]) >= pd.Timestamp(END)).all())
    write_report(summary, jump_tbl, go, crawled_at)
    print(summary.to_string(index=False))
    print(f"Bước nhảy SplyExNtv (|robust z| > {JUMP_Z}): {len(jumps)}")
    print("Đề xuất:", "GIỮ (go)" if go else "BỎ (no-go)")


def write_report(summary, jumps, go, crawled_at):
    lines = [
        "# Kiểm chứng nhóm Flows (task 1.10)", "",
        f"- Nguồn: CoinMetrics Community API `{API}`, gọi từng metric một; tải lúc {crawled_at} (UTC).",
        f"- Khoảng crawl: {START} → {END}. Tỷ lệ thiếu tính trên {CHECK_FROM} → {END}.",
        f"- Đối chiếu: file community trên GitHub `{GITHUB_CSV}`.", "",
        "## Kết quả theo metric", "",
        "| Metric | API | Ngày đầu | Ngày cuối | Thiếu từ 2018 | Trạng thái giá trị | Ngày trùng GitHub | Ngày cuối GitHub | Lệch tương đối lớn nhất | Số ngày lệch >1% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        if r["api"] != "OK":
            lines.append(f"| {r['metric']} | FAIL | – | – | – | {r.get('note')} | – | – | – | – |")
            continue
        lines.append(f"| {r['metric']} | OK | {r['first']} | {r['last']} | {r['missing_from_2018']:.2%} | "
                     f"{r['status']} | {r['gh_overlap']} | {r['gh_last']} | {r['gh_max_rel_diff']:.2e} | "
                     f"{r['gh_days_rel_diff_gt_1pct']} |")
    lines += ["", f"## Bước nhảy đột ngột của SplyExNtv (|robust z| > {JUMP_Z})", "",
              "Bước nhảy lớn trong lượng BTC trên sàn thường do CoinMetrics thêm/bớt ví của một sàn vào danh sách gắn nhãn, "
              "không phải dòng tiền thật. Các ngày này cần ghi vào Limitations.", "",
              "| Ngày | Δ SplyExNtv (BTC) | % lượng BTC trên sàn |", "|---|---|---|"]
    for d, r in jumps.sort_values("delta_btc", key=abs, ascending=False).head(20).iterrows():
        lines.append(f"| {d.date()} | {r['delta_btc']:,.0f} | {r['pct_of_exchange_supply']:.2f}% |")
    if len(jumps) > 20:
        lines.append(f"| … | còn {len(jumps) - 20} ngày | |")
    lines += ["", "![Chuỗi Flows](figures/flows_series.png)", "",
              "## Đề xuất", "",
              f"**{'GIỮ nhóm Flows (go)' if go else 'BỎ nhóm Flows (no-go)'}** — tiêu chí tự động: đủ 3 metric, "
              f"thiếu ≤ {MAX_MISSING:.0%} số ngày từ {CHECK_FROM}, có dữ liệu đến {END}. "
              "Quyết định cuối cùng do cả nhóm chốt ở họp go/no-go (task 1.19); "
              "phụ thuộc nhãn ví sàn phải được nêu trong Limitations dù giữ hay bỏ.", ""]
    (REPORTS / "flows_check.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
