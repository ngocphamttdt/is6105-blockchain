"""Góp ý 5 của thầy (Vi): Flows có thực sự có sẵn đúng lúc không, và bị CoinMetrics sửa lịch sử bao nhiêu.

Chạy: /opt/miniconda3/bin/python src/flows_vintage.py   (từ data/raw/vi_ohlcv_flows/)
- Tải lại FlowInExNtv, FlowOutExNtv, SplyExNtv (2016-01-01 → 2026-08-31) kèm `status` và `status-time`
  (thời điểm CoinMetrics tính ra giá trị của ngày đó). Lưu thành một vintage mới trong data/vintages/<ngày tải>/;
  không đụng data/raw/ hay data freeze.
- So vintage mới với giá trị trong data freeze v1.0 (master_daily.csv, tải 24/10/2026): ngày nào đã đổi, đổi bao nhiêu.
- Độ trễ công bố: status-time − 00:00 UTC ngày d+1. Ngày cũ có status-time gần đây nghĩa là đã được tính lại.
- Đẳng thức ΔSply = In − Out trên vintage mới: nếu phần 2026 đã được tính lại, đẳng thức sẽ đúng thêm nhiều tháng.
Kết quả: reports/flows_vintage.md, reports/figures/flows_vintage.png.
"""
import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from crawl_binance import OUT, http_get

API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["FlowInExNtv", "FlowOutExNtv", "SplyExNtv"]
START, END = "2016-01-01", "2026-08-31"
FREEZE = OUT.parents[2] / "data" / "processed" / "master_daily.csv"
FREEZE_DOWNLOADED = "2026-10-24"        # reports/flows_check.md
REPORTS = OUT / "reports"
TOL_BTC = 1.0


def fetch(metric):
    params = {"assets": "btc", "metrics": metric, "frequency": "1d", "start_time": START, "end_time": END,
              "page_size": 10000}
    url, rows = API + "?" + urllib.parse.urlencode(params), []
    while url:
        out = json.loads(http_get(url))
        rows.extend(out["data"])
        url = out.get("next_page_url")
        time.sleep(0.7)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return pd.DataFrame({"date": df["date"], metric: pd.to_numeric(df[metric]),
                         f"{metric}_status": df.get(f"{metric}-status"),
                         f"{metric}_status_time": pd.to_datetime(df.get(f"{metric}-status-time"), utc=True, format="ISO8601")})


def main():
    stamp = datetime.now(timezone.utc)
    vdir = OUT / "data" / "vintages" / stamp.strftime("%Y-%m-%d")
    vdir.mkdir(parents=True, exist_ok=True)
    path = vdir / "flows_vintage.csv"
    if path.exists():                    # đã tải hôm nay: dùng lại, không gọi API lần nữa
        new = pd.read_csv(path, parse_dates=["date"])
        for m in METRICS:
            new[f"{m}_status_time"] = pd.to_datetime(new[f"{m}_status_time"], utc=True, format="ISO8601")
    else:
        new = fetch(METRICS[0])
        for m in METRICS[1:]:
            new = new.merge(fetch(m), on="date", how="outer")
        new = new.sort_values("date").reset_index(drop=True)
        new.to_csv(path, index=False)

    old = pd.read_csv(FREEZE, parse_dates=["date"])[["date"] + [f"cm_{m}" for m in METRICS]]
    j = old.merge(new, on="date", how="inner")
    md = ["# Flows: thời điểm có sẵn và mức bị sửa lịch sử (góp ý 5)", "",
          f"Vintage mới tải lúc {stamp:%Y-%m-%d %H:%M} UTC, so với data freeze v1.0 (tải {FREEZE_DOWNLOADED}). "
          f"{len(j):,} ngày chung, {START} → {END}. Ngưỡng coi là \"đã đổi\": lệch > {TOL_BTC:g} BTC.", ""]

    # 1) Giá trị đã đổi giữa hai lần tải
    rows = []
    for m in METRICS:
        d = (j[m] - j[f"cm_{m}"]).abs()
        ch = d > TOL_BTC
        by = j.assign(ch=ch, d=d).groupby(j["date"].dt.year).agg(n_changed=("ch", "sum"), max_abs_btc=("d", "max"))
        rows.append({"metric": m, "n_days": len(j), "n_changed": int(ch.sum()), "max_abs_btc": float(d.max()),
                     "first_changed": j.loc[ch, "date"].min().date() if ch.any() else "-",
                     "last_changed": j.loc[ch, "date"].max().date() if ch.any() else "-"})
        by.to_csv(vdir / f"changed_by_year_{m}.csv")
    ch_t = pd.DataFrame(rows)
    md += ["## 1. Giá trị nào đã đổi giữa hai lần tải", "", ch_t.to_markdown(index=False), ""]

    # 2) Độ trễ công bố và dấu hiệu tính lại
    lag_rows = []
    for m in METRICS:
        st = j[f"{m}_status_time"]
        lag_h = (st - (j["date"] + pd.Timedelta(days=1)).dt.tz_localize("UTC")).dt.total_seconds() / 3600
        recent = lag_h > 48
        lag_rows.append({"metric": m, "median_lag_h": float(lag_h[~recent].median()),
                         "p95_lag_h": float(lag_h[~recent].quantile(0.95)),
                         "max_lag_h_on_time": float(lag_h[~recent].max()),
                         "n_days_recomputed_later": int(recent.sum()),
                         "share_recomputed": float(recent.mean()),
                         "latest_status_time": str(st.max())})
        j[f"{m}_lag_h"] = lag_h
    lag_t = pd.DataFrame(lag_rows)
    by_year = j.assign(rec=j["SplyExNtv_lag_h"] > 48).groupby(j["date"].dt.year)["rec"].mean().mul(100).round(1)
    rec = j["SplyExNtv_lag_h"] > 48
    rec_when = j.loc[rec, "SplyExNtv_status_time"]
    last_rec, first_pit = j.loc[rec, "date"].max(), j.loc[~rec & (j["date"] > j.loc[rec, "date"].max()), "date"].min()
    md += ["## 2. Thời điểm dữ liệu có sẵn", "",
           f"**Kết luận chính.** Mọi ngày từ {j['date'].min().date()} đến {last_rec.date()} có giá trị hiện tại được "
           f"tính lại trong khoảng {rec_when.min():%d/%m/%Y} – {rec_when.max():%d/%m/%Y}: CoinMetrics đã tính lại "
           f"**toàn bộ lịch sử** một lần trong đợt đó. Từ {first_pit.date()} trở đi, giá trị là giá trị công bố lần đầu "
           f"(độ trễ trung vị {lag_t.loc[2, 'median_lag_h']:.1f} giờ sau 00:00 UTC ngày d+1, 95% trong vòng "
           f"{lag_t.loc[2, 'p95_lag_h']:.1f} giờ). Hệ quả: với mọi ngày test trước {first_pit.date()}, Flows trong freeze "
           "không phải giá trị có sẵn lúc dự báo; chỉ phần từ ngày đó mới đúng thời điểm (point-in-time). Dịch 1 ngày "
           "đủ cho độ trễ công bố, nhưng không sửa được việc tính lại này.", "",
           "`status-time` là lúc CoinMetrics tính ra giá trị hiện có của ngày d. Độ trễ = status-time − 00:00 UTC "
           "ngày d+1. Ngày có độ trễ ≤ 48 giờ: giá trị hiện tại là giá trị công bố lần đầu. Ngày có độ trễ > 48 giờ: "
           "giá trị đã được tính lại sau đó.", "", lag_t.round(2).to_markdown(index=False), "",
           "% số ngày của SplyExNtv đã được tính lại (độ trễ > 48 giờ), theo năm:", "",
           by_year.to_frame("% ngày tính lại").to_markdown(), ""]

    # 3) Đẳng thức ΔSply = In − Out trên hai vintage
    def identity(df, sply, fin, fout):
        e = (df[sply].diff() - (df[fin] - df[fout])).abs()
        return e
    e_old = identity(j, "cm_SplyExNtv", "cm_FlowInExNtv", "cm_FlowOutExNtv")
    e_new = identity(j, "SplyExNtv", "FlowInExNtv", "FlowOutExNtv")
    month = j["date"].dt.to_period("M").astype(str)
    idt = pd.DataFrame({"tháng": month, "lệch_freeze": e_old > TOL_BTC, "lệch_vintage_mới": e_new > TOL_BTC})
    idt = idt[j["date"] >= "2025-09-01"].groupby("tháng").sum().astype(int)
    md += ["## 3. Đẳng thức ΔSply = In − Out", "",
           f"Số ngày lệch > {TOL_BTC:g} BTC theo tháng (từ 09/2025). Nếu phần cuối chuỗi đã được tính lại, cột vintage "
           "mới sẽ về 0 ở những tháng mà cột freeze còn lệch.", "", idt.to_markdown(), ""]

    # 4) Hình
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter(j["date"], j["SplyExNtv_lag_h"].clip(upper=24 * 400) / 24, s=3, color="#2a78d6")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_ylabel("status-time − ngày d+1 (ngày)")
    ax.set_title("SplyExNtv: lúc CoinMetrics tính ra giá trị hiện có của từng ngày", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "flows_vintage.png", dpi=150)
    md += ["![status-time](figures/flows_vintage.png)", ""]
    (REPORTS / "flows_vintage.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
