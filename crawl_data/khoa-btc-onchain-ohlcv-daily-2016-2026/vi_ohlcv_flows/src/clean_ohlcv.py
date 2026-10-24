"""Task 1.9 (Vi): làm sạch nến 1h và gộp thành OHLCV daily theo UTC.

Chạy sau src/crawl_binance.py: /opt/miniconda3/bin/python src/clean_ohlcv.py
Không chuẩn hoá, không tính đặc trưng (thuộc về pipeline, fit trong từng fold).

Output (vi_ohlcv_flows/data/clean/):
- ohlcv_1h_clean.csv: nến 1h sau khi bỏ trùng và kiểm tra tính hợp lệ (để tính realized volatility)
- ohlcv_daily.csv: open = nến đầu, high = max, low = min, close = nến cuối, volume = tổng,
  n_candles, flag_incomplete (< 24 nến), flag_imputed (ngày không có nến nào)
- reports/ohlcv_clean_log.json: thống kê kiểm tra
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crawl_binance import OUT, RAW, ROOT, read_kline_zip

CLEAN = OUT / "data" / "clean"
REPORTS = OUT / "reports"
OLD_DAILY_API = ROOT / "data" / "btc_spot_1d_20180101_20261021.csv"
JUMP_1H = 0.10  # |log return 1h| > 10% thì đánh dấu để xem lại, không tự xoá


def load_1h():
    frames = [read_kline_zip(p)[0] for p in sorted(RAW.glob("BTCUSDT-1h-*.zip"))]
    df = pd.concat(frames, ignore_index=True)
    return df[["datetime", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]]


def clean_1h(df, log):
    log["rows_raw"] = len(df)
    # Sau sự cố Binance 08–09/02/2018, nến bị lệch +28m14s trong khoảng 43 giờ: làm tròn xuống giờ và đánh dấu
    df["flag_ts_shifted"] = df["datetime"] != df["datetime"].dt.floor("h")
    shifted = df.loc[df["flag_ts_shifted"], "datetime"]
    log["ts_shifted_rows"] = int(len(shifted))
    log["ts_shifted_range"] = [str(shifted.min()), str(shifted.max())] if len(shifted) else []
    df["datetime"] = df["datetime"].dt.floor("h")

    log["duplicates_removed"] = int(df["datetime"].duplicated().sum())
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")

    bad_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    log["bad_ohlc_rows"] = int(bad_ohlc.sum())
    assert not bad_ohlc.any(), "có nến vi phạm high/low"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "có giá <= 0"

    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    missing = grid.difference(df.index)
    log["hours_expected"] = len(grid)
    log["hours_missing"] = len(missing)
    log["missing_hours_by_day"] = (pd.Series(1, index=missing).groupby(missing.floor("D")).sum()
                                   .rename(lambda d: d.strftime("%Y-%m-%d")).to_dict())

    log["zero_volume_hours"] = (df.index[df["volume"] == 0].strftime("%Y-%m-%d %H:%M")).tolist()
    # nhảy giá tính trên lưới đủ giờ để không so 2 nến cách nhau nhiều giờ
    r = np.log(df["close"].reindex(grid)).diff()
    jumps = r[r.abs() > JUMP_1H]
    log["price_jumps_1h"] = {t.strftime("%Y-%m-%d %H:%M"): round(float(v), 4) for t, v in jumps.items()}
    return df


def to_daily(h, log):
    g = h.groupby(h.index.floor("D"))
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
        "quote_volume": g["quote_volume"].sum(), "n_trades": g["n_trades"].sum(),
        "n_candles": g.size(),
    })
    days = pd.date_range(d.index.min(), d.index.max(), freq="D", tz="UTC")
    d = d.reindex(days)
    # ngày không có nến nào: giữ giá đóng cửa hôm trước, volume = 0, đánh dấu imputed
    d["flag_imputed"] = d["close"].isna()
    prev_close = d["close"].ffill()
    for c in ["open", "high", "low", "close"]:
        d[c] = d[c].fillna(prev_close)
    d[["volume", "quote_volume", "n_trades", "n_candles"]] = d[["volume", "quote_volume", "n_trades", "n_candles"]].fillna(0)
    d["n_candles"] = d["n_candles"].astype(int)
    d["flag_incomplete"] = d["n_candles"] < 24
    d.index = d.index.tz_localize(None).rename("date")

    log["days"] = len(d)
    log["date_range"] = [str(d.index.min().date()), str(d.index.max().date())]
    log["days_imputed"] = d.index[d["flag_imputed"]].strftime("%Y-%m-%d").tolist()
    log["days_incomplete"] = {k.strftime("%Y-%m-%d"): int(v) for k, v in d.loc[d["flag_incomplete"], "n_candles"].items()}
    log["days_zero_volume"] = d.index[d["volume"] == 0].strftime("%Y-%m-%d").tolist()
    r = np.log(d["close"]).diff()
    log["top5_abs_daily_logret"] = {k.strftime("%Y-%m-%d"): round(float(v), 4)
                                    for k, v in r.reindex(r.abs().sort_values(ascending=False).index[:5]).items()}
    return d


def compare_with_api(d, log):
    """So với file 1d lấy từ REST API trước đây: hai nguồn cùng Binance nên phải gần như trùng."""
    if not OLD_DAILY_API.exists():
        return
    old = pd.read_csv(OLD_DAILY_API)
    old.index = pd.to_datetime(old["datetime"], utc=True).dt.tz_localize(None).dt.normalize()
    j = d[["close", "volume"]].join(old[["close", "volume"]], rsuffix="_api", how="inner")
    full = j[~d.loc[j.index, "flag_incomplete"]]
    rel = (full["close"] - full["close_api"]).abs() / full["close_api"]
    log["compare_rest_api_1d"] = {
        "overlap_days": len(j), "complete_days_compared": len(full),
        "max_rel_diff_close": float(rel.max()),
        "days_rel_diff_close_gt_1e-6": int((rel > 1e-6).sum()),
    }


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    log = {}
    h = clean_1h(load_1h(), log)
    d = to_daily(h, log)
    compare_with_api(d, log)

    h.index = h.index.rename("datetime")
    h.to_csv(CLEAN / "ohlcv_1h_clean.csv")
    d.to_csv(CLEAN / "ohlcv_daily.csv")
    (REPORTS / "ohlcv_clean_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))

    print(f"1h: {log['rows_raw']} dòng, trùng {log['duplicates_removed']}, "
          f"thiếu {log['hours_missing']}/{log['hours_expected']} giờ, "
          f"volume=0: {len(log['zero_volume_hours'])} giờ, nhảy >10%/1h: {len(log['price_jumps_1h'])}")
    print(f"daily: {log['days']} ngày {log['date_range']}, thiếu nến: {len(log['days_incomplete'])} ngày, "
          f"imputed: {len(log['days_imputed'])} ngày")
    print("so với REST API 1d:", log.get("compare_rest_api_1d"))


if __name__ == "__main__":
    main()
