"""Task 1.8 (Vi): tải nến 1h BTCUSDT từ Binance Data Vision (file zip theo tháng).

Chạy: /opt/miniconda3/bin/python src/crawl_binance.py [--start 2017-09] [--end 2026-08]
- Mỗi file zip được kiểm bằng .CHECKSUM (sha256) đi kèm; sai checksum thì tải lại, vẫn sai thì dừng.
- File gốc lưu bất biến ở vi_ohlcv_flows/data/raw/binance_1h/; đã có và đúng checksum thì bỏ qua.
- Ghi manifest (ngày tải, sha256, số dòng, đơn vị timestamp) để tái lập.
"""
import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent  # vi_ohlcv_flows/
ROOT = OUT.parent
RAW = OUT / "data" / "raw" / "binance_1h"
BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]


def http_get(url, retries=3, timeout=60):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  lỗi {e}, thử lại...")
            time.sleep(2 * (attempt + 1))


def months(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def detect_ts_unit(v):
    """Binance chuyển sang microsecond từ 2025: ms ~1.7e12, us ~1.7e15."""
    return "us" if v > 1e14 else "ms"


def read_kline_zip(path):
    """Đọc 1 file zip kline, trả về DataFrame với cột datetime UTC."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    if not str(df.iloc[0, 0]).isdigit():  # một số file có dòng header
        df = df.iloc[1:].reset_index(drop=True)
    df["open_time"] = df["open_time"].astype("int64")
    unit = detect_ts_unit(df["open_time"].iloc[0])
    df["datetime"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype("int64")
    return df, unit


def fetch_month(m):
    name = f"BTCUSDT-1h-{m}.zip"
    zpath = RAW / name
    expected = http_get(BASE + name + ".CHECKSUM").decode().split()[0]
    if zpath.exists() and sha256(zpath.read_bytes()) == expected:
        return zpath, expected, "cached"
    for _ in range(2):
        data = http_get(BASE + name)
        if sha256(data) == expected:
            zpath.write_bytes(data)
            (RAW / (name + ".CHECKSUM")).write_text(f"{expected}  {name}\n")
            return zpath, expected, "downloaded"
        print(f"  {name}: sai checksum, tải lại")
    raise RuntimeError(f"{name}: checksum không khớp sau 2 lần tải")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-09")
    ap.add_argument("--end", default="2026-08")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    manifest = []
    for m in months(args.start, args.end):
        zpath, digest, status = fetch_month(m)
        df, unit = read_kline_zip(zpath)
        manifest.append({"month": m, "file": zpath.name, "sha256": digest, "status": status,
                         "rows": len(df), "ts_unit": unit,
                         "first": str(df["datetime"].min()), "last": str(df["datetime"].max())})
        print(f"{m}: {status:10s} {len(df):4d} nến, timestamp={unit}")
        time.sleep(0.1)

    meta = {"source": BASE, "symbol": "BTCUSDT", "interval": "1h",
            "range": [args.start, args.end],
            "crawled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": manifest}
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "binance_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    n_us = sum(f["ts_unit"] == "us" for f in manifest)
    print(f"\nXong {len(manifest)} tháng, {sum(f['rows'] for f in manifest)} nến; "
          f"{n_us} file dùng microsecond. Manifest: reports/binance_manifest.json")


if __name__ == "__main__":
    main()
