"""Task 1 (Hafid 2024): tải nến 15 phút BTCUSDT 2021-02 .. 2022-01 từ Binance Data Vision + bài báo arXiv.

Chạy:
  python3 src/download.py          # tải zip (kiểm .CHECKSUM sha256), ghép CSV, ghi manifest, tải PDF
  python3 src/download.py --test   # chạy _tests()
load_15m(): DataFrame index = open_time, UTC **naive** (không gắn tz), cột open/high/low/close/volume (float),
sắp tăng dần, không trùng. Không nội suy nến thiếu.
"""
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # experiment_paper/hafid2024/
RAW = BASE_DIR / "data" / "raw"
CSV = BASE_DIR / "data" / "btcusdt_15m.csv"
MANIFEST = BASE_DIR / "data" / "manifest.json"
PDF = BASE_DIR / "paper" / "2410.06935v2.pdf"
PDF_URL = "https://arxiv.org/pdf/2410.06935v2"
BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/"
MONTHS = [p.strftime("%Y-%m") for p in pd.period_range("2021-02", "2022-01", freq="M")]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]
EXPECTED_ROWS = 365 * 96  # 2021-02-01 .. 2022-01-31 = 365 ngày


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


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def detect_ts_unit(v):
    """ms ~1.6e12, us ~1.6e15 (Binance đổi sang us từ 2025)."""
    return "us" if v > 1e14 else "ms"


def read_kline_zip(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df["open_time"] = df["open_time"].astype("int64")
    unit = detect_ts_unit(df["open_time"].iloc[0])
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit)  # UTC naive
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]], unit


def fetch_month(m):
    name = f"BTCUSDT-15m-{m}.zip"
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


def download_pdf():
    try:
        data = http_get(PDF_URL)
        if not data.startswith(b"%PDF"):
            raise RuntimeError("phản hồi không phải PDF")
        PDF.parent.mkdir(parents=True, exist_ok=True)
        PDF.write_bytes(data)
        return {"status": "ok", "bytes": len(data), "sha256": sha256(data)}
    except Exception as e:
        print(f"Không tải được PDF: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    files, frames = [], []
    for m in MONTHS:
        zpath, digest, status = fetch_month(m)
        df, unit = read_kline_zip(zpath)
        frames.append(df)
        files.append({"month": m, "file": zpath.name, "sha256": digest, "status": status,
                      "rows": len(df), "ts_unit": unit})
        print(f"{m}: {status:10s} {len(df):5d} nến, timestamp={unit}")
    full = (pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
            .reset_index(drop=True))
    full = full[(full["open_time"] >= "2021-02-01") & (full["open_time"] < "2022-02-01")]
    full.to_csv(CSV, index=False)
    grid = pd.date_range("2021-02-01 00:00", "2022-01-31 23:45", freq="15min")
    missing = grid.difference(pd.DatetimeIndex(full["open_time"]))
    pdf = download_pdf()
    meta = {"source": BASE, "symbol": "BTCUSDT", "interval": "15m",
            "crawled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(full), "expected_rows": EXPECTED_ROWS, "missing_candles": len(missing),
            "missing_timestamps": [str(t) for t in missing[:200]],
            "csv_sha256": sha256(CSV.read_bytes()), "files": files, "paper_pdf": pdf}
    MANIFEST.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nXong: {len(full)} dòng, thiếu {len(missing)} nến. Manifest: {MANIFEST}")


def load_15m():
    """Đọc CSV -> DataFrame index open_time (UTC naive), float, tăng dần, không trùng."""
    df = pd.read_csv(CSV, parse_dates=["open_time"]).set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _tests():
    assert MANIFEST.exists() and CSV.exists(), "chưa có dữ liệu: chạy download.py trước"
    man = json.loads(MANIFEST.read_text())
    df = load_15m()
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[0] == pd.Timestamp("2021-02-01 00:00"), df.index[0]
    assert df.index[-1] == pd.Timestamp("2022-01-31 23:45"), df.index[-1]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df.dtypes == "float64").all()
    grid = pd.date_range("2021-02-01 00:00", "2022-01-31 23:45", freq="15min")
    assert len(grid) == EXPECTED_ROWS
    assert df.index.isin(grid).all(), "có nến ngoài lưới 15 phút/khoảng thời gian"
    miss = grid.difference(df.index)
    assert len(miss) == man["missing_candles"], (len(miss), man["missing_candles"])
    assert len(df) == EXPECTED_ROWS - len(miss)
    rec = man["missing_timestamps"]
    assert rec == [str(t) for t in miss[:len(rec)]], "missing_timestamps không khớp"
    assert len(man["files"]) == 12
    frames = []
    for f in man["files"]:
        z = RAW / f["file"]
        assert sha256(z.read_bytes()) == f["sha256"], f"sai sha256: {f['file']}"
        assert sha256(z.read_bytes()) == (RAW / (f["file"] + ".CHECKSUM")).read_text().split()[0]
        d, _ = read_kline_zip(z)
        assert len(d) == f["rows"], f"số dòng tháng {f['month']} lệch manifest"
        frames.append(d)
    ref = (pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
           .reset_index(drop=True))
    ref = ref[(ref["open_time"] >= "2021-02-01") & (ref["open_time"] < "2022-02-01")]
    ref = ref.set_index("open_time")
    pd.testing.assert_frame_equal(df, ref, check_freq=False)
    assert sha256(CSV.read_bytes()) == man["csv_sha256"], "CSV sai sha256 so với manifest"
    assert not df.isna().any().any()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] >= 0).all()
    assert PDF.exists() and PDF.read_bytes().startswith(b"%PDF"), "PDF thiếu/hỏng"
    assert man["paper_pdf"]["status"] == "ok", man["paper_pdf"]
    print(f"Tất cả test download đều PASS ({len(df)} dòng, thiếu {man['missing_candles']} nến)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    if ap.parse_args().test:
        _tests()
        sys.exit(0)
    main()
