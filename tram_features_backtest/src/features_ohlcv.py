"""
features_ohlcv.py  –  Task 1.13 (Trâm)
Đặc trưng OHLCV theo ngày cho bài toán dự báo hướng giá t+1.

Nguyên tắc: dòng ngày t chỉ dùng dữ liệu ĐẾN HẾT ngày t (UTC). Không dùng shift(-k).

Đầu vào (chỉ đọc data freeze, không đọc crawl_data/):
  data/processed/master_daily.csv     cột date, ohlcv_*        (giá ngày, cùng nguồn với nhãn y)
  data/processed/ohlcv_1h_clean.csv   cột datetime, close      (giá 1h, chỉ để tính RV)
  data/processed/MANIFEST.json        sha256: lệch thì dừng
Đầu ra:
  tram_features_backtest/data/features_ohlcv.csv : cột date + 11 đặc trưng (10 ban đầu + up_today)

Dùng trong pipeline của Ngọc (thay build_ohlcv_features tạm trong ngoc_pipeline/src/features.py):
  from features_ohlcv import build_ohlcv_features
  ohlcv = build_ohlcv_features(master)   # master = load_master(); trả cột date + 11 đặc trưng, cùng số dòng

Ghi chú giờ thiếu: file 1h thiếu 127 giờ (sàn bảo trì) và không được điền. Return 1h được tính giữa hai
nến liên tiếp có dữ liệu, nên phần biến động trong lúc sàn nghỉ vẫn được cộng vào RV của ngày chứa
nến sau, RV không bị đánh giá thấp ở những ngày có giờ thiếu.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]                    # thư mục gốc repo
PROCESSED = ROOT / "data/processed"
MASTER = PROCESSED / "master_daily.csv"
H1 = PROCESSED / "ohlcv_1h_clean.csv"
OUT = Path(__file__).resolve().parents[1] / "data/features_ohlcv.csv"

DATE = "date"
OHLCV_COLS = ["ohlcv_open", "ohlcv_high", "ohlcv_low", "ohlcv_close", "ohlcv_volume"]
# Bộ 10 đặc trưng ban đầu (task 1.13). Giữ tên này để các báo cáo cũ (baseline_check) tái tạo được.
FEATURES_10 = ["logret_lag1", "logret_lag2", "logret_lag3", "logret_lag4", "logret_lag5",
               "hl_range", "dlog_volume", "rsi_14", "rv_7d", "rv_30d"]
# 12/11/2026: thêm up_today = 1[return ngày t > 0]. Quy luật duy nhất tìm được (baseline_check) là đảo chiều theo
# DẤU return 1 ngày, trong khi 10 cột trên chỉ đo độ lớn. Đặt cuối danh sách để lrfs vẫn bắt đầu từ logret_lag1.
FEATURES = FEATURES_10 + ["up_today"]


# ---------------------------------------------------------------------------
# Các phép biến đổi
# ---------------------------------------------------------------------------
def log_return_lags(close: pd.Series, n_lags: int = 5) -> pd.DataFrame:
    """logret_lag1 = ln(C_t / C_{t-1}) là return của chính ngày t (cùng nghĩa ret_1 của Ngọc),
       logret_lagk = return của ngày t-k+1."""
    r = np.log(close / close.shift(1))
    return pd.DataFrame({f"logret_lag{k}": r.shift(k - 1) for k in range(1, n_lags + 1)})


def hl_range(high, low, close) -> pd.Series:
    return (high - low) / close


def dlog_volume(volume: pd.Series) -> pd.Series:
    """ln V_t − ln V_{t−1}. Ngày volume <= 0 (freeze hiện không có) thành NaN, không tự sửa dữ liệu."""
    return np.log(volume.where(volume > 0)).diff()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI kiểu Wilder (làm mượt hàm mũ alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def up_today(close: pd.Series) -> pd.Series:
    """1 nếu giá đóng cửa ngày t cao hơn ngày t−1, 0 nếu không; NaN ở ngày đầu chuỗi.
    Cùng thông tin với dấu của logret_lag1, nhưng ở dạng 0/1 để mô hình tuyến tính dùng thẳng quy luật đảo chiều
    (logret_lag1 đo độ lớn: ngày tăng 0,2% và ngày tăng 5% cách nhau rất xa, dù cùng là "ngày tăng")."""
    prev = close.shift(1)
    return (close > prev).astype(float).where(prev.notna())


def realized_vol(close_1h: pd.Series, windows=(7, 30)) -> pd.DataFrame:
    """Realized volatility từ log return 1h.
       RV_ngày(d) = tổng bình phương log return 1h trong ngày d (UTC)
       RV_k(t)    = sqrt( tổng RV_ngày của k ngày gần nhất, tính cả ngày t )
       Không năm hoá; nếu cần thì nhân sqrt(365/k)."""
    close_1h = close_1h.sort_index()
    r1h = np.log(close_1h / close_1h.shift(1)).dropna()
    rv_day = (r1h ** 2).groupby(r1h.index.floor("D")).sum()
    return pd.DataFrame({f"rv_{k}d": np.sqrt(rv_day.rolling(k, min_periods=k).sum())
                         for k in windows})


def build_features(daily: pd.DataFrame, close_1h: pd.Series) -> pd.DataFrame:
    """daily: index = ngày UTC (DatetimeIndex), cột open/high/low/close/volume, không có NaN."""
    daily = daily.sort_index()
    feats = [
        log_return_lags(daily["close"]),
        hl_range(daily["high"], daily["low"], daily["close"]).rename("hl_range"),
        dlog_volume(daily["volume"]).rename("dlog_volume"),
        rsi(daily["close"]).rename("rsi_14"),
        up_today(daily["close"]).rename("up_today"),
    ]
    out = pd.concat(feats, axis=1)
    out = out.join(realized_vol(close_1h), how="left")
    out.index.name = DATE
    return out[FEATURES]


# ---------------------------------------------------------------------------
# Đọc data freeze
# ---------------------------------------------------------------------------
def _sha256(path) -> str:
    """sha256 sau khi đổi CRLF -> LF. Trên Windows, Git (core.autocrlf=true) tự đổi xuống dòng của CSV
       sang CRLF khi checkout: số liệu y nguyên nhưng byte khác, nên phải chuẩn hoá trước khi so MANIFEST
       (MANIFEST được tạo trên máy dùng LF)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_freeze(names=("master_daily.csv", "ohlcv_1h_clean.csv"), directory=PROCESSED) -> str:
    """So sha256 với MANIFEST.json; lệch thì dừng (đang dùng sai phiên bản dữ liệu). Trả version."""
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [n for n in names if _sha256(directory / n) != manifest["files"][n]["sha256"]]
    if bad:
        raise ValueError(f"File không khớp MANIFEST {manifest['version']}: {bad}. "
                         "Kéo bản mới nhất (git pull) hoặc báo Khoa/Vi.")
    return manifest["version"]


def load_close_1h(path=H1) -> pd.Series:
    h1 = pd.read_csv(path)
    return pd.Series(h1["close"].to_numpy(), index=pd.to_datetime(h1["datetime"], utc=True), name="close")


def build_ohlcv_features(master: pd.DataFrame, close_1h: pd.Series | None = None) -> pd.DataFrame:
    """Giao diện cho pipeline của Ngọc: nhận master (cột date + ohlcv_*), trả DataFrame cột date
       (datetime64[ns], không múi giờ) + các cột FEATURES, cùng số dòng và thứ tự với master.
       Ngày chưa có giá (trước 17/08/2017) thì đặc trưng là NaN."""
    if close_1h is None:
        close_1h = load_close_1h()
    dates = pd.to_datetime(master[DATE]).astype("datetime64[ns]")
    daily = master[OHLCV_COLS].set_axis(pd.DatetimeIndex(dates).tz_localize("UTC"))
    daily = daily.rename(columns=lambda c: c.replace("ohlcv_", "")).dropna()
    feats = build_features(daily, close_1h)
    feats.index = feats.index.tz_localize(None)
    out = feats.reindex(pd.DatetimeIndex(dates)).reset_index(drop=True)
    out.insert(0, DATE, dates.to_numpy())
    return out


def load_and_build(out_path=OUT):
    version = verify_freeze()
    master = pd.read_csv(MASTER, parse_dates=[DATE])
    feats = build_ohlcv_features(master).dropna(subset=["hl_range"])   # bỏ 2016 – 16/08/2017 (chưa có giá)
    feats.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print(f"Data freeze {version}: sha256 khớp. Đã lưu {out_path}: {feats.shape}")
    print("Số NaN mỗi cột (do warm-up ở đầu chuỗi 2017):")
    print(feats.drop(columns=DATE).isna().sum().to_string())
    print("Từ 2017-12-31 (ngày train đầu của fold 0) còn NaN:",
          int(feats.loc[feats[DATE] >= "2017-12-31", FEATURES].isna().sum().sum()))
    return feats


# ----------------------------------------------------------------------------
# Test trên dữ liệu giả:  python features_ohlcv.py --test
# ----------------------------------------------------------------------------
def _fake_data(n_days=200, seed=1):
    rng = np.random.default_rng(seed)
    idx_h = pd.date_range("2020-01-01", periods=n_days * 24, freq="h", tz="UTC")
    c1h = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx_h)))), index=idx_h)
    g = c1h.groupby(c1h.index.floor("D"))
    daily = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(),
                          "close": g.last(),
                          "volume": rng.uniform(1e3, 5e3, n_days)})
    return daily, c1h


def _fake_master(daily: pd.DataFrame, n_before=10) -> pd.DataFrame:
    """Dạng master: cột date không múi giờ, có vài ngày đầu chưa có giá (NaN) như freeze thật."""
    m = daily.rename(columns=lambda c: f"ohlcv_{c}")
    m.index = m.index.tz_localize(None)
    before = pd.date_range(end=m.index[0] - pd.Timedelta(days=1), periods=n_before, freq="D")
    m = pd.concat([pd.DataFrame(index=before, columns=m.columns, dtype=float), m])
    return m.rename_axis(DATE).reset_index()


def _tests():
    daily, c1h = _fake_data()
    f = build_features(daily, c1h)

    # 1) Không nhìn tương lai: sửa dữ liệu SAU ngày cắt -> đặc trưng đến ngày cắt không đổi
    cut = daily.index[120]
    daily2, c1h2 = daily.copy(), c1h.copy()
    daily2.loc[daily2.index > cut, ["open", "high", "low", "close", "volume"]] *= 3
    c1h2[c1h2.index >= cut + pd.Timedelta(days=1)] *= 3
    f2 = build_features(daily2, c1h2)
    pd.testing.assert_frame_equal(f.loc[:cut], f2.loc[:cut])
    assert not f.loc[cut:].iloc[1:].equals(f2.loc[cut:].iloc[1:])      # phép thử có tác dụng

    # 2) Kiểm tra giá trị cụ thể
    t = daily.index[50]
    assert np.isclose(f.loc[t, "logret_lag1"],
                      np.log(daily.loc[t, "close"] / daily.loc[daily.index[49], "close"]))
    assert np.isclose(f.loc[t, "logret_lag2"], f.loc[daily.index[49], "logret_lag1"])
    assert np.isclose(f.loc[t, "dlog_volume"],
                      np.log(daily.loc[t, "volume"] / daily.loc[daily.index[49], "volume"]))
    assert f["rsi_14"].dropna().between(0, 100).all()
    # up_today: đúng dấu của logret_lag1, chỉ nhận 0/1, ngày đầu NaN
    ok = f["logret_lag1"].notna()
    assert (f.loc[ok, "up_today"] == (f.loc[ok, "logret_lag1"] > 0).astype(float)).all()
    assert set(f["up_today"].dropna().unique()) <= {0.0, 1.0} and np.isnan(f["up_today"].iloc[0])
    assert FEATURES[:len(FEATURES_10)] == FEATURES_10 and FEATURES[-1] == "up_today"
    assert (f["rv_30d"].dropna() > 0).all()

    # 3) Giao diện cho pipeline Ngọc: cột date + FEATURES, cùng số dòng/thứ tự với master,
    #    ngày chưa có giá thì NaN, giá trị khớp build_features
    m = _fake_master(daily)
    o = build_ohlcv_features(m, c1h)
    assert list(o.columns) == [DATE] + FEATURES and len(o) == len(m)
    assert o[DATE].dtype == "datetime64[ns]" and (o[DATE].to_numpy() == m[DATE].to_numpy()).all()
    assert o.iloc[:10, 1:].isna().all().all()
    np.testing.assert_allclose(o.iloc[10:, 1:].to_numpy(), f.to_numpy(), equal_nan=True)

    # 4) Giao diện cũng không nhìn tương lai
    m2 = m.copy()
    fut = m2[DATE] > cut.tz_localize(None)
    m2.loc[fut, OHLCV_COLS] *= 3
    o2 = build_ohlcv_features(m2, c1h2)
    pd.testing.assert_frame_equal(o.loc[~fut], o2.loc[~fut])

    # 5) verify_freeze bắt được file bị sửa
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "a.csv").write_text("x\n1\n")
        man = {"version": "vtest", "files": {"a.csv": {"sha256": _sha256(tmp / "a.csv")}}}
        (tmp / "MANIFEST.json").write_text(json.dumps(man))
        assert verify_freeze(["a.csv"], tmp) == "vtest"
        (tmp / "a.csv").write_bytes(b"x\r\n1\r\n")                     # Windows đổi xuống dòng: vẫn khớp
        assert verify_freeze(["a.csv"], tmp) == "vtest"
        (tmp / "a.csv").write_text("x\n2\n")
        try:
            verify_freeze(["a.csv"], tmp)
            raise AssertionError("Phải báo lỗi khi file lệch MANIFEST")
        except ValueError:
            pass

    print("Tất cả test features đều PASS")


if __name__ == "__main__":
    # python features_ohlcv.py         -> chạy test rồi tạo data/features_ohlcv.csv
    # python features_ohlcv.py --test  -> chỉ chạy test
    _tests()
    if "--test" not in sys.argv:
        load_and_build()
