"""
features_ohlcv.py  –  Task 1.13 (Trâm)
Đặc trưng OHLCV theo ngày cho bài toán dự báo hướng giá t+1.

Nguyên tắc: dòng ngày t chỉ dùng dữ liệu ĐẾN HẾT ngày t (UTC). Không dùng shift(-k).

Đầu vào:
  crawl_data/khoa-btc-onchain-ohlcv-daily-2016-2026/master_daily.csv  (cột ohlcv_*, task 1.7)
  crawl_data/vi_ohlcv_flows/data/clean/ohlcv_1h_clean.csv             (cột datetime, close, task 1.9)
Đầu ra:
  tram_features_backtest/data/features_ohlcv.csv : cột date + 10 đặc trưng

Ghi chú giờ thiếu: file 1h thiếu 127 giờ (sàn bảo trì). Return 1h được tính giữa hai nến
liên tiếp có dữ liệu, nên return qua khoảng trống vẫn được cộng vào RV của ngày chứa nến sau.
Như vậy RV không bị đánh giá thấp ở những ngày có giờ thiếu.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def log_return_lags(close: pd.Series, n_lags: int = 5) -> pd.DataFrame:
    """logret_lag1 = ln(C_t / C_{t-1}) (return của chính ngày t),
       logret_lag2 = return ngày t-1, ... đến lag5.
       -> nhóm cần thống nhất cách đặt tên 'trễ 1 ngày' với Ngọc/Khoa."""
    r = np.log(close / close.shift(1))
    return pd.DataFrame({f"logret_lag{k}": r.shift(k - 1) for k in range(1, n_lags + 1)})


def hl_range(high, low, close) -> pd.Series:
    return (high - low) / close


def dlog_volume(volume: pd.Series) -> pd.Series:
    # +1 để tránh log(0) nếu có ngày volume = 0 (Vi đã đánh dấu những ngày này)
    return np.log(volume + 1).diff()


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


def realized_vol(close_1h: pd.Series, windows=(7, 30)) -> pd.DataFrame:
    """Realized volatility từ log return 1h.
       RV_ngày(d)  = tổng bình phương log return 1h trong ngày d (UTC)
       RV_k(t)     = sqrt( tổng RV_ngày của k ngày gần nhất, tính cả ngày t )
       Không năm hoá; nếu cần thì nhân sqrt(365/k)."""
    close_1h = close_1h.sort_index()
    r1h = np.log(close_1h / close_1h.shift(1)).dropna()
    rv_day = (r1h ** 2).groupby(r1h.index.floor("D")).sum()
    return pd.DataFrame({f"rv_{k}d": np.sqrt(rv_day.rolling(k, min_periods=k).sum())
                         for k in windows})


def build_features(daily: pd.DataFrame, close_1h: pd.Series) -> pd.DataFrame:
    """daily: index = ngày UTC (DatetimeIndex), cột open/high/low/close/volume."""
    daily = daily.sort_index()
    feats = [
        log_return_lags(daily["close"]),
        hl_range(daily["high"], daily["low"], daily["close"]).rename("hl_range"),
        dlog_volume(daily["volume"]).rename("dlog_volume"),
        rsi(daily["close"]).rename("rsi_14"),
    ]
    out = pd.concat(feats, axis=1)
    out = out.join(realized_vol(close_1h), how="left")
    out.index.name = "date"
    return out


ROOT = Path(__file__).resolve().parents[2]          # thư mục gốc repo
MASTER = ROOT / "crawl_data/khoa-btc-onchain-ohlcv-daily-2016-2026/master_daily.csv"
H1 = ROOT / "crawl_data/vi_ohlcv_flows/data/clean/ohlcv_1h_clean.csv"
OUT = Path(__file__).resolve().parents[1] / "data/features_ohlcv.csv"


def load_and_build(master_path=MASTER, h1_path=H1, out_path=OUT):
    """Giá ngày lấy từ master_daily.csv (Khoa, cột ohlcv_*) để khớp với nhãn y;
       giá 1h lấy từ ohlcv_1h_clean.csv (Vi) để tính realized volatility."""
    m = pd.read_csv(master_path, parse_dates=["date"])
    m.index = pd.DatetimeIndex(m["date"]).tz_localize("UTC")
    daily = (m[["ohlcv_open", "ohlcv_high", "ohlcv_low", "ohlcv_close", "ohlcv_volume"]]
             .rename(columns=lambda c: c.replace("ohlcv_", ""))
             .dropna())                                  # bỏ 2016 – 08/2017 (chưa có Binance)
    h1 = pd.read_csv(h1_path)
    h1.index = pd.to_datetime(h1["datetime"], utc=True)
    feats = build_features(daily, h1["close"])
    feats.index = feats.index.tz_localize(None).date     # ghi ngày dạng YYYY-MM-DD như master
    feats.index.name = "date"
    feats.to_csv(out_path)
    print(f"Đã lưu {out_path}: {feats.shape}")
    print("Số NaN mỗi cột (do warm-up ở đầu chuỗi 2017):")
    print(feats.isna().sum().to_string())
    print("Từ 2018-01-01 còn NaN:", int(feats.loc[pd.Timestamp('2018-01-01').date():].isna().sum().sum()))
    return feats


# ----------------------------------------------------------------------------
# Test trên dữ liệu giả:  python features_ohlcv.py
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

    # 2) Kiểm tra giá trị cụ thể
    t = daily.index[50]
    assert np.isclose(f.loc[t, "logret_lag1"],
                      np.log(daily.loc[t, "close"] / daily.loc[daily.index[49], "close"]))
    assert np.isclose(f.loc[t, "logret_lag2"], f.loc[daily.index[49], "logret_lag1"])
    assert f["rsi_14"].dropna().between(0, 100).all()
    assert (f["rv_30d"].dropna() > 0).all()

    print("Tất cả test features đều PASS")
    print(f.dropna().head().round(4))


if __name__ == "__main__":
    # python features_ohlcv.py         -> chạy test rồi tạo data/features_ohlcv.csv
    # python features_ohlcv.py --test  -> chỉ chạy test
    _tests()
    if "--test" not in sys.argv:
        load_and_build()
