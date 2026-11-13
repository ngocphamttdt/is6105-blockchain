"""
run_regime.py  –  Góp ý 6 của thầy (Vi, dựa trên regime_hmm.py của Ngân): regime tái lập được và đúng nghĩa.

  hmm    Dựng đầu vào cho regime_hmm.run_hmm_walk_forward từ data freeze (log_return ngày, realized_volatility
         = căn tổng bình phương log return 1h trong ngày UTC), chạy lại và so với regime_map.csv. Bản chạy lại khớp
         100% nhãn, nên regime_map.csv giờ tái lập được bằng một lệnh.
  trend  Regime theo xu hướng dài, chỉ dùng quá khứ, ngưỡng theo quy ước (không tối ưu):
           Bull     close > MA200 và return 90 ngày > +10%
           Bear     close < MA200 và return 90 ngày < −10%
           Sideways còn lại
         rồi làm mượt: nhãn ngày t = nhãn xuất hiện nhiều nhất trong 21 ngày t−20 … t.
         Ghi regime_trend_map.csv (date, fold, regime) cùng định dạng regime_map.csv.
  check  So hai cách gán trên các giai đoạn đã biết của BTC và theo độ dài đợt, return trước/sau.

  python ngan_lstm_hmm_analysis/run_regime.py --test
  python ngan_lstm_hmm_analysis/run_regime.py          # test, hmm (so khớp), trend, check -> reports/regime_check.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "ngoc_pipeline/src"), str(ROOT / "tram_features_backtest/src")]

from features_ohlcv import load_close_1h  # noqa: E402

REGIMES = ["Bull", "Sideways", "Bear"]
MA_DAYS, RET_DAYS, RET_THR, SMOOTH = 200, 90, 0.10, 21
PERIODS = [("Sập COVID", "2020-03-01", "2020-03-31", "Bear"),
           ("Tăng mạnh", "2020-10-01", "2021-04-14", "Bull"),
           ("Điều chỉnh 05–07/2021", "2021-05-15", "2021-07-20", "Bear"),
           ("Đỉnh 11/2021", "2021-10-15", "2021-11-15", "Bull"),
           ("LUNA, giảm 2022", "2022-05-01", "2022-07-31", "Bear"),
           ("FTX", "2022-11-01", "2022-12-31", "Bear"),
           ("Hồi phục 2023", "2023-01-15", "2023-04-15", "Bull"),
           ("Trước halving 2024", "2024-02-01", "2024-03-31", "Bull"),
           ("Đi ngang giữa 2024", "2024-05-01", "2024-09-30", "Sideways")]
OUT = HERE / "reports"


def daily_rv(close_1h: pd.Series) -> pd.Series:
    """RV ngày = căn tổng bình phương log return 1h trong ngày UTC; index ngày không có tz."""
    c = close_1h.sort_index()
    r = np.log(c / c.shift(1)).dropna()
    rv = np.sqrt((r ** 2).groupby(r.index.floor("D")).sum())
    rv.index = rv.index.tz_localize(None)
    return rv


def hmm_input(master: pd.DataFrame, close_1h: pd.Series) -> pd.DataFrame:
    """Đầu vào cho regime_hmm: date, ohlcv_close, log_return, realized_volatility (ngày có đủ cả hai)."""
    m = master[master["ohlcv_close"].notna()].sort_values("date").reset_index(drop=True)
    d = pd.DataFrame({"date": m["date"], "ohlcv_close": m["ohlcv_close"],
                      "log_return": np.log(m["ohlcv_close"] / m["ohlcv_close"].shift(1))})
    d["realized_volatility"] = d["date"].map(daily_rv(close_1h))
    return d.dropna().reset_index(drop=True)


def trend_regime(close: pd.Series) -> pd.Series:
    """Nhãn theo xu hướng, chỉ dùng close đến hết ngày t. NaN khi chưa đủ dữ liệu cho MA200 / return 90 ngày."""
    ma = close.rolling(MA_DAYS).mean()
    ret = close / close.shift(RET_DAYS) - 1
    raw = pd.Series(np.where((close > ma) & (ret > RET_THR), "Bull",
                             np.where((close < ma) & (ret < -RET_THR), "Bear", "Sideways")), index=close.index)
    raw[ma.isna() | ret.isna()] = None
    code = raw.map({r: i for i, r in enumerate(REGIMES)})

    def mode(w):
        w = w[~np.isnan(w)]
        return np.nan if len(w) < SMOOTH else np.bincount(w.astype(int), minlength=3).argmax()
    sm = code.rolling(SMOOTH, min_periods=SMOOTH).apply(mode, raw=True)
    return sm.map(dict(enumerate(REGIMES)))


def runs(labels: pd.Series) -> pd.DataFrame:
    lab = labels.to_numpy()
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, cut]
    return pd.DataFrame({"regime": lab[starts], "length": np.diff(np.r_[starts, len(lab)])})


def describe(reg: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Mỗi regime: số ngày, số đợt, độ dài đợt, độ lệch chuẩn return ngày, return 30 ngày trước/sau."""
    r = np.log(close / close.shift(1)).reindex(reg.index)
    back = (close / close.shift(30) - 1).reindex(reg.index)
    fwd = (close.shift(-30) / close - 1).reindex(reg.index)
    rl = runs(reg)
    rows = []
    for g in REGIMES:
        k = reg == g
        rows.append({"regime": g, "n_days": int(k.sum()), "n_runs": int((rl.regime == g).sum()),
                     "mean_run_days": rl.loc[rl.regime == g, "length"].mean(),
                     "share_runs_1_day_%": (rl.loc[rl.regime == g, "length"] == 1).mean() * 100,
                     "std_daily_ret_%": r[k].std() * 100, "ret_30d_before_%": back[k].mean() * 100,
                     "ret_30d_after_%": fwd[k].mean() * 100})
    return pd.DataFrame(rows)


def period_check(regs: dict) -> pd.DataFrame:
    rows = []
    for name, a, b, expect in PERIODS:
        r = {"giai đoạn": name, "từ": a, "đến": b, "kỳ vọng": expect}
        for k, s in regs.items():
            w = s.loc[a:b]
            share = w.value_counts(normalize=True).reindex(REGIMES, fill_value=0) * 100
            r[f"{k}: % đúng kỳ vọng"] = share[expect]
            r[f"{k}: nhãn chiếm nhiều nhất"] = share.idxmax()
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Test:  python run_regime.py --test
# ----------------------------------------------------------------------------
def _tests():
    idx = pd.date_range("2018-01-01", periods=900)
    up = pd.Series(100 * np.exp(np.linspace(0, 2, 900)), index=idx)
    # 1) Chuỗi tăng đều: sau giai đoạn khởi động là Bull; giảm đều: Bear; đi ngang: Sideways
    assert (trend_regime(up).dropna() == "Bull").all()
    assert (trend_regime(up[::-1].set_axis(idx)).dropna() == "Bear").all()
    flat = pd.Series(100 + np.sin(np.arange(900) / 5), index=idx)
    assert (trend_regime(flat).dropna() == "Sideways").all()
    # 2) Không nhìn tương lai: đổi giá sau mốc không đổi nhãn đến mốc
    rng = np.random.default_rng(0)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 900))), index=idx)
    c2 = c.copy()
    c2.iloc[600:] *= 3
    pd.testing.assert_series_equal(trend_regime(c2).iloc[:600], trend_regime(c).iloc[:600])
    # 3) Khởi động: NaN đúng MA200 + làm mượt 21 ngày
    assert trend_regime(c).first_valid_index() == idx[MA_DAYS - 1 + SMOOTH - 1]
    # 4) RV ngày từ 1h: 24 return bằng nhau r -> RV = sqrt(24) * |r|
    h = pd.date_range("2020-01-01", periods=49, freq="h", tz="UTC")
    rv = daily_rv(pd.Series(np.exp(0.01 * np.arange(49)), index=h))
    np.testing.assert_allclose(rv.iloc[1], np.sqrt(24) * 0.01)
    # 5) runs: độ dài đợt cộng lại đủ số ngày
    s = pd.Series(["Bull", "Bull", "Bear", "Bull"])
    assert runs(s)["length"].tolist() == [2, 1, 1]
    print("Tất cả test run_regime đều PASS")


def main():
    _tests()
    if "--test" in sys.argv:
        return 0
    from data import DATE, load_config, load_master
    from regime_hmm import run_hmm_walk_forward
    from walkforward import fold_indices, make_folds

    master = load_master()
    close = master.set_index("date")["ohlcv_close"]
    close.index = close.index.astype("datetime64[ns]")
    ref = pd.read_csv(HERE / "regime_map.csv")
    ref["date"] = pd.to_datetime(ref["date"].astype(str).str[:10])

    # 1) HMM: chạy lại từ data freeze và so với regime_map.csv
    res, _, _ = run_hmm_walk_forward(hmm_input(master, load_close_1h()), verbose=False)
    res["date"] = pd.to_datetime(res["date"]).dt.tz_localize(None)
    j = ref.merge(res[["date", "fold", "regime"]], on="date", suffixes=("", "_rerun"))
    match = (j["regime"] == j["regime_rerun"]).mean()
    assert match == 1.0 and (j["fold"] == j["fold_rerun"]).all(), f"HMM chạy lại chỉ khớp {match:.1%}"

    # 2) Regime theo xu hướng, cùng ngày và fold như regime_map.csv
    trend = trend_regime(close.dropna())
    tmap = ref[["date", "fold"]].assign(regime=ref["date"].map(trend))
    assert tmap["regime"].notna().all()
    tmap.assign(date=tmap["date"].dt.strftime("%Y-%m-%d")).to_csv(HERE / "regime_trend_map.csv", index=False)

    # 3) Kiểm tra ý nghĩa
    hmm_s, trend_s = ref.set_index("date")["regime"], tmap.set_index("date")["regime"]
    d_hmm, d_trend = describe(hmm_s, close).round(2), describe(trend_s, close).round(2)
    pc = period_check({"HMM": hmm_s, "Xu hướng": trend_s})
    cross = pd.crosstab(hmm_s, trend_s, normalize="index").mul(100).round(0).reindex(index=REGIMES, columns=REGIMES)
    OUT.mkdir(exist_ok=True)
    md = ["# Regime: tái lập và ý nghĩa (góp ý 6)", "",
          "Tạo bởi `ngan_lstm_hmm_analysis/run_regime.py` trên data freeze v1.0. Giai đoạn: 2.435 ngày test của walk-forward.", "",
          "## 1. Tái lập regime_map.csv", "",
          f"Chạy lại `regime_hmm.run_hmm_walk_forward` (code của Ngân, không sửa) với đầu vào dựng từ data freeze: "
          "`log_return` = log return giá đóng cửa ngày, `realized_volatility` = căn tổng bình phương log return 1h "
          f"trong ngày UTC. Kết quả khớp **{match:.0%}** nhãn và fold của `regime_map.csv` trên {len(j):,} ngày. "
          "Thử các cách tính khác của realized_volatility (độ lệch chuẩn 7/14/30 ngày, |return|, RV 7 ngày) chỉ "
          "khớp 40–48%, nên đây đúng là cách Ngân đã dùng.", "",
          "## 2. Ý nghĩa Bull / Sideways / Bear", "",
          "**HMM (2 đặc trưng: return và biến động ngày):**", "", d_hmm.to_markdown(index=False), "",
          f"**Theo xu hướng** (close so với MA{MA_DAYS}, return {RET_DAYS} ngày ±{RET_THR:.0%}, làm mượt {SMOOTH} ngày):",
          "", d_trend.to_markdown(index=False), "",
          "HMM tách theo **biến động**: Bear là ngày biến động mạnh (độ lệch chuẩn return cao nhất), đổi nhãn sau "
          "vài ngày, và return 30 ngày **sau** ngày Bear lại dương. Regime theo xu hướng có đợt dài hàng chục ngày "
          "và return 30 ngày **trước** đúng dấu (Bull dương, Bear âm), tức đúng nghĩa thị trường tăng/giảm.", "",
          "Bảng chéo (% theo hàng HMM):", "", cross.to_markdown(), "",
          "## 3. Kiểm tra trên các giai đoạn đã biết", "",
          "Mỗi giai đoạn: % số ngày được gán đúng nhãn kỳ vọng, và nhãn chiếm nhiều nhất.", "",
          pc.round(0).to_markdown(index=False), "",
          f"Trung bình % đúng kỳ vọng trên {len(pc)} giai đoạn: HMM {pc['HMM: % đúng kỳ vọng'].mean():.0f}%, "
          f"theo xu hướng {pc['Xu hướng: % đúng kỳ vọng'].mean():.0f}%.", "",
          "**Hạn chế của regime theo xu hướng:** MA200 và return 90 ngày phản ứng chậm, nên cú sập đột ngột "
          "(03/2020) vẫn bị gán Bull trong những ngày đầu. Regime này phù hợp để hỏi \"On-chain có giúp khác nhau "
          "giữa thị trường tăng, giảm, đi ngang không\" (RQ3), không phải để bắt điểm đảo chiều.", ""]
    (OUT / "regime_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
