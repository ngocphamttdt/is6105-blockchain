"""
backtest.py  –  Task 1.12 (Trâm)
Backtest chiến lược long/flat từ xác suất dự báo p̂.

Quy ước thời gian (rất quan trọng):
  - p_hat tại dòng ngày t  = xác suất "giá ngày t+1 tăng", dự báo lúc cuối ngày t.
  - Vị thế cho ngày t+1    = 1 (mua giữ) nếu p_hat_t > 0.5, ngược lại 0 (cầm tiền).
  - Lợi nhuận ngày t+1     = vị thế_{t+1} * return_{t+1} - phí * |vị thế_{t+1} - vị thế_t|
  - return_{t+1}           = C_{t+1} / C_t - 1 (return đơn giản, không phải log return)
"""
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.001          # 0,1% mỗi lần đổi vị thế
DAYS_PER_YEAR = 365  # crypto giao dịch cả tuần


def run_backtest(close: pd.Series, p_hat: pd.Series,
                 threshold: float = 0.5, fee: float = FEE) -> pd.DataFrame:
    """
    close : giá đóng cửa ngày, index = ngày (DatetimeIndex, UTC)
    p_hat : xác suất dự báo, index = ngày t (dự báo cho t+1). Có thể chỉ phủ giai đoạn test.
    Trả về DataFrame index = ngày t+1 với các cột:
        position, ret, cost, strat_ret, equity, bh_ret, bh_equity
    """
    df = pd.DataFrame({"close": close}).sort_index()
    df["p_hat"] = p_hat.reindex(df.index)

    # Tín hiệu tạo ở cuối ngày t -> áp dụng cho ngày t+1  => shift(1)
    signal = (df["p_hat"] > threshold).astype(float).where(df["p_hat"].notna())
    df["position"] = signal.shift(1)
    df["ret"] = df["close"].pct_change()

    # Chỉ giữ những ngày có vị thế (tức là ngày t+1 của giai đoạn test)
    out = df.dropna(subset=["position", "ret"]).copy()

    # Phí: lần đổi vị thế đầu tiên tính từ trạng thái cầm tiền (0)
    prev_pos = out["position"].shift(1).fillna(0.0)
    out["cost"] = fee * (out["position"] - prev_pos).abs()
    out["strat_ret"] = out["position"] * out["ret"] - out["cost"]
    out["equity"] = (1 + out["strat_ret"]).cumprod()

    # Buy-and-hold: mua ngày đầu, trả phí 1 lần, giữ đến hết
    bh = out["ret"].copy()
    bh.iloc[0] -= fee
    out["bh_ret"] = bh
    out["bh_equity"] = (1 + bh).cumprod()
    return out[["position", "ret", "cost", "strat_ret", "equity", "bh_ret", "bh_equity"]]


def sharpe(daily_ret: pd.Series) -> float:
    """Sharpe năm hoá, lãi suất phi rủi ro = 0."""
    sd = daily_ret.std(ddof=1)
    return float("nan") if sd == 0 or np.isnan(sd) else daily_ret.mean() / sd * np.sqrt(DAYS_PER_YEAR)


def max_drawdown(equity: pd.Series) -> float:
    """MDD là số âm, ví dụ -0.35 = sụt 35% từ đỉnh."""
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """Bảng chỉ số cho chiến lược và buy-and-hold."""
    rows = {}
    for name, r, eq in [("strategy", bt["strat_ret"], bt["equity"]),
                        ("buy_and_hold", bt["bh_ret"], bt["bh_equity"])]:
        rows[name] = {
            "cum_return": eq.iloc[-1] - 1,
            "sharpe": sharpe(r),
            "mdd": max_drawdown(eq),
        }
    rows["strategy"]["n_trades"] = int((bt["cost"] > 0).sum())
    rows["strategy"]["pct_days_long"] = bt["position"].mean()
    rows["buy_and_hold"]["n_trades"] = 1
    rows["buy_and_hold"]["pct_days_long"] = 1.0
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# Test nhanh trên dữ liệu giả:  python backtest.py
# ----------------------------------------------------------------------------
def _fake_close(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n))), index=idx)


def _tests():
    close = _fake_close()
    test_idx = close.index[100:]

    # 1) Luôn long  ==  buy-and-hold (cùng trả 1 lần phí)
    bt = run_backtest(close, pd.Series(0.9, index=test_idx))
    assert np.allclose(bt["equity"], bt["bh_equity"]), "Luôn long phải bằng buy-and-hold"
    assert (bt["cost"] > 0).sum() == 1

    # 2) Luôn flat  ==  equity đứng yên 1.0, không phí
    bt = run_backtest(close, pd.Series(0.1, index=test_idx))
    assert np.allclose(bt["equity"], 1.0) and bt["cost"].sum() == 0

    # 3) Dự báo hoàn hảo phải thắng buy-and-hold  (kiểm tra shift đúng chiều)
    perfect = (close.shift(-1) > close).astype(float).loc[test_idx].iloc[:-1]
    bt = run_backtest(close, perfect)
    assert bt["equity"].iloc[-1] > bt["bh_equity"].iloc[-1]

    # 4) Dự báo "nhìn trộm" sai chiều (dùng return của chính ngày t) KHÔNG được thắng lớn
    #    -> nếu thắng lớn nghĩa là code đang shift sai
    peek_today = (close.pct_change() > 0).astype(float).loc[test_idx]
    bt = run_backtest(close, peek_today)
    assert bt["equity"].iloc[-1] < run_backtest(close, perfect)["equity"].iloc[-1]

    # 5) Phí được tính đúng: đổi vị thế mỗi ngày -> phí mỗi ngày = 0.1%
    alt = pd.Series(np.tile([0.9, 0.1], len(test_idx))[: len(test_idx)], index=test_idx)
    bt = run_backtest(close, alt)
    assert np.allclose(bt["cost"], FEE)

    print("Tất cả test backtest đều PASS")
    print(summarize(run_backtest(close, perfect)).round(3))


ROOT = Path(__file__).resolve().parents[2]                     # thư mục gốc repo
REPORTS = Path(__file__).resolve().parents[1] / "reports"
PRED_DIR = ROOT / "pipeline/results/predictions"


def load_close() -> pd.Series:
    """Giá đóng cửa ngày từ data freeze (data/processed), kiểm sha256 trước khi đọc."""
    from features_ohlcv import MASTER, verify_freeze
    verify_freeze(["master_daily.csv"])
    m = pd.read_csv(MASTER, parse_dates=["date"]).set_index("date")
    return m["ohlcv_close"].dropna()


def demo_real_data():
    """Chạy thử trên giá thật, giai đoạn test 01/2020 – 08/2026 của đề xuất,
       với 2 tín hiệu không cần mô hình: luôn long, và B0 persistence (ŷ_{t+1} = y_t)."""
    close = load_close()
    # p_hat của ngày t dự báo cho t+1; để ngày test đầu tiên là 2020-01-01 thì p_hat bắt đầu 2019-12-31
    t_idx = close.loc["2019-12-31":"2026-08-30"].index
    signals = {
        "always_long": pd.Series(1.0, index=t_idx),
        "B0_persistence": (close.pct_change() > 0).astype(float).loc[t_idx],
    }
    lines = ["# Backtest sanity check trên giá thật (01/2020 – 08/2026)", "",
             "Tạo bởi `src/backtest.py` (hàm `demo_real_data`). Phí 0,1%/lượt, Sharpe năm hoá √365.", ""]
    for name, p in signals.items():
        s = summarize(run_backtest(close, p))
        print(f"\n== {name}\n{s.round(3)}")
        lines += [f"## {name}", "", s.round(3).to_markdown(), ""]
    out = REPORTS / "backtest_sanity.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nĐã lưu {out}")


def backtest_predictions(pred_dir=PRED_DIR, threshold: float = 0.5, fee: float = FEE) -> pd.DataFrame:
    """Backtest mọi file {model}_{config}.csv trong pipeline/results/predictions/.
       Quy ước của Ngọc: dòng date = t có y_true = 1[close_{t+1} > close_t] và p_hat dự báo cho t+1,
       đúng với quy ước của run_backtest. Hàm kiểm lại quy ước này bằng y_true trước khi chạy."""
    close = load_close()
    y_check = (close.shift(-1) > close).astype(int)
    rows = []
    for f in sorted(Path(pred_dir).glob("*_*.csv")):
        model, config = f.stem.split("_", 1)
        p = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        if not (p["y_true"] == y_check.reindex(p.index)).all():
            raise ValueError(f"{f.name}: y_true không khớp 1[close_(t+1) > close_t], sai quy ước ngày")
        bt = run_backtest(close, p["p_hat"], threshold=threshold, fee=fee)
        s = summarize(bt)
        da = ((p["p_hat"] > threshold).astype(int) == p["y_true"]).mean()
        rows.append({"model": model, "config": config, "da": da, **s.loc["strategy"].to_dict(),
                     "first_day": bt.index[0].date(), "last_day": bt.index[-1].date(), "n_days": len(bt)})
    if not rows:
        raise FileNotFoundError(f"Không có file dự báo trong {pred_dir}")
    out = pd.DataFrame(rows)
    bh = summarize(bt).loc["buy_and_hold"].to_dict()                   # cùng khoảng ngày cho mọi file
    out = pd.concat([out, pd.DataFrame([{"model": "buy_and_hold", "config": "-", "da": np.nan, **bh,
                                         "first_day": bt.index[0].date(), "last_day": bt.index[-1].date(),
                                         "n_days": len(bt)}])], ignore_index=True)
    return out


def report_predictions():
    t = backtest_predictions()
    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "backtest_e1_draft.csv", index=False)
    show = t[["model", "config", "da", "cum_return", "sharpe", "mdd", "n_trades", "pct_days_long"]]
    lines = ["# Backtest trên dự báo E1 nháp của Ngọc", "",
             f"Tạo bởi `src/backtest.py` (hàm `report_predictions`) từ `pipeline/results/predictions/`. "
             f"Long nếu p̂ > 0,5, phí 0,1%/lượt, Sharpe năm hoá √365. "
             f"Ngày giữ vị thế: {t.first_day.iloc[0]} → {t.last_day.iloc[0]} ({t.n_days.iloc[0]} ngày).", "",
             "Chưa dùng để kết luận RQ1: dự báo E1 vẫn là bản nháp (OHLCV tạm 7 đặc trưng, chưa kiểm định).", "",
             show.round(3).to_markdown(index=False), ""]
    (REPORTS / "backtest_e1_draft.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + show.round(3).to_string(index=False))
    print(f"\nĐã lưu {REPORTS / 'backtest_e1_draft.md'}")


if __name__ == "__main__":
    # python backtest.py         -> chạy test, chạy thử trên giá thật, rồi backtest dự báo E1 của Ngọc
    # python backtest.py --test  -> chỉ chạy test
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _tests()
    if "--test" not in sys.argv:
        demo_real_data()
        if PRED_DIR.exists():
            report_predictions()
