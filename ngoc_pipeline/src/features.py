"""
features.py  –  Bước 1 (Ngọc, task 1.16): nhãn y và đặc trưng đầu vào cho mọi thí nghiệm.

Nguyên tắc: dòng ngày t chỉ dùng OHLCV đến hết ngày t, và On-chain đến hết ngày t−1 (CoinMetrics
công bố trễ, nên config đặt shift_days: 1). Không có shift(-k) nào ngoài make_target.

On-chain: mỗi chỉ số cho 3 đặc trưng, cùng tham số cho mọi cột (không chọn riêng theo cột).
  Cột dương           {col}_logdiff = log x_t − log x_{t−1}   {col}_ma7_30 = MA7 / MA30   {col}_z90 = z-score 90 ngày của log x
  Cột có dấu (Netflow) {col}_diff    = x_t − x_{t−1}           {col}_ma7m30 = MA7 − MA30   {col}_z90 = z-score 90 ngày của x
  Netflow âm ở khoảng một nửa số ngày nên không lấy log được; arcsinh cũng không dùng vì arcsinh(Netflow)
  gần như chỉ còn là dấu. Ngày nguồn không công bố được điền bằng giá trị trước đó, tối đa ffill_limit ngày.
OHLCV: 10 đặc trưng từ tram_features_backtest/src/features_ohlcv.py (task 1.13 của Trâm): logret_lag1..5,
  hl_range, dlog_volume, rsi_14, rv_7d, rv_30d. RV tính từ data/processed/ohlcv_1h_clean.csv.

python features.py         -> chạy test rồi dựng dữ liệu thật, kiểm không còn NaN từ ngày train đầu tiên
python features.py --test  -> chỉ chạy test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import (CLOSE, COL_GROUP, DATE, NETFLOW, OHLCV_COLS, ONCHAIN_GROUPS, TARGET, _raises,
                  load_config, load_master, make_synthetic_close_1h, make_synthetic_master, onchain_columns)

sys.path.append(str(Path(__file__).resolve().parents[2] / "tram_features_backtest/src"))
from features_ohlcv import FEATURES as OHLCV_FEATURES, build_ohlcv_features  # noqa: E402


def make_target(close: pd.Series) -> pd.Series:
    """y_t = 1 nếu close_{t+1} > close_t; NaN nếu thiếu một trong hai (ngày cuối, trước 17/08/2017)."""
    nxt = close.shift(-1)
    return (nxt > close).astype(float).where(nxt.notna() & close.notna()).rename(TARGET)


# ---- On-chain ----
def _level(s, signed):
    return s if signed else np.log(s.where(s > 0))


def _clean(s):
    return s.replace([np.inf, -np.inf], np.nan)


def log_diff(s, signed=False):
    return _clean(_level(s, signed).diff())


def ma_ratio(s, short=7, long=30, signed=False):
    if signed:
        return s.rolling(short).mean() - s.rolling(long).mean()
    return _clean(s.rolling(short).mean() / s.rolling(long).mean())


def rolling_zscore(s, window=90, signed=False):
    lv = _level(s, signed)
    sd = lv.rolling(window).std()
    return _clean((lv - lv.rolling(window).mean()) / sd.where(sd > 0))


def feature_names(col, ma_short=7, ma_long=30, z_window=90, signed=False) -> list[str]:
    if signed:
        return [f"{col}_diff", f"{col}_ma{ma_short}m{ma_long}", f"{col}_z{z_window}"]
    return [f"{col}_logdiff", f"{col}_ma{ma_short}_{ma_long}", f"{col}_z{z_window}"]


def build_onchain_features(df, cols=None, flows_set="A", signed_cols=(), shift_days=0, ffill_limit=0,
                           ma_short=7, ma_long=30, z_window=90, **_) -> pd.DataFrame:
    """Đặc trưng On-chain kèm cột date. cols=None: lấy theo flows_set. **_ nhận thẳng features_onchain.yaml."""
    if (df[DATE].diff().iloc[1:] != pd.Timedelta(days=1)).any():
        raise ValueError("Dữ liệu thiếu ngày: rolling theo số dòng sẽ sai cửa sổ.")
    cols = onchain_columns(flows_set) if cols is None else list(cols)
    raw = df[cols].ffill(limit=ffill_limit) if ffill_limit else df[cols]
    non_pos = [c for c in cols if c not in signed_cols and (raw[c] <= 0).any()]
    if non_pos:  # kiểm tra dữ liệu, không đổi giá trị: quên khai signed thì log ra NaN âm thầm
        raise ValueError(f"Cột có giá trị <= 0 nhưng không khai trong signed_cols: {non_pos}")
    parts = {}
    for c in cols:
        sg = c in signed_cols
        n_ld, n_ma, n_z = feature_names(c, ma_short, ma_long, z_window, sg)
        parts[n_ld] = log_diff(raw[c], sg)
        parts[n_ma] = ma_ratio(raw[c], ma_short, ma_long, sg)
        parts[n_z] = rolling_zscore(raw[c], z_window, sg)
    feats = pd.DataFrame(parts).shift(shift_days)
    feats.insert(0, DATE, df[DATE].to_numpy())
    return feats.reset_index(drop=True)


def feature_groups(cols=None, flows_set="A", signed_cols=(), ma_short=7, ma_long=30, z_window=90,
                   **_) -> dict[str, list[str]]:
    """{nhóm: [tên đặc trưng]} cho ablation E2. Truyền cùng **cfg với build_onchain_features."""
    cols = onchain_columns(flows_set) if cols is None else list(cols)
    groups = {g: [f for c in cols if COL_GROUP[c] == g
                  for f in feature_names(c, ma_short, ma_long, z_window, c in signed_cols)]
              for g in ONCHAIN_GROUPS}
    return {g: fs for g, fs in groups.items() if fs}


def build_dataset(master: pd.DataFrame, fe_cfg: dict, close_1h: pd.Series | None = None):
    """Ghép nhãn + OHLCV + On-chain theo ngày. Trả (data, {"ohlcv": [cột], "ohlcv_onchain": [cột]}).
    close_1h=None: đọc giá 1h của freeze (dữ liệu thật); test thì truyền make_synthetic_close_1h(master)."""
    df = master.copy()
    df[TARGET] = make_target(df[CLOSE])
    ohlcv = build_ohlcv_features(df, close_1h)
    data = df[[DATE, TARGET]].merge(ohlcv, on=DATE).merge(build_onchain_features(df, **fe_cfg), on=DATE)
    ohlcv_cols = [c for c in ohlcv.columns if c != DATE]
    onchain_cols = [f for fs in feature_groups(**fe_cfg).values() for f in fs]
    return data, {"ohlcv": ohlcv_cols, "ohlcv_onchain": ohlcv_cols + onchain_cols}


# ----------------------------------------------------------------------------
# Test:  python features.py --test
# ----------------------------------------------------------------------------
def _tests():
    cfg = load_config("features_onchain.yaml")
    m = make_synthetic_master(seed=42)
    h1 = make_synthetic_close_1h(m)
    rng = np.random.default_rng(1)

    # 1) Nhãn: y_t = [close_{t+1} > close_t]; không có giá thì NaN, không phải 0
    y = make_target(pd.Series([1.0, 2.0, 1.5, 1.5]))
    assert y.iloc[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(y.iloc[3])
    assert make_target(pd.Series([np.nan, np.nan, 1.0, 2.0])).isna().tolist() == [True, True, False, True]

    # 2) Không nhìn tương lai: sửa mọi giá trị sau mốc cut thì đặc trưng đến cut giống hệt
    for flows_set in ("A", "B"):
        for shift in (0, 1):
            c = {**cfg, "flows_set": flows_set, "shift_days": shift}
            cols = onchain_columns(flows_set)
            base = build_onchain_features(m, **c)
            for cut in (150, 2000):
                m2 = m.copy()
                fut = m2.index > cut
                m2.loc[fut, cols] = m2.loc[fut, cols].to_numpy() * rng.uniform(0.1, 10, (fut.sum(), len(cols)))
                m2.loc[m2.index[fut][3::11], cols] = np.nan
                f2 = build_onchain_features(m2, **c)
                pd.testing.assert_frame_equal(f2.iloc[:cut + 1], base.iloc[:cut + 1])
                assert not f2.iloc[cut + 1:].equals(base.iloc[cut + 1:])   # phép thử có tác dụng
    o = build_ohlcv_features(m, h1)
    m2, h2 = m.copy(), h1.copy()
    m2.loc[m2.index > 1000, OHLCV_COLS] *= 3
    h2[h2.index >= pd.Timestamp(m.loc[1001, DATE], tz="UTC")] *= 3
    o2 = build_ohlcv_features(m2, h2)
    pd.testing.assert_frame_equal(o2.iloc[:1001], o.iloc[:1001])
    assert not o2.iloc[1001:].equals(o.iloc[1001:])

    # 3) shift_days = 1 dịch đúng 1 dòng (On-chain ngày t−1 nằm ở dòng t)
    f0 = build_onchain_features(m, **{**cfg, "shift_days": 0}).drop(columns=DATE)
    f1 = build_onchain_features(m, **{**cfg, "shift_days": 1}).drop(columns=DATE)
    pd.testing.assert_frame_equal(f1.iloc[1:].reset_index(drop=True), f0.iloc[:-1].reset_index(drop=True))

    # 4) Biến đổi: cột dương khớp tính tay; cột có dấu dùng giá trị gốc
    s = pd.Series(np.exp(2.0 + 0.03 * np.arange(50)))
    np.testing.assert_allclose(log_diff(s).iloc[1:], 0.03)
    np.testing.assert_allclose(ma_ratio(pd.Series(np.full(60, 5.0))).iloc[29:], 1.0)
    s = pd.Series(np.random.default_rng(5).normal(0, 1e4, 200))
    pd.testing.assert_series_equal(log_diff(s, signed=True), s.diff())
    np.testing.assert_allclose(ma_ratio(s, signed=True).iloc[100], s.iloc[94:101].mean() - s.iloc[71:101].mean())
    w = s.iloc[11:101]
    np.testing.assert_allclose(rolling_zscore(s, signed=True).iloc[100], (w.iloc[-1] - w.mean()) / w.std())
    assert feature_names("x", signed=True) == ["x_diff", "x_ma7m30", "x_z90"]

    # 5) Kiểm tra đầu vào: cột <= 0 mà quên khai signed thì báo lỗi; thiếu ngày thì báo lỗi
    _raises(lambda: build_onchain_features(m, **{**cfg, "signed_cols": []}), NETFLOW)
    _raises(lambda: build_onchain_features(m.drop(index=100), **cfg), "thiếu ngày")

    # 6) ffill chỉ lấy quá khứ, tối đa 3 ngày
    c = {**cfg, "shift_days": 0, "ffill_limit": 3}
    m3 = m.copy()
    m3.loc[500:502, "cm_HashRate"] = np.nan                    # lỗ 3 ngày: được điền
    m3.loc[1000:1003, "cm_FeeTotNtv"] = np.nan                 # lỗ 4 ngày: ngày thứ 4 vẫn NaN
    f = build_onchain_features(m3, **c)
    assert f.loc[500:502, "cm_HashRate_logdiff"].eq(0).all()
    assert np.isnan(f.loc[1003, "cm_FeeTotNtv_logdiff"])

    # 7) Nhóm cho ablation khớp đúng cột của builder; Flows có 6 (A) hoặc 9 (B) đặc trưng
    for flows_set, n_flows in (("A", 6), ("B", 9)):
        c = {**cfg, "flows_set": flows_set}
        groups = feature_groups(**c)
        assert [x for fs in groups.values() for x in fs] == list(build_onchain_features(m, **c).columns[1:])
        assert len(groups["Flows"]) == n_flows

    # 8) build_dataset: bộ OHLCV-only là đúng 10 đặc trưng của Trâm, nằm trong bộ đầy đủ, không chứa cột On-chain
    data, sets = build_dataset(m, cfg, h1)
    assert sets["ohlcv"] == OHLCV_FEATURES and len(data) == len(m)
    assert set(sets["ohlcv"]) < set(sets["ohlcv_onchain"])
    assert not any(x.startswith(("cm_", "bc_", "puell", "nvt")) for x in sets["ohlcv"])
    print("Tất cả test features đều PASS")


if __name__ == "__main__":
    _tests()
    if "--test" not in sys.argv:
        cfg = load_config("features_onchain.yaml")
        data, sets = build_dataset(load_master(), cfg)
        window = data.loc[data[DATE] >= "2017-12-31", sets["ohlcv_onchain"]]   # ngày train đầu của fold 0
        print(f"Flows set {cfg['flows_set']}: {len(sets['ohlcv'])} đặc trưng OHLCV, "
              f"{len(sets['ohlcv_onchain']) - len(sets['ohlcv'])} On-chain; {len(data)} dòng")
        print("Từ 2017-12-31 còn NaN:", int(window.isna().sum().sum()))
