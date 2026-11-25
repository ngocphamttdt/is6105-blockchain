"""
ablation_tables.py  –  Gộp bảng ablation nhóm On-chain từ 7 thư mục (full + 3 plus_* + 3 drop_*).

  python pipeline/src/ablation_tables.py [--test]

Chỉ đọc CSV do rerun_tables.py ghi ({dir}/tables/*.csv, {dir}/predictions/*_ohlcv.csv), không gọi mô hình.
Ghi pipeline/results/ablation/:
  ablation_delta.csv      delta_onchain của các mô hình có học, mỗi cấu hình một khối (cột `config`)
  ablation_reproduce.csv  max |Δp̂| của p̂ OHLCV so với thư mục mốc `full` (kiểm tra cùng máy/cùng mô hình)
  ablation_rank.csv       xếp hạng nhóm: rank_plus (Δ log-loss plus_*), rank_drop (log-loss drop_g − full)
  ablation_summary.csv    bảng gọn cho báo cáo: Δ log-loss (×1000), Δ DA (điểm %) theo mô hình × cấu hình, * nếu Holm < 0,05
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data import RESULTS, ROOT, read_csv_dated, write_csv

OUT = RESULTS / "ablation"
CONFIGS = {"full": ROOT / "pipeline/results/no_flows",
           **{n: OUT / n for n in ("plus_activity", "plus_mining", "plus_valuation",
                                   "drop_activity", "drop_mining", "drop_valuation")}}
GROUPS = ["activity", "mining", "valuation"]
TOL = 1e-3


def _learned(m):
    return not str(m).startswith("b0")


def collect(configs):
    """Bảng dài delta_onchain của mô hình có học + danh sách cảnh báo (thư mục/mô hình thiếu)."""
    frames, warns = [], []
    for name, d in configs.items():
        f = Path(d) / "tables" / "delta_onchain.csv"
        if not f.exists():
            warns.append(f"Thiếu {name}: không có {f}")
            continue
        df = pd.read_csv(f, float_precision="round_trip")
        df = df[df.model.map(_learned)].copy()
        df.insert(0, "config", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["config", "model"])
    for name in out.config.unique():
        have = set(out.loc[out.config == name, "model"])
        for m in sorted(set(out.model) - have):
            warns.append(f"Thiếu mô hình {m} trong {name}")
    return out, warns


def reproduce(configs, ref="full"):
    """max |p̂_ohlcv(config) − p̂_ohlcv(ref)| theo mô hình; NaN nếu thiếu file, inf nếu khác tập ngày."""
    rows = []
    rdir = Path(configs[ref]) / "predictions"
    models = sorted(f.stem.rsplit("_", 1)[0] for f in rdir.glob("*_ohlcv.csv") if _learned(f.stem))
    for name, d in configs.items():
        if name == ref:
            continue
        for m in models:
            f = Path(d) / "predictions" / f"{m}_ohlcv.csv"
            if not f.exists():
                rows.append({"config": name, "model": m, "max_abs_dp": np.nan})
                continue
            a, b = read_csv_dated(rdir / f"{m}_ohlcv.csv"), read_csv_dated(f)
            ok = len(a) == len(b) and (a.date.to_numpy() == b.date.to_numpy()).all()
            dp = float(np.abs(a.p_hat.to_numpy() - b.p_hat.to_numpy()).max()) if ok else np.inf
            rows.append({"config": name, "model": m, "max_abs_dp": dp})
    return pd.DataFrame(rows, columns=["config", "model", "max_abs_dp"])


def _order(values):
    """{'Mining': -0.1, ...} -> 'Mining>Activity>...' theo giá trị tăng dần; NaN thì bỏ khỏi chuỗi."""
    s = pd.Series(values).dropna().sort_values(kind="stable")
    return ">".join(s.index) if len(s) else np.nan


def rank_groups(delta, metrics, repro=None):
    """Mỗi mô hình một dòng. rank_plus: d_logloss của plus_g tăng dần (âm nhiều = đóng góp nhiều).
    rank_drop: log-loss(drop_g) − log-loss(full) giảm dần (tăng nhiều khi bỏ = đóng góp nhiều);
    NaN nếu mô hình có max_abs_dp > TOL (hoặc NaN) trong `repro`."""
    bad = set()
    if repro is not None and len(repro):
        bad = set(repro.loc[~(repro.max_abs_dp <= TOL), "model"])

    def ll(cfg, m):
        t = metrics.get(cfg)
        if t is None:
            return np.nan
        r = t[(t.model == m) & (t.config == "ohlcv_onchain")]
        return float(r.log_loss.iloc[0]) if len(r) else np.nan

    rows = []
    for m in sorted(delta.model.unique()):
        plus = {g.capitalize(): delta.loc[(delta.config == f"plus_{g}") & (delta.model == m), "d_logloss"]
                for g in GROUPS}
        plus = {g: (v.iloc[0] if len(v) else np.nan) for g, v in plus.items()}
        full = ll("full", m)
        # đổi dấu để sort tăng dần = giảm dần theo mức tăng log-loss
        drop = {g.capitalize(): -(ll(f"drop_{g}", m) - full) for g in GROUPS}
        rows.append({"model": m, "rank_plus": _order(plus),
                     "rank_drop": np.nan if m in bad else _order(drop)})
    return pd.DataFrame(rows, columns=["model", "rank_plus", "rank_drop"])


def summary(delta, repro=None, alpha=0.05):
    """Bảng gọn cho báo cáo: mỗi dòng một mô hình có học; mỗi cấu hình 2 cột Δ log-loss (×1000) và Δ DA (điểm %).
    `*` = p sau Holm < alpha (DM cho log-loss, McNemar cho DA); `†` = cột full chạy trên máy khác, không tái lập được."""
    bad = set() if repro is None or repro.empty else set(repro.loc[repro.max_abs_dp > TOL, "model"])
    d = delta[delta.model.map(_learned)]
    out = pd.DataFrame(index=pd.Index(d.model.unique(), name="model"))
    for cfg in CONFIGS:
        s = d[d.config == cfg].set_index("model")
        for name, val, p, k in (("dLL_x1000", "d_logloss", "p_dm_holm", 1000), ("dDA_pct", "d_da", "p_mcnemar_holm", 100)):
            out[f"{cfg}_{name}"] = [
                f"{s.at[m, val] * k:.2f}" + ("*" if s.at[m, p] < alpha else "")
                + ("†" if cfg == "full" and m in bad else "") if m in s.index else ""
                for m in out.index]
    return out.reset_index()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--test" in argv:
        _tests()
        return 0
    delta, warns = collect(CONFIGS)
    rep = reproduce(CONFIGS) if (CONFIGS["full"] / "predictions").exists() else pd.DataFrame(
        columns=["config", "model", "max_abs_dp"])
    metrics = {n: pd.read_csv(Path(d) / "tables" / "metrics.csv") for n, d in CONFIGS.items()
               if (Path(d) / "tables" / "metrics.csv").exists()}
    rank = rank_groups(delta, metrics, rep)
    for w in warns:
        print(f"CẢNH BÁO: {w}")
    for r in rep[rep.max_abs_dp > TOL].itertuples():
        print(f"CẢNH BÁO: {r.config}/{r.model} lệch p̂ so với full: max|Δp̂|={r.max_abs_dp:.3g} > {TOL}")
    for k, df in {"ablation_delta": delta, "ablation_reproduce": rep, "ablation_rank": rank,
                  "ablation_summary": summary(delta, rep)}.items():
        print(f"Đã lưu {write_csv(df, OUT / f'{k}.csv')}")
    return 0


def _fake(root, name, ll_full=0.69, models=("b1", "b2"), dll=0.0, drop_ll=None):
    """Thư mục giả cùng cột như rerun_tables.py."""
    d = Path(root) / name
    (d / "tables").mkdir(parents=True)
    (d / "predictions").mkdir()
    dates = pd.date_range("2020-01-01", periods=5)
    pd.DataFrame({"model": ["b0maj", *models], "d_da": 0.0, "d_da_lo": -0.01, "d_da_hi": 0.01,
                  "d_logloss": [0.0, *[dll] * len(models)], "d_logloss_lo": -0.01, "d_logloss_hi": 0.01,
                  "p_mcnemar_holm": 1.0, "p_dm_holm": 1.0}).to_csv(d / "tables/delta_onchain.csv", index=False)
    pd.DataFrame([{"model": m, "config": c, "n_days": 5, "da": 0.5,
                   "log_loss": drop_ll if drop_ll is not None else ll_full, "brier": 0.25, "auc": 0.5, "pct_up": 0.5}
                  for m in models for c in ("ohlcv", "ohlcv_onchain")]).to_csv(d / "tables/metrics.csv", index=False)
    for m in models:
        pd.DataFrame({"date": dates, "fold": 0, "y_true": [0, 1, 0, 1, 1],
                      "p_hat": [0.4, 0.6, 0.5, 0.55, 0.45], "regime": "Bull"}).to_csv(
            d / f"predictions/{m}_ohlcv.csv", index=False)
    return d


def _tests():
    names = list(CONFIGS)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cfg = {n: _fake(t, n) for n in names}
        delta, warns = collect(cfg)
        assert len(delta) == 14 and warns == [], (len(delta), warns)
        assert not delta.model.str.startswith("b0").any()

        # thiếu thư mục drop_mining
        import shutil
        shutil.rmtree(cfg["drop_mining"])
        delta, warns = collect(cfg)
        assert len(delta) == 12 and any("drop_mining" in w for w in warns), (len(delta), warns)
        # thiếu b2 trong plus_activity
        f = cfg["plus_activity"] / "tables/delta_onchain.csv"
        x = pd.read_csv(f)
        x[x.model != "b2"].to_csv(f, index=False)
        delta, warns = collect(cfg)
        assert len(delta) == 11 and any("plus_activity" in w and "b2" in w for w in warns), warns

    # xếp hạng với số giả biết trước: log-loss thấp hơn (âm) = đóng góp nhiều hơn
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        dl = {"plus_activity": -0.002, "plus_mining": 0.001, "plus_valuation": -0.005}
        ll = {"full": 0.690, "drop_activity": 0.692, "drop_mining": 0.690, "drop_valuation": 0.697}
        cfg = {n: _fake(t, n, models=("b1",), dll=dl.get(n, 0.0), drop_ll=ll.get(n, 0.69)) for n in names}
        delta, _ = collect(cfg)
        metrics = {n: pd.read_csv(cfg[n] / "tables/metrics.csv") for n in names}
        rep = reproduce(cfg)
        assert (rep.max_abs_dp == 0).all() and len(rep) == 6, rep
        rk = rank_groups(delta, metrics, rep)
        assert rk.rank_plus.iloc[0] == "Valuation>Activity>Mining", rk
        assert rk.rank_drop.iloc[0] == "Valuation>Activity>Mining", rk

        # lệch 0,01 một ngày ở plus_mining -> max_abs_dp == 0.01 và rank_drop NaN
        p = cfg["plus_mining"] / "predictions/b1_ohlcv.csv"
        x = pd.read_csv(p)
        x.loc[2, "p_hat"] += 0.01
        x.to_csv(p, index=False)
        rep = reproduce(cfg)
        got = rep[(rep.config == "plus_mining") & (rep.model == "b1")].max_abs_dp.iloc[0]
        assert abs(got - 0.01) < 1e-12, got
        rk = rank_groups(delta, metrics, rep)
        assert pd.isna(rk.rank_drop.iloc[0]) and rk.rank_plus.iloc[0] == "Valuation>Activity>Mining", rk

        # summary: không có dòng b0, Δ log-loss ×1000, cột full có † vì b1 lệch ở plus_mining
        sm = summary(delta, rep)
        assert list(sm.model) == ["b1"], sm
        assert sm.at[0, "plus_valuation_dLL_x1000"] == f"{dl['plus_valuation'] * 1000:.2f}", sm
        assert sm.at[0, "full_dLL_x1000"].endswith("†"), sm
    print("Tất cả test ablation_tables đều PASS")


if __name__ == "__main__":
    sys.exit(main())
