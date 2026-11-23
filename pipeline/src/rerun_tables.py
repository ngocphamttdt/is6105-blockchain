"""
rerun_tables.py  –  Bảng kết quả và kiểm định cho thư mục chạy lại của rerun_e1.py.

  python pipeline/src/rerun_tables.py --dir pipeline/results/rerun_2026-11-19 [--old pipeline/results]

Đọc {dir}/predictions/{model}_{config}.csv, ghi {dir}/tables/:
  metrics.csv        DA, log-loss, Brier, AUC, % ngày đoán tăng của mỗi mô hình × cấu hình
  delta_onchain.csv  Δ = OHLCV + On-chain − OHLCV: ΔDA, Δlog-loss kèm CI 95% block bootstrap (15 ngày),
                     McNemar, Diebold–Mariano, p hiệu chỉnh Holm trên các mô hình có học
  vs_reversal.csv    mỗi mô hình có học so với b0rev: ΔDA, CI, McNemar, Holm; số năm thắng luật
  by_year.csv        DA, log-loss theo năm test
  by_regime.csv      DA, log-loss và ΔDA On-chain theo regime HMM
  reproduce.csv      (nếu có --old) so p̂ từng ngày với kết quả cũ cùng mô hình
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import stats
from data import DATE, ROOT, read_csv_dated, write_csv
from run_all import RULES

CONFIGS = ("ohlcv", "ohlcv_onchain")
ORDER = ["b0maj", "b0pers", "b0rev", "b1", "b1fs", "b1t", "b2", "b2t", "b3", "b3t", "b4", "b4t", "b5"]   # mô hình thiếu dự báo thì bỏ qua
LEARNED = [m for m in ORDER if not m.startswith("b0")]
OLD_PATH = {"b1t": "b1t/e1/{config}/predictions/b1t_{config}.csv"}


def load(pred_dir):
    out = {}
    for f in sorted(Path(pred_dir).glob("*_*.csv")):
        name, config = f.stem.split("_", 1)
        if config in CONFIGS:
            out[(name, config)] = read_csv_dated(f)
    return out


def metric_row(p, name):
    y, ph = p.y_true.to_numpy(), p.p_hat.to_numpy()
    r = {"da": stats.row_hit(y, ph).mean(), "log_loss": log_loss(y, ph, labels=[0, 1]),
         "brier": brier_score_loss(y, ph), "auc": roc_auc_score(y, ph), "pct_up": (ph > 0.5).mean()}
    if name in RULES:
        r["log_loss"] = r["brier"] = np.nan
    if name == "b0maj":
        r["auc"] = np.nan
    return r


def same_dates(a, b):
    return np.array_equal(a[DATE].to_numpy("datetime64[ns]"), b[DATE].to_numpy("datetime64[ns]"))


def tables(preds, old_dir=None):
    names = [m for m in ORDER if (m, "ohlcv") in preds and (m, "ohlcv_onchain") in preds]
    t = {}
    t["metrics"] = pd.DataFrame([{"model": m, "config": c, "n_days": len(preds[(m, c)]), **metric_row(preds[(m, c)], m)}
                                 for m in names for c in CONFIGS])

    rows = []
    for m in names:
        a, b = preds[(m, "ohlcv")], preds[(m, "ohlcv_onchain")]
        assert same_dates(a, b), m
        rows.append({"model": m, **stats.paired_row(a.y_true.to_numpy(), a.p_hat.to_numpy(), b.p_hat.to_numpy(),
                                                    with_logloss=m not in RULES)})
    d = pd.DataFrame(rows)
    learned = d.model.isin(LEARNED)
    for col in ("p_mcnemar", "p_dm"):
        d.loc[learned, col + "_holm"] = stats.holm(d.loc[learned, col].to_numpy())
    t["delta_onchain"] = d

    rev = preds[("b0rev", "ohlcv")]
    rows = []
    for m in names:
        if m not in LEARNED:
            continue
        for c in CONFIGS:
            p = preds[(m, c)]
            assert same_dates(p, rev), (m, c)
            y = p.y_true.to_numpy()
            r = {"model": m, "config": c, **stats.paired_row(y, rev.p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=False)}
            yrs = p[DATE].dt.year.to_numpy()
            hm, hr = stats.row_hit(y, p.p_hat), stats.row_hit(y, rev.p_hat)
            dy = {int(k): hm[yrs == k].mean() - hr[yrs == k].mean() for k in np.unique(yrs)}
            r |= {"n_years_model_better": sum(v > 0 for v in dy.values()), "n_years": len(dy)}
            rows.append(r)
    v = pd.DataFrame(rows)
    v["p_mcnemar_holm"] = stats.holm(v.p_mcnemar.to_numpy())
    t["vs_reversal"] = v

    rows = []
    for m in names:
        for c in CONFIGS:
            p = preds[(m, c)]
            for yr, q in p.groupby(p[DATE].dt.year):
                rows.append({"model": m, "config": c, "year": int(yr), "n_days": len(q),
                             "up_share": q.y_true.mean(), "da": stats.row_hit(q.y_true, q.p_hat).mean(),
                             "log_loss": np.nan if m in RULES else log_loss(q.y_true, q.p_hat, labels=[0, 1])})
    t["by_year"] = pd.DataFrame(rows)

    rows = []
    for m in names:
        a, b = preds[(m, "ohlcv")], preds[(m, "ohlcv_onchain")]
        for g in ("Bull", "Sideways", "Bear"):
            k = (a.regime == g).to_numpy()
            if not k.any():
                continue
            y = a.y_true.to_numpy()[k]
            pa, pb = a.p_hat.to_numpy()[k], b.p_hat.to_numpy()[k]
            dd, lo, hi = stats.bootstrap_delta(y, pa, pb, "da", stats.block_indices(int(k.sum())))
            rows.append({"model": m, "regime": g, "n_days": int(k.sum()), "up_share": y.mean(),
                         "da_ohlcv": stats.row_hit(y, pa).mean(), "da_ohlcv_onchain": stats.row_hit(y, pb).mean(),
                         "d_da": dd, "d_da_lo": lo, "d_da_hi": hi})
    t["by_regime"] = pd.DataFrame(rows)

    if old_dir is not None:
        rows = []
        for m in names:
            for c in CONFIGS:
                f = Path(old_dir) / OLD_PATH.get(m, "predictions/{model}_{config}.csv").format(model=m, config=c)
                if not f.exists():
                    rows.append({"model": m, "config": c, "old_file": str(f.relative_to(ROOT)), "found": False})
                    continue
                o, n = read_csv_dated(f), preds[(m, c)]
                ok = same_dates(o, n)
                diff = np.abs(o.p_hat.to_numpy() - n.p_hat.to_numpy()) if ok else np.array([np.nan])
                flips = int(((o.p_hat > 0.5) != (n.p_hat > 0.5)).sum()) if ok else -1
                rows.append({"model": m, "config": c, "old_file": str(f.relative_to(ROOT)), "found": True,
                             "same_dates": ok, "max_abs_diff_p_hat": float(np.nanmax(diff)),
                             "n_days_decision_differs": flips,
                             "da_old": stats.row_hit(o.y_true, o.p_hat).mean(),
                             "da_new": stats.row_hit(n.y_true, n.p_hat).mean()})
        t["reproduce"] = pd.DataFrame(rows)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--old", default=None)
    args = ap.parse_args(argv)
    base = ROOT / args.dir
    t = tables(load(base / "predictions"), ROOT / args.old if args.old else None)
    for k, df in t.items():
        print(f"Đã lưu {write_csv(df, base / 'tables' / f'{k}.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
