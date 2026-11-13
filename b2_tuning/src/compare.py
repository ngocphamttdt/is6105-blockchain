"""
compare.py – so B2 grid gốc với B2 đã tuning trên cùng 82 fold / 2.434 ngày test.
Ghi bảng metric, Δ On-chain, chênh lệch tuned − gốc (CI block bootstrap 15 ngày, 2.000 mẫu; McNemar cho DA)
và backtest long/flat (code backtest.py của repo).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO, RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(REPO / "tram_features_backtest/src"))
from backtest import load_close, run_backtest, summarize  # noqa: E402

B0MAJ_LL = 0.6938455526478804   # e1_delta_draft.csv
FILES = {("orig", "ohlcv"): RUNS / "orig/b2_ohlcv.csv", ("orig", "ohlcv_onchain"): RUNS / "orig/b2_ohlcv_onchain.csv",
         ("tuned", "ohlcv"): RUNS / "tuned_ohlcv/b2_ohlcv.csv",
         ("tuned", "ohlcv_onchain"): RUNS / "tuned_onchain/b2_ohlcv_onchain.csv"}
P = {k: pd.read_csv(f, parse_dates=["date"]) for k, f in FILES.items()}
dates = P[("orig", "ohlcv")]["date"]
assert all(p["date"].equals(dates) for p in P.values())
y = P[("orig", "ohlcv")]["y_true"].to_numpy()


def ll_vec(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def hits(p):
    return ((p > 0.5).astype(int) == y).astype(float)


def block_boot(d, block=15, n=2000, seed=0):
    rng, T = np.random.default_rng(seed), len(d)
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    m = d[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


close = load_close()
rows, bt_rows = [], []
for (grid, config), p in P.items():
    ph = p["p_hat"].to_numpy()
    s = summarize(run_backtest(close, p.set_index("date")["p_hat"])).loc["strategy"]
    rows.append({"grid": grid, "config": config, "da": hits(ph).mean(), "log_loss": log_loss(y, ph),
                 "brier": brier_score_loss(y, ph), "auc": roc_auc_score(y, ph),
                 "beats_b0maj_logloss": log_loss(y, ph) < B0MAJ_LL, "pct_pred_up": (ph > 0.5).mean(),
                 "p_hat_std": ph.std(), "cum_return": s["cum_return"], "sharpe": s["sharpe"], "mdd": s["mdd"],
                 "n_trades": int(s["n_trades"])})
metrics = pd.DataFrame(rows)

# Tuned − gốc, từng cấu hình
diff_rows = []
for config in ("ohlcv", "ohlcv_onchain"):
    a, b = P[("orig", config)]["p_hat"].to_numpy(), P[("tuned", config)]["p_hat"].to_numpy()
    dll = ll_vec(b) - ll_vec(a)
    ha, hb = hits(a), hits(b)
    n01, n10 = int(((hb == 1) & (ha == 0)).sum()), int(((hb == 0) & (ha == 1)).sum())
    lo, hi = block_boot(dll)
    dlo, dhi = block_boot(hb - ha)
    diff_rows.append({"config": config, "d_logloss": dll.mean(), "d_logloss_ci95": f"[{lo:+.4f}; {hi:+.4f}]",
                      "d_da": (hb - ha).mean(), "d_da_ci95": f"[{dlo:+.4f}; {dhi:+.4f}]",
                      "only_tuned_right": n01, "only_orig_right": n10,
                      "p_mcnemar": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0})
diff = pd.DataFrame(diff_rows)

# Δ On-chain trong từng grid
delta_rows = []
for grid in ("orig", "tuned"):
    a, b = P[(grid, "ohlcv")]["p_hat"].to_numpy(), P[(grid, "ohlcv_onchain")]["p_hat"].to_numpy()
    dll, dda = ll_vec(b) - ll_vec(a), hits(b) - hits(a)
    (llo, lhi), (dlo, dhi) = block_boot(dll), block_boot(dda)
    delta_rows.append({"grid": grid, "delta_da": dda.mean(), "delta_da_ci95": f"[{dlo:+.4f}; {dhi:+.4f}]",
                       "delta_logloss": dll.mean(), "delta_logloss_ci95": f"[{llo:+.4f}; {lhi:+.4f}]"})
delta = pd.DataFrame(delta_rows)

# Tham số được chọn
chosen = {}
for name, f in {"orig": RUNS / "orig/summary.json", "tuned_ohlcv": RUNS / "tuned_ohlcv/summary.json",
                "tuned_onchain": RUNS / "tuned_onchain/summary.json"}.items():
    for config, v in json.loads(f.read_text()).items():
        chosen[(name.split("_")[0], config)] = pd.DataFrame(v["chosen"]).astype(str).value_counts().to_dict()

OUT.mkdir(parents=True, exist_ok=True)
metrics.to_csv(OUT / "b2_metrics_orig_vs_tuned.csv", index=False)
diff.to_csv(OUT / "b2_tuned_minus_orig.csv", index=False)
delta.to_csv(OUT / "b2_delta_onchain.csv", index=False)
pd.set_option("display.width", 220)
print(metrics.round(4).to_string(index=False))
print(diff.round(4).to_string(index=False))
print(delta.round(4).to_string(index=False))
for k, v in chosen.items():
    print(k, v)
