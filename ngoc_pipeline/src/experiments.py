"""
experiments.py  –  Vi: góp ý 3–5 của thầy trên cùng pipeline (fold, mô hình, grid, cách tune của run_all).

  tuning  (góp ý 3)  Ghi tham số chọn ở mỗi lần tune (14 lần / 82 fold), điểm inner CV của tham số chọn và của
                     tham số mặc định, và kết quả test trước/sau tuning. Cấu hình E1 (features_onchain.yaml).
  e2      (góp ý 4)  Ablation trên bộ đặc trưng v2 (features_onchain_v2.yaml, sửa theo EDA) + winsorize 1%/99%
                     trong train: ohlcv, onchain (cả 20), plus_{nhóm}, no_flows. So với ohlcv và với luật đảo chiều.
          (góp ý 5)  no_flows = mọi On-chain v2 trừ Flows. Bảng Δ tách thêm hai tập ngày theo flows_vintage.md:
                     ngày ≤ 09/04/2026 (Flows đã được CoinMetrics tính lại hàng loạt trong 01–20/04/2026, không phải
                     giá trị có sẵn lúc dự báo) và ngày ≥ 10/04/2026 (giá trị công bố lần đầu, đúng thời điểm).

  python ngoc_pipeline/src/experiments.py --stage tuning [--models lr b3]
  python ngoc_pipeline/src/experiments.py --stage e2 [--models b2] [--configs plus_flows no_flows]
  python ngoc_pipeline/src/experiments.py --stage e2-tables
  python ngoc_pipeline/src/experiments.py --stage e3          (góp ý 6) Δ theo regime HMM và regime theo xu hướng,
                                                               cho E1 và E2; placebo xáo các đợt regime
Mỗi lệnh chạy test của mọi file trước (run_all.run_tests).

Output (ngoc_pipeline/results/):
  tuning/{tuned,default}/{model}_{config}.csv, tuning/tuning_params.csv, tuning/tuning_before_after.csv
  e2/predictions/{model}_{config}.csv, e2/tables/e2_{metrics,delta,vs_reversal,by_year}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score

import stats
from data import DATE, RESULTS, TARGET, load_config, load_master, read_csv_dated, write_csv
from features import build_dataset, feature_groups
from models import RULES, load_model
from run_all import attach_regime, load_regime_map, run_tests
from walkforward import fold_indices, inner_cv, make_folds, make_pipeline, run_walkforward

ML = ["lr", "b2", "b3", "b4"]
E2_CONFIGS = ["ohlcv", "onchain", "plus_activity", "plus_flows", "plus_mining", "plus_valuation", "no_flows"]
CLIP_Q = 0.01
POINT_IN_TIME_FROM = "2026-04-10"      # crawl_data/vi_ohlcv_flows/reports/flows_vintage.md
TUNE_DIR, E2_DIR = RESULTS / "tuning", RESULTS / "e2"


def setup(fe_name):
    fe, wf = load_config(fe_name), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf)
    return data, sets, folds, wf, fe


def metric_row(p, rule=False):
    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    r = {"n_days": len(p), "da": stats.row_hit(y, ph).mean()}
    if not rule:
        r |= {"log_loss": log_loss(y, ph, labels=[0, 1]), "auc": roc_auc_score(y, ph)}
    return r


# ----------------------------------------------------------------------------
# Góp ý 3: tuning
# ----------------------------------------------------------------------------
def default_estimator(name):
    """Mô hình với tham số cố định của config, còn tham số trong grid để ở mặc định của lớp."""
    est, grid = load_model(name)
    defaults = clone(est).__class__().get_params()
    return est.set_params(**{k: defaults[k] for k in grid}), grid


def inner_score(est, params, X, y, wf):
    pipe = make_pipeline(est).set_params(**{f"model__{k}": v for k, v in params.items()})
    return -cross_val_score(pipe, X, y, scoring="neg_log_loss", cv=inner_cv(wf["inner_splits"], wf["gap_days"])).mean()


def stage_tuning(models, configs=("ohlcv", "ohlcv_onchain")):
    data, sets, folds, wf, _ = setup("features_onchain.yaml")
    rmap = load_regime_map()
    for name in models:
        for config in configs:
            cols = sets[config]
            est, grid = load_model(name)
            log = []
            tuned = run_walkforward(data, cols, est, folds, param_grid=grid, tune_log=log, **wf)
            d_est, _ = default_estimator(name)
            default = run_walkforward(data, cols, d_est, folds, **wf)
            write_csv(attach_regime(tuned, rmap), TUNE_DIR / "tuned" / f"{name}_{config}.csv")
            write_csv(attach_regime(default, rmap), TUNE_DIR / "default" / f"{name}_{config}.csv")
            defaults = {k: d_est.get_params()[k] for k in grid}
            rows = []
            for r in log:
                f = next(x for x in folds if x.fold_id == r["fold"])
                tr, _ = fold_indices(data, f)
                train = data.loc[tr].dropna(subset=cols)
                X, y = train[cols].to_numpy(), train[TARGET].astype(int).to_numpy()
                chosen = {k: r[k] for k in grid}
                rows.append({"model": name, "config": config, "fold": r["fold"], "train_start": r["train_start"],
                             "train_end": r["train_end"], "n_combos": int(np.prod([len(v) for v in grid.values()])),
                             **{f"chosen_{k}": v for k, v in chosen.items()},
                             **{f"default_{k}": v for k, v in defaults.items()},
                             "inner_logloss_chosen": inner_score(est, chosen, X, y, wf),
                             "inner_logloss_default": inner_score(d_est, {}, X, y, wf)})
            out = TUNE_DIR / "params" / f"{name}_{config}.csv"
            write_csv(pd.DataFrame(rows), out)
            print(f"{name}_{config}: {len(log)} lần tune; test DA tuned {metric_row(tuned)['da']:.4f}, "
                  f"default {metric_row(default)['da']:.4f}", flush=True)


def tuning_tables():
    params = pd.concat([pd.read_csv(f) for f in sorted((TUNE_DIR / "params").glob("*.csv"))], ignore_index=True)
    write_csv(params, TUNE_DIR / "tuning_params.csv")
    rows = []
    for f in sorted((TUNE_DIR / "tuned").glob("*.csv")):
        name, config = f.stem.split("_", 1)
        t, d = read_csv_dated(f), read_csv_dated(TUNE_DIR / "default" / f.name)
        mt, md = metric_row(t), metric_row(d)
        pr = params[(params.model == name) & (params.config == config)]
        chosen_cols = [c for c in pr if c.startswith("chosen_")]
        r = stats.paired_row(t["y_true"].to_numpy(), d["p_hat"].to_numpy(), t["p_hat"].to_numpy())
        rows.append({"model": name, "config": config, "n_tunes": len(pr),
                     "n_distinct_choices": len(pr[chosen_cols].astype(str).drop_duplicates()),
                     "inner_logloss_default": pr.inner_logloss_default.mean(),
                     "inner_logloss_tuned": pr.inner_logloss_chosen.mean(),
                     "test_da_default": md["da"], "test_da_tuned": mt["da"],
                     "test_logloss_default": md["log_loss"], "test_logloss_tuned": mt["log_loss"],
                     "test_auc_default": md["auc"], "test_auc_tuned": mt["auc"],
                     **{k: r[k] for k in ("d_da", "d_da_lo", "d_da_hi", "d_logloss", "d_logloss_lo", "d_logloss_hi")}})
    t = pd.DataFrame(rows)
    write_csv(t, TUNE_DIR / "tuning_before_after.csv")
    return params, t


# ----------------------------------------------------------------------------
# Góp ý 4, 5: E2 trên bộ đặc trưng v2
# ----------------------------------------------------------------------------
def e2_feature_sets(sets, groups):
    oh = list(sets["ohlcv"])
    fs = {"ohlcv": oh, "onchain": oh + [f for g in groups.values() for f in g]}
    fs |= {f"plus_{g.lower()}": oh + cols for g, cols in groups.items()}
    fs["no_flows"] = oh + [f for g, cols in groups.items() if g != "Flows" for f in cols]
    return fs


def stage_e2(models, configs):
    data, sets, folds, wf, fe = setup("features_onchain_v2.yaml")
    fs = e2_feature_sets(sets, feature_groups(**fe))
    rmap = load_regime_map()
    pred_dir = E2_DIR / "predictions"
    if not (pred_dir / "b0rev_rule.csv").exists():
        write_csv(attach_regime(RULES["b0rev"](data, folds), rmap), pred_dir / "b0rev_rule.csv")
    for name in models:
        for config in configs:
            est, grid = load_model(name)
            p = run_walkforward(data, fs[config], est, folds, param_grid=grid, clip_q=CLIP_Q, **wf)
            write_csv(attach_regime(p, rmap), pred_dir / f"{name}_{config}.csv")
            print(f"{name}_{config}: {len(fs[config])} đặc trưng, DA {metric_row(p)['da']:.4f}", flush=True)


def e2_tables():
    pred_dir = E2_DIR / "predictions"
    preds = {tuple(f.stem.split("_", 1)): read_csv_dated(f) for f in sorted(pred_dir.glob("*_*.csv"))}
    rev = preds[("b0rev", "rule")]
    met, dl, vs, yr = [], [], [], []
    for (name, config), p in sorted(preds.items()):
        rule = name in RULES
        met.append({"model": name, "config": config, **metric_row(p, rule)})
        years = p[DATE].dt.year
        for y_ in sorted(years.unique()):
            q = p[years == y_]
            yr.append({"model": name, "config": config, "year": int(y_), "n_days": len(q),
                       "da": stats.row_hit(q.y_true, q.p_hat).mean()})
        if rule:
            continue
        r = stats.paired_row(p.y_true.to_numpy(), rev.p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=False)
        vs.append({"model": name, "config": config, "da_model": metric_row(p)["da"], "da_rule": metric_row(rev, True)["da"],
                   **r})
        if config == "ohlcv" or (name, "ohlcv") not in preds:
            continue
        base = preds[(name, "ohlcv")]
        assert np.array_equal(base[DATE].to_numpy(), p[DATE].to_numpy())
        y = p.y_true.to_numpy()
        r = stats.paired_row(y, base.p_hat.to_numpy(), p.p_hat.to_numpy())
        keep = ("d_da", "d_da_lo", "d_da_hi", "d_logloss", "d_logloss_lo", "d_logloss_hi")
        pit = (p[DATE] >= POINT_IN_TIME_FROM).to_numpy()
        for tag, mask in (("restated", ~pit), ("pit", pit)):
            sub = stats.paired_row(y[mask], base.p_hat.to_numpy()[mask], p.p_hat.to_numpy()[mask])
            r |= {f"{k}_{tag}": sub[k] for k in keep}
        r["n_days_pit"] = int(pit.sum())
        dl.append({"model": name, "config": config, **r})
    dl, vs = pd.DataFrame(dl), pd.DataFrame(vs)
    dl["p_mcnemar_holm"], dl["p_dm_holm"] = stats.holm(dl.p_mcnemar), stats.holm(dl.p_dm)
    vs["p_mcnemar_holm"] = stats.holm(vs.p_mcnemar)
    out = {"e2_metrics": pd.DataFrame(met), "e2_delta": dl, "e2_vs_reversal": vs, "e2_by_year": pd.DataFrame(yr)}
    for k, t in out.items():
        write_csv(t, E2_DIR / "tables" / f"{k}.csv")
    return out


# ----------------------------------------------------------------------------
# Góp ý 6: E3 theo regime (HMM của Ngân và regime theo xu hướng, ngan_lstm_hmm_analysis/run_regime.py)
# ----------------------------------------------------------------------------
NGAN = Path(__file__).resolve().parents[2] / "ngan_lstm_hmm_analysis"
REGIME_FILES = {"hmm": NGAN / "regime_map.csv", "trend": NGAN / "regime_trend_map.csv"}
REGIMES = ["Bull", "Sideways", "Bear"]
N_PLACEBO = 1000


def load_regimes():
    out = {}
    for k, f in REGIME_FILES.items():
        r = pd.read_csv(f)
        out[k] = pd.Series(r["regime"].to_numpy(), index=pd.to_datetime(r["date"].astype(str).str[:10]))
    return out


def regime_deltas(y, pa, pb, reg, metric="da"):
    f = stats.ROW_METRIC[metric]
    d = f(y, pb) - f(y, pa)
    return {g: d[reg == g].mean() for g in REGIMES if (reg == g).any()}


def e3_pairs():
    """{(nguồn, mô hình): (dự báo chỉ OHLCV, dự báo có On-chain)} cho E1 (33 đặc trưng v1) và E2 (20 đặc trưng v2)."""
    pairs = {}
    e1 = RESULTS / "predictions"
    for f in sorted(e1.glob("*_ohlcv.csv")):
        name = f.name.removesuffix("_ohlcv.csv")
        g = e1 / f"{name}_ohlcv_onchain.csv"
        if name not in RULES and not name.startswith("b0") and g.exists():
            pairs[("E1 (v1, 33)", name)] = (read_csv_dated(f), read_csv_dated(g))
    e2 = E2_DIR / "predictions"
    for name in ML:
        a, b = e2 / f"{name}_ohlcv.csv", e2 / f"{name}_onchain.csv"
        if a.exists() and b.exists():
            pairs[("E2 (v2, 20)", name)] = (read_csv_dated(a), read_csv_dated(b))
    return pairs


def stage_e3():
    regs = load_regimes()
    rows, plac = [], []
    for (src, name), (a, b) in e3_pairs().items():
        assert np.array_equal(a[DATE].to_numpy(), b[DATE].to_numpy())
        y, pa, pb = a.y_true.to_numpy(), a.p_hat.to_numpy(), b.p_hat.to_numpy()
        for scheme, s in regs.items():
            reg = s.reindex(a[DATE]).to_numpy()
            assert not pd.isna(reg).any(), f"{scheme}: thiếu regime cho ngày test"
            for g in REGIMES:
                k = reg == g
                r = {"source": src, "model": name, "scheme": scheme, "regime": g, "n_days": int(k.sum())}
                for metric in ("da", "log_loss"):
                    d, lo, hi = stats.bootstrap_delta(y[k], pa[k], pb[k], metric, stats.block_indices(int(k.sum())))
                    r |= {f"d_{metric}": d, f"d_{metric}_lo": lo, f"d_{metric}_hi": hi}
                rows.append(r)
            obs = regime_deltas(y, pa, pb, reg)
            rng = np.random.default_rng(stats.SEED)
            null = np.array([(lambda d: d["Bull"] - d["Bear"])(regime_deltas(y, pa, pb, stats.shuffle_runs(reg, rng)))
                             for _ in range(N_PLACEBO)])
            o = obs["Bull"] - obs["Bear"]
            plac.append({"source": src, "model": name, "scheme": scheme, "bull_minus_bear": o,
                         "null_lo": np.quantile(null, 0.025), "null_hi": np.quantile(null, 0.975),
                         "p": (1 + np.sum(np.abs(null) >= abs(o))) / (N_PLACEBO + 1),
                         "n_runs": len(stats.run_lengths(reg)[0])})
    dr, pl = pd.DataFrame(rows), pd.DataFrame(plac)
    for (src, scheme), idx in pl.groupby(["source", "scheme"]).groups.items():
        pl.loc[idx, "p_holm"] = stats.holm(pl.loc[idx, "p"])
    dr["ci_excludes_0"] = (dr.d_da_lo > 0) | (dr.d_da_hi < 0)
    write_csv(dr, RESULTS / "e3" / "e3_delta_regime.csv")
    write_csv(pl, RESULTS / "e3" / "e3_placebo.csv")
    return dr, pl


def figures():
    """Hình cho báo cáo góp ý: Δ của E2 theo mô hình × cấu hình; Δ của E3 theo regime xu hướng."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    ink, muted, grid, surf = "#2a78d6", "#52514e", "#e6e5e1", "#fcfcfb"
    names = {"lr": "B1 Logistic Regression", "b2": "B2 Random Forest", "b3": "B3 SVM", "b4": "B4 XGBoost",
             "b5": "B5 LSTM", "lrfs": "B1 + chọn đặc trưng"}
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def panel(ax, s, k, labels, title, xlabel=None):
        y = np.arange(len(s))
        ax.set_facecolor(surf)
        ax.hlines(y, s[f"{k}_lo"] * 100, s[f"{k}_hi"] * 100, color=ink, lw=2)
        ax.plot(s[k] * 100, y, "o", ms=6, color=ink, mec=surf, mew=1.5)
        ax.axvline(0, color=muted, ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        if xlabel:
            ax.set_xlabel(xlabel)

    dl = pd.read_csv(E2_DIR / "tables" / "e2_delta.csv")
    cfg_lab = {"onchain": "+ cả 20 On-chain", "no_flows": "+ On-chain trừ Flows", "plus_activity": "+ Activity",
               "plus_flows": "+ Flows", "plus_mining": "+ Mining", "plus_valuation": "+ Valuation"}
    cfgs = list(cfg_lab)[::-1]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True, sharex=True, facecolor=surf)
    for ax, m in zip(axs, ML):
        s = dl[dl.model == m].set_index("config").loc[cfgs]
        panel(ax, s, "d_da", [cfg_lab[c] for c in cfgs], names[m])
    fig.supxlabel("Δ accuracy so với chỉ OHLCV (điểm %), CI 95%  ·  bộ đặc trưng v2, winsorize 1%/99%", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_e2_delta_v2.png", dpi=200, facecolor=surf)

    dr = pd.read_csv(RESULTS / "e3" / "e3_delta_regime.csv")
    dr = dr[dr.scheme == "trend"]
    srcs = list(dict.fromkeys(dr.source))
    fig, axs = plt.subplots(len(srcs), 1, figsize=(8, 2.0 + 2.2 * len(srcs)), facecolor=surf, sharex=True)
    for ax, src in zip(np.atleast_1d(axs), srcs):
        s = dr[dr.source == src]
        s = s.assign(lab=s.model.map(names) + " · " + s.regime).iloc[::-1]
        panel(ax, s, "d_da", s.lab, src)
    fig.supxlabel("Δ accuracy (điểm %) trong từng regime xu hướng, CI 95%", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_e3_trend_regime.png", dpi=200, facecolor=surf)
    plt.close("all")


# ----------------------------------------------------------------------------
# Test:  chạy cùng run_tests() ở đầu mỗi stage
# ----------------------------------------------------------------------------
def _tests():
    from data import make_synthetic_close_1h, make_synthetic_master
    fe = load_config("features_onchain_v2.yaml")
    m = make_synthetic_master(seed=42)
    _, sets = build_dataset(m, fe, make_synthetic_close_1h(m))
    groups = feature_groups(**fe)
    fs = e2_feature_sets(sets, groups)

    # 1) Mọi cấu hình bắt đầu bằng đúng 10 đặc trưng OHLCV; onchain = đủ 20; no_flows = onchain trừ Flows
    oh = sets["ohlcv"]
    assert set(fs) == set(E2_CONFIGS) and all(fs[c][:len(oh)] == oh for c in fs)
    assert len(fs["onchain"]) - len(oh) == sum(map(len, groups.values()))
    assert set(fs["no_flows"]) == set(fs["onchain"]) - set(groups["Flows"])
    assert fs["plus_flows"] == oh + groups["Flows"]

    # 2) Mô hình mặc định: tham số trong grid về mặc định của lớp, tham số cố định của config giữ nguyên
    est, grid = default_estimator("b2")
    assert est.get_params()["max_depth"] is None and est.get_params()["n_estimators"] == 100
    assert est.get_params()["random_state"] == 0 and set(grid) == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert default_estimator("lr")[0].get_params()["C"] == 1.0

    # 3) Δ theo regime: trung bình có trọng số theo số ngày bằng Δ gộp
    rng = np.random.default_rng(1)
    y, pa, pb = rng.integers(0, 2, 300), rng.uniform(0, 1, 300), rng.uniform(0, 1, 300)
    reg = np.repeat(np.array(REGIMES), 100)
    d = regime_deltas(y, pa, pb, reg)
    np.testing.assert_allclose(np.mean(list(d.values())), (stats.row_hit(y, pb) - stats.row_hit(y, pa)).mean())
    print("Tất cả test experiments đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["tuning", "tuning-tables", "e2", "e2-tables", "e3"], required=True)
    ap.add_argument("--models", nargs="+", default=ML)
    ap.add_argument("--configs", nargs="+", default=None)
    args = ap.parse_args(argv)
    run_tests()
    stats._tests()
    _tests()
    pd.set_option("display.width", 250)
    if args.stage == "tuning":
        stage_tuning(args.models, args.configs or ("ohlcv", "ohlcv_onchain"))
    elif args.stage == "tuning-tables":
        print(tuning_tables()[1].round(4).to_string(index=False))
    elif args.stage == "e2":
        stage_e2(args.models, args.configs or E2_CONFIGS)
    elif args.stage == "e3":
        dr, pl = stage_e3()
        print(dr[dr.regime.isin(["Bull", "Bear"])].round(4).to_string(index=False))
        print(pl.round(4).to_string(index=False))
    else:
        for k, t in e2_tables().items():
            if k != "e2_by_year":
                print(k)
                print(t.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
