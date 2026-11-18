"""
gate.py  –  Vi: phần còn lại của góp ý 4 ("thêm On-chain từng bước chỉ sau khi baseline OHLCV ổn định").

Hai bước, đúng thứ tự thầy yêu cầu. Mọi quyết định chỉ dùng inner CV trong cửa sổ train (TimeSeriesSplit 5, gap 1),
không dùng ngày test. Tiêu chí và lưới dưới đây chốt TRƯỚC khi chạy.

  gate      Baseline OHLCV (10 đặc trưng, winsorize 1%/99% như E2) cho B1–B4 với lưới MỞ RỘNG (lưới cũ chạm biên:
            LR chọn C nhỏ nhất 14/14 lần, RF min_samples_leaf lớn nhất 14/14 lần). Ở mỗi lần tune (14 lần) ghi
            log-loss inner CV của tham số chọn và của B0 "đoán theo tỷ lệ ngày tăng trong phần train" trên cùng split.
            Một mô hình QUA CỔNG khi cả hai:
              G1  vượt B0 trên validation: log-loss inner < log-loss B0 ở ≥ 70% số lần tune (≥ 10/14)
              G2  không chạm biên lưới: mỗi tham số nằm ở giá trị đầu/cuối của lưới ở ≤ 50% số lần tune
            Baseline chính thức = mô hình qua cổng có log-loss inner trung bình thấp nhất. Kết quả test (DA, so với
            luật đảo chiều b0rev, DA theo năm) chỉ để báo cáo, không dùng để chọn.

  stepwise  Chọn nhóm On-chain tiến dần (Activity, Flows, Mining, Valuation của features_onchain_v2.yaml) trong mỗi
            cửa sổ tune: bắt đầu từ OHLCV với tham số vừa tune; mỗi bước thử thêm từng nhóm còn lại, lấy nhóm cho
            log-loss inner thấp nhất; chỉ thêm nếu giảm > TOL; dừng khi không nhóm nào đạt. Tập đặc trưng được chọn
            dùng cho 6 fold tiếp theo. So test với baseline cùng mô hình (Δ, CI bootstrap khối, McNemar, DM, Holm).

  python pipeline/src/gate.py --stage gate [--models lr b2]
  python pipeline/src/gate.py --stage stepwise [--models lr b2]
  python pipeline/src/gate.py --stage tables

Output: results/gate/{baseline,stepwise}/{model}.csv (dự báo), results/gate/log/{gate,stepwise}_{model}.csv (mỗi lần
tune), results/gate/tables/{gate_summary,gate_by_year,stepwise_groups,stepwise_delta}.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

import stats
from data import DATE, RESULTS, TARGET, read_csv_dated, write_csv
from experiments import CLIP_Q, E2_DIR, ML, metric_row, setup
from features import feature_groups
from models import RULES, load_model
from run_all import attach_regime, load_regime_map, run_tests
from walkforward import PREDICTION_COLS, fold_indices, inner_cv, make_pipeline, tune

GATE_DIR = RESULTS / "gate"
GRID_WIDE = {
    "lr": {"C": [1e-4, 1e-3, 1e-2, 0.1, 1]},
    "b2": {"n_estimators": [300], "max_depth": [2, 3, 6], "min_samples_leaf": [50, 100, 200]},
    "b3": {"C": [0.01, 0.1, 1, 10], "gamma": [1e-4, 1e-3, 1e-2, 0.1]},
    "b4": {"learning_rate": [0.005, 0.01, 0.05], "max_depth": [1, 2, 3, 4]},
}
G1_SHARE, G2_SHARE = 0.70, 0.50
TOL = 0.001                     # log-loss inner phải giảm hơn 0,001 (≈ 0,15% của ln 2) mới thêm một nhóm
GROUP_ORDER = ["Activity", "Flows", "Mining", "Valuation"]


# ----------------------------------------------------------------------------
# Inner CV
# ----------------------------------------------------------------------------
def cv_logloss(est, params, X, y, wf):
    """Log-loss trung bình trên inner CV, pipeline giống hệt lúc chạy test (winsorize + scale + mô hình)."""
    losses = []
    for tr, va in inner_cv(wf["inner_splits"], wf["gap_days"]).split(X):
        pipe = make_pipeline(clone(est), CLIP_Q).set_params(**{f"model__{k}": v for k, v in params.items()})
        losses.append(log_loss(y[va], pipe.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1], labels=[0, 1]))
    return float(np.mean(losses))


def cv_logloss_prior(y, wf):
    """B0 trên cùng các split: p̂ = tỷ lệ ngày tăng của phần train."""
    return float(np.mean([log_loss(y[va], np.full(len(va), y[tr].mean()), labels=[0, 1])
                          for tr, va in inner_cv(wf["inner_splits"], wf["gap_days"]).split(y)]))


def at_edge(value, grid_values):
    """True nếu giá trị chọn là đầu hoặc cuối lưới (chỉ tính tham số có ≥ 2 giá trị)."""
    return len(grid_values) > 1 and value in (grid_values[0], grid_values[-1])


def forward_groups(est, params, data_tr, base_cols, groups, wf, tol=TOL):
    """Chọn nhóm tiến dần theo log-loss inner. Trả (nhóm đã chọn theo thứ tự, log-loss từng bước)."""
    y = data_tr[TARGET].astype(int).to_numpy()
    cols, chosen = list(base_cols), []
    best = cv_logloss(est, params, data_tr[cols].to_numpy(), y, wf)
    path = [best]
    while True:
        trial = {g: cv_logloss(est, params, data_tr[cols + groups[g]].to_numpy(), y, wf)
                 for g in groups if g not in chosen}
        if not trial:
            break
        g = min(trial, key=trial.get)
        if best - trial[g] <= tol:
            break
        chosen.append(g)
        cols += groups[g]
        best = trial[g]
        path.append(best)
    return chosen, path


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def run(data, folds, name, wf, base_cols, groups=None):
    """Walk-forward như run_walkforward (tune mỗi retune_every fold) và ghi log mỗi lần tune.
    groups=None: chỉ OHLCV (stage gate). Có groups: thêm bước chọn nhóm On-chain sau khi tune (stage stepwise).
    Mọi so sánh trong một lần tune dùng cùng các dòng train (không NaN ở mọi cột ứng viên)."""
    est, _ = load_model(name)
    grid = GRID_WIDE[name]
    all_cols = list(base_cols) + [c for g in (groups or {}).values() for c in g]
    params, cols, out, log = {}, list(base_cols), [], []
    for i, fold in enumerate(folds):
        tr, te = fold_indices(data, fold)
        if i % wf["retune_every"] == 0:
            train_all = data.loc[tr].dropna(subset=all_cols)
            X, y = train_all[list(base_cols)].to_numpy(), train_all[TARGET].astype(int).to_numpy()
            params = tune(make_pipeline(est, CLIP_Q), grid, X, y, inner_cv(wf["inner_splits"], wf["gap_days"]))
            r = {"fold": fold.fold_id, "train_start": fold.train_start.date(), "train_end": fold.train_end.date(),
                 **{f"chosen_{k}": v for k, v in params.items()},
                 "inner_logloss": cv_logloss(est, params, X, y, wf), "inner_logloss_prior": cv_logloss_prior(y, wf)}
            if groups:
                chosen, path = forward_groups(est, params, train_all, base_cols, groups, wf)
                cols = list(base_cols) + [c for g in chosen for c in groups[g]]
                r |= {"groups": "+".join(chosen) or "(không)", "inner_logloss_selected": path[-1],
                      **{f"has_{g.lower()}": g in chosen for g in groups}}
            log.append(r)
        train, test = data.loc[tr].dropna(subset=cols), data.loc[te]
        if test[cols].isna().any().any():
            raise ValueError(f"Fold {fold.fold_id}: có NaN trong đặc trưng của tập test.")
        pipe = make_pipeline(est, CLIP_Q).set_params(**{f"model__{k}": v for k, v in params.items()})
        pipe.fit(train[cols].to_numpy(), train[TARGET].astype(int).to_numpy())
        out.append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": fold.fold_id,
                                 "y_true": test[TARGET].astype(int).to_numpy(),
                                 "p_hat": pipe.predict_proba(test[cols].to_numpy())[:, 1], "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS], pd.DataFrame(log)


def stage(kind, models):
    data, sets, folds, wf, fe = setup("features_onchain_v2.yaml")
    groups = feature_groups(**fe)
    groups = {g: groups[g] for g in GROUP_ORDER}
    rmap = load_regime_map()
    for name in models:
        p, log = run(data, folds, name, wf, sets["ohlcv"], groups if kind == "stepwise" else None)
        sub = "baseline" if kind == "gate" else "stepwise"
        write_csv(attach_regime(p, rmap), GATE_DIR / sub / f"{name}.csv")
        write_csv(log, GATE_DIR / "log" / f"{kind}_{name}.csv")
        extra = f"; nhóm chọn: {log.groups.value_counts().to_dict()}" if kind == "stepwise" else ""
        print(f"{kind} {name}: DA {metric_row(p)['da']:.4f}, inner < B0 ở "
              f"{(log.inner_logloss < log.inner_logloss_prior).sum()}/{len(log)} lần tune{extra}", flush=True)


# ----------------------------------------------------------------------------
# Bảng
# ----------------------------------------------------------------------------
def gate_row(name, log):
    grid = GRID_WIDE[name]
    n = len(log)
    beats = int((log.inner_logloss < log.inner_logloss_prior).sum())
    edge = {k: int(sum(at_edge(v, grid[k]) for v in log[f"chosen_{k}"])) for k in grid if len(grid[k]) > 1}
    g1, g2 = beats >= G1_SHARE * n, all(e <= G2_SHARE * n for e in edge.values())
    return {"model": name, "n_tunes": n, "n_beats_prior": beats, "G1": g1,
            "edge_hits": "; ".join(f"{k} {e}/{n}" for k, e in edge.items()), "G2": g2, "pass": g1 and g2,
            "inner_logloss": log.inner_logloss.mean(), "inner_logloss_prior": log.inner_logloss_prior.mean(),
            "choices": "; ".join(f"{k}: " + ", ".join(f"{v:g}×{c}" for v, c in log[f'chosen_{k}'].value_counts()
                                                      .sort_index().items()) for k in grid if len(grid[k]) > 1)}


def tables():
    rev = read_csv_dated(E2_DIR / "predictions" / "b0rev_rule.csv")
    da_rev = metric_row(rev, True)["da"]
    summ, yr, grp, dl = [], [], [], []
    for name in ML:
        f = GATE_DIR / "baseline" / f"{name}.csv"
        if not f.exists():
            continue
        p, log = read_csv_dated(f), pd.read_csv(GATE_DIR / "log" / f"gate_{name}.csv")
        assert np.array_equal(p[DATE].to_numpy(), rev[DATE].to_numpy())
        m = metric_row(p)
        r = stats.paired_row(p.y_true.to_numpy(), rev.p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=False)
        old = E2_DIR / "predictions" / f"{name}_ohlcv.csv"
        summ.append({**gate_row(name, log), "test_da": m["da"], "test_logloss": m["log_loss"], "test_auc": m["auc"],
                     "test_da_old_grid": metric_row(read_csv_dated(old))["da"] if old.exists() else np.nan,
                     "test_da_b0rev": da_rev, "d_da_vs_b0rev": r["d_da"], "d_da_vs_b0rev_lo": r["d_da_lo"],
                     "d_da_vs_b0rev_hi": r["d_da_hi"], "p_mcnemar_vs_b0rev": r["p_mcnemar"]})
        for y_, q in p.groupby(p[DATE].dt.year):
            yr.append({"model": name, "year": int(y_), "n_days": len(q), "da": stats.row_hit(q.y_true, q.p_hat).mean(),
                       "da_b0rev": stats.row_hit(rev.loc[q.index, "y_true"], rev.loc[q.index, "p_hat"]).mean()})
        g = GATE_DIR / "stepwise" / f"{name}.csv"
        if not g.exists():
            continue
        s, slog = read_csv_dated(g), pd.read_csv(GATE_DIR / "log" / f"stepwise_{name}.csv")
        assert np.array_equal(s[DATE].to_numpy(), p[DATE].to_numpy())
        grp.append({"model": name, "n_tunes": len(slog), "n_no_group": int((slog.groups == "(không)").sum()),
                    **{f"n_{g_.lower()}": int(slog[f"has_{g_.lower()}"].sum()) for g_ in GROUP_ORDER},
                    "selections": "; ".join(f"{k}×{v}" for k, v in slog.groups.value_counts().items()),
                    "inner_gain_mean": (slog.inner_logloss - slog.inner_logloss_selected).mean()})
        dl.append({"model": name, "da_baseline": m["da"], "da_stepwise": metric_row(s)["da"],
                   "logloss_baseline": m["log_loss"], "logloss_stepwise": metric_row(s)["log_loss"],
                   **stats.paired_row(p.y_true.to_numpy(), p.p_hat.to_numpy(), s.p_hat.to_numpy())})
    out = {"gate_summary": pd.DataFrame(summ), "gate_by_year": pd.DataFrame(yr)}
    if grp:
        d = pd.DataFrame(dl)
        d["p_mcnemar_holm"], d["p_dm_holm"] = stats.holm(d.p_mcnemar), stats.holm(d.p_dm)
        out |= {"stepwise_groups": pd.DataFrame(grp), "stepwise_delta": d}
    for k, t in out.items():
        write_csv(t, GATE_DIR / "tables" / f"{k}.csv")
    return out


def figure():
    """DA test theo năm của baseline mới (lưới mở rộng) và của luật đảo chiều; đường 50% làm mốc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    ink, rule_c, muted, grid_c, surf = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1", "#fcfcfb"
    names = {"lr": "B1 Logistic Regression", "b2": "B2 Random Forest", "b3": "B3 SVM", "b4": "B4 XGBoost"}
    yr = pd.read_csv(GATE_DIR / "tables" / "gate_by_year.csv")
    models = [m for m in ML if m in set(yr.model)]
    fig, axs = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.0), sharey=True, facecolor=surf)
    for ax, m in zip(np.atleast_1d(axs), models):
        s = yr[yr.model == m]
        ax.set_facecolor(surf)
        ax.axhline(50, color=muted, ls="--", lw=1)
        ax.plot(s.year, s.da_b0rev * 100, "-o", color=rule_c, lw=2, ms=5, mec=surf, label="Luật đảo chiều b0rev")
        ax.plot(s.year, s.da * 100, "-o", color=ink, lw=2, ms=5, mec=surf, label="Mô hình (chỉ OHLCV)")
        ax.set_title(names[m], fontsize=9)
        ax.grid(axis="y", color=grid_c)
        ax.set_axisbelow(True)
        ax.set_xticks(s.year)
        ax.set_xticklabels([f"'{y % 100:02d}" for y in s.year])
    np.atleast_1d(axs)[0].set_ylabel("Accuracy trên test (%)")
    h, lab = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = RESULTS / "figures" / "fig_gate_by_year.png"
    fig.savefig(out, dpi=200, facecolor=surf)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------
def _tests():
    wf = {"inner_splits": 5, "gap_days": 1}
    # 1) at_edge: chỉ tính tham số có ≥ 2 giá trị
    assert at_edge(1e-4, GRID_WIDE["lr"]["C"]) and at_edge(1, GRID_WIDE["lr"]["C"])
    assert not at_edge(1e-2, GRID_WIDE["lr"]["C"]) and not at_edge(300, [300])

    # 2) B0 inner = log-loss của tỷ lệ train; với y cân bằng ≈ ln 2
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    assert abs(cv_logloss_prior(y, wf) - np.log(2)) < 0.01

    # 3) forward_groups chọn nhóm có tín hiệu, bỏ nhóm nhiễu; nhóm nhiễu không làm giảm log-loss quá TOL
    n = 900
    df = pd.DataFrame({"o1": rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n),
                       "sig": rng.normal(size=n)})
    df[TARGET] = (df.sig + 0.3 * rng.normal(size=n) > 0).astype(int)
    est, _ = load_model("lr")
    chosen, path = forward_groups(est, {"C": 1.0}, df, ["o1"], {"Noise": ["noise1", "noise2"], "Signal": ["sig"]}, wf)
    assert chosen == ["Signal"] and path[0] - path[1] > 0.2, (chosen, path)

    # 4) Không có nhóm nào có tín hiệu → không chọn gì
    df[TARGET] = rng.integers(0, 2, n)
    chosen, path = forward_groups(est, {"C": 1.0}, df, ["o1"], {"Noise": ["noise1", "noise2"], "Signal": ["sig"]}, wf)
    assert chosen == [] and len(path) == 1

    # 5) gate_row: G1/G2 đúng ngưỡng
    log = pd.DataFrame({"chosen_C": [1e-4] * 8 + [1e-2] * 6, "inner_logloss": [0.69] * 10 + [0.70] * 4,
                        "inner_logloss_prior": [0.695] * 14})
    r = gate_row("lr", log)
    assert r["n_beats_prior"] == 10 and r["G1"] and not r["G2"] and not r["pass"]
    print("Tất cả test gate đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gate", "stepwise", "tables", "test"], required=True)
    ap.add_argument("--models", nargs="+", default=ML)
    args = ap.parse_args(argv)
    _tests()
    if args.stage == "test":
        return 0
    run_tests()
    pd.set_option("display.width", 250)
    if args.stage in ("gate", "stepwise"):
        stage(args.stage, args.models)
    else:
        for k, t in tables().items():
            print(k)
            print(t.round(4).to_string(index=False))
        if (GATE_DIR / "tables" / "gate_by_year.csv").exists():
            print(figure())
    return 0


if __name__ == "__main__":
    sys.exit(main())
