"""
b1t_tuning.py  –  Vi: mô hình b1t (tên cũ: lrt / lr_tuned) = tuning đầy đủ cho baseline B1 Logistic Regression (góp ý 3: tune bằng train/validation,
không chọn theo test). Không gian tìm và tiêu chí chốt TRƯỚC khi chạy.

Ở mỗi lần tune (14 lần, mỗi 6 fold; cùng 82 fold walk-forward), trong cửa sổ train 730 ngày, TimeSeriesSplit(5, gap 1):
  tập đặc trưng  ohlcv10 | ohlcv10+sign | lag1 | sign | lag1+sign
                 (ohlcv10 = 10 đặc trưng OHLCV; lag1 = logret_lag1; sign = up_lag1 = 1[logret_lag1 > 0])
  C              1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0,1, 0,3, 1, 10
  l1_ratio       0 (L2), 0,5 (elastic-net), 1 (L1)              solver L-BFGS-B tất định (ElasticNetLR)
  class_weight   None, balanced
  = 300 tổ hợp. Tiền xử lý như E2: winsorize 1%/99% + StandardScaler, fit trong từng split.
Hai tiêu chí chọn (đều chỉ trên validation):
  logloss   log-loss inner thấp nhất (tiêu chí chính, như mọi mô hình khác của pipeline)
  acc       accuracy inner cao nhất (độ nhạy: B1 được chọn theo đúng metric báo cáo)
Hoà điểm (làm tròn 10 chữ số) thì lấy tổ hợp đơn giản hơn: ít đặc trưng hơn, C nhỏ hơn.

So sánh trên test: LR E1 (lưới C cũ, 10 OHLCV), LR gate (lưới C mở rộng), hai bản tuning mới, luật đảo chiều.

  python pipeline/src/b1t_tuning.py --stage run       (~1 phút)
  python pipeline/src/b1t_tuning.py --stage tables
  python pipeline/src/b1t_tuning.py --stage e1        E1 với B1 tune: cấu hình E1 (v1, 33 On-chain), tiêu chí log-loss;
                                                          +On-chain = mỗi tập ứng viên + 33 đặc trưng On-chain
Output: results/b1t/{predictions/b1t_{logloss,acc}.csv, log, tables/b1t_*.csv}, figures/fig_b1t_tuning.png;
        results/b1t/e1/{ohlcv,ohlcv_onchain}/{predictions/b1t_{config}.csv, log/}
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

import stats
from data import DATE, PREDICTION_COLS, RESULTS, TARGET, load_config, read_csv_dated, write_csv
from experiments import E2_DIR, setup
from hybrid import SIGN, with_sign
from rules30 import OUT as R30, score
from run_all import attach_regime, load_regime_map
from walkforward import fold_indices, inner_cv, make_pipeline

OUT = RESULTS / "b1t"
CFG = load_config("models/b1t.yaml")          # tập đặc trưng, grid, tiêu chí: chốt trong file config
GRID = CFG["param_grid"]
CRITERIA = CFG["criteria"]
CLIP_Q = CFG["params"]["clip_q"]


def feature_sets(ohlcv):
    """Tập ứng viên theo thứ tự trong config, từ đơn giản đến phức tạp (dùng để phá hoà).
    "ohlcv" trong config được thay bằng danh sách đặc trưng OHLCV."""
    return {name: [c for col in cols for c in (list(ohlcv) if col == "ohlcv" else [col])]
            for name, cols in CFG["feature_sets"].items()}


class ElasticNetLR(ClassifierMixin, BaseEstimator):
    """Logistic regression elastic-net, cùng hàm mục tiêu với sklearn LogisticRegression:
        min  Σ w_i·logloss_i + (1/C)·[(1 − l1_ratio)/2·‖β‖² + l1_ratio·‖β‖₁],  intercept không bị phạt.
    Không dùng solver saga vì điều kiện dừng của nó (sklearn 1.9) chỉ xét hệ số, bỏ qua intercept: khi phạt mạnh
    làm mọi hệ số = 0, saga dừng sau 1 vòng với intercept còn sai và phụ thuộc random_state (fold 0, tập sign,
    C=1e-4, L1: intercept đúng 0,097 nhưng saga cho 0,29 / 1,48 / 0,40 với seed 0 / 42 / 7, kể cả khi tol=0).
    Ở đây giải tất định bằng L-BFGS-B với β = β⁺ − β⁻, β± ≥ 0 (L1 thành trơn; hệ số về 0 đúng nghĩa ở biên)."""

    def __init__(self, C=1.0, l1_ratio=0.0, class_weight=None, tol=1e-10, max_iter=20000):
        self.C, self.l1_ratio, self.class_weight, self.tol, self.max_iter = C, l1_ratio, class_weight, tol, max_iter

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        y = (y == self.classes_[1]).astype(float)
        n, p = X.shape
        w = compute_sample_weight(self.class_weight, y) / n                  # chia n: cùng nghiệm, số học ổn hơn
        lam, a = 1.0 / (self.C * n), self.l1_ratio

        def f(z):
            bp, bm, b0 = z[:p], z[p:2 * p], z[-1]
            beta = bp - bm
            m = X @ beta + b0
            loss = w @ (np.logaddexp(0, m) - y * m)
            r = w * (expit(m) - y)
            g = X.T @ r + lam * (1 - a) * beta
            return (loss + lam * ((1 - a) / 2 * beta @ beta + a * (bp.sum() + bm.sum())),
                    np.concatenate([g + lam * a, -g + lam * a, [r.sum()]]))

        pr = np.clip(w @ y / w.sum(), 1e-12, 1 - 1e-12)
        z0 = np.zeros(2 * p + 1)
        z0[-1] = np.log(pr / (1 - pr))
        res = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * (2 * p) + [(None, None)],
                       options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": self.tol})
        self.coef_ = (res.x[:p] - res.x[p:2 * p])[None, :]
        self.intercept_ = res.x[-1:]
        self.n_iter_ = np.array([res.nit])
        self.converged_ = bool(res.success)
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


def estimator():
    return ElasticNetLR()


def search(train, sets, wf):
    """Mọi tổ hợp (tập đặc trưng × GRID) trên inner CV. Trả DataFrame một dòng / tổ hợp, đủ điểm của hai tiêu chí."""
    y = train[TARGET].astype(int).to_numpy()
    rows = []
    for k, (name, cols) in enumerate(sets.items()):
        gs = GridSearchCV(make_pipeline(estimator(), CLIP_Q), {f"model__{p}": v for p, v in GRID.items()},
                          scoring=CRITERIA, refit=False, cv=inner_cv(wf["inner_splits"], wf["gap_days"]), n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = gs.fit(train[cols].to_numpy(), y).cv_results_
        for i, prm in enumerate(res["params"]):
            prm = {p.removeprefix("model__"): v for p, v in prm.items()}
            prm["class_weight"] = prm["class_weight"] or "none"       # None trong DataFrame thành NaN
            rows.append({"set": name, "set_rank": k, **prm,
                         "inner_logloss": -res["mean_test_logloss"][i], "inner_acc": res["mean_test_acc"][i]})
    return pd.DataFrame(rows)


def pick(res, criterion):
    """Tổ hợp tốt nhất theo tiêu chí; hoà (10 chữ số) thì lấy tập đơn giản hơn rồi C nhỏ hơn."""
    key = res.inner_logloss.round(10) if criterion == "logloss" else -res.inner_acc.round(10)
    return res.assign(_k=key).sort_values(["_k", "set_rank", "C"], kind="stable").iloc[0]


def stage_run(fe="features_onchain_v2.yaml", onchain=False, out_dir=OUT, criteria=tuple(CRITERIA),
              names=None):
    """Tune B1 trên mọi fold. onchain=True: mỗi tập ứng viên được thêm toàn bộ đặc trưng On-chain của `fe`
    (cùng lưới, cùng tiêu chí), để Δ On-chain so hai bản B1 tune cùng quy trình (góp ý 4)."""
    data, sets_, folds, wf, _ = setup(fe)
    data = with_sign(data)
    sets = feature_sets(sets_["ohlcv"])
    if onchain:
        oc = [c for c in sets_["ohlcv_onchain"] if c not in sets_["ohlcv"]]
        sets = {f"{k}+oc": v + oc for k, v in sets.items()}
    names = names or {c: f"b1t_{c}" for c in criteria}
    all_cols = sorted({c for v in sets.values() for c in v})
    rmap = load_regime_map()
    out = {c: [] for c in criteria}
    logs, chosen = [], {}
    for i, f in enumerate(folds):
        tr, te = fold_indices(data, f)
        train, test = data.loc[tr].dropna(subset=all_cols), data.loc[te]
        if i % wf["retune_every"] == 0:
            res = search(train, sets, wf)
            prior = np.mean([log_loss(train[TARGET].to_numpy()[va], np.full(len(va), train[TARGET].to_numpy()[t].mean()),
                                      labels=[0, 1]) for t, va in inner_cv(wf["inner_splits"], wf["gap_days"])
                             .split(train)])
            write_csv(res, out_dir / "log" / "grid" / f"fold{f.fold_id:02d}.csv")
            for c in criteria:
                b = pick(res, c)
                chosen[c] = b
                logs.append({"criterion": c, "fold": f.fold_id, "train_start": f.train_start.date(),
                             "train_end": f.train_end.date(), "set": b["set"], "C": b["C"], "l1_ratio": b["l1_ratio"],
                             "class_weight": b["class_weight"], "inner_logloss": b["inner_logloss"],
                             "inner_acc": b["inner_acc"], "inner_logloss_prior": prior})
            print(f"fold {f.fold_id}: " + "; ".join(f"{c}: {chosen[c]['set']}, C={chosen[c]['C']:g}, "
                                                   f"l1={chosen[c]['l1_ratio']:g}, cw={chosen[c]['class_weight']}"
                                                   for c in criteria), flush=True)
        for c in criteria:
            b = chosen[c]
            cols = sets[b["set"]]
            assert not test[cols].isna().any().any(), f"fold {f.fold_id}: thiếu đặc trưng ở ngày test"
            pipe = make_pipeline(estimator(), CLIP_Q).set_params(
                model__C=b["C"], model__l1_ratio=b["l1_ratio"],
                model__class_weight=None if b["class_weight"] == "none" else b["class_weight"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pipe.fit(train[cols].to_numpy(), train[TARGET].astype(int).to_numpy())
            out[c].append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": f.fold_id,
                                        "y_true": test[TARGET].astype(int).to_numpy(),
                                        "p_hat": pipe.predict_proba(test[cols].to_numpy())[:, 1], "regime": pd.NA}))
    for c in criteria:
        p = pd.concat(out[c], ignore_index=True)[PREDICTION_COLS]
        write_csv(attach_regime(p, rmap), out_dir / "predictions" / f"{names[c]}.csv")
        print(f"{names[c]}: accuracy {score(p)['accuracy']:.4f}", flush=True)
    write_csv(pd.DataFrame(logs), out_dir / "log" / "chosen.csv")


# ----------------------------------------------------------------------------
# Bảng, hình
# ----------------------------------------------------------------------------
def full_metrics(p, rule=False):
    y, ph = p.y_true.to_numpy(), p.p_hat.to_numpy()
    r = score(p) | {"mcc": matthews_corrcoef(y, (ph > 0.5).astype(int))}
    if not rule:
        r |= {"log_loss": log_loss(y, ph, labels=[0, 1]), "auc": roc_auc_score(y, ph)}
    return r


def tables():
    rev = read_csv_dated(R30 / "predictions" / "rev1.csv")
    old = read_csv_dated(E2_DIR / "predictions" / "b1_ohlcv.csv")
    cands = {"B1 cũ: 10 OHLCV, C ∈ {0,01..10}, L2": (old, False),
             "B1 lưới C mở rộng (gate): 10 OHLCV, C ∈ {1e-4..1}, L2": (read_csv_dated(RESULTS / "gate" / "baseline" / "b1.csv"), False),
             "B1 tune đầy đủ, chọn theo log-loss": (read_csv_dated(OUT / "predictions" / "b1t_logloss.csv"), False),
             "B1 tune đầy đủ, chọn theo accuracy": (read_csv_dated(OUT / "predictions" / "b1t_acc.csv"), False),
             "Luật đảo chiều (b0rev)": (rev, True)}
    rows = []
    for lab, (p, rule) in cands.items():
        assert np.array_equal(p[DATE].to_numpy("datetime64[ns]"), rev[DATE].to_numpy("datetime64[ns]")), lab
        r = {"method": lab, **full_metrics(p, rule)}
        if not rule:
            for tag, base in (("old", old), ("rev1", rev)):
                if base is p:
                    continue
                d = stats.paired_row(p.y_true.to_numpy(), base.p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=False)
                r |= {f"d_acc_vs_{tag}": d["d_da"], f"d_acc_vs_{tag}_lo": d["d_da_lo"],
                      f"d_acc_vs_{tag}_hi": d["d_da_hi"], f"p_mcnemar_vs_{tag}": d["p_mcnemar"]}
        rows.append(r)
    t = pd.DataFrame(rows)
    log = pd.read_csv(OUT / "log" / "chosen.csv")
    ch = []
    for c, g in log.groupby("criterion"):
        n = len(g)
        ch.append({"criterion": c, "n_tunes": n,
                   "sets": "; ".join(f"{k}×{v}" for k, v in g.set.value_counts().items()),
                   "C": "; ".join(f"{k:g}×{v}" for k, v in g.C.value_counts().sort_index().items()),
                   "C_at_edge": int(g.C.isin([GRID['C'][0], GRID['C'][-1]]).sum()),
                   "l1_ratio": "; ".join(f"{k:g}×{v}" for k, v in g.l1_ratio.value_counts().sort_index().items()),
                   "class_weight": "; ".join(f"{k}×{v}" for k, v in g.class_weight.value_counts().items()),
                   "inner_logloss": g.inner_logloss.mean(), "inner_logloss_prior": g.inner_logloss_prior.mean(),
                   "n_beats_prior": int((g.inner_logloss < g.inner_logloss_prior).sum()), "inner_acc": g.inner_acc.mean()})
    ch = pd.DataFrame(ch)
    yr = []
    for lab, (p, _) in cands.items():
        for y_, q in p.groupby(p[DATE].dt.year):
            yr.append({"method": lab, "year": int(y_), "accuracy": stats.row_hit(q.y_true, q.p_hat).mean()})
    yr = pd.DataFrame(yr)
    for k, v in {"b1t_summary": t, "b1t_choices": ch, "b1t_by_year": yr}.items():
        write_csv(v, OUT / "tables" / f"{k}.csv")
    return t, ch, yr


def figure(t, yr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    ink, rule_c, muted, grid_c, surf = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1", "#fcfcfb"
    short = {"B1 cũ: 10 OHLCV, C ∈ {0,01..10}, L2": "B1 cũ",
             "B1 lưới C mở rộng (gate): 10 OHLCV, C ∈ {1e-4..1}, L2": "B1 lưới C mở rộng",
             "B1 tune đầy đủ, chọn theo log-loss": "B1 tune đầy đủ (log-loss)",
             "B1 tune đầy đủ, chọn theo accuracy": "B1 tune đầy đủ (accuracy)",
             "Luật đảo chiều (b0rev)": "Luật đảo chiều"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=surf, gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]
    ax.set_facecolor(surf)
    s = t.iloc[::-1]
    y = np.arange(len(s))
    ax.axvline(50, color=muted, ls="--", lw=1)
    ax.plot(s.accuracy * 100, y + 0.15, "o", color=ink, ms=7, mec=surf, mew=1.5, label="Accuracy")
    ax.plot(s.f1_macro * 100, y - 0.15, "D", color=rule_c, ms=6, mec=surf, mew=1.5, label="F1 macro")
    ax.set_yticks(y)
    ax.set_yticklabels(s.method.map(short))
    ax.set_xlabel("% trên 2.434 ngày test")
    ax.grid(axis="x", color=grid_c)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Trước và sau tuning", fontsize=9)
    ax = axs[1]
    ax.set_facecolor(surf)
    ax.axhline(50, color=muted, ls="--", lw=1)
    show = {"B1 cũ": "#9a9893", "B1 tune đầy đủ (log-loss)": ink, "Luật đảo chiều": rule_c}
    for lab, c in show.items():
        full = next(k for k, v in short.items() if v == lab)
        q = yr[yr.method == full]
        ax.plot(q.year, q.accuracy * 100, "-o", color=c, lw=2, ms=5, mec=surf, label=lab)
    ax.set_xticks(sorted(yr.year.unique()))
    ax.set_xticklabels([f"'{v % 100:02d}" for v in sorted(yr.year.unique())])
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color=grid_c)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Theo năm", fontsize=9)
    fig.tight_layout()
    out = RESULTS / "figures" / "fig_b1t_tuning.png"
    fig.savefig(out, dpi=200, facecolor=surf)
    plt.close(fig)
    return out


def _tests():
    res = pd.DataFrame({"set": ["a", "b", "a"], "set_rank": [0, 1, 0], "C": [1.0, 0.1, 0.01],
                        "inner_logloss": [0.69, 0.69, 0.70], "inner_acc": [0.52, 0.55, 0.55]})
    assert pick(res, "logloss")["C"] == 1.0                      # hoà log-loss → tập đơn giản hơn (rank 0)
    assert pick(res, "acc")["C"] == 0.01                          # hoà accuracy → tập rank 0, rồi C nhỏ hơn
    assert len(GRID["C"]) * len(GRID["l1_ratio"]) * len(GRID["class_weight"]) * 5 == 300
    # Config b1t.yaml: tập đặc trưng đúng thứ tự, "ohlcv" được thay bằng cột OHLCV, cột dấu là up_lag1 của hybrid
    o = ["a", "b"]
    assert feature_sets(o) == {"sign": [SIGN], "lag1": ["logret_lag1"], "lag1+sign": ["logret_lag1", SIGN],
                               "ohlcv10": o, "ohlcv10+sign": o + [SIGN]}
    assert all(isinstance(c, (int, float)) for c in GRID["C"]) and GRID["class_weight"] == [None, "balanced"]
    print("Tất cả test b1t_tuning đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run", "e1", "tables", "test"], required=True)
    args = ap.parse_args(argv)
    _tests()
    if args.stage == "run":
        stage_run()
    elif args.stage == "e1":
        for cfg in ("ohlcv", "ohlcv_onchain"):
            stage_run("features_onchain.yaml", onchain=cfg == "ohlcv_onchain", out_dir=OUT / "e1" / cfg,
                      criteria=("logloss",), names={"logloss": f"b1t_{cfg}"})
    elif args.stage == "tables":
        pd.set_option("display.width", 300)
        t, ch, yr = tables()
        print(t.round(4).to_string(index=False))
        print(ch.round(4).to_string(index=False))
        print(figure(t, yr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
