"""
horizon7.py  –  Task 6 (Vi): dự báo hướng giá 7 ngày tới, On-chain có giúp ích không?

Câu hỏi: ở tầm 1 ngày (E1, bỏ Flows) On-chain không cải thiện. Tín hiệu On-chain đổi chậm (MA7/MA30, z-score 90 ngày),
nên có thể chỉ có ích ở tầm dài hơn. Ở đây đo lại cùng phép so sánh với nhãn 7 ngày:

    y_t = 1[close_{t+7} > close_t]            Δ = M(OHLCV + On-chain) − M(OHLCV-only)

Thiết kế (chốt trước khi chạy, không đổi theo kết quả test):
  - Dữ liệu, đặc trưng như pipeline/results/no_flows: 10 đặc trưng OHLCV + On-chain v1 bỏ Flows (27 đặc trưng).
  - Cùng lịch fold như E1 (train 730 ngày, test 30 ngày, bước 30, test từ 01/01/2020). Chỉ đổi gap_days 1 -> 7:
    nhãn ngày train cuối dùng close của 7 ngày sau, nên phải bỏ 7 ngày để nhãn train không chạm ngày test.
    Inner CV để tune: TimeSeriesSplit(5, gap=7). SVM (b3t) hiệu chỉnh xác suất cũng dùng gap 7.
  - 7 ngày cuối (25–31/08/2026) chưa có nhãn nên test kết thúc 24/08/2026: còn 81 fold (fold 81 của E1 chỉ
    test 28–30/08), cùng ngày test với E1 từ 01/01/2020 đến 24/08/2026.
  - Mô hình: b0maj, b0pers, b0rev (luật theo hướng 7 ngày qua), b1, b1t, b2, b2t, b3t; cùng grid/config như E1.
    (b4/b4t cần xgboost, b5 cần torch: không chạy ở đây.)
  - Nhãn 7 ngày chồng lấn nhau (ngày t và t+1 chung 6/7 đoạn giá) nên các ngày test không độc lập:
      * CI 95% bằng circular block bootstrap, khối 30 ngày (≥ 4 × tầm dự báo).
      * Δ log-loss kiểm bằng Diebold–Mariano, lag Newey–West = max(mặc định, 2 × 7).
      * Không dùng McNemar (giả định các ngày độc lập).
      * Kiểm thêm trên mẫu không chồng lấn: mỗi 7 ngày lấy 1 ngày (7 cách chọn ngày bắt đầu).
    Holm trên các mô hình có học, riêng cho DA và log-loss.
  - DA là metric chính, log-loss là metric phụ (đồng thời là tiêu chí tune), như kế hoạch cuối kỳ.

Chạy từ gốc repo:
  python analyses/horizon7/src/horizon7.py --test                       # test nhãn, fold, luật
  python analyses/horizon7/src/horizon7.py --stage run [--models b1 b2t]  # walk-forward -> results/predictions/
  python analyses/horizon7/src/horizon7.py --stage tables               # bảng + hình -> results/tables, reports/figures
  python analyses/horizon7/src/horizon7.py --stage forecast             # dự báo 7 ngày sau ngày cuối của data freeze
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pipeline/src"))

import b1t_tuning  # noqa: E402
import stats  # noqa: E402
from data import CLOSE, DATE, PREDICTION_COLS, TARGET, load_config, load_master, read_csv_dated, write_csv  # noqa: E402
from features import build_dataset  # noqa: E402
from hybrid import SIGN, with_sign  # noqa: E402
from models import load_model  # noqa: E402
from run_all import attach_regime, drop_groups_sets, load_regime_map, ohlcv10_sets  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from walkforward import fold_indices, inner_cv, make_folds, make_pipeline, run_walkforward, tune  # noqa: E402

H = 7
BLOCK = 30
CONFIGS = ("ohlcv", "ohlcv_onchain")
RULES = ("b0pers", "b0rev")
LEARNED = ("b1", "b1t", "b2", "b2t", "b3t")
MODELS = ("b0maj",) + RULES + LEARNED
INTERNAL_GAP = {"b3", "b3t", "b4", "b4t"}          # mô hình có CV/validation nội bộ, nhận tham số gap
BASE = ROOT / "analyses/horizon7"
OUT = BASE / "results"
FIG = BASE / "reports/figures"
H1_DIR = ROOT / "pipeline/results/no_flows"         # cùng bộ đặc trưng, tầm 1 ngày (để so)
NAMES = {"b0maj": "B0 Majority", "b0pers": "B0 Persistence 7d", "b0rev": "B0 Đảo chiều 7d", "b1": "B1 LR",
         "b1t": "B1t LR tune", "b2": "B2 RF", "b2t": "B2t RF tune", "b3t": "B3t SVM tune"}


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def make_target_h(close: pd.Series, h=H) -> pd.Series:
    """y_t = 1 nếu close_{t+h} > close_t; NaN nếu thiếu một trong hai (h ngày cuối)."""
    nxt = close.shift(-h)
    return (nxt > close).astype(float).where(nxt.notna() & close.notna())


def setup(h=H):
    """(data, sets, folds, wf). data[TARGET] là nhãn h ngày; sets như no_flows (10 OHLCV + On-chain v1 bỏ Flows)."""
    fe, wf = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    master = load_master()                                        # kiểm sha256 với MANIFEST
    data, sets = build_dataset(master, fe)
    sets = drop_groups_sets(ohlcv10_sets(sets), fe, ["Flows"])
    y = make_target_h(master.set_index(DATE)[CLOSE], h)
    data[TARGET] = data[DATE].map(y).to_numpy()
    data = with_sign(data)                                        # up_lag1 cho tập ứng viên của b1t
    wf = {**wf, "gap_days": h}
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf)
    return data, sets, folds, wf


# ----------------------------------------------------------------------------
# Mô hình
# ----------------------------------------------------------------------------
def direction_rule(data, folds, h=H, flip_below=None) -> pd.DataFrame:
    """Luật theo hướng h ngày qua: p̂_t = 1[close_t > close_{t−h}] = y_{t−h} (đã biết cuối ngày t).
    flip_below=None: persistence (xu hướng tiếp diễn). Có flip_below: đảo chiều nếu trong cửa sổ train persistence
    đúng < flip_below (như b0rev của E1, chiều chọn lại mỗi fold, chỉ dùng train)."""
    y_prev, out = data[TARGET].shift(h), []
    for f in folds:
        tr, te = fold_indices(data, f)
        ok = tr[y_prev.loc[tr].notna()]
        p = y_prev.loc[te].to_numpy(dtype=float)
        if flip_below is not None and (data.loc[ok, TARGET] == y_prev.loc[ok]).mean() < flip_below:
            p = 1.0 - p
        out.append(pd.DataFrame({"date": data.loc[te, DATE].to_numpy(), "fold": f.fold_id,
                                 "y_true": data.loc[te, TARGET].astype(int).to_numpy(), "p_hat": p, "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def b1t_sets(sets, config):
    s = b1t_tuning.feature_sets(sets["ohlcv"])
    if config == "ohlcv_onchain":
        oc = [c for c in sets["ohlcv_onchain"] if c not in sets["ohlcv"]]
        s = {f"{k}+oc": v + oc for k, v in s.items()}
    return s


def b1t_fit(train, cand, b):
    cols = cand[b["set"]]
    pipe = make_pipeline(b1t_tuning.estimator(), b1t_tuning.CLIP_Q).set_params(
        model__C=b["C"], model__l1_ratio=b["l1_ratio"],
        model__class_weight=None if b["class_weight"] == "none" else b["class_weight"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(train[cols].to_numpy(), train[TARGET].astype(int).to_numpy())
    return pipe, cols


def run_b1t(data, sets, folds, wf, config):
    """Như b1t_tuning.stage_run (tiêu chí log-loss), với nhãn h ngày và inner CV gap h."""
    cand = b1t_sets(sets, config)
    all_cols = sorted({c for v in cand.values() for c in v})
    out, log, b = [], [], None
    for i, f in enumerate(folds):
        tr, te = fold_indices(data, f)
        train, test = data.loc[tr].dropna(subset=all_cols), data.loc[te]
        if i % wf["retune_every"] == 0:
            b = b1t_tuning.pick(b1t_tuning.search(train, cand, wf), "logloss")
            log.append({"fold": f.fold_id, "set": b["set"], "C": b["C"], "l1_ratio": b["l1_ratio"],
                        "class_weight": b["class_weight"], "inner_logloss": b["inner_logloss"]})
        pipe, cols = b1t_fit(train, cand, b)
        out.append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": f.fold_id,
                                 "y_true": test[TARGET].astype(int).to_numpy(),
                                 "p_hat": pipe.predict_proba(test[cols].to_numpy())[:, 1], "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS], pd.DataFrame(log)


def estimator_for(name, h=H):
    est, grid = load_model(name)
    if name in INTERNAL_GAP:
        est.set_params(gap=h)
    return est, grid


def run_model(name, data, sets, folds, wf, config):
    """(predictions, tune_log)."""
    if name == "b0pers":
        return direction_rule(data, folds), pd.DataFrame()
    if name == "b0rev":
        return direction_rule(data, folds, flip_below=load_config("models/b0rev.yaml")["params"]["flip_below"]), \
            pd.DataFrame()
    if name == "b1t":
        return run_b1t(data, sets, folds, wf, config)
    est, grid = estimator_for(name)
    log = []
    p = run_walkforward(data, sets[config], est, folds, param_grid=grid, tune_log=log, **wf)
    return p, pd.DataFrame(log)


def stage_run(models):
    data, sets, folds, wf = setup()
    print(f"h={H}: {len(sets['ohlcv'])} OHLCV, {len(sets['ohlcv_onchain'])} tổng (bỏ Flows); {len(folds)} fold, "
          f"gap {wf['gap_days']}; test {folds[0].test_start.date()} → {folds[-1].test_end.date()}", flush=True)
    (OUT / "feature_sets.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_sets.json").write_text(json.dumps(
        {"horizon": H, "gap_days": wf["gap_days"], "drop_groups": ["Flows"], "sets": sets}, indent=1) + "\n")
    rmap = load_regime_map()
    for name in models:
        t0 = time.time()
        for config in CONFIGS:
            p, log = run_model(name, data, sets, folds, wf, config)
            write_csv(attach_regime(p, rmap), OUT / "predictions" / f"{name}_{config}.csv")
            if len(log):
                write_csv(log, OUT / "tune_log" / f"{name}_{config}.csv")
        print(f"== {name}: {time.time() - t0:.0f} giây", flush=True)


# ----------------------------------------------------------------------------
# Bảng
# ----------------------------------------------------------------------------
def load_preds(d):
    files = {tuple(f.stem.split("_", 1)): f for f in sorted(Path(d).glob("*_*.csv"))}
    return {k: read_csv_dated(f) for k, f in files.items() if k[1] in CONFIGS}


def metric_row(p, name):
    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    r = {"n_days": len(p), "pct_up": y.mean(), "da": stats.row_hit(y, ph).mean(), "pct_pred_up": (ph > 0.5).mean()}
    if name not in RULES:
        r |= {"log_loss": log_loss(y, ph, labels=[0, 1]), "brier": brier_score_loss(y, ph)}
    if name not in RULES and name != "b0maj":
        r["auc"] = roc_auc_score(y, ph)
    return r


def dm_lag(T, h=H):
    return max(int(np.floor(4 * (T / 100) ** (2 / 9))), 2 * h)


def delta_row(pa, pb, idx):
    """Δ = M(pb) − M(pa) cho DA (+CI) và log-loss (+CI, DM)."""
    if not np.array_equal(pa[DATE].to_numpy(), pb[DATE].to_numpy()):
        raise ValueError("Hai cấu hình không cùng ngày test.")
    y = pa["y_true"].to_numpy()
    d, lo, hi = stats.bootstrap_delta(y, pa["p_hat"], pb["p_hat"], "da", idx)
    r = {"d_da": d, "d_da_lo": lo, "d_da_hi": hi}
    la, lb = stats.row_log_loss(y, pa["p_hat"]), stats.row_log_loss(y, pb["p_hat"])
    d, lo, hi = stats.bootstrap_delta(y, pa["p_hat"], pb["p_hat"], "log_loss", idx)
    s, pv = stats.diebold_mariano(la, lb, lag=dm_lag(len(y)))
    return r | {"d_logloss": d, "d_logloss_lo": lo, "d_logloss_hi": hi, "dm_stat": s, "p_dm": pv}


def p_boot_da(y, pa, pb, idx):
    """p hai phía cho ΔDA = 0 từ phân phối bootstrap đã dời tâm về 0."""
    d = stats.row_hit(y, pb) - stats.row_hit(y, pa)
    boots = d[idx].mean(1) - d.mean()
    return float(min(1.0, (np.abs(boots) >= abs(d.mean())).mean() + 1 / len(boots)))


def nonoverlap(pa, pb):
    """ΔDA trên 7 mẫu không chồng lấn (mỗi 7 ngày một ngày, ngày bắt đầu lệch 0..6)."""
    k = (pa[DATE] - pa[DATE].iloc[0]).dt.days.to_numpy() % H
    hit = lambda p, m: stats.row_hit(p["y_true"].to_numpy()[m], p["p_hat"].to_numpy()[m]).mean()  # noqa: E731
    d = np.array([hit(pb, k == o) - hit(pa, k == o) for o in range(H)])
    return {"nonoverlap_n": int((k == 0).sum()), "nonoverlap_d_da_mean": d.mean(), "nonoverlap_d_da_min": d.min(),
            "nonoverlap_d_da_max": d.max(), "nonoverlap_n_pos": int((d > 0).sum())}


def year_rows(name, pa, pb):
    rows = []
    years = pa[DATE].dt.year.to_numpy()
    for yr in np.unique(years):
        m = years == yr
        qa, qb = pa[m].reset_index(drop=True), pb[m].reset_index(drop=True)
        r = {"model": name, "year": int(yr), "n_days": int(m.sum()),
             "da_ohlcv": stats.row_hit(qa.y_true, qa.p_hat).mean(), "da_onchain": stats.row_hit(qb.y_true, qb.p_hat).mean()}
        r |= delta_row(qa, qb, stats.block_indices(int(m.sum()), BLOCK))
        rows.append(r)
    return rows


def h1_delta():
    """Δ của cùng mô hình ở tầm 1 ngày (pipeline/results/no_flows), cùng cách tính (khối 30 ngày)."""
    preds = load_preds(H1_DIR / "predictions")
    rows = []
    for name in LEARNED:
        if (name, "ohlcv") in preds and (name, "ohlcv_onchain") in preds:
            pa, pb = preds[(name, "ohlcv")], preds[(name, "ohlcv_onchain")]
            r = {"model": name, "horizon": 1, "da_ohlcv": stats.row_hit(pa.y_true, pa.p_hat).mean(),
                 "da_onchain": stats.row_hit(pb.y_true, pb.p_hat).mean()}
            rows.append(r | delta_row(pa, pb, stats.block_indices(len(pa), BLOCK)))
    return pd.DataFrame(rows)


def stage_tables():
    preds = load_preds(OUT / "predictions")
    tdir = OUT / "tables"
    # 1) metric từng mô hình × cấu hình
    met = pd.DataFrame([{"model": n, "config": c, **metric_row(p, n)} for (n, c), p in sorted(preds.items())])
    write_csv(met, tdir / "metrics.csv")
    # 2) Δ On-chain
    models = [m for m in LEARNED if (m, "ohlcv") in preds and (m, "ohlcv_onchain") in preds]
    T = len(preds[(models[0], "ohlcv")])
    idx = stats.block_indices(T, BLOCK)
    rows, yrows = [], []
    for n in models:
        pa, pb = preds[(n, "ohlcv")], preds[(n, "ohlcv_onchain")]
        r = {"model": n, "horizon": H, "da_ohlcv": stats.row_hit(pa.y_true, pa.p_hat).mean(),
             "da_onchain": stats.row_hit(pb.y_true, pb.p_hat).mean()}
        r |= delta_row(pa, pb, idx)
        r["p_boot_da"] = p_boot_da(pa.y_true.to_numpy(), pa.p_hat, pb.p_hat, idx)
        rows.append(r | nonoverlap(pa, pb))
        yrows += year_rows(n, pa, pb)
    delta = pd.DataFrame(rows)
    delta["p_boot_da_holm"] = stats.holm(delta["p_boot_da"])
    delta["p_dm_holm"] = stats.holm(delta["p_dm"])
    write_csv(delta, tdir / "delta_onchain.csv")
    yr = pd.DataFrame(yrows)
    write_csv(yr, tdir / "delta_by_year.csv")
    cons = yr.groupby("model").agg(n_years=("year", "size"), n_years_da_up=("d_da", lambda s: int((s > 0).sum())),
                                   n_years_ll_down=("d_logloss", lambda s: int((s < 0).sum())),
                                   n_years_da_ci_above0=("d_da_lo", lambda s: int((s > 0).sum())),
                                   n_years_da_ci_below0=("d_da_hi", lambda s: int((s < 0).sum()))).reset_index()
    write_csv(cons, tdir / "consistency_by_year.csv")
    # 3) mô hình vs luật / majority (cùng tầm 7 ngày)
    vs = []
    for n in models:
        for c in CONFIGS:
            p = preds[(n, c)]
            for ref in ("b0maj", "b0pers", "b0rev"):
                q = preds[(ref, "ohlcv")]
                d, lo, hi = stats.bootstrap_delta(p.y_true.to_numpy(), q.p_hat, p.p_hat, "da", idx)
                vs.append({"model": n, "config": c, "ref": ref, "d_da": d, "d_da_lo": lo, "d_da_hi": hi})
    write_csv(pd.DataFrame(vs), tdir / "vs_baselines.csv")
    # 4) so với tầm 1 ngày
    h1 = h1_delta()
    both = pd.concat([h1, delta[h1.columns.intersection(delta.columns)]], ignore_index=True)
    write_csv(both, tdir / "h1_vs_h7.csv")
    # 5) Δ theo regime
    reg = []
    for n in models:
        pa, pb = preds[(n, "ohlcv")], preds[(n, "ohlcv_onchain")]
        for g in sorted(pa["regime"].dropna().unique()):
            m = (pa["regime"] == g).to_numpy()
            reg.append({"model": n, "regime": g, "n_days": int(m.sum()),
                        "d_da": stats.row_hit(pb.y_true[m], pb.p_hat[m]).mean() - stats.row_hit(pa.y_true[m], pa.p_hat[m]).mean(),
                        "d_logloss": (stats.row_log_loss(pb.y_true[m], pb.p_hat[m]) - stats.row_log_loss(pa.y_true[m], pa.p_hat[m])).mean()})
    write_csv(pd.DataFrame(reg), tdir / "delta_by_regime.csv")
    figures(delta, h1, yr)
    print(delta.round(4).to_string())
    return delta


def figures(delta, h1, yr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    models = list(delta["model"])
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (col, lab) in zip(axes, (("d_da", "ΔDA (cao hơn = On-chain tốt hơn)"),
                                     ("d_logloss", "Δ log-loss (thấp hơn = On-chain tốt hơn)"))):
        for j, (df, hz, colr) in enumerate(((h1, 1, "#9AA5B1"), (delta, H, "#2F6DB5"))):
            d = df.set_index("model").reindex(models)
            ax.errorbar(x + (j - 0.5) * 0.3, d[col], yerr=[d[col] - d[col + "_lo"], d[col + "_hi"] - d[col]],
                        fmt="o", color=colr, capsize=3, label=f"Tầm {hz} ngày")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [NAMES[m] for m in models], rotation=20)
        ax.set_title(lab)
        ax.legend()
    fig.suptitle("Δ = OHLCV + On-chain (bỏ Flows) − OHLCV-only, CI 95% block bootstrap (khối 30 ngày)")
    fig.tight_layout()
    fig.savefig(FIG / "delta_h1_vs_h7.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for n, g in yr.groupby("model"):
        ax.plot(g["year"], g["d_da"], marker="o", label=NAMES[n])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔDA")
    ax.set_title(f"ΔDA On-chain theo năm, tầm {H} ngày")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "delta_da_by_year_h7.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Dự báo 7 ngày sau ngày cuối của data freeze (không chấm được: freeze chưa có giá sau 31/08/2026)
# ----------------------------------------------------------------------------
def stage_forecast(models=("b1", "b1t", "b2t")):
    data, sets, folds, wf = setup()
    lab = data.loc[data[TARGET].notna()]
    train_end = lab[DATE].max()                                   # ngày cuối có nhãn 7 ngày
    tr = data.index[data[DATE].between(train_end - pd.Timedelta(days=wf["train_days"] - 1), train_end)]
    te = data.index[data[DATE] > train_end]                       # 7 ngày cuối: chưa biết kết quả
    rows = []
    for name in models:
        for config in CONFIGS:
            if name == "b1t":
                cand = b1t_sets(sets, config)
                train = data.loc[tr].dropna(subset=sorted({c for v in cand.values() for c in v}))
                b = b1t_tuning.pick(b1t_tuning.search(train, cand, wf), "logloss")
                pipe, cols = b1t_fit(train, cand, b)
                prm = f"{b['set']}, C={b['C']:g}, l1={b['l1_ratio']:g}, cw={b['class_weight']}"
            else:
                cols = sets[config]
                train = data.loc[tr].dropna(subset=cols)
                est, grid = estimator_for(name)
                X, y = train[cols].to_numpy(), train[TARGET].astype(int).to_numpy()
                pipe = make_pipeline(est)
                best = tune(pipe, grid, X, y, inner_cv(wf["inner_splits"], wf["gap_days"]))
                pipe.set_params(**{f"model__{k}": v for k, v in best.items()}).fit(X, y)
                prm = ", ".join(f"{k}={v}" for k, v in best.items())
            p = pipe.predict_proba(data.loc[te, cols].to_numpy())[:, 1]
            rows += [{"date": d.date(), "target_date": (d + pd.Timedelta(days=H)).date(), "model": name,
                      "config": config, "p_up": pr, "params": prm} for d, pr in zip(data.loc[te, DATE], p)]
            print(f"{name}_{config}: {prm}", flush=True)
    out = pd.DataFrame(rows)
    write_csv(out, OUT / "tables" / "forecast_next7.csv")
    print(out.pivot_table(index="date", columns=["model", "config"], values="p_up").round(3).to_string())


# ----------------------------------------------------------------------------
def _tests():
    close = pd.Series([1.0, 2, 3, 2, 1, 5, 6, 7, 8, 0.5])
    y = make_target_h(close, 7)
    assert y.iloc[:3].tolist() == [1.0, 1.0, 0.0] and y.iloc[3:].isna().all()
    data, sets, folds, wf = setup()
    assert len(sets["ohlcv"]) == 10 and len(sets["ohlcv_onchain"]) == 37
    assert not any("Netflow" in c or "SplyEx" in c for c in sets["ohlcv_onchain"])
    assert len(folds) == 81 and folds[0].test_start == pd.Timestamp("2020-01-01")
    c = load_master().set_index(DATE)[CLOSE]
    for f in folds:
        tr, te = fold_indices(data, f)
        # ngày giá cuối mà nhãn train dùng tới vẫn trước ngày test đầu
        assert data.loc[tr, DATE].max() + pd.Timedelta(days=H) < data.loc[te, DATE].min()
        assert len(tr) == wf["train_days"]
    d = data.set_index(DATE)
    t = pd.Timestamp("2023-05-10")
    assert d.loc[t, TARGET] == float(c[t + pd.Timedelta(days=7)] > c[t])
    for a, b in inner_cv(5, H).split(np.zeros((730, 1))):
        assert a.max() + H < b.min()
    p = direction_rule(data, folds[:3])
    np.testing.assert_array_equal(p["p_hat"], (c > c.shift(7)).astype(float).loc[p["date"]].to_numpy())
    assert SIGN in data
    print("Tất cả test horizon7 đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run", "tables", "forecast"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)
    if a.test or a.stage is None:
        _tests()
    if a.stage == "run":
        stage_run(a.models)
    elif a.stage == "tables":
        stage_tables()
    elif a.stage == "forecast":
        stage_forecast()


if __name__ == "__main__":
    main()
