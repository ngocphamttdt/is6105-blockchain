"""
shap_analysis.py  –  SHAP cho 7 mô hình có học trên cấu hình ohlcv_onchain bỏ nhóm Flows (mốc pipeline/results/no_flows).

Dựng lại từng fold đúng mô hình đã sinh p̂ trong no_flows/predictions, kiểm p̂ dựng lại (lệch > 1e-3 thì bỏ model),
rồi tính SHAP từng fold, gộp theo nhóm và theo năm, so với ablation, vẽ hình. Kết quả ở pipeline/results/shap/:
shap_reproduce.csv, shap_values/, shap_features.csv, shap_groups.csv, shap_groups_by_year.csv, shap_vs_ablation.csv, *.png.

  python pipeline/src/shap_analysis.py                       # đủ 7 model, 82 fold
  python pipeline/src/shap_analysis.py --models b1 b2 --every 20   # chạy thử nhanh
  python pipeline/src/shap_analysis.py --test
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import features
from b1t_tuning import CLIP_Q, ElasticNetLR, feature_sets
from data import (DATE, RESULTS, ROOT, TARGET, load_config, load_master, make_synthetic_close_1h,
                  make_synthetic_master, read_csv_dated, write_csv)
from features import build_dataset
from features_ohlcv import FEATURES_10
from hybrid import SIGN, with_sign
from models import load_model
from run_all import drop_groups_sets, ohlcv10_sets
from walkforward import fold_indices, inner_cv, make_folds, make_pipeline, run_walkforward, tune

UNIT = {"b1": "log-odds", "b1fs": "log-odds", "b1t": "log-odds", "b2": "probability", "b2t": "probability",
        "b4": "log-odds", "b4t": "log-odds"}
ONCHAIN = ["Activity", "Mining", "Valuation"]
SHAP_MODELS = ["b1", "b1fs", "b1t", "b2", "b2t", "b4", "b4t"]
REF = ROOT / "pipeline/results/no_flows"
OUT = RESULTS / "shap"
TOL = 1e-3


def setup(master=None, close_1h=None):
    """(data, cols, folds, wf_cfg, fe_cfg) như rerun_e1.main --no-flows: 10 OHLCV + On-chain v1 bỏ Flows.
    data có thêm up_lag1 (cột của b1t); các mô hình khác chỉ dùng cols nên không bị ảnh hưởng."""
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    master = load_master() if master is None else master
    data, sets = build_dataset(master, fe_cfg, close_1h) if close_1h is not None else build_dataset(master, fe_cfg)
    cols = drop_groups_sets(ohlcv10_sets(sets), fe_cfg, ["Flows"])["ohlcv_onchain"]
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)
    return with_sign(data), cols, folds, wf_cfg, fe_cfg


def _b1t_row(chosen, fold_id):
    """Dòng chosen.csv (tiêu chí log-loss) có fold lớn nhất <= fold_id: lần tune đang có hiệu lực."""
    c = chosen[(chosen["criterion"] == "logloss") & (chosen["fold"] <= fold_id)]
    if c.empty:
        raise ValueError(f"b1t_chosen không có dòng nào cho fold <= {fold_id}")
    return c.loc[c["fold"].idxmax()]


def refit_folds(name, data, cols, folds, wf, b1t_chosen=None, every=1):
    """Dựng lại từng fold đúng như mô hình đã sinh p̂ trong no_flows.
    Yield (fold, pipe đã fit, cột mô hình dùng, test_df, train_X = train[used] đúng các dòng đã fit) cho folds[::every] (chỉ số theo danh sách `folds` đầy đủ).
    Tune vẫn rơi vào i % retune_every == 0 của danh sách đầy đủ, nên tham số của fold được chọn đúng như bản gốc;
    fold không được chọn và không phải fold tune thì bỏ qua (không fit). b1t cần b1t_chosen (DataFrame của chosen.csv)."""
    cols = list(cols)
    if name == "b1t":
        if b1t_chosen is None:
            raise ValueError("b1t cần b1t_chosen (chosen.csv)")
        oc = [c for c in cols if c not in FEATURES_10]
        sets = {f"{k}+oc": v + oc for k, v in feature_sets(FEATURES_10).items()}
        sets |= feature_sets(FEATURES_10)           # tên không "+oc" cho tập chỉ-OHLCV nếu chosen.csv có
        union = sorted({c for k, v in sets.items() if k.endswith("+oc") for c in v})   # như b1t_tuning.stage_run
    else:
        est, grid = load_model(name)
        best = {}
    for i, fold in enumerate(folds):
        selected = i % every == 0
        retune = name != "b1t" and bool(grid) and i % wf["retune_every"] == 0
        if not selected and not retune:
            continue
        tr, te = fold_indices(data, fold)
        window = (fold.train_end - fold.train_start).days + 1
        if len(tr) < window:
            raise ValueError(f"Fold {fold.fold_id}: chỉ {len(tr)}/{window} ngày train có nhãn.")
        if name == "b1t":
            b = _b1t_row(b1t_chosen, fold.fold_id)
            used = list(sets[b["set"]])
            train, test = data.loc[tr].dropna(subset=union), data.loc[te]
            cw = b["class_weight"]
            pipe = make_pipeline(ElasticNetLR(), CLIP_Q).set_params(
                model__C=float(b["C"]), model__l1_ratio=float(b["l1_ratio"]),
                model__class_weight=None if cw == "none" or pd.isna(cw) else cw)
        else:
            used = cols
            train, test = data.loc[tr].dropna(subset=used), data.loc[te]
            pipe = make_pipeline(est, wf.get("clip_q"))
        if test[used].isna().any().any():
            raise ValueError(f"Fold {fold.fold_id}: có NaN trong đặc trưng của tập test.")
        X, y = train[used].to_numpy(), train[TARGET].astype(int).to_numpy()
        if name != "b1t":
            if retune:
                best = tune(pipe, grid, X, y, inner_cv(wf["inner_splits"], wf["gap_days"]))
            if not selected:
                continue
            pipe.set_params(**{f"model__{k}": v for k, v in best.items()})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X, y)
        yield fold, pipe, used, test, train[used]


def check_reproduce(name, p_rebuilt, ref_dir=REF):
    """max |Δp̂| theo date giữa p̂ dựng lại (cột date, p_hat) và {ref_dir}/predictions/{name}_ohlcv_onchain.csv."""
    ref = read_csv_dated(Path(ref_dir) / "predictions" / f"{name}_ohlcv_onchain.csv")
    new = p_rebuilt[[DATE, "p_hat"]]
    missing = ~new[DATE].isin(ref[DATE])
    if missing.any():
        raise ValueError(f"{name}: {int(missing.sum())} ngày dựng lại không có trong file tham chiếu")
    m = new.merge(ref[[DATE, "p_hat"]], on=DATE, suffixes=("_new", "_ref"))     # chỉ so các ngày dựng lại
    if m.empty:
        raise ValueError(f"{name}: không có ngày nào để so")
    if m[["p_hat_new", "p_hat_ref"]].isna().any().any():
        raise ValueError(f"{name}: p̂ có NaN trên các ngày so sánh")
    return float((m["p_hat_ref"] - m["p_hat_new"]).abs().max())


def _lr_parts(model):
    """(coef, intercept) của LR/ElasticNetLR; đầu ra mô hình = coef·x + intercept (log-odds)."""
    return np.asarray(model.coef_, float).ravel(), float(np.ravel(model.intercept_)[0])


def _explain(name, pipe, used, train_X, test_X, all_cols):
    """(SHAP (len(test_X), len(all_cols)), expected_value). Cột ngoài `used` (hay ngoài support_ của b1fs) = 0.
    SHAP tính trên đặc trưng đã qua các bước trước "model" (clip, scaler), đúng đầu vào của mô hình."""
    import shap
    pre, model = pipe[:-1], pipe["model"]
    Xtr, Xte = pre.transform(np.asarray(train_X, float)), pre.transform(np.asarray(test_X, float))
    idx = np.arange(len(used))                       # cột (trong `used`) mà mô hình thực sự dùng
    if name == "b1fs":
        idx = np.asarray(model.support_)
        inner = model.model_                         # Pipeline(StandardScaler, LR) nằm trong ForwardSelectLR
        Xtr, Xte = inner[0].transform(Xtr[:, idx]), inner[0].transform(Xte[:, idx])
        lr = inner[1]
    elif name in ("b1", "b1t"):
        lr = model
    if name in ("b1", "b1fs", "b1t"):
        expl = shap.LinearExplainer(_lr_parts(lr), shap.maskers.Independent(Xtr, max_samples=len(Xtr)))
        sv, ev = np.asarray(expl.shap_values(Xte)), float(np.ravel(expl.expected_value)[0])
    else:
        expl = shap.TreeExplainer(model if name in ("b2", "b2t") else model.model_)
        sv, ev = expl.shap_values(Xte), np.ravel(expl.expected_value)
        if isinstance(sv, list):                     # RF: danh sách theo lớp
            sv = sv[1]
        elif np.ndim(sv) == 3:                       # RF: (n, p, lớp)
            sv = sv[:, :, 1]
        sv = np.asarray(sv)
        ev = float(ev[1] if len(ev) > 1 else ev[0])
    out = np.zeros((len(Xte), len(all_cols)))
    pos = {c: j for j, c in enumerate(all_cols)}
    for k, i in enumerate(idx):
        out[:, pos[used[i]]] = sv[:, k]
    return out, ev


def explain_fold(name, pipe, used, train_X, test_X, all_cols):
    """SHAP (len(test_X), len(all_cols)); đơn vị theo UNIT[name]."""
    return _explain(name, pipe, used, train_X, test_X, all_cols)[0]


def run(models, every=1, ctx=None, b1t_chosen=None, ref_dir=REF, out_dir=OUT, verbose=True):
    """SHAP từng fold cho từng mô hình; kiểm p̂ dựng lại với mốc, vượt TOL thì bỏ mô hình (ghi lý do).
    Ghi {out_dir}/shap_values/{model}.csv và shap_reproduce.csv. Trả {"values", "reproduce", "dropped"}."""
    data, cols, folds, wf, fe_cfg = setup() if ctx is None else ctx
    all_cols = list(cols) + [SIGN]
    if b1t_chosen is None and "b1t" in models:
        b1t_chosen = pd.read_csv(Path(ref_dir) / "log" / "b1t_ohlcv_onchain" / "log" / "chosen.csv")
    values, dropped, rows = {}, {}, []
    for name in models:
        parts, preds = [], []
        for fold, pipe, used, test, train in refit_folds(name, data, cols, folds, wf, b1t_chosen, every):
            sv = explain_fold(name, pipe, used, train, test[used], all_cols)
            dates = test[DATE].to_numpy()
            parts.append(pd.DataFrame(sv, columns=all_cols).assign(date=dates, fold=fold.fold_id, unit=UNIT[name]))
            preds.append(pd.DataFrame({DATE: dates, "p_hat": pipe.predict_proba(test[used].to_numpy())[:, 1]}))
        df = pd.concat(parts, ignore_index=True)[["date", "fold", "unit"] + all_cols]
        dp = check_reproduce(name, pd.concat(preds, ignore_index=True), ref_dir)
        kept = dp <= TOL
        rows.append({"model": name, "max_abs_dp": dp, "kept": kept})
        if verbose:
            print(f"{name}: max_abs_dp = {dp:.3e} {'giữ' if kept else 'BỎ (> ' + str(TOL) + ')'}", flush=True)
        if kept:
            values[name] = df
            write_csv(df, Path(out_dir) / "shap_values" / f"{name}.csv")
        else:
            dropped[name] = f"max|Δp̂| = {dp:.3e} > {TOL}"
    rep = pd.DataFrame(rows, columns=["model", "max_abs_dp", "kept"])
    write_csv(rep, Path(out_dir) / "shap_reproduce.csv")
    return {"values": values, "reproduce": rep, "dropped": dropped}


def summarize(values, groups):
    """(shap_features, shap_groups, shap_groups_by_year) từ {model: df(date, fold, unit, cột SHAP)}.
    Tổng theo nhóm = tổng mean|SHAP| của các đặc trưng trong nhóm, so sánh chỉ trong cùng mô hình (đơn vị khác nhau)."""
    feats, grp, yr = [], [], []
    for model, df in values.items():
        fc = [c for c in df.columns if c not in ("date", "fold", "unit")]
        a = df[fc].abs()
        m = a.mean().sort_values(ascending=False, kind="stable")
        f = pd.DataFrame({"model": model, "unit": df["unit"].iloc[0], "feature": m.index, "mean_abs_shap": m.to_numpy()})
        f["rank"] = np.arange(1, len(f) + 1)
        feats.append(f)

        def agg(sub):
            g = {k: float(sub[[c for c in v if c in fc]].sum(axis=1).mean()) for k, v in groups.items()}
            tot = sum(g.values())
            g["onchain_share"] = sum(x for k, x in g.items() if k != "OHLCV") / tot if tot > 0 else np.nan
            return g
        grp.append({"model": model, "unit": df["unit"].iloc[0], **agg(a)})
        years = pd.to_datetime(df["date"]).dt.year
        for y in sorted(years.unique()):
            yr.append({"model": model, "unit": df["unit"].iloc[0], "year": int(y), **agg(a[(years == y).to_numpy()])})
    return pd.concat(feats, ignore_index=True), pd.DataFrame(grp), pd.DataFrame(yr)


def compare_ablation(shap_groups, ablation_rank):
    """Xếp hạng 3 nhóm On-chain theo mean|SHAP| giảm dần (hoà giữ thứ tự Activity, Mining, Valuation)
    so với rank_plus / rank_drop của ablation."""
    ab = ablation_rank.set_index("model")
    rows = []
    for _, r in shap_groups.iterrows():
        order = sorted(ONCHAIN, key=lambda g: -r[g])          # sorted ổn định
        shap_rank = ">".join(order)
        plus = ab["rank_plus"].get(r["model"], np.nan)
        drop = ab["rank_drop"].get(r["model"], np.nan)
        if not isinstance(plus, str):
            match = "không có ablation"
        elif shap_rank == plus:
            match = "khớp hoàn toàn"
        elif shap_rank.split(">")[0] == plus.split(">")[0]:
            match = "khớp nhóm đứng đầu"
        else:
            match = "không khớp"
        rows.append({"model": r["model"], "rank_shap": shap_rank, "rank_plus": plus, "rank_drop": drop, "match": match})
    return pd.DataFrame(rows, columns=["model", "rank_shap", "rank_plus", "rank_drop", "match"])


def make_figures(shap_groups, shap_features, out_dir=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gcols = [c for c in shap_groups.columns if c not in ("model", "unit", "onchain_share")]
    pct = shap_groups[gcols].div(shap_groups[gcols].sum(axis=1), axis=0) * 100
    ax = pct.set_axis(shap_groups["model"]).plot.bar(stacked=True, figsize=(8, 4.5))
    ax.set_ylabel("% tổng mean |SHAP| (trong cùng mô hình)")
    ax.set_title("Đóng góp SHAP theo nhóm")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "shap_groups.png", dpi=150)
    plt.close(ax.figure)
    models = [m for m in ("b1t", "b2t") if m in set(shap_features["model"])]
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 7), squeeze=False)
        for a, m in zip(axes[0], models):
            t = shap_features[shap_features.model == m].nsmallest(20, "rank").iloc[::-1]
            a.barh(t["feature"], t["mean_abs_shap"])
            a.set_title(f"{m}: 20 đặc trưng đầu ({t['unit'].iloc[0]})")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_top_features.png", dpi=150)
        plt.close(fig)


def _tests_shap(data, cols, folds, wf, fe_cfg):
    from scipy.special import logit
    all_cols = list(cols) + [SIGN]
    ch = pd.DataFrame([{"criterion": "logloss", "fold": folds[0].fold_id, "set": "lag1+sign+oc", "C": 0.001,
                        "l1_ratio": 0.0, "class_weight": "balanced"},
                       {"criterion": "logloss", "fold": folds[6].fold_id, "set": "sign", "C": 0.01,
                        "l1_ratio": 0.5, "class_weight": "none"}])
    # 6) cộng lại đúng đầu ra mô hình theo đơn vị; cột không dùng = 0; không NaN
    for name in SHAP_MODELS:
        n = 0
        # b1t không tune (chọn từ chosen.csv) nên chạy fold 0 và 6 (tập "lag1+sign+oc" rồi "sign"); mô hình còn lại
        # chỉ fold 0 (tune RF/XGB/b1fs trên 730 ngày rất chậm)
        fs, ev_k = (folds[:7], 6) if name == "b1t" else (folds[:1], 1)
        for fold, pipe, used, test, train in refit_folds(name, data, cols, fs, wf, b1t_chosen=ch, every=ev_k):
            sv, ev = _explain(name, pipe, used, train, test[used], all_cols)
            assert sv.shape == (len(test), len(all_cols)) and not np.isnan(sv).any(), name
            p = pipe.predict_proba(test[used].to_numpy())[:, 1]
            out = logit(p) if UNIT[name] == "log-odds" else p
            err = np.abs(sv.sum(axis=1) + ev - out).max()
            assert err < (1e-6 if name in ("b1", "b1fs", "b1t") else 1e-4), (name, fold.fold_id, err)
            assert np.allclose(explain_fold(name, pipe, used, train, test[used], all_cols), sv)
            zero = [j for j, c in enumerate(all_cols) if c not in used]
            if name == "b1fs":
                zero = [j for j, c in enumerate(all_cols) if c not in [used[k] for k in pipe["model"].support_]]
                assert 0 < len(pipe["model"].support_) <= len(used)
            assert (sv[:, zero] == 0).all(), name
            if name == "b1t" and used == [SIGN]:
                assert (np.delete(sv, all_cols.index(SIGN), axis=1) == 0).all()
            n += 1
        assert n == (2 if name == "b1t" else 1), (name, n)

    # 7) summarize trên SHAP giả biết trước
    groups = {"OHLCV": ["o1", "o2"], "Activity": ["a1"], "Mining": ["m1"], "Valuation": ["v1"]}
    d = pd.DataFrame({"date": pd.date_range("2020-12-30", periods=4), "fold": 0, "unit": "log-odds",
                      "o1": [0.3, -0.3, 0.3, -0.3], "o2": [0.3, 0.3, -0.3, -0.3], "a1": [0.1, -0.1, 0.1, -0.1],
                      "m1": [0.2, 0.2, -0.2, -0.2], "v1": [-0.1, 0.1, 0.1, -0.1]})
    feats, grp, by_year = summarize({"b1": d}, groups)
    assert abs(grp.loc[0, "onchain_share"] - 0.4) < 1e-12 and abs(grp.loc[0, "OHLCV"] - 0.6) < 1e-12
    assert feats[feats.feature == "o1"]["rank"].iloc[0] == 1 and sorted(feats["rank"]) == [1, 2, 3, 4, 5]
    assert set(by_year["year"]) == {2020, 2021} and len(by_year) == 2
    cmp = compare_ablation(grp, pd.DataFrame({"model": ["b1"], "rank_plus": ["Mining>Activity>Valuation"],
                                              "rank_drop": ["Mining>Valuation>Activity"]}))
    assert cmp.loc[0, "rank_shap"] == "Mining>Activity>Valuation" and cmp.loc[0, "match"] == "khớp hoàn toàn"
    for plus, want in (("Mining>Valuation>Activity", "khớp nhóm đứng đầu"), ("Valuation>Mining>Activity", "không khớp")):
        c = compare_ablation(grp, pd.DataFrame({"model": ["b1"], "rank_plus": [plus], "rank_drop": [np.nan]}))
        assert c.loc[0, "match"] == want, (plus, c.loc[0, "match"])
    c = compare_ablation(grp, pd.DataFrame({"model": ["b2"], "rank_plus": [np.nan], "rank_drop": [np.nan]}))
    assert c.loc[0, "match"] == "không có ablation"

    # 8) run: mô hình khớp mốc được giữ, mô hình lệch > TOL bị bỏ kèm lý do
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("b1", "b2"):
            est, grid = load_model(name)
            ref = run_walkforward(data, cols, est, folds[:1], param_grid=grid, **wf)
            if name == "b2":
                ref["p_hat"] += 0.01
            write_csv(ref, Path(tmp) / "ref" / "predictions" / f"{name}_ohlcv_onchain.csv")
        res = run(["b1", "b2"], ctx=(data, cols, folds[:1], wf, fe_cfg), ref_dir=Path(tmp) / "ref", out_dir=Path(tmp) / "out", verbose=False)
        rep = pd.read_csv(Path(tmp) / "out" / "shap_reproduce.csv").set_index("model")
        assert rep.loc["b1", "kept"] and not rep.loc["b2", "kept"] and abs(rep.loc["b2", "max_abs_dp"] - 0.01) < 1e-9
        assert list(res["values"]) == ["b1"] and "b2" in res["dropped"]
        v = pd.read_csv(Path(tmp) / "out" / "shap_values" / "b1.csv")
        assert list(v.columns) == ["date", "fold", "unit"] + all_cols and (v["unit"] == "log-odds").all()
        assert len(v) == 30 and not (Path(tmp) / "out" / "shap_values" / "b2.csv").exists()


def _tests():
    m = make_synthetic_master(seed=42)
    data, cols, folds, wf, fe_cfg = setup(m, make_synthetic_close_1h(m))
    folds = folds[:7]
    # 1) 10 OHLCV + 27 On-chain, không có Flows, data có up_lag1
    flows = features.feature_groups(**fe_cfg)["Flows"]
    assert len(cols) == 37 and cols[:10] == FEATURES_10 and not set(cols) & set(flows), len(cols)
    assert SIGN in data.columns

    # 2) p̂ dựng lại == p̂ của run_walkforward (cùng tune, cùng fold)
    for name in ("b1", "b2"):
        est, grid = load_model(name)
        ref = run_walkforward(data, cols, est, folds, param_grid=grid, **wf)
        parts = []
        for fold, pipe, used, test, _ in refit_folds(name, data, cols, folds, wf):
            parts.append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": fold.fold_id,
                                       "p_hat": pipe.predict_proba(test[used].to_numpy())[:, 1]}))
        got = pd.concat(parts, ignore_index=True)
        assert len(got) == len(ref) and (got[DATE].to_numpy() == ref[DATE].to_numpy()).all(), name
        err = np.abs(got.p_hat.to_numpy() - ref.p_hat.to_numpy()).max()
        assert err < 1e-12, (name, err)

    # 2b) tập con không phải tiền tố: every=4 trên 13 fold -> fold 0,4,8,12; fold 8 dùng tune của fold 6
    full = make_folds(data.loc[data[TARGET].notna(), DATE], **wf)[:13]
    assert len(full) == 13
    est, grid = load_model("b2")
    ref = run_walkforward(data, cols, est, full, param_grid=grid, **wf).set_index("fold")
    got = list(refit_folds("b2", data, cols, full, wf, every=4))
    assert [g[0].fold_id for g in got] == [full[i].fold_id for i in (0, 4, 8, 12)]
    for fold, pipe, used, test, _ in got:
        r = ref.loc[fold.fold_id]
        assert (r[DATE].to_numpy() == test[DATE].to_numpy()).all()
        err = np.abs(pipe.predict_proba(test[used].to_numpy())[:, 1] - r.p_hat.to_numpy()).max()
        assert err < 1e-12, (fold.fold_id, err)

    # 3) b1t: tập đặc trưng theo b1t_chosen (hàng có fold lớn nhất <= fold hiện tại)
    ch = pd.DataFrame([{"criterion": "logloss", "fold": folds[0].fold_id, "set": "lag1+sign+oc", "C": 0.001,
                        "l1_ratio": 0.0, "class_weight": "balanced"},
                       {"criterion": "logloss", "fold": folds[6].fold_id, "set": "sign", "C": 0.01,
                        "l1_ratio": 0.5, "class_weight": "none"}])
    oc = cols[10:]
    seen = []
    for fold, pipe, used, test, _ in refit_folds("b1t", data, cols, folds, wf, b1t_chosen=ch):
        seen.append(used)
        assert pipe.predict_proba(test[used].to_numpy()).shape == (len(test), 2)
        assert pipe.named_steps["model"].coef_.shape[1] == len(used)
    assert len(seen) == 7
    for u in seen[:6]:
        assert u == ["logret_lag1", SIGN] + oc, u
    assert seen[6] == [SIGN], seen[6]
    assert pipe.named_steps["model"].class_weight is None and pipe.named_steps["model"].l1_ratio == 0.5

    # 4) check_reproduce: giống → 0.0; lệch một ngày 0,002 → 0.002
    with tempfile.TemporaryDirectory() as tmp:
        p = pd.DataFrame({DATE: pd.date_range("2020-01-01", periods=5), "fold": 0, "y_true": [0, 1, 0, 1, 1],
                          "p_hat": [0.4, 0.5, 0.6, 0.55, 0.45], "regime": pd.NA})
        write_csv(p, Path(tmp) / "predictions" / "b1_ohlcv_onchain.csv")
        assert check_reproduce("b1", p[[DATE, "p_hat"]], ref_dir=tmp) == 0.0
        q = p.copy()
        q.loc[2, "p_hat"] += 0.002
        assert abs(check_reproduce("b1", q[[DATE, "p_hat"]], ref_dir=tmp) - 0.002) < 1e-9
    # 5) check_reproduce: tập con ngày OK; ngày lạ, rỗng, NaN thì báo lỗi
        sub = p.iloc[[1, 3]][[DATE, "p_hat"]]
        assert check_reproduce("b1", sub, ref_dir=tmp) == 0.0
        for bad in (pd.DataFrame({DATE: [pd.Timestamp("2030-01-01")], "p_hat": [0.5]}),
                    sub.iloc[:0], sub.assign(p_hat=[np.nan, 0.5])):
            try:
                check_reproduce("b1", bad, ref_dir=tmp)
            except ValueError:
                continue
            raise AssertionError("check_reproduce phải báo lỗi")

    _tests_shap(data, cols, folds, wf, fe_cfg)
    print("Tất cả test shap_analysis đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=SHAP_MODELS, choices=SHAP_MODELS)
    ap.add_argument("--every", type=int, default=1, help="chỉ fold 0, k, 2k, ... (chạy thử nhanh)")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args(argv)
    if args.test:
        _tests()
        return 0
    ctx = setup()
    res = run(args.models, args.every, ctx=ctx)
    for m, why in res["dropped"].items():
        print(f"BỎ {m} khỏi SHAP: {why}")
    if not res["values"]:
        print("Không còn mô hình nào để tổng hợp.")
        return 1
    groups = {k: v for k, v in features.feature_groups(**ctx[4]).items() if k != "Flows"}
    groups = {"OHLCV": list(FEATURES_10) + [SIGN], **groups}
    feats, grp, yr = summarize(res["values"], groups)
    write_csv(feats, OUT / "shap_features.csv")
    write_csv(grp, OUT / "shap_groups.csv")
    write_csv(yr, OUT / "shap_groups_by_year.csv")
    ab = RESULTS / "ablation" / "ablation_rank.csv"
    if ab.exists():
        write_csv(compare_ablation(grp, pd.read_csv(ab)), OUT / "shap_vs_ablation.csv")
    else:
        print(f"CẢNH BÁO: chưa có {ab}, bỏ qua so sánh SHAP vs ablation.")
    make_figures(grp, feats)
    print(grp.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
