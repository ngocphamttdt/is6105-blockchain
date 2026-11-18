"""
simple_split.py  –  Chạy 5 nhóm baseline (B0, LR, SVM RBF, RF, LSTM) trên CÙNG một cách chia
train 2018–2021 / validation 2022–2023 / test 2024–08/2026 (ý 1 của thầy), cùng 10 đặc trưng OHLCV.
Mỗi mô hình: tune trên validation theo log-loss, học lại trên train + validation, chấm test đúng 1 lần,
xem trực tiếp dự báo đúng/sai. Grid lấy đúng của người phụ trách (Vi: LR, Trâm: SVM, Ngân: RF, Ngọc: LSTM).
Thiết kế: docs/superpowers/specs/2026-11-13-simple-split-design.md

Chạy từ thư mục gốc repo:
    python analyses/simple_split/src/simple_split.py --test     # test
    python analyses/simple_split/src/simple_split.py --quick    # chạy thử: 2 tổ hợp / mô hình, 1 seed, không ghi báo cáo
    python analyses/simple_split/src/simple_split.py            # chạy đủ, ghi analyses/simple_split/reports/
"""
import sys
from pathlib import Path

import torch  # noqa: F401  nạp torch TRƯỚC sklearn/scipy: ngược lại LSTM bị segfault (xung đột OpenMP trên macOS)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "analyses/baseline_check/src"), str(ROOT / "features/src"),
                str(ROOT / "pipeline/src"), str(ROOT / "analyses/lstm_tuning/src")]
import baseline_check as bc                                   # noqa: E402
from features_ohlcv import FEATURES_10                        # noqa: E402

SIGN = "up_lag1"
SEEDS = [42, 43, 44]
MODELS = {}


def load():
    """df (data freeze v1.0, 10 đặc trưng OHLCV, nhãn y, thêm cột dấu up_lag1) và parts train / val / test."""
    df = bc.load_data()
    df[SIGN] = (df["logret_lag1"] > 0).astype(float).where(df["logret_lag1"].notna())
    return df, bc.split(df)


def _ll(y, p):
    return bc.log_loss(np.asarray(y, int), np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


def tune(name, fit_fn, grid, parts):
    """Mỗi tổ hợp: học trên train, chấm train và val. Chọn min val log-loss (làm tròn 4 chữ số); hoà thì tổ hợp
    đứng trước trong grid (grid xếp từ đơn giản đến phức tạp). Không đọc parts["test"]."""
    tr, va = parts["train"], parts["val"]
    rows = []
    for k, prm in enumerate(grid):
        p_tr, p_va = fit_fn(prm, tr, tr), fit_fn(prm, tr, va)
        rows.append({**prm, "train_log_loss": _ll(tr["y"], p_tr), "val_log_loss": _ll(va["y"], p_va),
                     "train_DA": ((p_tr > .5) == tr["y"]).mean(), "val_DA": ((p_va > .5) == va["y"]).mean(),
                     "_order": k})
        print(f"  {name} {k + 1}/{len(grid)} val_log_loss={rows[-1]['val_log_loss']:.4f}", flush=True)
    t = pd.DataFrame(rows)
    k_best = int(t.assign(_r=t["val_log_loss"].round(4)).sort_values(["_r", "_order"]).iloc[0]["_order"])
    t["chosen"] = t["_order"] == k_best
    return t.drop(columns="_order"), grid[k_best]


def final(fit_fn, params, parts):
    """Học lại trên train + val với tham số đã chọn, trả p̂ cho test (lần duy nhất test được dùng)."""
    return fit_fn(params, pd.concat([parts["train"], parts["val"]]), parts["test"])


def rules(parts):
    """4 luật B0 trên test; tỷ lệ tăng và chiều luật đảo chiều lấy từ train + val (như baseline_check)."""
    r, _ = bc.rule_predictions(pd.concat([parts["train"], parts["val"]]), parts["test"])
    names = {"B0 Majority (p̂ = tỷ lệ tăng của train)": "Majority", "B0 Luôn đoán tăng": "Buy-and-hold",
             "B0 Persistence (mai cùng chiều hôm nay)": "Persistence", "B0 Đảo chiều (mai ngược chiều hôm nay)": "Đảo chiều"}
    return {names[k]: np.asarray(v, float) for k, v in r.items()}


# ---------------------------------------------------------------------------
# Mô hình: mỗi mô hình có "before" (đặt trước), "grid" (của người phụ trách) và
# fit(params, learn_df, target_df) -> p̂ cho target_df. Chuẩn hoá fit lại trong mỗi lần học.
# ---------------------------------------------------------------------------
def _sk_fit(make, cols, learn, target, seeds=(None,), _cols_seen=None):
    """Học pipeline sklearn trên learn[cols], trả p̂ trên target; có seed thì trung bình p̂ các seed."""
    if _cols_seen is not None:
        _cols_seen.append(list(cols))
    ps = [make(s).fit(learn[cols].to_numpy(), learn["y"].astype(int).to_numpy())
          .predict_proba(target[cols].to_numpy())[:, 1] for s in seeds]
    return np.mean(ps, axis=0)


def _build_models():
    import itertools
    import warnings
    import yaml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    import b1t_tuning                                     # B1 tune đầy đủ (b1t)
    from models import CalibratedSVC
    from walkforward import make_pipeline
    warnings.simplefilter("ignore", ConvergenceWarning)

    # LR (Vi): 5 tập đặc trưng × b1t_tuning.GRID, winsorize CLIP_Q; trước tune = C 1, L2, 10 đặc trưng, không winsorize
    sets = b1t_tuning.feature_sets(FEATURES_10)

    def lr_fit(prm, learn, target, seeds=None, _cols_seen=None):
        cw = None if prm["class_weight"] in (None, "none") else prm["class_weight"]
        est = b1t_tuning.ElasticNetLR(C=prm["C"], l1_ratio=prm["l1_ratio"], class_weight=cw)
        return _sk_fit(lambda s: make_pipeline(est, b1t_tuning.CLIP_Q if prm["clip"] else None), sets[prm["set"]],
                       learn, target, _cols_seen=_cols_seen)

    lr_grid = [{"set": st, "C": c, "l1_ratio": l1, "class_weight": cw, "clip": True}
               for st in sets for c in b1t_tuning.GRID["C"] for l1 in b1t_tuning.GRID["l1_ratio"]
               for cw in b1t_tuning.GRID["class_weight"]]
    MODELS["LR"] = {"before": {"set": "ohlcv10", "C": 1.0, "l1_ratio": 0.0, "class_weight": None, "clip": False},
                    "grid": lr_grid, "fit": lr_fit}

    # SVM RBF (Trâm): GRID_WIDE 5 × 6; trước tune = sklearn mặc định (C 1, gamma "scale")
    def svm_fit(prm, learn, target, seeds=None, _cols_seen=None):
        return _sk_fit(lambda s: make_pipeline(CalibratedSVC(C=prm["C"], gamma=prm["gamma"])), FEATURES_10,
                       learn, target, _cols_seen=_cols_seen)

    svm_grid = [{"C": c, "gamma": g} for c in [0.01, 0.1, 1, 10, 100]
                for g in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]]
    MODELS["SVM RBF"] = {"before": {"C": 1.0, "gamma": "scale"}, "grid": svm_grid, "fit": svm_fit}

    # RF (Ngân): param_grid của b2t.yaml, 500 cây, max_depth None; trước tune = sklearn mặc định + 500 cây.
    # Có ngẫu nhiên: p̂ trung bình SEEDS.
    b2t = yaml.safe_load((ROOT / "pipeline/configs/models/b2t.yaml").read_text())["param_grid"]

    def rf_fit(prm, learn, target, seeds=None, _cols_seen=None):
        return _sk_fit(lambda s: make_pipeline(RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1,
                                                                      random_state=s, **prm)),
                       FEATURES_10, learn, target, seeds=seeds or SEEDS, _cols_seen=_cols_seen)

    keys = list(b2t)
    rf_grid = [dict(zip(keys, v)) for v in itertools.product(*b2t.values())]
    MODELS["RF"] = {"before": {}, "grid": rf_grid, "fit": rf_fit}

    # LSTM (Ngọc): lstm_tuning.GRID 16 cấu hình; trước tune = lstm_tuning.DEFAULT. Early stopping trong 20% cuối
    # phần đang học (có gap), học lại trên 100% với số epoch tốt nhất. Có ngẫu nhiên: p̂ trung bình SEEDS.
    import lstm_tuning
    lstm_tuning.COLS = list(FEATURES_10)
    feats = lstm_tuning.load_full()
    feats["_y"] = bc.load_data()["y"].reindex(feats.index)

    def lstm_fit(prm, learn, target, seeds=None, _cols_seen=None):
        if _cols_seen is not None:
            _cols_seen.append(list(lstm_tuning.COLS))
        return np.mean([lstm_tuning.fit_predict(feats, learn.index, {"t": target.index}, prm, s)[0]["t"]
                        for s in (seeds or SEEDS)], axis=0)

    g = lstm_tuning.GRID
    lstm_grid = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    MODELS["LSTM"] = {"before": dict(lstm_tuning.DEFAULT), "grid": lstm_grid, "fit": lstm_fit}


_build_models()


# ---------------------------------------------------------------------------
# Xem dự báo, bảng so sánh, báo cáo
# ---------------------------------------------------------------------------
def inspect(name, p, test, rev):
    """Các mục xem trực tiếp dự báo của một mô hình trên test (markdown)."""
    d = test[["ohlcv_close", "logret_lag1", "rsi_14", "y", "ret_next"]].copy()
    d["p_hat"] = p
    d["dự báo"] = (p > .5).astype(int)
    d["đúng"] = d["dự báo"] == d["y"]
    d["return ngày mai (%)"] = d["ret_next"] * 100
    cols = ["ohlcv_close", "logret_lag1", "rsi_14", "p_hat", "dự báo", "y", "đúng", "return ngày mai (%)"]
    show = lambda x: x[cols].round(3).to_markdown()                                     # noqa: E731
    cm = pd.crosstab(d["y"].map({0: "thật: giảm", 1: "thật: tăng"}),
                     d["dự báo"].map({0: "đoán: giảm", 1: "đoán: tăng"}))
    wrong = d[~d["đúng"]].assign(conf=lambda x: (x["p_hat"] - .5).abs()).nlargest(8, "conf")
    big = d.loc[d["ret_next"].abs().nlargest(10).index]
    q = pd.qcut(d["ret_next"].abs(), 4, labels=["25% ngày yên nhất", "Q2", "Q3", "25% ngày mạnh nhất"])
    by_vol = pd.DataFrame({"số ngày": d.groupby(q, observed=True).size(),
                           "mô hình đúng (%)": d.groupby(q, observed=True)["đúng"].mean() * 100,
                           "đảo chiều đúng (%)": ((rev > .5) == d["y"]).groupby(q, observed=True).mean() * 100}).round(1)
    agree = ((p > .5) == (rev > .5)).mean() * 100
    return [f"### {name}", "",
            f"p̂: trung vị {np.median(p):.3f}, 90% số ngày trong [{np.quantile(p, .05):.3f}; {np.quantile(p, .95):.3f}]; "
            f"đoán tăng {(p > .5).mean() * 100:.1f}% số ngày; cùng hướng với luật đảo chiều {agree:.1f}% số ngày.", "",
            "**Ma trận nhầm lẫn**", "", cm.to_markdown(), "",
            "**10 ngày ngẫu nhiên**", "", show(d.sample(10, random_state=0).sort_index()), "",
            "**8 ngày sai mà mô hình tự tin nhất**", "", show(wrong), "",
            "**10 ngày biến động mạnh nhất** (theo return ngày mai): mô hình đúng "
            f"{int(big['đúng'].sum())}/10", "", show(big), "",
            "**Độ chính xác theo mức biến động của ngày mai**", "", by_vol.to_markdown(), ""]


def compare(preds, test, rev):
    """Bảng metric trên test cho mọi dự báo + McNemar (chính xác) so với luật đảo chiều."""
    from scipy.stats import binomtest
    y = test["y"].to_numpy(int)
    t = bc.fmt_metrics({k: bc.metrics(y, v) for k, v in preds.items()})
    r_ok = (rev > .5) == y
    mc = []
    for v in preds.values():
        m_ok = (np.asarray(v) > .5) == y
        b, c = int((m_ok & ~r_ok).sum()), int((~m_ok & r_ok).sum())
        mc.append(np.nan if b + c == 0 else binomtest(b, b + c, .5).pvalue)
    t["McNemar p (so với đảo chiều)"] = np.round(mc, 3)
    return t


def run(quick=False):
    """Chạy đủ 5 nhóm mô hình trên cách chia chung, ghi báo cáo vào analyses/simple_split/reports/."""
    global SEEDS
    import time
    if quick:
        SEEDS = [42]
    df, parts = load()
    te = parts["test"]
    rb = rules(parts)
    rev = rb["Đảo chiều"]
    preds, tunings, chosen = dict(rb), {}, {}
    t0 = time.time()
    for name, m in MODELS.items():
        grid = [m["grid"][0], m["grid"][-1]] if quick else m["grid"]
        print(f"{name}: {len(grid)} tổ hợp × {len(SEEDS) if name in ('RF', 'LSTM') else 1} seed", flush=True)
        preds[f"{name} trước tune"] = final(m["fit"], m["before"], parts)
        tunings[name], chosen[name] = tune(name, m["fit"], grid, parts)
        preds[f"{name} sau tune"] = final(m["fit"], chosen[name], parts)
        print(f"  chọn {chosen[name]} ({(time.time() - t0) / 60:.1f} phút)", flush=True)
    out = pd.DataFrame({"date": te.index, "y": te["y"].to_numpy(int), **{k: np.asarray(v) for k, v in preds.items()}})
    assert len(out) == 973 and not out.isna().any().any(), "dự báo thiếu ngày hoặc có NaN"
    cmp = compare(preds, te, rev)
    print(cmp.to_string())
    if quick:
        return
    report(cmp, tunings, chosen, preds, te, rev, parts, out)


def report(cmp, tunings, chosen, preds, te, rev, parts, out):
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "predictions_test.csv", index=False)
    cmp.to_csv(OUT / "test_metrics.csv")
    md = ["# So sánh baseline trên cùng một cách chia (ý 1 của thầy)", "",
          "Tạo bởi `analyses/simple_split/src/simple_split.py`. Thiết kế: `docs/superpowers/specs/2026-11-13-simple-split-design.md`.", "",
          "## Thiết lập chung", "",
          "| Hạng mục | Giá trị |", "|---|---|",
          "| Dữ liệu | data freeze v1.0, 10 đặc trưng OHLCV (`FEATURES_10`) |",
          *[f"| {k} | {v.index.min().date()} → {v.index.max().date()} ({len(v)} ngày) |" for k, v in parts.items()],
          "| Chọn tham số | log-loss trên validation; hoà (4 chữ số) thì tổ hợp đơn giản hơn |",
          "| Mô hình cuối | học lại trên train + validation, chấm test đúng 1 lần |",
          f"| Seed | RF và LSTM: p̂ trung bình {SEEDS}; B0, LR, SVM tất định |", "",
          "Lưu ý độ tin cậy: 973 ngày test cho sai số DA khoảng ±1,6 điểm (CI 95% rộng khoảng ±3 điểm). "
          "Chênh lệch nhỏ hơn mức này giữa hai mô hình chưa đủ để nói mô hình nào tốt hơn.", "",
          "## 1. Bảng so sánh trên test 2024 – 08/2026", "", cmp.to_markdown(), "",
          "Log-loss và Brier của luật B0 (p̂ = 0,99/0,01) không có nghĩa, chỉ so DA với chúng. "
          "McNemar p: so số ngày chỉ mô hình đúng và chỉ luật đảo chiều đúng.", "",
          "## 2. Tuning từng mô hình (train → validation, chưa đụng test)", ""]
    for name, t in tunings.items():
        t.to_csv(OUT / f"tuning_{name.replace(' ', '_')}.csv", index=False)
        s = t.sort_values("val_log_loss").copy()
        for c in ["train_log_loss", "val_log_loss"]:
            s[c] = s[c].round(4)
        for c in ["train_DA", "val_DA"]:
            s[c] = (s[c] * 100).round(1)
        md += [f"### {name}: {len(t)} tổ hợp; chọn `{chosen[name]}`", "",
               (s.head(15).to_markdown(index=False) + (f"\n\n(15/{len(t)} tổ hợp tốt nhất; đủ ở "
                f"`tuning_{name.replace(' ', '_')}.csv`)" if len(t) > 15 else "")), ""]
    md += ["## 3. Xem trực tiếp dự báo (bản sau tune, test 2024 – 08/2026)", ""]
    for name in tunings:
        md += inspect(f"{name} sau tune", np.asarray(preds[f"{name} sau tune"]), te, rev)
    (OUT / "simple_split.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Đã lưu {OUT / 'simple_split.md'}")


OUT = ROOT / "analyses" / "simple_split" / "reports"


def _tests():
    df, parts = load()
    assert (df[SIGN].dropna() == (df.loc[df[SIGN].notna(), "logret_lag1"] > 0).astype(float)).all()
    assert len(parts["test"]) == 973
    fake = lambda prm, learn, target: np.full(len(target), prm["p"])       # noqa: E731
    t, best = tune("fake", fake, [{"p": 0.5}, {"p": 0.4}], parts)
    want = min([0.5, 0.4], key=lambda p: bc.log_loss(parts["val"]["y"], np.full(len(parts["val"]), p)))
    assert best == {"p": want} and t["chosen"].sum() == 1
    blind = {k: v.copy() for k, v in parts.items()}
    blind["test"]["y"] = np.nan
    assert tune("fake", fake, [{"p": 0.5}, {"p": 0.4}], blind)[1] == best      # tune không đọc test
    r = rules(parts)
    assert round(((r["Đảo chiều"] > .5) == parts["test"]["y"]).mean(), 4) == 0.5098
    assert round(((r["Majority"] > .5) == parts["test"]["y"]).mean(), 4) == 0.5067
    # Task 2: LR / SVM / RF
    assert {len(MODELS[m]["grid"]) for m in ("LR", "SVM RBF", "RF")} == {300, 30, 24}
    assert [len(MODELS[m]["grid"]) for m in ("LR", "SVM RBF", "RF")] == [300, 30, 24]
    seen = []
    small = {k: v.iloc[:300] for k, v in parts.items()}
    for m, prm in [("LR", {**MODELS["LR"]["grid"][0]}), ("SVM RBF", MODELS["SVM RBF"]["before"]),
                   ("RF", MODELS["RF"]["before"])]:
        MODELS[m]["fit"](prm, small["train"], small["val"], _cols_seen=seen)
    assert seen[0] == [SIGN] and seen[1] == list(FEATURES_10) and seen[2] == list(FEATURES_10)
    a = MODELS["RF"]["fit"](MODELS["RF"]["before"], small["train"], small["val"], seeds=[42])
    b = MODELS["RF"]["fit"](MODELS["RF"]["before"], small["train"], small["val"], seeds=[42])
    assert np.array_equal(a, b)
    # Task 3: LSTM
    import lstm_tuning
    assert lstm_tuning.COLS == list(FEATURES_10)
    q = MODELS["LSTM"]["fit"]({"lookback": 10, "hidden": 8, "dropout": 0.0, "lr": 1e-3},
                              parts["train"], parts["val"], seeds=[42])
    assert len(q) == len(parts["val"]) and np.isfinite(q).all()
    # Task 4: xem dự báo, bảng so sánh
    te = parts["test"]
    rev = rules(parts)["Đảo chiều"]
    pz = np.random.default_rng(0).uniform(0.4, 0.6, len(te))
    md = "\n".join(inspect("X", pz, te, rev))
    for head in ["Ma trận nhầm lẫn", "10 ngày ngẫu nhiên", "8 ngày sai mà mô hình tự tin nhất",
                 "10 ngày biến động mạnh nhất", "theo mức biến động", "cùng hướng với luật đảo chiều"]:
        assert head in md, head
    assert md == "\n".join(inspect("X", pz, te, rev))                       # ngày ngẫu nhiên cố định
    cmp = compare({"Đảo chiều": rev, "X": pz}, te, rev)
    assert "McNemar p (so với đảo chiều)" in cmp.columns and len(cmp) == 2
    print("Tất cả test simple_split đều PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.test:
        _tests()
    else:
        run(quick=args.quick)
