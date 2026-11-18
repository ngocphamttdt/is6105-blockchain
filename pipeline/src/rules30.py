"""
rules30.py  –  Vi: luật 30 ngày (rule-based) và bảng accuracy / F1 cho cả chuỗi bước thầy đề ra:
  luật đơn giản  →  mô hình B1–B4 chỉ OHLCV  →  thêm On-chain có chọn lọc.

Luật (chỉ dùng giá đóng cửa đến hết ngày t; dự báo hướng ngày t+1):
  b0maj   đoán theo lớp nhiều hơn trong cửa sổ train (730 ngày)
  pers1   mai cùng chiều hôm nay;  rev1  luật đảo chiều b0rev (chọn chiều trong train)
  mom30   động lượng 30 ngày: close_t > close_{t−30}
  ma30    giá trên trung bình 30 ngày: close_t > MA30_t
  maj30   đa số 30 ngày qua là ngày tăng: #ngày tăng trong 30 ngày ≥ 15
Mỗi luật 30 ngày có hai bản:
  {luật}_follow  luôn đi theo tín hiệu (không học gì)
  {luật}         chọn "theo" hay "ngược" tín hiệu trong từng cửa sổ train (accuracy train < 50% thì đảo), như b0rev.
  Danh sách luật chốt trước khi chạy; không chọn luật theo test.

Metric: accuracy, F1 lớp "tăng", F1 macro (trung bình F1 hai lớp), balanced accuracy, % ngày đoán tăng,
Δ accuracy so với b0rev (CI bootstrap khối, McNemar). Ngưỡng p̂ > 0,5 như mọi bảng khác.

  python pipeline/src/rules30.py            → results/rules30/{predictions/,tables/}, figures/fig_rules30.png
  python pipeline/src/rules30.py --test
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

import stats
from data import CLOSE, DATE, PREDICTION_COLS, RESULTS, TARGET, load_master, read_csv_dated, write_csv
from experiments import E2_DIR, setup
from walkforward import fold_indices

OUT = RESULTS / "rules30"
GATE_DIR = RESULTS / "gate"
WINDOW = 30
NAMES = {"b0maj": "Đoán lớp nhiều hơn (train)", "pers1": "Mai cùng chiều hôm nay", "rev1": "Đảo chiều 1 ngày (b0rev)",
         "mom30_follow": "Động lượng 30 ngày (theo)", "mom30": "Động lượng 30 ngày (chọn chiều trong train)",
         "ma30_follow": "Giá > MA30 (theo)", "ma30": "Giá > MA30 (chọn chiều trong train)",
         "maj30_follow": "Đa số 30 ngày tăng (theo)", "maj30": "Đa số 30 ngày tăng (chọn chiều trong train)"}


def signals(close: pd.Series, window=WINDOW) -> pd.DataFrame:
    """Tín hiệu 0/1 tại ngày t, chỉ dùng close đến hết ngày t. NaN khi chưa đủ dữ liệu."""
    up = (close > close.shift(1)).astype(float).where(close.shift(1).notna())
    out = pd.DataFrame(index=close.index)
    out["pers1"] = up
    out["mom30"] = (close > close.shift(window)).astype(float).where(close.shift(window).notna())
    ma = close.rolling(window).mean()
    out["ma30"] = (close > ma).astype(float).where(ma.notna())
    n_up = up.rolling(window).sum()
    out["maj30"] = (n_up >= window / 2).astype(float).where(n_up.notna())
    return out


def rule_predictions(data, folds, sig, choose):
    """p̂ = tín hiệu (hoặc 1 − tín hiệu nếu choose và tín hiệu đúng < 50% trong cửa sổ train). Trả (pred, số fold đảo)."""
    out, flips = [], 0
    for f in folds:
        tr, te = fold_indices(data, f)
        s_tr = sig.loc[tr]
        ok = s_tr.notna()
        flip = choose and (s_tr[ok] == data.loc[tr[ok.to_numpy()], TARGET]).mean() < 0.5
        flips += int(flip)
        p = sig.loc[te].to_numpy(float)
        if np.isnan(p).any():
            raise ValueError(f"Fold {f.fold_id}: tín hiệu thiếu ở ngày test.")
        out.append(pd.DataFrame({"date": data.loc[te, DATE].to_numpy(), "fold": f.fold_id,
                                 "y_true": data.loc[te, TARGET].astype(int).to_numpy(),
                                 "p_hat": 1 - p if flip else p, "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS], flips


def majority_predictions(data, folds):
    out = []
    for f in folds:
        tr, te = fold_indices(data, f)
        rate = data.loc[tr, TARGET].mean()
        out.append(pd.DataFrame({"date": data.loc[te, DATE].to_numpy(), "fold": f.fold_id,
                                 "y_true": data.loc[te, TARGET].astype(int).to_numpy(),
                                 "p_hat": np.full(len(te), rate), "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def build_rules():
    data, _, folds, _, _ = setup("features_onchain_v2.yaml")
    m = load_master()[[DATE, CLOSE]]
    close = data[[DATE]].merge(m, on=DATE, how="left")[CLOSE]
    close.index = data.index
    # Giá Binance bắt đầu giữa 2017 (On-chain từ 2016); từ ngày train đầu của fold 0 trở đi phải đủ giá
    first_train = folds[0].train_start - pd.Timedelta(days=WINDOW)
    assert close[data[DATE] >= first_train].notna().all()
    assert (data[DATE].diff().iloc[1:] == pd.Timedelta(days=1)).all()
    sig = signals(close)
    preds, flips = {"b0maj": majority_predictions(data, folds)}, {}
    preds["pers1"], _ = rule_predictions(data, folds, sig["pers1"], choose=False)
    preds["rev1"], flips["rev1"] = rule_predictions(data, folds, sig["pers1"], choose=True)
    for k in ("mom30", "ma30", "maj30"):
        preds[f"{k}_follow"], _ = rule_predictions(data, folds, sig[k], choose=False)
        preds[k], flips[k] = rule_predictions(data, folds, sig[k], choose=True)
    for k, p in preds.items():
        write_csv(p, OUT / "predictions" / f"{k}.csv")
    return preds, flips, len(folds)


def score(p):
    y, yhat = p.y_true.to_numpy(), (p.p_hat.to_numpy() > 0.5).astype(int)
    return {"n_days": len(p), "accuracy": (y == yhat).mean(), "f1_up": f1_score(y, yhat, zero_division=0),
            "f1_down": f1_score(1 - y, 1 - yhat, zero_division=0),
            "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, yhat), "pct_pred_up": yhat.mean()}


def tables(preds, flips, n_folds):
    rev = preds["rev1"]
    steps = [("1. Luật", k, NAMES[k], p) for k, p in preds.items()]
    ml = {"b1": "B1 LR", "b2": "B2 RF", "b3": "B3 SVM", "b4": "B4 XGB"}
    for k, lab in ml.items():
        f = GATE_DIR / "baseline" / f"{k}.csv"
        if f.exists():
            steps.append(("2. ML chỉ OHLCV", k, lab, read_csv_dated(f)))
    for k, lab in ml.items():
        f = GATE_DIR / "stepwise" / f"{k}.csv"
        if f.exists():
            steps.append(("3. + On-chain chọn lọc (stepwise)", k, lab, read_csv_dated(f)))
    for k, lab in ml.items():
        f = E2_DIR / "predictions" / f"{k}_onchain.csv"
        if f.exists():
            steps.append(("3'. + cả 20 On-chain (đối chứng)", k, lab, read_csv_dated(f)))
    rows = []
    for step, key, lab, p in steps:
        assert np.array_equal(p[DATE].to_numpy("datetime64[ns]"), rev[DATE].to_numpy("datetime64[ns]")), key
        r = {"step": step, "method": key, "label": lab, **score(p),
             "n_folds_flipped": flips.get(key, np.nan) if step.startswith("1") else np.nan}
        if key != "rev1":
            d = stats.paired_row(p.y_true.to_numpy(), rev.p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=False)
            r |= {"d_acc_vs_rev1": d["d_da"], "d_acc_lo": d["d_da_lo"], "d_acc_hi": d["d_da_hi"],
                  "p_mcnemar": d["p_mcnemar"]}
        rows.append(r)
    t = pd.DataFrame(rows)
    t["p_mcnemar_holm"] = np.nan
    has = t.p_mcnemar.notna()
    t.loc[has, "p_mcnemar_holm"] = stats.holm(t.loc[has, "p_mcnemar"])
    t["n_folds"] = n_folds
    yr = []
    for k, p in preds.items():
        for y_, q in p.groupby(p[DATE].dt.year):
            yr.append({"method": k, "year": int(y_), "n_days": len(q), **score(q)})
    yr = pd.DataFrame(yr)
    write_csv(t, OUT / "tables" / "rules30_summary.csv")
    write_csv(yr, OUT / "tables" / "rules30_by_year.csv")
    return t, yr


def figure(t, yr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    ink, rule_c, muted, grid_c, surf = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1", "#fcfcfb"
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=surf, gridspec_kw={"width_ratios": [1.25, 1]})
    s = t.iloc[::-1]
    y = np.arange(len(s))
    ax = axs[0]
    ax.set_facecolor(surf)
    ax.axvline(50, color=muted, ls="--", lw=1)
    ax.plot(s.accuracy * 100, y + 0.17, "o", color=ink, ms=7, mec=surf, mew=1.5, label="Accuracy")
    ax.plot(s.f1_macro * 100, y - 0.17, "D", color=rule_c, ms=6, mec=surf, mew=1.5, label="F1 macro")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a.split('.')[0]}. {b}" for a, b in zip(s.step, s.label)], fontsize=8)
    ax.set_xlabel("% trên 2.434 ngày test (01/2020 → 08/2026)")
    ax.set_title("Accuracy và F1 macro theo từng bước", fontsize=9)
    ax.grid(axis="x", color=grid_c)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax = axs[1]
    ax.set_facecolor(surf)
    ax.axhline(50, color=muted, ls="--", lw=1)
    show = {"rev1": (rule_c, "Đảo chiều 1 ngày"), "mom30": (ink, "Động lượng 30 ngày"),
            "ma30": ("#7a8c3a", "Giá > MA30"), "maj30": ("#8a5cb8", "Đa số 30 ngày tăng")}
    for k, (c, lab) in show.items():
        q = yr[yr.method == k]
        ax.plot(q.year, q.accuracy * 100, "-o", color=c, lw=2, ms=5, mec=surf, label=lab)
    ax.set_xticks(sorted(yr.year.unique()))
    ax.set_xticklabels([f"'{v % 100:02d}" for v in sorted(yr.year.unique())])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Luật (chọn chiều trong train) theo năm", fontsize=9)
    ax.grid(axis="y", color=grid_c)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    out = RESULTS / "figures" / "fig_rules30.png"
    fig.savefig(out, dpi=200, facecolor=surf)
    plt.close(fig)
    return out


def _tests():
    # 1) Tín hiệu chỉ dùng quá khứ: đổi giá sau ngày t không đổi tín hiệu tại t
    rng = np.random.default_rng(0)
    close = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.03, 200))) * 100)
    s1 = signals(close)
    c2 = close.copy()
    c2.iloc[120:] *= rng.uniform(0.5, 2, 80)
    s2 = signals(c2)
    pd.testing.assert_frame_equal(s1.iloc[:120], s2.iloc[:120])
    # 2) Đúng định nghĩa trên ví dụ nhỏ
    c = pd.Series(np.arange(1, 41, dtype=float))          # giá tăng đều
    s = signals(c)
    assert s.pers1.iloc[1:].eq(1).all() and s.mom30.iloc[30:].eq(1).all() and s.ma30.iloc[29:].eq(1).all()
    assert s.maj30.iloc[30:].eq(1).all() and s.mom30.iloc[:30].isna().all() and s.maj30.iloc[:30].isna().all()
    # 3) F1 tính tay: y = [1,1,0,0], ŷ = [1,0,1,0] → F1 tăng = 0,5; F1 macro = 0,5
    r = score(pd.DataFrame({"y_true": [1, 1, 0, 0], "p_hat": [0.9, 0.1, 0.8, 0.2]}))
    assert r["accuracy"] == 0.5 and abs(r["f1_up"] - 0.5) < 1e-12 and abs(r["f1_macro"] - 0.5) < 1e-12
    # Luôn đoán tăng: F1 tăng = 2·0,5/(1+0,5) = 0,667; F1 giảm = 0 → F1 macro 0,333
    r = score(pd.DataFrame({"y_true": [1, 1, 0, 0], "p_hat": [0.6] * 4}))
    assert abs(r["f1_up"] - 2 / 3) < 1e-12 and r["f1_down"] == 0 and abs(r["f1_macro"] - 1 / 3) < 1e-12
    print("Tất cả test rules30 đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args(argv)
    _tests()
    if args.test:
        return 0
    preds, flips, n_folds = build_rules()
    t, yr = tables(preds, flips, n_folds)
    pd.set_option("display.width", 250)
    print(t.drop(columns=["label"]).round(4).to_string(index=False))
    print(yr.pivot(index="method", columns="year", values="accuracy").round(3).to_string())
    print(figure(t, yr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
