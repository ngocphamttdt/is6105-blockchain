"""
hybrid.py  –  Vi: áp dụng luật (rule-based) VÀO mô hình, ba cách. Thiết kế và ngưỡng chốt trước khi chạy.

  A  Luật làm đặc trưng: thêm up_lag1 = 1[logret_lag1 > 0] (dấu return hôm nay, thứ luật đảo chiều dùng) vào 10
     đặc trưng OHLCV; chạy B1–B4 như gate.py (lưới mở rộng, winsorize, tune mỗi 6 fold). Thêm "lr_sign": LR chỉ
     dùng up_lag1, để xem mô hình có tự học lại được luật không.
  C  Luật mặc định, mô hình ghi đè: p̂ = mô hình nếu |p̂ − 0,5| > MARGIN (0,05), ngược lại p̂ = luật đảo chiều.
     Áp cho dự báo của gate (chỉ OHLCV), stepwise (+ On-chain chọn lọc) và A.
  B  Luật On-chain cực trị: với mỗi z90 (7 cột On-chain v2), ngày có |z| > Z (2) là ngày "kích hoạt". Chiều dự báo
     (z dương → tăng, hay z dương → giảm) chọn theo accuracy trên các ngày kích hoạt trong cửa sổ train của fold.
     Ngày không kích hoạt dùng luật đảo chiều. So trên đúng các ngày kích hoạt của test: luật On-chain vs đảo chiều.

So với luật đảo chiều rev1 (results/rules30/predictions/rev1.csv): accuracy, F1 macro, Δ (CI bootstrap khối),
McNemar, Holm trong từng cách.

  python pipeline/src/hybrid.py --stage a [--models lr b2]   (A: chạy mô hình, ~10 phút)
  python pipeline/src/hybrid.py --stage tables                (B, C và bảng tổng)
"""
import argparse
import sys

import numpy as np
import pandas as pd

import stats
from data import DATE, PREDICTION_COLS, RESULTS, TARGET, read_csv_dated, write_csv
from experiments import ML, setup
from gate import GATE_DIR, GRID_WIDE, run
from rules30 import OUT as R30, score
from run_all import attach_regime, load_regime_map
from walkforward import fold_indices

OUT = RESULTS / "hybrid"
MARGIN, Z = 0.05, 2.0
Z_COLS = ["cm_NetflowExNtv_z90", "cm_AdrActCnt_z90", "cm_TxCnt_z90", "bc_est_tx_vol_usd_z90",
          "cm_HashRate_z90", "cm_FeeTotNtv_z90", "puell_multiple_z90"]
SIGN = "up_lag1"
GRID_WIDE.setdefault("lr_sign", GRID_WIDE["lr"])


def with_sign(data):
    data = data.copy()
    data[SIGN] = (data["logret_lag1"] > 0).astype(float).where(data["logret_lag1"].notna())
    return data


# ----------------------------------------------------------------------------
# A
# ----------------------------------------------------------------------------
def stage_a(models):
    data, sets, folds, wf, _ = setup("features_onchain_v2.yaml")
    data = with_sign(data)
    rmap = load_regime_map()
    for name in models:
        cols, est = (sets["ohlcv"] + [SIGN], name) if name != "lr_sign" else ([SIGN], "lr")
        GRID_WIDE.setdefault(est, GRID_WIDE["lr"])
        p, log = run(data, folds, est, wf, cols)
        write_csv(attach_regime(p, rmap), OUT / "a" / f"{name}.csv")
        write_csv(log, OUT / "log" / f"a_{name}.csv")
        print(f"A {name}: {len(cols)} đặc trưng, accuracy {score(p)['accuracy']:.4f}", flush=True)


# ----------------------------------------------------------------------------
# C
# ----------------------------------------------------------------------------
def override(model, rule, margin=MARGIN):
    """p̂ = mô hình khi đủ tự tin, ngược lại = luật. Trả (dự báo, mặt nạ ngày mô hình ghi đè)."""
    assert np.array_equal(model[DATE].to_numpy("datetime64[ns]"), rule[DATE].to_numpy("datetime64[ns]"))
    use = (model.p_hat - 0.5).abs().to_numpy() > margin
    out = rule.copy()
    out["p_hat"] = np.where(use, model.p_hat.to_numpy(), rule.p_hat.to_numpy())
    return out, use


# ----------------------------------------------------------------------------
# B
# ----------------------------------------------------------------------------
def extreme_rule(data, folds, col, rev, z=Z):
    """Luật cực trị cho một cột z. Trả (dự báo cả test, mặt nạ ngày kích hoạt trên test, số fold chọn 'z dương → tăng')."""
    out, n_pos = [], 0
    rev_p = rev.set_index(DATE).p_hat
    for f in folds:
        tr, te = fold_indices(data, f)
        ztr, ytr = data.loc[tr, col], data.loc[tr, TARGET]
        hit = ztr.abs() > z
        up_if_pos = ((ztr[hit] > 0).astype(int) == ytr[hit]).mean() >= 0.5 if hit.any() else True
        n_pos += int(up_if_pos)
        zte = data.loc[te, col].to_numpy()
        if np.isnan(zte).any():
            raise ValueError(f"{col}, fold {f.fold_id}: thiếu z ở ngày test.")
        sig = (zte > 0).astype(float) if up_if_pos else (zte < 0).astype(float)
        d = data.loc[te, DATE].to_numpy()
        p = np.where(np.abs(zte) > z, sig, rev_p.loc[d].to_numpy())
        out.append(pd.DataFrame({"date": d, "fold": f.fold_id, "y_true": data.loc[te, TARGET].astype(int).to_numpy(),
                                 "p_hat": p, "regime": pd.NA, "trig": np.abs(zte) > z}))
    res = pd.concat(out, ignore_index=True)
    return res[PREDICTION_COLS], res.trig.to_numpy(), n_pos


# ----------------------------------------------------------------------------
# Bảng
# ----------------------------------------------------------------------------
def vs_rev(label, kind, p, rev, mask=None):
    y = p.y_true.to_numpy()
    r = {"approach": kind, "method": label, **score(p)}
    d = stats.paired_row(y, rev.p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=False)
    r |= {"d_acc_vs_rev1": d["d_da"], "d_acc_lo": d["d_da_lo"], "d_acc_hi": d["d_da_hi"], "p_mcnemar": d["p_mcnemar"]}
    if mask is not None:
        m = mask
        r["n_days_active"] = int(m.sum())
        r["coverage"] = m.mean()
        if m.sum():
            r["acc_active_method"] = stats.row_hit(y[m], p.p_hat.to_numpy()[m]).mean()
            r["acc_active_rev1"] = stats.row_hit(y[m], rev.p_hat.to_numpy()[m]).mean()
            if m.sum() >= 30:
                da = stats.paired_row(y[m], rev.p_hat.to_numpy()[m], p.p_hat.to_numpy()[m], with_logloss=False)
                r |= {"d_acc_active": da["d_da"], "d_acc_active_lo": da["d_da_lo"], "d_acc_active_hi": da["d_da_hi"],
                      "p_mcnemar_active": da["p_mcnemar"]}
    return r


def tables():
    rev = read_csv_dated(R30 / "predictions" / "rev1.csv")
    rows = []
    # A
    a = {}
    for f in sorted((OUT / "a").glob("*.csv")):
        a[f.stem] = read_csv_dated(f)
        rows.append(vs_rev(f"A: {f.stem} + dấu return hôm nay", "A", a[f.stem], rev))
    for k in ML:
        g = GATE_DIR / "baseline" / f"{k}.csv"
        if g.exists() and k in a:
            d = stats.paired_row(a[k].y_true.to_numpy(), read_csv_dated(g).p_hat.to_numpy(), a[k].p_hat.to_numpy(),
                                 with_logloss=False)
            rows[[r["method"] for r in rows].index(f"A: {k} + dấu return hôm nay")] |= {
                "acc_without_sign": score(read_csv_dated(g))["accuracy"], "d_acc_vs_without_sign": d["d_da"],
                "d_acc_vs_without_sign_lo": d["d_da_lo"], "d_acc_vs_without_sign_hi": d["d_da_hi"]}
    # C
    sources = {f"gate {k}": GATE_DIR / "baseline" / f"{k}.csv" for k in ML}
    sources |= {f"stepwise {k}": GATE_DIR / "stepwise" / f"{k}.csv" for k in ML}
    sources |= {f"A {k}": OUT / "a" / f"{k}.csv" for k in ML}
    for lab, f in sources.items():
        if f.exists():
            p, use = override(read_csv_dated(f), rev)
            write_csv(p, OUT / "c" / f"{lab.replace(' ', '_')}.csv")
            rows.append(vs_rev(f"C: đảo chiều, {lab} ghi đè khi |p̂−0,5| > {MARGIN}", "C", p, rev, use))
    # B
    data, _, folds, _, _ = setup("features_onchain_v2.yaml")
    for col in Z_COLS:
        p, trig, n_pos = extreme_rule(data, folds, col, rev)
        write_csv(p, OUT / "b" / f"{col}.csv")
        r = vs_rev(f"B: {col} |z| > {Z:g}", "B", p, rev, trig)
        r["n_folds_up_if_pos"] = n_pos
        rows.append(r)
    t = pd.DataFrame(rows)
    for kind, idx in t.groupby("approach").groups.items():
        t.loc[idx, "p_mcnemar_holm"] = stats.holm(t.loc[idx, "p_mcnemar"])
        if "p_mcnemar_active" in t:
            ok = [i for i in idx if pd.notna(t.loc[i, "p_mcnemar_active"])]
            if ok:
                t.loc[ok, "p_mcnemar_active_holm"] = stats.holm(t.loc[ok, "p_mcnemar_active"])
    write_csv(t, OUT / "tables" / "hybrid_summary.csv")
    return t


def _tests():
    d = pd.date_range("2020-01-01", periods=6)
    rule = pd.DataFrame({"date": d, "y_true": [1, 0, 1, 0, 1, 0], "p_hat": [1.0, 0, 1, 0, 1, 0]})
    model = rule.assign(p_hat=[0.52, 0.6, 0.4, 0.5, 0.56, 0.44])
    p, use = override(model, rule)
    assert use.tolist() == [False, True, True, False, True, True]
    assert p.p_hat.tolist() == [1.0, 0.6, 0.4, 0.0, 0.56, 0.44]
    x = pd.DataFrame({"logret_lag1": [0.1, -0.2, 0.0, np.nan]})
    assert with_sign(x)[SIGN].tolist()[:3] == [1.0, 0.0, 0.0] and np.isnan(with_sign(x)[SIGN].iloc[3])
    print("Tất cả test hybrid đều PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["a", "tables", "test"], required=True)
    ap.add_argument("--models", nargs="+", default=["lr_sign"] + ML)
    args = ap.parse_args(argv)
    _tests()
    if args.stage == "a":
        stage_a(args.models)
    elif args.stage == "tables":
        pd.set_option("display.width", 300)
        t = tables()
        print(t.drop(columns=[c for c in t if c.startswith("f1_") and c != "f1_macro"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
