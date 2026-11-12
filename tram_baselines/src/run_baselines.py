"""
run_baselines.py  –  Trâm: chạy và tuning B0 (Majority, Persistence, Buy-and-hold) và B3 (SVM kernel RBF)

Theo góp ý 3 và 4 của thầy:
  - Ghi rõ phần tuning nào đã thực sự chạy, chọn tham số thế nào, kết quả trước/sau tuning.
  - Tuning bằng train/validation (inner CV trong cửa sổ train), không chọn mô hình hay tham số theo test.
  - Hoàn thiện baseline OHLCV-only trước, rồi mới thêm On-chain với cùng mô hình và cùng protocol.

Dùng lại nguyên dữ liệu, đặc trưng, fold, chuẩn hoá và CalibratedSVC của ngoc_pipeline, để kết quả so được
trực tiếp với bảng E1 của nhóm. Khác biệt duy nhất: vòng walk-forward ở đây ghi lại MỌI tổ hợp tham số đã thử.

Mô hình (tên file không có dấu "_" để khớp quy ước {model}_{config}.csv của ngoc_pipeline):
  b0maj    Majority: p̂ = tỷ lệ ngày tăng của cả cửa sổ train 730 ngày (như ngoc_pipeline). Không tham số.
  b0majlb  Majority có tune độ dài cửa sổ nhìn lại L ∈ LOOKBACKS: p̂ = tỷ lệ ngày tăng của L ngày train cuối.
           L chọn theo log-loss trên inner CV trong cửa sổ train. Đây là tham số duy nhất của Majority.
  b0pers   Persistence: mai cùng hướng hôm nay. Không tham số.
  b0rev    Đảo chiều: mai ngược hướng hôm nay; chiều chọn trên cửa sổ train (luật của ngoc_pipeline).
  b0bh     Buy-and-hold: luôn đoán tăng (p̂ = 1). Không tham số; chỉ có DA và backtest.
  b3def    SVM RBF, tham số mặc định của sklearn (C = 1, gamma = "scale"), KHÔNG tune  -> "trước tuning"
  b3g9     SVM RBF, grid 3 × 3 của configs/models/b3.yaml (C × gamma)                 -> tái tạo b3 của Ngọc
  b3g30    SVM RBF, grid rộng 5 × 6 (GRID_WIDE)                                        -> "sau tuning"
Mọi SVM: xác suất hiệu chỉnh sigmoid (CalibratedSVC), tune lại mỗi 6 fold, inner TimeSeriesSplit(5, gap 1),
tiêu chí log-loss, hoà thì lấy tổ hợp đứng trước (giống walkforward.tune của Ngọc).

Chạy từ thư mục gốc repo:
    python tram_baselines/src/run_baselines.py            # chạy mô hình (~5 phút) rồi viết báo cáo
    python tram_baselines/src/run_baselines.py --report   # chỉ dựng lại bảng và báo cáo từ dự báo đã lưu
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ngoc_pipeline/src"), str(ROOT / "tram_features_backtest/src")]
from data import DATE, PREDICTION_COLS, TARGET, load_config, load_master, read_csv_dated, write_csv  # noqa: E402
from features import build_dataset                                                                     # noqa: E402
from models import CalibratedSVC, load_model, persistence_predictions, reversal_predictions            # noqa: E402
from walkforward import fold_indices, inner_cv, make_folds, make_pipeline                              # noqa: E402
from backtest import run_backtest, summarize                                                           # noqa: E402
from features_ohlcv import FEATURES_10, verify_freeze as verify_freeze_crlf                            # noqa: E402

FREEZE_FILES = ("master_daily.csv", "ohlcv_1h_clean.csv", "data_dictionary_A.csv", "data_dictionary_B.csv")


def load_master_checked():
    """Kiểm sha256 của mọi file freeze theo cách bỏ qua khác biệt xuống dòng CRLF/LF (Git trên Windows tự đổi
    sang CRLF khi checkout), rồi mới đọc. load_master() của ngoc_pipeline so sha256 trên byte thô nên báo lỗi
    trên Windows; ở đây gọi nó với verify=False sau khi đã tự kiểm."""
    verify_freeze_crlf(FREEZE_FILES)
    return load_master(verify=False)

BASE = Path(__file__).resolve().parents[1]
PRED = BASE / "results/predictions"
TAB = BASE / "results/tables"
FIG = BASE / "reports/figures"
CONFIGS = ("ohlcv", "ohlcv_onchain")            # OHLCV-only chạy trước
LOOKBACKS = [30, 90, 180, 365, 730]
GRID_WIDE = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]}
SVM_DEFAULT = {"C": 1.0, "gamma": "scale"}
RULE_MODELS = {"b0pers", "b0rev", "b0bh"}       # p̂ chỉ 0/1: không có log-loss, Brier, AUC
LABELS = {"b0maj": "B0 Majority (730 ngày)", "b0majlb": "B0 Majority, tune cửa sổ L", "b0pers": "B0 Persistence",
          "b0rev": "B0 Đảo chiều", "b0bh": "B0 Buy-and-hold (luôn tăng)", "b3def": "B3 SVM mặc định (chưa tune)",
          "b3g9": "B3 SVM grid 3×3 (bản của Ngọc)", "b3g30": "B3 SVM grid 5×6 (tune rộng)"}


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def load_all():
    fe, wf = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master_checked(), fe)
    # Báo cáo này làm với 10 đặc trưng OHLCV ban đầu (trước up_today, 12/11/2026); giữ nguyên để tái tạo đúng.
    onchain = [c for c in sets["ohlcv_onchain"] if c not in sets["ohlcv"]]
    sets = {"ohlcv": list(FEATURES_10), "ohlcv_onchain": list(FEATURES_10) + onchain}
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf)
    return data, sets, folds, wf, fe


def _frame(data, te, fold, p):
    return pd.DataFrame({"date": data.loc[te, DATE].to_numpy(), "fold": fold.fold_id,
                         "y_true": data.loc[te, TARGET].astype(int).to_numpy(), "p_hat": p,
                         "regime": pd.NA})


# ---------------------------------------------------------------------------
# B0
# ---------------------------------------------------------------------------
def majority(data, folds, lookback_grid=None, wf=None, log_rows=None):
    """lookback_grid=None: p̂ = tỷ lệ tăng của cả cửa sổ train. Có grid: chọn L trên inner CV mỗi retune_every fold."""
    out, best_L = [], None
    for i, fold in enumerate(folds):
        tr, te = fold_indices(data, fold)
        y = data.loc[tr, TARGET].astype(int).to_numpy()
        if lookback_grid is None:
            out.append(_frame(data, te, fold, np.full(len(te), y.mean())))
            continue
        if i % wf["retune_every"] == 0:
            scores = {}
            for L in lookback_grid:
                losses, accs = [], []
                for itr, iva in inner_cv(wf["inner_splits"], wf["gap_days"]).split(y):
                    p = np.clip(y[itr][-L:].mean(), 1e-6, 1 - 1e-6)
                    losses.append(log_loss(y[iva], np.full(len(iva), p), labels=[0, 1]))
                    accs.append(((p > 0.5) == y[iva]).mean())
                scores[L] = (np.mean(losses), np.mean(accs))
            best_L = min(lookback_grid, key=lambda L: (round(scores[L][0], 10), lookback_grid.index(L)))
            for L, (ll, acc) in scores.items():
                log_rows.append({"model": "b0majlb", "config": "-", "retune": i // wf["retune_every"],
                                 "fold": fold.fold_id, "train_start": fold.train_start.date(),
                                 "train_end": fold.train_end.date(), "params": f"L={L}", "C": np.nan,
                                 "gamma": np.nan, "L": L, "val_log_loss": ll, "val_da": acc,
                                 "chosen": L == best_L, "is_default": L == 730})
        out.append(_frame(data, te, fold, np.full(len(te), y[-best_L:].mean())))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def buy_and_hold(data, folds):
    out = []
    for fold in folds:
        _, te = fold_indices(data, fold)
        out.append(_frame(data, te, fold, np.ones(len(te))))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


# ---------------------------------------------------------------------------
# B3 SVM với log tuning
# ---------------------------------------------------------------------------
def tune_logged(X, y, grid, wf):
    """GridSearchCV theo log-loss như walkforward.tune, nhưng trả cả bảng điểm của mọi tổ hợp,
    và điểm của tham số mặc định (C = 1, gamma = 'scale') trên cùng các split để so trước/sau tuning."""
    pipe = make_pipeline(CalibratedSVC(**load_model("b3")[0].get_params()))
    cv = inner_cv(wf["inner_splits"], wf["gap_days"])
    gs = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                      scoring={"ll": "neg_log_loss", "acc": "accuracy"}, refit=False, cv=cv, n_jobs=-1).fit(X, y)
    r = gs.cv_results_
    best_i = int(np.argmax(np.round(r["mean_test_ll"], 10)))
    rows = [{"C": prm["model__C"], "gamma": prm["model__gamma"], "val_log_loss": -r["mean_test_ll"][j],
             "val_log_loss_sd": r["std_test_ll"][j], "val_da": r["mean_test_acc"][j], "chosen": j == best_i,
             "is_default": False} for j, prm in enumerate(r["params"])]
    d = cross_validate(pipe.set_params(model__C=SVM_DEFAULT["C"], model__gamma=SVM_DEFAULT["gamma"]), X, y,
                       cv=cv, scoring={"ll": "neg_log_loss", "acc": "accuracy"}, n_jobs=-1)
    rows.append({"C": SVM_DEFAULT["C"], "gamma": SVM_DEFAULT["gamma"], "val_log_loss": -d["test_ll"].mean(),
                 "val_log_loss_sd": d["test_ll"].std(), "val_da": d["test_acc"].mean(), "chosen": False,
                 "is_default": True})
    best = {k.removeprefix("model__"): v for k, v in r["params"][best_i].items()}
    return best, rows


def svm_walkforward(data, cols, folds, wf, grid, name, config, log_rows):
    """Giống walkforward.run_walkforward của Ngọc (cùng dropna, retune mỗi 6 fold), thêm ghi log tuning.
    grid=None -> tham số mặc định, không tune."""
    est = load_model("b3")[0]
    out, best = [], dict(SVM_DEFAULT) if grid is None else {}
    for i, fold in enumerate(folds):
        tr, te = fold_indices(data, fold)
        train, test = data.loc[tr].dropna(subset=cols), data.loc[te]
        X, y = train[cols].to_numpy(), train[TARGET].astype(int).to_numpy()
        if grid is not None and i % wf["retune_every"] == 0:
            best, rows = tune_logged(X, y, grid, wf)
            for r in rows:
                log_rows.append({"model": name, "config": config, "retune": i // wf["retune_every"],
                                 "fold": fold.fold_id, "train_start": fold.train_start.date(),
                                 "train_end": fold.train_end.date(), "params": f"C={r['C']}, gamma={r['gamma']}",
                                 "L": np.nan, **r})
        pipe = make_pipeline(est).set_params(**{f"model__{k}": v for k, v in best.items()}).fit(X, y)
        out.append(_frame(data, te, fold, pipe.predict_proba(test[cols].to_numpy())[:, 1]))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


# ---------------------------------------------------------------------------
# Chạy
# ---------------------------------------------------------------------------
def run_models():
    PRED.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    data, sets, folds, wf, fe = load_all()
    print(f"{len(folds)} fold; {len(sets['ohlcv'])} OHLCV, {len(sets['ohlcv_onchain'])} tổng; flows {fe['flows_set']}")
    log_rows, t0 = [], time.time()

    # B0 không dùng đặc trưng: một bộ dự báo, ghi cho cả hai cấu hình (Δ = 0)
    b0 = {"b0maj": majority(data, folds), "b0majlb": majority(data, folds, LOOKBACKS, wf, log_rows),
          "b0pers": persistence_predictions(data, folds), "b0rev": reversal_predictions(data, folds),
          "b0bh": buy_and_hold(data, folds)}
    for name, p in b0.items():
        for config in CONFIGS:
            write_csv(p, PRED / f"{name}_{config}.csv")
    print(f"B0 xong ({time.time()-t0:.0f}s)")

    grid9 = load_model("b3")[1]
    for config in CONFIGS:                                   # OHLCV-only trước
        for name, grid in [("b3def", None), ("b3g9", grid9), ("b3g30", GRID_WIDE)]:
            p = svm_walkforward(data, sets[config], folds, wf, grid, name, config, log_rows)
            write_csv(p, PRED / f"{name}_{config}.csv")
            print(f"{name}_{config} xong ({time.time()-t0:.0f}s)")
    # Thử thêm: SVM chỉ với 1 đặc trưng (logret_lag1), không ghi vào PRED vì không có cặp On-chain
    for name, grid in [("b3def", None), ("b3g30", GRID_WIDE)]:
        p = svm_walkforward(data, ["logret_lag1"], folds, wf, grid, f"{name}f1", "ohlcv_f1", log_rows)
        write_csv(p, TAB / f"f1_{name}.csv")
        print(f"{name} 1 đặc trưng xong ({time.time()-t0:.0f}s)")
    pd.DataFrame(log_rows).to_csv(TAB / "tuning_log.csv", index=False)


# ---------------------------------------------------------------------------
# Bảng
# ---------------------------------------------------------------------------
def hits(p):
    return ((p["p_hat"].to_numpy() > 0.5).astype(int) == p["y_true"].to_numpy()).astype(int)


def metric_row(p, name):
    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    h = hits(p)
    k, n = int(h.sum()), len(h)
    ci = binomtest(k, n).proportion_ci()
    rule = name in RULE_MODELS
    return {"n": n, "da": k / n, "da_ci_low": ci.low, "da_ci_high": ci.high, "p_vs_50": binomtest(k, n, 0.5).pvalue,
            "log_loss": np.nan if rule else log_loss(y, ph, labels=[0, 1]),
            "brier": np.nan if rule else brier_score_loss(y, ph),
            "auc": np.nan if rule or name.startswith("b0maj") else roc_auc_score(y, ph),
            "pct_pred_up": (ph > 0.5).mean()}


def mcnemar(a, b):
    """a, b: mảng 0/1 đúng/sai cùng ngày. Trả (chỉ b đúng, chỉ a đúng, p chính xác)."""
    only_b, only_a = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
    n = only_a + only_b
    return only_b, only_a, (binomtest(only_b, n, 0.5).pvalue if n else 1.0)


def build_tables():
    preds = {tuple(f.stem.split("_", 1)): read_csv_dated(f) for f in sorted(PRED.glob("*_*.csv"))}
    order = list(LABELS)
    keys = sorted(preds, key=lambda k: (CONFIGS.index(k[1]), order.index(k[0])))
    m = pd.DataFrame([{"model": n, "config": c, **metric_row(preds[(n, c)], n)} for n, c in keys])

    rev = {c: hits(preds[("b0rev", c)]) for c in CONFIGS}
    vs = []
    for n, c in keys:
        if n.startswith("b3"):
            ob, oa, p = mcnemar(rev[c], hits(preds[(n, c)]))
            vs.append({"model": n, "config": c, "da_model": hits(preds[(n, c)]).mean(), "da_rev": rev[c].mean(),
                       "d_da": hits(preds[(n, c)]).mean() - rev[c].mean(), "only_model_right": ob,
                       "only_rev_right": oa, "p_mcnemar": p})
    vs = pd.DataFrame(vs)

    delta = []
    for n in [k for k in order if k.startswith("b3")]:
        a, b = preds[(n, "ohlcv")], preds[(n, "ohlcv_onchain")]
        ob, oa, p = mcnemar(hits(a), hits(b))
        ma, mb = metric_row(a, n), metric_row(b, n)
        delta.append({"model": n, "da_ohlcv": ma["da"], "da_onchain": mb["da"], "d_da": mb["da"] - ma["da"],
                      "p_mcnemar": p, "ll_ohlcv": ma["log_loss"], "ll_onchain": mb["log_loss"],
                      "d_log_loss": mb["log_loss"] - ma["log_loss"]})
    delta = pd.DataFrame(delta)

    yr = []
    for n, c in keys:
        p = preds[(n, c)]
        years = p[DATE].dt.year
        for y_ in sorted(years.unique()):
            q = p[years == y_]
            yr.append({"model": n, "config": c, "year": int(y_), "n": len(q), "da": hits(q).mean(),
                       "pct_pred_up": (q["p_hat"] > 0.5).mean(), "up_share": q["y_true"].mean()})
    yr = pd.DataFrame(yr)

    master = load_master_checked().set_index(DATE)
    close = master["ohlcv_close"].dropna()
    bt = []
    for n, c in keys:
        p = preds[(n, c)].set_index(DATE)["p_hat"]
        s = summarize(run_backtest(close, p))
        bt.append({"model": n, "config": c, **s.loc["strategy"].to_dict()})
    bh = summarize(run_backtest(close, preds[("b0bh", "ohlcv")].set_index(DATE)["p_hat"])).loc["buy_and_hold"]
    bt = pd.DataFrame(bt)

    for name, t in [("test_metrics", m), ("vs_reversal", vs), ("delta_onchain", delta), ("by_year", yr),
                    ("backtest", bt)]:
        t.to_csv(TAB / f"{name}.csv", index=False)
    return preds, m, vs, delta, yr, bt, bh


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def pct(x, d=2):
    return f"{x*100:.{d}f}".replace(".", ",")


def num(x, d=4):
    return "–" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def pval(x):
    return "<0,001" if x < 0.001 else num(x, 3)


def write_report(preds, m, vs, delta, yr, bt, bh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(TAB / "tuning_log.csv")
    log["gamma_num"] = pd.to_numeric(log["gamma"], errors="coerce")      # "scale" của mặc định -> NaN
    data, sets, folds, wf, _ = load_all()
    md = []
    M = m.set_index(["model", "config"])

    def mt(names, config):
        rows = []
        for n in names:
            r = M.loc[(n, config)]
            rows.append(f"| {LABELS[n]} | {pct(r.da)} | [{pct(r.da_ci_low,1)}; {pct(r.da_ci_high,1)}] | {pval(r.p_vs_50)} "
                        f"| {num(r.log_loss)} | {num(r.brier)} | {num(r.auc,3)} | {pct(r.pct_pred_up,1)} |")
        return ["| Mô hình | DA (%) | CI 95% | p (DA = 50%) | Log-loss | Brier | AUC | % ngày đoán tăng |",
                "|---|---|---|---|---|---|---|---|"] + rows

    up_all = preds[("b0bh", "ohlcv")]["y_true"].mean()
    rev = M.loc[("b0rev", "ohlcv")]
    md += ["# Chạy và tuning baseline B0 (Majority, Persistence, Buy-and-hold) và B3 (SVM RBF) – Trâm", "",
           "Tạo bởi `tram_baselines/src/run_baselines.py`. Dữ liệu, đặc trưng, 82 fold walk-forward (train 730 ngày, "
           "gap 1, test 30, 2020-01-01 → 2026-08-30), chuẩn hoá và CalibratedSVC lấy nguyên từ `ngoc_pipeline`. "
           f"{len(sets['ohlcv'])} đặc trưng OHLCV; {len(sets['ohlcv_onchain'])} đặc trưng khi thêm On-chain (Flows A). "
           f"Tỷ lệ ngày tăng trong {len(preds[('b0bh','ohlcv')])} ngày test: {pct(up_all,1)}%.", ""]

    # ---- tóm tắt (điền sau) ----
    summary_at = len(md)

    # ---- 1. protocol ----
    md += ["## 1. Tuning đã chạy thực tế", "",
           "| Mô hình | Tham số | Giá trị đã thử | Chọn bằng | Bao lâu chọn lại |", "|---|---|---|---|---|",
           "| B0 Majority (730 ngày) | – | không có tham số | – | – |",
           f"| B0 Majority, tune cửa sổ L | L = số ngày train cuối dùng để tính tỷ lệ tăng | {', '.join(map(str, LOOKBACKS))} | "
           "log-loss trung bình trên inner TimeSeriesSplit(5, gap 1) trong cửa sổ train | mỗi 6 fold (14 lần) |",
           "| B0 Persistence | – | không có tham số | – | – |",
           "| B0 Đảo chiều | chiều của luật (cùng / ngược hôm nay) | 2 chiều | % đúng của persistence trên cửa sổ train (< 50% thì đảo) | mỗi fold |",
           "| B0 Buy-and-hold | – | không có tham số | – | – |",
           "| B3 SVM mặc định | C = 1, gamma = \"scale\" (mặc định sklearn) | không tune | – | – |",
           f"| B3 SVM grid 3×3 | C, gamma | C ∈ {load_model('b3')[1]['C']}, gamma ∈ {load_model('b3')[1]['gamma']} | "
           "log-loss trên inner TimeSeriesSplit(5, gap 1) trong cửa sổ train | mỗi 6 fold (14 lần) |",
           f"| B3 SVM grid 5×6 | C, gamma | C ∈ {GRID_WIDE['C']}, gamma ∈ {GRID_WIDE['gamma']} | như trên | mỗi 6 fold (14 lần) |",
           "", "Ngày test không bao giờ nằm trong inner CV: các split chỉ cắt trong 730 ngày train của fold. "
           "Xác suất của SVM được hiệu chỉnh sigmoid trên điểm ngoài mẫu theo thời gian, cũng trong cửa sổ train. "
           "Toàn bộ điểm của mọi tổ hợp ở mọi lần tune: `results/tables/tuning_log.csv`.", "",
           "**Kiểm tra tái tạo:** `b3g9` dùng đúng grid và code của Ngọc, nên phải ra cùng dự báo với "
           "`ngoc_pipeline/results/predictions/b3_*.csv`."]
    for c in CONFIGS:
        f = ROOT / f"ngoc_pipeline/results/predictions/b3_{c}.csv"
        if f.exists():
            old, new = read_csv_dated(f), preds[("b3g9", c)]
            flips = int(((old.p_hat > .5) != (new.p_hat.to_numpy() > .5)).sum())
            md.append(f"- {c}: {flips} ngày đổi hướng dự báo, lệch p̂ lớn nhất {np.abs(old.p_hat.to_numpy()-new.p_hat.to_numpy()).max():.1e}.")
    md.append("")

    # ---- 2. B0 ----
    b0n = ["b0maj", "b0majlb", "b0pers", "b0rev", "b0bh"]
    lb = log[log.model == "b0majlb"]
    lb_ch = lb[lb.chosen].L.value_counts().sort_index()
    lb_sum = lb.groupby("L").agg(val_log_loss=("val_log_loss", "mean"), val_da=("val_da", "mean"),
                                 so_lan_chon=("chosen", "sum"))
    md += ["## 2. B0: Majority, Persistence, Buy-and-hold (không dùng đặc trưng)", "",
           "B0 không dùng đặc trưng nên hai cấu hình cho cùng dự báo. Persistence, đảo chiều, buy-and-hold chỉ cho p̂ = 0 hoặc 1, "
           "nên chỉ so được DA (log-loss, Brier, AUC để trống).", ""] + mt(b0n, "ohlcv") + [
           "", "### Tuning cửa sổ của Majority", "",
           "Điểm trung bình qua 14 lần tune (inner validation trong cửa sổ train):", "",
           "| L (ngày) | Val log-loss TB | Val DA TB (%) | Số lần được chọn |", "|---|---|---|---|"]
    md += [f"| {int(L)} | {num(r.val_log_loss)} | {pct(r.val_da,1)} | {int(r.so_lan_chon)} |" for L, r in lb_sum.iterrows()]
    maj, majlb = M.loc[("b0maj", "ohlcv")], M.loc[("b0majlb", "ohlcv")]
    md += ["", f"Trước tuning (L = 730): DA {pct(maj.da)}%, log-loss {num(maj.log_loss)}. "
           f"Sau tuning: DA {pct(majlb.da)}%, log-loss {num(majlb.log_loss)}. "
           f"L = 730 được chọn nhiều nhất ({int(lb_sum.loc[730,'so_lan_chon'])}/14 lần). DA nhích lên nhưng log-loss tệ đi: "
           "cửa sổ ngắn bám theo xu hướng vài tháng gần đây, nhưng tỷ lệ tăng gần đây gần như không nói gì về ngày mai.", "",
           f"**Buy-and-hold như một bộ dự báo** đúng {pct(M.loc[('b0bh','ohlcv')].da)}%, đúng bằng tỷ lệ ngày tăng. "
           f"**Đảo chiều** đúng {pct(rev.da)}% (p = {pval(rev.p_vs_50)}), là B0 mạnh nhất và là mốc mà B3 phải vượt.", ""]

    # ---- 3. SVM tuning ----
    md += ["## 3. B3 SVM RBF: tuning", ""]
    figs_tune = []
    for c in CONFIGS:
        L = log[(log.model == "b3g30") & (log.config == c)]
        ch = L[L.chosen]
        df_ = L[L.is_default]
        piv = (L[~L.is_default].groupby(["C", "gamma_num"]).val_log_loss.mean().unstack("gamma_num"))
        cnt = ch.groupby(["C", "gamma_num"]).size().rename("lần chọn").reset_index()
        rows = []
        for (_, a), (_, b) in zip(ch.sort_values("retune").iterrows(), df_.sort_values("retune").iterrows()):
            rows.append(f"| {int(a.retune)} | {a.train_start} → {a.train_end} | C = {a.C:g}, γ = {a.gamma_num:g} | "
                        f"{num(b.val_log_loss)} | {num(a.val_log_loss)} | {pct(b.val_da,1)} | {pct(a.val_da,1)} |")
        md += [f"### Cấu hình {c} ({len(sets[c])} đặc trưng), grid 5×6", "",
               "Mỗi lần tune: tham số được chọn, và điểm inner validation trước tuning (mặc định C = 1, γ = scale) "
               "so với sau tuning (tổ hợp được chọn). Log-loss thấp hơn là tốt hơn; đoán 50/50 cho log-loss 0,6931.", "",
               "| Lần tune | Cửa sổ train | Tham số chọn | Val log-loss trước | Val log-loss sau | Val DA trước (%) | Val DA sau (%) |",
               "|---|---|---|---|---|---|---|"] + rows + [
               "", f"14 lần tune chọn ra **{len(cnt)} tổ hợp khác nhau**, C nhảy từ {ch.C.min():g} đến {ch.C.max():g} và "
               f"γ từ {ch.gamma_num.min():g} đến {ch.gamma_num.max():g}. Số lần mỗi tổ hợp được chọn: "
               + "; ".join(f"C = {r.C:g}, γ = {r.gamma_num:g}: {int(r['lần chọn'])}" for _, r in cnt.iterrows()) + ". "
               "Tham số tốt nhất đổi gần như mỗi lần tune, nghĩa là các tổ hợp cho điểm validation gần bằng nhau và "
               "lựa chọn chủ yếu do nhiễu.", ""]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [f"{g:g}" for g in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{C:g}" for C in piv.index])
        ax.set_xlabel("gamma")
        ax.set_ylabel("C")
        for i_ in range(piv.shape[0]):
            for j_ in range(piv.shape[1]):
                ax.text(j_, i_, f"{piv.values[i_, j_]:.4f}", ha="center", va="center", fontsize=7,
                        color="white" if piv.values[i_, j_] > np.nanmedian(piv.values) else "black")
        ax.set_title(f"Val log-loss TB qua 14 lần tune ({c})", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fname = f"svm_grid_{c}.png"
        fig.savefig(FIG / fname, dpi=110)
        plt.close(fig)
        bad = piv.stack()
        bad = bad[bad > 0.75]
        n_bad_chosen = sum((r.C, r.gamma_num) in set(bad.index) for r in ch.itertuples())
        md += [f"![grid {c}](figures/{fname})", ""]
        if len(bad):
            md += [f"{len(bad)}/30 tổ hợp có val log-loss trung bình trên 0,75 (tệ hơn hẳn đoán 50/50), đều ở C nhỏ "
                   f"(C ≤ {bad.index.get_level_values(0).max():g}). Với C nhỏ, SVM gần như cho cùng một điểm quyết định cho mọi ngày, "
                   "nên bước hiệu chỉnh sigmoid nhiều lần bị dốc quá mức và cho xác suất cực đoan. "
                   + (f"Dù trung bình kém, các tổ hợp này vẫn được chọn {n_bad_chosen}/14 lần tune, vì ở những cửa sổ đó chúng "
                      "tình cờ tốt nhất: khi điểm SVM gần như hằng số và hiệu chỉnh không gãy, p̂ gần bằng tỷ lệ ngày tăng của train, "
                      "tức SVM trở thành Majority." if n_bad_chosen else "Không lần tune nào chọn các tổ hợp này."), ""]
        figs_tune.append(fname)

    # ---- 4. kết quả test ----
    md += ["## 4. Kết quả trên 2.434 ngày test", "", "### 4.1 OHLCV-only (làm trước)", ""] + \
          mt(["b0maj", "b0rev", "b3def", "b3g9", "b3g30"], "ohlcv") + \
          ["", "### 4.2 OHLCV + On-chain (cùng mô hình, cùng protocol, cùng grid)", ""] + \
          mt(["b0maj", "b0rev", "b3def", "b3g9", "b3g30"], "ohlcv_onchain") + [""]
    md += ["### 4.3 SVM so với luật đảo chiều (cùng ngày test, McNemar chính xác)", "",
           "| SVM | Cấu hình | DA SVM (%) | DA đảo chiều (%) | ΔDA (điểm) | Ngày chỉ SVM đúng | Ngày chỉ luật đúng | p |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {LABELS[r.model]} | {r.config} | {pct(r.da_model)} | {pct(r.da_rev)} | {pct(r.d_da)} | {r.only_model_right} | "
           f"{r.only_rev_right} | {pval(r.p_mcnemar)} |" for r in vs.itertuples()]
    md += ["", "### 4.4 Δ khi thêm On-chain (OHLCV + On-chain − OHLCV-only)", "",
           "| SVM | DA OHLCV (%) | DA + On-chain (%) | ΔDA (điểm) | p McNemar | Log-loss OHLCV | Log-loss + On-chain | ΔLog-loss |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {LABELS[r.model]} | {pct(r.da_ohlcv)} | {pct(r.da_onchain)} | {pct(r.d_da)} | {pval(r.p_mcnemar)} | "
           f"{num(r.ll_ohlcv)} | {num(r.ll_onchain)} | {num(r.d_log_loss)} |" for r in delta.itertuples()]
    md += ["", "ΔLog-loss âm nghĩa là On-chain làm xác suất tốt hơn.", ""]

    # ---- 5. theo năm ----
    yy = yr[(yr.config == "ohlcv") & yr.model.isin(["b0rev", "b3def", "b3g9", "b3g30"])].pivot(
        index="year", columns="model", values="da")[["b0rev", "b3def", "b3g9", "b3g30"]]
    up = yr[(yr.config == "ohlcv") & (yr.model == "b0bh")].set_index("year").up_share
    md += ["## 5. DA theo năm (OHLCV-only)", "",
           "| Năm | Tỷ lệ ngày tăng | Đảo chiều | SVM mặc định | SVM 3×3 | SVM 5×6 |", "|---|---|---|---|---|---|"]
    md += [f"| {y_} | {pct(up[y_],1)} | {pct(r.b0rev,1)} | {pct(r.b3def,1)} | {pct(r.b3g9,1)} | {pct(r.b3g30,1)} |"
           for y_, r in yy.iterrows()]
    nb = {k: int((yy[k] > yy["b0rev"]).sum()) for k in ["b3def", "b3g9", "b3g30"]}
    md += ["", "Số năm SVM thắng luật đảo chiều: " + "; ".join(f"{LABELS[k]} {v}/{len(yy)}" for k, v in nb.items()) + ".", ""]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    (yy * 100).rename(columns=LABELS).plot(ax=ax, marker="o")
    ax.axhline(50, color="grey", ls="--", lw=1)
    ax.set_ylabel("DA (%)")
    ax.set_xlabel("năm")
    ax.set_title("DA theo năm, OHLCV-only")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "da_by_year.png", dpi=110)
    plt.close(fig)
    md += ["![theo năm](figures/da_by_year.png)", ""]

    # ---- 6. chẩn đoán SVM ----
    diag_rows = []
    for n in ["b3def", "b3g9", "b3g30"]:
        for c in CONFIGS:
            p = preds[(n, c)]
            fold_up = p.groupby("fold").apply(lambda q: (q.p_hat > .5).mean(), include_groups=False)
            train_up = pd.Series({f.fold_id: data.loc[fold_indices(data, f)[0], TARGET].mean() for f in folds})
            q = p.p_hat.quantile([.05, .5, .95])
            diag_rows.append({"model": n, "config": c, "p05": q[.05], "p50": q[.5], "p95": q[.95],
                              "pct_up": (p.p_hat > .5).mean(),
                              "folds_all_same": int(((fold_up == 0) | (fold_up == 1)).sum()),
                              "corr_trainup_foldup": np.corrcoef(train_up.loc[fold_up.index], fold_up)[0, 1]})
    dg = pd.DataFrame(diag_rows)
    dg.to_csv(TAB / "svm_diagnostics.csv", index=False)
    md += ["## 6. Chẩn đoán: SVM đang dự báo gì?", "",
           "| SVM | Cấu hình | p̂ 5% | p̂ trung vị | p̂ 95% | % ngày đoán tăng | Số fold (/82) đoán cùng một hướng cả 30 ngày | Tương quan: tỷ lệ tăng của train ↔ % đoán tăng trong fold |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {LABELS[r.model]} | {r.config} | {num(r.p05,3)} | {num(r.p50,3)} | {num(r.p95,3)} | {pct(r.pct_up,1)} | "
           f"{r.folds_all_same} | {num(r.corr_trainup_foldup,2)} |" for r in dg.itertuples()]
    md += ["", "Đọc bảng: p̂ của SVM gần như luôn nằm sát 0,5; ở nhiều fold SVM đoán cùng một hướng cho cả 30 ngày; "
           "và % ngày đoán tăng trong fold đi theo tỷ lệ tăng của cửa sổ train. Nghĩa là SVM chủ yếu lặp lại "
           "tỷ lệ ngày tăng của train (giống Majority), chứ không phân biệt được ngày tăng với ngày giảm. "
           "Đây cũng là lý do SVM \"lệch về tăng\" mà báo cáo giữa kỳ ghi nhận: các cửa sổ train phần lớn có hơn 50% ngày tăng.", ""]

    # ---- 7. backtest ----
    g30_da = M.loc[("b3g30", "ohlcv")].da
    g30_long = bt[(bt.model == "b3g30") & (bt.config == "ohlcv")].pct_days_long.iloc[0]
    md += ["## 7. Backtest long/flat (phí 0,1%/lượt, code `tram_features_backtest/src/backtest.py`)", "",
           "| Mô hình | Cấu hình | Lợi nhuận tích luỹ | Sharpe | MDD | Số lần giao dịch | % ngày giữ BTC |", "|---|---|---|---|---|---|---|"]
    for r in bt.itertuples():
        if r.model in ("b0majlb",) or (r.model.startswith("b0") and r.config != "ohlcv"):
            continue
        md.append(f"| {LABELS[r.model]} | {r.config if r.model.startswith('b3') else '–'} | {pct(r.cum_return,0)}% | "
                  f"{num(r.sharpe,2)} | {pct(r.mdd,0)}% | {int(r.n_trades)} | {pct(r.pct_days_long,0)}% |")
    md += ["", f"Buy-and-hold thật (mua ngày đầu, trả phí một lần): {pct(bh.cum_return,0)}%, Sharpe {num(bh.sharpe,2)}, "
           f"MDD {pct(bh.mdd,0)}%. SVM grid 5×6 OHLCV-only lời nhiều nhất trong các SVM dù DA chỉ {pct(g30_da)}%, vì nó giữ BTC "
           f"{pct(g30_long,0)}% số ngày nên hưởng phần lớn đà tăng dài hạn, gần giống buy-and-hold. "
           "Như báo cáo giữa kỳ, lợi nhuận phụ thuộc vài ngày biến động mạnh; chỉ dùng tham khảo.", ""]

    # ---- 8. SVM 1 đặc trưng ----
    f1 = {}
    for n in ["b3def", "b3g30"]:
        f = TAB / f"f1_{n}.csv"
        if f.exists():
            q = read_csv_dated(f)
            f1[n] = (metric_row(q, n), mcnemar(hits(preds[("b0rev", "ohlcv")]), hits(q)), (q.p_hat > .5).to_numpy() != preds[("b0rev", "ohlcv")].p_hat.to_numpy().astype(bool))
    if f1:
        md += ["## 8. Thử thêm: SVM chỉ với 1 đặc trưng (logret_lag1)", "",
               "Cùng protocol, cùng 82 fold, chỉ khác bộ đặc trưng. Mục đích: xem SVM có học được quy luật đảo chiều "
               "khi bỏ 9 đặc trưng còn lại không.", "",
               "| SVM 1 đặc trưng | DA (%) | CI 95% | Log-loss | % ngày đoán tăng | So với đảo chiều: ΔDA (điểm), p | % ngày cùng hướng với luật đảo chiều |",
               "|---|---|---|---|---|---|---|"]
        for n, (r, (ob, oa, pv), diff) in f1.items():
            md.append(f"| {LABELS[n].replace('B3 ', '')} | {pct(r['da'])} | [{pct(r['da_ci_low'],1)}; {pct(r['da_ci_high'],1)}] | "
                      f"{num(r['log_loss'])} | {pct(r['pct_pred_up'],1)} | {pct(r['da']-rev.da)}, p = {pval(pv)} | {pct(1-diff.mean(),1)} |")
        md += [""]

    # ---- tóm tắt ----
    g30, g9, de = M.loc[("b3g30", "ohlcv")], M.loc[("b3g9", "ohlcv")], M.loc[("b3def", "ohlcv")]
    best_svm = max(["b3def", "b3g9", "b3g30"], key=lambda n: M.loc[(n, "ohlcv")].da)
    v30 = vs[(vs.model == "b3g30") & (vs.config == "ohlcv")].iloc[0]
    d30 = delta[delta.model == "b3g30"].iloc[0]
    L30 = log[(log.model == "b3g30") & (log.config == "ohlcv")]
    gain = (L30[L30.is_default].sort_values("retune").val_log_loss.to_numpy()
            - L30[L30.chosen].sort_values("retune").val_log_loss.to_numpy())
    d9 = dg[(dg.model == "b3g9") & (dg.config == "ohlcv")].iloc[0]
    summary = ["## Tóm tắt", "",
               f"1. **B0.** Majority đúng {pct(maj.da)}%, Persistence {pct(M.loc[('b0pers','ohlcv')].da)}%, "
               f"Buy-and-hold (luôn đoán tăng) {pct(M.loc[('b0bh','ohlcv')].da)}%, Đảo chiều {pct(rev.da)}% (p = {pval(rev.p_vs_50)}). "
               f"Majority chỉ có một tham số (độ dài cửa sổ L); tune L cho DA {pct(majlb.da)}% nhưng log-loss tệ hơn "
               f"({num(maj.log_loss)} → {num(majlb.log_loss)}), nên giữ bản 730 ngày.",
               f"2. **Tuning SVM có chạy và đã ghi lại đủ.** Grid 5×6 = 30 tổ hợp × 14 lần tune × 2 cấu hình, chọn bằng inner CV "
               f"trong cửa sổ train. Trên inner validation, tổ hợp được chọn có log-loss thấp hơn mặc định trung bình "
               f"{num(gain.mean())} ({num(L30[L30.is_default].val_log_loss.mean())} → {num(L30[L30.chosen].val_log_loss.mean())}). "
               f"Nhưng mức cải thiện này lạc quan sẵn (chọn tổ hợp tốt nhất trong 30 trên chính validation đó), và 14 lần tune "
               f"chọn ra {L30[L30.chosen].params.nunique()} tổ hợp khác nhau: tuning không ổn định.",
               f"3. **Trên test, tuning không giúp.** OHLCV-only: DA mặc định {pct(de.da)}%, grid 3×3 {pct(g9.da)}%, grid 5×6 {pct(g30.da)}%; "
               f"log-loss {num(de.log_loss)} → {num(g9.log_loss)} → {num(g30.log_loss)}, tức bản **chưa tune có log-loss tốt nhất**. "
               "Không bản nào khác 50% có ý nghĩa.",
               f"4. **Không SVM nào thắng luật đảo chiều.** Grid 5×6 OHLCV-only kém luật {pct(-v30.d_da)} điểm (p = {pval(v30.p_mcnemar)}).",
               f"5. **Vì sao:** p̂ của SVM gần như luôn trong khoảng {num(d9.p05,2)}–{num(d9.p95,2)}, và % ngày đoán tăng trong mỗi fold "
               f"đi theo tỷ lệ tăng của cửa sổ train (tương quan {num(d9.corr_trainup_foldup,2)} với grid 3×3). SVM gần như là Majority: "
               "không tìm được ranh giới giữa ngày tăng và ngày giảm trong 10 đặc trưng OHLCV.",
               f"6. **Thêm On-chain (cùng protocol):** ΔDA của grid 5×6 = {pct(d30.d_da)} điểm (p = {pval(d30.p_mcnemar)}), "
               f"Δlog-loss {num(d30.d_log_loss)}. Vì baseline OHLCV-only chưa vượt luật đảo chiều, Δ này chưa đủ cơ sở để kết luận.",
               ]
    if f1:
        r1 = f1["b3g30"][0]
        same = 1 - f1["b3g30"][2].mean()
        summary.append(f"7. **SVM chỉ với 1 đặc trưng (logret_lag1)** đúng {pct(r1['da'])}% (grid 5×6), so với {pct(g30.da)}% "
                       f"của bản 10 đặc trưng và {pct(rev.da)}% của luật đảo chiều. Dù chỉ nhìn đúng thông tin mà luật đảo chiều dùng, "
                       f"SVM vẫn đoán tăng {pct(r1['pct_pred_up'],1)}% số ngày và chỉ cùng hướng với luật ở {pct(same,1)}% số ngày: "
                       "SVM RBF không khai thác được quy luật đảo chiều.")
    summary += ["", "**Đề xuất cho nhóm:**", "",
               "- Ghi B3 vào báo cáo với cả ba bản (mặc định, 3×3, 5×6) và bảng tuning ở mục 3, để thầy thấy tuning đã chạy và không cải thiện.",
               "- Không mở rộng grid SVM thêm: tham số tốt nhất đổi mỗi lần tune và bản chưa tune có log-loss tốt nhất trên test.",
               "- Giữ luật đảo chiều làm mốc. Chỉ so On-chain cho mô hình nào đã vượt mốc này trên OHLCV-only; hiện SVM chưa vượt.", ""]
    md[summary_at:summary_at] = summary
    (BASE / "reports/baselines_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(summary))
    print(f"Đã lưu {BASE / 'reports/baselines_report.md'}")


def main():
    if "--report" not in sys.argv:
        run_models()
    write_report(*build_tables())


if __name__ == "__main__":
    main()
