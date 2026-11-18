"""
compare_up_today.py  –  Trâm: thêm đặc trưng up_today, chạy lại B1 (lr, lrfs) và B3 (SVM RBF)

up_today = 1 nếu giá đóng cửa ngày t cao hơn ngày t−1 (features/src/features_ohlcv.py).
So hai bộ đặc trưng OHLCV trên cùng mô hình, cùng config, cùng 82 fold của pipeline:
  10 đặc trưng  = FEATURES_10 (bộ ban đầu, như bảng E1 hiện tại)
  11 đặc trưng  = FEATURES    (10 + up_today)
và cho cả hai cấu hình: OHLCV-only và OHLCV + On-chain (33 cột On-chain, Flows A).

Mô hình và grid lấy nguyên từ pipeline/configs/models/{lr,lrfs,b3}.yaml; tune bằng inner CV trong cửa sổ train
(walkforward.tune), tune lại mỗi 6 fold. Vòng walk-forward giống walkforward.run_walkforward, chỉ thêm ghi lại
tham số được chọn ở mỗi lần tune và các cột mà lrfs giữ lại ở mỗi fold.

Chạy từ thư mục gốc repo:
    python analyses/up_today/src/compare_up_today.py            # chạy mô hình (~5 phút) rồi viết báo cáo
    python analyses/up_today/src/compare_up_today.py --report   # chỉ dựng lại báo cáo
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "pipeline/src"), str(ROOT / "features/src"),
                str(ROOT / "analyses/baselines/src")]
from data import DATE, PREDICTION_COLS, TARGET, load_config, read_csv_dated, write_csv   # noqa: E402
from features import build_dataset                                                        # noqa: E402
from features_ohlcv import FEATURES, FEATURES_10                                          # noqa: E402
from models import load_model, reversal_predictions                                      # noqa: E402
from walkforward import fold_indices, inner_cv, make_folds, make_pipeline, tune          # noqa: E402
from run_baselines import hits, load_master_checked, mcnemar, metric_row                 # noqa: E402
from backtest import run_backtest, summarize                                              # noqa: E402

BASE = Path(__file__).resolve().parents[1]
PRED = BASE / "results/predictions"
TAB = BASE / "results/tables"
FIG = BASE / "reports/figures"
MODELS = ["lr", "lrfs", "b3"]
MODEL_LABEL = {"lr": "B1 Logistic Regression (lr)", "lrfs": "B1 LR chọn đặc trưng tiến dần (lrfs)",
               "b3": "B3 SVM RBF (b3)"}
CONFIGS = ("ohlcv", "ohlcv_onchain")
FSETS = ("f10", "f11")                      # f10 = 10 đặc trưng ban đầu, f11 = 10 + up_today


def load_all():
    fe, wf = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master_checked(), fe)
    if sets["ohlcv"] != FEATURES:
        raise ValueError(f"Bộ OHLCV của pipeline khác FEATURES: {sets['ohlcv']}")
    onchain = [c for c in sets["ohlcv_onchain"] if c not in FEATURES]
    cols = {("f10", "ohlcv"): list(FEATURES_10), ("f11", "ohlcv"): list(FEATURES),
            ("f10", "ohlcv_onchain"): list(FEATURES_10) + onchain, ("f11", "ohlcv_onchain"): list(FEATURES) + onchain}
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf)
    return data, cols, folds, wf


def walkforward_logged(data, cols, folds, wf, name, tag, log):
    """Như walkforward.run_walkforward, thêm ghi tham số đã chọn và cột lrfs giữ lại."""
    est, grid = load_model(name)
    out, best = [], {}
    for i, fold in enumerate(folds):
        tr, te = fold_indices(data, fold)
        train, test = data.loc[tr].dropna(subset=cols), data.loc[te]
        X, y = train[cols].to_numpy(), train[TARGET].astype(int).to_numpy()
        pipe = make_pipeline(est)
        if grid and i % wf["retune_every"] == 0:
            best = tune(pipe, grid, X, y, inner_cv(wf["inner_splits"], wf["gap_days"]))
        pipe.set_params(**{f"model__{k}": v for k, v in best.items()}).fit(X, y)
        m = pipe.named_steps["model"]
        row = {"model": name, "tag": tag, "fold": fold.fold_id, "retuned": bool(grid) and i % wf["retune_every"] == 0,
               **{f"param_{k}": v for k, v in best.items()}}
        if hasattr(m, "support_"):
            kept = [cols[j] for j in m.support_]
            row |= {"n_kept": len(kept), "kept": "|".join(kept), "kept_up_today": "up_today" in kept}
        log.append(row)
        out.append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": fold.fold_id,
                                 "y_true": test[TARGET].astype(int).to_numpy(),
                                 "p_hat": pipe.predict_proba(test[cols].to_numpy())[:, 1], "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def lr_fixed_c(data, cols, folds, C=1.0):
    """LR với C cố định (không tune), cùng fold và scaler: để thấy tuning theo log-loss đổi DA thế nào."""
    from sklearn.linear_model import LogisticRegression
    out = []
    for fold in folds:
        tr, te = fold_indices(data, fold)
        train, test = data.loc[tr].dropna(subset=cols), data.loc[te]
        pipe = make_pipeline(LogisticRegression(C=C, max_iter=1000)).fit(train[cols].to_numpy(),
                                                                       train[TARGET].astype(int).to_numpy())
        out.append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": fold.fold_id,
                                 "y_true": test[TARGET].astype(int).to_numpy(),
                                 "p_hat": pipe.predict_proba(test[cols].to_numpy())[:, 1], "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def run_models():
    PRED.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    data, cols, folds, wf = load_all()
    print(f"{len(folds)} fold; OHLCV 10 → 11 cột; On-chain {len(cols[('f10','ohlcv_onchain')]) - 10} cột")
    t0 = time.time()
    (TAB / "fit_logs").mkdir(exist_ok=True)
    write_csv(reversal_predictions(data, folds), PRED / "b0rev.csv")
    for config in CONFIGS:                      # OHLCV-only trước
        for name in MODELS:
            for fs in FSETS:
                tag = f"{fs}_{config}"
                out = PRED / f"{name}__{tag}.csv"
                if out.exists():                # chạy tiếp từ chỗ dừng: lrfs + On-chain mất ~25 phút mỗi lần
                    print(f"{name} {tag}: đã có, bỏ qua")
                    continue
                log = []
                p = walkforward_logged(data, cols[(fs, config)], folds, wf, name, tag, log)
                write_csv(p, out)
                pd.DataFrame(log).to_csv(TAB / "fit_logs" / f"{name}__{tag}.csv", index=False)
                print(f"{name} {tag} xong ({time.time()-t0:.0f}s)", flush=True)
    logs = [pd.read_csv(f) for f in sorted((TAB / "fit_logs").glob("*.csv"))]
    pd.concat(logs, ignore_index=True).to_csv(TAB / "fit_log.csv", index=False)


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def pct(x, d=2):
    return f"{x*100:.{d}f}".replace(".", ",")


def num(x, d=4):
    return "–" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def pval(x):
    return "<0,001" if x < 0.001 else num(x, 3)


def write_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    preds = {f.stem: read_csv_dated(f) for f in PRED.glob("*.csv")}
    rev = preds["b0rev"]
    hrev = hits(rev)
    flog = pd.read_csv(TAB / "fit_log.csv")
    master = load_master_checked().set_index(DATE)
    close = master["ohlcv_close"].dropna()

    rows = []
    for config in CONFIGS:
        for name in MODELS:
            for fs in FSETS:
                p = preds[f"{name}__{fs}_{config}"]
                if not np.array_equal(p[DATE].to_numpy(), rev[DATE].to_numpy()):
                    raise ValueError(f"{name} {fs} {config}: khác ngày test với b0rev")
                mr = metric_row(p, name)
                ob, oa, pv = mcnemar(hrev, hits(p))
                s = summarize(run_backtest(close, p.set_index(DATE)["p_hat"])).loc["strategy"]
                rows.append({"model": name, "fs": fs, "config": config, **mr, "d_vs_rev": mr["da"] - hrev.mean(),
                             "p_vs_rev": pv, "cum_return": s.cum_return, "sharpe": s.sharpe, "mdd": s.mdd,
                             "pct_long": s.pct_days_long})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "metrics.csv", index=False)
    T = t.set_index(["model", "fs", "config"])

    # Δ do up_today (f11 − f10) và Δ do On-chain (trong từng bộ)
    d_up, d_on = [], []
    for name in MODELS:
        for config in CONFIGS:
            a, b = preds[f"{name}__f10_{config}"], preds[f"{name}__f11_{config}"]
            ob, oa, pv = mcnemar(hits(a), hits(b))
            d_up.append({"model": name, "config": config, "da_f10": T.loc[(name, "f10", config)].da,
                         "da_f11": T.loc[(name, "f11", config)].da, "d_da": T.loc[(name, "f11", config)].da - T.loc[(name, "f10", config)].da,
                         "p": pv, "ll_f10": T.loc[(name, "f10", config)].log_loss, "ll_f11": T.loc[(name, "f11", config)].log_loss})
        for fs in FSETS:
            a, b = preds[f"{name}__{fs}_ohlcv"], preds[f"{name}__{fs}_ohlcv_onchain"]
            ob, oa, pv = mcnemar(hits(a), hits(b))
            d_on.append({"model": name, "fs": fs, "da_ohlcv": T.loc[(name, fs, "ohlcv")].da,
                         "da_onchain": T.loc[(name, fs, "ohlcv_onchain")].da,
                         "d_da": T.loc[(name, fs, "ohlcv_onchain")].da - T.loc[(name, fs, "ohlcv")].da, "p": pv,
                         "d_ll": T.loc[(name, fs, "ohlcv_onchain")].log_loss - T.loc[(name, fs, "ohlcv")].log_loss})
    d_up, d_on = pd.DataFrame(d_up), pd.DataFrame(d_on)
    d_up.to_csv(TAB / "delta_up_today.csv", index=False)
    d_on.to_csv(TAB / "delta_onchain.csv", index=False)

    # theo năm (OHLCV-only)
    yr = []
    years = rev[DATE].dt.year.to_numpy()
    for name in MODELS:
        for fs in FSETS:
            h = hits(preds[f"{name}__{fs}_ohlcv"])
            for y_ in np.unique(years):
                yr.append({"model": name, "fs": fs, "year": int(y_), "da": h[years == y_].mean(),
                           "rev": hrev[years == y_].mean()})
    yr = pd.DataFrame(yr)
    yr.to_csv(TAB / "by_year_ohlcv.csv", index=False)

    md = ["# Thêm đặc trưng up_today: chạy lại B1 (lr, lrfs) và B3 (SVM RBF) – Trâm", "",
          "Tạo bởi `analyses/up_today/src/compare_up_today.py`. `up_today` = 1 nếu giá đóng cửa ngày t cao hơn ngày t−1 "
          "(`features/src/features_ohlcv.py`). Mô hình, grid, 82 fold, chuẩn hoá lấy nguyên từ `pipeline`; "
          "tune bằng inner CV trong cửa sổ train, không dùng ngày test. Mốc so sánh: luật đảo chiều `b0rev` "
          f"đúng {pct(hrev.mean())}% trên {len(rev)} ngày test.", "",
          "**Vì sao thêm cột này:** `baseline_check` cho thấy quy luật duy nhất có ý nghĩa là hướng giá đảo chiều sau 1 ngày, "
          "tức nằm ở *dấu* của return hôm nay. 10 đặc trưng ban đầu chỉ đo *độ lớn* (return, biên độ, RSI, biến động), "
          "nên mô hình tuyến tính phải tự suy ra dấu từ độ lớn, và bị các ngày biến động mạnh lấn át.", ""]
    summary_at = len(md)

    def table(config):
        out = ["| Mô hình | Bộ đặc trưng | DA (%) | CI 95% | p (DA = 50%) | Log-loss | AUC | % đoán tăng | So với đảo chiều (điểm) | p |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for name in MODELS:
            for fs in FSETS:
                r = T.loc[(name, fs, config)]
                out.append(f"| {MODEL_LABEL[name]} | {'10 + up_today' if fs == 'f11' else '10 ban đầu'} | {pct(r.da)} | "
                           f"[{pct(r.da_ci_low,1)}; {pct(r.da_ci_high,1)}] | {pval(r.p_vs_50)} | {num(r.log_loss)} | "
                           f"{num(r.auc,3)} | {pct(r.pct_pred_up,1)} | {pct(r.d_vs_rev)} | {pval(r.p_vs_rev)} |")
        return out

    md += ["## 1. OHLCV-only (làm trước)", ""] + table("ohlcv") + [
           "", "Log-loss: đoán 50/50 mọi ngày cho 0,6931; thấp hơn là tốt hơn. \"So với đảo chiều\" = DA mô hình − DA luật, "
           "p theo McNemar chính xác trên các ngày hai bên khác nhau.", "",
           "## 2. OHLCV + On-chain (cùng mô hình, cùng protocol)", ""] + table("ohlcv_onchain") + [""]

    md += ["## 3. Tác dụng của up_today (11 − 10 đặc trưng, cùng mô hình, cùng cấu hình)", "",
           "| Mô hình | Cấu hình | DA 10 (%) | DA 11 (%) | ΔDA (điểm) | p McNemar | Log-loss 10 | Log-loss 11 |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {MODEL_LABEL[r.model]} | {r.config} | {pct(r.da_f10)} | {pct(r.da_f11)} | {pct(r.d_da)} | {pval(r.p)} | "
           f"{num(r.ll_f10)} | {num(r.ll_f11)} |" for r in d_up.itertuples()]
    md += ["", "## 4. Δ khi thêm On-chain, trên từng bộ đặc trưng OHLCV", "",
           "| Mô hình | Bộ OHLCV | DA OHLCV-only (%) | DA + On-chain (%) | ΔDA (điểm) | p McNemar | ΔLog-loss |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {MODEL_LABEL[r.model]} | {'10 + up_today' if r.fs == 'f11' else '10 ban đầu'} | {pct(r.da_ohlcv)} | {pct(r.da_onchain)} | "
           f"{pct(r.d_da)} | {pval(r.p)} | {num(r.d_ll)} |" for r in d_on.itertuples()]
    md += ["", "ΔLog-loss âm nghĩa là On-chain làm xác suất tốt hơn.", ""]

    # 5. tham số và cột được chọn
    md += ["## 5. Tuning đã chạy và cột được chọn", "",
           "Tham số chọn ở 14 lần tune (inner TimeSeriesSplit(5, gap 1) trong cửa sổ train, tiêu chí log-loss):", ""]
    for name in MODELS:
        for tag in [f"{fs}_{c}" for c in CONFIGS for fs in FSETS]:
            L = flog[(flog.model == name) & (flog.tag == tag) & flog.retuned]
            pcols = [c for c in L.columns if c.startswith("param_") and L[c].notna().any()]
            if not pcols:
                continue
            combos = L[pcols].apply(lambda r: ", ".join(f"{c.replace('param_', '')} = {r[c]:g}" for c in pcols), axis=1)
            vc = combos.value_counts()
            md.append(f"- {MODEL_LABEL[name]}, {tag.replace('f10_', '10 đặc trưng, ').replace('f11_', '11 đặc trưng, ')}: "
                      + "; ".join(f"{k} ({v} lần)" for k, v in vc.items()))
    md.append("")
    md += ["", "lrfs 10 đặc trưng + On-chain không có log tham số vì lần chạy đó (24 phút) được giữ lại từ lần chạy trước khi "
           "thêm phần ghi log; dự báo của nó vẫn dùng đầy đủ trong các bảng.", ""]
    lf = flog[(flog.model == "lrfs")]
    if len(lf):
        md += ["**lrfs giữ lại những cột nào** (tính trên 82 fold):", "",
               "| Cấu hình | Bộ đặc trưng | Số cột giữ TB | % fold giữ up_today | Cột hay được giữ nhất |", "|---|---|---|---|---|"]
        for tag in [f"{fs}_{c}" for c in CONFIGS for fs in FSETS]:
            L = lf[lf.tag == tag]
            if not len(L):
                md.append(f"| {tag.split('_', 1)[1]} | {'11' if tag.startswith('f11') else '10'} | không có log | – | – |")
                continue
            cnt = pd.Series("|".join(L.kept).split("|")).value_counts()
            top = ", ".join(f"{k} ({v}/{len(L)})" for k, v in cnt.head(4).items())
            md.append(f"| {tag.split('_', 1)[1]} | {'11' if tag.startswith('f11') else '10'} | {L.n_kept.mean():.1f} | "
                      f"{pct(L.kept_up_today.mean(),0) if tag.startswith('f11') else '–'} | {top} |")
        md.append("")

    # 6. theo năm
    piv = yr.pivot_table(index="year", columns=["model", "fs"], values="da")
    revy = yr.drop_duplicates("year").set_index("year").rev
    md += ["## 6. DA theo năm, OHLCV-only", "",
           "| Năm | Đảo chiều | " + " | ".join(f"{m} {'11' if f == 'f11' else '10'}" for m in MODELS for f in FSETS) + " |",
           "|---" * (2 + len(MODELS) * 2) + "|"]
    for y_ in piv.index:
        md.append(f"| {y_} | {pct(revy[y_],1)} | " + " | ".join(pct(piv.loc[y_, (m, f)], 1) for m in MODELS for f in FSETS) + " |")
    md.append("")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(revy.index, revy * 100, marker="o", color="k", label="Đảo chiều")
    for m, c in zip(MODELS, ["tab:blue", "tab:green", "tab:red"]):
        ax.plot(piv.index, piv[(m, "f10")] * 100, ls="--", color=c, alpha=0.6, label=f"{m} 10 đặc trưng")
        ax.plot(piv.index, piv[(m, "f11")] * 100, marker="o", color=c, label=f"{m} 11 đặc trưng")
    ax.axhline(50, color="grey", lw=0.8)
    ax.set_ylabel("DA (%)")
    ax.set_xlabel("năm")
    ax.set_title("DA theo năm, OHLCV-only")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "da_by_year.png", dpi=110)
    plt.close(fig)
    md += ["![theo năm](figures/da_by_year.png)", ""]

    # 7. backtest
    bh = summarize(run_backtest(close, rev.set_index(DATE)["p_hat"])).loc["buy_and_hold"]
    rv = summarize(run_backtest(close, rev.set_index(DATE)["p_hat"])).loc["strategy"]
    md += ["## 7. Backtest long/flat (phí 0,1%/lượt, tham khảo)", "",
           "| Mô hình | Bộ | Cấu hình | Lợi nhuận | Sharpe | MDD | % ngày giữ BTC |", "|---|---|---|---|---|---|---|",
           f"| Buy-and-hold | – | – | {pct(bh.cum_return,0)}% | {num(bh.sharpe,2)} | {pct(bh.mdd,0)}% | 100% |",
           f"| Luật đảo chiều | – | – | {pct(rv.cum_return,0)}% | {num(rv.sharpe,2)} | {pct(rv.mdd,0)}% | {pct(rv.pct_days_long,0)}% |"]
    md += [f"| {r.model} | {'11' if r.fs == 'f11' else '10'} | {r.config} | {pct(r.cum_return,0)}% | {num(r.sharpe,2)} | "
           f"{pct(r.mdd,0)}% | {pct(r.pct_long,0)}% |" for r in t.itertuples()]
    md += [""]

    # LR C = 1 cố định (nhanh, tính lại mỗi lần dựng báo cáo)
    data, cols, folds, _ = load_all()
    fixed = {}
    for fs in FSETS:
        q = lr_fixed_c(data, cols[(fs, "ohlcv")], folds)
        r = metric_row(q, "lr")
        fixed[fs] = (r["da"], r["log_loss"], r["p_vs_50"])
    pd.DataFrame({fs: {"da": v[0], "log_loss": v[1], "p_vs_50": v[2]} for fs, v in fixed.items()}).T.to_csv(
        TAB / "lr_fixed_C1_ohlcv.csv")

    # Tóm tắt
    def g(n, fs, c="ohlcv"):
        return T.loc[(n, fs, c)]
    best = max(MODELS, key=lambda n: g(n, "f11").da)
    up_lr = d_up[(d_up.model == "lr") & (d_up.config == "ohlcv")].iloc[0]
    up_b3 = d_up[(d_up.model == "b3") & (d_up.config == "ohlcv")].iloc[0]
    on11 = d_on[d_on.fs == "f11"]
    lf11 = lf[lf.tag == "f11_ohlcv"]
    summary = ["## Tóm tắt", "",
               f"1. **B1 (lr) OHLCV-only:** {pct(up_lr.da_f10)}% → **{pct(up_lr.da_f11)}%** khi thêm up_today "
               f"(ΔDA {pct(up_lr.d_da)} điểm, p = {pval(up_lr.p)}); log-loss {num(up_lr.ll_f10)} → {num(up_lr.ll_f11)}. "
               f"So với 50%: p = {pval(g('lr','f11').p_vs_50)}.",
               f"2. **lrfs OHLCV-only:** {pct(g('lrfs','f10').da)}% → {pct(g('lrfs','f11').da)}%; "
               f"lrfs giữ up_today ở {pct(lf11.kept_up_today.mean(),0)}% số fold." if len(lf11) else "",
               f"3. **B3 SVM OHLCV-only:** {pct(up_b3.da_f10)}% → {pct(up_b3.da_f11)}% (ΔDA {pct(up_b3.d_da)} điểm, p = {pval(up_b3.p)}). "
               "SVM RBF gần như không dùng được cột 0/1 này.",
               f"4. **So với luật đảo chiều ({pct(hrev.mean())}%):** mô hình tốt nhất sau khi thêm up_today là "
               f"{MODEL_LABEL[best]} với {pct(g(best,'f11').da)}%, "
               f"{'vẫn kém' if g(best,'f11').d_vs_rev < 0 else 'hơn'} luật {pct(abs(g(best,'f11').d_vs_rev))} điểm "
               f"(p = {pval(g(best,'f11').p_vs_rev)}).",
               (lambda L, cmin: f"__ITEM6__6. **Tuning của lr chọn C = {cmin:g} (giá trị nhỏ nhất của grid, regularization mạnh nhất) ở "
                f"{int((L.param_C == cmin).sum())}/{len(L)} lần tune.** Khi tín hiệu yếu, tiêu chí log-loss luôn ưu tiên thu hệ số về gần 0 "
                "để p̂ sát tỷ lệ tăng của train: log-loss tốt hơn nhưng phần lớn tác dụng của up_today lên DA bị triệt tiêu. "
                f"Cùng mô hình nhưng C = 1 cố định (không tune): DA {pct(fixed['f11'][0])}% (p = {pval(fixed['f11'][2])}), "
                f"log-loss {num(fixed['f11'][1])} (10 đặc trưng, C = 1: DA {pct(fixed['f10'][0])}%). "
                "Nhóm cần chốt metric chính (DA hay log-loss) trước, vì hai metric đang kéo tuning về hai hướng.")(
                   flog[(flog.model == "lr") & (flog.tag == "f11_ohlcv") & flog.retuned], min(load_model("lr")[1]["C"])),
               "5. **Thêm On-chain trên bộ 11 đặc trưng:** " + "; ".join(
                   f"{r.model} ΔDA {pct(r.d_da)} điểm (p = {pval(r.p)})" for r in on11.itertuples()) + ".",
               "", "**Ảnh hưởng tới cả nhóm:** `pipeline` lấy mọi cột mà `build_ohlcv_features` trả về, nên sau khi gộp PR này, "
               "lần chạy `run_all.py --stage e1` tiếp theo sẽ dùng 11 đặc trưng OHLCV cho mọi mô hình (B1–B5). "
               "Các file dự báo hiện có trong `pipeline/results/predictions/` vẫn là bản 10 đặc trưng cho tới khi chạy lại. "
               "Muốn giữ 10 đặc trưng thì dùng `FEATURES_10`.", ""]
    item6 = [x for x in summary if x.startswith("__ITEM6__")]
    summary = [x for x in summary if x != "" and not x.startswith("__ITEM6__")]
    pos = next(i for i, x in enumerate(summary) if x.startswith("5."))
    summary.insert(pos + 1, item6[0].replace("__ITEM6__", ""))
    md[summary_at:summary_at] = summary + [""]
    (BASE / "reports/up_today_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(summary))


def main():
    if "--report" not in sys.argv:
        run_models()
    write_report()


if __name__ == "__main__":
    main()
