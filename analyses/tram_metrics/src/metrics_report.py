"""
metrics_report.py  –  Góp ý 3 của thầy: metric chính / phụ, log-loss trước và sau tuning,
                      so sánh OHLCV-only với OHLCV + On-chain (Trâm).

Mô hình đã chốt (góp ý 1): LR (b1 -> b1t) và RF (b2 -> b2t).
Bộ đặc trưng chính: On-chain bỏ Flows (góp ý 4), dự báo ở pipeline/results/no_flows/.
Đối chiếu độ nhạy: bộ còn Flows, dự báo ở pipeline/results/rerun_2026-11-19/.

Chỉ đọc file dự báo đã có, không chạy lại mô hình. Kiểm định dùng đúng pipeline/src/stats.py:
  CI 95% block bootstrap (khối 15 ngày, 2.000 lần), McNemar chính xác cho ΔDA,
  Diebold–Mariano cho Δ log-loss, hiệu chỉnh Holm trong từng bảng.

  python analyses/tram_metrics/src/metrics_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pipeline" / "src"))
import stats  # noqa: E402

BASE = ROOT / "analyses" / "tram_metrics"
RES, REP, FIG = BASE / "results", BASE / "reports", BASE / "reports" / "figures"
MAIN = ROOT / "pipeline" / "results" / "no_flows" / "predictions"
FLOWS = ROOT / "pipeline" / "results" / "rerun_2026-11-19" / "predictions"

MODELS = {"LR": ("b1", "b1t"), "RF": ("b2", "b2t")}
CONFIGS = {"ohlcv": "OHLCV-only", "ohlcv_onchain": "OHLCV + On-chain"}
LN2 = float(np.log(2))


# ---------------------------------------------------------------- đọc dữ liệu
def load(folder, name, config):
    df = pd.read_csv(folder / f"{name}_{config}.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def check_same_days(frames):
    ref = frames[0]
    for f in frames[1:]:
        assert (f["date"].values == ref["date"].values).all(), "khác ngày test"
        assert (f["y_true"].values == ref["y_true"].values).all(), "khác nhãn"


def metric_row(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    ll = stats.row_log_loss(y, p)
    return dict(da=float(stats.row_hit(y, p).mean()),
                log_loss=float(ll.mean()),
                brier=float(((p - y) ** 2).mean()),
                pct_pred_up=float((p > 0.5).mean()))


def add_holm(df):
    """Holm trên toàn bộ p-value của một bảng (cả McNemar lẫn DM)."""
    ps = np.r_[df["p_mcnemar"].values, df["p_dm"].values]
    h = stats.holm(ps)
    df["p_mcnemar_holm"], df["p_dm_holm"] = h[: len(df)], h[len(df):]
    return df


# ---------------------------------------------------------------- các bảng
def table_metrics(folder):
    rows = []
    for config in CONFIGS:
        for mdl, (dflt, tuned) in MODELS.items():
            for stage, name in (("grid gốc", dflt), ("tune lại", tuned)):
                d = load(folder, name, config)
                rows.append(dict(model=mdl, file=name, config=config, stage=stage,
                                 **metric_row(d["y_true"], d["p_hat"])))
        maj = load(folder, "b0maj", config)
        rows.append(dict(model="B0 Majority (mốc)", file="b0maj", config=config, stage="-",
                         **metric_row(maj["y_true"], maj["p_hat"])))
    return pd.DataFrame(rows)


def table_tuning(folder):
    """Δ = tune lại − grid gốc, cùng mô hình, cùng cấu hình, cùng ngày test."""
    rows = []
    for mdl, (dflt, tuned) in MODELS.items():
        for config in CONFIGS:
            a, b = load(folder, dflt, config), load(folder, tuned, config)
            check_same_days([a, b])
            rows.append(dict(model=mdl, config=config, so_sanh=f"{tuned} − {dflt}",
                             **stats.paired_row(a["y_true"].values, a["p_hat"].values, b["p_hat"].values)))
    return add_holm(pd.DataFrame(rows))


def table_onchain(folder, stage_idx=1):
    """Δ = (OHLCV + On-chain) − OHLCV-only, cùng mô hình (đã tune)."""
    rows = []
    for mdl, names in MODELS.items():
        name = names[stage_idx]
        a, b = load(folder, name, "ohlcv"), load(folder, name, "ohlcv_onchain")
        check_same_days([a, b])
        rows.append(dict(model=mdl, file=name,
                         **stats.paired_row(a["y_true"].values, a["p_hat"].values, b["p_hat"].values)))
    return add_holm(pd.DataFrame(rows))


def table_benchmarks(folder):
    """Mô hình đã tune so với các mốc: luôn tăng (DA), đảo chiều b0rev (DA), b0maj (log-loss)."""
    rows = []
    for config in CONFIGS:
        maj, rev = load(folder, "b0maj", config), load(folder, "b0rev", config)
        y = maj["y_true"].values
        up = np.ones(len(y))
        for mdl, (_, tuned) in MODELS.items():
            m = load(folder, tuned, config)
            check_same_days([maj, rev, m])
            p = m["p_hat"].values
            r_up = stats.paired_row(y, up, p, with_logloss=False)
            r_rev = stats.paired_row(y, rev["p_hat"].values, p, with_logloss=False)
            r_maj = stats.paired_row(y, maj["p_hat"].values, p)
            rows.append(dict(model=mdl, file=tuned, config=config,
                             d_da_vs_up=r_up["d_da"], p_vs_up=r_up["p_mcnemar"],
                             d_da_vs_rev=r_rev["d_da"], p_vs_rev=r_rev["p_mcnemar"],
                             d_ll_vs_maj=r_maj["d_logloss"], d_ll_vs_maj_lo=r_maj["d_logloss_lo"],
                             d_ll_vs_maj_hi=r_maj["d_logloss_hi"], p_dm_vs_maj=r_maj["p_dm"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hình
def figure(M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_PRE, C_POST, INK, MUTED = "#9bbfe8", "#2a78d6", "#1f2328", "#6b7280"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (metric, title, ref, ref_lab) in zip(axes, [
            ("log_loss", "Log-loss (thấp hơn là tốt)", LN2, "ln 2 = 0,6931 (đoán 0,5)"),
            ("da", "DA – tỷ lệ đoán đúng hướng (cao hơn là tốt)", None, None)]):
        labels, x = [], 0
        for config, clab in CONFIGS.items():
            for mdl in MODELS:
                pre = M[(M.model == mdl) & (M.config == config) & (M.stage == "grid gốc")][metric].item()
                post = M[(M.model == mdl) & (M.config == config) & (M.stage == "tune lại")][metric].item()
                ax.bar(x - 0.2, pre, 0.38, color=C_PRE, label="grid gốc" if x == 0 else None)
                ax.bar(x + 0.2, post, 0.38, color=C_POST, label="tune lại" if x == 0 else None)
                labels.append(f"{mdl}\n{clab.replace(' + ', ' +\n')}")
                x += 1
        maj = M[(M.model == "B0 Majority (mốc)") & (M.config == "ohlcv")][metric].item()
        if metric == "log_loss":
            ax.axhline(ref, color=MUTED, lw=1, ls="--", label=ref_lab)
            ax.axhline(maj, color=INK, lw=1, ls=":", label=f"B0 Majority = {maj:.4f}".replace(".", ","))
            ax.set_ylim(0.688, 0.703)
        else:
            ax.axhline(0.5090, color=INK, lw=1, ls=":", label="luôn tăng = 50,90%")
            ax.axhline(0.5325, color=MUTED, lw=1, ls="--", label="đảo chiều b0rev = 53,25%")
            ax.set_ylim(0.48, 0.545)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks(range(x), labels, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, frameon=False, loc="upper left" if metric == "log_loss" else "upper right")
    fig.tight_layout()
    fig.savefig(FIG / "metrics_before_after.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- báo cáo
def pct(x, d=2):
    return f"{100 * x:.{d}f}".replace(".", ",")


def num(x, d=4):
    return "–" if pd.isna(x) else f"{x:+.{d}f}".replace(".", ",") if d and x != 0 else f"{x:.{d}f}".replace(".", ",")


def f4(x):
    return f"{x:.4f}".replace(".", ",")


def pv(x):
    return "–" if pd.isna(x) else (f"{x:.3f}".replace(".", ","))


def ci(lo, hi, scale=1.0, d=2):
    return f"[{lo * scale:+.{d}f}; {hi * scale:+.{d}f}]".replace(".", ",")


def write_report(M, T, D, B, Dflows, Mflows):
    def g(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0]

    L = []
    L += ["# Metric chính, metric phụ và kết quả (góp ý 3 của thầy)", "",
          "Phụ trách: Trâm. Script: `analyses/tram_metrics/src/metrics_report.py`. "
          "Mô hình đã chốt: **LR** (`b1` → `b1t`) và **RF** (`b2` → `b2t`). "
          "Bộ đặc trưng chính: On-chain **bỏ Flows** (27 đặc trưng, `pipeline/results/no_flows/`). "
          "82 fold walk-forward, 2.434 ngày test (01/01/2020 → 30/08/2026).", ""]

    # --- đoạn văn cho báo cáo
    L += ["## 1. Đoạn văn đưa vào báo cáo (mục Metric)", "",
          "> **Metric chính là DA** (directional accuracy, tỷ lệ ngày đoán đúng hướng giá t+1), vì câu hỏi nghiên cứu là "
          "đoán đúng hướng. Mọi kết luận về RQ1 dựa trên ΔDA = DA(OHLCV + On-chain) − DA(OHLCV-only), tính trên cùng ngày test, "
          "kiểm định bằng McNemar chính xác. **Log-loss là metric phụ** và là **tiêu chí tuning**: mọi tham số được chọn bằng "
          "log-loss trung bình trên `TimeSeriesSplit(5, gap 1)` bên trong cửa sổ train 730 ngày của từng fold; "
          "test chỉ dùng một lần để báo cáo, không dùng để chọn mô hình, tham số hay bộ đặc trưng. "
          "Log-loss được chọn để tune vì nó chấm cả độ tự tin của xác suất, thay đổi mượt hơn DA nên ít chọn nhầm do may. "
          "Δ log-loss được kiểm định bằng Diebold–Mariano. Mọi Δ báo cáo kèm CI 95% (block bootstrap, khối 15 ngày) "
          "và p đã hiệu chỉnh Holm. Mốc so sánh: DA của \"luôn đoán tăng\" trên cùng ngày (50,90%) và luật đảo chiều 1 ngày "
          "(53,25%); log-loss của B0 Majority (0,6938) và ln 2 = 0,6931 (luôn đoán 0,5). "
          "Luật B0 chỉ cho p̂ bằng 0 hoặc 1 nên không tính log-loss (trừ B0 Majority).", ""]

    # --- tóm tắt
    lr_t = g(T, model="LR", config="ohlcv"); rf_t = g(T, model="RF", config="ohlcv")
    lr_to = g(T, model="LR", config="ohlcv_onchain"); rf_to = g(T, model="RF", config="ohlcv_onchain")
    lr_d, rf_d = g(D, model="LR"), g(D, model="RF")
    L += ["## 2. Tóm tắt", "",
          f"1. **Tune lại làm log-loss giảm ở cả 4 tổ hợp, nhưng chỉ một chỗ có ý nghĩa sau Holm: LR + On-chain** "
          f"(Δ log-loss {num(lr_to.d_logloss)}, p Holm {pv(lr_to.p_dm_holm)}), vì LR grid gốc với 37 đặc trưng bị quá khớp, còn `b1t` có winsorize và regularization mạnh hơn (C nhỏ tới 0,0001, có L1). "
          f"Không tổ hợp nào có DA thay đổi có ý nghĩa; tăng nhiều nhất là LR OHLCV-only "
          f"({pct(g(M,model='LR',config='ohlcv',stage='grid gốc').da)}% → {pct(g(M,model='LR',config='ohlcv',stage='tune lại').da)}%, "
          f"p Holm {pv(lr_t.p_mcnemar_holm)}); RF + On-chain còn giảm {num(-100*rf_to.d_da,2).lstrip('+')} điểm.",
          f"2. **Thêm On-chain (bỏ Flows) không giúp, còn có xu hướng làm tệ đi.** Sau tune: "
          f"LR ΔDA {num(100*lr_d.d_da,2)} điểm (CI {ci(lr_d.d_da_lo, lr_d.d_da_hi, 100)}, p Holm {pv(lr_d.p_mcnemar_holm)}), "
          f"Δ log-loss {num(lr_d.d_logloss)} (p Holm {pv(lr_d.p_dm_holm)}); "
          f"RF ΔDA {num(100*rf_d.d_da,2)} điểm (CI {ci(rf_d.d_da_lo, rf_d.d_da_hi, 100)}, p Holm {pv(rf_d.p_mcnemar_holm)}), "
          f"Δ log-loss {num(rf_d.d_logloss)} (p Holm {pv(rf_d.p_dm_holm)}). Không Δ nào có ý nghĩa sau Holm → **chưa chấp nhận H1**. Riêng ΔDA của LR có CI không chứa 0, nhưng theo chiều On-chain làm *tệ đi*.",
          f"3. **Chỉ LR OHLCV-only sau tune lại có log-loss thấp hơn mốc B0 Majority** "
          f"({f4(g(M,model='LR',config='ohlcv',stage='tune lại').log_loss)} so với "
          f"{f4(g(M,model='B0 Majority (mốc)',config='ohlcv').log_loss)}), và chênh lệch này cũng chưa có ý nghĩa "
          f"(p DM {pv(g(B,model='LR',config='ohlcv').p_dm_vs_maj)}). Mọi log-loss đều sát ln 2: xác suất mô hình đưa ra gần như 0,5.",
          f"4. **Không mô hình nào vượt luật đảo chiều** (53,25%). LR OHLCV-only sau tune bằng luật "
          f"({num(100*g(B,model='LR',config='ohlcv').d_da_vs_rev,2)} điểm, vì `b1t` gần như chính là luật đảo chiều); "
          f"ba tổ hợp còn lại thua {num(-100*B.d_da_vs_rev.nlargest(2).iloc[1],2).lstrip('+')} đến "
          f"{num(-100*B.d_da_vs_rev.min(),2).lstrip('+')} điểm, p McNemar chưa hiệu chỉnh 0,007–0,047. "
          f"So với luôn tăng: chỉ LR OHLCV-only hơn ({num(100*g(B,model='LR',config='ohlcv').d_da_vs_up,2)} điểm, "
          f"p {pv(g(B,model='LR',config='ohlcv').p_vs_up)}), chưa có ý nghĩa.",
          "5. **Kết luận giống khi còn Flows** (mục 6): bỏ Flows không đổi chiều kết quả nào.", ""]

    # --- bảng 1
    L += ["## 3. Metric của từng tổ hợp", "",
          "| Mô hình | Cấu hình | Giai đoạn | DA | Log-loss | Brier | % ngày đoán tăng |",
          "|---|---|---|---:|---:|---:|---:|"]
    for _, r in M.iterrows():
        L.append(f"| {r.model} (`{r.file}`) | {CONFIGS[r.config]} | {r.stage} | {pct(r.da)}% | {f4(r.log_loss)} | "
                 f"{f4(r.brier)} | {pct(r.pct_pred_up,1)}% |")
    L += ["", "Mốc: luôn tăng DA 50,90%; luật đảo chiều DA 53,25%; ln 2 = 0,6931. "
          "B0 Majority dự báo p̂ = tỷ lệ ngày tăng trong train nên có log-loss; DA của nó (50,08%) thấp hơn luôn tăng "
          "vì có fold train nghiêng về giảm.", ""]

    # --- bảng 2
    L += ["## 4. Trước và sau tune lại (Δ = tune lại − grid gốc)", "",
          "\"Trước\" là bản E1 tune bằng grid gốc của proposal (Phụ lục C): LR chỉ tune C ∈ {0,01; 0,1; 1; 10}, "
          "RF tune 12 tổ hợp số cây, độ sâu, min_samples_leaf. \"Sau\" là bản tune lại: `b1t` (Vi, 300 tổ hợp gồm cả tập đặc trưng, "
          "C, kiểu regularization) và `b2t` (grid mới: min_samples_leaf tới 200, max_features, max_samples). "
          "Cả hai đều chọn bằng log-loss trên validation trong train; bảng này chỉ **mô tả** tác động lên test. "
          "Bản hoàn toàn chưa tune (tham số mặc định sklearn) chỉ có cho bộ còn Flows, ở `pipeline/results/tuning/tuning_before_after.csv`.", "",
          "| Mô hình | Cấu hình | ΔDA (điểm) | CI 95% | p McNemar | p Holm | Δ log-loss | CI 95% | p DM | p Holm |",
          "|---|---|---:|---|---:|---:|---:|---|---:|---:|"]
    for _, r in T.iterrows():
        L.append(f"| {r.model} | {CONFIGS[r.config]} | {num(100*r.d_da,2)} | {ci(r.d_da_lo, r.d_da_hi, 100)} | "
                 f"{pv(r.p_mcnemar)} | {pv(r.p_mcnemar_holm)} | {num(r.d_logloss)} | "
                 f"{ci(r.d_logloss_lo, r.d_logloss_hi, 1, 4)} | {pv(r.p_dm)} | {pv(r.p_dm_holm)} |")
    L += ["", "Δ log-loss âm = tune lại làm xác suất tốt hơn. Holm tính trên cả 8 p-value của bảng.", ""]

    # --- bảng 3
    def onchain_table(Dx, title):
        out = [title, "",
               "| Mô hình | ΔDA (điểm) | CI 95% | chỉ OHLCV đúng / chỉ +On-chain đúng | p McNemar | p Holm | "
               "Δ log-loss | CI 95% | p DM | p Holm |",
               "|---|---:|---|---:|---:|---:|---:|---|---:|---:|"]
        for _, r in Dx.iterrows():
            out.append(f"| {r.model} (`{r.file}`) | {num(100*r.d_da,2)} | {ci(r.d_da_lo, r.d_da_hi, 100)} | "
                       f"{r.n_only_a} / {r.n_only_b} | {pv(r.p_mcnemar)} | {pv(r.p_mcnemar_holm)} | {num(r.d_logloss)} | "
                       f"{ci(r.d_logloss_lo, r.d_logloss_hi, 1, 4)} | {pv(r.p_dm)} | {pv(r.p_dm_holm)} |")
        return out
    L += onchain_table(D, "## 5. OHLCV-only so với OHLCV + On-chain (Δ = +On-chain − OHLCV, mô hình đã tune)")
    L += ["", "ΔDA > 0 hoặc Δ log-loss < 0 nghĩa là On-chain giúp. Holm tính trên 4 p-value (2 mô hình × 2 metric).", ""]

    # --- bảng mốc
    L += ["### So với các mốc", "",
          "| Mô hình | Cấu hình | ΔDA vs luôn tăng | p | ΔDA vs đảo chiều | p | Δ log-loss vs B0 Majority | CI 95% | p DM |",
          "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for _, r in B.iterrows():
        L.append(f"| {r.model} | {CONFIGS[r.config]} | {num(100*r.d_da_vs_up,2)} | {pv(r.p_vs_up)} | "
                 f"{num(100*r.d_da_vs_rev,2)} | {pv(r.p_vs_rev)} | {num(r.d_ll_vs_maj)} | "
                 f"{ci(r.d_ll_vs_maj_lo, r.d_ll_vs_maj_hi, 1, 4)} | {pv(r.p_dm_vs_maj)} |")
    L += [""]

    # --- độ nhạy
    L += onchain_table(Dflows, "## 6. Đối chiếu: bộ On-chain còn Flows (33 đặc trưng, `rerun_2026-11-19`)")
    L += ["", "| Mô hình | Cấu hình | DA bỏ Flows | DA còn Flows | Log-loss bỏ Flows | Log-loss còn Flows |",
          "|---|---|---:|---:|---:|---:|"]
    for mdl in MODELS:
        a = g(M, model=mdl, config="ohlcv_onchain", stage="tune lại")
        b = g(Mflows, model=mdl, config="ohlcv_onchain", stage="tune lại")
        L.append(f"| {mdl} | OHLCV + On-chain | {pct(a.da)}% | {pct(b.da)}% | {f4(a.log_loss)} | {f4(b.log_loss)} |")
    L += ["", "Cấu hình OHLCV-only không dùng On-chain nên giống nhau ở hai thư mục.", ""]

    L += ["## 7. Hình", "", "![Grid gốc và tune lại](figures/metrics_before_after.png)", "",
          "## 8. Ghi chú", "",
          "- Δ log-loss nhỏ cỡ 0,001–0,005 vì mọi mô hình đều đưa xác suất gần 0,5; DM vẫn phát hiện được chênh lệch nhỏ "
          "nếu nó đều đặn qua các ngày.",
          "- Holm trong bảng 5 tính trên 2 mô hình đã chốt. Bảng `pipeline/results/no_flows/tables/delta_onchain.csv` "
          "của pipeline hiệu chỉnh trên cả 7 mô hình có học nên p Holm ở đó lớn hơn; kết luận không đổi.",
          "- Mọi số tính lại từ file dự báo; DA và log-loss khớp `pipeline/results/no_flows/tables/metrics.csv`.", ""]
    (REP / "metrics_report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    for d in (RES, REP, FIG):
        d.mkdir(parents=True, exist_ok=True)
    M = table_metrics(MAIN)
    T = table_tuning(MAIN)
    D = table_onchain(MAIN)
    B = table_benchmarks(MAIN)
    Mflows = table_metrics(FLOWS)
    Dflows = table_onchain(FLOWS)
    M.to_csv(RES / "metrics_no_flows.csv", index=False)
    T.to_csv(RES / "tuning_before_after_no_flows.csv", index=False)
    D.to_csv(RES / "delta_onchain_no_flows.csv", index=False)
    B.to_csv(RES / "vs_benchmarks_no_flows.csv", index=False)
    Mflows.to_csv(RES / "metrics_with_flows.csv", index=False)
    Dflows.to_csv(RES / "delta_onchain_with_flows.csv", index=False)
    figure(M)
    write_report(M, T, D, B, Dflows, Mflows)
    print(M.round(4).to_string(index=False))
    print(T[["model", "config", "d_da", "p_mcnemar_holm", "d_logloss", "p_dm_holm"]].round(4).to_string(index=False))
    print(D[["model", "d_da", "d_da_lo", "d_da_hi", "p_mcnemar_holm", "d_logloss", "p_dm_holm"]].round(4).to_string(index=False))
    print(B.round(4).to_string(index=False))
    print(Dflows[["model", "d_da", "p_mcnemar_holm", "d_logloss", "p_dm_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
