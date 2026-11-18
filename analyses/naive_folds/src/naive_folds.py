"""
naive_folds.py  –  Trâm: baseline ngây thơ LUÔN TĂNG / LUÔN GIẢM trên từng fold, đặt cạnh các baseline đã tuning

Câu hỏi của nhóm: "nếu lúc nào cũng dự đoán tăng thì kết quả là bao nhiêu?" Ví dụ một đoạn có 60% ngày tăng thì
luôn tăng đúng 60%. Script này tính:
  - luôn tăng (p̂ = 1) và luôn giảm (p̂ = 0) trên từng fold của walk-forward (82 fold × 30 ngày, 2020 – 08/2026)
    và trên đoạn test của simple_split (2024 – 08/2026) chia thành khối 30 ngày;
  - DA của các baseline đã tuning trên cùng từng fold, và mô hình thắng luôn tăng / luôn giảm ở bao nhiêu fold;
  - phân phối tỷ lệ ngày tăng giữa các fold so với mức kỳ vọng nếu ngày tăng/giảm là ngẫu nhiên (nhị thức);
  - mô hình có "đi theo" xu hướng của fold không (tương quan % đoán tăng với tỷ lệ tăng thật của fold).

Chỉ đọc file dự báo đã có trong repo, không chạy lại mô hình:
  pipeline/results/predictions/{model}_{config}.csv         (b0*, b1, b2t, b3, b4, b5)
  pipeline/results/b1t/predictions/b1t_logloss.csv   (B1 tune đầy đủ của Vi, OHLCV)
  pipeline/results/b1t/e1/ohlcv_onchain/predictions/b1t_ohlcv_onchain.csv
  analyses/simple_split/reports/predictions_test.csv

Chạy từ thư mục gốc repo:   python analyses/naive_folds/src/naive_folds.py
Đầu ra: analyses/naive_folds/results/*.csv, analyses/naive_folds/reports/naive_folds.md, reports/figures/*.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, chi2

ROOT = Path(__file__).resolve().parents[3]
NP = ROOT / "pipeline/results"
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
FIG = BASE / "reports/figures"
BLOCK = 30                                      # độ dài khối cho simple_split, bằng độ dài fold test

# Mô hình walk-forward: (nhãn, file). OHLCV-only trước; bản có On-chain để đối chiếu.
WF_OHLCV = {
    "B0 Majority (train)": NP / "predictions/b0maj_ohlcv.csv",
    "B0 Persistence": NP / "predictions/b0pers_ohlcv.csv",
    "B0 Đảo chiều": NP / "predictions/b0rev_ohlcv.csv",
    "B1 LR (E1)": NP / "predictions/b1_ohlcv.csv",
    "B1 LR tune đầy đủ (Vi)": NP / "b1t/predictions/b1t_logloss.csv",
    "B2 RF tuned (b2t)": NP / "predictions/b2t_ohlcv.csv",
    "B3 SVM RBF": NP / "predictions/b3_ohlcv.csv",
    "B4 XGBoost": NP / "predictions/b4_ohlcv.csv",
    "B5 LSTM": NP / "predictions/b5_ohlcv.csv",
}
WF_ONCHAIN = {
    "B1 LR (E1)": NP / "predictions/b1_ohlcv_onchain.csv",
    "B1 LR tune đầy đủ (Vi)": NP / "b1t/e1/ohlcv_onchain/predictions/b1t_ohlcv_onchain.csv",
    "B2 RF tuned (b2t)": NP / "predictions/b2t_ohlcv_onchain.csv",
    "B3 SVM RBF": NP / "predictions/b3_ohlcv_onchain.csv",
    "B4 XGBoost": NP / "predictions/b4_ohlcv_onchain.csv",
    "B5 LSTM": NP / "predictions/b5_ohlcv_onchain.csv",
}
SIMPLE = ROOT / "analyses/simple_split/reports/predictions_test.csv"
SIMPLE_MODELS = ["Majority", "Persistence", "Đảo chiều", "LR sau tune", "SVM RBF sau tune", "RF sau tune",
                 "LSTM sau tune"]


# ---------------------------------------------------------------------------
# Đọc và ghép
# ---------------------------------------------------------------------------
def load_wf(files: dict) -> pd.DataFrame:
    """Bảng ngày × mô hình: y, fold và dự báo 0/1 của từng mô hình (p̂ > 0,5)."""
    base = None
    for name, f in files.items():
        p = pd.read_csv(f, parse_dates=["date"])
        if base is None:
            base = p[["date", "fold", "y_true"]].rename(columns={"y_true": "y"}).copy()
        elif not (p["date"].to_numpy() == base["date"].to_numpy()).all() or not (p["y_true"].to_numpy() == base["y"].to_numpy()).all():
            raise ValueError(f"{f.name}: khác ngày hoặc nhãn so với file đầu")
        base[name] = (p["p_hat"].to_numpy() > 0.5).astype(int)
    return base


def load_simple() -> pd.DataFrame:
    s = pd.read_csv(SIMPLE, parse_dates=["date"])
    out = s[["date", "y"]].copy()
    out["fold"] = np.arange(len(s)) // BLOCK                      # khối 30 ngày liên tiếp
    for m in SIMPLE_MODELS:
        out[m] = (s[m].to_numpy() > 0.5).astype(int)
    return out


# ---------------------------------------------------------------------------
# Tính theo fold
# ---------------------------------------------------------------------------
def per_fold(df: pd.DataFrame, models: list) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby("fold"):
        r = {"fold": int(k), "start": g["date"].min().date(), "end": g["date"].max().date(), "n": len(g),
             "up_share": g["y"].mean()}
        r["always_up"] = r["up_share"]
        r["always_down"] = 1 - r["up_share"]
        for m in models:
            r[m] = (g[m] == g["y"]).mean()
            r[f"{m} | % đoán tăng"] = g[m].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, pf: pd.DataFrame, models: list) -> pd.DataFrame:
    """Toàn kỳ + đếm fold. McNemar chính xác so với luôn tăng: các ngày mô hình đoán giảm."""
    rows = []
    y = df["y"].to_numpy()
    up_hits = (y == 1).astype(int)
    for name, hits in [("Luôn tăng", up_hits), ("Luôn giảm", 1 - up_hits)]:
        k, n = int(hits.sum()), len(hits)
        rows.append({"model": name, "da": k / n, "ci_low": binomtest(k, n).proportion_ci().low,
                     "ci_high": binomtest(k, n).proportion_ci().high, "p_vs_50": binomtest(k, n, .5).pvalue,
                     "d_vs_up": k / n - up_hits.mean(), "p_vs_up": np.nan, "folds_beat_up": np.nan,
                     "folds_beat_down": np.nan, "folds_beat_both": np.nan, "corr_lean": np.nan, "corr_da": np.nan,
                     "pct_up": 1.0 if name == "Luôn tăng" else 0.0})
    for m in models:
        hits = (df[m].to_numpy() == y).astype(int)
        k, n = int(hits.sum()), len(hits)
        only_m = int(((hits == 1) & (up_hits == 0)).sum())          # ngày mô hình đúng, luôn tăng sai
        only_up = int(((hits == 0) & (up_hits == 1)).sum())
        p_up = binomtest(only_m, only_m + only_up, .5).pvalue if only_m + only_up else 1.0
        rows.append({"model": m, "da": k / n, "ci_low": binomtest(k, n).proportion_ci().low,
                     "ci_high": binomtest(k, n).proportion_ci().high, "p_vs_50": binomtest(k, n, .5).pvalue,
                     "d_vs_up": k / n - up_hits.mean(), "p_vs_up": p_up,
                     "folds_beat_up": int((pf[m] > pf["always_up"]).sum()),
                     "folds_beat_down": int((pf[m] > pf["always_down"]).sum()),
                     "folds_beat_both": int((pf[m] > pf[["always_up", "always_down"]].max(axis=1)).sum()),
                     "corr_lean": np.corrcoef(pf[f"{m} | % đoán tăng"], pf["up_share"])[0, 1]
                     if pf[f"{m} | % đoán tăng"].std() > 0 else np.nan,
                     "corr_da": np.corrcoef(pf[m], pf["up_share"])[0, 1],
                     "pct_up": df[m].mean()})
    return pd.DataFrame(rows)


def dispersion(pf: pd.DataFrame, p_all: float):
    """Tỷ lệ ngày tăng giữa các fold có phân tán hơn nhị thức (ngày độc lập, xác suất p_all) không."""
    k = (pf["up_share"] * pf["n"]).round().astype(int)
    stat = (((k - pf["n"] * p_all) ** 2) / (pf["n"] * p_all * (1 - p_all))).sum()
    dof = len(pf) - 1
    full = pf[pf["n"] == BLOCK]
    exp_ge60 = 1 - binom.cdf(np.ceil(0.6 * BLOCK) - 1, BLOCK, p_all)
    exp_le40 = binom.cdf(np.floor(0.4 * BLOCK), BLOCK, p_all)
    return {"chi2": stat, "dof": dof, "p": 1 - chi2.cdf(stat, dof), "ratio": stat / dof,
            "n_ge60": int((full["up_share"] >= 0.6).sum()), "exp_ge60": exp_ge60 * len(full),
            "n_le40": int((full["up_share"] <= 0.4).sum()), "exp_le40": exp_le40 * len(full),
            "min": pf["up_share"].min(), "max": pf["up_share"].max(), "n_full": len(full)}


def by_year(df: pd.DataFrame, models: list) -> pd.DataFrame:
    y = df["date"].dt.year
    out = pd.DataFrame({"Luôn tăng": df.groupby(y)["y"].mean(), "Luôn giảm": 1 - df.groupby(y)["y"].mean()})
    for m in models:
        out[m] = (df[m] == df["y"]).groupby(y).mean()
    out.index.name = "year"
    return out


# ---------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # thứ tự cố định: xanh, cam, aqua


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_folds(pf, best, fname, title):
    import matplotlib.pyplot as plt
    pf = pf[pf["n"] == BLOCK].reset_index(drop=True)     # bỏ fold cuối 4 ngày: % trên 4 ngày gây hiểu nhầm
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(pf))
    ax.bar(x, pf["always_up"] * 100, width=0.8, color="#c9c8c1", label="Luôn tăng (= tỷ lệ ngày tăng của fold)")
    ax.plot(x, pf["B0 Đảo chiều"] * 100 if "B0 Đảo chiều" in pf else pf["Đảo chiều"] * 100, color=C2, lw=1.4,
            label="Đảo chiều")
    ax.plot(x, pf[best] * 100, color=C1, lw=1.6, label=best)
    ax.axhline(50, color=INK2, lw=0.8, ls="--")
    years = pd.to_datetime(pf["start"]).dt.year
    ticks = [i for i in range(len(pf)) if i == 0 or years.iloc[i] != years.iloc[i - 1]]
    ax.set_xticks(ticks, [str(years.iloc[i]) for i in ticks])
    ax.set_ylim(20, 85)
    ax.set_ylabel("% ngày đoán đúng trong fold", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120)
    plt.close(fig)


def fig_hist(pf, p_all, fname):
    import matplotlib.pyplot as plt
    full = pf[pf["n"] == BLOCK]
    k = (full["up_share"] * BLOCK).round().astype(int)
    ks = np.arange(BLOCK + 1)
    obs = np.bincount(k, minlength=BLOCK + 1)
    exp = binom.pmf(ks, BLOCK, p_all) * len(full)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(ks / BLOCK * 100, obs, width=100 / BLOCK * 0.8, color=C1, label=f"Thực tế ({len(full)} fold)")
    ax.plot(ks / BLOCK * 100, exp, color=C2, lw=2, marker="o", ms=4,
            label=f"Kỳ vọng nếu mỗi ngày tăng ngẫu nhiên với xác suất {p_all*100:.1f}%".replace(".", ","))
    ax.axvline(50, color=INK2, lw=0.8, ls="--")
    ax.set_xlim(15, 85)
    ax.set_xlabel("Tỷ lệ ngày tăng trong fold = % đúng của luôn tăng", color=INK2, fontsize=9)
    ax.set_ylabel("Số fold", color=INK2, fontsize=9)
    ax.set_title("Luôn tăng đúng bao nhiêu % trong một fold 30 ngày?", fontsize=10, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120)
    plt.close(fig)


def fig_lean(pf, models, fname):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.4), sharey=True)
    for ax, m, c in zip(np.atleast_1d(axes), models, [C1, C2, C3]):
        ax.scatter(pf["up_share"] * 100, pf[f"{m} | % đoán tăng"] * 100, s=22, color=c, edgecolor="white", lw=0.6)
        r = np.corrcoef(pf["up_share"], pf[f"{m} | % đoán tăng"])[0, 1]
        ax.set_title(f"{m}  (r = {r:.2f})".replace(".", ","), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Tỷ lệ ngày tăng thật của fold (%)", color=INK2, fontsize=8)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("% ngày mô hình đoán tăng", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def pct(x, d=2):
    return "–" if pd.isna(x) else f"{x*100:.{d}f}".replace(".", ",")


def num(x, d=2):
    return "–" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def pval(x):
    return "–" if pd.isna(x) else ("<0,001" if x < 0.001 else num(x, 3))


def table_summary(s, n_folds):
    out = ["| Mô hình | DA toàn kỳ (%) | CI 95% | So với luôn tăng (điểm) | p (McNemar, so với luôn tăng) | "
           f"Fold thắng luôn tăng (/{n_folds}) | Fold thắng luôn giảm | Fold thắng cả hai | % ngày đoán tăng |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        out.append(f"| {r.model} | {pct(r.da)} | [{pct(r.ci_low,1)}; {pct(r.ci_high,1)}] | {pct(r.d_vs_up)} | {pval(r.p_vs_up)} | "
                   f"{'–' if pd.isna(r.folds_beat_up) else int(r.folds_beat_up)} | "
                   f"{'–' if pd.isna(r.folds_beat_down) else int(r.folds_beat_down)} | "
                   f"{'–' if pd.isna(r.folds_beat_both) else int(r.folds_beat_both)} | {pct(r.pct_up,1)} |")
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    wf = load_wf(WF_OHLCV)
    m_wf = list(WF_OHLCV)
    pf = per_fold(wf, m_wf)
    s = summary(wf, pf, m_wf)
    p_all = wf["y"].mean()
    disp = dispersion(pf, p_all)
    yr = by_year(wf, m_wf)

    wf_on = load_wf(WF_ONCHAIN)
    if not (wf_on["date"].to_numpy() == wf["date"].to_numpy()).all():
        raise ValueError("Bản On-chain khác ngày test")
    m_on = list(WF_ONCHAIN)
    pf_on = per_fold(wf_on, m_on)
    s_on = summary(wf_on, pf_on, m_on)

    ss = load_simple()
    pf_s = per_fold(ss, SIMPLE_MODELS)
    s_s = summary(ss, pf_s, SIMPLE_MODELS)
    disp_s = dispersion(pf_s, ss["y"].mean())

    for name, t in [("folds_walkforward_ohlcv", pf), ("summary_walkforward_ohlcv", s),
                    ("folds_walkforward_onchain", pf_on), ("summary_walkforward_onchain", s_on),
                    ("by_year_walkforward_ohlcv", yr), ("blocks_simple_split", pf_s), ("summary_simple_split", s_s)]:
        t.to_csv(RES / f"{name}.csv", index=name.startswith("by_year"))

    learned = [m for m in m_wf if not m.startswith("B0")]
    best = max(learned, key=lambda m: s.set_index("model").loc[m, "da"])
    fig_folds(pf, best, "folds_walkforward.png",
              "Luôn tăng, đảo chiều và mô hình có học tốt nhất trên từng fold (walk-forward, OHLCV-only)")
    fig_hist(pf, p_all, "hist_up_share.png")
    fig_lean(pf, ["B0 Majority (train)", "B1 LR tune đầy đủ (Vi)", "B3 SVM RBF"], "lean_vs_up_share.png")

    S = s.set_index("model")
    up, down = S.loc["Luôn tăng"], S.loc["Luôn giảm"]
    rev = S.loc["B0 Đảo chiều"]
    nf = len(pf)
    full = pf[pf["n"] == BLOCK]
    fmax, fmin = full.loc[full["up_share"].idxmax()], full.loc[full["up_share"].idxmin()]
    beat_up_counts = {m: int(S.loc[m, "folds_beat_up"]) for m in learned}

    md = ["# Baseline ngây thơ: luôn tăng / luôn giảm trên từng fold – Trâm", "",
          "Tạo bởi `analyses/naive_folds/src/naive_folds.py`. Chỉ đọc các file dự báo đã có trong repo, không chạy lại mô hình. "
          "\"Luôn tăng\" đoán tăng mọi ngày (p̂ = 1), \"luôn giảm\" đoán giảm mọi ngày (p̂ = 0); hai luật này không học gì, "
          "nên % đúng của luôn tăng **chính bằng tỷ lệ ngày tăng** của đoạn đang xét.", ""]
    md += ["## Tóm tắt", "",
           f"1. **Toàn kỳ 2020 – 08/2026 (2.434 ngày):** luôn tăng đúng **{pct(up.da)}%**, luôn giảm **{pct(down.da)}%**. "
           f"Tỷ lệ ngày tăng của BTC chỉ hơn 50% một chút, nên luôn tăng không phải mốc cao.",
           f"2. **Nhưng trên từng fold 30 ngày, luôn tăng dao động từ {pct(disp['min'],0)}% đến {pct(disp['max'],0)}%.** "
           f"Có {disp['n_ge60']}/{disp['n_full']} fold luôn tăng đúng ≥ 60% (ví dụ fold {int(fmax.fold)}, "
           f"{fmax.start} → {fmax.end}: {pct(fmax.up_share,0)}%) và {disp['n_le40']} fold chỉ ≤ 40% "
           f"(fold {int(fmin.fold)}, {fmin.start} → {fmin.end}: {pct(fmin.up_share,0)}%). "
           "Một con số \"60%\" trên một đoạn ngắn **không chứng minh mô hình giỏi**: nó có thể chỉ do đoạn đó rơi vào thị trường tăng.",
           f"3. **Mức dao động này gần đúng như kỳ vọng nếu mỗi ngày tăng/giảm ngẫu nhiên** (kiểm định χ²: p = {pval(disp['p'])}; "
           f"kỳ vọng {num(disp['exp_ge60'],1)} fold ≥ 60%, thực tế {disp['n_ge60']}). Tức là các fold \"tăng nhiều\" phần lớn là may rủi "
           "của 30 ngày, không phải xu hướng mà một mô hình có thể bám theo.",
           f"4. **Mô hình đã tuning so với luôn tăng (toàn kỳ, OHLCV-only):** tốt nhất là {best} với {pct(S.loc[best,'da'])}% "
           f"(+{pct(S.loc[best,'d_vs_up'])} điểm, p = {pval(S.loc[best,'p_vs_up'])}). Luật đảo chiều {pct(rev.da)}% "
           f"(+{pct(rev.d_vs_up)} điểm, p = {pval(rev.p_vs_up)}). Các mô hình còn lại chênh luôn tăng từ "
           f"{pct(min(S.loc[m,'d_vs_up'] for m in learned))} đến {pct(max(S.loc[m,'d_vs_up'] for m in learned))} điểm.",
           f"5. **Theo fold:** mô hình có học thắng luôn tăng ở {min(beat_up_counts.values())}–{max(beat_up_counts.values())}/{nf} fold; "
           f"luật đảo chiều thắng ở {int(rev.folds_beat_up)}/{nf}. Thắng **cả** luôn tăng lẫn luôn giảm (tức hơn cả luật ngây thơ đoán đúng "
           f"hướng đa số của fold khi đã biết trước) chỉ ở {int(min(S.loc[m,'folds_beat_both'] for m in learned))}–"
           f"{int(max(S.loc[m,'folds_beat_both'] for m in learned))} fold.",
           f"6. **Không mô hình nào \"bám\" được xu hướng của fold, vì xu hướng đó không kéo dài:** tỷ lệ tăng của một fold gần như "
           f"không liên quan tới fold trước (tự tương quan {num(pf['up_share'].autocorr(1))}). Majority, vốn học tỷ lệ tăng của 730 ngày "
           f"trước, chỉ thắng luôn tăng ở {int(S.loc['B0 Majority (train)','folds_beat_up'])}/{nf} fold. "
           f"B3 SVM giống luôn tăng nhất: đoán tăng {pct(S.loc['B3 SVM RBF','pct_up'],0)}% số ngày, DA của nó cao lên ở fold tăng nhiều "
           f"(r = {num(S.loc['B3 SVM RBF','corr_da'])}) và chỉ thắng luôn tăng ở {int(S.loc['B3 SVM RBF','folds_beat_up'])}/{nf} fold. "
           f"Hai mô hình tốt nhất (B1 tune đầy đủ, luật đảo chiều) lại đoán tăng *ít* hơn ở fold tăng nhiều "
           f"(r = {num(S.loc['B1 LR tune đầy đủ (Vi)','corr_lean'])} và {num(rev.corr_lean)}), vì chúng dựa vào quy luật đảo chiều: "
           "fold có nhiều ngày tăng thì có nhiều ngày \"hôm nay tăng\", nên luật đoán \"mai giảm\" nhiều hơn.",
           f"7. **simple_split (test 2024 – 08/2026, {len(ss)} ngày, {len(pf_s)} khối 30 ngày):** luôn tăng {pct(S_s(s_s,'Luôn tăng'))}%, "
           f"luôn giảm {pct(S_s(s_s,'Luôn giảm'))}%; khối 30 ngày dao động {pct(disp_s['min'],0)}–{pct(disp_s['max'],0)}%. "
           "Không mô hình đã tuning nào hơn luôn tăng có ý nghĩa thống kê (bảng ở mục 4).",
           "",
           "**Cách đọc cho nhóm:** mốc so sánh đúng của một mô hình là **luôn tăng trên cùng đoạn dữ liệu**, không phải 50%. "
           "Một mô hình chỉ có giá trị khi hơn luôn tăng (và luôn giảm) trên cùng các ngày, kiểm định bằng McNemar, "
           "và hơn ở nhiều fold chứ không chỉ một vài fold may mắn.", ""]

    md += ["## 1. Toàn kỳ walk-forward (82 fold, 2.434 ngày), OHLCV-only", ""] + table_summary(s, nf) + [
           "", "\"Fold thắng\" = DA của mô hình trong fold cao hơn hẳn luật ngây thơ (bằng nhau không tính). "
           "\"Thắng cả hai\" = cao hơn max(luôn tăng, luôn giảm) của fold đó; max này là luật biết trước hướng đa số của fold, "
           "không dùng được trong thực tế, chỉ là mốc trên.", "",
           "![theo fold](figures/folds_walkforward.png)", "",
           "Hình chỉ vẽ 81 fold đủ 30 ngày. Cột xám là luôn tăng ở từng fold, tức tỷ lệ ngày tăng của fold. Các đường dao động quanh 50% và không bám sát cột xám: "
           "mô hình không nắm được fold nào sẽ tăng nhiều.", ""]

    md += ["## 2. Luôn tăng đúng bao nhiêu trong một fold?", "",
           f"Tỷ lệ ngày tăng trên {disp['n_full']} fold đủ 30 ngày: thấp nhất {pct(disp['min'],1)}%, cao nhất {pct(disp['max'],1)}%. "
           f"Nếu mỗi ngày tăng ngẫu nhiên với xác suất {pct(p_all,1)}% thì 30 ngày có độ lệch chuẩn khoảng "
           f"{num(np.sqrt(p_all*(1-p_all)/BLOCK)*100,1)} điểm, nên một fold 60% hay 40% là bình thường.", "",
           "| | Thực tế | Kỳ vọng nếu ngẫu nhiên |", "|---|---|---|",
           f"| Số fold luôn tăng đúng ≥ 60% | {disp['n_ge60']} | {num(disp['exp_ge60'],1)} |",
           f"| Số fold luôn tăng đúng ≤ 40% | {disp['n_le40']} | {num(disp['exp_le40'],1)} |",
           f"| Độ phân tán so với nhị thức (χ²/bậc tự do) | {num(disp['ratio'])} | 1,00 |",
           f"| p (χ², {disp['dof']} bậc tự do) | {pval(disp['p'])} | |", "",
           "![phân phối](figures/hist_up_share.png)", ""]

    md += ["## 3. Theo năm (OHLCV-only)", "",
           "| Năm | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    md += [f"| {y_} | " + " | ".join(pct(v, 1) for v in r) + " |" for y_, r in yr.iterrows()]
    md += ["", "Năm 2020 thị trường tăng mạnh nên luôn tăng đúng 57,7%, năm 2022 giảm mạnh nên luôn giảm đúng 54,2%. "
           "Mô hình nào lệch về đoán tăng sẽ trông tốt ở 2020 và tệ ở 2022, dù không dự báo được gì.", "",
           "## 4. Mô hình có \"đi theo\" xu hướng của fold không?", "",
           "| Mô hình | Tương quan: % đoán tăng ↔ tỷ lệ tăng thật của fold | Tương quan: DA ↔ tỷ lệ tăng thật của fold |",
           "|---|---|---|"]
    md += [f"| {m} | {num(S.loc[m,'corr_lean'])} | {num(S.loc[m,'corr_da'])} |" for m in m_wf]
    md += ["", f"Tự tương quan của tỷ lệ tăng giữa hai fold liền nhau: {num(pf['up_share'].autocorr(1))}, gần 0. "
           "Biết fold trước tăng nhiều gần như không cho biết fold sau sẽ thế nào, nên một mô hình không thể thắng luôn tăng "
           "bằng cách bám theo xu hướng gần đây.", "",
           "- Tương quan đầu **dương** (SVM, Persistence) nghĩa là mô hình đoán tăng nhiều hơn ở fold thật sự tăng nhiều. Với SVM điều này "
           "đến từ việc nó gần như luôn đoán tăng, nên DA của nó lên xuống theo tỷ lệ tăng của fold, giống hệt luôn tăng.",
           "- Tương quan đầu **âm** (đảo chiều, B1 tune đầy đủ, RF) nghĩa là mô hình đoán tăng ít hơn ở fold tăng nhiều. "
           "Đây là dấu hiệu mô hình dùng quy luật đảo chiều: nhiều ngày \"hôm nay tăng\" thì nhiều lần đoán \"mai giảm\".",
           "- Cột thứ hai gần 0 với hầu hết mô hình: DA của mô hình không phụ thuộc nhiều vào việc fold tăng hay giảm, trừ SVM.", "",
           "![đi theo xu hướng](figures/lean_vs_up_share.png)", "",
           "## 5. Cùng phép so sánh khi thêm On-chain", ""] + table_summary(s_on, nf) + [
           "", "## 6. simple_split: test 2024 – 08/2026 chia thành khối 30 ngày", "",
           f"Dự báo lấy từ `analyses/simple_split/reports/predictions_test.csv` (mô hình học trên 2018–2023, tune trên validation 2022–2023, "
           f"chạy test một lần). Test {len(ss)} ngày được chia thành {len(pf_s)} khối 30 ngày liên tiếp (khối cuối "
           f"{int(pf_s['n'].iloc[-1])} ngày)."] + [""] + table_summary(s_s, len(pf_s)) + [
           "", f"Luôn tăng trên từng khối dao động {pct(disp_s['min'],0)}–{pct(disp_s['max'],0)}%. "
           f"Majority của simple_split đoán tăng mọi ngày (tỷ lệ tăng của 2018–2023 là hơn 50%), nên trùng với luôn tăng.", "",
           "## Ghi chú", "",
           "- DA của mỗi mô hình lấy từ file dự báo hiện có trong repo; con số khớp với các bảng E1, `b1t_summary.csv`, "
           "`analyses/b2_tuning` và `simple_split` của nhóm.",
           f"- B1 tune đầy đủ: file `b1t_logloss.csv` và bảng `pipeline/results/b1t/tables/b1t_summary.csv` đều cho "
           f"{pct(S.loc['B1 LR tune đầy đủ (Vi)','da'])}%, trong khi `docs/feedback_review/report.md` còn ghi 52,59% (bản chạy cũ hơn). "
           "Nên cập nhật report đó cho khớp.",
           "- p so với luôn tăng: McNemar chính xác trên các ngày mô hình và luôn tăng khác nhau (tức các ngày mô hình đoán giảm).",
           "- Fold cuối của walk-forward chỉ có 4 ngày; nó có trong bảng đếm fold nhưng không có trong thống kê fold đủ 30 ngày.", ""]
    (BASE / "reports/naive_folds.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[4:13]))


def S_s(s, name):
    return s.set_index("model").loc[name, "da"]


if __name__ == "__main__":
    main()
