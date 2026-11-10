"""
eda_pre_training.py  -  EDA dữ liệu trước khi đưa vào training (theo nhanxet.md).

Phần đã có ở chỗ khác, không làm lại:
  - EDA nhãn, tự tương quan return, baseline OHLCV-only, xem dự báo đúng/sai -> baseline_check/
  - Chuỗi giá, bản đồ ô trống, tương quan đặc trưng sơ bộ                    -> tram_features_backtest/notebooks/eda.ipynb

File này bổ sung phần còn trống:
  0. Toàn vẹn dữ liệu (sha256, ngày liên tục, logic OHLC, quy ước ngày giữa các nguồn)
  1. Nhãn và metric: tỷ lệ tăng theo fold, giá trị tham chiếu của metric, ví dụ tính tay       (góp ý 1, 2)
  2. On-chain thô: mùa vụ theo thứ trong tuần, bước nhảy, tính dừng                            (góp ý 4)
  3. Ma trận đặc trưng đúng như pipeline: ô trống, đuôi dày, trùng thông tin, trôi phân phối  (góp ý 4)
  4. Flows: kiểm đồng nhất Netflow = ΔSply, bước nhảy do gắn nhãn ví                           (góp ý 5)
  5. Regime: độ dài đợt, ý nghĩa Bull/Bear/Sideways theo giá                                   (góp ý 6)

QUY TẮC: không tính bất kỳ thống kê nào giữa đặc trưng On-chain và nhãn y (data/processed/README.md,
lưu ý 6). Chỉ xem On-chain với chính nó, với giá, và với đặc trưng OHLCV.

Chạy từ thư mục gốc repo:  python eda_pre_training/src/eda_pre_training.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ngoc_pipeline/src"))
from data import CLOSE, DATE, TARGET, load_config, load_master, onchain_columns, verify_freeze  # noqa: E402
from features import build_dataset  # noqa: E402
from walkforward import make_folds  # noqa: E402

OUT = ROOT / "eda_pre_training/reports"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
WINDOW_START = "2017-12-31"          # ngày train đầu của fold 0
STUDY_START = "2018-01-01"
HALVINGS = ["2020-05-11", "2024-04-20"]
WEEKDAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

# Bảng màu: 3 slot đầu của palette mặc định (đã kiểm định CVD cho mọi cặp) + xám trung tính
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.2, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
warnings.filterwarnings("ignore")   # KPSS báo InterpolationWarning khi p nằm ngoài bảng tra

REPORT: list[str] = []


def md(text=""):
    REPORT.append(text)


def table(df: pd.DataFrame, floatfmt=".3f", index=True):
    md(df.to_markdown(floatfmt=floatfmt, index=index))
    md()


def fig_link(name, caption):
    md(f"![{caption}](figures/{name})")
    md()


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


SUM: dict = {}      # số liệu chính cho phần Tóm tắt


def robust_z(s: pd.Series) -> pd.Series:
    """(x − trung vị) / (1,4826·MAD); MAD = 0 (chuỗi phần lớn bằng một giá trị) thì dùng độ lệch chuẩn."""
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / (mad if mad > 0 else s.std())


# ---------------------------------------------------------------------------
def part0_integrity(master: pd.DataFrame, onchain_all: list[str]):
    md("## Phần 0. Toàn vẹn dữ liệu")
    md()
    sha = verify_freeze()
    md(f"- sha256 của {len(sha)} file khớp `MANIFEST.json` (data freeze v1.0).")
    d = master[DATE]
    md(f"- `master_daily.csv`: {len(master):,} dòng, {d.min().date()} → {d.max().date()}; "
       f"ngày trùng: {d.duplicated().sum()}; bước ngày khác 1: {(d.diff().dropna() != pd.Timedelta(days=1)).sum()}.")
    w = master[master[DATE] >= STUDY_START]
    o, h, l, c, v = (w[f"ohlcv_{k}"] for k in ("open", "high", "low", "close", "volume"))
    bad = ((l > o) | (l > c) | (h < o) | (h < c) | (l > h) | (v <= 0)).sum()
    md(f"- Logic OHLC từ 2018 (low ≤ open, close ≤ high; volume > 0): {bad} ngày vi phạm.")
    md()

    rows = []
    for col in onchain_all:
        s = w[col]
        rows.append({"cột": col, "ô trống từ 2018": int(s.isna().sum()),
                     "≤ 0": int((s <= 0).sum()) if col != "cm_NetflowExNtv" else "(có dấu)",
                     "giá trị lặp y hệt ngày trước": int((s.diff() == 0).sum())})
    md("**Ô trống, giá trị không dương và giá trị bị \"đứng\" (bằng hệt ngày trước) trong 2018 → 08/2026:**")
    md()
    table(pd.DataFrame(rows), index=False)
    hr = w.set_index(DATE)["cm_HashRate"]
    hr_rep = hr[hr.diff() == 0]
    md("Giá trị lặp của `bc_difficulty` là bình thường: difficulty chỉ đổi mỗi 2.016 block (khoảng 2 tuần). "
       f"Hệ quả là `bc_difficulty_logdiff` bằng 0 ở {(np.log(w['bc_difficulty']).diff() == 0).mean():.0%} số ngày; "
       "đặc trưng này gần như là một biến \"ngày điều chỉnh difficulty\". "
       f"`cm_HashRate` có {len(hr_rep)} ngày bằng hệt ngày trước, rải đều {hr_rep.index.year.min()}–{hr_rep.index.year.max()} "
       f"(mỗi năm {hr_rep.index.year.value_counts().min()}–{hr_rep.index.year.value_counts().max()} ngày, không có đợt dài). "
       "Có thể CoinMetrics đã giữ nguyên giá trị hôm trước cho những ngày này. Ảnh hưởng nhỏ, nhưng nên hỏi lại người crawl (Khoa).")
    md()

    # Quy ước ngày: return của giá CoinMetrics so với giá Binance ở các độ trễ
    r_cm = np.log(w.set_index(DATE)["cm_PriceUSD"]).diff()
    r_bn = np.log(w.set_index(DATE)[CLOSE]).diff()
    lag_corr = {k: r_bn.corr(r_cm.shift(k)) for k in (-1, 0, 1)}
    gap = (w["cm_PriceUSD"] / w[CLOSE] - 1).abs()
    md("**Quy ước ngày giữa CoinMetrics và Binance.** Tương quan giữa log return giá CoinMetrics (`cm_PriceUSD`) "
       "và giá đóng cửa Binance:")
    md()
    table(pd.DataFrame({"CM dịch k ngày": list(lag_corr), "tương quan": list(lag_corr.values())}), index=False)
    md(f"Độ trễ 0 cho tương quan cao nhất ({lag_corr[0]:.3f}), nên hai nguồn cùng quy ước ngày UTC. "
       f"Chênh lệch giá trung vị {gap.median():.3%}, lớn nhất {gap.max():.2%} (giá tham chiếu của CoinMetrics "
       "là giá tổng hợp nhiều sàn, không phải giá Binance). Như vậy dòng On-chain ngày d mô tả cùng ngày với nến ngày d, "
       "và `shift_days: 1` của pipeline là đủ, không cần dịch thêm vì lệch quy ước.")
    md()


# ---------------------------------------------------------------------------
def part1_target_metrics(data: pd.DataFrame, folds):
    md("## Phần 1. Nhãn và metric")
    md()
    md("Phần EDA nhãn đầy đủ (tự tương quan, persistence, đảo chiều theo năm) nằm ở "
       "[`baseline_check/reports/baseline_check.md`](../../baseline_check/reports/baseline_check.md). "
       "Ở đây chỉ bổ sung hai điểm ảnh hưởng trực tiếp đến cách đọc bảng kết quả.")
    md()

    # 1a. Tỷ lệ tăng trong train vs test theo fold
    rows = []
    for f in folds:
        tr = data.loc[data[DATE].between(f.train_start, f.train_end), TARGET].dropna()
        te = data.loc[data[DATE].between(f.test_start, f.test_end), TARGET].dropna()
        rows.append({"fold": f.fold_id, "test_start": f.test_start, "n_test": len(te),
                     "up_train": tr.mean(), "up_test": te.mean()})
    fr = pd.DataFrame(rows)
    fr["maj_pred_up"] = fr.up_train > 0.5
    fr["maj_DA"] = np.where(fr.maj_pred_up, fr.up_test, 1 - fr.up_test)
    se30 = np.sqrt(0.25 / 30)
    md("### 1a. Tỷ lệ ngày tăng: cửa sổ train so với cửa sổ test (82 fold)")
    md()
    md(f"- Tỷ lệ tăng trong cửa sổ train 730 ngày chỉ dao động {fr.up_train.min():.1%} – {fr.up_train.max():.1%}. "
       f"B0 majority đoán \"tăng\" ở {fr.maj_pred_up.mean():.0%} số fold và \"giảm\" ở {1 - fr.maj_pred_up.mean():.0%}; "
       "hướng của nó đổi theo một chênh lệch rất nhỏ quanh 50%, nên DA của B0 majority về bản chất là một đồng xu.")
    md(f"- Tỷ lệ tăng trong cửa sổ test 30 ngày dao động {fr.up_test.min():.1%} – {fr.up_test.max():.1%}, "
       f"độ lệch chuẩn {fr.up_test.std():.1%}. Với 30 ngày, chỉ riêng nhiễu lấy mẫu đã cho độ lệch chuẩn "
       f"{se30:.1%}. **DA của từng fold vì vậy gần như không mang thông tin**; chỉ nên đọc DA gộp nhiều fold, "
       "hoặc theo năm.")
    md(f"- Tương quan giữa up_train và up_test qua các fold: {fr.up_train.corr(fr.up_test):.2f}. "
       "Tỷ lệ tăng của quá khứ gần không báo trước tỷ lệ tăng của tháng sau.")
    md()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axhline(0.5, color=GREY, lw=0.8)
    ax.fill_between(fr.test_start, 0.5 - 1.96 * se30, 0.5 + 1.96 * se30, color=GREY, alpha=0.15, lw=0,
                    label="±1,96·SE với 30 ngày")
    ax.plot(fr.test_start, fr.up_test, color=C1, marker="o", ms=3, label="tỷ lệ tăng trong test (30 ngày)")
    ax.plot(fr.test_start, fr.up_train, color=C2, lw=2, label="tỷ lệ tăng trong train (730 ngày)")
    ax.set_ylabel("tỷ lệ ngày tăng")
    ax.set_title("Tỷ lệ ngày tăng theo fold walk-forward")
    ax.legend(loc="lower left", frameon=False, ncol=3)
    savefig(fig, "eda_10_uprate_by_fold.png")
    fig_link("eda_10_uprate_by_fold.png", "Tỷ lệ tăng theo fold")
    fr.to_csv(OUT / "uprate_by_fold.csv", index=False)

    # 1b. Giá trị tham chiếu và ví dụ tính tay
    y_all = data.loc[data[DATE].between(folds[0].test_start, folds[-1].test_end), TARGET].dropna()
    n, p = len(y_all), y_all.mean()
    md("### 1b. Giá trị tham chiếu của metric trên giai đoạn test 2020 → 08/2026")
    md()
    md(f"n = {n:,} ngày, tỷ lệ tăng p = {p:.4f}. Mọi bảng kết quả nên in kèm các mốc này:")
    md()
    ref = pd.DataFrame([
        {"dự báo": "p̂ = 0,5 cho mọi ngày", "DA": np.nan, "log-loss": np.log(2), "Brier": 0.25, "AUC": 0.5},
        {"dự báo": f"p̂ = p = {p:.3f} (biết trước tỷ lệ tăng)", "DA": max(p, 1 - p),
         "log-loss": -(p * np.log(p) + (1 - p) * np.log(1 - p)), "Brier": p * (1 - p), "AUC": 0.5},
        {"dự báo": "luôn đoán tăng", "DA": p, "log-loss": np.nan, "Brier": np.nan, "AUC": 0.5},
    ])
    table(ref.set_index("dự báo"), floatfmt=".4f")
    se = np.sqrt(0.25 / n)
    md(f"- Sai số chuẩn của DA với n = {n:,}: {se:.2%}. Khoảng 95% của mô hình đoán ngẫu nhiên: "
       f"{0.5 - 1.96 * se:.1%} – {0.5 + 1.96 * se:.1%}.")
    md("- **Log-loss phải nhỏ hơn 0,6931 thì mô hình mới hơn đoán 50/50.** Log-loss cao hơn mức này nghĩa là "
       "xác suất của mô hình tệ hơn không biết gì, dù DA có thể trên 50%.")
    md("- AUC chỉ đo khả năng xếp hạng ngày tăng trên ngày giảm, không phụ thuộc ngưỡng 0,5. Một mô hình có p̂ "
       "luôn trong 0,49–0,51 vẫn có thể có AUC 0,53; ngược lại DA phụ thuộc hoàn toàn vào phía nào của 0,5.")
    md()
    ex = pd.DataFrame({"ngày": ["d1", "d2", "d3", "d4"], "y": [1, 0, 1, 0], "p̂": [0.60, 0.55, 0.45, 0.30]})
    ex["dự đoán (p̂ > 0,5)"] = (ex["p̂"] > 0.5).astype(int)
    ex["đúng?"] = (ex["dự đoán (p̂ > 0,5)"] == ex.y).astype(int)
    ex["−log p(y)"] = -np.log(np.where(ex.y == 1, ex["p̂"], 1 - ex["p̂"]))
    ex["(p̂ − y)²"] = (ex["p̂"] - ex.y) ** 2
    md("**Ví dụ tính tay (đối chiếu định nghĩa metric):**")
    md()
    table(ex, floatfmt=".4f", index=False)
    da, ll, br = ex["đúng?"].mean(), ex["−log p(y)"].mean(), ex["(p̂ − y)²"].mean()
    ll_sk, br_sk, auc_sk = log_loss(ex.y, ex["p̂"]), brier_score_loss(ex.y, ex["p̂"]), roc_auc_score(ex.y, ex["p̂"])
    assert np.isclose(ll, ll_sk) and np.isclose(br, br_sk)
    md(f"DA = 3/4 = {da:.2f}; log-loss = trung bình cột −log p(y) = {ll:.4f} (sklearn: {ll_sk:.4f}); "
       f"Brier = {br:.4f} (sklearn: {br_sk:.4f}); AUC = {auc_sk:.2f}: trong 4 cặp (ngày tăng, ngày giảm), "
       "có 3 cặp ngày tăng được xếp p̂ cao hơn (d3 = 0,45 thấp hơn d2 = 0,55). "
       "Khi kiểm bảng của pipeline, dùng đúng cách tính này trên một fold rồi so với con số trong bảng.")
    md()


# ---------------------------------------------------------------------------
def part2_onchain_raw(master: pd.DataFrame, onchain_all: list[str], groups: dict):
    md("## Phần 2. Chỉ số On-chain thô")
    md()
    w = master[master[DATE] >= STUDY_START].set_index(DATE)

    # 2a. Chuỗi thô, thang log
    cols = [c for c in onchain_all if c != "cm_NetflowExNtv"]
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.0 * nrow), sharex=True)
    for ax, c in zip(axes.flat, cols):
        ax.plot(w.index, w[c], color=C1, lw=0.7)
        ax.plot(w.index, w[c].rolling(30).median(), color=INK, lw=1.2)
        ax.set_yscale("log")
        ax.set_title(f"{c}  [{groups.get(c, 'input')}]", loc="left", fontsize=8.5)
        for hv in HALVINGS:
            ax.axvline(pd.Timestamp(hv), color=GREY, lw=0.8, ls="--")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Chỉ số On-chain thô, thang log (xanh: ngày; đen: trung vị 30 ngày; nét đứt: halving)",
                 fontsize=10, x=0.01, ha="left")
    savefig(fig, "eda_20_onchain_raw.png")
    md("### 2a. Chuỗi thô")
    md()
    fig_link("eda_20_onchain_raw.png", "Chuỗi On-chain thô")
    md("HashRate, difficulty, TxCnt, SplyExNtv có xu hướng dài hạn rõ (không dừng). MVRV và Puell dao động theo chu kỳ giá. "
       "Vì vậy **không được đưa mức thô vào mô hình**; pipeline đã dùng logdiff, MA7/MA30 và z-score 90 ngày (xem 2c).")
    md()

    # 2b. Mùa vụ theo thứ trong tuần
    md("### 2b. Mùa vụ theo thứ trong tuần")
    md()
    rows = {}
    for c in cols:
        lv = np.log(w[c].where(w[c] > 0))
        dev = lv - lv.rolling(7, center=True).mean()      # chỉ để mô tả, không phải đặc trưng
        prof = dev.groupby(dev.index.dayofweek).mean()
        ld = lv.diff()
        share = 1 - (ld - ld.groupby(ld.index.dayofweek).transform("mean")).var() / ld.var()
        rows[c] = {"biên độ (%): max − min theo thứ": (prof.max() - prof.min()) * 100,
                   "thứ thấp nhất": WEEKDAYS[int(prof.idxmin())],
                   "% phương sai logdiff do thứ trong tuần": share * 100}
    wk = pd.DataFrame(rows).T.sort_values("% phương sai logdiff do thứ trong tuần", ascending=False)
    table(wk, floatfmt=".1f")
    top = wk.index[:3].tolist()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for c, col in zip(top, (C1, C2, C3)):
        lv = np.log(w[c])
        dev = (lv - lv.rolling(7, center=True).mean()).groupby(w.index.dayofweek).mean() * 100
        ax.plot(WEEKDAYS, dev.values, marker="o", ms=5, color=col, lw=2, label=c)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("% lệch so với trung bình 7 ngày")
    ax.set_title("Hồ sơ theo thứ trong tuần, 2018 → 08/2026")
    ax.legend(frameon=False)
    savefig(fig, "eda_21_weekday.png")
    fig_link("eda_21_weekday.png", "Mùa vụ theo thứ")
    strong = wk.index[wk.iloc[:, 2] > 25].tolist()
    SUM["weekday"] = strong
    SUM["weekday_top"] = (wk.index[0], wk.iloc[0, 2])
    md(f"**Phát hiện:** {len(strong)} chỉ số có hơn 25% phương sai của `logdiff` đến từ thứ trong tuần: "
       f"{', '.join('`'+c+'`' for c in strong)}. Cả giao dịch, phí và dòng tiền vào/ra sàn đều giảm vào cuối tuần. "
       f"Với `{top[0]}`, {wk.iloc[0, 2]:.0f}% phương sai của `logdiff` chỉ là hiệu ứng thứ trong tuần. "
       "Các đặc trưng `*_logdiff` này vì vậy chủ yếu mã hoá \"hôm qua là thứ mấy\", không phải thay đổi thật của hoạt động mạng. "
       "Mô hình cây có thể học thuộc mẫu này, rồi tầm quan trọng của đặc trưng (SHAP, ablation) bị thổi phồng.")
    md()
    md("Đề xuất: với các cột trên, thay `logdiff` (t so với t−1) bằng sai phân 7 ngày (log x_t − log x_{t−7}) "
       "hoặc log(x_t / MA7). MA7/MA30 và z-score 90 ngày ít bị ảnh hưởng hơn, vì trung bình 7 ngày đã triệt tiêu chu kỳ tuần. "
       "Thay đổi này chỉ dựa trên chính chuỗi On-chain, không xem nhãn, nên không vi phạm quy tắc chống data snooping. "
       "Phải áp dụng như nhau cho mọi cấu hình rồi mới chạy E1.")
    md()

    # 2c. Tính dừng
    md("### 2c. Tính dừng (2018 → 08/2026)")
    md()
    md("ADF: H0 = có nghiệm đơn vị (không dừng), p < 0,05 → dừng. KPSS: H0 = dừng, p < 0,05 → không dừng. "
       "Một chuỗi được coi là dừng khi ADF p < 0,05 **và** KPSS p ≥ 0,05. KPSS tra bảng nên p bị chặn trong [0,01; 0,10].")
    md()
    res = []
    for c in onchain_all:
        s = w[c].dropna()
        signed = c == "cm_NetflowExNtv"
        lv = s if signed else np.log(s)
        trans = {"mức (log)" if not signed else "mức": lv,
                 "logdiff" if not signed else "diff": lv.diff(),
                 "MA7/MA30" if not signed else "MA7−MA30":
                     (s.rolling(7).mean() / s.rolling(30).mean()) if not signed else (s.rolling(7).mean() - s.rolling(30).mean()),
                 "z90": (lv - lv.rolling(90).mean()) / lv.rolling(90).std()}
        row = {"cột": c}
        for key, x in zip(("lv", "d", "ma", "z"), trans.values()):
            x = x.replace([np.inf, -np.inf], np.nan).dropna()
            p_adf = adfuller(x, autolag="AIC")[1]
            p_kpss = kpss(x, regression="c", nlags="auto")[1]
            row[key] = f"{p_adf:.2f} / {p_kpss:.2f}" + (" ✓" if p_adf < 0.05 and p_kpss >= 0.05 else "")
        res.append(row)
    st = pd.DataFrame(res)
    st.columns = ["cột", "mức: ADF / KPSS", "sai phân: ADF / KPSS", "MA7÷MA30: ADF / KPSS", "z90: ADF / KPSS"]
    table(st, index=False)
    feat_cols = st.columns[2:]
    never = st.loc[~st[feat_cols].apply(lambda r: r.str.contains("✓")).any(axis=1), "cột"].tolist()
    partial = st.loc[st[feat_cols].apply(lambda r: r.str.contains("✓")).any(axis=1)
                     & ~st[feat_cols].apply(lambda r: r.str.contains("✓")).all(axis=1), "cột"].tolist()
    SUM["nonstat"] = never
    md("✓ = dừng theo cả hai kiểm định. Mức thô (log) của gần như mọi chỉ số không dừng, đúng như hình 2a, "
       "nên không được đưa mức thô vào mô hình. Ba biến đổi của pipeline xử lý được phần lớn các cột, trừ:")
    md()
    md(f"- **Không biến đổi nào dừng:** {', '.join('`'+c+'`' for c in never) or '(không có)'}. "
       "KPSS bác bỏ cả sai phân, nghĩa là trung bình của chuỗi đã biến đổi vẫn đổi theo từng giai đoạn. "
       "Với `cm_SplyExNtv`, nguyên nhân là lượng BTC trên sàn giảm đều nhiều năm, kèm các bước nhảy (xem Phần 4).")
    md(f"- **Chỉ một phần dừng:** {', '.join('`'+c+'`' for c in partial) or '(không có)'}. Hash rate và difficulty "
       "tăng gần như đơn điệu, nên MA7/MA30 và z90 của chúng hầu như luôn lệch về một phía.")
    md()
    md("Chuỗi không dừng làm quan hệ học trên cửa sổ train 730 ngày dễ không còn đúng ở cửa sổ test (xem 3d). "
       "Nên cân nhắc bỏ, hoặc chỉ giữ sai phân (bỏ z90, MA7/MA30) cho các cột trên.")
    md()


# ---------------------------------------------------------------------------
def part3_feature_matrix(data: pd.DataFrame, sets: dict, groups: dict, folds):
    md("## Phần 3. Ma trận đặc trưng đúng như pipeline sẽ dùng")
    md()
    ohlcv = sets["ohlcv"]
    onch = [c for c in sets["ohlcv_onchain"] if c not in ohlcv]
    X = data.loc[data[DATE] >= WINDOW_START].set_index(DATE)
    X = X.loc[X[TARGET].notna()]
    md(f"Dựng bằng `ngoc_pipeline/src/features.py::build_dataset` (flows_set A, shift_days 1): "
       f"{len(ohlcv)} đặc trưng OHLCV + {len(onch)} đặc trưng On-chain, {len(X):,} dòng có nhãn từ {WINDOW_START}. "
       f"Ô trống: {int(X[sets['ohlcv_onchain']].isna().sum().sum())}.")
    md()

    # 3a. Phân phối
    md("### 3a. Phân phối và giá trị cực trị")
    md()
    rows = []
    for c in sets["ohlcv_onchain"]:
        s = X[c]
        rz = robust_z(s)
        rows.append({"đặc trưng": c, "nhóm": "OHLCV" if c in ohlcv else groups[c],
                     "skew": stats.skew(s), "kurtosis dư": stats.kurtosis(s),
                     "% |robust z| > 5": (rz.abs() > 5).mean() * 100, "max |robust z|": rz.abs().max(),
                     "ngày cực trị": rz.abs().idxmax().date()})
    dist = pd.DataFrame(rows).sort_values("kurtosis dư", ascending=False)
    dist.to_csv(OUT / "feature_distribution.csv", index=False)
    table(dist.head(15), floatfmt=".2f", index=False)
    heavy = dist.loc[dist["kurtosis dư"] > 10, "đặc trưng"].tolist()
    md(f"Bảng trên là 15 đặc trưng đuôi dày nhất (đủ 43 dòng: `reports/feature_distribution.csv`). "
       f"{len(heavy)} đặc trưng có kurtosis dư > 10. StandardScaler không xử lý được đuôi dày: một ngày cực trị "
       "(thường là 12/03/2020 hoặc ngày nhảy của Flows) chiếm phần lớn phương sai, các ngày còn lại bị nén về gần 0. "
       "LR và SVM nhạy với điều này, cây (RF, XGBoost) thì không. Đề xuất cho LR/SVM: winsorize theo phân vị 1%/99% "
       "**tính trên cửa sổ train của từng fold**, hoặc dùng `QuantileTransformer`/`RobustScaler` trong pipeline.")
    md()

    # 3b. Trùng thông tin
    md("### 3b. Trùng thông tin giữa các đặc trưng (Spearman, không có nhãn)")
    md()
    corr = X[sets["ohlcv_onchain"]].corr(method="spearman")
    pairs = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename("rho").reset_index()
             .rename(columns={"level_0": "a", "level_1": "b"}))
    pairs = pairs.loc[pairs.rho.abs() >= 0.8].sort_values("rho", key=np.abs, ascending=False)
    table(pairs.head(20), floatfmt=".3f", index=False)
    SUM["redundant_pairs"] = len(pairs)
    md(f"Có {len(pairs)} cặp |ρ| ≥ 0,8. Đọc từng cặp:")
    md()
    md("- `bc_est_tx_vol_usd_*` và `nvt_ratio_bc_*`: NVT = vốn hoá / giá trị giao dịch, nên logdiff của NVT ≈ return − logdiff "
       "của giá trị giao dịch. Hai đặc trưng gần như là một.")
    md("- `logret_lag2` và `cm_CapMVRVCur_logdiff` (ρ ≈ 0,99): MVRV = vốn hoá / vốn hoá thực, mẫu số đổi rất chậm, "
       "nên thay đổi của MVRV gần như chính là return. Sau khi dịch 1 ngày, nó trùng `logret_lag2` (xem 3c).")
    md("- `cm_HashRate_logdiff` và `puell_multiple_logdiff`: Puell = phát hành USD / trung bình 365 ngày; "
       "lượng BTC phát hành mỗi ngày tỷ lệ với số block, mà số block đi theo hash rate.")
    md("- `cm_SplyExNtv_logdiff` và `cm_NetflowExNtv_z90`: từ 2018 đến 03/2026, ΔSplyExNtv **bằng đúng** Netflow (Phần 4), "
       "nên logdiff của Sply ≈ Netflow / Sply. **Trong phương án A, hai cột Flows thực chất là một chuỗi và mức tích luỹ của chính nó.**")
    md()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=6)
    ax.grid(False)
    nb = len(ohlcv)
    ax.axhline(nb - 0.5, color=INK, lw=0.8)
    ax.axvline(nb - 0.5, color=INK, lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
    ax.set_title("Tương quan Spearman giữa 43 đặc trưng (10 OHLCV trước đường kẻ)")
    savefig(fig, "eda_30_feature_corr.png")
    fig_link("eda_30_feature_corr.png", "Tương quan đặc trưng")

    # 3c. Phần On-chain giải thích được bằng OHLCV
    md("### 3c. On-chain có thêm thông tin ngoài giá không?")
    md()
    md("Hồi quy tuyến tính từng đặc trưng On-chain theo 10 đặc trưng OHLCV **cùng dòng** (không có nhãn). "
       "R² cao nghĩa là đặc trưng đó gần như chỉ lặp lại thông tin giá mà mô hình OHLCV-only đã có. "
       "Vì On-chain bị dịch 1 ngày, logdiff của chỉ số định giá theo giá (MVRV, Puell, NVT) ở dòng t "
       "chủ yếu là return ngày t−1, tức `logret_lag2`.")
    md()
    Xo = X[ohlcv].to_numpy()
    rr = []
    for c in onch:
        y = X[c].to_numpy()
        lr = LinearRegression().fit(Xo, y)
        r2 = lr.score(Xo, y)
        best = max(ohlcv, key=lambda o: abs(np.corrcoef(X[o], X[c])[0, 1]))
        rr.append({"đặc trưng": c, "nhóm": groups[c], "R² theo 10 OHLCV": r2,
                   "đặc trưng OHLCV gần nhất": best, "Pearson r": np.corrcoef(X[best], X[c])[0, 1]})
    rr = pd.DataFrame(rr).sort_values("R² theo 10 OHLCV", ascending=False)
    rr.to_csv(OUT / "onchain_r2_on_ohlcv.csv", index=False)
    table(rr, floatfmt=".3f", index=False)
    hi = rr.loc[rr["R² theo 10 OHLCV"] > 0.5, "đặc trưng"].tolist()
    SUM["r2_hi"] = hi
    SUM["r2_median"] = rr["R² theo 10 OHLCV"].median()
    md(f"{len(hi)} đặc trưng có R² > 0,5: {', '.join('`'+h+'`' for h in hi) if hi else '(không có)'}. "
       "Nếu nhóm Valuation cải thiện kết quả trong ablation, cần kiểm tra lại xem cải thiện đến từ thông tin On-chain "
       "hay chỉ từ việc thêm một bản sao của return trễ. Cách kiểm: chạy thêm một cấu hình đối chứng "
       "\"OHLCV + phần dư\" (dùng phần dư của các đặc trưng này sau khi hồi quy theo OHLCV, fit trong train).")
    md()

    # 3d. Trôi phân phối train -> test
    md("### 3d. Trôi phân phối giữa cửa sổ train và test")
    md()
    md("Với mỗi fold: tỷ lệ ngày test có giá trị nằm ngoài khoảng [P1, P99] của cửa sổ train cùng fold. "
       "Nếu không trôi, tỷ lệ kỳ vọng là 2%.")
    md()
    out = []
    for f in folds:
        tr = X.loc[f.train_start:f.train_end, sets["ohlcv_onchain"]]
        te = X.loc[f.test_start:f.test_end, sets["ohlcv_onchain"]]
        if te.empty:
            continue
        lo, hi_ = tr.quantile(0.01), tr.quantile(0.99)
        out.append(((te < lo) | (te > hi_)).mean().rename(f.test_start))
    drift = pd.DataFrame(out)
    summ = pd.DataFrame({"trung bình % ngoài [P1,P99]": drift.mean() * 100,
                         "% fold có > 20% ngày ngoài": (drift > 0.2).mean() * 100})
    summ["nhóm"] = ["OHLCV" if c in ohlcv else groups[c] for c in summ.index]
    summ = summ.sort_values("trung bình % ngoài [P1,P99]", ascending=False)
    summ.to_csv(OUT / "feature_drift.csv")
    table(summ.head(15), floatfmt=".1f")
    gcols = {g: [c for c in summ.index if summ.loc[c, "nhóm"] == g] for g in summ["nhóm"].unique()}
    fig, ax = plt.subplots(figsize=(9, 3))
    for (g, cs), col in zip(sorted(gcols.items()), (C1, C2, C3, GREY, INK)):
        ax.plot(drift.index, drift[cs].mean(axis=1) * 100, color=col, lw=1.6, label=g)
    ax.axhline(2, color=GREY, ls="--", lw=0.8)
    ax.set_ylabel("% ngày test ngoài [P1,P99] train")
    ax.set_title("Trôi phân phối theo fold, trung bình theo nhóm đặc trưng (nét đứt: 2% kỳ vọng)")
    ax.legend(frameon=False, ncol=5)
    savefig(fig, "eda_31_drift.png")
    fig_link("eda_31_drift.png", "Trôi phân phối")
    by_g = summ.groupby("nhóm")["trung bình % ngoài [P1,P99]"].mean()
    worst_fold = drift.mean(axis=1).idxmax()
    md(f"Trung bình theo nhóm: {', '.join(f'{g} {v:.1f}%' for g, v in by_g.sort_values(ascending=False).items())} "
       "(kỳ vọng 2%; đủ 43 dòng ở `reports/feature_drift.csv`). Trôi phân phối nhìn chung **vừa phải** và không riêng gì On-chain: "
       "`rv_30d`, `rv_7d` của OHLCV cũng nằm trong nhóm trôi nhiều nhất, vì biến động của BTC giảm dần qua các năm. "
       f"Trôi tập trung vào một số fold (nhiều nhất: fold bắt đầu {worst_fold.date()}), thường là lúc thị trường chuyển pha. "
       "Ở các fold này, mô hình tuyến tính ngoại suy ra ngoài miền đã học, còn mô hình cây gán giá trị của lá ngoài cùng. "
       "Khi đọc kết quả theo fold hoặc theo năm, nên đối chiếu với hình trên.")
    md()


# ---------------------------------------------------------------------------
def part4_flows(master: pd.DataFrame):
    md("## Phần 4. Flows: sửa lịch sử và thời điểm có dữ liệu")
    md()
    w = master.set_index(DATE)
    w = w[w.index >= STUDY_START]
    d_sply = w["cm_SplyExNtv"].diff()
    net = w["cm_NetflowExNtv"]
    resid = (d_sply - net).dropna()
    off = resid.abs() > 1                                   # lệch hơn 1 BTC
    by_m = off.groupby(resid.index.to_period("M")).sum()
    # tháng đầu tiên mà từ đó trở đi mọi tháng đều lệch ≥ 15 ngày
    full = by_m >= 15
    start_bad = full[::-1].cumprod()[::-1]
    first_bad = start_bad[start_bad == 1].index.min()
    hist = resid[resid.index < first_bad.start_time]
    SUM["flows_first_bad"] = str(first_bad)
    SUM["flows_hist_off"] = int((hist.abs() > 1).sum())

    md("### 4a. Netflow có khớp với thay đổi lượng BTC trên sàn không?")
    md()
    md("Theo định nghĩa, lượng BTC trên sàn hôm nay bằng hôm qua cộng dòng ròng: ΔSplyExNtv_t = Netflow_t = In_t − Out_t. "
       "Nếu CoinMetrics thêm ví của một sàn vào danh sách mà không tính lại quá khứ, đẳng thức sẽ gãy đúng ngày đó.")
    md()
    md(f"- **{hist.index.min().date()} → {hist.index.max().date()}: đẳng thức đúng tới từng BTC.** "
       f"Chỉ {SUM['flows_hist_off']} trong {len(hist):,} ngày lệch quá 1 BTC (lệch lớn nhất {hist.abs().max():,.0f} BTC).")
    md(f"- **Từ {first_bad} trở đi: ngày nào cũng lệch.** {int(off[off.index >= first_bad.start_time].sum())} / "
       f"{int((resid.index >= first_bad.start_time).sum())} ngày, có ngày lệch tới {resid.abs().max():,.0f} BTC.")
    md()
    tb = by_m[by_m > 0].rename("số ngày lệch > 1 BTC").to_frame()
    tb["lệch lớn nhất (BTC)"] = resid.abs().groupby(resid.index.to_period("M")).max()
    md("Số ngày lệch theo tháng (chỉ liệt kê tháng có lệch):")
    md()
    table(tb, floatfmt=",.0f")

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    axes[0].plot(w.index, w["cm_SplyExNtv"] / 1e6, color=C1)
    axes[0].set_ylabel("triệu BTC")
    axes[0].set_title("Lượng BTC trên sàn (cm_SplyExNtv)", loc="left")
    axes[1].plot(resid.index, resid / 1e3, color=C2, lw=0.8)
    axes[1].set_ylabel("nghìn BTC")
    axes[1].set_title("ΔSplyExNtv − Netflow: bằng 0 suốt lịch sử, chỉ khác 0 ở các tháng gần ngày tải", loc="left")
    for ax in axes:
        ax.axvline(first_bad.start_time, color=GREY, lw=0.8, ls="--")
    savefig(fig, "eda_40_flows_identity.png")
    fig_link("eda_40_flows_identity.png", "Kiểm đồng nhất Flows")

    md("**Cách hiểu.** Dữ liệu được tải cuối tháng 10/2026 (`flows_check.md` ghi 24/10/2026). Phần lịch sử đã được CoinMetrics **tính lại cho nhất quán** "
       "với danh sách ví sàn hiện tại, nên đẳng thức đúng tuyệt đối. Vài tháng gần ngày tải chưa được tính lại "
       "nên đẳng thức gãy. Nói cách khác: **giá trị một người dự báo thấy vào lúc đó (giống phần cuối chuỗi) khác giá trị "
       "đang nằm trong file cho cùng ngày (phần lịch sử đã sửa)**. Đây là bằng chứng trực tiếp trong dữ liệu cho góp ý 5.")
    md()
    md("Hai hệ quả:")
    md()
    md("1. Backtest 2020 → 03/2026 dùng Flows **đã sửa**, đẹp hơn và nhất quán hơn dữ liệu thời gian thực. "
       "Mọi cải thiện từ Flows trong backtest là cận trên, không phải con số thực tế.")
    md(f"2. Các fold test từ {first_bad} dùng Flows **chưa sửa**, tức là dữ liệu khác bản chất với dữ liệu train của chính fold đó. "
       "Lần tải sau, các giá trị này sẽ đổi và kết quả của các fold cuối sẽ không tái lập được.")
    md()

    md("### 4b. Bước nhảy của Netflow")
    md()
    rz = robust_z(net)
    jumps = rz[rz.abs() > 8].sort_values(key=np.abs, ascending=False)
    SUM["flows_jumps"] = len(jumps)
    jt = pd.DataFrame({"Netflow (BTC)": net.loc[jumps.index], "robust z": jumps})
    jt.index = jt.index.date
    md(f"{len(jumps)} ngày có |robust z| > 8 (10 ngày lớn nhất):")
    md()
    table(jt.head(10), floatfmt=",.1f")
    md("Nhiều ngày rơi vào các sự kiện có thật (12/03/2020, LUNA 05/2022, FTX 11/2022), nên không thể bỏ hết như lỗi. "
       "Nhưng chuỗi dao động lên xuống liên tiếp trong 11/2022 (+88.036 ngày 18, −52.318 ngày 17) giống việc gắn và chuyển nhãn ví "
       "hơn là dòng tiền thật. Không có vintage thì không phân biệt được hai trường hợp này.")
    md()

    md("### 4c. Vì sao dịch trễ 1 ngày không giải quyết được")
    md()
    md("- `shift_days: 1` xử lý **độ trễ công bố**: CoinMetrics công bố ngày d khoảng 2–9,5 giờ sau 00:00 UTC ngày d+1. "
       "Nó không xử lý **sửa lịch sử**. Khi một ví được gắn nhãn sàn vào năm 2024, Flows của ví đó được tính lại cho cả 2018–2023, "
       "nên giá trị ngày 2021-05-01 trong file khác giá trị có thật vào ngày 2021-05-02, dù dịch bao nhiêu ngày.")
    md("- File GitHub và API trong `crawl_data/vi_ohlcv_flows/reports/flows_check.md` khớp 100%, nhưng chúng được tải "
       "gần như cùng lúc, nên phép so đó không đo được mức sửa. Phép kiểm ở 4a là bằng chứng duy nhất hiện có.")
    md()
    md("**Đề xuất (không cần xem nhãn):**")
    md()
    md("1. Chạy E1 thêm cấu hình **OHLCV + On-chain bỏ Flows**, cùng model, fold và tuning. Kết luận RQ1 nên dựa trên cấu hình này; "
       "cấu hình có Flows báo cáo riêng, ghi rõ là cận trên.")
    md(f"2. Ghi trong báo cáo: các fold từ {first_bad} dùng dữ liệu Flows chưa được tính lại. Có thể báo cáo thêm kết quả bỏ các fold này.")
    md("3. Tải lại Flows hằng tuần đến hết đề tài, lưu kèm thời điểm tải. So giá trị của cùng một ngày lịch sử giữa các lần tải "
       "để định lượng mức sửa. Trong vài tuần đã thấy được phần 2026 thay đổi bao nhiêu.")
    md("4. Trong phương án A, Netflow chính là ΔSplyExNtv, nên `cm_SplyExNtv_logdiff` gần như trùng thông tin Netflow (3b). "
       "Chỉ mức (z90, MA7/MA30) của SplyExNtv là thêm thông tin, nhưng đó lại là chuỗi không dừng (2c). Nếu vẫn giữ Flows, "
       "phương án B (In, Out, Sply) cũng không tốt hơn, vì In và Out thuộc nhóm có mùa vụ tuần mạnh nhất (2b).")
    md()


# ---------------------------------------------------------------------------
def runs(labels: pd.Series) -> pd.DataFrame:
    grp = (labels != labels.shift()).cumsum()
    r = labels.groupby(grp).agg(["first", "size"])
    r.columns = ["regime", "len"]
    return r.reset_index(drop=True)


def part5_regime(master: pd.DataFrame):
    md("## Phần 5. Regime")
    md()
    rm = pd.read_csv(ROOT / "ngan_lstm_hmm_analysis/regime_map.csv")
    rm[DATE] = pd.to_datetime(rm[DATE]).dt.tz_localize(None)
    dup = int(rm[DATE].duplicated().sum())
    rm = rm.drop_duplicates(DATE, keep="last").set_index(DATE).sort_index()
    px = master.set_index(DATE)[CLOSE]
    lr = np.log(px).diff()
    df = rm.join(lr.rename("ret")).join(px.rename("close"))
    df["ret_next"] = lr.shift(-1).reindex(df.index)
    df["ret_back30"] = np.log(px).diff(30).reindex(df.index)
    df["ret_fwd30"] = (np.log(px).shift(-30) - np.log(px)).reindex(df.index)

    md(f"Đọc `ngan_lstm_hmm_analysis/regime_map.csv`: {len(rm):,} ngày ({dup} ngày trùng đã bỏ). "
       "Script tạo file này chưa có trong repo nên phần này chỉ mô tả nhãn, không tái lập được.")
    md()
    md("### 5a. Độ dài mỗi đợt regime")
    md()
    r = runs(df["regime"])
    rs = r.groupby("regime")["len"].agg(["count", "mean", "median", "max"])
    rs["% đợt chỉ 1 ngày"] = r.groupby("regime")["len"].apply(lambda s: (s == 1).mean() * 100)
    rs["số ngày"] = df["regime"].value_counts()
    rs.columns = ["số đợt", "dài TB (ngày)", "trung vị", "dài nhất", "% đợt 1 ngày", "số ngày"]
    table(rs, floatfmt=".1f")
    sw = df["regime"].groupby(df["fold"]).apply(lambda s: (s != s.shift()).sum() - 1)
    md(f"Tổng {len(r)} đợt trong {len(df):,} ngày, tức trung bình đổi regime mỗi {len(df) / len(r):.1f} ngày; "
       f"trung vị {sw.median():.0f} lần đổi trong một fold 30 ngày.")
    md()

    md("### 5b. Bull/Bear/Sideways có đúng nghĩa theo giá không?")
    md()
    md("Chỉ dùng giá (không có On-chain). Nếu nhãn đúng nghĩa, Bull phải có return 30 ngày trước/sau dương rõ, Bear âm rõ.")
    md()
    prof = df.groupby("regime").agg(
        n=("ret", "size"),
        ret_ngay_TB=("ret", lambda s: s.mean() * 100),
        bien_dong_ngay=("ret", lambda s: s.std() * 100),
        ret_30_truoc_TB=("ret_back30", lambda s: s.mean() * 100),
        ret_30_sau_TB=("ret_fwd30", lambda s: s.mean() * 100),
        ty_le_tang_t1=("ret_next", lambda s: (s > 0).mean() * 100),
    )
    prof.columns = ["số ngày", "return ngày t TB (%)", "độ lệch chuẩn return (%)",
                    "return 30 ngày trước TB (%)", "return 30 ngày sau TB (%)", "tỷ lệ tăng ngày t+1 (%)"]
    table(prof, floatfmt=".2f")

    # Tham chiếu: luật đơn giản, nhân quả, minh bạch
    ma200 = px.rolling(200).mean()
    r90 = np.log(px).diff(90)
    rule = pd.Series("Sideways", index=px.index)
    rule[(px > ma200) & (r90 > 0.10)] = "Bull"
    rule[(px < ma200) & (r90 < -0.10)] = "Bear"
    # Làm mượt nhân quả: nhãn ngày t = nhãn xuất hiện nhiều nhất trong 21 ngày t−20..t (chỉ quá khứ)
    code = rule.map({"Bear": 0, "Sideways": 1, "Bull": 2}).astype(float)
    code[ma200.isna()] = np.nan
    sm = code.rolling(21).apply(lambda a: np.bincount(a.astype(int), minlength=3).argmax(), raw=True)
    rule = sm.map({0: "Bear", 1: "Sideways", 2: "Bull"}).reindex(df.index)
    r2 = runs(rule)
    rs2 = r2.groupby("regime")["len"].agg(["count", "mean", "median"])
    rs2["số ngày"] = rule.value_counts()
    rs2.columns = ["số đợt", "dài TB (ngày)", "trung vị", "số ngày"]
    md("**Tham chiếu: một luật đơn giản, chỉ dùng quá khứ.** Bull nếu giá > MA200 và return 90 ngày > +10%; "
       "Bear nếu giá < MA200 và return 90 ngày < −10%; còn lại Sideways. Sau đó làm mượt: nhãn ngày t là nhãn xuất hiện nhiều "
       "nhất trong 21 ngày t−20 → t. Ngưỡng chọn theo quy ước, không tối ưu, chỉ để so độ dài đợt và ý nghĩa nhãn:")
    md()
    table(rs2, floatfmt=".1f")
    prof2 = df.assign(rule=rule).groupby("rule").agg(
        r30b=("ret_back30", lambda s: s.mean() * 100), r30f=("ret_fwd30", lambda s: s.mean() * 100),
        sd=("ret", lambda s: s.std() * 100))
    prof2.columns = ["return 30 ngày trước TB (%)", "return 30 ngày sau TB (%)", "độ lệch chuẩn return (%)"]
    prof2.index.name = "luật"
    table(prof2, floatfmt=".2f")
    agree = (rule == df["regime"]).mean()
    ct = pd.crosstab(df["regime"], rule, normalize="index") * 100
    ct.index.name, ct.columns.name = "HMM \\ luật", None
    md(f"Hai cách gán trùng nhau ở {agree:.0%} số ngày. Bảng chéo (% theo hàng HMM):")
    md()
    table(ct, floatfmt=".0f")

    cmap = {"Bull": C1, "Sideways": GREY, "Bear": C2}
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, lab, title in ((axes[0], df["regime"], "HMM (regime_map.csv)"),
                           (axes[1], rule, "Luật MA200 + return 90 ngày")):
        ax.plot(df.index, df["close"], color=INK, lw=0.8)
        ax.set_yscale("log")
        for reg, col in cmap.items():
            m = (lab == reg).to_numpy()
            ax.fill_between(df.index, 0, 1, where=m, color=col, alpha=0.35, lw=0,
                            transform=ax.get_xaxis_transform(), label=reg)
        ax.set_title(title, loc="left")
        ax.set_ylabel("giá BTC (log)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    savefig(fig, "eda_50_regime.png")
    fig_link("eda_50_regime.png", "Regime HMM so với luật đơn giản")
    SUM["hmm_mean_run"] = len(df) / len(r)
    SUM["hmm_bear_fwd"] = prof.loc["Bear", "return 30 ngày sau TB (%)"]
    SUM["hmm_sd"] = prof["độ lệch chuẩn return (%)"].round(1).to_dict()
    SUM["rule_mean_run"] = len(rule.dropna()) / len(r2)
    md("**Nhận xét.**")
    md()
    md(f"- **Nhãn HMM đổi trung bình mỗi {SUM['hmm_mean_run']:.1f} ngày**, hơn một nửa số đợt Bear chỉ dài 1 ngày (5a). "
       "Không có đợt Bull hay Bear nào kéo dài như các chu kỳ thật của BTC (nhiều tháng).")
    md(f"- **HMM tách theo biến động, không theo xu hướng.** Độ lệch chuẩn return ngày: Bear {SUM['hmm_sd']['Bear']}%, "
       f"Bull {SUM['hmm_sd']['Bull']}%, Sideways {SUM['hmm_sd']['Sideways']}%. Return 30 ngày **sau** các ngày Bear lại "
       f"cao nhất ({SUM['hmm_bear_fwd']:+.2f}%), và Sideways có return 30 ngày trước còn cao hơn Bull. Trên thực tế \"Bear\" của HMM "
       "là \"ngày biến động mạnh, thường là ngày giảm\", \"Sideways\" là \"ngày biến động thấp\".")
    md(f"- Luật đơn giản đã làm mượt cho đợt dài trung bình {SUM['rule_mean_run']:.0f} ngày, và Bull/Bear của nó "
       "có return 30 ngày trước đúng dấu. Hai cách gán chỉ trùng nhau ở mức xấp xỉ ngẫu nhiên (bảng chéo).")
    md()
    md("Với nhãn HMM hiện tại, E3 sẽ trả lời câu hỏi \"Δ có khác nhau giữa ngày biến động cao và thấp không\", "
       "chứ không phải RQ3 (\"Δ có khác nhau giữa thị trường tăng, giảm, đi ngang không\"). Đề xuất trước khi chạy E3:")
    md()
    md("1. Đưa script tạo đầu vào HMM vào repo, chạy lại ra đúng `regime_map.csv` (tái lập được).")
    md("2. Định nghĩa regime theo xu hướng trên khung dài (ví dụ return 60–90 ngày, hoặc HMM trên return trung bình "
       "trượt 30 ngày), đặt ngưỡng bằng quy ước hoặc chỉ trên dữ liệu train; yêu cầu độ dài đợt tối thiểu (ví dụ ≥ 20 ngày) "
       "hoặc làm mượt xác suất trạng thái.")
    md("3. Kiểm tra bằng mắt trên các giai đoạn đã biết: 2020-03 (sập COVID), 2020-10 → 2021-04 (tăng), "
       "2022-05 → 2022-11 (LUNA, FTX, giảm), 2023 (đi ngang rồi tăng), 2024-03 (đỉnh trước halving).")
    md()


# ---------------------------------------------------------------------------
def main():
    master = load_master()
    cfg = load_config("features_onchain.yaml")
    wf = load_config("walkforward.yaml")
    data, sets = build_dataset(master, cfg)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf)
    onchain_A = onchain_columns("A")
    onchain_all = onchain_A + [c for c in onchain_columns("B") if c not in onchain_A]
    from data import COL_GROUP
    groups = {**COL_GROUP}
    ohlcv = set(sets["ohlcv"])
    for f in sets["ohlcv_onchain"]:
        if f not in ohlcv:
            base = next(c for c in onchain_all if f.startswith(c + "_"))
            groups[f] = COL_GROUP[base]

    md("# EDA dữ liệu trước khi training")
    md()
    md(f"Tạo bởi `eda_pre_training/src/eda_pre_training.py` trên data freeze v1.0. Lịch walk-forward: {len(folds)} fold, "
       f"test {folds[0].test_start.date()} → {folds[-1].test_end.date()}. Phương án Flows: {cfg['flows_set']}.")
    md()
    md("**Không có thống kê nào giữa đặc trưng On-chain và nhãn y** (data/processed/README.md, lưu ý 6). "
       "Mọi phát hiện về On-chain dưới đây chỉ dựa trên chính chuỗi On-chain, giá và đặc trưng OHLCV, "
       "nên các thay đổi đề xuất được phép áp dụng trước khi chạy thí nghiệm.")
    md()
    md("<!-- SUMMARY -->")
    md()
    part0_integrity(master, onchain_all)
    part1_target_metrics(data, folds)
    part2_onchain_raw(master, onchain_all, groups)
    part3_feature_matrix(data, sets, groups, folds)
    part4_flows(master)
    part5_regime(master)

    s = SUM
    summary = [
        "## Tóm tắt",
        "",
        "| # | Kiểm tra | Kết quả | Việc cần làm trước khi train |",
        "|---|---|---|---|",
        "| 0 | Toàn vẹn dữ liệu | Đúng phiên bản, không thiếu ngày, OHLC hợp lệ, CoinMetrics và Binance cùng quy ước ngày | Không |",
        "| 1 | Nhãn và metric | DA từng fold (30 ngày) có SE ≈ 9 điểm: không đọc được. SE của DA gộp ≈ 1 điểm. "
        "Log-loss phải < 0,6931 mới hơn đoán 50/50 | In kèm các mốc tham chiếu (1b) trong mọi bảng kết quả |",
        f"| 2 | Mùa vụ tuần | {len(s['weekday'])} chỉ số có > 25% phương sai `logdiff` do thứ trong tuần "
        f"({', '.join('`'+c+'`' for c in s['weekday'])}; nặng nhất `{s['weekday_top'][0]}`: {s['weekday_top'][1]:.0f}%) | Đổi `logdiff` thành sai phân 7 ngày cho các cột này |",
        f"| 2 | Tính dừng | Không biến đổi nào của {', '.join('`'+c+'`' for c in s['nonstat'])} dừng | "
        "Cân nhắc bỏ, hoặc chỉ giữ sai phân |",
        f"| 3 | Trùng thông tin với giá | {len(s['r2_hi'])} đặc trưng On-chain có R² > 0,5 theo 10 đặc trưng OHLCV, "
        "đều là MVRV; `cm_CapMVRVCur_logdiff` ≈ `logret_lag2` (ρ 0,99) | Ablation phải tách riêng MVRV, "
        "nếu không thì Valuation \"đóng góp\" chỉ là return trễ |",
        f"| 3 | Đuôi dày, trôi phân phối | Đặc trưng đuôi dày nhất là của Flows, difficulty và OHLCV; trôi vừa phải và tập trung ở vài fold | "
        "Winsorize hoặc QuantileTransformer trong pipeline cho LR/SVM |",
        f"| 4 | Flows bị sửa lịch sử | ΔSply = Netflow chính xác đến {s['flows_first_bad']}, sau đó ngày nào cũng lệch: "
        "lịch sử đã được tính lại, mấy tháng cuối chưa | Chạy đối chứng bỏ Flows; ghi các fold cuối là dữ liệu chưa sửa; "
        "bắt đầu lưu vintage |",
        f"| 5 | Regime | HMM đổi nhãn mỗi {s['hmm_mean_run']:.1f} ngày; tách theo biến động chứ không theo xu hướng "
        f"(return 30 ngày sau Bear: {s['hmm_bear_fwd']:+.1f}%) | Định nghĩa lại regime theo xu hướng dài hạn, "
        "đưa script vào repo, rồi mới chạy E3 |",
        "",
        "Thứ tự đề xuất (theo góp ý 7): (1) sửa biến đổi On-chain ở mục 2–3 → (2) chốt baseline OHLCV-only theo "
        "`baseline_check` → (3) thêm On-chain **từng nhóm**, giữ nguyên model, fold và tuning, trong đó có cấu hình bỏ Flows → "
        "(4) sửa regime rồi mới làm E3.",
    ]
    i = REPORT.index("<!-- SUMMARY -->")
    REPORT[i:i + 1] = summary
    (OUT / "eda_pre_training.md").write_text("\n".join(REPORT), encoding="utf-8")
    print("Đã ghi", OUT / "eda_pre_training.md")


if __name__ == "__main__":
    main()
