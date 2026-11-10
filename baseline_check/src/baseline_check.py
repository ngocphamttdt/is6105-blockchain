"""
baseline_check.py  –  Góp ý 1 của thầy (Vi + Trâm)
"Kiểm tra data, làm EDA và chạy một baseline cơ bản trước; xem trực tiếp một số dự báo đúng/sai
 và bảo đảm kết quả có ý nghĩa rồi mới mở rộng."

Chỉ dùng OHLCV và data freeze v1.0 (data/processed/). Không dùng On-chain, không dùng ngoc_pipeline.
Mọi thứ cố ý đơn giản để ai cũng kiểm tra tay được.

Chạy từ thư mục gốc repo:
    python baseline_check/src/baseline_check.py
Đầu ra: baseline_check/reports/ (baseline_check.md, figures/, predictions_test.csv, ...)

Bố cục:
  Phần 1  Kiểm tra dữ liệu và nhãn bằng ví dụ cụ thể
  Phần 2  EDA cho nhãn: return, tự tương quan, vì sao persistence < 50%
  Phần 3  Baseline cơ bản: chia train / validation / test theo thời gian, tune trên validation,
          test chạy đúng 1 lần; kiểm chứng metric bằng tay; phép thử xáo nhãn và phép thử rò rỉ cố ý
  Phần 4  Xem trực tiếp dự báo đúng/sai
  Phần 5  Đối chiếu với walk-forward (cùng giai đoạn test 2020–2026 như báo cáo giữa kỳ)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             log_loss, matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tram_features_backtest/src"))
from features_ohlcv import FEATURES, MASTER, build_ohlcv_features, verify_freeze   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
FIG = OUT / "figures"

# Chia theo thời gian. Ngày cuối của train và validation bị bỏ (gap 1 ngày) vì nhãn của ngày đó
# dùng giá đóng cửa của ngày đầu tiên của đoạn sau.
SPLITS = {
    "train": ("2018-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2023-12-31"),
    "test":  ("2024-01-01", "2026-08-30"),     # 2026-08-30 là ngày cuối còn nhãn (cần giá 31/08)
}
WF_TEST = ("2020-01-01", "2026-08-30")         # giai đoạn test của walk-forward trong báo cáo giữa kỳ
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 0


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Một dòng mỗi ngày t: giá, 10 đặc trưng OHLCV (đến hết ngày t), nhãn y_t = 1[C_{t+1} > C_t]."""
    verify_freeze(["master_daily.csv", "ohlcv_1h_clean.csv"])
    master = pd.read_csv(MASTER, parse_dates=["date"])
    feats = build_ohlcv_features(master).set_index("date")
    m = master.set_index("date")
    df = feats.join(m[["ohlcv_open", "ohlcv_close", "ohlcv_n_bars_1h"]])
    df["close_next"] = df["ohlcv_close"].shift(-1)
    df["ret_next"] = np.log(df["close_next"] / df["ohlcv_close"])          # chỉ để kiểm tra, KHÔNG là đặc trưng
    df["y"] = (df["close_next"] > df["ohlcv_close"]).astype(float).where(df["close_next"].notna())
    df["up_today"] = (df["logret_lag1"] > 0).astype(float).where(df["logret_lag1"].notna())
    # Từ 31/12/2017: fold 0 của walk-forward cần đủ 730 ngày train trước gap (31/12/2017 → 30/12/2019),
    # như ngoc_pipeline. Bắt đầu từ 01/01/2018 thì chỉ số train của fold 0 thành −1, tức ngày cuối 2026.
    return df.loc["2017-12-31":"2026-08-30"]


def split(df):
    parts = {k: df.loc[a:b] for k, (a, b) in SPLITS.items()}
    parts["train"] = parts["train"].iloc[:-1]      # gap 1 ngày
    parts["val"] = parts["val"].iloc[:-1]
    return parts


# ---------------------------------------------------------------------------
# Metric (viết lại bằng công thức để đối chiếu với sklearn)
# ---------------------------------------------------------------------------
def metrics(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    pred = (p > 0.5).astype(int)
    n, k = len(y), int((pred == y).sum())
    out = {
        "n": n,
        "DA": k / n,
        "DA_CI_low": stats.binomtest(k, n).proportion_ci().low,
        "DA_CI_high": stats.binomtest(k, n).proportion_ci().high,
        "p_vs_50": stats.binomtest(k, n, 0.5).pvalue,
        "bal_acc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "AUC": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "pct_pred_up": pred.mean(),
    }
    return out


def hand_check(y, p):
    """Tính DA, log-loss, Brier bằng tay trên vài dòng và so với sklearn."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p > 0.5).astype(int)
    rows = pd.DataFrame({"y": y, "p_hat": p, "dự báo": pred, "đúng?": pred == y,
                         "−log p(nhãn thật)": -np.log(np.where(y == 1, p, 1 - p)),
                         "(p − y)²": (p - y) ** 2})
    manual = {"DA": rows["đúng?"].mean(), "log_loss": rows["−log p(nhãn thật)"].mean(),
              "brier": rows["(p − y)²"].mean()}
    sk = {"DA": accuracy_score(y, pred), "log_loss": log_loss(y, p, labels=[0, 1]),
          "brier": brier_score_loss(y, p)}
    assert all(np.isclose(manual[k], sk[k]) for k in manual), (manual, sk)
    return rows, manual, sk


# ---------------------------------------------------------------------------
# Mô hình baseline
# ---------------------------------------------------------------------------
def rule_predictions(train, target):
    """Các luật không học tham số (hoặc chỉ học 1 con số từ train)."""
    up_share = train["y"].mean()
    # Chiều của luật đảo chiều chọn từ train: nếu trên train "mai cùng chiều hôm nay" < 50% thì đảo
    pers_train = (train["y"] == train["up_today"]).mean()
    contrarian = pers_train < 0.5
    return {
        "B0 Majority (p̂ = tỷ lệ tăng của train)": np.full(len(target), up_share),
        "B0 Luôn đoán tăng": np.full(len(target), 0.99),
        "B0 Persistence (mai cùng chiều hôm nay)": np.where(target["up_today"] == 1, 0.99, 0.01),
        "B0 Đảo chiều (mai ngược chiều hôm nay)": (np.where(target["up_today"] == 1, 0.01, 0.99)
                                                   if contrarian else np.full(len(target), np.nan)),
    }, {"up_share_train": up_share, "persistence_acc_train": pers_train, "contrarian_chosen": contrarian}


def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def tune_lr(train, val, cols):
    """Tune C trên validation theo log-loss. Trả bảng mọi C (trước/sau tuning) và C tốt nhất."""
    rows = []
    for C in C_GRID:
        mdl = lr(C).fit(train[cols], train["y"].astype(int))
        pv = mdl.predict_proba(val[cols])[:, 1]
        pt = mdl.predict_proba(train[cols])[:, 1]
        rows.append({"C": C, "train_log_loss": log_loss(train["y"], pt), "val_log_loss": log_loss(val["y"], pv),
                     "train_DA": ((pt > .5) == train["y"]).mean(), "val_DA": ((pv > .5) == val["y"]).mean()})
    t = pd.DataFrame(rows)
    best = float(t.loc[t["val_log_loss"].idxmin(), "C"])
    return t, best


# ---------------------------------------------------------------------------
# Phép thử để chắc pipeline có ý nghĩa
# ---------------------------------------------------------------------------
def sanity_controls(parts, cols, C):
    tr, te = parts["train"], parts["test"]
    rng = np.random.default_rng(SEED)
    out = {}
    # (a) Xáo nhãn train 200 lần: DA test phải quanh 50%
    das = []
    for _ in range(200):
        ys = rng.permutation(tr["y"].astype(int).to_numpy())
        p = lr(C).fit(tr[cols], ys).predict_proba(te[cols])[:, 1]
        das.append(((p > .5) == te["y"]).mean())
    out["shuffle_DA_mean"], out["shuffle_DA_p2.5"], out["shuffle_DA_p97.5"] = (
        float(np.mean(das)), float(np.percentile(das, 2.5)), float(np.percentile(das, 97.5)))
    # (b) Rò rỉ cố ý: thêm return của NGÀY MAI vào đặc trưng -> DA phải gần 100%.
    #     Nếu không gần 100% thì pipeline đang căn sai ngày giữa đặc trưng và nhãn.
    leak_cols = cols + ["ret_next"]
    p = lr(C).fit(tr[leak_cols], tr["y"].astype(int)).predict_proba(te[leak_cols])[:, 1]
    out["leak_DA"] = float(((p > .5) == te["y"]).mean())
    # (c) "Rò rỉ" sai chiều: dùng return của chính ngày t làm nhãn mục tiêu -> chỉ đặc trưng hôm nay dự báo được
    p = lr(C).fit(tr[cols], tr["up_today"].astype(int)).predict_proba(te[cols])[:, 1]
    out["predict_today_DA"] = float(((p > .5) == te["up_today"]).mean())
    return out, das


# ---------------------------------------------------------------------------
# Walk-forward đơn giản (cùng lịch với ngoc_pipeline: train 730, gap 1, test 30, bước 30)
# ---------------------------------------------------------------------------
def walk_forward(df, cols_sets, C_fixed=1.0):
    test_days = df.loc[WF_TEST[0]:WF_TEST[1]].index
    preds = {name: pd.Series(np.nan, index=test_days) for name in cols_sets}
    rules = {"Majority": pd.Series(np.nan, index=test_days), "Persistence": pd.Series(np.nan, index=test_days),
             "Đảo chiều": pd.Series(np.nan, index=test_days)}
    pos = df.index.get_indexer(test_days)
    for start in range(0, len(test_days), 30):
        te_idx = pos[start:start + 30]
        tr_idx = np.arange(te_idx[0] - 1 - 730, te_idx[0] - 1)
        assert tr_idx[0] >= 0, "Cửa sổ train vượt đầu dữ liệu: iloc âm sẽ lấy nhầm ngày cuối chuỗi"
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        for name, cols in cols_sets.items():
            preds[name].loc[te.index] = lr(C_fixed).fit(tr[cols], tr["y"].astype(int)).predict_proba(te[cols])[:, 1]
        rules["Majority"].loc[te.index] = tr["y"].mean()
        rules["Persistence"].loc[te.index] = np.where(te["up_today"] == 1, .99, .01)
        contr = (tr["y"] == tr["up_today"]).mean() < 0.5               # chiều đảo chọn lại trên mỗi cửa sổ train
        rules["Đảo chiều"].loc[te.index] = (np.where(te["up_today"] == 1, .01, .99) if contr
                                            else np.where(te["up_today"] == 1, .99, .01))
    return {**rules, **preds}, test_days


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def fmt_metrics(rows: dict) -> pd.DataFrame:
    t = pd.DataFrame(rows).T
    t["DA (%)"] = (t["DA"] * 100).round(2)
    t["CI 95% DA"] = [f"[{a*100:.1f}; {b*100:.1f}]" for a, b in zip(t["DA_CI_low"], t["DA_CI_high"])]
    t["p (DA = 50%)"] = t["p_vs_50"].map(lambda v: f"{v:.3f}" if v >= 0.001 else "<0,001")
    keep = ["n", "DA (%)", "CI 95% DA", "p (DA = 50%)", "bal_acc", "MCC", "AUC", "log_loss", "brier", "pct_pred_up"]
    t = t[keep]
    for c in ["bal_acc", "MCC", "AUC", "brier", "pct_pred_up"]:
        t[c] = t[c].astype(float).round(3)
    t["log_loss"] = t["log_loss"].astype(float).round(4)
    t["n"] = t["n"].astype(int)
    return t


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    md = []
    df = load_data()
    parts = split(df)

    # ---------------- Phần 1: dữ liệu và nhãn ----------------
    master = pd.read_csv(MASTER, parse_dates=["date"]).set_index("date")
    h1 = pd.read_csv(ROOT / "data/processed/ohlcv_1h_clean.csv")
    h1["datetime"] = pd.to_datetime(h1["datetime"], utc=True)
    h1["d"] = h1["datetime"].dt.tz_localize(None).dt.floor("D")
    g = h1.groupby("d").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                            close=("close", "last"), n=("close", "size"), last_bar=("datetime", "max"))
    j = master.join(g, how="inner")
    diff = {c: float((j[f"ohlcv_{c}"] - j[c]).abs().max()) for c in ["open", "high", "low", "close"]}
    not_23 = j.index[j["last_bar"].dt.hour != 23].strftime("%Y-%m-%d").tolist()
    ties = int((df["close_next"] == df["ohlcv_close"]).sum())
    inc = df[df["ohlcv_n_bars_1h"] < 24]

    md += ["# Kiểm tra baseline OHLCV-only (góp ý 1 của thầy)", "",
           "Tạo bởi `baseline_check/src/baseline_check.py` trên data freeze v1.0 (sha256 khớp MANIFEST). "
           "Chỉ dùng 10 đặc trưng OHLCV của `tram_features_backtest`; không dùng On-chain, không dùng `ngoc_pipeline`.", "",
           "## Phần 1. Dữ liệu và nhãn", "",
           "**Nhãn.** Dòng ngày t có y_t = 1 nếu giá đóng cửa ngày t+1 > ngày t. Đặc trưng của dòng t chỉ dùng dữ liệu đến hết ngày t. Ví dụ quanh ngày sập 12/03/2020:", ""]
    ex = df.loc["2020-03-10":"2020-03-14", ["ohlcv_close", "close_next", "logret_lag1", "y"]].copy()
    ex.columns = ["close_t", "close_t+1", "logret_lag1 (return ngày t)", "y_t"]
    ex.index = ex.index.strftime("%Y-%m-%d")
    md += [ex.round(4).to_markdown(), "",
           "Đọc dòng 2020-03-11: giá đóng cửa 7.934,52 → ngày 12/03 đóng cửa 4.800,00, nên y = 0 (giảm). "
           "Đặc trưng của dòng này chỉ biết return của ngày 11/03 (+0,5%), chưa biết cú sập ngày 12/03.", "",
           "**Giá ngày so với nến 1h.** Gộp lại 78.761 nến 1h theo ngày UTC rồi so với `master_daily.csv`: "
           f"lệch tối đa open {diff['open']:.0e}, high {diff['high']:.0e}, low {diff['low']:.0e}, close {diff['close']:.0e} USD (chỉ là sai số làm tròn).", "",
           f"**Ngày giá đóng cửa không lấy từ nến 23:00 UTC:** {not_23}. Ngày này chỉ có 1 nến (00:00) vì Binance bảo trì, "
           "nên close của 08/02/2018 thực ra là giá lúc 01:00 UTC. Nhãn của 07/02 và 08/02/2018 vì vậy không đúng nghĩa \"cuối ngày\"; "
           "hai ngày này chỉ nằm trong train.", "",
           f"**Ngày giá đóng cửa bằng nhau (y không xác định):** {ties} ngày.", "",
           f"**Ngày thiếu nến 1h (< 24 nến) trong 2018–08/2026:** {len(inc)} ngày; "
           f"{int((inc.index >= '2020-01-01').sum())} ngày nằm trong giai đoạn test walk-forward. "
           "Giá đóng cửa của các ngày thiếu 1–10 nến vẫn là nến 23:00, nên nhãn vẫn đúng.", ""]

    # ---------------- Phần 2: EDA cho nhãn ----------------
    r = df["logret_lag1"]
    periods = {"2018–2019 (train fold 0)": ("2018-01-01", "2019-12-31"),
               "2020–08/2026 (test walk-forward)": WF_TEST,
               "Train baseline 2018–2021": SPLITS["train"], "Validation 2022–2023": SPLITS["val"],
               "Test baseline 2024–08/2026": SPLITS["test"]}
    erows = {}
    for name, (a, b) in periods.items():
        s = df.loc[a:b]
        pers = (s["y"] == s["up_today"]).mean()
        k = int((s["y"] == s["up_today"]).sum())
        erows[name] = {"số ngày": len(s), "tỷ lệ ngày tăng (%)": round(s["y"].mean() * 100, 1),
                       "ACF return trễ 1": round(s["logret_lag1"].autocorr(1), 3),
                       "ACF dấu return trễ 1": round(np.sign(s["logret_lag1"]).autocorr(1), 3),
                       "persistence đúng (%)": round(pers * 100, 1),
                       "p (persistence = 50%)": round(stats.binomtest(k, len(s)).pvalue, 4),
                       "độ lệch chuẩn return ngày (%)": round(s["logret_lag1"].std() * 100, 2)}
    et = pd.DataFrame(erows).T
    by_year = df.loc[WF_TEST[0]:WF_TEST[1]].groupby(lambda d: d.year).apply(
        lambda s: pd.Series({"tỷ lệ tăng (%)": s["y"].mean() * 100,
                             "persistence (%)": (s["y"] == s["up_today"]).mean() * 100,
                             "đảo chiều (%)": (s["y"] != s["up_today"]).mean() * 100})).round(1)
    se = np.sqrt(.25 / len(df.loc[WF_TEST[0]:WF_TEST[1]])) * 100
    pers_wf = erows["2020–08/2026 (test walk-forward)"]["persistence đúng (%)"]
    pers_f0 = erows["2018–2019 (train fold 0)"]["persistence đúng (%)"]
    pers_bt = erows["Test baseline 2024–08/2026"]["persistence đúng (%)"]
    strong = by_year.index[by_year["đảo chiều (%)"] >= 54].astype(str).tolist()
    weak = by_year.index[(by_year["đảo chiều (%)"] > 50) & (by_year["đảo chiều (%)"] < 52)].astype(str).tolist()
    rev = by_year.index[by_year["đảo chiều (%)"] < 50].astype(str).tolist()
    md += ["## Phần 2. EDA cho nhãn", "",
           "Chỉ dùng chính chuỗi giá và nhãn (không xem đặc trưng nào với nhãn).", "",
           et.to_markdown(), "",
           "**Phát hiện chính: hướng giá có xu hướng đảo chiều sau 1 ngày.** "
           f"Tự tương quan của dấu return là âm ở mọi giai đoạn, nên \"mai cùng chiều hôm nay\" chỉ đúng {pers_wf:.1f}% "
           f"trên 2020–08/2026, thấp hơn 50% có ý nghĩa thống kê. Ngược lại, luật \"mai ngược chiều hôm nay\" đúng {100-pers_wf:.1f}%. "
           f"Quy luật này đã thấy ngay trên train 2018–2019 (persistence {pers_f0:.1f}%), nên được phép dùng làm baseline mà không nhìn test.", "",
           "Theo năm trong giai đoạn test walk-forward:", "", by_year.to_markdown(), "",
           f"Quy luật yếu dần theo thời gian: đảo chiều đúng ≥ 54% ở {', '.join(strong)}; chỉ 50–52% ở {', '.join(weak)}"
           + (f"; dưới 50% ở {', '.join(rev)}" if rev else "") +
           f". Trên đoạn test của baseline (2024–08/2026) đảo chiều chỉ còn đúng {100-pers_bt:.1f}%.", "",
           f"**Ngưỡng nhiễu.** Với 2.434 ngày test, sai số chuẩn của DA khoảng {se:.1f} điểm. "
           f"Một mô hình đoán ngẫu nhiên có 95% khả năng rơi vào khoảng {50-1.96*se:.1f}–{50+1.96*se:.1f}%. "
           "Mọi DA 49–52% trong báo cáo giữa kỳ đều nằm trong vùng này.", ""]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    acf_r = [r.loc[WF_TEST[0]:WF_TEST[1]].autocorr(k) for k in range(1, 11)]
    acf_s = [np.sign(r.loc[WF_TEST[0]:WF_TEST[1]]).autocorr(k) for k in range(1, 11)]
    band = 1.96 / np.sqrt(len(r.loc[WF_TEST[0]:WF_TEST[1]]))
    for ax, v, t in [(axes[0], acf_r, "Tự tương quan của log return"), (axes[1], acf_s, "Tự tương quan của dấu return")]:
        ax.bar(range(1, 11), v, color=["tab:red" if abs(x) > band else "tab:grey" for x in v])
        ax.axhspan(-band, band, color="tab:blue", alpha=0.12, label="vùng nhiễu 95%")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("độ trễ (ngày)")
        ax.set_title(t + ", 2020–08/2026")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bc_01_acf.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    by_year[["persistence (%)", "đảo chiều (%)"]].plot.bar(ax=ax, color=["tab:grey", "tab:green"])
    ax.axhline(50, color="red", ls="--", lw=1)
    ax.set_ylim(35, 65)
    ax.set_ylabel("% ngày đoán đúng")
    ax.set_title("Hai luật đơn giản theo năm (test walk-forward)")
    ax.set_xticklabels(by_year.index.astype(str), rotation=0)
    ax.set_xlabel("năm")
    fig.tight_layout()
    fig.savefig(FIG / "bc_02_rules_by_year.png", dpi=110)
    plt.close(fig)
    md += ["![ACF](figures/bc_01_acf.png)", "", "![Luật theo năm](figures/bc_02_rules_by_year.png)", ""]

    # ---------------- Phần 3: baseline cơ bản ----------------
    tr, va, te = parts["train"], parts["val"], parts["test"]
    md += ["## Phần 3. Baseline cơ bản: train / validation / test", "",
           "| Đoạn | Từ | Đến | Số ngày |", "|---|---|---|---|"]
    for k in ["train", "val", "test"]:
        md.append(f"| {k} | {parts[k].index[0].date()} | {parts[k].index[-1].date()} | {len(parts[k])} |")
    md += ["", "Ngày cuối của train và validation bị bỏ (gap 1 ngày) vì nhãn của ngày đó dùng giá của đoạn sau. "
           "Chuẩn hoá (StandardScaler) nằm trong pipeline nên chỉ học trên dữ liệu train. "
           "Tham số C của Logistic Regression chọn trên validation theo log-loss; sau đó học lại trên train + validation "
           "và chạy test **đúng 1 lần**.", ""]

    col_sets = {"LR 1 đặc trưng (logret_lag1)": ["logret_lag1"], "LR 10 đặc trưng OHLCV": FEATURES}
    tuning_md, best_C = [], {}
    for name, cols in col_sets.items():
        t, C = tune_lr(tr, va, cols)
        best_C[name] = C
        default = t.loc[t["C"] == 1.0].iloc[0]
        chosen = t.loc[t["C"] == C].iloc[0]
        tuning_md += [f"**{name}.** C tốt nhất trên validation = {C:g} "
                      f"(val log-loss {chosen.val_log_loss:.4f}, val DA {chosen.val_DA*100:.1f}%; "
                      f"mặc định C = 1: val log-loss {default.val_log_loss:.4f}, val DA {default.val_DA*100:.1f}%).", "",
                      t.assign(train_DA=lambda x: (x.train_DA*100).round(1), val_DA=lambda x: (x.val_DA*100).round(1))
                       .round(4).to_markdown(index=False), ""]
    md += ["### Tuning (chỉ train → validation, chưa đụng test)", "",
           "Bảng ghi đủ mọi giá trị C đã chạy, kèm kết quả trước tuning (C = 1 mặc định) và sau tuning.", ""] + tuning_md
    md += ["**Lưu ý:** trên validation 2022–2023, không giá trị C nào của cả hai mô hình đạt DA trên 51% "
           "và val log-loss của mọi C đều cao hơn mức đoán 50/50 (0,6931). Mô hình học trên 2018–2021 "
           "không dùng được cho 2022–2023; kết quả trên test ở dưới cần đọc với lưu ý này.", ""]

    trva = pd.concat([tr, va.iloc[:]])                       # va đã bỏ ngày cuối
    rule_p, rule_info = rule_predictions(trva, te)
    test_rows, test_probs = {}, {}
    for name, p in rule_p.items():
        if not np.isnan(p).any():
            test_rows[name] = metrics(te["y"], p)
            test_probs[name] = p
    for name, cols in col_sets.items():
        mdl = lr(best_C[name]).fit(trva[cols], trva["y"].astype(int))
        p = mdl.predict_proba(te[cols])[:, 1]
        test_rows[f"{name} (C = {best_C[name]:g})"] = metrics(te["y"], p)
        test_probs[name] = p
        if name == "LR 10 đặc trưng OHLCV":
            coefs = pd.Series(mdl[-1].coef_[0], index=cols).sort_values()
            lr10_model, lr10_p = mdl, p
    tt = fmt_metrics(test_rows)
    md += ["### Kết quả trên test (2024 – 08/2026)", "",
           f"Trên train + validation: tỷ lệ ngày tăng {rule_info['up_share_train']*100:.1f}%, "
           f"persistence đúng {rule_info['persistence_acc_train']*100:.1f}% → chọn luật đảo chiều.", "",
           tt.to_markdown(), "",
           "Cột: DA = % ngày đoán đúng hướng; CI 95% và p theo kiểm định nhị thức so với 50%; bal_acc = trung bình độ chính xác của hai lớp; "
           "MCC từ −1 đến 1 (0 = ngẫu nhiên); AUC 0,5 = ngẫu nhiên; log-loss và Brier càng thấp càng tốt "
           "(đoán 50/50 mọi ngày: log-loss 0,6931, Brier 0,25); pct_pred_up = tỷ lệ ngày mô hình nói \"tăng\". "
           "Các luật B0 dùng p̂ = 0,99/0,01 nên log-loss của chúng rất lớn; chỉ so DA với chúng.", "",
           "Hệ số của LR 10 đặc trưng (đã chuẩn hoá; âm = đặc trưng càng lớn thì càng nghiêng về \"giảm\"):", "",
           coefs.round(4).to_frame("hệ số").to_markdown(), ""]

    # Kiểm chứng metric bằng tay
    sample = te.index[:6]
    rows, manual, sk = hand_check(te.loc[sample, "y"], lr10_p[:6])
    rows.index = sample.strftime("%Y-%m-%d")
    md += ["### Kiểm chứng metric bằng tay (6 ngày đầu của test, LR 10 đặc trưng)", "",
           rows.round(4).to_markdown(), "",
           f"Tính tay: DA = {manual['DA']:.4f}, log-loss = {manual['log_loss']:.4f}, Brier = {manual['brier']:.4f}. "
           f"sklearn: DA = {sk['DA']:.4f}, log-loss = {sk['log_loss']:.4f}, Brier = {sk['brier']:.4f}. Khớp.", "",
           "Cách đọc: log-loss của một ngày = −log(xác suất mô hình gán cho điều thật sự xảy ra). "
           "Ngày tăng (y = 1) mà mô hình nói p̂ = 0,52 thì phạt −log 0,52 = 0,654; nói p̂ = 0,48 thì phạt −log 0,48 = 0,734.", ""]

    # Phép thử
    sc, das = sanity_controls({"train": trva, "test": te}, FEATURES, best_C["LR 10 đặc trưng OHLCV"])
    md += ["### Phép thử để chắc kết quả có ý nghĩa", "",
           "| Phép thử | Kỳ vọng nếu pipeline đúng | Kết quả |", "|---|---|---|",
           f"| Xáo trộn nhãn train 200 lần, học lại, đo DA test | quanh 50% | TB {sc['shuffle_DA_mean']*100:.1f}%, 95% trong [{sc['shuffle_DA_p2.5']*100:.1f}; {sc['shuffle_DA_p97.5']*100:.1f}] |",
           f"| Rò rỉ cố ý: thêm return của ngày mai vào đặc trưng | gần 100% | {sc['leak_DA']*100:.1f}% |",
           f"| Đổi mục tiêu thành \"hôm nay tăng?\" (đặc trưng có chứa đáp án) | gần 100% | {sc['predict_today_DA']*100:.1f}% |",
           "", "Phép thử 2 và 3 cho thấy pipeline nhận ra tín hiệu khi tín hiệu có thật, và nhãn được căn đúng ngày. "
           "Phép thử 1 cho biết vùng \"ngẫu nhiên\" để so các DA ở trên.", ""]

    # ---------------- Phần 4: xem dự báo ----------------
    pt = te[["ohlcv_close", "close_next", "logret_lag1", "rsi_14", "rv_7d", "y"]].copy()
    pt["p_hat"] = lr10_p
    pt["dự báo"] = (pt["p_hat"] > .5).astype(int)
    pt["đúng"] = pt["dự báo"] == pt["y"]
    pt["return ngày mai (%)"] = (np.log(pt["close_next"] / pt["ohlcv_close"]) * 100)
    pt["đảo chiều"] = (1 - te["up_today"]).astype(int)
    pt.to_csv(OUT / "predictions_test.csv", date_format="%Y-%m-%d")

    cm = confusion_matrix(pt["y"].astype(int), pt["dự báo"])
    cm_df = pd.DataFrame(cm, index=["thật: giảm", "thật: tăng"], columns=["đoán: giảm", "đoán: tăng"])
    show = ["ohlcv_close", "logret_lag1", "rsi_14", "p_hat", "dự báo", "y", "đúng", "return ngày mai (%)"]
    rnd = pt.sample(10, random_state=SEED).sort_index()[show]
    conf_wrong = pt[~pt["đúng"]].assign(conf=lambda x: (x.p_hat - .5).abs()).nlargest(8, "conf")[show]
    big = pt.reindex(pt["return ngày mai (%)"].abs().nlargest(10).index)[show]
    for t in (rnd, conf_wrong, big):
        t.index = t.index.strftime("%Y-%m-%d")
    pt["|return ngày mai|"] = pt["return ngày mai (%)"].abs()
    pt["nhóm biến động"] = pd.qcut(pt["|return ngày mai|"], 4, labels=["25% ngày yên nhất", "Q2", "Q3", "25% ngày mạnh nhất"])
    by_move = pt.groupby("nhóm biến động", observed=True).agg(
        số_ngày=("đúng", "size"), LR_đúng=("đúng", "mean"),
        đảo_chiều_đúng=("đảo chiều", lambda s: (s == pt.loc[s.index, "y"]).mean()))
    by_move[["LR_đúng", "đảo_chiều_đúng"]] = (by_move[["LR_đúng", "đảo_chiều_đúng"]] * 100).round(1)
    pt["bin"] = pd.cut(pt["p_hat"], [0, .45, .48, .5, .52, .55, 1])
    calib = pt.groupby("bin", observed=True).agg(số_ngày=("y", "size"), p_hat_TB=("p_hat", "mean"),
                                                 tỷ_lệ_tăng_thật=("y", "mean")).round(3)
    agree = (pt["dự báo"] == pt["đảo chiều"]).mean()

    md += ["## Phần 4. Xem trực tiếp dự báo (LR 10 đặc trưng, test 2024 – 08/2026)", "",
           "Toàn bộ dự báo từng ngày: `reports/predictions_test.csv`.", "",
           "**Ma trận nhầm lẫn**", "", cm_df.to_markdown(), "",
           f"**p̂ rất sát 0,5:** trung vị {pt.p_hat.median():.3f}, 90% số ngày nằm trong "
           f"[{pt.p_hat.quantile(.05):.3f}; {pt.p_hat.quantile(.95):.3f}]. Mô hình gần như không tự tin ngày nào.", "",
           f"**LR có học được quy luật đảo chiều không?** Có: hệ số lớn nhất là logret_lag1 = {coefs['logret_lag1']:+.3f} "
           f"(âm = hôm nay tăng thì nghiêng về mai giảm), và LR đồng ý với luật đảo chiều ở {agree*100:.0f}% số ngày test. "
           "Khác biệt là LR dựa vào *độ lớn* return, còn luật đảo chiều chỉ dựa vào *dấu*.", "",
           "**10 ngày ngẫu nhiên**", "", rnd.round(3).to_markdown(), "",
           "**8 ngày sai mà mô hình tự tin nhất**", "", conf_wrong.round(3).to_markdown(), "",
           f"Cả {int((conf_wrong['logret_lag1'] < 0).sum())}/{len(conf_wrong)} ngày đều là ngày t vừa giảm mạnh "
           f"(logret_lag1 từ {conf_wrong['logret_lag1'].min()*100:.1f}% đến {conf_wrong['logret_lag1'].max()*100:.1f}%): "
           "mô hình đoán \"sẽ bật lại\" nhưng giá giảm tiếp. Mô hình tự tin nhất đúng vào lúc quy luật đảo chiều dễ sai nhất.", "",
           "**10 ngày biến động mạnh nhất (theo return ngày mai)**", "", big.round(3).to_markdown(), "",
           f"Mô hình đúng {int(big['đúng'].sum())}/10 ngày, tức không đoán được những ngày quyết định lời lỗ của backtest.", "",
           "**Độ chính xác theo mức biến động của ngày mai**", "", by_move.to_markdown(), "",
           "**Calibration: p̂ có khớp tỷ lệ tăng thật không?**", "", calib.to_markdown(), ""]

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(pt["p_hat"], bins=40, color="tab:blue")
    ax.axvline(.5, color="red", ls="--")
    ax.set_title("Phân phối p̂ của LR 10 đặc trưng trên test")
    ax.set_xlabel("p̂ (xác suất ngày mai tăng)")
    fig.tight_layout()
    fig.savefig(FIG / "bc_03_phat_hist.png", dpi=110)
    plt.close(fig)
    md += ["![p̂](figures/bc_03_phat_hist.png)", ""]

    # ---------------- Phần 5: walk-forward ----------------
    wf, days = walk_forward(df, {"LR 1 đặc trưng (logret_lag1), C = 1": ["logret_lag1"],
                                 "LR 10 đặc trưng OHLCV, C = 1": FEATURES})
    yw = df.loc[days, "y"]
    wrows = {k: metrics(yw, v.to_numpy()) for k, v in wf.items()}
    wt = fmt_metrics(wrows)
    n_folds_pers = sum(1 for s in range(0, len(days), 30)
                       if (df.iloc[np.arange(df.index.get_loc(days[s]) - 731, df.index.get_loc(days[s]) - 1)]
                           .pipe(lambda t: (t["y"] == t["up_today"]).mean())) >= 0.5)
    md += ["## Phần 5. Đối chiếu walk-forward (82 fold, 2020 – 08/2026)", "",
           "Cùng lịch với `ngoc_pipeline` (train 730, gap 1, test 30, bước 30). LR dùng C = 1 cố định, không tune "
           "(lịch này không có validation riêng). Luật đảo chiều chọn lại chiều trên từng cửa sổ train "
           f"({n_folds_pers}/82 fold cửa sổ train cho persistence ≥ 50% nên dùng persistence).", "",
           wt.to_markdown(), "",
           f"So với báo cáo giữa kỳ: Majority {wrows['Majority']['DA']*100:.2f}% và Persistence "
           f"{wrows['Persistence']['DA']*100:.2f}% khớp đúng Bảng 8.1 (hai code độc lập). "
           f"Luật đảo chiều đúng {wrows['Đảo chiều']['DA']*100:.2f}% (p = {wrows['Đảo chiều']['p_vs_50']:.3f}), "
           "cao hơn mọi mô hình có học trong Bảng 8.1 (49,0–51,9%). "
           f"LR chỉ với 1 đặc trưng ({wrows['LR 1 đặc trưng (logret_lag1), C = 1']['DA']*100:.2f}%) tốt hơn LR với 10 đặc trưng "
           f"({wrows['LR 10 đặc trưng OHLCV, C = 1']['DA']*100:.2f}%): thêm đặc trưng đang thêm nhiễu.", "",
           "**Thận trọng với luật đảo chiều:** chiều của luật được chọn lại trên từng cửa sổ train nên không nhìn test, "
           "nhưng quy luật được phát hiện khi làm EDA trên cả giai đoạn 2018–2026. Nên coi đây là một baseline cần thêm vào, "
           "không phải một phát hiện đã kiểm định độc lập.", ""]
    # ---------------- Tóm tắt đầu báo cáo ----------------
    lr10_key = f"LR 10 đặc trưng OHLCV (C = {best_C['LR 10 đặc trưng OHLCV']:g})"
    summary = ["", "## Tóm tắt", "",
               "| # | Câu hỏi | Kết quả |", "|---|---|---|",
               "| 1 | Dữ liệu và nhãn có đúng không? | Đúng. Giá ngày khớp nến 1h; nhãn kiểm bằng ví dụ; không có ngày giá bằng nhau. "
               "Chỉ ngày 08/02/2018 lấy giá đóng cửa lúc 01:00 UTC (nằm trong train) |",
               f"| 2 | Pipeline có nhận ra tín hiệu khi tín hiệu có thật? | Có. Rò rỉ cố ý → DA {sc['leak_DA']*100:.1f}%; "
               f"xáo nhãn → DA {sc['shuffle_DA_mean']*100:.1f}% |",
               f"| 3 | Có quy luật đơn giản nào không? | Có: hướng giá **đảo chiều sau 1 ngày**. Luật \"mai ngược hôm nay\" "
               f"đúng {wrows['Đảo chiều']['DA']*100:.2f}% trên walk-forward 2020–08/2026 (p = {wrows['Đảo chiều']['p_vs_50']:.3f}), "
               "cao hơn mọi mô hình có học trong báo cáo giữa kỳ, nhưng yếu dần và đảo ngược năm 2026 |",
               f"| 4 | Baseline OHLCV-only có ổn định? | Chưa. Walk-forward: LR 1 đặc trưng {wrows['LR 1 đặc trưng (logret_lag1), C = 1']['DA']*100:.2f}% "
               f"> LR 10 đặc trưng {wrows['LR 10 đặc trưng OHLCV, C = 1']['DA']*100:.2f}%. Chia cố định: validation dưới 51% với mọi C, "
               f"test {test_rows[lr10_key]['DA']*100:.2f}% (p = {test_rows[lr10_key]['p_vs_50']:.2f}) |",
               "| 5 | Dự báo trông thế nào? | p̂ gần như luôn trong 0,47–0,55; sai tự tin nhất đều sau ngày giảm mạnh; "
               f"đúng {int(big['đúng'].sum())}/10 ngày biến động mạnh nhất |", "",
               "**Đề xuất cho nhóm trước khi so On-chain:**", "",
               "1. Thêm luật đảo chiều vào B0. Mọi mô hình có học phải được so với nó, không chỉ với Majority.",
               "2. Dựng baseline OHLCV từ ít đặc trưng lên (bắt đầu từ logret_lag1), thêm từng đặc trưng và giữ lại cái nào "
               "cải thiện trên validation. Hiện 10 đặc trưng kém hơn 1 đặc trưng.",
               "3. Báo cáo kết quả theo từng năm. Tín hiệu thay đổi mạnh giữa các năm, con số gộp 2020–2026 che mất điều này.",
               "4. Chỉ so On-chain sau khi baseline OHLCV vượt luật đảo chiều một cách ổn định. Δ ±1 điểm hiện nằm trong nhiễu "
               "(sai số chuẩn của DA khoảng 1 điểm).", ""]
    md[3:3] = summary
    (OUT / "baseline_check.md").write_text("\n".join(md), encoding="utf-8")
    tt.to_csv(OUT / "test_metrics.csv")
    wt.to_csv(OUT / "walkforward_metrics.csv")
    print(tt.to_string())
    print(wt.to_string())
    print(sc)
    print(f"Đã lưu {OUT / 'baseline_check.md'}")


if __name__ == "__main__":
    main()
