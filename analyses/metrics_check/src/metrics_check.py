"""
metrics_check.py  –  Góp ý 2 của thầy (Vi)
"Kiểm tra lại định nghĩa, cách tính và ý nghĩa của các metrics. Đối chiếu bằng một vài ví dụ cụ thể,
 bảo đảm các bảng đang đo đúng bài toán dự báo hướng giá."

Kiểm trên file dự báo CHÍNH THỨC (pipeline/results/predictions/) và code tạo bảng (run_all.metrics),
không phải trên dữ liệu giả. Mọi phép kiểm là assert: sai thì script dừng, không in bảng.

Chạy từ thư mục gốc repo:
    python analyses/metrics_check/src/metrics_check.py
Đầu ra: analyses/metrics_check/reports/metrics_check.md

Bố cục:
  Phần 1  Nhãn trong file dự báo có đúng là hướng giá ngày mai không (mọi file, mọi dòng)
  Phần 2  Định nghĩa, mốc tham chiếu, chiều tốt và chiều của Δ cho từng metric
  Phần 3  Tính tay trên ngày thật, rồi khớp với run_all.metrics và với bảng e1_delta_draft.csv
  Phần 4  Δ: tính lại từ hai file và tách thành số ngày chỉ một bên đúng
  Phần 5  Ô không có nghĩa: log-loss, Brier của luật; AUC của Majority
  Phần 6  Backtest có dùng p̂ đúng ngày không
  Phần 7  Các bảng và các code độc lập có cho cùng con số không
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "reports"
sys.path[:0] = [str(ROOT / "pipeline/src"), str(ROOT / "features/src")]

from data import RESULTS, load_master  # noqa: E402
from models import RULES  # noqa: E402
from run_all import load_predictions, metrics  # noqa: E402
from backtest import run_backtest  # noqa: E402

PRED_DIR = RESULTS / "predictions"
TABLE = RESULTS / "tables/e1_delta_draft.csv"
EXAMPLE = ("lr", "ohlcv")                     # B1 Logistic Regression chỉ OHLCV
EPS = np.finfo(float).eps                     # sklearn log_loss cắt p̂ vào [eps, 1 − eps]


def close_series():
    m = load_master(verify=False).set_index("date")["ohlcv_close"]
    m.index = m.index.astype("datetime64[ns]")
    return m


def per_day(p):
    """Bảng từng ngày: dự báo, đúng/sai, phạt log-loss, phạt Brier, viết bằng công thức."""
    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    return pd.DataFrame({"date": p["date"].dt.strftime("%Y-%m-%d"), "y": y, "p_hat": ph,
                         "dự báo (p̂ > 0,5)": (ph > 0.5).astype(int),
                         "đúng": ((ph > 0.5).astype(int) == y).astype(int),
                         "−log p(y)": -np.log(np.clip(np.where(y == 1, ph, 1 - ph), EPS, 1)),
                         "(p̂ − y)²": (ph - y) ** 2})


def auc_by_pairs(y, p):
    """AUC = tỷ lệ cặp (ngày tăng, ngày giảm) mà ngày tăng có p̂ cao hơn; hoà tính nửa."""
    pos, neg = p[y == 1], p[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg)), int(gt), int(eq), len(pos) * len(neg)


def md_table(df, digits=4):
    return df.round(digits).to_markdown(index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(PRED_DIR)
    close = close_series()
    table = pd.read_csv(TABLE)
    md, summary = [], []

    # ---------------- Phần 1: nhãn ----------------
    nxt = close.shift(-1)
    truth = (nxt > close).astype(int)
    ref_dates = preds[EXAMPLE]["date"]
    for k, p in preds.items():
        d = p["date"].to_numpy()
        assert np.array_equal(d, ref_dates.to_numpy()), f"{k}: khác ngày test"
        assert (p["y_true"].to_numpy() == truth.loc[d].to_numpy()).all(), f"{k}: y_true sai hướng giá"
        assert (nxt.loc[d] != close.loc[d]).all(), "có ngày giá bằng nhau: nhãn không xác định"
    p = preds[EXAMPLE]
    ex_days = pd.to_datetime(["2020-03-11", "2020-03-12", "2021-11-09", "2021-11-10"])
    ex = pd.DataFrame({"date": ex_days.strftime("%Y-%m-%d"), "close_t": close.loc[ex_days].to_numpy(),
                       "close_t+1": nxt.loc[ex_days].to_numpy(),
                       "y_true trong file": p.set_index("date").loc[ex_days, "y_true"].to_numpy()})
    ex["tăng?"] = np.where(ex["close_t+1"] > ex["close_t"], "tăng", "giảm")
    n_files, n_days = len(preds), len(ref_dates)
    md += ["## Phần 1. Nhãn trong file dự báo có đúng là hướng giá ngày mai?", "",
           f"Kiểm **mọi dòng của {n_files} file** trong `pipeline/results/predictions/`: cùng {n_days} ngày test "
           f"({ref_dates.min().date()} → {ref_dates.max().date()}), và `y_true` của dòng ngày t bằng đúng "
           "1[close_{t+1} > close_t] tính lại từ `master_daily.csv`. Không có ngày nào giá đóng cửa bằng nhau. "
           "Ví dụ:", "", md_table(ex, 2), "",
           "Đọc dòng 2020-03-11: dòng này dự báo cho ngày 12/03 (ngày sập), giá đi từ 7.934,52 xuống 4.800, nên y = 0. "
           "Một mô hình dự báo đúng cú sập phải có p̂ < 0,5 ở dòng **11/03**, không phải dòng 12/03.", ""]
    summary.append(("Nhãn trong file dự báo là hướng giá ngày mai?",
                    f"Đúng ở mọi dòng của {n_files} file, {n_days} ngày; cùng ngày test giữa các file"))

    # ---------------- Phần 2: định nghĩa ----------------
    up = p["y_true"].mean()
    ref_rows = pd.DataFrame([
        {"metric": "DA (accuracy)", "công thức": "tỷ lệ ngày 1[p̂ > 0,5] = y",
         "đo gì trong bài toán hướng giá": "đoán đúng hướng bao nhiêu % số ngày",
         "mốc đoán bừa": f"≈ 50%; luôn đoán tăng: {up*100:.2f}%", "tốt khi": "cao", "Δ > 0 nghĩa là": "On-chain giúp"},
        {"metric": "Log-loss", "công thức": "trung bình −log p̂(nhãn thật)",
         "đo gì trong bài toán hướng giá": "xác suất có đáng tin không; phạt nặng khi tự tin mà sai",
         "mốc đoán bừa": f"p̂ = 0,5: {np.log(2):.4f}; p̂ = {up:.3f}: {log_loss(p['y_true'], np.full(n_days, up)):.4f}",
         "tốt khi": "thấp", "Δ > 0 nghĩa là": "On-chain làm XẤU đi"},
        {"metric": "Brier", "công thức": "trung bình (p̂ − y)²",
         "đo gì trong bài toán hướng giá": "như log-loss nhưng phạt nhẹ hơn khi sai tự tin",
         "mốc đoán bừa": "p̂ = 0,5: 0,2500", "tốt khi": "thấp", "Δ > 0 nghĩa là": "On-chain làm XẤU đi"},
        {"metric": "AUC", "công thức": "tỷ lệ cặp (ngày tăng, ngày giảm) mà ngày tăng có p̂ cao hơn",
         "đo gì trong bài toán hướng giá": "xếp hạng ngày tăng trên ngày giảm, không phụ thuộc ngưỡng 0,5",
         "mốc đoán bừa": "0,5", "tốt khi": "cao", "Δ > 0 nghĩa là": "On-chain giúp"},
    ])
    md += ["## Phần 2. Định nghĩa và cách đọc từng metric", "",
           f"Ngày test có y = 1 (giá ngày mai tăng) chiếm {up*100:.2f}%. Mọi metric dùng ngưỡng 0,5 "
           "(`run_all.metrics`): p̂ > 0,5 là đoán tăng; p̂ đúng bằng 0,5 tính là đoán giảm.", "",
           ref_rows.to_markdown(index=False), "",
           "**Hai điểm dễ đọc nhầm:**", "",
           "- DA và AUC: càng cao càng tốt, Δ dương là On-chain giúp. Log-loss và Brier: càng thấp càng tốt, "
           "**Δ âm** mới là On-chain giúp. Bảng `e1_delta_draft.csv` để chung các metric trong một cột `delta`, "
           "nên phải đọc dấu theo từng dòng.",
           "- DA chỉ xem p̂ nằm phía nào của 0,5; log-loss xem p̂ cách 0,5 bao xa. Một mô hình có thể DA trên 50% "
           "nhưng log-loss tệ hơn đoán 50/50 (Phần 3).", ""]

    # ---------------- Phần 3: tính tay ----------------
    pdays = per_day(p)
    pick = pd.concat([pdays[(pdays["y"] == a) & (pdays["đúng"] == b)].head(2) for a in (0, 1) for b in (0, 1)])
    pick = pick.sort_values("date")
    y8, p8 = pick["y"].to_numpy(), pick["p_hat"].to_numpy()
    hand8 = {"DA": pick["đúng"].mean(), "log-loss": pick["−log p(y)"].mean(), "Brier": pick["(p̂ − y)²"].mean(),
             "AUC": auc_by_pairs(y8, p8)[0]}
    sk8 = {"DA": ((p8 > .5) == y8).mean(), "log-loss": log_loss(y8, p8, labels=[0, 1]),
           "Brier": brier_score_loss(y8, p8), "AUC": roc_auc_score(y8, p8)}
    assert all(np.isclose(hand8[k], sk8[k]) for k in hand8), (hand8, sk8)
    _, gt8, eq8, n8 = auc_by_pairs(y8, p8)

    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    hand = {"da": pdays["đúng"].mean(), "log_loss": pdays["−log p(y)"].mean(), "brier": pdays["(p̂ − y)²"].mean(),
            "auc": auc_by_pairs(y, ph)[0]}
    pipe = metrics(p, EXAMPLE[0])
    tab = table[(table.model == EXAMPLE[0])].set_index("metric")[EXAMPLE[1]]
    for k in hand:
        assert np.isclose(hand[k], pipe[k], atol=1e-12), (k, hand[k], pipe[k])
        assert np.isclose(hand[k], tab[k], atol=1e-12), (k, hand[k], tab[k])
    full = pd.DataFrame({"metric": list(hand), "tính tay (công thức)": list(hand.values()),
                         "run_all.metrics": [pipe[k] for k in hand], "e1_delta_draft.csv": [tab[k] for k in hand]})
    md += ["## Phần 3. Tính tay trên ngày thật", "",
           f"File `{EXAMPLE[0]}_{EXAMPLE[1]}.csv` (B1 Logistic Regression, chỉ OHLCV). 8 ngày: 2 ngày cho mỗi tổ hợp "
           "(giảm/tăng) × (sai/đúng).", "", md_table(pick), "",
           f"Trên 8 ngày này: DA = {hand8['DA']:.4f}; log-loss = {hand8['log-loss']:.4f}; Brier = {hand8['Brier']:.4f}; "
           f"AUC = {hand8['AUC']:.4f} ({gt8} trong {n8} cặp (ngày tăng, ngày giảm) có ngày tăng được p̂ cao hơn"
           + (f", {eq8} cặp hoà" if eq8 else "") + "). sklearn cho cùng 4 con số.", "",
           f"Trên đủ {n_days} ngày, công thức tay, hàm `run_all.metrics` và bảng `e1_delta_draft.csv` khớp nhau "
           "(lệch ≤ 1e-12):", "", md_table(full, 6), "",
           f"**Ví dụ cho điểm dễ nhầm ở Phần 2:** B1 có DA {hand['da']*100:.2f}% (trên 50%) nhưng log-loss "
           f"{hand['log_loss']:.4f} > {np.log(2):.4f}: xác suất của B1 tệ hơn việc luôn nói 50/50.", ""]
    summary.append(("Code tạo bảng tính đúng định nghĩa?",
                    "Đúng: tính tay trên 8 ngày khớp sklearn; trên đủ 2.434 ngày khớp run_all.metrics và e1_delta_draft.csv"))

    # ---------------- Phần 4: Δ ----------------
    drows = []
    for name in sorted({m for m, _ in preds}):
        if (name, "ohlcv_onchain") not in preds:
            continue
        a, b = preds[(name, "ohlcv")], preds[(name, "ohlcv_onchain")]
        ma, mb = metrics(a, name), metrics(b, name)
        t = table[table.model == name].set_index("metric")
        for k in ma:
            d = mb[k] - ma[k]
            assert (np.isnan(d) and np.isnan(t.loc[k, "delta"])) or np.isclose(d, t.loc[k, "delta"], atol=1e-12), (name, k)
        ha, hb = ((a.p_hat > .5).astype(int) == a.y_true), ((b.p_hat > .5).astype(int) == b.y_true)
        n_a, n_b = int((ha & ~hb).sum()), int((~ha & hb).sum())
        assert np.isclose((n_b - n_a) / len(a), mb["da"] - ma["da"])
        drows.append({"model": name, "ΔDA (điểm %)": (mb["da"] - ma["da"]) * 100,
                      "chỉ OHLCV đúng": n_a, "chỉ +On-chain đúng": n_b, "(b − a)/N × 100": (n_b - n_a) / len(a) * 100,
                      "Δ log-loss": mb["log_loss"] - ma["log_loss"]})
    dt = pd.DataFrame(drows)
    md += ["## Phần 4. Δ được tính đúng không?", "",
           "Δ = metric(OHLCV + On-chain) − metric(chỉ OHLCV), cùng mô hình, cùng 2.434 ngày. Tính lại từ hai file dự báo "
           "khớp cột `delta` của `e1_delta_draft.csv` cho mọi mô hình và metric. ΔDA còn tách được thành số ngày chỉ một "
           "bên đúng: ΔDA = (chỉ +On-chain đúng − chỉ OHLCV đúng) / N. Đây cũng là hai số mà kiểm định McNemar dùng.", "",
           md_table(dt, 4), "",
           "Đọc dòng `lr`: thêm On-chain làm đúng thêm ở một số ngày nhưng sai ở gần chừng ấy ngày khác; ΔDA chỉ là "
           "hiệu của hai con số lớn gần bằng nhau, nên dễ đổi dấu khi đổi mô hình hoặc giai đoạn.", ""]
    summary.append(("Δ có đúng dấu, đúng cách tính?",
                    "Đúng: tính lại từ hai file khớp bảng; ΔDA = (số ngày chỉ +On-chain đúng − số ngày chỉ OHLCV đúng)/N"))

    # ---------------- Phần 5: ô không có nghĩa ----------------
    pr = preds[("b0rev", "ohlcv")]
    wrong = 1 - ((pr.p_hat > .5).astype(int) == pr.y_true).mean()
    raw_ll = log_loss(pr.y_true, pr.p_hat, labels=[0, 1])
    raw_brier = brier_score_loss(pr.y_true, pr.p_hat)
    assert np.isclose(raw_ll, wrong * -np.log(EPS)) and np.isclose(raw_brier, wrong)
    yr, pr_ = pr.y_true.to_numpy(), pr.p_hat.to_numpy()
    bal = 0.5 * (((pr_ > .5) & (yr == 1)).sum() / (yr == 1).sum() + ((pr_ <= .5) & (yr == 0)).sum() / (yr == 0).sum())
    assert np.isclose(roc_auc_score(yr, pr_), bal)
    pm = preds[("b0maj", "ohlcv")]
    n_levels = pm.groupby("fold")["p_hat"].nunique().max()
    t_rules = table[table.model.isin(list(RULES) + ["b0maj"])][["model", "metric", "ohlcv"]]
    md += ["## Phần 5. Ô nào trong bảng không có nghĩa", "",
           "**Log-loss và Brier của luật** (`b0pers`, `b0rev`). Luật cho p̂ chỉ là 0 hoặc 1. Khi đó:",
           f"- log-loss = (tỷ lệ ngày sai) × (−log eps) = {wrong:.4f} × {-np.log(EPS):.2f} = {raw_ll:.2f}. "
           "Con số chỉ đo tỷ lệ ngày sai nhân một hằng số do sklearn chọn; trước khi sửa, bảng in 19,28 cho `b0pers` "
           "(và 16,85 cho `b0rev` khi mới thêm), dễ bị đọc như mô hình cực tệ.",
           f"- Brier = tỷ lệ ngày sai = {raw_brier:.4f}, tức 1 − DA, không thêm thông tin.",
           f"- AUC của luật bằng đúng balanced accuracy ({bal:.4f}), vẫn đọc được.", "",
           f"**AUC của `b0maj`.** p̂ của Majority không đổi trong mỗi fold ({n_levels} giá trị/fold), chỉ khác giữa các fold. "
           "AUC khi đó xếp hạng các fold theo tỷ lệ tăng của cửa sổ train, không phải xếp hạng các ngày.", "",
           "**Đã sửa** `run_all.metrics`: các ô này ghi NaN (ô trống) thay vì một con số. Bảng hiện tại:", "",
           md_table(t_rules), ""]
    summary.append(("Ô nào không nên đọc?",
                    "Log-loss, Brier của luật (log-loss của `b0pers` từng in 19,28) và AUC của Majority: "
                    "đã đổi thành ô trống"))

    # ---------------- Phần 6: backtest ----------------
    bt = run_backtest(close, p.set_index("date")["p_hat"])
    d0 = p["date"].iloc[:4]
    rows = []
    for t in d0:
        t1 = t + pd.Timedelta(days=1)
        rows.append({"dòng dự báo t": t.strftime("%Y-%m-%d"), "p̂_t": p.set_index("date").loc[t, "p_hat"],
                     "y_t": int(p.set_index("date").loc[t, "y_true"]), "ngày giữ vị thế": t1.strftime("%Y-%m-%d"),
                     "vị thế t+1": bt.loc[t1, "position"], "return t+1 (%)": bt.loc[t1, "ret"] * 100})
    btx = pd.DataFrame(rows)
    sign_ok = (bt["ret"].shift(-1).dropna() > 0).astype(int)
    common = p.set_index("date").index.intersection(sign_ok.index)
    assert (sign_ok.loc[common].to_numpy() == p.set_index("date").loc[common, "y_true"].to_numpy()).all()
    assert ((bt["position"].shift(-1).loc[common] == 1) == (p.set_index("date").loc[common, "p_hat"] > .5)).all()
    md += ["## Phần 6. Backtest dùng p̂ đúng ngày không", "",
           "`features/src/backtest.py`: p̂ ở dòng ngày t quyết định vị thế của ngày t+1, lời lỗ là return "
           "của ngày t+1. Kiểm trên đủ giai đoạn test: (a) vị thế ngày t+1 = 1 đúng khi p̂_t > 0,5; (b) return ngày t+1 "
           "> 0 đúng khi y_t = 1. Nên backtest đo đúng cùng bài toán với DA, chỉ khác là mỗi ngày có trọng số bằng "
           "độ lớn return.", "", md_table(btx, 4), ""]
    summary.append(("Backtest có lệch ngày?", "Không: p̂_t → vị thế t+1 → return t+1, và dấu return t+1 khớp y_t ở mọi ngày"))

    # ---------------- Phần 7: nhất quán giữa các bảng ----------------
    yt = pd.read_csv(RESULTS / "tables/e1_by_year.csv")
    vs = pd.read_csv(RESULTS / "tables/e1_vs_reversal.csv")
    for (name, config), q in preds.items():
        s = yt[(yt.model == name) & (yt.config == config)]
        assert s.n_days.sum() == len(q)
        assert np.isclose((s.da * s.n_days).sum() / len(q), metrics(q, name)["da"])
    for _, r in vs.iterrows():
        assert np.isclose(r.da_model, metrics(preds[(r.model, r.config)], r.model)["da"])
        assert np.isclose(r.da_rule, metrics(preds[("b0rev", "ohlcv")], "b0rev")["da"])
    bc = pd.read_csv(ROOT / "analyses/baseline_check/reports/walkforward_metrics.csv", index_col=0)
    indep = pd.DataFrame([
        {"luật": "Majority", "pipeline": metrics(preds[("b0maj", "ohlcv")])["da"] * 100,
         "baseline_check (code riêng)": bc.loc["Majority", "DA (%)"]},
        {"luật": "Persistence", "pipeline": metrics(preds[("b0pers", "ohlcv")])["da"] * 100,
         "baseline_check (code riêng)": bc.loc["Persistence", "DA (%)"]},
        {"luật": "Đảo chiều", "pipeline": metrics(preds[("b0rev", "ohlcv")])["da"] * 100,
         "baseline_check (code riêng)": bc.loc["Đảo chiều", "DA (%)"]}])
    assert np.allclose(indep["pipeline"].round(2), indep["baseline_check (code riêng)"])
    md += ["## Phần 7. Các bảng có cho cùng con số không", "",
           "- `e1_by_year.csv`: với mọi mô hình và cấu hình, số ngày các năm cộng lại đủ 2.434 và DA gộp theo trọng số "
           "số ngày bằng đúng DA trong `e1_delta_draft.csv`.",
           "- `e1_vs_reversal.csv`: DA của mô hình và của luật đảo chiều khớp `run_all.metrics`.",
           "- Hai code viết độc lập (`pipeline` và `baseline_check`) cho cùng DA của ba luật:", "",
           md_table(indep, 2), ""]
    summary.append(("Các bảng có nhất quán?",
                    "Có: theo năm, so với luật đảo chiều và bảng Δ khớp nhau; hai code độc lập cho cùng DA của 3 luật"))

    head = ["# Kiểm tra metric (góp ý 2 của thầy)", "",
            "Tạo bởi `analyses/metrics_check/src/metrics_check.py`. Kiểm trên file dự báo chính thức của `pipeline` và code "
            "tạo bảng `run_all.metrics`. Mọi phép kiểm là `assert`; báo cáo này chỉ được ghi khi tất cả đều đúng.", "",
            "## Tóm tắt", "", "| # | Câu hỏi | Kết quả |", "|---|---|---|"]
    head += [f"| {i} | {q} | {a} |" for i, (q, a) in enumerate(summary, 1)]
    head += ["", "**Cách đọc bảng kết quả từ nay:** DA và log-loss là hai metric chính, luôn in kèm mốc (DA 50%, "
             "luôn đoán tăng 50,90%, luật đảo chiều 53,25%; log-loss 0,6931). Δ log-loss âm mới là On-chain giúp. "
             "Không đọc log-loss, Brier của luật và AUC của Majority.", ""]
    (OUT / "metrics_check.md").write_text("\n".join(head + md), encoding="utf-8")
    print("\n".join(head))
    print(f"Đã lưu {OUT / 'metrics_check.md'}")


if __name__ == "__main__":
    main()
