"""
report_sections.py  –  Sinh bản nháp mục 5 (Baselines) và mục 8 (Kết quả trước/sau tuning) cho báo cáo giữa kỳ,
TRỰC TIẾP từ kết quả của simple_split.py (analyses/simple_split/reports/*.csv) và cấu hình trong simple_split.py.
Không gõ tay con số nào. Chỉ OHLCV, chỉ cách chia train/val/test chung; không có kết quả On-chain, không có XGBoost
(không chạy trong simple_split).

    python analyses/simple_split/src/report_sections.py --test   # test
    python analyses/simple_split/src/report_sections.py          # ghi docs/midterm_review3_baseline_sections.md
Chạy lại sau mỗi lần chạy simple_split.py.
"""
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simple_split as ss                                              # noqa: E402  (nạp torch trước sklearn)
import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402

REP = ss.ROOT / "analyses" / "simple_split" / "reports"
OUT_MD = ss.ROOT / "docs" / "midterm_review3_baseline_sections.md"
LEARNED = ["LR", "SVM RBF", "RF", "LSTM"]
IDS = {"LR": "B1", "RF": "B2", "SVM RBF": "B3", "LSTM": "B5"}
RULES = ["Majority", "Buy-and-hold", "Persistence", "Đảo chiều"]
NOTE_ONCHAIN = "Kết quả có On-chain tạm hoãn cho đến khi baseline OHLCV ổn định (góp ý 1, 4)."
COIN = 0.6931          # log-loss khi đoán 50/50 mọi ngày (ln 2)


def vn(x, d=2):
    """Số kiểu Việt: dấu phẩy thập phân."""
    return f"{x:.{d}f}".replace(".", ",")


def _val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:g}".replace(".", ",")
    return str(v)


def _params(d):
    return "; ".join(f"{k} = {_val(v)}" for k, v in d.items()) or "mặc định sklearn"


def _grid_desc(grid):
    keys = list(grid[0])
    out = []
    for k in keys:
        vals = list(dict.fromkeys(_val(g[k]) for g in grid))
        if len(vals) > 1:
            out.append(f"{k} ∈ {{{'; '.join(vals)}}}")
    return " × ".join(out)


def section5():
    _, parts = ss.load()
    rows = ["## 5. Baselines", "",
            "Mọi baseline chỉ dùng 10 đặc trưng OHLCV (mục 4.1), cùng dữ liệu data freeze v1.0 và cùng một cách chia "
            "theo thời gian. Ngày cuối của train và validation bị bỏ (gap 1 ngày) vì nhãn của ngày đó dùng giá của "
            "đoạn sau.", "",
            "| Đoạn | Từ | Đến | Số ngày | Dùng để |", "|---|---|---|---|---|"]
    use = {"train": "học", "val": "chọn tham số", "test": "chấm 1 lần"}
    for k, v in parts.items():
        rows.append(f"| {k} | {v.index.min().date()} | {v.index.max().date()} | {len(v)} | {use[k]} |")
    rows += ["", "Bảng 5.1. Cách chia train / validation / test", "",
             "| ID | Mô hình | Trước tune | Grid sau tune | Số tổ hợp | Seed |", "|---|---|---|---|---|---|",
             "| B0 | Majority; Persistence; Buy-and-hold; Đảo chiều | không có tham số | – | – | – |"]
    names = {"LR": "Logistic Regression", "RF": "Random Forest", "SVM RBF": "SVM kernel RBF", "LSTM": "LSTM 1 lớp"}
    for m in ["LR", "RF", "SVM RBF", "LSTM"]:
        M = ss.MODELS[m]
        before = _params(M["before"]) + ("; n_estimators = 500" if m == "RF" else "")
        seed = ", ".join(map(str, ss.SEEDS)) + " (trung bình p̂)" if m in ("RF", "LSTM") else "không (tất định)"
        rows.append(f"| {IDS[m]} | {names[m]} | {before} | {_grid_desc(M['grid'])} | {len(M['grid'])} | {seed} |")
    rows += ["", "Bảng 5.2. Baseline, tham số trước tune và grid sau tune", "",
             "- B0: Majority đoán theo tỷ lệ ngày tăng của train + validation; Persistence: mai cùng hướng hôm nay; "
             "Buy-and-hold: luôn đoán tăng; Đảo chiều: mai ngược hướng hôm nay (chiều luật chọn trên train + validation).",
             "- LR: tập đặc trưng `sign` = 1[return hôm nay > 0], `lag1` = return hôm nay, `ohlcv10` = 10 đặc trưng; "
             "bản sau tune có winsorize 1%/99% trước StandardScaler. SVM: xác suất hiệu chỉnh sigmoid. "
             "RF: 500 cây, max_depth không giới hạn. LSTM: early stopping (patience 5) trên 20% cuối phần đang học, có "
             "gap 1 ngày, rồi học lại trên toàn bộ phần đang học.",
             "- Chọn tham số: log-loss nhỏ nhất trên validation (làm tròn 4 chữ số; hoà thì chọn tổ hợp đứng trước, tức "
             "đơn giản hơn). Sau đó học lại trên train + validation với tham số đã chọn và chấm test đúng 1 lần. "
             "StandardScaler fit trên phần đang học.",
             "- B4 XGBoost: chưa chạy ở cách chia này.",
             f"- {NOTE_ONCHAIN}", ""]
    return "\n".join(rows)


def _ci(s):
    return s.replace(".", ",")


def _p(v):
    return "–" if pd.isna(v) else ("< 0,001" if v < 0.001 else vn(v, 3))


def section8():
    tm = pd.read_csv(REP / "test_metrics.csv", index_col=0)
    pr = pd.read_csv(REP / "predictions_test.csv", parse_dates=["date"]).set_index("date")
    _, parts = ss.load()
    te = parts["test"]
    assert (pr.index == te.index).all() and (pr["y"].to_numpy() == te["y"].to_numpy()).all()
    n = int(tm["n"].iloc[0])
    rows = ["## 8. Kết quả", "",
            f"Mọi số liệu tính trên đoạn test {te.index.min().date()} → {te.index.max().date()} ({n} ngày; "
            f"tỷ lệ ngày tăng {vn(te['y'].mean() * 100, 1)}%), sau khi học lại trên train + validation. "
            "Tạo bởi `analyses/simple_split/src/simple_split.py`.", "",
            "### 8.1. So sánh trên test, trước và sau tuning", "",
            "| Mô hình | DA (%) | CI 95% DA | p (DA = 50%) | Log-loss | % ngày đoán tăng | McNemar p so với Đảo chiều |",
            "|---|---|---|---|---|---|---|"]
    for name, r in tm.iterrows():
        ll = "–" if name in RULES[1:] else vn(r["log_loss"], 4)
        rows.append(f"| {name} | {vn(r['DA (%)'])} | {_ci(r['CI 95% DA'])} | {_p(r['p (DA = 50%)'])} | {ll} | "
                    f"{vn(r['pct_pred_up'] * 100, 1)} | {_p(r['McNemar p (so với đảo chiều)'])} |")
    rows += ["", f"Bảng 8.1. {n} ngày test. Log-loss của Persistence, Buy-and-hold, Đảo chiều không tính vì p̂ chỉ là "
             f"0/1. Đoán 50/50 mọi ngày có log-loss {vn(COIN, 4)}. Với {n} ngày, sai số chuẩn của DA khoảng "
             f"{vn(np.sqrt(.25 / n) * 100, 1)} điểm.", "",
             "### 8.2. Tuning trên validation (chưa dùng test)", "",
             "| Mô hình | Tham số được chọn | Val log-loss | Val DA (%) | Số tổ hợp có val log-loss < 0,6931 |",
             "|---|---|---|---|---|"]
    tun = {}
    for m in LEARNED:
        t = pd.read_csv(REP / f"tuning_{m.replace(' ', '_')}.csv")
        tun[m] = t
        c = t[t["chosen"]].iloc[0]
        prm = {k: c[k] for k in t.columns if k not in
               ("train_log_loss", "val_log_loss", "train_DA", "val_DA", "chosen", "clip")}
        rows.append(f"| {m} | {_params(prm)} | {vn(c['val_log_loss'], 4)} | {vn(c['val_DA'] * 100, 1)} | "
                    f"{(t['val_log_loss'] < COIN).sum()}/{len(t)} |")
    rows += ["", "Bảng 8.2. Bảng đầy đủ mọi tổ hợp: `analyses/simple_split/reports/tuning_<mô hình>.csv`.", "",
             "### 8.3. Xem trực tiếp dự báo (bản sau tune)", "",
             "| Mô hình | Thật giảm, đoán giảm | Thật giảm, đoán tăng | Thật tăng, đoán giảm | Thật tăng, đoán tăng | "
             "Cùng hướng Đảo chiều (%) | Đúng / 10 ngày biến động mạnh nhất |",
             "|---|---|---|---|---|---|---|"]
    y = te["y"].to_numpy(int)
    big = te["ret_next"].abs().nlargest(10).index
    rev = pr["Đảo chiều"].to_numpy() > .5
    for m in LEARNED:
        d = pr[f"{m} sau tune"].to_numpy() > .5
        cm = [int(((y == a) & (d == b)).sum()) for a in (0, 1) for b in (0, 1)]
        okb = int(((pr.loc[big, f"{m} sau tune"] > .5) == pr.loc[big, "y"].astype(bool)).sum())
        rows.append(f"| {m} | {cm[0]} | {cm[1]} | {cm[2]} | {cm[3]} | {vn((d == rev).mean() * 100, 1)} | {okb}/10 |")
    rows += ["", "Bảng 8.3. Ma trận nhầm lẫn, mức trùng hướng với luật Đảo chiều, và số ngày đoán đúng trong 10 ngày có "
             "return ngày mai lớn nhất (theo trị tuyệt đối). Danh sách từng ngày: `analyses/simple_split/reports/simple_split.md`, "
             "mục 3.", "", "### 8.4. Nhận xét", ""]
    rows += [f"- {x}" for x in _observations(tm, tun)] + [""]
    return "\n".join(rows)


def _observations(tm, tun):
    """Nhận xét sinh từ số liệu; câu nào cũng có số."""
    obs = []
    lo = tm["CI 95% DA"].str.extract(r"\[([\d.]+);\s*([\d.]+)\]").astype(float)
    k = int(((lo[0] <= 50) & (lo[1] >= 50)).sum())
    obs.append(f"{k}/{len(tm)} dòng có CI 95% của DA chứa 50%.")
    best = tm["DA (%)"].idxmax()
    obs.append(f"DA cao nhất: {best} {vn(tm.loc[best, 'DA (%)'])}% (p = {_p(tm.loc[best, 'p (DA = 50%)'])}); "
               f"luật Đảo chiều {vn(tm.loc['Đảo chiều', 'DA (%)'])}%.")
    worse = [m for m in LEARNED if tm.loc[f"{m} sau tune", "DA (%)"] < tm.loc[f"{m} trước tune", "DA (%)"]]
    detail = "; ".join(f"{m} {vn(tm.loc[f'{m} trước tune', 'DA (%)'])} → {vn(tm.loc[f'{m} sau tune', 'DA (%)'])}"
                       for m in worse)
    obs.append(f"DA sau tune thấp hơn trước tune ở {len(worse)}/{len(LEARNED)} mô hình"
               + (f": {detail}" if worse else "") + ".")
    below = [f"{m} {vn(tm.loc[f'{m} sau tune', 'log_loss'], 4)}" for m in LEARNED
             if tm.loc[f"{m} sau tune", "log_loss"] < COIN]
    obs.append(f"Trên test, bản sau tune có log-loss thấp hơn {vn(COIN, 4)}: " + ("; ".join(below) if below else "không mô hình nào") + ".")
    vb = "; ".join(f"{m} {(t['val_log_loss'] < COIN).sum()}/{len(t)}" for m, t in tun.items())
    obs.append(f"Trên validation, số tổ hợp có log-loss thấp hơn {vn(COIN, 4)}: {vb}.")
    mc = tm.drop(index="Đảo chiều")["McNemar p (so với đảo chiều)"]
    obs.append(f"So với luật Đảo chiều, McNemar p nhỏ nhất là {_p(mc.min())} ({mc.idxmin()}).")
    return obs


def edits():
    """Những chỗ phải sửa ở các mục khác để báo cáo không tự mâu thuẫn. Văn bản cố định, không thêm kết quả."""
    items = [
        ("Mục 1", "Giữ bài toán Δ On-chain và RQ1–RQ3 làm mục tiêu đề tài; thêm một câu: giữa kỳ tập trung hoàn thiện "
                  "baseline OHLCV theo góp ý của giảng viên, phần On-chain làm sau khi baseline ổn định."),
        ("Mục 4.2", "Thêm cách chia cố định train 2018–2021 / validation 2022–2023 / test 2024 – 08/2026 dùng cho "
                    "baseline giữa kỳ (mục 5). Giữ đoạn walk-forward và ghi là protocol cho các thí nghiệm sau."),
        ("Mục 4.3–4.4", "Regime HMM, kiểm định Δ, Diebold–Mariano, Holm, placebo: chuyển thành kế hoạch (chưa dùng ở giữa "
                        "kỳ). Giữ McNemar và CI của DA vì mục 8 có dùng."),
        ("Bảng 4.1", "Dòng MCC: đổi \"Chưa triển khai\" thành đã có (simple_split và baseline_check có tính MCC). "
                     "Dòng F1 macro: vẫn chưa có trong simple_split, giữ \"chưa triển khai\"."),
        ("Mục 7", "E0: (a) xong; (b) xáo nhãn đã chạy trong baseline_check (cho LR); (c) chia cố định đã chạy (mục 8). "
                  "E1, E2, E3, E4: đổi trạng thái thành \"tạm hoãn đến khi baseline OHLCV ổn định\"."),
        ("Mục 8", "Thay toàn bộ 8.1–8.4 cũ (2.434 ngày, có On-chain, backtest, regime) bằng mục 8 mới ở trên."),
        ("Mục 9", "Bỏ câu \"B1, B2, B4 hiện là bản chạy thử\" và \"B5 không tune bằng grid\" (đã tune, mục 5). "
                  "Thêm: test chỉ 973 ngày nên CI của DA rộng; B4 XGBoost chưa chạy ở cách chia này; kết quả mới có một "
                  "lần chia, chưa đối chiếu walk-forward. Giữ các hạn chế về dữ liệu; câu về Flows và regime chuyển "
                  "sang phần kế hoạch."),
        ("Mục 10", "Thay kết luận \"On-chain không cải thiện\" bằng kết luận rút từ mục 8.4 (chỉ về baseline OHLCV). "
                   "Hướng tiếp theo: chạy B4, đối chiếu walk-forward, rồi mới thêm On-chain theo từng nhóm."),
        ("Phụ lục B", "Bỏ (kết quả E3 theo regime)."),
        ("Phụ lục C", "Thay grid cũ bằng Bảng 5.2."),
    ]
    return "\n".join(["## Các chỗ cần sửa ở phần còn lại của báo cáo", ""] + [f"- **{k}:** {v}" for k, v in items]) + "\n"


def main():
    s5, s8, ed = section5(), section8(), edits()
    head = ["# Bản nháp mục 5 và mục 8 (báo cáo giữa kỳ)", "",
            f"Sinh ngày {date.today():%d/%m/%Y} bởi `python analyses/simple_split/src/report_sections.py` từ kết quả trong "
            "`analyses/simple_split/reports/`. Chạy lại lệnh này sau mỗi lần chạy `simple_split.py`. Mọi con số lấy từ file "
            "kết quả; không có số gõ tay.", ""]
    OUT_MD.write_text("\n".join(head) + "\n" + s5 + "\n" + s8 + "\n" + ed, encoding="utf-8")
    print(f"Đã lưu {OUT_MD}")


def _tests():
    s5 = section5()
    for v in ["973", "1460", "729", "42, 43, 44", "B4 XGBoost: chưa chạy"]:
        assert v in s5, v
    for m, n in [("LR", 300), ("SVM RBF", 30), ("RF", 24), ("LSTM", 16)]:
        line = next(x for x in s5.splitlines() if x.startswith(f"| {IDS[m]} "))
        assert f"| {n} |" in line, (m, line)
    assert s5.replace(NOTE_ONCHAIN, "").count("On-chain") == 0
    # Task 2: mục 8 khớp CSV, không có On-chain / Δ / XGBoost
    s8 = section8()
    tm = pd.read_csv(REP / "test_metrics.csv", index_col=0)
    lines = s8.splitlines()
    for name, r in tm.iterrows():
        line = next(x for x in lines if x.startswith(f"| {name} |"))
        assert f"| {r['DA (%)']:.2f}".replace(".", ",") + " |" in line, (name, line)
        if name not in RULES[1:]:
            assert f"| {r['log_loss']:.4f}".replace(".", ",") + " |" in line, (name, line)
    for m in LEARNED:
        t = pd.read_csv(REP / f"tuning_{m.replace(' ', '_')}.csv")
        c = t[t["chosen"]].iloc[0]
        line = next(x for x in lines if x.startswith(f"| {m} |"))
        assert f"{(t['val_log_loss'] < COIN).sum()}/{len(t)}" in line, (m, line)
        assert f"| {c['val_log_loss']:.4f}".replace(".", ",") + " |" in line, (m, line)
    for bad in ["On-chain", "Δ", "XGBoost"]:
        assert bad not in s8, bad
    # Task 3: danh sách sửa đủ mục, không thêm con số kết quả mới
    ed = edits()
    for sec in ["Mục 1", "Mục 4.2", "Mục 4.3–4.4", "Mục 7", "Mục 9", "Mục 10", "Phụ lục B", "Phụ lục C", "Bảng 4.1"]:
        assert sec in ed, sec
    for num in re.findall(r"\d+,\d+", ed):
        assert num in s8 or num in s5, num
    print("Tất cả test report_sections đều PASS")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _tests()
    else:
        main()
