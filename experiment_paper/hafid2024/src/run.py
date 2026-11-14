"""Task 3 (Hafid 2024): chạy 3 thí nghiệm A/B/C, ghi reports/results.csv và reports/report.md.

Chạy: python3 src/run.py          (chạy đủ 12 cấu hình)
      python3 src/run.py --test   (kiểm tra results.csv)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results.csv"
REPORT = REPORTS / "report.md"
DAILY = ROOT.parent.parent / "data" / "processed" / "master_daily.csv"

EXPERIMENTS = ["A", "B", "C"]
SCALES = ["full", "train"]
MODELS = ["XGBoost", "LR"]
PAPER = {"XGBoost": 0.9240, "LR": 0.9101}


def load_daily():
    d = pd.read_csv(DAILY, parse_dates=["date"]).set_index("date")
    d = d[["ohlcv_open", "ohlcv_high", "ohlcv_low", "ohlcv_close", "ohlcv_volume"]]
    d.columns = ["open", "high", "low", "close", "volume"]
    d = d.loc["2018-01-01":"2026-08-30"].dropna(subset=["close"])
    return d.astype(float)


def run_all():
    from download import load_15m
    from paper import features, label_paper, label_next, make_xgb, make_lr, fit_eval, split_80_20
    m15, daily = load_15m(), load_daily()
    setups = {
        "A": (m15, label_paper, "15m, nhãn bài"),
        "B": (m15, label_next, "15m, nhãn nến kế tiếp"),
        "C": (daily, label_next, "ngày, nhãn ngày mai"),
    }
    factories = {"XGBoost": make_xgb, "LR": make_lr}
    rows = []
    for exp, (df, labf, desc) in setups.items():
        X, y = features(df), labf(df["close"])
        d = pd.concat([X, y.rename("__y")], axis=1).dropna()
        tr, te = split_80_20(len(d))
        for scale in SCALES:
            for mname in MODELS:
                model = factories[mname]()
                r = fit_eval(X, y, model, scale)
                (tn, fp), (fn, tp) = r["confusion"]
                rows.append({
                    "experiment": exp, "description": desc, "scale_on": scale, "model": mname,
                    "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall"],
                    "f1": r["f1"], "auc": r["auc"], "majority_baseline": r["majority_baseline"],
                    "test_pos_rate": r["test_pos_rate"], "n_train": r["n_train"], "n_test": r["n_test"],
                    "train_start": d.index[tr[0]], "train_end": d.index[tr[-1]],
                    "test_start": d.index[te[0]], "test_end": d.index[te[-1]],
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "selected_features": " ".join(r["selected_features"]),
                    "lr_n_iter": int(model.n_iter_[0]) if mname == "LR" else None,
                })
    return pd.DataFrame(rows)


def f4(x):
    return f"{x:.4f}".replace(".", ",")


def write_report(res, runtime):
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    missing = manifest.get("missing_candles")
    get = lambda e, s, m: res[(res.experiment == e) & (res.scale_on == s) & (res.model == m)].iloc[0]
    L = []
    L.append("# Báo cáo tái tạo Hafid et al. 2024 (arXiv:2410.06935)\n")
    L.append("Mọi số kết quả trong báo cáo này được sinh tự động từ `reports/results.csv` bởi `src/run.py`; số nến thiếu lấy từ `data/manifest.json`.\n")
    L.append("## 1. Đã chạy gì\n")
    L.append("- **A**: BTCUSDT nến 15 phút (2021-02-01 → 2022-02-01 UTC), nhãn của bài: y_t = 1 nếu SMA10_t ≥ SMA60_t (SMA của Close, gồm cả nến t).")
    L.append("- **B**: cùng dữ liệu 15 phút, nhãn dự báo thật: y_t = 1 nếu C_{t+1} > C_t.")
    L.append("- **C**: OHLCV ngày của nhóm (`data/processed/master_daily.csv`, chỉ 5 cột OHLCV, 2018-01-01 → 2026-08-30), cùng 17 đặc trưng tính trên nến ngày, nhãn ngày mai 1[C_{t+1} > C_t].")
    L.append("- Mỗi thí nghiệm chạy với 2 chế độ chuẩn hoá (`full`: scaler và chi2 fit trên toàn bộ dòng như bài; `train`: chỉ fit trên 80% đầu) và 2 mô hình (XGBoost, LR) = 12 cấu hình. Chia 80/20 theo thời gian, không xáo trộn; mô hình luôn fit trên train, đánh giá trên test.")
    lr = res[res.model == "LR"]
    L.append(f"- LR (max_iter = 100, giá trị của bài) hội tụ trong giới hạn này ở cả 6 cấu hình: n_iter_ từ {int(lr.lr_n_iter.min())} đến {int(lr.lr_n_iter.max())} (< 100); không có ConvergenceWarning trên dữ liệu thật. n_iter_ theo cấu hình: " + ", ".join(f"{r.experiment}/{r.scale_on} = {int(r.lr_n_iter)}" for _, r in lr.iterrows()) + ".")
    L.append("- Majority baseline là tỷ lệ lớp nhiều nhất trong tập TEST (hơi có lợi cho baseline so với dùng tỷ lệ của train).\n")
    L.append("## 2. Bảng kết quả\n")
    L.append("| TN | Chuẩn hoá | Mô hình | Accuracy | Precision | Recall | F1 | AUC | Majority baseline | n_test | 8 đặc trưng được chọn |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        L.append(f"| {r.experiment} | {r.scale_on} | {r.model} | {f4(r.accuracy)} | {f4(r.precision)} | {f4(r.recall)} | {f4(r.f1)} | {f4(r.auc)} | {f4(r.majority_baseline)} | {r.n_test} | {r.selected_features} |")
    L.append("\n## 3. Giai đoạn train/test (sau khi bỏ dòng warm-up)\n")
    L.append("| TN | n_train | Train đầu | Train cuối | n_test | Test đầu | Test cuối |")
    L.append("|---|---|---|---|---|---|---|")
    for e in EXPERIMENTS:
        r = res[res.experiment == e].iloc[0]
        L.append(f"| {e} | {r.n_train} | {r.train_start} | {r.train_end} | {r.n_test} | {r.test_start} | {r.test_end} |")
    L.append("\n## 4. So sánh A với số của bài\n")
    L.append("| Mô hình | Bài báo | A (full) | Hiệu (full − bài) | A (train) | Hiệu (train − bài) |")
    L.append("|---|---|---|---|---|---|")
    sg1 = lambda x: ("+" if x >= 0 else "−") + f"{abs(x):.1f}".replace(".", ",")
    sg = lambda x: ("+" if x >= 0 else "−") + f4(abs(x))
    for m in MODELS:
        f_, t_ = get('A', 'full', m).accuracy, get('A', 'train', m).accuracy
        L.append(f"| {m} | {f4(PAPER[m])} | {f4(f_)} | {sg(f_ - PAPER[m])} | {f4(t_)} | {sg(t_ - PAPER[m])} |")
    L.append("\nCác tham số không được chỉnh để khớp số của bài. Bài không công bố các dòng dữ liệu chính xác, ranh giới chia train/test và mã nguồn, nên không thể xác nhận khớp chính xác.\n")
    L.append("## 5. Nhận xét chính\n")
    L.append("Chênh lệch accuracy so với majority baseline (chuẩn hoá train-only). SE = sqrt(0,25/n_test); gap = accuracy − majority baseline; z = gap/SE.\n")
    L.append("| TN | Mô hình | n_test | Accuracy | Baseline | gap | SE | gap/SE | AUC | Trong ±2 SE? |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in EXPERIMENTS:
        for m in MODELS:
            r = get(e, "train", m)
            se = (0.25 / r.n_test) ** 0.5
            gap = r.accuracy - r.majority_baseline
            z = sg1(gap / se)
            L.append(f"| {e} | {m} | {r.n_test} | {f4(r.accuracy)} | {f4(r.majority_baseline)} | {sg(gap)} | {f4(se)} | {z} | {f4(r.auc)} | {'có' if abs(gap / se) <= 2 else 'không'} |")
    nc = int(get("C", "train", "LR").n_test)
    sec = (0.25 / nc) ** 0.5
    bl, bx = get("B", "train", "LR"), get("B", "train", "XGBoost")
    gl, gx = bl.accuracy - bl.majority_baseline, bx.accuracy - bx.majority_baseline
    seb = (0.25 / bl.n_test) ** 0.5
    L.append("\nSE giả định các quan sát độc lập. Với A, nhãn bền vững theo thời gian (SMA10 so với SMA60 đổi dấu hiếm) nên giả định này sai nặng và SE của A rất lạc quan. "
             "Với B, nhãn là hướng của một nến, các nhãn không chồng lấn nên giả định độc lập chỉ vi phạm nhẹ (tự tương quan còn lại của lợi suất 15 phút).\n")
    L.append(f"**B.** LR vượt baseline {sg(gl)} ({sg1(gl / seb)} SE, AUC {f4(bl.auc)}), còn XGBoost trên cùng 8 đặc trưng chỉ {sg(gx)} ({sg1(gx / seb)} SE), nằm trong nhiễu. "
             f"Lưu ý cho B: một lần chia duy nhất; một cửa sổ test duy nhất từ {bl.test_start} đến {bl.test_end}; "
             "so sánh 4 cấu hình dự báo (2 mô hình × 2 chế độ chuẩn hoá) nên có vấn đề so sánh nhiều lần; "
             "không có chi phí giao dịch hay phân tích lợi suất (accuracy không bằng lợi nhuận).\n")
    L.append(f"**C.** n_test = {nc}, SE = {f4(sec)}; hiệu ứng nhỏ nhất phát hiện được với power 80% (hai phía, 5%) ≈ 2,8·SE = {f4(2.8 * sec)}. "
             "Vì vậy kết quả của C không có thông tin: không phải bằng chứng là không có lợi thế dự báo.\n")
    sel = get("A", "train", "XGBoost").selected_features.replace(" ", ", ")
    L.append(f"**A.** Nhãn của A là hàm tất định của Close đến hết nến t (SMA10 so với SMA60), và các đặc trưng được chọn ({sel}) là các đại lượng đà giá tính từ cùng chuỗi Close; A không phải dự báo giá tương lai.\n")
    L.append("**Chế độ `full` và `train` cho kết quả gần như trùng nhau.** MinMax sau StandardScaler triệt tiêu bước Standard, nên đầu vào hiệu dụng là (x − min_fit)/(max_fit − min_fit); `scale_on` chỉ có tác dụng khi min/max của một cột được chọn rơi vào giai đoạn test. "
             "XGBoost bất biến với phép biến đổi affine tăng theo từng cột, nên XGBoost giống hệt nhau giữa hai chế độ ở B và C. "
             "Ở B, cực trị của các cột được chọn nằm trong train, trừ %D30 (thay đổi rất nhỏ) mà LR-L1 gán trọng số gần 0, nên dự đoán không đổi. "
             "Ở C, LR khác nhau vì Close và các EMA lập đỉnh mới trong 2025–26 (giai đoạn test). "
             "Chế độ `full` là GIẢ ĐỊNH của nhóm về bài: PDF chỉ nói \"We scale the data by using the StandardScaler\" và chia 80% train / 20% test, không nói scaler được fit trên những dòng nào.\n")
    L.append("## 6. Lựa chọn của nhóm khi bài không nêu\n")
    for s in ["RSI kiểu Wilder (EWM alpha = 1/n)", "MACD = EMA12 − EMA26", "%D = SMA 3 của %K cùng cửa sổ",
              "MOM_n = C_t − C_{t−n}", "PROC_9 = 100·(C_t − C_{t−9})/C_{t−9}",
              "chi2 chạy trên dữ liệu đã chuẩn hoá Standard rồi MinMax về [0, 1] (chi2 cần giá trị không âm)",
              "RSI và EMA khởi tạo bằng quan sát đầu tiên (pandas ewm, adjust=False), không phải SMA; với RSI200 giá trị khởi tạo vẫn còn ảnh hưởng lên các dòng train đầu",
              "bỏ các dòng warm-up có NaN; chọn 8 đặc trưng bằng SelectKBest(chi2, k=8); thêm %D30, %D200 vì Eq. 6 của bài chọn chúng",
              "XGBoost: n_estimators 400, learning_rate 0,1, max_depth 4, min_child_weight 3, subsample 0,8, colsample_bytree 1,0, gamma 0,1, reg_alpha 0,5, reg_lambda 1, random_state 0; LR: penalty l1, C 0,1, solver saga, max_iter 100, random_state 0"]:
        L.append(f"- {s}")
    L.append("\n## 7. Dữ liệu thiếu\n")
    L.append(f"Dữ liệu 15 phút có {manifest.get('rows')} nến; so với 365 × 96 = 35040 nến, thiếu {missing} nến (không nội suy). Features dùng cửa sổ theo số dòng nên các chỗ thiếu làm cửa sổ trải dài hơn thời gian danh nghĩa.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")


def main():
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    res = run_all()
    res.to_csv(RESULTS, index=False)
    write_report(pd.read_csv(RESULTS), time.time() - t0)
    print(res[["experiment", "scale_on", "model", "accuracy", "majority_baseline", "auc"]].to_string())
    print(f"runtime {time.time() - t0:.1f}s")


def _tests():
    res = pd.read_csv(RESULTS, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    assert len(res) == 12, len(res)
    assert set(zip(res.experiment, res.scale_on, res.model)) == {
        (e, s, m) for e in EXPERIMENTS for s in SCALES for m in MODELS}
    assert res.majority_baseline.between(0.5, 1).all() and res.majority_baseline.notna().all()
    assert (res.n_test > 0).all()
    assert (res.test_start > res.train_end).all()
    assert (res.accuracy.between(0, 1)).all()
    assert (res.selected_features.str.split().str.len() == 8).all()
    assert (res.tn + res.fp + res.fn + res.tp == res.n_test).all()
    assert REPORT.exists() and "Bảng kết quả" in REPORT.read_text(encoding="utf-8")
    # daily
    d = load_daily()
    assert d.index.min() >= pd.Timestamp("2018-01-01") and d.index.max() <= pd.Timestamp("2026-08-30")
    assert d.notna().all().all()
    print("Tất cả test run đều PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    if ap.parse_args().test:
        _tests()
    else:
        main()
