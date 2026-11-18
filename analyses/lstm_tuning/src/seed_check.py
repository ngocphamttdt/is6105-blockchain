"""
seed_check.py  –  Kết quả của B5 LSTM có phải do may mắn của random seed không?

Giữ nguyên dữ liệu, cách chia, cấu hình trước tune (DEFAULT) và cấu hình sau tune (chọn ở lstm_tuning.py,
đọc lại từ grid_validation.csv), chỉ đổi seed. Mỗi seed chạy riêng (không lấy trung bình), trên cả
test cố định 2024–08/2026 và walk-forward 82 fold. Ghi phân bố DA, log-loss theo seed và chênh lệch
sau − trước tune ghép cặp theo seed.

Cần chạy lstm_tuning.py (cùng --features) trước để có grid_validation.csv.

    python analyses/lstm_tuning/src/seed_check.py --features lag1            # 10 seed (0–9), mặc định
    python analyses/lstm_tuning/src/seed_check.py --features all --n-seeds 5 # đổi số seed
    python analyses/lstm_tuning/src/seed_check.py --features lag1 --quick    # 2 seed, 3 fold, không ghi báo cáo
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

import lstm_tuning as lt
bc = lt.bc

LR_WF, LR_TEST = 52.22, 52.42          # LR 1 đặc trưng (baseline_check), không phụ thuộc seed
REV_WF, REV_TEST = 53.25, 50.98        # luật đảo chiều

# Vì sao 10 seed: số seed cần để phát hiện chênh lệch d khi so ghép cặp ≈ (2σ/d)², với σ là độ lệch chuẩn của
# chênh lệch giữa các seed. Đo được σ ≈ 0,6–0,8 điểm DA → phát hiện chênh lệch 0,5 điểm cần khoảng 10 seed
# (1 điểm chỉ cần khoảng 3). Với 10 seed, sai số của DA trung bình ≈ 0,6/√10 ≈ 0,2 điểm, đủ để so với LR
# (cách 1,2 điểm). 20–30 seed chỉ thu hẹp thêm khoảng 0,1 điểm mà tốn gấp 2–3 lần thời gian.
N_SEEDS = 10
SEED_RATIONALE = ("Vì sao 10 seed: số seed cần để phát hiện chênh lệch d (so ghép cặp) ≈ (2σ/d)². Với σ ≈ 0,6–0,8 "
                  "điểm DA, phát hiện chênh lệch 0,5 điểm cần khoảng 10 seed; sai số của DA trung bình 10 seed ≈ 0,2 "
                  "điểm, đủ nhỏ so với khoảng cách tới LR (1,2 điểm). Seed được cố định trước (0, 1, …), không chọn "
                  "seed theo kết quả.")


def best_from_grid(path):
    """Chọn lại cấu hình như lstm_tuning.step_a: val log-loss làm tròn 4 chữ số, hoà thì đơn giản hơn."""
    t = pd.read_csv(path)
    t["_rank"] = t["val_log_loss"].round(4)
    r = t.sort_values(["_rank", "hidden", "lookback", "dropout", "lr"],
                      ascending=[True, True, True, False, False]).iloc[0]
    return {"lookback": int(r["lookback"]), "hidden": int(r["hidden"]),
            "dropout": float(r["dropout"]), "lr": float(r["lr"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(lt.FEATURE_SETS), default="all")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS, help=f"số seed (0 … n−1), mặc định {N_SEEDS}")
    ap.add_argument("--quick", action="store_true", help="2 seed, 3 fold, không ghi báo cáo")
    args = ap.parse_args()
    lt.COLS, lt.FEAT_DESC, sub = lt.FEATURE_SETS[args.features]
    out = lt.OUT / sub if sub else lt.OUT

    df = bc.load_data()
    parts = bc.split(df)
    feats = lt.load_full()
    feats["_y"] = df["y"].reindex(feats.index)
    best = best_from_grid(out / "grid_validation.csv")
    cfgs = {"trước tune": lt.DEFAULT, "sau tune": best}
    seeds = list(range(2 if args.quick else args.n_seeds))
    max_folds = 3 if args.quick else None
    print(f"{lt.FEAT_DESC}: {len(seeds)} seed; trước tune = {lt.cfg_name(lt.DEFAULT)}; sau tune = {lt.cfg_name(best)}")

    te = parts["test"]
    trva = parts["train"].index.append(parts["val"].index)
    rows = []
    t0 = time.time()
    for s in seeds:
        for label, cfg in cfgs.items():
            p_test, _, _ = lt.ensemble(feats, trva, {"test": te.index}, cfg, [s])
            m_test = bc.metrics(te["y"], p_test["test"])
            wf_p, _, test_days = lt.step_b(feats, df, {label: cfg}, [s], max_folds=max_folds)
            p = wf_p[label]
            ok = p.notna()
            m_wf = bc.metrics(df.loc[test_days[ok], "y"], p[ok])
            rows.append({"seed": s, "cấu hình": label,
                         "test_DA": m_test["DA"] * 100, "test_log_loss": m_test["log_loss"],
                         "wf_DA": m_wf["DA"] * 100, "wf_log_loss": m_wf["log_loss"],
                         "wf_pct_pred_up": m_wf["pct_pred_up"] * 100})
        r = [x for x in rows if x["seed"] == s]
        print(f"  seed {s}: " + "; ".join(f"{x['cấu hình']} WF DA {x['wf_DA']:.2f} test DA {x['test_DA']:.2f}"
                                          for x in r) + f"  ({(time.time() - t0) / 60:.1f} phút)")
    t = pd.DataFrame(rows)
    if args.quick:
        print(t.to_string())
        return

    # Phân bố theo seed
    summ = t.groupby("cấu hình")[["wf_DA", "wf_log_loss", "test_DA", "test_log_loss"]].agg(
        ["mean", "std", "min", "max"]).round(4).T
    summ = summ[["trước tune", "sau tune"]]
    # Chênh lệch ghép cặp theo seed: sau − trước
    w = t.pivot(index="seed", columns="cấu hình")
    d_wf = w["wf_DA"]["sau tune"] - w["wf_DA"]["trước tune"]
    d_wf_ll = w["wf_log_loss"]["sau tune"] - w["wf_log_loss"]["trước tune"]
    d_te = w["test_DA"]["sau tune"] - w["test_DA"]["trước tune"]
    paired = pd.DataFrame({
        "metric": ["ΔDA walk-forward (điểm %)", "Δlog-loss walk-forward", "ΔDA test (điểm %)"],
        "TB": [d_wf.mean(), d_wf_ll.mean(), d_te.mean()],
        "độ lệch chuẩn": [d_wf.std(), d_wf_ll.std(), d_te.std()],
        "số seed sau tune tốt hơn": [int((d_wf > 0).sum()), int((d_wf_ll < 0).sum()), int((d_te > 0).sum())],
        "p (t-test ghép cặp)": [stats.ttest_1samp(d_wf, 0).pvalue, stats.ttest_1samp(d_wf_ll, 0).pvalue,
                                stats.ttest_1samp(d_te, 0).pvalue],
    }).round(4)
    beats = pd.DataFrame({
        "cấu hình": ["trước tune", "sau tune"],
        f"seed có WF DA > LR 1 đặc trưng ({LR_WF}%)": [int((w["wf_DA"][c] > LR_WF).sum()) for c in cfgs],
        f"seed có WF DA > luật đảo chiều ({REV_WF}%)": [int((w["wf_DA"][c] > REV_WF).sum()) for c in cfgs],
        "seed có WF log-loss < 0,6931": [int((w["wf_log_loss"][c] < 0.6931).sum()) for c in cfgs],
    })
    show = t.copy()
    for c in ["test_DA", "wf_DA", "wf_pct_pred_up"]:
        show[c] = show[c].round(2)
    for c in ["test_log_loss", "wf_log_loss"]:
        show[c] = show[c].round(4)

    t.to_csv(out / "seed_check.csv", index=False)
    md = [
        f"# Kiểm tra theo random seed: B5 LSTM, {lt.FEAT_DESC}", "",
        "Tạo bởi `analyses/lstm_tuning/src/seed_check.py`. Giữ nguyên dữ liệu, cách chia và hai cấu hình, chỉ đổi seed "
        f"({len(seeds)} seed: {seeds[0]}–{seeds[-1]}). Mỗi seed là **một mô hình riêng**, không lấy trung bình "
        "(báo cáo chính `lstm_tuning.md` dùng p̂ trung bình 3 seed 42–44).", "",
        f"- Trước tune: {lt.cfg_name(lt.DEFAULT)}",
        f"- Sau tune (chọn trên validation ở `lstm_tuning.py`): {lt.cfg_name(best)}",
        f"- Mốc không phụ thuộc seed: LR 1 đặc trưng (WF {LR_WF}%, test {LR_TEST}%), luật đảo chiều "
        f"(WF {REV_WF}%, test {REV_TEST}%), đoán 50/50 (log-loss 0,6931). LR dùng solver tất định nên chạy lại "
        "luôn ra cùng kết quả.",
        "- " + SEED_RATIONALE, "",
        "## Phân bố theo seed", "",
        summ.to_markdown(), "",
        "## Sau tune có thật sự tốt hơn trước tune? (ghép cặp theo seed)", "",
        paired.to_markdown(index=False), "",
        "## Có seed nào vượt các mốc không?", "",
        beats.to_markdown(index=False), "",
        "## Từng seed", "",
        show.to_markdown(index=False), "",
    ]
    (out / "seed_check.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Đã lưu {out / 'seed_check.md'}")


if __name__ == "__main__":
    main()
