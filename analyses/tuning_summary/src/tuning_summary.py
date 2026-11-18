"""
tuning_summary.py  –  Bảng tổng hợp trước / sau tuning của mọi baseline B0–B5, trên cùng 2.434 ngày test.

Chỉ ĐỌC file dự báo đã có trong repo, không chạy lại mô hình. Mọi metric được tính lại theo cùng một cách
(run_all.metrics, stats.paired_row) từ p̂ của từng ngày, nên các dòng so được trực tiếp với nhau.

Ba mức tune (trùng trường `tuning` trong pipeline/configs/models/*.yaml):
  none           không tune: luật B0, tham số mặc định của sklearn, hoặc tham số cố định (B5)
  original_grid  tune bằng grid gốc (Phụ lục C): dự báo chính thức của E1
  retuned        đã tune lại: b1t, b2t, b3t, grid mở rộng của gate.py, lstm_tuning của Ngọc

Δ của mỗi dòng so với dòng `so_với` (bản chưa tune cùng code), CI 95% block bootstrap 15 ngày, McNemar cho DA,
Diebold–Mariano cho log-loss. Thêm ΔDA so với luật đảo chiều b0rev (mốc khó vượt nhất).

Chạy từ thư mục gốc repo:
    python analyses/tuning_summary/src/tuning_summary.py          # ghi results/ và reports/
    python analyses/tuning_summary/src/tuning_summary.py --test   # chỉ chạy test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pipeline/src"))
import stats                                   # noqa: E402
from run_all import metrics                    # noqa: E402

BASE = Path(__file__).resolve().parents[1]
P = ROOT / "pipeline/results"
AN = ROOT / "analyses"
CONFIGS = ("ohlcv", "ohlcv_onchain")
LSTM_WF = AN / "lstm_tuning/reports/predictions_walkforward.csv"
LEVELS = {"none": "chưa tune", "original_grid": "grid gốc", "retuned": "tune lại"}

GATE_NOTE = "grid mở rộng của gate.py; 10 OHLCV + winsorize 1%/99% (như E2); chỉ OHLCV"
B4_NOTE = ("dự báo E1. Bản chạy lại cùng grid trong pipeline/results/tuning/tuned/ lệch 312–359 ngày quyết định: "
           "XGBoost không tái lập được giữa hai lần chạy (khác máy / phiên bản)")
LSTM_NOTE = "code analyses/lstm_tuning (Ngọc), khác regime/model_lstm.py của B5 trong E1; chỉ OHLCV"


def rows_spec():
    """(id, baseline, config, mức, mô hình, nguồn, cột p̂ hoặc None, so_với, ghi chú)."""
    r = []
    for c in CONFIGS:
        for m, note in (("b0maj", "p̂ = tỷ lệ ngày tăng của cửa sổ train"),
                        ("b0pers", "ngày mai cùng hướng hôm nay"),
                        ("b0rev", "ngày mai ngược hướng hôm nay; chiều chọn trên train")):
            r.append((f"{m}_{c}", "B0", c, "none", m, P / f"predictions/{m}_{c}.csv", None, None, note))
        for b in ("b1", "b2", "b3", "b4"):
            B = b.upper()
            r.append((f"{b}def_{c}", B, c, "none", f"{b} tham số mặc định", P / f"tuning/default/{b}_{c}.csv", None,
                      None, "tham số trong grid để ở mặc định của sklearn (experiments.py --stage tuning)"))
            r.append((f"{b}_{c}", B, c, "original_grid", b, P / f"predictions/{b}_{c}.csv", None, f"{b}def_{c}",
                      B4_NOTE if b == "b4" else "dự báo E1"))
        r.append((f"b1t_{c}", "B1", c, "retuned", "b1t", P / f"b1t/e1/{c}/predictions/b1t_{c}.csv", None, f"b1def_{c}",
                  "tune cả tập đặc trưng, C, L1/L2, class_weight (b1t_tuning.py --stage e1)"))
        r.append((f"b2t_{c}", "B2", c, "retuned", "b2t", P / f"predictions/b2t_{c}.csv", None, f"b2def_{c}",
                  "grid thiết kế từ dữ liệu trước 2020 (analyses/b2_tuning)"))
        r.append((f"b3t_{c}", "B3", c, "retuned", "b3t", P / f"predictions/b3t_{c}.csv", None, f"b3def_{c}",
                  "grid C × gamma 5 × 6 của Trâm, chạy qua pipeline (run_all.py --ohlcv10); "
                  "trùng b3g30 trong analyses/baselines (lệch p̂ < 1e-12)"))
        r.append((f"b5_{c}", "B5", c, "none", "b5", P / f"predictions/b5_{c}.csv", None, None,
                  "dự báo E1, tham số cố định, trung bình 3 seed"))
    for b in ("b1", "b2", "b3", "b4"):
        r.append((f"{b}gate_ohlcv", b.upper(), "ohlcv", "retuned", f"{b} grid mở rộng (gate)", P / f"gate/baseline/{b}.csv",
                  None, f"{b}def_ohlcv", GATE_NOTE))
    r.append(("b5lt0_ohlcv", "B5", "ohlcv", "none", "b5 lstm_tuning trước tune", LSTM_WF, "B5 LSTM trước tune", None,
              LSTM_NOTE))
    r.append(("b5lt1_ohlcv", "B5", "ohlcv", "retuned", "b5 lstm_tuning sau tune", LSTM_WF, "B5 LSTM sau tune",
              "b5lt0_ohlcv", LSTM_NOTE + "; grid 16 cấu hình chọn trên validation 2022–2023"))
    return r


def load(path, col, ref):
    """p̂ của một nguồn, kiểm cùng ngày và cùng y_true với file tham chiếu."""
    d = pd.read_csv(path, parse_dates=["date"])
    y = d["y_true"] if "y_true" in d else d["y"]
    ph = d[col] if col else d["p_hat"]
    if len(d) != len(ref) or not (d["date"].to_numpy() == ref["date"].to_numpy()).all():
        raise ValueError(f"{path}: ngày test khác file tham chiếu")
    if not np.array_equal(y.astype(int).to_numpy(), ref["y_true"].to_numpy()):
        raise ValueError(f"{path}: y_true khác file tham chiếu")
    return pd.DataFrame({"date": d["date"], "y_true": ref["y_true"].to_numpy(), "p_hat": ph.to_numpy(float)})


def build():
    ref = pd.read_csv(P / "predictions/b0rev_ohlcv.csv", parse_dates=["date"])
    spec = rows_spec()
    preds = {s[0]: load(s[5], s[6], ref) for s in spec}
    y, rev = ref["y_true"].to_numpy(), ref["p_hat"].to_numpy()
    out = []
    for rid, base, cfg, level, model, path, col, vs, note in spec:
        p = preds[rid]
        is_rule = model in ("b0pers", "b0rev")
        mtr = metrics(p, "b0rev" if is_rule else ("b0maj" if model == "b0maj" else None))
        row = {"baseline": base, "config": cfg, "level": level, "mức": LEVELS[level], "model": model,
               "n_days": len(p), "da": mtr["da"], "log_loss": mtr["log_loss"], "brier": mtr["brier"], "auc": mtr["auc"],
               "pct_pred_up": (p.p_hat > 0.5).mean(), "so_với": spec_label(spec, vs) if vs else ""}
        if vs:
            d = stats.paired_row(y, preds[vs].p_hat.to_numpy(), p.p_hat.to_numpy(), with_logloss=True)
            row |= {k: d[k] for k in ("d_da", "d_da_lo", "d_da_hi", "p_mcnemar",
                                      "d_logloss", "d_logloss_lo", "d_logloss_hi", "p_dm")}
        if model != "b0rev":
            d = stats.paired_row(y, rev, p.p_hat.to_numpy(), with_logloss=False)
            row |= {"d_da_vs_b0rev": d["d_da"], "d_da_vs_b0rev_lo": d["d_da_lo"], "d_da_vs_b0rev_hi": d["d_da_hi"],
                    "p_mcnemar_vs_b0rev": d["p_mcnemar"]}
        row |= {"nguồn": path.relative_to(ROOT).as_posix() + (f" [cột {col}]" if col else ""), "ghi_chú": note}
        out.append(row)
    order = {"none": 0, "original_grid": 1, "retuned": 2}
    t = pd.DataFrame(out)
    t["_o"] = t.level.map(order)
    return t.sort_values(["baseline", "config", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)


def spec_label(spec, rid):
    return next(s[4] for s in spec if s[0] == rid)


def pct(x, d=2):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}".replace(".", ",")


def num(x, d=4):
    return "–" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def report(t):
    def delta(r, k, scale, d):
        if pd.isna(r.get(k)):
            return ""
        f = (lambda v: f"{v * scale:+.{d}f}".replace(".", ","))
        return f"{f(r[k])} [{f(r[k + '_lo'])}; {f(r[k + '_hi'])}]"

    md = ["# Trước và sau tuning của mọi baseline (B0–B5)", "",
          "Tạo bởi `analyses/tuning_summary/src/tuning_summary.py`. Chỉ đọc file dự báo đã có, không chạy lại mô hình. "
          "Mọi dòng dùng cùng 2.434 ngày test walk-forward (01/01/2020 → 30/08/2026, 82 fold) và cùng cách tính metric.", "",
          "- **Mức tune:** chưa tune (luật, tham số mặc định hoặc cố định) · grid gốc (Phụ lục C, dự báo E1) · tune lại.",
          "- **Δ so với chưa tune:** dòng hiện tại trừ dòng ở cột *so với* (bản chưa tune cùng code). ΔDA tính bằng điểm %, "
          "CI 95% block bootstrap 15 ngày. Δlog-loss < 0 là tốt hơn.",
          "- **ΔDA vs đảo chiều:** so với luật `b0rev` (53,25%), mốc mà chưa mô hình có học nào vượt được.",
          "- Bảng đầy đủ (p-value McNemar, DM, Brier, nguồn từng dòng): `results/tuning_summary.csv`.", ""]
    for cfg, lab in (("ohlcv", "OHLCV-only"), ("ohlcv_onchain", "OHLCV + On-chain")):
        md += [f"## {lab}", "",
               "| Baseline | Mức | Mô hình | DA (%) | Log-loss | AUC | % đoán tăng | ΔDA so với chưa tune | "
               "Δlog-loss so với chưa tune | ΔDA vs đảo chiều |",
               "|---|---|---|---:|---:|---:|---:|---|---|---|"]
        for _, r in t[t.config == cfg].iterrows():
            md.append(f"| {r.baseline} | {r['mức']} | `{r.model}` | {pct(r.da)} | {num(r.log_loss)} | {num(r.auc, 3)} | "
                      f"{pct(r.pct_pred_up, 1)} | {delta(r, 'd_da', 100, 2)} | {delta(r, 'd_logloss', 1, 4)} | "
                      f"{delta(r, 'd_da_vs_b0rev', 100, 2)} |")
        md.append("")
    ml = t[~t.baseline.eq("B0")]
    best = ml.loc[ml.groupby(["baseline", "config"]).log_loss.idxmin()]
    md += ["## Nhận xét (tính tự động từ bảng)", ""]
    tuned = ml[ml.level != "none"].dropna(subset=["d_logloss"])
    better = tuned[tuned.d_logloss_hi < 0]
    worse = tuned[tuned.d_logloss_lo > 0]
    md += [f"- Có {len(tuned)} lần so một bản đã tune với bản chưa tune cùng code. Log-loss tốt lên có ý nghĩa "
           f"(CI nằm hẳn dưới 0) ở {len(better)} lần"
           + (": " + ", ".join(f"`{r.model}` {r.config}" for _, r in better.iterrows()) if len(better) else "")
           + f"; xấu đi có ý nghĩa ở {len(worse)} lần"
           + (": " + ", ".join(f"`{r.model}` {r.config}" for _, r in worse.iterrows()) if len(worse) else "") + ".",
           f"- DA cao nhất trong các mô hình có học: {pct(ml.da.max())}% (`{ml.loc[ml.da.idxmax(), 'model']}`, "
           f"{ml.loc[ml.da.idxmax(), 'config']}); luật đảo chiều đúng {pct(t[t.model.eq('b0rev')].da.iloc[0])}%. "
           f"Số mô hình có học có DA cao hơn đảo chiều với CI không chứa 0: "
           f"{int((ml.d_da_vs_b0rev_lo > 0).sum())}.",
           "- Bản có log-loss thấp nhất của từng baseline: "
           + "; ".join(f"{r.baseline} {r.config}: `{r.model}` ({r['mức']}, {num(r.log_loss)})" for _, r in best.iterrows())
           + ".", "",
           "## Ghi chú về nguồn", ""]
    for _, r in t.drop_duplicates("model").iterrows():
        md.append(f"- `{r.model}`: {r.ghi_chú}. Nguồn (ví dụ cấu hình {r.config}): `{r.nguồn}`")
    return "\n".join(md) + "\n"


def _tests():
    spec = rows_spec()
    ids = [s[0] for s in spec]
    assert len(ids) == len(set(ids)), "id trùng"
    for s in spec:
        assert s[3] in LEVELS
        if s[7]:
            assert s[7] in ids, f"{s[0]}: so_với {s[7]} không có"
            assert next(x for x in spec if x[0] == s[7])[3] == "none", f"{s[0]}: so_với phải là bản chưa tune"
            assert next(x for x in spec if x[0] == s[7])[2] == s[2], f"{s[0]}: so_với phải cùng config"
    # metric và Δ trên ví dụ nhỏ: p̂ đúng hướng mọi ngày thì DA = 1 và ΔDA so với p̂ = 0,5 cố định là dương
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(y == 1, 0.6, 0.4)
    d = stats.paired_row(y, np.full(400, 0.49), good, with_logloss=True)
    assert d["d_da"] > 0 and d["d_logloss"] < 0
    print("Tất cả test tuning_summary đều PASS")


if __name__ == "__main__":
    _tests()
    if "--test" not in sys.argv:
        t = build()
        (BASE / "results").mkdir(parents=True, exist_ok=True)
        (BASE / "reports").mkdir(parents=True, exist_ok=True)
        t.to_csv(BASE / "results/tuning_summary.csv", index=False)
        (BASE / "reports/tuning_summary.md").write_text(report(t), encoding="utf-8")
        print(f"{len(t)} dòng -> {BASE / 'results/tuning_summary.csv'}, {BASE / 'reports/tuning_summary.md'}")
