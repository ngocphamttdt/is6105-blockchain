"""
run_all.py  –  Task 2.8 (Ngọc): chạy thí nghiệm bằng một lệnh. Mỗi lần chạy đều chạy test trước.

  python ngoc_pipeline/src/run_all.py --test                                    # chỉ chạy test của mọi file
  python ngoc_pipeline/src/run_all.py --stage data                              # kiểm freeze, đặc trưng, fold trên dữ liệu thật
  python ngoc_pipeline/src/run_all.py --stage e1 --models lr b0pers b0maj b3    # E1: đủ 82 fold x 2 cấu hình
  python ngoc_pipeline/src/run_all.py --stage e1 --models b3 --every 10         # chạy thử 1/10 số fold

Output (ngoc_pipeline/results/):
  predictions/{model}_{config}.csv   config = ohlcv | ohlcv_onchain; cột date, fold, y_true, p_hat, regime
  tables/e1_delta_draft.csv          Δ = ohlcv_onchain − ohlcv cho mọi mô hình đã có predictions (chưa kiểm định)
Chạy thử (--every > 1) ghi vào thư mục con every{k}/, không đè kết quả chính.
Phương án Flows (A/B) đọc từ flows_set trong configs/features_onchain.yaml.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import data as data_mod
import features
import models
import walkforward
from data import DATE, PREDICTION_COLS, RESULTS, TARGET, load_config, load_master, make_synthetic_master, \
    read_csv_dated, verify_freeze, write_csv
from features import build_dataset
from models import RULES, load_model
from walkforward import make_folds, run_walkforward

CONFIGS = ("ohlcv", "ohlcv_onchain")
METRICS = ("da", "log_loss", "brier", "auc")


def run_e1(data, feature_sets, model_names, folds, wf_cfg, pred_dir, log=print) -> dict:
    """Mỗi mô hình chạy cả 2 cấu hình, cùng fold và cùng grid; ghi {model}_{config}.csv."""
    preds = {}
    for name in model_names:
        for config in CONFIGS:
            if name in RULES:
                p = RULES[name](data, folds)
            else:
                est, grid = load_model(name)
                p = run_walkforward(data, feature_sets[config], est, folds, param_grid=grid, **wf_cfg)
            write_csv(p, Path(pred_dir) / f"{name}_{config}.csv")
            preds[(name, config)] = p
            log(f"  {name}_{config}: {len(p)} ngày test, {p['fold'].nunique()} fold")
    return preds


def load_predictions(pred_dir) -> dict:
    """{(model, config): predictions} của mọi file {model}_{config}.csv có config E1 trong thư mục."""
    files = {tuple(f.stem.split("_", 1)): f for f in sorted(Path(pred_dir).glob("*_*.csv"))}
    return {k: read_csv_dated(f) for k, f in files.items() if k[1] in CONFIGS}


def metrics(p) -> dict:
    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    return {"da": ((ph > 0.5).astype(int) == y).mean(), "log_loss": log_loss(y, ph, labels=[0, 1]),
            "brier": brier_score_loss(y, ph), "auc": roc_auc_score(y, ph)}


def delta_table(preds) -> pd.DataFrame:
    """Mỗi dòng một (model, metric); delta = ohlcv_onchain − ohlcv. Log-loss, Brier: delta âm là On-chain
    tốt hơn. Kiểm định (McNemar, DM, bootstrap) là module của Vi (task 2.3)."""
    rows = []
    for name in sorted({m for m, _ in preds}):
        base, full = preds[(name, "ohlcv")], preds[(name, "ohlcv_onchain")]
        if not np.array_equal(base[DATE].to_numpy("datetime64[ns]"), full[DATE].to_numpy("datetime64[ns]")):
            raise ValueError(f"{name}: hai cấu hình không cùng ngày test, không so cặp được.")
        mb, mf = metrics(base), metrics(full)
        rows += [{"model": name, "metric": k, "ohlcv": mb[k], "ohlcv_onchain": mf[k],
                  "delta": mf[k] - mb[k], "n_days": len(base)} for k in METRICS]
    return pd.DataFrame(rows)


def stage_data():
    """Kiểm freeze (sha256), đặc trưng không còn NaN từ ngày train đầu tiên, và số fold. Không ghi file."""
    man = verify_freeze()
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(verify=False), fe_cfg)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)
    n_nan = int(data.loc[data[DATE] >= folds[0].train_start, sets["ohlcv_onchain"]].isna().sum().sum())
    print(f"Data freeze {man['version']}: sha256 khớp {len(man['files'])} file; flows set {fe_cfg['flows_set']}; "
          f"{len(sets['ohlcv'])} OHLCV + {len(sets['ohlcv_onchain']) - len(sets['ohlcv'])} On-chain; "
          f"{len(folds)} fold, train fold 0 từ {folds[0].train_start.date()}; NaN từ ngày đó: {n_nan}")
    if n_nan:
        raise ValueError("Còn NaN trong giai đoạn train/test: kiểm tra dữ liệu hoặc ffill_limit.")


def stage_e1(model_names, every=1):
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe_cfg)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)[::every]
    print(f"Flows set {fe_cfg['flows_set']}; {len(sets['ohlcv'])} OHLCV, {len(sets['ohlcv_onchain'])} tổng; "
          f"{len(folds)} fold")
    sub = "" if every == 1 else f"every{every}"
    run_e1(data, sets, model_names, folds, wf_cfg, RESULTS / "predictions" / sub)
    table = delta_table(load_predictions(RESULTS / "predictions" / sub))   # gồm cả mô hình chạy từ trước
    print("Đã lưu", write_csv(table, RESULTS / "tables" / sub / "e1_delta_draft.csv"))
    print(table.to_string(index=False, float_format="%.4f"))


# ----------------------------------------------------------------------------
# Test:  python run_all.py --test   (chạy test của mọi file, rồi test của run_all)
# ----------------------------------------------------------------------------
def _tests():
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(make_synthetic_master(seed=42), fe_cfg)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)[::40]
    with tempfile.TemporaryDirectory() as tmp:
        # 1) E1 chạy hết trên dữ liệu giả; CSV đọc lại khớp y nguyên; hai cấu hình cùng ngày test; B0 giống nhau
        preds = run_e1(data, sets, ["lr", "b0pers", "b0maj"], folds, wf_cfg, tmp, log=lambda _: None)
        back = load_predictions(tmp)
        assert set(back) == set(preds)
        for k, p in preds.items():
            assert list(back[k].columns) == PREDICTION_COLS
            np.testing.assert_array_equal(back[k]["p_hat"], p["p_hat"])
        pd.testing.assert_frame_equal(preds[("b0pers", "ohlcv")], preds[("b0pers", "ohlcv_onchain")])
        table = delta_table(back)
        np.testing.assert_allclose(table["delta"], table["ohlcv_onchain"] - table["ohlcv"])
        # 2) Bảng Δ bỏ qua file không phải cấu hình E1 (vd. ablation sau này)
        (Path(tmp) / "b4_ohlcv_plus_flows.csv").write_text("x")
        assert set(load_predictions(tmp)) == set(preds)
    # 3) Hai cấu hình khác ngày test thì không so cặp
    p = preds[("lr", "ohlcv")]
    data_mod._raises(lambda: delta_table({("lr", "ohlcv"): p, ("lr", "ohlcv_onchain"): p.iloc[1:]}), "không cùng")
    print("Tất cả test run_all đều PASS")


def run_tests():
    for mod in (data_mod, features, walkforward, models):
        mod._tests()
    _tests()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["data", "e0", "e1"])
    ap.add_argument("--models", nargs="+", default=["lr"])
    ap.add_argument("--every", type=int, default=1, help="chỉ chạy mỗi k fold (để chạy thử)")
    ap.add_argument("--test", action="store_true", help="chỉ chạy test")
    args = ap.parse_args(argv)
    if not args.test and not args.stage:
        ap.error("cần --stage hoặc --test")

    run_tests()
    if args.stage == "data":
        stage_data()
    elif args.stage == "e1":
        stage_e1(args.models, args.every)
    elif args.stage == "e0":
        print("Stage e0 chưa cài (task 2.9, 2.10).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
