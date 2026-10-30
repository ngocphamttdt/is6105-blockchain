"""
run_all.py  –  Task 2.8 (Ngọc): chạy thí nghiệm bằng một lệnh. Mỗi lần chạy đều chạy test trước.

  python ngoc_pipeline/src/run_all.py --test                                    # chỉ chạy test của mọi file
  python ngoc_pipeline/src/run_all.py --stage data                              # kiểm freeze, đặc trưng, fold trên dữ liệu thật
  python ngoc_pipeline/src/run_all.py --stage e1 --models lr b0pers b0maj b3    # E1: đủ 82 fold x 2 cấu hình
  python ngoc_pipeline/src/run_all.py --stage e1 --models b3 --every 10         # chạy thử 1/10 số fold
  python ngoc_pipeline/src/run_all.py --stage regime                            # ghép lại regime vào mọi predictions

Output (ngoc_pipeline/results/):
  predictions/{model}_{config}.csv   config = ohlcv | ohlcv_onchain; cột date, fold, y_true, p_hat, regime
  predictions/b5_seeds/              B5 LSTM từng seed; b5_{config}.csv là p̂ trung bình các seed
  tables/e1_delta_draft.csv          Δ = ohlcv_onchain − ohlcv cho mọi mô hình đã có predictions (chưa kiểm định)
Chạy thử (--every > 1) ghi vào thư mục con every{k}/, không đè kết quả chính.
Phương án Flows (A/B) đọc từ flows_set trong configs/features_onchain.yaml.
Cột regime lấy từ ngan_lstm_hmm_analysis/regime_map.csv (HMM của Ngân, task 2.13), ghép theo date + fold.
"""
import argparse
import contextlib
import importlib.util
import io
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import data as data_mod
import features
import models
import walkforward
from data import DATE, PREDICTION_COLS, RESULTS, TARGET, load_config, load_master, make_synthetic_close_1h, \
    make_synthetic_master, read_csv_dated, verify_freeze, write_csv
from features import build_dataset
from models import RULES, load_model
from walkforward import make_folds, run_walkforward

CONFIGS = ("ohlcv", "ohlcv_onchain")
METRICS = ("da", "log_loss", "brier", "auc")
NGAN_DIR = Path(__file__).resolve().parents[2] / "ngan_lstm_hmm_analysis"
REGIME_MAP = NGAN_DIR / "regime_map.csv"
LSTM = "b5"


def load_regime_map(path=REGIME_MAP) -> pd.DataFrame | None:
    """Bảng date, fold, regime của HMM; None nếu chưa có file."""
    if not Path(path).exists():
        return None
    r = pd.read_csv(path)
    r[DATE] = pd.to_datetime(r[DATE], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    return r[[DATE, "fold", "regime"]]


def attach_regime(p, regime_map) -> pd.DataFrame:
    """Điền cột regime theo date + fold. Ngày dự báo nào không có regime thì báo lỗi, không tự gán."""
    if regime_map is None:
        return p
    out = p.drop(columns="regime").merge(regime_map, on=[DATE, "fold"], how="left", validate="one_to_one")
    if out["regime"].isna().any():
        raise ValueError(f"{int(out['regime'].isna().sum())} ngày dự báo không có regime trong regime_map.")
    return out[PREDICTION_COLS]


def _lstm_job(data, folds, feature_cols, seed, params) -> pd.DataFrame:
    """Chạy trong tiến trình con: torch không được nạp chung tiến trình với xgboost (trên macOS hai bản
    OpenMP của chúng xung đột, torch bị treo khi tạo tensor nếu xgboost đã chạy trước)."""
    if str(NGAN_DIR) not in sys.path:
        sys.path.append(str(NGAN_DIR))
    import torch
    torch.set_num_threads(4)                                      # cố định để chạy song song nhiều seed
    from model_lstm import run_lstm
    with contextlib.redirect_stdout(io.StringIO()):               # model_lstm in log từng epoch
        return run_lstm(data, folds, feature_cols, seed=seed, **params)


def run_lstm_seeds(data, feature_cols, folds, seed_dir, config, regime_map=None) -> pd.DataFrame:
    """B5: chạy model_lstm.run_lstm của Ngân với từng seed (mỗi seed một tiến trình con, chạy song song),
    lưu riêng từng seed, trả p̂ trung bình."""
    cfg = load_config(f"models/{LSTM}.yaml")
    with ProcessPoolExecutor(len(cfg["seeds"]), mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(_lstm_job, data, folds, list(feature_cols), s, cfg["params"]) for s in cfg["seeds"]]
        runs = [j.result() for j in jobs]
    for seed, p in zip(cfg["seeds"], runs):
        write_csv(attach_regime(p, regime_map), Path(seed_dir) / f"{LSTM}s{seed}_{config}.csv")
    if any(not r[DATE].equals(runs[0][DATE]) for r in runs):
        raise ValueError("Các seed của B5 không cùng ngày test.")
    mean = runs[0].copy()
    mean["p_hat"] = np.mean([r["p_hat"].to_numpy() for r in runs], axis=0)
    return mean


def run_e1(data, feature_sets, model_names, folds, wf_cfg, pred_dir, log=print,
           regime_map=None) -> dict:
    """Mỗi mô hình chạy cả 2 cấu hình, cùng fold và cùng grid; ghi {model}_{config}.csv."""
    preds = {}
    for name in model_names:
        for config in CONFIGS:
            if name in RULES:
                p = RULES[name](data, folds)
            elif name == LSTM:
                p = run_lstm_seeds(data, feature_sets[config], folds, Path(pred_dir) / f"{LSTM}_seeds", config,
                                   regime_map)
            else:
                est, grid = load_model(name)
                p = run_walkforward(data, feature_sets[config], est, folds, param_grid=grid, **wf_cfg)
            p = attach_regime(p, regime_map)
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
    run_e1(data, sets, model_names, folds, wf_cfg, RESULTS / "predictions" / sub, regime_map=load_regime_map())
    table = delta_table(load_predictions(RESULTS / "predictions" / sub))   # gồm cả mô hình chạy từ trước
    print("Đã lưu", write_csv(table, RESULTS / "tables" / sub / "e1_delta_draft.csv"))
    print(table.to_string(index=False, float_format="%.4f"))


def stage_regime(pred_dir=RESULTS / "predictions"):
    """Ghép lại regime vào mọi file predictions (cả b5_seeds/), dùng khi Ngân cập nhật regime_map.csv."""
    regime_map = load_regime_map()
    if regime_map is None:
        raise ValueError(f"Chưa có {REGIME_MAP}.")
    files = sorted(Path(pred_dir).glob("*_*.csv")) + sorted(Path(pred_dir).glob(f"{LSTM}_seeds/*_*.csv"))
    for f in files:
        write_csv(attach_regime(read_csv_dated(f), regime_map), f)
    print(f"Đã ghép regime vào {len(files)} file trong {pred_dir}")


# ----------------------------------------------------------------------------
# Test:  python run_all.py --test   (chạy test của mọi file, rồi test của run_all)
# ----------------------------------------------------------------------------
def _tests():
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    m = make_synthetic_master(seed=42)
    data, sets = build_dataset(m, fe_cfg, make_synthetic_close_1h(m))
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

    # 4) Ghép regime theo date + fold, giữ nguyên thứ tự và p̂; thiếu regime thì báo lỗi
    rmap = p[[DATE, "fold"]].assign(regime=np.where(np.arange(len(p)) % 2, "Bull", "Bear")).iloc[::-1]
    out = attach_regime(p, rmap)
    assert list(out.columns) == PREDICTION_COLS and out["regime"].notna().all()
    np.testing.assert_array_equal(out["p_hat"], p["p_hat"])
    assert out["regime"].tolist() == rmap.iloc[::-1]["regime"].tolist()
    data_mod._raises(lambda: attach_regime(p, rmap.iloc[1:]), "không có regime")

    # 5) B5 (nếu cài được torch): đủ seed, p̂ trung bình đúng, cùng định dạng.
    #    Không import torch ở tiến trình chính (xem _lstm_job), chỉ kiểm có cài hay không.
    if importlib.util.find_spec("torch") is None:
        print("  Bỏ qua test b5: chưa cài torch")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mean = run_lstm_seeds(data, sets["ohlcv"], folds[:1], tmp, "ohlcv")
            seeds = [read_csv_dated(f) for f in sorted(Path(tmp).glob("*.csv"))]
            assert len(seeds) == len(load_config(f"models/{LSTM}.yaml")["seeds"])
            np.testing.assert_allclose(mean["p_hat"], np.mean([s["p_hat"] for s in seeds], axis=0), rtol=1e-6)
            assert list(mean.columns) == PREDICTION_COLS and mean["p_hat"].between(0, 1).all()
    print("Tất cả test run_all đều PASS")


def run_tests():
    for mod in (data_mod, features, walkforward, models):
        mod._tests()
    _tests()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["data", "e0", "e1", "regime"])
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
    elif args.stage == "regime":
        stage_regime()
    elif args.stage == "e0":
        print("Stage e0 chưa cài (task 2.9, 2.10).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
