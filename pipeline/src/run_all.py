"""
run_all.py  –  Task 2.8 (Ngọc): chạy thí nghiệm bằng một lệnh. Mỗi lần chạy đều chạy test trước.

  python pipeline/src/run_all.py --test                                    # chỉ chạy test của mọi file
  python pipeline/src/run_all.py --stage data                              # kiểm freeze, đặc trưng, fold trên dữ liệu thật
  python pipeline/src/run_all.py --stage e1 --models b1 b0pers b0maj b3    # E1: đủ 82 fold x 2 cấu hình
  python pipeline/src/run_all.py --stage e1 --models b3 --every 10         # chạy thử 1/10 số fold
  python pipeline/src/run_all.py --stage e1 --models b3t --ohlcv10         # dùng 10 đặc trưng OHLCV như dự báo E1 hiện có
  python pipeline/src/run_all.py --stage regime                            # ghép lại regime vào mọi predictions
  python pipeline/src/run_all.py --stage tables                            # tính lại các bảng từ predictions
  python pipeline/src/run_all.py --stage e1 --models b1 b2 --out pipeline/results/my_run          # ghi vào thư mục chỉ định
  python pipeline/src/run_all.py --stage e1 --models b1 b2 b3 b4 b5 --no-flows --out pipeline/results/no_flows
                                 # bỏ nhóm Flows khỏi cấu hình ohlcv_onchain; --no-flows bắt buộc có --out
  python pipeline/src/run_all.py --stage tables --out pipeline/results/no_flows                    # bảng cho thư mục đó
  python pipeline/src/run_all.py --stage e1 --models b1 b2 --no-flows --no-activity --out pipeline/results/no_flows_activity
                                 # bỏ nhiều nhóm cùng lúc: --no-flows, --no-activity, --no-mining, --no-valuation

Output (pipeline/results/):
  predictions/{model}_{config}.csv   config = ohlcv | ohlcv_onchain; cột date, fold, y_true, p_hat, regime
  predictions/b5_seeds/              B5 LSTM từng seed; b5_{config}.csv là p̂ trung bình các seed
  tables/e1_delta_draft.csv          Δ = ohlcv_onchain − ohlcv cho mọi mô hình đã có predictions (chưa kiểm định)
  tables/e1_by_year.csv              DA, log-loss của từng mô hình × cấu hình × năm test
  tables/e1_vs_reversal.csv          mỗi mô hình so với luật đảo chiều b0rev: ΔDA gộp + McNemar, ΔDA từng năm
Chạy thử (--every > 1) ghi vào thư mục con every{k}/, không đè kết quả chính.
--out DIR: cùng cấu trúc predictions/, tables/ nhưng nằm trong DIR, kèm DIR/feature_sets.json (bộ đặc trưng đã dùng;
  chạy tiếp vào DIR với bộ khác, vd. bật/tắt --no-flows, sẽ báo lỗi để không trộn kết quả).
--no-flows / --no-activity / --no-mining / --no-valuation: mỗi cờ bỏ một nhóm On-chain khỏi cấu hình ohlcv_onchain,
  dùng riêng hoặc kết hợp (không được bỏ cả 4 nhóm). Tên cấu hình vẫn là ohlcv | ohlcv_onchain (rerun_tables.py,
  rerun_figures.py đọc được). Flows A: 6 đặc trưng (cm_SplyExNtv, cm_NetflowExNtv); Activity, Mining, Valuation:
  mỗi nhóm 9 đặc trưng (3 cột x logdiff, ma7_30, z90). Bỏ nhóm nào cũng bắt buộc có --out.
Phương án Flows (A/B) đọc từ flows_set trong configs/features_onchain.yaml.
Cột regime lấy từ regime/regime_map.csv (HMM của Ngân, task 2.13), ghép theo date + fold.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import data as data_mod
import features
import models
import walkforward
from data import DATE, PREDICTION_COLS, RESULTS, TARGET, load_config, load_master, make_synthetic_close_1h, \
    make_synthetic_master, read_csv_dated, verify_freeze, write_csv
from features import build_dataset
from features_ohlcv import FEATURES_10                   # noqa: E402  (features.py thêm đường dẫn)
from models import RULES, load_model, run_rule
from walkforward import make_folds, run_walkforward

CONFIGS = ("ohlcv", "ohlcv_onchain")
METRICS = ("da", "log_loss", "brier", "auc")
NGAN_DIR = Path(__file__).resolve().parents[2] / "regime"
REGIME_MAP = NGAN_DIR / "regime_map.csv"
LSTM = "b5"
REVERSAL = "b0rev"


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
                p = run_rule(name, data, folds)
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


def metrics(p, name=None) -> dict:
    """DA, log-loss, Brier, AUC. Ô không có nghĩa thì để NaN thay vì in một con số:
    luật (RULES) cho p̂ chỉ 0/1, log-loss và Brier khi đó chỉ đếm ngày sai (log-loss ~ 16,8); b0maj cho p̂ không
    đổi trong mỗi fold, AUC khi đó chỉ xếp hạng tỷ lệ tăng train giữa các fold, không phải giữa các ngày."""
    y, ph = p["y_true"].to_numpy(), p["p_hat"].to_numpy()
    out = {"da": ((ph > 0.5).astype(int) == y).mean(), "log_loss": log_loss(y, ph, labels=[0, 1]),
           "brier": brier_score_loss(y, ph), "auc": roc_auc_score(y, ph)}
    if name in RULES:
        out["log_loss"] = out["brier"] = np.nan
    if name == "b0maj":
        out["auc"] = np.nan
    return out


def delta_table(preds) -> pd.DataFrame:
    """Mỗi dòng một (model, metric); delta = ohlcv_onchain − ohlcv. Log-loss, Brier: delta âm là On-chain
    tốt hơn. Kiểm định (McNemar, DM, bootstrap) là module của Vi (task 2.3)."""
    rows = []
    for name in sorted({m for m, _ in preds}):
        base, full = preds[(name, "ohlcv")], preds[(name, "ohlcv_onchain")]
        if not np.array_equal(base[DATE].to_numpy("datetime64[ns]"), full[DATE].to_numpy("datetime64[ns]")):
            raise ValueError(f"{name}: hai cấu hình không cùng ngày test, không so cặp được.")
        mb, mf = metrics(base, name), metrics(full, name)
        rows += [{"model": name, "metric": k, "ohlcv": mb[k], "ohlcv_onchain": mf[k],
                  "delta": mf[k] - mb[k], "n_days": len(base)} for k in METRICS]
    return pd.DataFrame(rows)


def _hits(p) -> np.ndarray:
    return ((p["p_hat"].to_numpy() > 0.5).astype(int) == p["y_true"].to_numpy()).astype(float)


def by_year_table(preds) -> pd.DataFrame:
    """DA và log-loss theo năm test. Luật (p̂ chỉ 0/1) không có log-loss."""
    rows = []
    for (name, config), p in sorted(preds.items()):
        years = p[DATE].dt.year
        for yr in sorted(years.unique()):
            q = p[years == yr]
            ll = np.nan if name in RULES else log_loss(q["y_true"], q["p_hat"], labels=[0, 1])
            rows.append({"model": name, "config": config, "year": int(yr), "n_days": len(q),
                         "da": _hits(q).mean(), "log_loss": ll})
    return pd.DataFrame(rows)


def vs_reversal_table(preds, rule=REVERSAL) -> pd.DataFrame:
    """Mỗi mô hình (không phải luật) so với luật đảo chiều cùng ngày test: ΔDA = DA(mô hình) − DA(luật),
    McNemar chính xác trên các ngày hai bên bất đồng, và ΔDA từng năm. Dùng để kiểm đề xuất 4 của baseline_check:
    chỉ so On-chain khi baseline OHLCV thắng luật này ổn định."""
    ref = next((p for (m, _), p in preds.items() if m == rule), None)
    if ref is None:
        return pd.DataFrame()
    rows = []
    for (name, config), p in sorted(preds.items()):
        if name in RULES or name.startswith("b0"):
            continue
        if not np.array_equal(p[DATE].to_numpy("datetime64[ns]"), ref[DATE].to_numpy("datetime64[ns]")):
            raise ValueError(f"{name}_{config}: không cùng ngày test với {rule}.")
        a, b = _hits(ref), _hits(p)
        only_model, only_rule = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
        n = only_model + only_rule
        r = {"model": name, "config": config, "da_model": b.mean(), "da_rule": a.mean(), "d_da": b.mean() - a.mean(),
             "only_model_right": only_model, "only_rule_right": only_rule,
             "p_mcnemar": binomtest(only_model, n, 0.5).pvalue if n else 1.0}
        years = p[DATE].dt.year.to_numpy()
        d_year = {int(yr): b[years == yr].mean() - a[years == yr].mean() for yr in np.unique(years)}
        r |= {f"d_da_{yr}": v for yr, v in d_year.items()}
        r["n_years_model_better"] = int(sum(v > 0 for v in d_year.values()))
        r["n_years"] = len(d_year)
        rows.append(r)
    return pd.DataFrame(rows)


def write_tables(pred_dir, table_dir, log=print):
    """Δ On-chain, theo năm, so với luật đảo chiều: tính lại từ mọi file predictions trong thư mục."""
    preds = load_predictions(pred_dir)
    tables = {"e1_delta_draft": delta_table(preds), "e1_by_year": by_year_table(preds),
              "e1_vs_reversal": vs_reversal_table(preds)}
    for name, t in tables.items():
        log(f"Đã lưu {write_csv(t, Path(table_dir) / f'{name}.csv')}")
    return tables


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


def ohlcv10_sets(sets):
    """Bộ đặc trưng của các dự báo E1 hiện có: 10 đặc trưng OHLCV ban đầu (trước khi thêm up_today, 13/11/2026)
    + đặc trưng On-chain. build_dataset nay trả 11 đặc trưng OHLCV; dùng hàm này khi cần so với dự báo E1 cũ."""
    onchain = [c for c in sets["ohlcv_onchain"] if c not in sets["ohlcv"]]
    return {"ohlcv": list(FEATURES_10), "ohlcv_onchain": list(FEATURES_10) + onchain}


# Cờ dòng lệnh -> nhóm On-chain (data.ONCHAIN_GROUPS). Mỗi nhóm một cờ riêng, kết hợp được: --no-flows --no-activity.
DROP_FLAGS = {"no_flows": "Flows", "no_activity": "Activity", "no_mining": "Mining", "no_valuation": "Valuation"}


def drop_groups_sets(sets, fe_cfg, groups):
    """Bỏ mọi đặc trưng thuộc các nhóm On-chain `groups` (theo feature_groups của fe_cfg) khỏi mọi cấu hình.
    Chỉ ohlcv_onchain chứa On-chain nên bộ ohlcv giữ nguyên: dự báo OHLCV-only không đổi.
    Báo lỗi nếu nhóm không có trong cấu hình đặc trưng hoặc bỏ hết mọi nhóm (ohlcv_onchain sẽ trùng ohlcv)."""
    fg = features.feature_groups(**fe_cfg)
    missing = [g for g in groups if not fg.get(g)]
    if missing:
        raise ValueError(f"Cấu hình đặc trưng không có nhóm {', '.join(missing)} để bỏ.")
    if set(fg) <= set(groups):
        raise ValueError(f"Không được bỏ cả {len(fg)} nhóm On-chain: ohlcv_onchain sẽ trùng ohlcv.")
    drop = {f for g in groups for f in fg[g]}
    return {k: [c for c in v if c not in drop] for k, v in sets.items()}


def no_flows_sets(sets, fe_cfg):
    """Như drop_groups_sets(..., ["Flows"]) (giữ cho mã cũ)."""
    return drop_groups_sets(sets, fe_cfg, ["Flows"])


def add_drop_args(ap):
    """Thêm --no-flows, --no-activity, --no-mining, --no-valuation vào parser."""
    for dest, g in DROP_FLAGS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, action="store_true",
                        help=f"bỏ nhóm {g} khỏi cấu hình ohlcv_onchain (bắt buộc có --out)")


def dropped_groups(args):
    """Danh sách nhóm bị bỏ theo các cờ --no-*, thứ tự như DROP_FLAGS."""
    return [g for dest, g in DROP_FLAGS.items() if getattr(args, dest, False)]


def drop_label(groups):
    return f" (BỎ {', '.join(groups)})" if groups else ""


def out_base(out):
    """Thư mục kết quả: None -> pipeline/results/ (mặc định); còn lại tính từ gốc repo (đường dẫn tuyệt đối giữ nguyên)."""
    return RESULTS if out is None else data_mod.ROOT / out


def check_drop_out(base, groups):
    """groups: danh sách nhóm bị bỏ (hoặc bool như cũ). Có bỏ nhóm thì không được ghi vào pipeline/results/."""
    if groups and Path(base).resolve() == RESULTS.resolve():
        flags = " ".join("--" + d.replace("_", "-") for d, g in DROP_FLAGS.items()
                         if groups is True and g == "Flows" or groups is not True and g in groups)
        raise ValueError(f"{flags} cần --out <thư mục riêng>: không ghi dự báo bỏ nhóm On-chain đè lên kết quả E1 "
                         "chính trong pipeline/results/.")


check_no_flows_out = check_drop_out   # tên cũ


def _normalize_rec(rec):
    """feature_sets.json cũ ghi no_flows: bool; đổi sang drop_groups để so được với bản mới."""
    rec = dict(rec)
    if "no_flows" in rec and "drop_groups" not in rec:
        rec["drop_groups"] = ["Flows"] if rec.pop("no_flows") else []
    return rec


def record_feature_sets(out_dir, sets, **info):
    """Ghi {out_dir}/feature_sets.json: tuỳ chọn chạy (flows_set, drop_groups) và bộ đặc trưng của từng cấu hình.
    Thư mục đã có file này với nội dung khác thì báo lỗi, để không trộn dự báo bỏ các nhóm khác nhau
    (hoặc 10/11 đặc trưng OHLCV) trong cùng một thư mục. File cũ ghi no_flows: bool vẫn so được."""
    path = Path(out_dir) / "feature_sets.json"
    rec = {**info, "sets": {k: list(v) for k, v in sets.items()}}
    if path.exists():
        if _normalize_rec(json.loads(path.read_text())) != _normalize_rec(rec):
            raise ValueError(f"{path} ghi bộ đặc trưng khác lần chạy này (vd. bỏ nhóm On-chain khác). "
                             f"Chọn --out khác để không trộn kết quả.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return path


def stage_e1(model_names, every=1, ohlcv10=False, no_flows=False, out=None, drop_groups=()):
    """out=None: ghi vào pipeline/results/ như trước. Có out: ghi {out}/predictions/, {out}/tables/ và
    {out}/feature_sets.json. drop_groups: các nhóm On-chain bỏ khỏi ohlcv_onchain (bắt buộc có out);
    no_flows=True tương đương thêm "Flows" vào drop_groups."""
    groups = [g for g in DROP_FLAGS.values() if g in drop_groups or (no_flows and g == "Flows")]
    base = out_base(out)
    check_drop_out(base, groups)
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe_cfg)
    if ohlcv10:
        sets = ohlcv10_sets(sets)
    if groups:
        sets = drop_groups_sets(sets, fe_cfg, groups)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)[::every]
    print(f"Flows set {fe_cfg['flows_set']}{drop_label(groups)}; {len(sets['ohlcv'])} OHLCV, "
          f"{len(sets['ohlcv_onchain'])} tổng; {len(folds)} fold; ghi vào {base}")
    if base.resolve() != RESULTS.resolve():
        record_feature_sets(base, sets, flows_set=fe_cfg["flows_set"], drop_groups=groups)
    sub = "" if every == 1 else f"every{every}"
    run_e1(data, sets, model_names, folds, wf_cfg, base / "predictions" / sub, regime_map=load_regime_map())
    stage_tables(base / "predictions" / sub, base / "tables" / sub)   # gồm cả mô hình chạy từ trước


def stage_tables(pred_dir=RESULTS / "predictions", table_dir=RESULTS / "tables"):
    tables = write_tables(pred_dir, table_dir)
    print(tables["e1_delta_draft"].to_string(index=False, float_format="%.4f"))
    vs = tables["e1_vs_reversal"]
    if len(vs):
        print(vs[["model", "config", "da_model", "da_rule", "d_da", "p_mcnemar", "n_years_model_better", "n_years"]]
              .to_string(index=False, float_format="%.4f"))


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
        preds = run_e1(data, sets, ["b1", "b0pers", "b0maj"], folds, wf_cfg, tmp, log=lambda _: None)
        back = load_predictions(tmp)
        assert set(back) == set(preds)
        for k, p in preds.items():
            assert list(back[k].columns) == PREDICTION_COLS
            np.testing.assert_array_equal(back[k]["p_hat"], p["p_hat"])
        pd.testing.assert_frame_equal(preds[("b0pers", "ohlcv")], preds[("b0pers", "ohlcv_onchain")])
        table = delta_table(back)
        np.testing.assert_allclose(table["delta"], table["ohlcv_onchain"] - table["ohlcv"])
        # 1b) Ô không có nghĩa để trống: log-loss, Brier của luật; AUC của b0maj. lr thì đủ 4 metric
        t = table.set_index(["model", "metric"])["ohlcv"]
        assert np.isnan(t[("b0pers", "log_loss")]) and np.isnan(t[("b0pers", "brier")])
        assert np.isnan(t[("b0maj", "auc")]) and not np.isnan(t[("b0maj", "log_loss")])
        assert t.loc["b1"].notna().all()
        # 2) Bảng Δ bỏ qua file không phải cấu hình E1 (vd. ablation sau này)
        (Path(tmp) / "b4_ohlcv_plus_flows.csv").write_text("x")
        assert set(load_predictions(tmp)) == set(preds)
        # 2b) Bảng theo năm: số ngày cộng lại đủ, DA gộp theo trọng số ngày = DA cả giai đoạn; luật không có log-loss
        run_e1(data, sets, ["b0rev"], folds, wf_cfg, tmp, log=lambda _: None)
        back = load_predictions(tmp)
        yt = by_year_table(back)
        for (name, config), p in back.items():
            s = yt[(yt.model == name) & (yt.config == config)]
            assert s.n_days.sum() == len(p)
            np.testing.assert_allclose((s.da * s.n_days).sum() / s.n_days.sum(), _hits(p).mean())
            assert s.log_loss.isna().all() == (name in RULES)
        # 2c) So với luật đảo chiều: chỉ mô hình có học; ΔDA gộp và McNemar khớp tính tay; luật so với chính nó = 0
        vs = vs_reversal_table(back)
        assert set(vs.model) == {"b1"}
        row = vs[vs.config == "ohlcv"].iloc[0]
        a, b = _hits(back[("b0rev", "ohlcv")]), _hits(back[("b1", "ohlcv")])
        np.testing.assert_allclose(row.d_da, b.mean() - a.mean())
        assert row.only_model_right - row.only_rule_right == int(b.sum() - a.sum())
        assert vs_reversal_table({k: v for k, v in back.items() if k[0] != "b0rev"}).empty
    # 3) Hai cấu hình khác ngày test thì không so cặp
    p = preds[("b1", "ohlcv")]
    data_mod._raises(lambda: delta_table({("b1", "ohlcv"): p, ("b1", "ohlcv_onchain"): p.iloc[1:]}), "không cùng")

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
    # ohlcv10_sets: 10 đặc trưng OHLCV ban đầu, giữ nguyên phần On-chain
    s10 = ohlcv10_sets({"ohlcv": FEATURES_10 + ["up_today"], "ohlcv_onchain": FEATURES_10 + ["up_today", "oc1", "oc2"]})
    assert s10 == {"ohlcv": FEATURES_10, "ohlcv_onchain": FEATURES_10 + ["oc1", "oc2"]}

    # 6) no_flows_sets: bỏ đúng các đặc trưng nhóm Flows, giữ thứ tự phần còn lại; bộ ohlcv không đổi
    flows = features.feature_groups(**fe_cfg)["Flows"]
    nf = no_flows_sets(sets, fe_cfg)
    assert nf["ohlcv"] == sets["ohlcv"]
    assert nf["ohlcv_onchain"] == [c for c in sets["ohlcv_onchain"] if c not in flows]
    assert len(nf["ohlcv_onchain"]) == len(sets["ohlcv_onchain"]) - len(flows) and flows
    assert no_flows_sets(ohlcv10_sets(sets), fe_cfg)["ohlcv"] == FEATURES_10
    data_mod._raises(lambda: no_flows_sets(sets, {**fe_cfg, "drop_features": flows}), "không có nhóm Flows")
    # 6a) drop_groups_sets: mỗi nhóm bỏ riêng hoặc kết hợp; bỏ cả 4 nhóm thì báo lỗi
    fg = features.feature_groups(**fe_cfg)
    for combo in (["Activity"], ["Mining"], ["Valuation"], ["Flows", "Activity"], ["Flows", "Mining", "Valuation"]):
        d = drop_groups_sets(sets, fe_cfg, combo)
        gone = {f for g in combo for f in fg[g]}
        assert d["ohlcv"] == sets["ohlcv"]
        assert d["ohlcv_onchain"] == [c for c in sets["ohlcv_onchain"] if c not in gone] and gone
    assert drop_groups_sets(sets, fe_cfg, ["Flows"]) == nf
    data_mod._raises(lambda: drop_groups_sets(sets, fe_cfg, list(fg)), "cả 4 nhóm")
    data_mod._raises(lambda: drop_groups_sets(sets, fe_cfg, ["Nope"]), "không có nhóm Nope")
    ap_t = argparse.ArgumentParser()
    add_drop_args(ap_t)
    assert dropped_groups(ap_t.parse_args(["--no-activity", "--no-flows"])) == ["Flows", "Activity"]
    assert dropped_groups(ap_t.parse_args([])) == []
    # 6b) feature_sets.json: ghi lần đầu, chạy lại cùng bộ thì được, khác bộ (có Flows) thì báo lỗi
    with tempfile.TemporaryDirectory() as tmp:
        p = record_feature_sets(tmp, nf, flows_set="A", drop_groups=["Flows"])
        assert json.loads(p.read_text())["sets"]["ohlcv_onchain"] == nf["ohlcv_onchain"]
        record_feature_sets(tmp, nf, flows_set="A", drop_groups=["Flows"])
        data_mod._raises(lambda: record_feature_sets(tmp, sets, flows_set="A", drop_groups=[]), "bộ đặc trưng khác")
        na = drop_groups_sets(sets, fe_cfg, ["Flows", "Activity"])
        data_mod._raises(lambda: record_feature_sets(tmp, na, flows_set="A", drop_groups=["Flows", "Activity"]),
                         "bộ đặc trưng khác")
    # file cũ (no_flows: true) vẫn khớp lần chạy mới bỏ Flows
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "feature_sets.json").write_text(json.dumps({"flows_set": "A", "no_flows": True, "sets": nf}))
        record_feature_sets(tmp, nf, flows_set="A", drop_groups=["Flows"])
        data_mod._raises(lambda: record_feature_sets(tmp, na, flows_set="A", drop_groups=["Flows", "Activity"]),
                         "bộ đặc trưng khác")
    # 6c) bỏ nhóm nào cũng không được ghi vào pipeline/results/ mặc định
    data_mod._raises(lambda: check_no_flows_out(out_base(None), True), "--no-flows cần --out")
    data_mod._raises(lambda: check_drop_out(out_base(None), ["Activity"]), "--no-activity cần --out")
    check_drop_out(out_base(None), [])
    print("Tất cả test run_all đều PASS")


def run_tests():
    for mod in (data_mod, features, walkforward, models):
        mod._tests()
    _tests()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["data", "e0", "e1", "regime", "tables"])
    ap.add_argument("--models", nargs="+", default=["b1"])
    ap.add_argument("--every", type=int, default=1, help="chỉ chạy mỗi k fold (để chạy thử)")
    ap.add_argument("--ohlcv10", action="store_true",
                    help="dùng 10 đặc trưng OHLCV ban đầu (như dự báo E1 hiện có), bỏ up_today")
    add_drop_args(ap)   # --no-flows, --no-activity, --no-mining, --no-valuation (chỉ --stage e1, bắt buộc có --out)
    ap.add_argument("--out", default=None,
                    help="thư mục kết quả (tính từ gốc repo) cho --stage e1 / tables; mặc định pipeline/results/")
    ap.add_argument("--test", action="store_true", help="chỉ chạy test")
    args = ap.parse_args(argv)
    if not args.test and not args.stage:
        ap.error("cần --stage hoặc --test")
    groups = dropped_groups(args)
    if groups and args.stage != "e1":
        ap.error("--no-flows / --no-activity / --no-mining / --no-valuation chỉ dùng với --stage e1")
    if args.out is not None and args.stage not in ("e1", "tables"):
        ap.error("--out chỉ dùng với --stage e1 hoặc tables")
    if groups:
        try:
            check_drop_out(out_base(args.out), groups)
        except ValueError as e:
            ap.error(str(e))

    run_tests()
    if args.stage == "data":
        stage_data()
    elif args.stage == "e1":
        stage_e1(args.models, args.every, args.ohlcv10, out=args.out, drop_groups=groups)
    elif args.stage == "regime":
        stage_regime()
    elif args.stage == "tables":
        base = out_base(args.out)
        stage_tables(base / "predictions", base / "tables")
    elif args.stage == "e0":
        print("Stage e0 chưa cài (task 2.9, 2.10).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
