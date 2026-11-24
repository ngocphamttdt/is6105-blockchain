"""
rerun_e1.py  –  Chạy lại E1 cho các baseline B0–B3 vào một thư mục riêng, không đè kết quả cũ.

Dùng đúng code hiện có (run_all.run_e1, b1t_tuning.stage_run), không sửa mô hình nào.
Bộ đặc trưng như mọi dự báo E1 đã có: 10 đặc trưng OHLCV ban đầu (run_all.ohlcv10_sets) + 33 On-chain v1.

  python pipeline/src/rerun_e1.py --out pipeline/results/rerun_2026-11-19
  python pipeline/src/rerun_e1.py --out ... --models b0maj b0pers b0rev b1 b1fs b3 b3t b2 b2t b1t
  python pipeline/src/rerun_e1.py --out pipeline/results/no_flows --no-flows \
      --models b0maj b0pers b0rev b1 b1fs b1t b2 b2t b3 b3t b4 b4t b5
                                       bỏ nhóm Flows khỏi cấu hình ohlcv_onchain cho mọi mô hình (kể cả b1t, b5)
  python pipeline/src/rerun_e1.py --out pipeline/results/no_flows_activity --no-flows --no-activity --models ...
                                       bỏ nhiều nhóm: --no-flows, --no-activity, --no-mining, --no-valuation

Output trong --out:
  predictions/{model}_{config}.csv     cùng định dạng với pipeline/results/predictions/
  log/b1t_{config}/                    log chọn tham số của b1t (như results/b1t/e1/)
  run_log.csv                          thời gian chạy mỗi mô hình
  feature_sets.json                    bộ đặc trưng đã dùng (flows_set, drop_groups); --out đã có file này với bộ
                                       khác thì dừng, không trộn kết quả bỏ các nhóm khác nhau
--no-flows / --no-activity / --no-mining / --no-valuation: mỗi cờ bỏ một nhóm On-chain, dùng riêng hoặc kết hợp.
  Tên cấu hình vẫn là ohlcv | ohlcv_onchain nên rerun_tables.py / rerun_figures.py dùng được nguyên;
  ohlcv giữ 10 đặc trưng, ohlcv_onchain = 10 OHLCV + 33 On-chain trừ nhóm bị bỏ
  (Flows A: 6 đặc trưng; Activity, Mining, Valuation: mỗi nhóm 9). Vd. --no-flows --no-activity: 10 + 18.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import b1t_tuning
from data import DATE, ROOT, TARGET, load_config, load_master, read_csv_dated, write_csv
from features import build_dataset
from run_all import (CONFIGS, add_drop_args, check_drop_out, drop_groups_sets, drop_label, dropped_groups,
                     load_regime_map, ohlcv10_sets, record_feature_sets, run_e1)
from walkforward import make_folds

DEFAULT_MODELS = ["b0maj", "b0pers", "b0rev", "b1", "b1fs", "b3", "b3t", "b1t", "b2", "b2t"]


def run_b1t(out: Path, drop_groups=()):
    """b1t chạy bằng b1t_tuning.stage_run (cấu hình E1: features_onchain.yaml, tiêu chí log-loss).
    setup() của b1t_tuning trả 11 đặc trưng OHLCV; thay bằng 10 đặc trưng ban đầu như dự báo E1 đã có.
    drop_groups: phần On-chain thêm vào mỗi tập ứng viên của b1t cũng bỏ các nhóm này."""
    orig = b1t_tuning.setup

    def setup10(fe):
        data, sets, folds, wf, cfg = orig(fe)
        sets = ohlcv10_sets(sets)
        return data, drop_groups_sets(sets, cfg, drop_groups) if drop_groups else sets, folds, wf, cfg

    b1t_tuning.setup = setup10
    try:
        for cfg in CONFIGS:
            tmp = out / "log" / f"b1t_{cfg}"
            b1t_tuning.stage_run("features_onchain.yaml", onchain=cfg == "ohlcv_onchain", out_dir=tmp,
                                 criteria=("logloss",), names={"logloss": f"b1t_{cfg}"})
            p = tmp / "predictions" / f"b1t_{cfg}.csv"
            write_csv(read_csv_dated(p), out / "predictions" / f"b1t_{cfg}.csv")
            p.unlink()
            p.parent.rmdir()
    finally:
        b1t_tuning.setup = orig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="thư mục kết quả, tính từ gốc repo")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    add_drop_args(ap)   # --no-flows, --no-activity, --no-mining, --no-valuation
    args = ap.parse_args(argv)
    out = ROOT / args.out
    groups = dropped_groups(args)
    try:
        check_drop_out(out, groups)
    except ValueError as e:
        ap.error(str(e))
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe_cfg)          # load_master kiểm sha256 với MANIFEST
    sets = ohlcv10_sets(sets)
    if groups:
        try:
            sets = drop_groups_sets(sets, fe_cfg, groups)
        except ValueError as e:
            ap.error(str(e))
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)
    print(f"{len(sets['ohlcv'])} OHLCV, {len(sets['ohlcv_onchain'])} tổng{drop_label(groups)}; "
          f"{len(folds)} fold; ghi vào {out}", flush=True)
    try:
        record_feature_sets(out, sets, flows_set=fe_cfg["flows_set"], drop_groups=groups)
    except ValueError as e:
        ap.error(str(e))
    rmap = load_regime_map()
    log_path = out / "run_log.csv"
    for name in args.models:
        t0 = time.time()
        if name == "b1t":
            run_b1t(out, groups)
        else:
            run_e1(data, sets, [name], folds, wf_cfg, out / "predictions", regime_map=rmap,
                   log=lambda s: print(s, flush=True))
        row = pd.DataFrame([{"model": name, "seconds": round(time.time() - t0, 1),
                             "finished_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")}])
        old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        write_csv(pd.concat([old[old.get("model", pd.Series(dtype=str)) != name] if len(old) else old, row]), log_path)
        print(f"== {name} xong sau {time.time() - t0:.0f} giây", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
