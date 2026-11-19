"""
rerun_e1.py  –  Chạy lại E1 cho các baseline B0–B3 vào một thư mục riêng, không đè kết quả cũ.

Dùng đúng code hiện có (run_all.run_e1, b1t_tuning.stage_run), không sửa mô hình nào.
Bộ đặc trưng như mọi dự báo E1 đã có: 10 đặc trưng OHLCV ban đầu (run_all.ohlcv10_sets) + 33 On-chain v1.

  python pipeline/src/rerun_e1.py --out pipeline/results/rerun_2026-11-19
  python pipeline/src/rerun_e1.py --out ... --models b0maj b0pers b0rev b1 b1fs b3 b3t b2 b2t b1t

Output trong --out:
  predictions/{model}_{config}.csv     cùng định dạng với pipeline/results/predictions/
  log/b1t_{config}/                    log chọn tham số của b1t (như results/b1t/e1/)
  run_log.csv                          thời gian chạy mỗi mô hình
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import b1t_tuning
from data import DATE, ROOT, TARGET, load_config, load_master, read_csv_dated, write_csv
from features import build_dataset
from run_all import CONFIGS, load_regime_map, ohlcv10_sets, run_e1
from walkforward import make_folds

DEFAULT_MODELS = ["b0maj", "b0pers", "b0rev", "b1", "b1fs", "b3", "b3t", "b1t", "b2", "b2t"]


def run_b1t(out: Path):
    """b1t chạy bằng b1t_tuning.stage_run (cấu hình E1: features_onchain.yaml, tiêu chí log-loss).
    setup() của b1t_tuning trả 11 đặc trưng OHLCV; thay bằng 10 đặc trưng ban đầu như dự báo E1 đã có."""
    orig = b1t_tuning.setup

    def setup10(fe):
        data, sets, folds, wf, cfg = orig(fe)
        return data, ohlcv10_sets(sets), folds, wf, cfg

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
    args = ap.parse_args(argv)
    out = ROOT / args.out
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe_cfg)          # load_master kiểm sha256 với MANIFEST
    sets = ohlcv10_sets(sets)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)
    print(f"{len(sets['ohlcv'])} OHLCV, {len(sets['ohlcv_onchain'])} tổng; {len(folds)} fold", flush=True)
    rmap = load_regime_map()
    log_path = out / "run_log.csv"
    for name in args.models:
        t0 = time.time()
        if name == "b1t":
            run_b1t(out)
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
