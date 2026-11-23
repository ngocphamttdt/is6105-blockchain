"""
rerun_e1.py  –  Chạy lại E1 cho các baseline B0–B3 vào một thư mục riêng, không đè kết quả cũ.

Dùng đúng code hiện có (run_all.run_e1, b1t_tuning.stage_run), không sửa mô hình nào.
Bộ đặc trưng như mọi dự báo E1 đã có: 10 đặc trưng OHLCV ban đầu (run_all.ohlcv10_sets) + 33 On-chain v1.

  python pipeline/src/rerun_e1.py --out pipeline/results/rerun_2026-11-19
  python pipeline/src/rerun_e1.py --out ... --models b0maj b0pers b0rev b1 b1fs b3 b3t b2 b2t b1t
  python pipeline/src/rerun_e1.py --out pipeline/results/no_flows --no-flows \
      --models b0maj b0pers b0rev b1 b1fs b1t b2 b2t b3 b3t b4 b4t b5
                                       bỏ nhóm Flows khỏi cấu hình ohlcv_onchain cho mọi mô hình (kể cả b1t, b5)

Output trong --out:
  predictions/{model}_{config}.csv     cùng định dạng với pipeline/results/predictions/
  log/b1t_{config}/                    log chọn tham số của b1t (như results/b1t/e1/)
  run_log.csv                          thời gian chạy mỗi mô hình
  feature_sets.json                    bộ đặc trưng đã dùng (flows_set, no_flows); --out đã có file này với bộ khác
                                       thì dừng, không trộn kết quả có Flows và bỏ Flows
--no-flows: tên cấu hình vẫn là ohlcv | ohlcv_onchain nên rerun_tables.py / rerun_figures.py dùng được nguyên;
  ohlcv giữ 10 đặc trưng, ohlcv_onchain = 10 OHLCV + 27 On-chain (33 trừ 6 đặc trưng Flows A).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import b1t_tuning
from data import DATE, ROOT, TARGET, load_config, load_master, read_csv_dated, write_csv
from features import build_dataset
from run_all import CONFIGS, check_no_flows_out, load_regime_map, no_flows_sets, ohlcv10_sets, record_feature_sets, run_e1
from walkforward import make_folds

DEFAULT_MODELS = ["b0maj", "b0pers", "b0rev", "b1", "b1fs", "b3", "b3t", "b1t", "b2", "b2t"]


def run_b1t(out: Path, no_flows=False):
    """b1t chạy bằng b1t_tuning.stage_run (cấu hình E1: features_onchain.yaml, tiêu chí log-loss).
    setup() của b1t_tuning trả 11 đặc trưng OHLCV; thay bằng 10 đặc trưng ban đầu như dự báo E1 đã có.
    no_flows: phần On-chain thêm vào mỗi tập ứng viên của b1t cũng bỏ nhóm Flows."""
    orig = b1t_tuning.setup

    def setup10(fe):
        data, sets, folds, wf, cfg = orig(fe)
        sets = ohlcv10_sets(sets)
        return data, no_flows_sets(sets, cfg) if no_flows else sets, folds, wf, cfg

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
    ap.add_argument("--no-flows", action="store_true", help="bỏ nhóm Flows khỏi cấu hình ohlcv_onchain cho mọi mô hình")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    try:
        check_no_flows_out(out, args.no_flows)
    except ValueError as e:
        ap.error(str(e))
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe_cfg)          # load_master kiểm sha256 với MANIFEST
    sets = ohlcv10_sets(sets)
    if args.no_flows:
        sets = no_flows_sets(sets, fe_cfg)
    folds = make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)
    print(f"{len(sets['ohlcv'])} OHLCV, {len(sets['ohlcv_onchain'])} tổng{' (BỎ Flows)' if args.no_flows else ''}; "
          f"{len(folds)} fold; ghi vào {out}", flush=True)
    record_feature_sets(out, sets, flows_set=fe_cfg["flows_set"], no_flows=args.no_flows)
    rmap = load_regime_map()
    log_path = out / "run_log.csv"
    for name in args.models:
        t0 = time.time()
        if name == "b1t":
            run_b1t(out, args.no_flows)
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
