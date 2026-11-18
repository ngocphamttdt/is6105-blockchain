"""
rf_runner.py – chạy B2 Random Forest qua đúng walk-forward của pipeline với một grid tuỳ chọn,
ghi predictions cho 2 cấu hình, log tham số được chọn ở mỗi lần tune, và in metric.

python rf_runner.py --repo <repo> --grid <grid.yaml> --out <dir> [--every k]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--grid", required=True, help="yaml có params, param_grid (cùng định dạng configs/models/b2.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--configs", nargs="+", default=["ohlcv", "ohlcv_onchain"])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "pipeline/src"))
    import walkforward
    from data import DATE, TARGET, load_config, load_master, write_csv
    from features import build_dataset
    from run_all import metrics
    from sklearn.ensemble import RandomForestClassifier

    cfg = yaml.safe_load(Path(args.grid).read_text())
    fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
    data, sets = build_dataset(load_master(), fe_cfg)
    folds = walkforward.make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)[::args.every]

    # ghi lại tham số được chọn ở mỗi lần tune (không đổi logic tune của repo)
    chosen, orig_tune = [], walkforward.tune

    def logged_tune(pipe, param_grid, X, y, cv):
        best = orig_tune(pipe, param_grid, X, y, cv)
        chosen.append(best)
        return best

    walkforward.tune = logged_tune

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for config in args.configs:
        chosen.clear()
        t0 = time.time()
        est = RandomForestClassifier(**(cfg.get("params") or {}))
        p = walkforward.run_walkforward(data, sets[config], est, folds, param_grid=cfg.get("param_grid") or {},
                                        **wf_cfg)
        write_csv(p, out / f"b2_{config}.csv")
        m = metrics(p, "b2")
        m["pct_up"] = float((p["p_hat"] > 0.5).mean())
        m["seconds"] = round(time.time() - t0, 1)
        m["chosen"] = [dict(c) for c in chosen]
        summary[config] = m
        print(config, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "chosen"},
              flush=True)
        print("  chosen:", pd.DataFrame(chosen).value_counts().to_dict(), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
