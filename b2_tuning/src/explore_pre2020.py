"""
explore_pre2020.py – khảo sát siêu tham số RF CHỈ trên cửa sổ train của fold 0 (2017-12-31 → 2019-12-30),
tức hoàn toàn trước ngày test đầu tiên (2020-01-01). Không chạm dữ liệu test nên không gây data snooping.
Dùng đúng inner CV của repo: TimeSeriesSplit(5, gap 1), log-loss, scaler trong pipeline.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO / "ngoc_pipeline/src"))
import walkforward  # noqa: E402
from data import DATE, TARGET, load_config, load_master  # noqa: E402
from features import build_dataset  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

fe_cfg, wf_cfg = load_config("features_onchain.yaml"), load_config("walkforward.yaml")
data, sets = build_dataset(load_master(), fe_cfg)
folds = walkforward.make_folds(data.loc[data[TARGET].notna(), DATE], **wf_cfg)
tr, _ = walkforward.fold_indices(data, folds[0])
assert data.loc[tr, DATE].max() < pd.Timestamp("2020-01-01")
cv = walkforward.inner_cv(wf_cfg["inner_splits"], wf_cfg["gap_days"])

space = {
    "max_features": ["sqrt", 0.1, 0.33, 0.5, 1.0],
    "min_samples_leaf": [10, 20, 50, 100, 150],
    "max_depth": [3, 6, None],
    "max_samples": [None, 0.5],
}
rows = []
for config in ("ohlcv", "ohlcv_onchain"):
    train = data.loc[tr].dropna(subset=sets[config])
    X, y = train[sets[config]].to_numpy(), train[TARGET].astype(int).to_numpy()
    t0 = time.time()
    for vals in itertools.product(*space.values()):
        params = dict(zip(space, vals))
        est = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, **params)
        s = cross_val_score(walkforward.make_pipeline(est), X, y, cv=cv, scoring="neg_log_loss")
        rows.append({"config": config, **params, "cv_logloss": -s.mean(), "cv_std": s.std()})
    print(config, "done", round(time.time() - t0), "s", flush=True)

df = pd.DataFrame(rows)
df["max_depth"] = df["max_depth"].astype(object).where(df["max_depth"].notna(), "None")
df["max_samples"] = df["max_samples"].astype(object).where(df["max_samples"].notna(), "None")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
prior = float(np.mean([-cross_val_score(walkforward.make_pipeline(
    __import__("sklearn.dummy", fromlist=["DummyClassifier"]).DummyClassifier(strategy="prior")),
    X, y, cv=cv, scoring="neg_log_loss").mean()]))
print("b0maj (prior) inner-CV log-loss:", round(prior, 5))
for config, g in df.groupby("config"):
    print(f"\n== {config}: top 10")
    print(g.sort_values("cv_logloss").head(10).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    for col in space:
        print(f"  mean log-loss theo {col}:", g.groupby(col)["cv_logloss"].mean().round(5).to_dict())
