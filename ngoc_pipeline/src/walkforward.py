"""
walkforward.py  –  Task 1.15 (Ngọc): walk-forward chống rò rỉ.

Protocol: train trượt 730 ngày, test 30 ngày, bước 30 ngày, test 01/2020 – 08/2026 (82 fold).
- Bỏ gap_days = 1 ngày giữa train và test: nhãn ngày t dùng close ngày t+1, nên nếu train kết thúc ngay
  trước test thì nhãn ngày train cuối đã chứa giá của ngày test đầu tiên.
- Scaler nằm trong sklearn Pipeline nên được fit lại trong từng cửa sổ train và từng inner fold.
- Tune bằng GridSearchCV trên TimeSeriesSplit(5, gap=1), tiêu chí log-loss, tune lại mỗi 6 fold.
- Output: DataFrame 5 cột date, fold, y_true, p_hat, regime (regime để trống, Ngân điền).

python walkforward.py         -> chạy test rồi in fold trên ngày thật, vẽ results/figures/walkforward.png
python walkforward.py --test  -> chỉ chạy test
"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (CLOSE, DATE, PREDICTION_COLS, RESULTS, TARGET, _raises, load_config, load_master,
                  make_synthetic_master)

DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(dates, train_days=730, test_days=30, step_days=30, gap_days=1,
               test_start="2020-01-01", test_end="2026-08-31", **_) -> list[Fold]:
    """`dates` nên là các ngày CÓ NHÃN (master từ 2016 nhưng OHLCV chỉ có từ 17/08/2017).
    Fold cuối có thể ngắn hơn test_days. **_ nhận thẳng walkforward.yaml."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    start, end = pd.Timestamp(test_start), min(pd.Timestamp(test_end), dates.max())
    folds, k = [], 0
    while (ts := start + k * step_days * DAY) <= end:
        tr_end = ts - (gap_days + 1) * DAY
        folds.append(Fold(k, tr_end - (train_days - 1) * DAY, tr_end, ts, min(ts + (test_days - 1) * DAY, end)))
        k += 1
    if not folds:
        raise ValueError("Không có fold nào: kiểm tra test_start/test_end.")
    if dates.min() > folds[0].train_start:
        raise ValueError(f"Dữ liệu bắt đầu {dates.min().date()} nhưng fold đầu cần train từ "
                         f"{folds[0].train_start.date()}.")
    return folds


def fold_indices(df, fold, target=TARGET):
    """Index dòng train/test của fold; bỏ dòng không có nhãn."""
    has_y = df[target].notna()
    train = df.index[df[DATE].between(fold.train_start, fold.train_end) & has_y]
    test = df.index[df[DATE].between(fold.test_start, fold.test_end) & has_y]
    return train, test


def inner_cv(n_splits=5, gap=1):
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])


def tune(pipe, param_grid, X, y, cv) -> dict:
    """Grid search theo log-loss; param_grid dùng tên tham số của mô hình (không có tiền tố).
    Điểm làm tròn 10 chữ số, hoà thì lấy tổ hợp đứng trước trong grid: các tổ hợp cho cùng mô hình
    (vd. RF max_depth None và 6 khi min_samples_leaf lớn) chỉ lệch nhau ~1e-16 do cộng song song,
    nếu không làm tròn thì lần chạy lại có thể chọn tổ hợp khác."""
    grid = {f"model__{k}": v for k, v in param_grid.items()}
    res = GridSearchCV(pipe, grid, scoring="neg_log_loss", cv=cv, refit=False).fit(X, y).cv_results_
    best = res["params"][int(np.argmax(np.round(res["mean_test_score"], 10)))]
    return {k.removeprefix("model__"): v for k, v in best.items()}


def run_walkforward(df, feature_cols, estimator, folds, param_grid=None, retune_every=6,
                    inner_splits=5, gap_days=1, target=TARGET, **_) -> pd.DataFrame:
    """Chạy một mô hình sklearn qua mọi fold, trả predictions đúng định dạng chung.
    Cửa sổ train thiếu nhãn là lỗi; dòng train còn NaN (warm-up) bị bỏ; NaN trong test là lỗi."""
    feature_cols, best, out = list(feature_cols), {}, []
    for i, fold in enumerate(folds):
        tr, te = fold_indices(df, fold, target)
        window = (fold.train_end - fold.train_start).days + 1
        if len(tr) < window:
            raise ValueError(f"Fold {fold.fold_id}: chỉ {len(tr)}/{window} ngày train có nhãn "
                             f"(train từ {fold.train_start.date()}).")
        train, test = df.loc[tr].dropna(subset=feature_cols), df.loc[te]
        if test[feature_cols].isna().any().any():
            raise ValueError(f"Fold {fold.fold_id}: có NaN trong đặc trưng của tập test.")
        X, y = train[feature_cols].to_numpy(), train[target].astype(int).to_numpy()
        pipe = make_pipeline(estimator)
        if param_grid and i % retune_every == 0:
            best = tune(pipe, param_grid, X, y, inner_cv(inner_splits, gap_days))
        pipe.set_params(**{f"model__{k}": v for k, v in best.items()}).fit(X, y)
        out.append(pd.DataFrame({"date": test[DATE].to_numpy(), "fold": fold.fold_id,
                                 "y_true": test[target].astype(int).to_numpy(),
                                 "p_hat": pipe.predict_proba(test[feature_cols].to_numpy())[:, 1],
                                 "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def plot_folds(folds, path, every=1):
    """Sơ đồ walk-forward cho mục Method: train (xanh), gap (xám), test (cam)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#4C78A8", "#BBBBBB", "#F58518")
    shown = folds[::every]
    fig, ax = plt.subplots(figsize=(10, max(3, len(shown) * 0.12)))
    for row, f in enumerate(shown):
        ax.barh(row, f.train_end - f.train_start + DAY, left=f.train_start, color=colors[0], height=0.8)
        ax.barh(row, f.test_start - f.train_end - DAY, left=f.train_end + DAY, color=colors[1], height=0.8)
        ax.barh(row, f.test_end - f.test_start + DAY, left=f.test_start, color=colors[2], height=0.8)
    ax.invert_yaxis()
    ax.set_ylabel("Fold")
    ax.set_title(f"Walk-forward: {len(folds)} fold, train {(folds[0].train_end - folds[0].train_start).days + 1} "
                 f"ngày, test {(folds[0].test_end - folds[0].test_start).days + 1} ngày")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels=["Train", "Gap", "Test"],
              loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Test:  python walkforward.py --test
# ----------------------------------------------------------------------------
def _tests():
    cfg = load_config("walkforward.yaml")
    m = make_synthetic_master(seed=42)
    m[TARGET] = (m[CLOSE].shift(-1) > m[CLOSE]).astype(float).where(m[CLOSE].shift(-1).notna())
    folds = make_folds(m[DATE], **cfg)

    # 1) Train luôn trước test, cách đúng gap; nhãn ngày train cuối không chạm ngày test
    for f in folds:
        tr, te = fold_indices(m, f)
        assert (f.test_start - f.train_end).days == cfg["gap_days"] + 1
        assert m.loc[tr, DATE].max() + DAY < m.loc[te, DATE].min()
        assert len(tr) == cfg["train_days"]

    # 2) Các khối test liền nhau, không chồng lấn, phủ đúng một lần mọi ngày có nhãn 01/2020 – 08/2026
    test_dates = pd.concat([m.loc[fold_indices(m, f)[1], DATE] for f in folds])
    assert test_dates.is_unique and folds[0].test_start == pd.Timestamp("2020-01-01")
    assert test_dates.reset_index(drop=True).equals(pd.Series(pd.date_range("2020-01-01", "2026-08-30"), name=DATE))

    # 3) Scaler chỉ học từ train: phá dữ liệu test (x1000) thì dự báo vẫn khớp pipeline chỉ fit trên train gốc
    cols = ["cm_AdrActCnt", "cm_HashRate", "ohlcv_volume"]
    tr, te = fold_indices(m, folds[10])
    pipe = make_pipeline(LogisticRegression()).fit(m.loc[tr, cols].to_numpy(), m.loc[tr, TARGET].astype(int))
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, m.loc[tr, cols].to_numpy().mean(axis=0))
    broken = m.copy()
    broken.loc[te, cols] *= 1000
    res = run_walkforward(broken, cols, LogisticRegression(), folds[10:11])
    np.testing.assert_allclose(res["p_hat"], pipe.predict_proba(broken.loc[te, cols].to_numpy())[:, 1])

    # 4) Inner CV theo thời gian, có gap
    for a, b in inner_cv(5, 1).split(np.zeros((730, 1))):
        assert a.max() + 1 < b.min()

    # 5) Đặc trưng ngẫu nhiên thì DA quanh 50% (bản nhỏ của E0(b)); output đúng 5 cột, có tune
    noise = m.assign(n1=np.random.default_rng(0).normal(size=len(m)), n2=np.random.default_rng(1).normal(size=len(m)))
    res = run_walkforward(noise, ["n1", "n2"], LogisticRegression(), folds[::4], param_grid={"C": [0.1, 1.0]})
    assert list(res.columns) == PREDICTION_COLS and res["date"].is_unique
    assert abs(((res["p_hat"] > 0.5).astype(int) == res["y_true"]).mean() - 0.5) < 0.06

    # 5b) Tune: hai tổ hợp chỉ lệch ~1e-15 điểm thì coi là hoà, luôn lấy tổ hợp đứng trước
    tr, _ = fold_indices(noise, folds[0])
    Xn, yn = noise.loc[tr, ["n1", "n2"]].to_numpy(), noise.loc[tr, TARGET].astype(int).to_numpy()
    for grid in ([1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]):
        assert tune(make_pipeline(LogisticRegression()), {"C": grid}, Xn, yn, inner_cv())["C"] == grid[0]

    # 6) Báo lỗi: train thiếu nhãn, NaN trong test, dữ liệu không đủ cho fold đầu (cửa sổ 1.095 ngày)
    bad = m.copy()
    bad.loc[fold_indices(m, folds[0])[0][:10], TARGET] = np.nan
    _raises(lambda: run_walkforward(bad, ["cm_AdrActCnt"], LogisticRegression(), folds[:1]), "720/730")
    bad = m.copy()
    bad.loc[fold_indices(m, folds[0])[1][0], "cm_AdrActCnt"] = np.nan
    _raises(lambda: run_walkforward(bad, ["cm_AdrActCnt"], LogisticRegression(), folds[:1]), "NaN")
    _raises(lambda: make_folds(m[DATE], **{**cfg, "train_days": 1095}), "2016-12-31")
    print("Tất cả test walkforward đều PASS")


if __name__ == "__main__":
    _tests()
    if "--test" not in sys.argv:
        m = load_master()
        y = (m[CLOSE].shift(-1) > m[CLOSE]).where(m[CLOSE].shift(-1).notna() & m[CLOSE].notna())
        folds = make_folds(m.loc[y.notna(), DATE], **load_config("walkforward.yaml"))
        print(f"{len(folds)} fold; fold 0: train {folds[0].train_start.date()} → {folds[0].train_end.date()}, "
              f"test {folds[0].test_start.date()} → {folds[0].test_end.date()}; "
              f"fold cuối test {folds[-1].test_start.date()} → {folds[-1].test_end.date()}")
        print("Đã lưu", plot_folds(folds, RESULTS / "figures/walkforward.png"))
