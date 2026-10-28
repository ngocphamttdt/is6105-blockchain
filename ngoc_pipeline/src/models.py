"""
models.py  –  Task 2.8, 2.11, 3.8 (Ngọc): interface chung cho mô hình, và B0, B3.

Mỗi mô hình là một estimator kiểu sklearn (fit, predict_proba, get_params/set_params), đăng ký trong
REGISTRY và có config configs/models/{tên}.yaml:
  class: b3                       # khoá trong REGISTRY
  params: {cal_splits: 3}         # tham số cố định
  param_grid: {C: [0.1, 1, 10]}   # grid để tune, tên tham số của mô hình
Nhờ vậy mô hình đi thẳng qua make_pipeline (scaler fit trong train) và tune của walkforward.py.

Mô hình đã có:
  lr      B1 Logistic Regression (bản pipeline mỏng; Trâm sở hữu B1)
  b0maj   B0 majority class: p̂ = tỷ lệ ngày tăng trong train. Mức sàn cho log-loss, Brier
  b0pers  B0 persistence: ngày mai cùng hướng hôm nay, p̂_t = 1[close_t > close_{t−1}]. Mức sàn cho DA.
          Không fit nên nằm trong RULES, không qua run_walkforward. p̂ chỉ là 0/1 nên log-loss không dùng được
  b3      B3 SVM RBF, xác suất hiệu chỉnh sigmoid trên điểm ngoài mẫu theo thời gian
B0 không dùng đặc trưng, nên hai cấu hình OHLCV và OHLCV + On-chain cho cùng dự báo (Δ = 0).

python models.py --test  -> chạy test
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC

from data import CLOSE, CONFIG_DIR, DATE, PREDICTION_COLS, TARGET, _raises, load_config, make_synthetic_master
from walkforward import fold_indices, make_folds, run_walkforward


class CalibratedSVC(ClassifierMixin, BaseEstimator):
    """SVM RBF. Lấy điểm quyết định ngoài mẫu qua TimeSeriesSplit(cal_splits, gap) trong cửa sổ train,
    fit sigmoid (Platt) trên các điểm đó, rồi fit SVM cuối trên toàn cửa sổ train.
    Không dùng CalibratedClassifierCV: với TimeSeriesSplit nó chỉ chạy kiểu ensemble, khi đó các SVM
    không học phần dữ liệu gần test nhất."""

    def __init__(self, C=1.0, gamma="scale", cal_splits=3, gap=1):
        self.C, self.gamma, self.cal_splits, self.gap = C, gamma, cal_splits, gap

    def _svc(self):
        return SVC(kernel="rbf", C=self.C, gamma=self.gamma)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        scores, targets = [], []
        for tr, va in TimeSeriesSplit(self.cal_splits, gap=self.gap).split(X):
            scores.append(self._svc().fit(X[tr], y[tr]).decision_function(X[va]))
            targets.append(y[va])
        self.calibrator_ = LogisticRegression(C=1e6).fit(np.concatenate(scores)[:, None], np.concatenate(targets))
        self.svc_ = self._svc().fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X):
        return self.calibrator_.predict_proba(self.svc_.decision_function(np.asarray(X))[:, None])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


def persistence_predictions(data, folds, target=TARGET) -> pd.DataFrame:
    """B0 persistence: p̂_t = y_{t−1} = 1[close_t > close_{t−1}], đã biết khi ngày t kết thúc."""
    if (data[DATE].diff().iloc[1:] != pd.Timedelta(days=1)).any():
        raise ValueError("Dữ liệu thiếu ngày: dòng trước không còn là ngày t−1.")
    y_prev, out = data[target].shift(1), []
    for fold in folds:
        _, te = fold_indices(data, fold, target)
        out.append(pd.DataFrame({"date": data.loc[te, DATE].to_numpy(), "fold": fold.fold_id,
                                 "y_true": data.loc[te, target].astype(int).to_numpy(),
                                 "p_hat": y_prev.loc[te].to_numpy(dtype=float), "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


REGISTRY = {"lr": LogisticRegression, "b0maj": DummyClassifier, "b3": CalibratedSVC}
RULES = {"b0pers": persistence_predictions}     # mô hình theo luật: không fit, không tune


def register(name, cls):
    """Thêm mô hình mới, ví dụ register("rf", RandomForestClassifier), rồi tạo configs/models/rf.yaml."""
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"Tên mô hình '{name}' đã được đăng ký.")
    REGISTRY[name] = cls


def load_model(name):
    """Đọc configs/models/{name}.yaml, trả (estimator, param_grid)."""
    path = CONFIG_DIR / "models" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"Chưa có config cho mô hình '{name}': {path}")
    cfg = yaml.safe_load(path.read_text())
    return REGISTRY[cfg["class"]](**(cfg.get("params") or {})), cfg.get("param_grid") or {}


# ----------------------------------------------------------------------------
# Test:  python models.py --test
# ----------------------------------------------------------------------------
def _tests():
    m = make_synthetic_master(seed=42)
    m[TARGET] = (m[CLOSE].shift(-1) > m[CLOSE]).astype(float).where(m[CLOSE].shift(-1).notna())
    folds = make_folds(m[DATE], **load_config("walkforward.yaml"))

    # 1) Đọc config; tên chưa có config hoặc trùng tên thì báo lỗi
    est, grid = load_model("b3")
    assert isinstance(est, CalibratedSVC) and set(grid) == {"C", "gamma"}
    assert isinstance(load_model("b0maj")[0], DummyClassifier)
    _raises(lambda: load_model("khong_co"), "Chưa có config")
    _raises(lambda: register("lr", SVC), "đã được đăng ký")

    # 2) b0maj: p̂ = tỷ lệ ngày tăng trong cửa sổ train của từng fold
    res = run_walkforward(m, ["cm_AdrActCnt"], load_model("b0maj")[0], folds[:3])
    for f in folds[:3]:
        np.testing.assert_allclose(res.loc[res["fold"] == f.fold_id, "p_hat"], m.loc[fold_indices(m, f)[0], TARGET].mean())

    # 3) b0pers: p̂_t = hướng giá của chính ngày t; đổi giá sau ngày test cuối không đổi dự báo
    res = persistence_predictions(m, folds[:5])
    close = m.set_index(DATE)[CLOSE]
    np.testing.assert_array_equal(res["p_hat"], (close > close.shift(1)).astype(float).loc[res["date"]])
    m2 = m.copy()
    m2.loc[m2[DATE] > res["date"].max() + pd.Timedelta(days=1), CLOSE] *= 3
    m2[TARGET] = (m2[CLOSE].shift(-1) > m2[CLOSE]).astype(float).where(m2[CLOSE].shift(-1).notna())
    pd.testing.assert_frame_equal(persistence_predictions(m2, folds[:5]), res)

    # 4) CalibratedSVC: tương thích sklearn; xác suất hợp lệ và xếp hạng đúng hướng khi có tín hiệu
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + rng.normal(0, 1, 400) > 0).astype(int)
    p = clone(CalibratedSVC(gamma=0.1)).fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and ((p > 0) & (p < 1)).all()
    assert p[X[:, 0] > 1, 1].mean() > 0.6 > 0.4 > p[X[:, 0] < -1, 1].mean()

    # 5) b3 chạy qua walk-forward có tune
    res = run_walkforward(m, ["cm_AdrActCnt", "cm_TxCnt", "cm_HashRate"], est, folds[:2], param_grid=grid)
    assert res["p_hat"].between(0, 1).all() and res["fold"].nunique() == 2
    print("Tất cả test models đều PASS")


if __name__ == "__main__":
    _tests()
