"""
models.py  –  Task 2.8, 2.11, 3.8 (Ngọc): interface chung cho mô hình, và B0, B3.

Mỗi mô hình là một estimator kiểu sklearn (fit, predict_proba, get_params/set_params), đăng ký trong
REGISTRY và có config configs/models/{tên}.yaml:
  class: svm_rbf                  # thuật toán: khoá trong REGISTRY (tên file là mã mô hình, vd. b3)
  params: {cal_splits: 3}         # tham số cố định
  param_grid: {C: [0.1, 1, 10]}   # grid để tune, tên tham số của mô hình
Nhờ vậy mô hình đi thẳng qua make_pipeline (scaler fit trong train) và tune của walkforward.py.

Mô hình đã có (mã = tên file config; hậu tố t = đã tune lại):
  b1      B1 Logistic Regression (bản pipeline mỏng; Trâm sở hữu B1). Tên cũ: lr
  b0maj   B0 majority class: p̂ = tỷ lệ ngày tăng trong train. Mức sàn cho log-loss, Brier
  b0pers  B0 persistence: ngày mai cùng hướng hôm nay, p̂_t = 1[close_t > close_{t−1}]. Mức sàn cho DA.
          Không fit nên nằm trong RULES, không qua run_walkforward. p̂ chỉ là 0/1 nên log-loss không dùng được
  b0rev   B0 đảo chiều: ngày mai ngược hướng hôm nay, p̂_t = 1 − 1[close_t > close_{t−1}]. Chiều chọn lại trên
          từng cửa sổ train: nếu persistence đúng ≥ 50% trong train thì dùng persistence (baseline_check, đề xuất 1).
          Mọi mô hình có học phải được so với luật này. Cũng nằm trong RULES
  b1fs    B1 có chọn đặc trưng tiến dần (tên cũ: lrfs): bắt đầu từ cột đầu tiên (logret_lag1), mỗi bước thêm cột làm log-loss
          inner CV (TimeSeriesSplit trong cửa sổ train) giảm nhiều nhất, dừng khi không giảm quá tol
          (baseline_check, đề xuất 2)
  b3      B3 SVM RBF, xác suất hiệu chỉnh sigmoid trên điểm ngoài mẫu theo thời gian
  b2      B2 Random Forest (bản chạy thử; Trâm sở hữu B2)
  b2t     B2 Random Forest đã tuning (configs/models/b2t.yaml; chi tiết analyses/b2_tuning/README.md)
  b4      B4 XGBoost, early stopping trên 20% cuối cửa sổ train (bản chạy thử; Khoa sở hữu B4)
  b1t     B1 tune đầy đủ (tên cũ: lrt / lr_tuned), chạy bằng b1t_tuning.py, không qua load_model
B0 không dùng đặc trưng, nên hai cấu hình OHLCV và OHLCV + On-chain cho cùng dự báo (Δ = 0).

python models.py --test  -> chạy test
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


class EarlyStoppingXGB(ClassifierMixin, BaseEstimator):
    """XGBoost. Chọn số cây bằng early stopping (log-loss) trên val_frac cuối cửa sổ train, bỏ gap dòng
    giữa hai phần (cùng lý do gap của walk-forward), rồi fit lại trên toàn cửa sổ với số cây đã chọn.
    Import xgboost lúc fit: máy chưa cài xgboost vẫn chạy được các mô hình khác."""

    def __init__(self, learning_rate=0.05, max_depth=3, n_estimators=1000, early_stopping_rounds=50,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1, val_frac=0.2, gap=1,
                 n_jobs=4, random_state=0):
        self.learning_rate, self.max_depth, self.n_estimators = learning_rate, max_depth, n_estimators
        self.early_stopping_rounds, self.subsample, self.colsample_bytree = early_stopping_rounds, subsample, colsample_bytree
        self.min_child_weight, self.val_frac, self.gap = min_child_weight, val_frac, gap
        self.n_jobs, self.random_state = n_jobs, random_state

    def _xgb(self, **kw):
        from xgboost import XGBClassifier
        return XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,
                             subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                             min_child_weight=self.min_child_weight, eval_metric="logloss",
                             n_jobs=self.n_jobs, random_state=self.random_state, **kw)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n_val = int(len(X) * self.val_frac)
        cut = len(X) - n_val
        probe = self._xgb(n_estimators=self.n_estimators, early_stopping_rounds=self.early_stopping_rounds)
        probe.fit(X[:cut - self.gap], y[:cut - self.gap], eval_set=[(X[cut:], y[cut:])], verbose=False)
        self.best_n_estimators_ = probe.best_iteration + 1
        self.model_ = self._xgb(n_estimators=self.best_n_estimators_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


class ForwardSelectLR(ClassifierMixin, BaseEstimator):
    """Logistic Regression chọn đặc trưng tiến dần. Luôn giữ `start` cột đầu; mỗi bước thử thêm từng cột còn lại,
    lấy cột cho log-loss trung bình trên TimeSeriesSplit(cv_splits, gap) thấp nhất, dừng khi không giảm quá tol.
    Mỗi split có scaler riêng, fit trên phần train của split. Chỉ dùng dữ liệu được fit (cửa sổ train của fold)."""

    def __init__(self, C=1.0, start=1, cv_splits=5, gap=1, tol=1e-4, max_iter=1000):
        self.C, self.start, self.cv_splits, self.gap, self.tol, self.max_iter = C, start, cv_splits, gap, tol, max_iter

    def _lr(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter))

    def _cv_loss(self, X, y, cols):
        losses = [log_loss(y[va], self._lr().fit(X[tr][:, cols], y[tr]).predict_proba(X[va][:, cols])[:, 1],
                           labels=[0, 1])
                  for tr, va in TimeSeriesSplit(self.cv_splits, gap=self.gap).split(X)]
        return float(np.mean(losses))

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        selected = list(range(self.start))
        best = self._cv_loss(X, y, selected)
        while len(selected) < X.shape[1]:
            trials = {j: self._cv_loss(X, y, selected + [j]) for j in range(X.shape[1]) if j not in selected}
            j, loss = min(trials.items(), key=lambda kv: kv[1])
            if loss > best - self.tol:
                break
            selected.append(j)
            best = loss
        self.support_ = np.array(selected)
        self.cv_loss_ = best
        self.model_ = self._lr().fit(X[:, self.support_], y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X)[:, self.support_])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


def _direction_rule(data, folds, target, choose_flip) -> pd.DataFrame:
    """Luật theo hướng giá hôm nay: p̂_t = y_{t−1} = 1[close_t > close_{t−1}], đã biết khi ngày t kết thúc.
    choose_flip(train_y, train_y_prev) -> True thì đảo thành 1 − y_{t−1} cho cả fold."""
    if (data[DATE].diff().iloc[1:] != pd.Timedelta(days=1)).any():
        raise ValueError("Dữ liệu thiếu ngày: dòng trước không còn là ngày t−1.")
    y_prev, out = data[target].shift(1), []
    for fold in folds:
        tr, te = fold_indices(data, fold, target)
        ok = tr[y_prev.loc[tr].notna()]
        p = y_prev.loc[te].to_numpy(dtype=float)
        if choose_flip(data.loc[ok, target], y_prev.loc[ok]):
            p = 1.0 - p
        out.append(pd.DataFrame({"date": data.loc[te, DATE].to_numpy(), "fold": fold.fold_id,
                                 "y_true": data.loc[te, target].astype(int).to_numpy(),
                                 "p_hat": p, "regime": pd.NA}))
    return pd.concat(out, ignore_index=True)[PREDICTION_COLS]


def persistence_predictions(data, folds, target=TARGET) -> pd.DataFrame:
    """B0 persistence: p̂_t = y_{t−1} = 1[close_t > close_{t−1}], đã biết khi ngày t kết thúc."""
    return _direction_rule(data, folds, target, lambda y, y_prev: False)


def reversal_predictions(data, folds, target=TARGET, flip_below=0.5) -> pd.DataFrame:
    """B0 đảo chiều: p̂_t = 1 − y_{t−1} nếu trong cửa sổ train persistence đúng < flip_below, ngược lại p̂_t = y_{t−1}.
    Nhãn của ngày train cuối dùng close của ngày gap, nên việc chọn chiều không chạm ngày test nào."""
    return _direction_rule(data, folds, target, lambda y, y_prev: (y == y_prev).mean() < flip_below)


# Khoá = thuật toán (trường class của config), không phải mã mô hình: b2 và b2t cùng class random_forest
REGISTRY = {"logreg": LogisticRegression, "majority": DummyClassifier, "svm_rbf": CalibratedSVC,
            "random_forest": RandomForestClassifier, "xgboost": EarlyStoppingXGB, "logreg_fs": ForwardSelectLR}
RULES = {"b0pers": persistence_predictions,     # mô hình theo luật: không fit, không tune
         "b0rev": reversal_predictions}         # tham số trong configs/models/{b0pers,b0rev}.yaml
RULE_FUNCS = {"persistence": persistence_predictions, "reversal": reversal_predictions}


def run_rule(name, data, folds, target=TARGET) -> pd.DataFrame:
    """Chạy luật B0 theo configs/models/{name}.yaml (class: rule, rule: <tên luật>, params: tham số của luật)."""
    cfg = yaml.safe_load((CONFIG_DIR / "models" / f"{name}.yaml").read_text())
    if cfg.get("class") != "rule":
        raise ValueError(f"'{name}' không phải luật (class: {cfg.get('class')}); dùng load_model.")
    return RULE_FUNCS[cfg["rule"]](data, folds, target, **(cfg.get("params") or {}))


def register(name, cls):
    """Thêm thuật toán mới, ví dụ register("extra_trees", ExtraTreesClassifier), rồi tạo configs/models/<mã>.yaml với class: extra_trees."""
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"Tên mô hình '{name}' đã được đăng ký.")
    REGISTRY[name] = cls


def load_model(name):
    """Đọc configs/models/{name}.yaml, trả (estimator, param_grid)."""
    path = CONFIG_DIR / "models" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"Chưa có config cho mô hình '{name}': {path}")
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("class") not in REGISTRY:
        raise ValueError(f"'{name}' (class: {cfg.get('class')}) không chạy qua load_model: luật B0 dùng run_rule, "
                         f"b1t dùng b1t_tuning.py.")
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
    _raises(lambda: register("logreg", SVC), "đã được đăng ký")

    # 1b) Quy ước tên: mọi config có baseline (B0–B5) và tuning; hậu tố "t" <=> đã tune lại (retuned)
    for path in sorted((CONFIG_DIR / "models").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert cfg.get("baseline") in {f"B{i}" for i in range(6)}, f"{path.name}: thiếu/sai baseline"
        assert cfg.get("tuning") in {"none", "original_grid", "retuned"}, f"{path.name}: thiếu/sai tuning"
        assert path.stem.endswith("t") == (cfg["tuning"] == "retuned"), f"{path.name}: hậu tố t không khớp tuning"

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

    # 3b) b0rev: mỗi fold là đúng persistence hoặc đúng 1 − persistence, theo persistence trong cửa sổ train;
    #     đổi giá sau ngày test cuối không đổi dự báo
    rev = reversal_predictions(m, folds[:5])
    for f in folds[:5]:
        tr, _ = fold_indices(m, f)
        y_tr, y_prev = m.loc[tr, TARGET], m[TARGET].shift(1).loc[tr]
        flip = (y_tr == y_prev).mean() < 0.5
        k = res["fold"] == f.fold_id
        np.testing.assert_array_equal(rev.loc[k, "p_hat"], 1 - res.loc[k, "p_hat"] if flip else res.loc[k, "p_hat"])
    pd.testing.assert_frame_equal(reversal_predictions(m2, folds[:5]), rev)
    #     Đọc từ config cho đúng kết quả gọi hàm trực tiếp; config luật không chạy qua load_model
    pd.testing.assert_frame_equal(run_rule("b0rev", m, folds[:5]), rev)
    pd.testing.assert_frame_equal(run_rule("b0pers", m, folds[:5]), res)
    _raises(lambda: load_model("b0rev"), "không chạy qua load_model")
    _raises(lambda: load_model("b1t"), "không chạy qua load_model")
    _raises(lambda: run_rule("b1", m, folds[:5]), "không phải luật")
    #     Chuỗi đảo chiều hoàn hảo trong train thì luật lật chiều và đúng 100% trên test
    alt = m.copy()
    alt[CLOSE] = 100 * np.where(np.arange(len(alt)) % 2, 1.1, 1.0)
    alt[TARGET] = (alt[CLOSE].shift(-1) > alt[CLOSE]).astype(float).where(alt[CLOSE].shift(-1).notna())
    r = reversal_predictions(alt, folds[:2])
    assert ((r["p_hat"] > 0.5) == r["y_true"]).all()

    # 3c) b1fs: luôn giữ cột đầu; bắt được cột có tín hiệu; bỏ cột nhiễu; tune được C qua walk-forward
    rng = np.random.default_rng(1)
    Xs = rng.normal(size=(700, 6))
    ys = (0.8 * Xs[:, 3] + rng.normal(0, 1, 700) > 0).astype(int)
    fs = ForwardSelectLR().fit(Xs, ys)
    assert fs.support_[0] == 0 and 3 in fs.support_ and len(fs.support_) <= 3
    assert fs.predict_proba(Xs).shape == (700, 2)
    est, grid = load_model("b1fs")
    assert isinstance(est, ForwardSelectLR) and set(grid) == {"C"}
    res_fs = run_walkforward(m, ["cm_AdrActCnt", "cm_TxCnt", "cm_HashRate"], est, folds[:2], param_grid={"C": [0.1, 1]})
    assert res_fs["p_hat"].between(0, 1).all() and res_fs["fold"].nunique() == 2

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

    # 6) b2 đọc đúng config và chạy qua walk-forward
    est, grid = load_model("b2")
    assert isinstance(est, RandomForestClassifier) and set(grid) == {"n_estimators", "max_depth", "min_samples_leaf"}
    res = run_walkforward(m, ["cm_AdrActCnt", "cm_TxCnt"], est, folds[:1], param_grid={"n_estimators": [50]})
    assert res["p_hat"].between(0, 1).all()
    #    b2t (B2 đã tuning): cùng lớp RandomForestClassifier, grid lá lớn + max_features + max_samples
    est, grid = load_model("b2t")
    assert isinstance(est, RandomForestClassifier) and set(grid) == {"min_samples_leaf", "max_features", "max_samples"}
    res = run_walkforward(m, ["cm_AdrActCnt", "cm_TxCnt"], est.set_params(n_estimators=50), folds[:1],
                          param_grid={"min_samples_leaf": [50], "max_features": ["sqrt"], "max_samples": [None, 0.5]})
    assert res["p_hat"].between(0, 1).all()

    # 7) b4 (nếu cài được xgboost): xác suất hợp lệ, xếp hạng đúng hướng; số cây chọn bằng early stopping
    try:
        import xgboost  # noqa: F401
    except Exception as e:
        print(f"  Bỏ qua test b4: không import được xgboost ({type(e).__name__})")
    else:
        est, grid = load_model("b4")
        assert isinstance(est, EarlyStoppingXGB) and set(grid) == {"learning_rate", "max_depth"}
        fitted = clone(est).set_params(n_estimators=300).fit(X, y)
        p = fitted.predict_proba(X)
        assert np.allclose(p.sum(axis=1), 1) and 1 <= fitted.best_n_estimators_ <= 300
        assert p[X[:, 0] > 1, 1].mean() > 0.6 > 0.4 > p[X[:, 0] < -1, 1].mean()
    print("Tất cả test models đều PASS")


if __name__ == "__main__":
    _tests()
