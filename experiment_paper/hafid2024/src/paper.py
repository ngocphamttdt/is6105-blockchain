"""Task 2 (Hafid 2024): chỉ báo kỹ thuật, nhãn, mô hình đúng như bài.

Chạy: python3 src/paper.py --test
features(df) tổng quát cho mọi DataFrame OHLCV (cửa sổ theo số dòng, không giả định tần suất).
Nhãn là float (0/1, NaN ở warm-up / dòng cuối) để dropna; fit_eval ép về int.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

FEATURE_COLS = ["Close", "Volume", "RSI14", "RSI30", "RSI200", "MOM10", "MOM30", "MACD",
                "PROC9", "EMA10", "EMA30", "EMA200", "%K10", "%K30", "%K200", "%D30", "%D200"]


def ema(close, n):
    """EMA_n: alpha = 2/(n+1), khởi tạo bằng giá đầu, chỉ hợp lệ sau n quan sát."""
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close, n):
    """RSI Wilder: RS = avg_gain/avg_loss, avg = EWM alpha=1/n (adjust=False); RSI = 100 - 100/(1+RS)."""
    d = close.diff()
    gain = d.clip(lower=0).where(d.notna())
    loss = (-d).clip(lower=0).where(d.notna())
    ag = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + ag / al)
    out[(al == 0) & (ag > 0)] = 100.0
    out[(al == 0) & (ag == 0)] = 50.0
    return out


def mom(close, n):
    """MOM_n = C_t - C_{t-n}."""
    return close - close.shift(n)


def proc(close, n):
    """PROC_n = 100 * (C_t - C_{t-n}) / C_{t-n}."""
    return 100.0 * (close - close.shift(n)) / close.shift(n)


def stoch_k(df, n):
    """%K_n = 100 * (C_t - LL_n) / (HH_n - LL_n); HH/LL = max high / min low của n dòng gần nhất gồm t."""
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    rng = (hh - ll).replace(0, np.nan)
    return 100.0 * (df["close"] - ll) / rng


def features(df):
    """17 đặc trưng; dòng t chỉ dùng dữ liệu đến hết dòng t. Giữ NaN warm-up (caller dropna)."""
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["Close"] = c
    f["Volume"] = df["volume"]
    for n in (14, 30, 200):
        f[f"RSI{n}"] = rsi(c, n)
    f["MOM10"] = mom(c, 10)
    f["MOM30"] = mom(c, 30)
    f["MACD"] = ema(c, 12) - ema(c, 26)
    f["PROC9"] = proc(c, 9)
    for n in (10, 30, 200):
        f[f"EMA{n}"] = ema(c, n)
    k = {n: stoch_k(df, n) for n in (10, 30, 200)}
    for n in (10, 30, 200):
        f[f"%K{n}"] = k[n]
    for n in (30, 200):
        f[f"%D{n}"] = k[n].rolling(3).mean()
    return f[FEATURE_COLS]


def label_paper(close):
    """Nhãn bài: 1 nếu SMA10_t >= SMA60_t (SMA gồm nến t), ngược lại 0; NaN khi chưa đủ 60 dòng."""
    s10 = close.rolling(10).mean()
    s60 = close.rolling(60).mean()
    y = (s10 >= s60).astype(float)
    return y.where(s60.notna())


def label_next(close):
    """Nhãn dự báo thật: 1[C_{t+1} > C_t]; NaN ở dòng cuối."""
    nxt = close.shift(-1)
    y = (nxt > close).astype(float)
    return y.where(nxt.notna())


def make_xgb():
    return XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, min_child_weight=3,
                         subsample=0.8, colsample_bytree=1.0, gamma=0.1, reg_alpha=0.5,
                         reg_lambda=1, objective="binary:logistic", random_state=0)


def make_lr():
    return LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=100, random_state=0)


def split_80_20(n):
    k = int(n * 0.8)
    return np.arange(0, k), np.arange(k, n)


def fit_eval(X, y, model, scale_on="full"):
    """Standard -> MinMax (cho chi2) -> SelectKBest(chi2, 8) -> model. Model luôn fit trên train, đánh giá trên test.
    scale_on="full": scaler + chi2 fit trên mọi dòng (như bài); "train": chỉ trên train."""
    assert scale_on in ("full", "train")
    d = pd.concat([X, y.rename("__y")], axis=1).dropna()
    Xd, yd = d[list(X.columns)], d["__y"].astype(int).to_numpy()
    tr, te = split_80_20(len(d))
    A = Xd.to_numpy(dtype=float)
    fit_rows = np.arange(len(d)) if scale_on == "full" else tr
    sc = StandardScaler().fit(A[fit_rows])
    mm = MinMaxScaler().fit(sc.transform(A[fit_rows]))
    Z = mm.transform(sc.transform(A))
    sel = SelectKBest(chi2, k=8).fit(Z[fit_rows], yd[fit_rows])
    Zs = sel.transform(Z)
    model.fit(Zs[tr], yd[tr])
    proba = model.predict_proba(Zs[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yt = yd[te]
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    auc = roc_auc_score(yt, proba) if len(set(yt)) == 2 else float("nan")
    pr = yt.mean()
    return {
        "accuracy": accuracy_score(yt, pred),
        "precision": precision_score(yt, pred, zero_division=0),
        "recall": recall_score(yt, pred, zero_division=0),
        "f1": f1_score(yt, pred, zero_division=0),
        "auc": auc,
        "confusion": cm.tolist(),  # [[tn, fp],[fn, tp]]
        "test_pos_rate": float(pr),
        "majority_baseline": float(max(pr, 1 - pr)),
        "selected_features": list(Xd.columns[sel.get_support()]),
        "n_train": len(tr),
        "n_test": len(te),
        "fitted": {"scaler": sc, "minmax": mm, "selector": sel},
    }


def _tests():
    close = pd.Series([1.0, 2, 3, 2, 4, 5, 3])
    # EMA span 3 (alpha .5): 1, 1.5, 2.25, 2.125, 3.0625 ; hợp lệ từ quan sát thứ 3
    e = ema(close, 3)
    assert e.iloc[:2].isna().all()
    assert np.allclose(e.iloc[2:5], [2.25, 2.125, 3.0625])
    # RSI n=2 (alpha .5): delta nan,1,1,-1 -> gain avg [1,1,.5], loss avg [0,0,.5]
    r = rsi(close, 2)
    assert r.iloc[:2].isna().all()
    assert np.isclose(r.iloc[2], 100.0) and np.isclose(r.iloc[3], 50.0)
    # dòng 4: delta=+2 -> ag=(.5+2)/2=1.25 ... ag3=.5, al3=.5 ; ag4=1.25, al4=.25 -> RS=5 -> 100-100/6
    assert np.isclose(r.iloc[4], 100 - 100 / 6)
    # MOM, PROC
    assert np.isclose(mom(close, 2).iloc[4], 4 - 3) and mom(close, 2).iloc[:2].isna().all()
    assert np.isclose(proc(close, 2).iloc[4], 100 * (4 - 3) / 3)
    assert np.isclose(proc(close, 3).iloc[6], 100 * (3 - 2) / 2)
    # %K n=3
    df = pd.DataFrame({"high": [2, 4, 3, 6, 5.0], "low": [1, 2, 1, 3, 2.0], "close": [1.5, 3, 2, 5, 4.0]})
    k = stoch_k(df, 3)
    assert k.iloc[:2].isna().all()
    assert np.isclose(k.iloc[2], 100 * (2 - 1) / (4 - 1))   # HH=4 LL=1
    assert np.isclose(k.iloc[3], 100 * (5 - 1) / (6 - 1))   # HH=6 LL=1
    assert np.isclose(k.iloc[4], 100 * (4 - 1) / (6 - 1))   # low[2:5]=1,3,2 -> LL=1; HH=6
    assert np.isclose(stoch_k(df, 3).rolling(3).mean().iloc[4], np.mean(k.iloc[2:5]))  # %D = SMA3(%K)
    # nhãn
    c = pd.Series(np.arange(1.0, 101.0))
    lp = label_paper(c)
    assert lp.iloc[:59].isna().all() and (lp.iloc[59:] == 1).all()  # tăng đều: SMA10 > SMA60
    assert (label_paper(c[::-1].reset_index(drop=True)).iloc[59:] == 0).all()
    ln = label_next(pd.Series([1.0, 3, 2, 2, 5]))
    assert ln.iloc[:4].tolist() == [1, 0, 0, 1] and np.isnan(ln.iloc[4])
    # split
    tr, te = split_80_20(100)
    assert tr[0] == 0 and tr[-1] == 79 and te[0] == 80 and te[-1] == 99
    # synthetic OHLCV
    rng = np.random.default_rng(0)
    n = 600
    cl = 100 + np.cumsum(rng.normal(0, 1, n))
    ohlc = pd.DataFrame({"open": cl, "high": cl + rng.random(n), "low": cl - rng.random(n),
                         "close": cl, "volume": rng.random(n) * 10 + 1},
                        index=pd.date_range("2021-01-01", periods=n, freq="15min"))
    F = features(ohlc)
    assert list(F.columns) == FEATURE_COLS and len(F.columns) == 17
    # không nhìn tương lai
    t = 400
    ohlc2 = ohlc.copy()
    ohlc2.iloc[t + 1:, :] = ohlc2.iloc[t + 1:, :] * 7 + 3
    F2 = features(ohlc2)
    pd.testing.assert_frame_equal(F.iloc[:t + 1], F2.iloc[:t + 1])
    pd.testing.assert_series_equal(label_paper(ohlc.close).iloc[:t + 1], label_paper(ohlc2.close).iloc[:t + 1])
    # fit_eval
    y = label_paper(ohlc.close)
    res = fit_eval(F, y, make_lr(), "full")
    assert len(res["selected_features"]) == 8
    assert sum(map(sum, res["confusion"])) == res["n_test"]
    # train-only không đọc test
    d = pd.concat([F, y.rename("y")], axis=1).dropna()
    Xd, yd = d[FEATURE_COLS], d["y"]
    Xb = Xd.copy()
    Xb.iloc[int(len(Xb) * 0.8):] *= 1000
    a = fit_eval(Xd, yd, make_lr(), "train")["fitted"]
    b = fit_eval(Xb, yd, make_lr(), "train")["fitted"]
    assert np.array_equal(a["scaler"].mean_, b["scaler"].mean_)
    assert np.array_equal(a["scaler"].scale_, b["scaler"].scale_)
    assert np.array_equal(a["minmax"].data_min_, b["minmax"].data_min_)
    assert np.array_equal(a["selector"].scores_, b["selector"].scores_)
    c_ = fit_eval(Xb, yd, make_lr(), "full")["fitted"]
    assert not np.array_equal(a["scaler"].mean_, c_["scaler"].mean_)
    # LR có random_state: hai lần chạy giống hệt nhau (hệ số, xác suất, chỉ số)
    m1, m2 = make_lr(), make_lr()
    r1 = fit_eval(F, y, m1, "train")
    r2 = fit_eval(F, y, m2, "train")
    assert np.array_equal(m1.coef_, m2.coef_) and np.array_equal(m1.intercept_, m2.intercept_)
    for k_ in ("accuracy", "precision", "recall", "f1", "auc", "confusion"):
        assert r1[k_] == r2[k_], k_
    Z_ = np.random.default_rng(1).random((5, 8))
    assert np.array_equal(m1.predict_proba(Z_), m2.predict_proba(Z_))
    # xgb nhanh trên dữ liệu nhỏ
    rx = fit_eval(F, y, make_xgb(), "train")
    assert 0 <= rx["accuracy"] <= 1
    # smoke trên dữ liệu thật
    try:
        from download import load_15m
        df = load_15m().iloc[:2000]
    except Exception as e:  # chưa có dữ liệu thì bỏ qua
        print("bỏ qua smoke:", e)
    else:
        Fr = features(df)
        assert Fr.shape == (2000, 17) and Fr.iloc[250:].notna().all().all()
        rr = fit_eval(Fr, label_paper(df.close), make_lr(), "full")
        assert 0 <= rr["accuracy"] <= 1
    print("Tất cả test paper đều PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    if ap.parse_args().test:
        _tests()
