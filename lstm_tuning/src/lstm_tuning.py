"""
lstm_tuning.py  –  Tuning B5 LSTM (góp ý 3 của thầy): ghi rõ tuning đã chạy, chọn tham số trên validation,
kết quả trước/sau tuning. Script độc lập: không sửa và không gọi ngan_lstm_hmm_analysis/model_lstm.py
hay ngoc_pipeline. Dùng lại dữ liệu, cách chia, metric và luật B0 của baseline_check để số liệu so được.

Đặc trưng: 10 đặc trưng OHLCV (mặc định) hoặc chỉ logret_lag1 (--features lag1, như LR 1 đặc trưng của
baseline_check). Data freeze v1.0.

Khác với model_lstm.py hiện tại (cùng kiến trúc LSTM 1 lớp + Linear):
  - Validation nội bộ cho early stopping (20% cuối phần train) có gap 1 ngày.
  - Batch được xáo trộn trong phần train (mỗi chuỗi lookback ngày đã độc lập; không rò rỉ).
  - Có dropout trước lớp Linear.
  - Sau early stopping, học lại từ đầu trên 100% phần train với số epoch tốt nhất.

Bước A (chia cố định, như baseline_check): grid trên train 2018–2021 → chọn theo log-loss validation
2022–2023 (p̂ trung bình 3 seed) → học lại trên train + validation → test 2024–08/2026 đúng 1 lần.
Bước B (walk-forward 82 fold, lịch như ngoc_pipeline): chạy cấu hình mặc định (trước tune) và cấu hình
đã chọn ở Bước A (sau tune), cùng code, để so trước/sau.

Chạy từ thư mục gốc repo:
    python lstm_tuning/src/lstm_tuning.py            # đầy đủ (khoảng 30–60 phút trên CPU)
    python lstm_tuning/src/lstm_tuning.py --quick    # chạy thử nhanh, không ghi báo cáo
    python lstm_tuning/src/lstm_tuning.py --seeds 42 43 44 45 46   # đổi danh sách seed lấy trung bình
"""
import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline_check/src"))
sys.path.insert(0, str(ROOT / "tram_features_backtest/src"))
import baseline_check as bc                                              # noqa: E402
from features_ohlcv import FEATURES, MASTER, build_ohlcv_features       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
FEAT_DESC = "10 đặc trưng OHLCV"
COLS = list(FEATURES)                    # đặt lại trong main() theo --features
FEATURE_SETS = {"all": (list(FEATURES), "10 đặc trưng OHLCV", ""), "lag1": (["logret_lag1"], "chỉ logret_lag1", "lag1")}
# Số seed. Kết quả chính dùng p̂ trung bình 3 seed (như configs/models/b5.yaml): trung bình nhiều seed làm dự báo
# ổn định hơn nhưng không làm nó cao hơn. Muốn biết kết quả có phải do may rủi của seed không thì dùng
# seed_check.py (10 seed, mỗi seed một mô hình riêng). KHÔNG chọn seed theo kết quả: đó là chọn mô hình theo test.
SEEDS = [42, 43, 44]
SEED_NOTE = ("Số seed: kết quả chính lấy p̂ trung bình {n} seed để giảm dao động do khởi tạo ngẫu nhiên; độ nhạy theo "
             "seed được kiểm riêng bằng `seed_check.py` (10 seed, mỗi seed một mô hình, `seed_check.md`). Seed được cố "
             "định trước khi chạy, không chọn seed theo kết quả.")
DEFAULT = {"lookback": 30, "hidden": 32, "dropout": 0.0, "lr": 1e-3}     # cấu hình B5 hiện tại = trước tune
GRID = {"lookback": [10, 30], "hidden": [8, 32], "dropout": [0.0, 0.2], "lr": [1e-3, 1e-4]}
FIXED = {"batch_size": 32, "max_epochs": 50, "patience": 5, "val_frac": 0.2, "gap": 1}

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# ---------------------------------------------------------------------------
# Dữ liệu: chuỗi lookback ngày kết thúc ở ngày t, nhãn y_t = 1[C_{t+1} > C_t]
# ---------------------------------------------------------------------------
def load_full():
    """Đặc trưng cho mọi ngày có giá (kể cả trước 31/12/2017, để chuỗi đầu tiên có đủ lịch sử)."""
    master = pd.read_csv(MASTER, parse_dates=["date"])
    feats = build_ohlcv_features(master).set_index("date")[FEATURES]
    return feats[np.isfinite(feats).all(axis=1)]


def make_sequences(feats_scaled: np.ndarray, pos: np.ndarray, lookback: int) -> np.ndarray:
    """Chuỗi cho từng vị trí trong pos: các dòng pos-lookback+1 … pos (chỉ dữ liệu đến hết ngày t)."""
    idx = pos[:, None] + np.arange(-lookback + 1, 1)[None, :]
    assert idx.min() >= 0, "Thiếu lịch sử cho chuỗi đầu tiên"
    return feats_scaled[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


def _train(X, y, cfg, seed, epochs=None, X_val=None, y_val=None):
    """Học LSTM. Có X_val: early stopping theo log-loss validation, trả (model, best_epoch).
       Không có X_val: học đúng `epochs` epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = LSTMClassifier(X.shape[2], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    n_epochs = FIXED["max_epochs"] if epochs is None else epochs
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), FIXED["batch_size"]):
            b = perm[i:i + FIXED["batch_size"]]
            opt.zero_grad()
            crit(model(Xt[b]), yt[b]).backward()
            opt.step()
        if X_val is None:
            continue
        model.eval()
        with torch.no_grad():
            v = crit(model(torch.from_numpy(X_val)), torch.from_numpy(y_val)).item()
        if v < best - 1e-6:
            best, best_state, best_epoch, wait = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= FIXED["patience"]:
                break
    if X_val is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def fit_predict(feats, train_dates, target_sets, cfg, seed):
    """Học trên train_dates, trả p̂ cho từng tập ngày trong target_sets (dict tên → DatetimeIndex).
       1) scaler fit trên các ngày train; 2) early stopping trên 20% cuối train, bỏ 1 ngày gap;
       3) học lại từ đầu trên 100% train với số epoch tốt nhất."""
    scaler = StandardScaler().fit(feats.loc[train_dates, COLS].to_numpy())
    Z = scaler.transform(feats[COLS].to_numpy()).astype(np.float32)
    y_all = feats["_y"].to_numpy(np.float32)
    pos_tr = feats.index.get_indexer(train_dates)
    X_tr = make_sequences(Z, pos_tr, cfg["lookback"])
    y_tr = y_all[pos_tr]
    cut = int(len(X_tr) * (1 - FIXED["val_frac"]))
    fit_end = cut - FIXED["gap"]                       # nhãn ngày fit cuối dùng giá ngày val đầu → bỏ
    _, best_epoch = _train(X_tr[:fit_end], y_tr[:fit_end], cfg, seed, X_val=X_tr[cut:], y_val=y_tr[cut:])
    model, _ = _train(X_tr, y_tr, cfg, seed, epochs=max(best_epoch, 1))
    model.eval()
    out = {}
    with torch.no_grad():
        for name, dates in target_sets.items():
            X = make_sequences(Z, feats.index.get_indexer(dates), cfg["lookback"])
            out[name] = torch.sigmoid(model(torch.from_numpy(X))).numpy().astype(float)
    return out, best_epoch


def ensemble(feats, train_dates, target_sets, cfg, seeds):
    """p̂ trung bình các seed (như b5_{config}.csv của ngoc_pipeline)."""
    per_seed, epochs = [], []
    for s in seeds:
        p, e = fit_predict(feats, train_dates, target_sets, cfg, s)
        per_seed.append(p)
        epochs.append(e)
    mean = {k: np.mean([p[k] for p in per_seed], axis=0) for k in target_sets}
    return mean, per_seed, epochs


def cfg_name(c):
    return f"lookback={c['lookback']}, hidden={c['hidden']}, dropout={c['dropout']}, lr={c['lr']:g}"


# ---------------------------------------------------------------------------
# Bước A: grid trên train/validation, test 1 lần
# ---------------------------------------------------------------------------
def step_a(feats, parts, grid, seeds):
    tr, va, te = parts["train"].index, parts["val"].index, parts["test"].index
    y_tr, y_va = parts["train"]["y"].to_numpy(int), parts["val"]["y"].to_numpy(int)
    rows = []
    for cfg in grid:
        t0 = time.time()
        mean, per_seed, epochs = ensemble(feats, tr, {"train": tr, "val": va}, cfg, seeds)
        seed_ll = [log_loss(y_va, p["val"], labels=[0, 1]) for p in per_seed]
        rows.append({**cfg, "trước tune": cfg == DEFAULT,
                     "train_log_loss": log_loss(y_tr, mean["train"], labels=[0, 1]),
                     "val_log_loss": log_loss(y_va, mean["val"], labels=[0, 1]),
                     "val_log_loss_sd_seed": float(np.std(seed_ll)),
                     "train_DA": ((mean["train"] > .5) == y_tr).mean(),
                     "val_DA": ((mean["val"] > .5) == y_va).mean(),
                     "val_pct_pred_up": (mean["val"] > .5).mean(),
                     "best_epoch_seeds": "/".join(map(str, epochs))})
        print(f"  {cfg_name(cfg):<55} val_log_loss={rows[-1]['val_log_loss']:.4f} "
              f"val_DA={rows[-1]['val_DA']:.3f} epochs={rows[-1]['best_epoch_seeds']} ({time.time() - t0:.0f}s)")
    t = pd.DataFrame(rows)
    # Chọn theo val log-loss làm tròn 4 chữ số; hoà thì chọn mô hình đơn giản hơn (hidden nhỏ, lookback ngắn)
    t["_rank"] = t["val_log_loss"].round(4)
    best_row = t.sort_values(["_rank", "hidden", "lookback", "dropout", "lr"],
                             ascending=[True, True, True, False, False]).iloc[0]
    best = {k: (int(best_row[k]) if k in ("lookback", "hidden") else float(best_row[k])) for k in DEFAULT}
    t = t.drop(columns="_rank")

    # Test đúng 1 lần cho cấu hình đã chọn (và cấu hình mặc định đặt trước, không chọn theo test)
    trva = parts["train"].index.append(parts["val"].index)
    test_p = {}
    for label, cfg in [("B5 LSTM trước tune (" + cfg_name(DEFAULT) + ")", DEFAULT),
                       ("B5 LSTM sau tune (" + cfg_name(best) + ")", best)]:
        mean, _, _ = ensemble(feats, trva, {"test": te}, cfg, seeds)
        test_p[label] = mean["test"]
    return t, best, test_p


# ---------------------------------------------------------------------------
# Bước B: walk-forward 82 fold, cùng lịch baseline_check / ngoc_pipeline
# ---------------------------------------------------------------------------
def wf_schedule(df):
    test_days = df.loc[bc.WF_TEST[0]:bc.WF_TEST[1]].index
    pos = df.index.get_indexer(test_days)
    for start in range(0, len(test_days), 30):
        te_idx = pos[start:start + 30]
        tr_idx = np.arange(te_idx[0] - 1 - 730, te_idx[0] - 1)
        assert tr_idx[0] >= 0
        yield df.index[tr_idx], df.index[te_idx]


def step_b(feats, df, cfgs, seeds, max_folds=None):
    test_days = df.loc[bc.WF_TEST[0]:bc.WF_TEST[1]].index
    preds = {label: pd.Series(np.nan, index=test_days) for label in cfgs}
    epochs = {label: [] for label in cfgs}
    for k, (tr, te) in enumerate(wf_schedule(df)):
        if max_folds is not None and k >= max_folds:
            break
        for label, cfg in cfgs.items():
            mean, _, ep = ensemble(feats, tr, {"test": te}, cfg, seeds)
            preds[label].loc[te] = mean["test"]
            epochs[label].append(ep)
        if k % 10 == 0:
            print(f"  fold {k} xong ({te[0].date()} → {te[-1].date()})")
    return preds, epochs, test_days


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="all",
                    help="all: 10 đặc trưng OHLCV (ghi vào reports/); lag1: chỉ logret_lag1 (ghi vào reports/lag1/)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help=f"các seed để lấy trung bình p̂ (mặc định {' '.join(map(str, SEEDS))})")
    ap.add_argument("--quick", action="store_true", help="2 cấu hình, 1 seed, 3 fold; không ghi báo cáo")
    args = ap.parse_args()
    global COLS, OUT, FEAT_DESC
    COLS, FEAT_DESC, sub = FEATURE_SETS[args.features]
    OUT = OUT / sub if sub else OUT
    print(f"Đặc trưng: {FEAT_DESC} → {OUT}")

    df = bc.load_data()
    parts = bc.split(df)
    feats = load_full()
    feats["_y"] = df["y"].reindex(feats.index)          # nhãn đi kèm, chỉ đọc ở các ngày train/target
    for name, p in parts.items():                        # mọi ngày cần dùng đều có đặc trưng và nhãn
        assert p.index.isin(feats.index).all() and p["y"].notna().all(), name

    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    seeds = list(args.seeds)
    if args.quick:
        grid, seeds = [DEFAULT, {"lookback": 10, "hidden": 8, "dropout": 0.2, "lr": 1e-3}], seeds[:1]
    assert DEFAULT in grid

    t0 = time.time()
    print(f"Bước A: {len(grid)} cấu hình × {len(seeds)} seed (train 2018–2021 → validation 2022–2023)")
    grid_t, best, test_p = step_a(feats, parts, grid, seeds)
    print(f"  chọn: {cfg_name(best)}")

    te = parts["test"]
    rules, rinfo = bc.rule_predictions(parts["train"].pipe(lambda x: pd.concat([x, parts["val"]])), te)
    test_rows = {k: bc.metrics(te["y"], v) for k, v in rules.items()
                 if k in ("B0 Majority (p̂ = tỷ lệ tăng của train)", "B0 Đảo chiều (mai ngược chiều hôm nay)")}
    test_rows.update({k: bc.metrics(te["y"], v) for k, v in test_p.items()})
    test_t = bc.fmt_metrics(test_rows)
    print(test_t.to_string())

    print("Bước B: walk-forward 82 fold, trước tune và sau tune")
    wf_cfgs = {"B5 LSTM trước tune": DEFAULT, "B5 LSTM sau tune": best}
    if best == DEFAULT:
        wf_cfgs = {"B5 LSTM trước tune = sau tune": DEFAULT}
    wf_p, wf_epochs, test_days = step_b(feats, df, wf_cfgs, seeds, max_folds=3 if args.quick else None)
    rule_wf, _ = bc.walk_forward(df, {})
    wf_rows = {}
    for k in ("Majority", "Đảo chiều"):
        wf_rows[k] = bc.metrics(df.loc[test_days, "y"], rule_wf[k])
    for k, p in wf_p.items():
        ok = p.notna()
        wf_rows[k] = bc.metrics(df.loc[test_days[ok], "y"], p[ok])
    wf_t = bc.fmt_metrics(wf_rows)
    print(wf_t.to_string())
    print(f"Tổng thời gian: {(time.time() - t0) / 60:.1f} phút")
    if args.quick:
        return

    # Theo năm (walk-forward): DA của LSTM sau tune so với luật đảo chiều
    yrs = pd.DataFrame({"y": df.loc[test_days, "y"].to_numpy(),
                        "Đảo chiều": (rule_wf["Đảo chiều"].to_numpy() > .5),
                        **{k: (v.to_numpy() > .5) for k, v in wf_p.items()}}, index=test_days)
    by_year = pd.DataFrame({k: (yrs[k] == yrs["y"]).groupby(yrs.index.year).mean() * 100
                            for k in ["Đảo chiều", *wf_p]}).round(1)
    by_year.insert(0, "số ngày", yrs.groupby(yrs.index.year).size())

    write_report(grid_t, best, test_t, wf_t, by_year, wf_epochs, rinfo, seeds)
    pd.DataFrame({"date": test_days, "y": df.loc[test_days, "y"].to_numpy(),
                  **{k: v.to_numpy() for k, v in wf_p.items()}}).to_csv(OUT / "predictions_walkforward.csv",
                                                                         index=False)


def write_report(grid_t, best, test_t, wf_t, by_year, wf_epochs, rinfo, seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    grid_t.to_csv(OUT / "grid_validation.csv", index=False)
    test_t.to_csv(OUT / "test_metrics.csv")
    wf_t.to_csv(OUT / "walkforward_metrics.csv")
    show = grid_t.copy()
    for c in ["train_log_loss", "val_log_loss", "val_log_loss_sd_seed"]:
        show[c] = show[c].round(4)
    for c in ["train_DA", "val_DA", "val_pct_pred_up"]:
        show[c] = (show[c] * 100).round(1)
    show = show.sort_values("val_log_loss")
    ep = {k: np.array(v) for k, v in wf_epochs.items()}
    md = [
        "# Tuning B5 LSTM (góp ý 3 của thầy): " + FEAT_DESC, "",
        "Tạo bởi `lstm_tuning/src/lstm_tuning.py`. Đặc trưng: **" + FEAT_DESC + "**, data freeze v1.0. Dữ liệu, cách chia, "
        "metric và luật B0 lấy từ `baseline_check` nên số liệu so được trực tiếp với bảng LR ở đó.", "",
        "## Thiết lập", "",
        f"- Kiến trúc: LSTM 1 lớp → dropout → Linear, BCE, Adam, batch {FIXED['batch_size']}, "
        f"tối đa {FIXED['max_epochs']} epoch, patience {FIXED['patience']}; p̂ trung bình {len(seeds)} seed "
        f"({', '.join(map(str, seeds))}).",
        "- " + SEED_NOTE.format(n=len(seeds)),
        f"- Mỗi lần học: scaler fit trên ngày train; early stopping trên {int(FIXED['val_frac'] * 100)}% cuối phần "
        f"train, **bỏ {FIXED['gap']} ngày gap** giữa phần fit và phần early stopping; sau đó học lại từ đầu trên "
        "100% phần train với số epoch tốt nhất. Batch xáo trộn trong phần train.",
        "- Khác `model_lstm.py` (B5 trong `ngoc_pipeline`): code đó không có gap ở early stopping, không xáo batch, "
        "không dropout và không học lại trên 100% train. Vì vậy cột \"trước tune\" ở đây được chạy lại bằng code "
        "này, không lấy con số 50,74% của báo cáo giữa kỳ.",
        f"- Grid (chốt trước khi chạy): lookback {GRID['lookback']} × hidden {GRID['hidden']} × "
        f"dropout {GRID['dropout']} × lr {GRID['lr']} = {len(grid_t)} cấu hình. "
        f"Trước tune = cấu hình B5 hiện tại ({cfg_name(DEFAULT)}).",
        "- Chọn theo log-loss validation (p̂ trung bình các seed), làm tròn 4 chữ số; hoà thì chọn mô hình đơn "
        "giản hơn (hidden nhỏ, lookback ngắn).", "",
        "## Bước A. Grid: train 2018–2021 → validation 2022–2023 (chưa đụng test)", "",
        f"**Cấu hình được chọn: {cfg_name(best)}.** Mốc: đoán 50/50 mọi ngày có log-loss 0,6931.", "",
        show.to_markdown(index=False), "",
        "Cột: `val_log_loss_sd_seed` = độ lệch chuẩn val log-loss giữa các seed (cho biết chênh lệch giữa hai cấu "
        "hình có lớn hơn dao động do seed không); `best_epoch_seeds` = số epoch early stopping chọn ở từng seed.", "",
        "## Bước A. Kết quả test 2024 – 08/2026 (chạy 1 lần)", "",
        f"Học lại trên train + validation. Trên train + validation: persistence đúng "
        f"{rinfo['persistence_acc_train'] * 100:.1f}% → luật đảo chiều.", "",
        test_t.to_markdown(), "",
        "## Bước B. Walk-forward 82 fold (2020 – 08/2026), trước và sau tune", "",
        "Cùng lịch `ngoc_pipeline` (train 730, gap 1, test 30, bước 30); tham số giữ cố định theo Bước A ở mọi fold.", "",
        wf_t.to_markdown(), "",
        "Số epoch early stopping chọn (trung bình qua fold × seed): " +
        "; ".join(f"{k}: {v.mean():.1f} (min {v.min()}, max {v.max()})" for k, v in ep.items()) + ".", "",
        "### Theo năm (DA %, walk-forward)", "",
        by_year.to_markdown(), "",
        "Cột Đảo chiều ở đây là luật B0 của walk-forward: chiều được chọn lại trên từng cửa sổ train (một số fold "
        "chọn persistence), nên năm 2026 khác con số 46,3% của bảng EDA trong `baseline_check` (luật đảo chiều "
        "thuần). DA gộp 53,25% thì khớp.", "",
    ]
    (OUT / "lstm_tuning.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Đã lưu {OUT / 'lstm_tuning.md'}")


if __name__ == "__main__":
    main()
