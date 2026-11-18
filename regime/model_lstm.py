
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data import DATE, TARGET, PREDICTION_COLS
from walkforward import fold_indices


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def run_lstm_fold(data, fold, feature_cols, lookback=30,
                  hidden_size=32, num_layers=1,
                  max_epochs=50, patience=5,
                  seed=42, batch_size=32, lr=1e-3):

    df = data.sort_values(DATE).reset_index(drop=True).copy()

    if df[DATE].duplicated().any():
        raise ValueError("Ngày bị trùng.")
    if (df[DATE].diff().iloc[1:] != pd.Timedelta(days=1)).any():
        raise ValueError("Dataset phải có đủ ngày liên tục.")

    train_idx, test_idx = fold_indices(df, fold, TARGET)
    expected_train = (fold.train_end - fold.train_start).days + 1

    if len(train_idx) != expected_train:
        raise ValueError(
            f"Fold {fold.fold_id}: train có {len(train_idx)}/{expected_train} nhãn."
        )

    # Fit scaler chỉ trên dữ liệu feature thuộc train window
    train_rows = df.loc[train_idx, feature_cols].to_numpy(dtype=np.float64)
    train_rows = train_rows[np.isfinite(train_rows).all(axis=1)]

    if len(train_rows) == 0:
        raise ValueError("Không có feature train hợp lệ.")

    scaler = StandardScaler().fit(train_rows)

    raw = df[feature_cols].to_numpy(dtype=np.float64)
    valid_rows = np.isfinite(raw).all(axis=1)

    scaled = np.full(raw.shape, np.nan, dtype=np.float32)
    scaled[valid_rows] = scaler.transform(raw[valid_rows]).astype(np.float32)

    y = df[TARGET].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(df[DATE])

    # Tạo sequence kết thúc ở ngày t; target y_t dự báo hướng close[t+1]
    X_seq, y_seq, seq_dates = [], [], []

    for i in range(lookback - 1, len(df)):
        start = i - lookback + 1

        if not np.isfinite(scaled[start:i + 1]).all():
            continue
        if not np.isfinite(y[i]):
            continue

        X_seq.append(scaled[start:i + 1])
        y_seq.append(int(y[i]))
        seq_dates.append(dates[i])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    seq_dates = pd.DatetimeIndex(seq_dates)

    train_mask = (
        (seq_dates >= fold.train_start) &
        (seq_dates <= fold.train_end)
    )
    test_mask = (
        (seq_dates >= fold.test_start) &
        (seq_dates <= fold.test_end)
    )

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]
    test_dates = seq_dates[test_mask]

    if len(X_train) < 10 or len(X_test) == 0:
        raise ValueError(
            f"Fold {fold.fold_id}: sequence train/test không đủ "
            f"({len(X_train)}, {len(X_test)})."
        )

    # Validation là 20% cuối của train, giữ nguyên thứ tự thời gian
    split = int(len(X_train) * 0.8)
    if split <= 0 or split >= len(X_train):
        raise ValueError("Không đủ dữ liệu để tạo validation.")

    X_fit, y_fit = X_train[:split], y_train[:split]
    X_val, y_val = X_train[split:], y_train[split:]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_fit), torch.tensor(y_fit)),
        batch_size=batch_size,
        shuffle=False,
    )

    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        print(f"Fold {fold.fold_id} | Epoch {epoch + 1:02d} | val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("Không tìm được trạng thái model tốt nhất.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, device=device))
        p_hat = torch.sigmoid(logits).cpu().numpy()

    result = pd.DataFrame({
        "date": test_dates,
        "fold": fold.fold_id,
        "y_true": y_test.astype(int),
        "p_hat": p_hat,
        "regime": pd.NA,
    })

    return result[PREDICTION_COLS], best_loss


def run_lstm(data, folds, feature_cols, **kwargs):
    outputs = []

    for fold in folds:
        pred, val_loss = run_lstm_fold(
            data, fold, feature_cols, **kwargs
        )
        outputs.append(pred)
        print(
            f"Fold {fold.fold_id}: {len(pred)} ngày test; "
            f"best_val_loss={val_loss:.4f}"
        )

    return pd.concat(outputs, ignore_index=True)[PREDICTION_COLS]
