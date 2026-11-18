"""
Bitcoin market-regime HMM module.

Functions
---------
forward_filter(model, X_train_scaled, X_test_scaled)
    Causal forward-filtering probabilities for test observations.
run_hmm_fold(train_df, test_df, ...)
    Fit one HMM fold and classify test observations.
run_hmm_walk_forward(data, ...)
    Run the full walk-forward evaluation and optionally save CSV files.

Input data must contain:
    date, log_return, realized_volatility
Optional:
    ohlcv_close

Windows use ROW COUNTS, not calendar-day counts. Missing dates are not filled.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable log(sum(exp(values)))."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    maximum = np.max(values)
    if not np.isfinite(maximum):
        return float(maximum)
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def _filter_sequence(
    model: GaussianHMM,
    X: np.ndarray,
    initial_prob: np.ndarray | None = None,
) -> np.ndarray:
    """
    Forward-filter a sequence using only observations up to each time.

    initial_prob is the prior state distribution immediately before X[0].
    """
    X = np.asarray(X, dtype=float)
    n_states = int(model.n_components)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[1] != model.means_.shape[1]:
        raise ValueError("Feature count does not match the fitted HMM.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    if len(X) == 0:
        return np.empty((0, n_states), dtype=float)

    log_emissions = model._compute_log_likelihood(X)
    trans = np.asarray(model.transmat_, dtype=float)

    prior = (
        np.asarray(model.startprob_, dtype=float)
        if initial_prob is None
        else np.asarray(initial_prob, dtype=float)
    )
    if prior.shape != (n_states,):
        raise ValueError("initial_prob must have one value per state.")
    prior = np.clip(prior, 0.0, None)
    if not np.isfinite(prior).all() or prior.sum() <= 0:
        raise ValueError("Invalid initial state probabilities.")
    prior /= prior.sum()

    log_prior = np.full(n_states, -np.inf)
    mask = prior > 0
    log_prior[mask] = np.log(prior[mask])

    log_alpha = log_prior + log_emissions[0]
    normalizer = _logsumexp(log_alpha)
    if not np.isfinite(normalizer):
        raise FloatingPointError("Could not normalize initial filtering step.")
    log_alpha -= normalizer

    filtered = np.zeros((len(X), n_states), dtype=float)
    filtered[0] = np.exp(log_alpha)

    log_trans = np.full_like(trans, -np.inf, dtype=float)
    mask_trans = trans > 0
    log_trans[mask_trans] = np.log(trans[mask_trans])

    for t in range(1, len(X)):
        next_log_alpha = np.empty(n_states, dtype=float)
        for j in range(n_states):
            next_log_alpha[j] = (
                _logsumexp(log_alpha + log_trans[:, j])
                + log_emissions[t, j]
            )
        normalizer = _logsumexp(next_log_alpha)
        if not np.isfinite(normalizer):
            raise FloatingPointError(
                f"Could not normalize filtering step at observation {t}."
            )
        log_alpha = next_log_alpha - normalizer
        filtered[t] = np.exp(log_alpha)

    return filtered


def forward_filter(
    model: GaussianHMM,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
) -> np.ndarray:
    """
    Filter train and then test in chronological order; return test probabilities.

    This carries the filtered posterior at the end of train into test and
    does not use future test observations to revise earlier test probabilities.
    """
    X_train_scaled = np.asarray(X_train_scaled, dtype=float)
    X_test_scaled = np.asarray(X_test_scaled, dtype=float)

    if X_train_scaled.ndim != 2 or X_test_scaled.ndim != 2:
        raise ValueError("Train and test arrays must be 2D.")
    if len(X_train_scaled) == 0:
        raise ValueError("Training array must not be empty.")
    if X_train_scaled.shape[1] != X_test_scaled.shape[1]:
        raise ValueError("Train and test feature counts do not match.")
    if not np.isfinite(X_train_scaled).all():
        raise ValueError("Training array contains non-finite values.")
    if not np.isfinite(X_test_scaled).all():
        raise ValueError("Test array contains non-finite values.")
    if len(X_test_scaled) == 0:
        return np.empty((0, model.n_components), dtype=float)

    # Filtering over the concatenated sequence makes test filtering inherit
    # the posterior state distribution from the final training observation.
    all_filtered = _filter_sequence(
        model, np.vstack([X_train_scaled, X_test_scaled])
    )
    return all_filtered[len(X_train_scaled):]


def run_hmm_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str] = ("log_return", "realized_volatility"),
    n_components: int = 3,
    n_init: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit one HMM training fold and classify its test observations."""
    train_df = train_df.sort_values("date").copy()
    test_df = test_df.sort_values("date").copy()

    if train_df.empty or test_df.empty:
        raise ValueError("train_df and test_df must not be empty.")
    if n_components != 3:
        raise ValueError(
            "This module maps exactly 3 states to Bear/Sideways/Bull; "
            "set n_components=3."
        )
    if n_init < 1:
        raise ValueError("n_init must be >= 1.")

    features = list(feature_cols)
    required = set(features + ["date", "log_return"])
    missing_train = sorted(required - set(train_df.columns))
    missing_test = sorted(required - set(test_df.columns))
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns; train={missing_train}, test={missing_test}"
        )

    X_train_raw = train_df[features].to_numpy(dtype=float)
    X_test_raw = test_df[features].to_numpy(dtype=float)
    if not np.isfinite(X_train_raw).all():
        raise ValueError("Training features contain NaN or infinity.")
    if not np.isfinite(X_test_raw).all():
        raise ValueError("Test features contain NaN or infinity.")

    # Fit scaler only on training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    best_model = None
    best_score = -np.inf
    init_errors: List[str] = []

    for init in range(n_init):
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            tol=1e-3,
            min_covar=1e-4,
            random_state=random_state + init,
        )
        try:
            model.fit(X_train)
            score = float(model.score(X_train))
            if np.isfinite(score) and score > best_score:
                best_model = model
                best_score = score
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            init_errors.append(f"init {init}: {type(exc).__name__}: {exc}")

    if best_model is None:
        details = "; ".join(init_errors) if init_errors else "no finite score"
        raise RuntimeError(f"All HMM initializations failed ({details}).")

    # Map states using training returns only (Viterbi assignments).
    train_states = best_model.predict(X_train)
    state_mean_return: Dict[int, float] = {}

    for state in range(n_components):
        state_returns = train_df.loc[
            train_states == state, "log_return"
        ].astype(float)
        state_mean_return[state] = (
            float(state_returns.mean()) if len(state_returns) else np.nan
        )

    if any(not np.isfinite(v) for v in state_mean_return.values()):
        raise RuntimeError(
            "At least one state has no training observations; "
            "cannot reliably map every state to a regime."
        )

    ordered_states = sorted(
        state_mean_return, key=lambda state: state_mean_return[state]
    )
    state_to_regime = {
        ordered_states[0]: "Bear",
        ordered_states[1]: "Sideways",
        ordered_states[2]: "Bull",
    }

    test_probs = forward_filter(best_model, X_train, X_test)
    test_states = test_probs.argmax(axis=1)

    result = test_df[["date"]].copy().reset_index(drop=True)
    result["state"] = test_states
    result["regime"] = [state_to_regime[int(s)] for s in test_states]

    for state in range(n_components):
        result[f"prob_state_{state}"] = test_probs[:, state]

    if "fold" in test_df.columns:
        result["fold"] = test_df["fold"].to_numpy()

    metadata: Dict[str, Any] = {
        "model": best_model,
        "scaler": scaler,
        "state_to_regime": state_to_regime,
        "state_mean_return": state_mean_return,
        "train_log_likelihood": best_score,
        "converged": bool(best_model.monitor_.converged),
        "n_iter": int(best_model.monitor_.iter),
        "initialization_errors": init_errors,
    }
    return result, metadata


def run_hmm_walk_forward(
    data: pd.DataFrame,
    first_test_date: str = "2020-01-01",
    train_rows: int = 730,
    test_rows: int = 30,
    step_rows: int = 30,
    gap_rows: int = 1,
    feature_cols: Sequence[str] = ("log_return", "realized_volatility"),
    n_components: int = 3,
    n_init: int = 5,
    random_state: int = 42,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run sequential walk-forward HMM folds.

    The first test begins at first_test_date. Each fold trains on train_rows
    rows ending gap_rows rows before its test window. Test windows advance by
    step_rows. The final test window may contain fewer than test_rows rows.

    Returns (results, fold_summary, fold_metadata).
    """
    if train_rows <= 0 or test_rows <= 0 or step_rows <= 0:
        raise ValueError("train_rows, test_rows, and step_rows must be positive.")
    if gap_rows < 0:
        raise ValueError("gap_rows must be >= 0.")

    required = {"date", "log_return", *feature_cols}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    start_ts = pd.Timestamp(first_test_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    test_start_idx = int(df["date"].searchsorted(start_ts))
    first_train_start = test_start_idx - gap_rows - train_rows

    if first_train_start < 0:
        raise ValueError(
            "Not enough rows before first_test_date for the requested "
            "training window and gap."
        )

    all_results: List[pd.DataFrame] = []
    fold_stats: List[Dict[str, Any]] = []
    fold_metadata: List[Dict[str, Any]] = []
    fold = 0
    total_start = time.time()

    while True:
        train_start = first_train_start + fold * step_rows
        train_end = train_start + train_rows
        test_start = test_start_idx + fold * step_rows
        test_end = min(test_start + test_rows, len(df))

        if train_end > len(df) or test_start >= len(df):
            break

        train = df.iloc[train_start:train_end].copy()
        test = df.iloc[test_start:test_end].copy()

        if len(train) != train_rows or test.empty:
            break

        # Assert the requested row gap between train and test.
        if train_end + gap_rows > test_start:
            raise RuntimeError(
                f"Fold {fold}: train/test windows violate gap_rows={gap_rows}."
            )

        started = time.time()
        fold_result, info = run_hmm_fold(
            train_df=train,
            test_df=test,
            feature_cols=feature_cols,
            n_components=n_components,
            n_init=n_init,
            random_state=random_state,
        )
        elapsed = time.time() - started

        # Attach original daily data fields for downstream analysis.
        extra_cols = [
            c for c in ("ohlcv_close", "log_return", "realized_volatility")
            if c in test.columns
        ]
        fold_result = fold_result.merge(
            test[["date", *extra_cols]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        fold_result["fold"] = fold
        all_results.append(fold_result)

        fold_stats.append({
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "test_start": test["date"].min(),
            "test_end": test["date"].max(),
            "converged": info["converged"],
            "iterations": info["n_iter"],
            "train_log_likelihood": info["train_log_likelihood"],
            "seconds": round(elapsed, 3),
        })
        fold_metadata.append(info)

        if verbose:
            print(f"Fold {fold} completed in {elapsed:.2f}s")
        fold += 1

    if not all_results:
        raise RuntimeError("No walk-forward folds were completed.")

    results = (
        pd.concat(all_results, ignore_index=True)
        .sort_values(["date", "fold"])
        .reset_index(drop=True)
    )
    summary = pd.DataFrame(fold_stats)

    if out_dir is not None:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "hmm_walk_forward_results.csv", index=False)
        summary.to_csv(output_dir / "hmm_walk_forward_summary.csv", index=False)

    if verbose:
        print(f"\nCompleted folds: {fold}")
        print(f"Predicted rows: {len(results)}")
        print(f"Date range: {results['date'].min()} -> {results['date'].max()}")
        print(f"Total elapsed: {time.time() - total_start:.2f}s")

    return results, summary, fold_metadata
