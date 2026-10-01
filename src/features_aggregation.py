"""
Causal uid behavioral aggregations.

Every feature for a transaction uses ONLY rows that strictly precede it in time within
the same uid. This makes the features leakage-safe by construction, so they can be
computed once on the full time-sorted frame and then sliced per fold (a row's value
never depends on a future row). This is the "zfix": the amount z-score is computed
against the prior mean/std, never including the current transaction.

Feature families (per uid):
    uid_prior_count        - number of prior transactions
    uid_velocity_{d}d      - prior transactions within the last d days
    uid_amt_zdev           - (amt - prior_mean) / prior_std   over prior transactions
    uid_amt_prior_mean     - mean amount over prior transactions
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from experiment_config import ExperimentConfig


def _prior_count_within_window(times: np.ndarray, window: float) -> np.ndarray:
    """For each position i (group-local, time-sorted), count prior rows j<i with
    times[j] >= times[i] - window. Vectorized via searchsorted."""
    lo = np.searchsorted(times, times - window, side="left")
    idx = np.arange(len(times))
    return (idx - lo).astype(np.float32)  # prior rows within window (excludes self)


def add_uid_aggregations(df: pd.DataFrame, cfg: ExperimentConfig) -> List[str]:
    """Add causal uid features in-place. Returns the list of added column names.

    Assumes df is sorted by cfg.time_col (stage-1 guarantees this).
    """
    assert df[cfg.time_col].is_monotonic_increasing, "df must be time-sorted"

    amt = df[cfg.amount_col].to_numpy(dtype="float64")
    uid = df[cfg.uid_col].to_numpy()
    has_uid = pd.notna(uid)

    added: List[str] = []

    # --- prior count, prior mean/std via cumulative sums (exclude current row) ---
    g = df.groupby(cfg.uid_col, sort=False)
    n_prior = g.cumcount().to_numpy().astype(np.float64)  # 0,1,2,... within uid

    amt_s = pd.Series(amt, index=df.index)
    cumsum_incl = amt_s.groupby(df[cfg.uid_col], sort=False).cumsum().to_numpy()
    cumsumsq_incl = (amt_s**2).groupby(df[cfg.uid_col], sort=False).cumsum().to_numpy()
    prior_sum = cumsum_incl - amt
    prior_sumsq = cumsumsq_incl - amt**2

    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(n_prior > 0, prior_sum / n_prior, np.nan)
        prior_var = np.where(n_prior > 1, prior_sumsq / n_prior - prior_mean**2, np.nan)
        prior_var = np.clip(prior_var, 0, None)
        prior_std = np.sqrt(prior_var)
        zdev = np.where((n_prior > 1) & (prior_std > 0), (amt - prior_mean) / prior_std, np.nan)

    # rows without a uid get no behavioral history
    n_prior_out = np.where(has_uid, n_prior, np.nan).astype(np.float32)
    df["uid_prior_count"] = n_prior_out
    df["uid_amt_prior_mean"] = np.where(has_uid, prior_mean, np.nan).astype(np.float32)
    df["uid_amt_zdev"] = np.where(has_uid, zdev, np.nan).astype(np.float32)
    added += ["uid_prior_count", "uid_amt_prior_mean", "uid_amt_zdev"]

    # --- windowed velocity (prior count within d days) ---
    times = df[cfg.time_col].to_numpy(dtype="float64")
    for d in cfg.velocity_windows_days:
        col = f"uid_velocity_{d}d"
        out = np.full(len(df), np.nan, dtype=np.float32)
        window = d * cfg.seconds_per_day
        # per-uid, operate on time-sorted positions
        for _, pos in g.indices.items():
            pos = np.asarray(pos)
            if pos.size == 0:
                continue
            out[pos] = _prior_count_within_window(times[pos], window)
        df[col] = np.where(has_uid, out, np.nan).astype(np.float32)
        added.append(col)

    return added
