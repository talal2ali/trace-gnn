"""
Rolling-origin weekly folds (leakage-safe temporal evaluation).

For each test week w (from min_train_weeks onward), train on prior weeks and test on
week w. Expanding window uses all prior weeks; sliding uses a fixed-width window.
Periodic retraining is implicit: every fold refits the model and the preprocessor.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

from experiment_config import ExperimentConfig


def make_folds(df: pd.DataFrame, cfg: ExperimentConfig) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Return list of (test_week, train_idx, test_idx).

    Indices are positional (iloc) into df, which must be time-sorted.
    """
    weeks = np.sort(df[cfg.week_col].unique())
    pos = np.arange(len(df))
    wk = df[cfg.week_col].to_numpy()

    folds = []
    for w in weeks:
        if w < cfg.min_train_weeks:
            continue
        if cfg.expanding_window:
            train_mask = wk < w
        else:
            train_mask = (wk < w) & (wk >= w - cfg.min_train_weeks)
        test_mask = wk == w
        train_idx = pos[train_mask]
        test_idx = pos[test_mask]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((int(w), train_idx, test_idx))
    return folds


def describe_folds(folds, df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for w, tr, te in folds:
        rows.append({
            "test_week": w,
            "n_train": len(tr),
            "n_test": len(te),
            "train_frauds": int(df[cfg.target_col].to_numpy()[tr].sum()),
            "test_frauds": int(df[cfg.target_col].to_numpy()[te].sum()),
        })
    return pd.DataFrame(rows)
