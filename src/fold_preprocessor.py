"""
FoldPreprocessor - model-pluggable, fold-aware preprocessing.

Fit ONLY on the training window of a fold, then applied unchanged to validation/test.
This is the single rule that keeps the protocol leakage-safe across model families.

Two orthogonal knobs:
  profile     : "tree" | "linear" | "mlp"   (which transforms run)
  column_set  : "full" | "drop_high_missing" | "redundancy_pruned"

Profiles
  tree    : keep NaN (native handling); label-encode categoricals (train map,
            0 = missing/unseen); no scaling.
  linear  : impute (train median / sentinel) + missingness indicators + standardize;
  mlp       one-hot low-card categoricals, frequency-encode high-card (train-fit).

Nothing here is fit on anything but the training rows passed to .fit().
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiment_config import ExperimentConfig

_MISSING = "__MISSING__"


class FoldPreprocessor:
    def __init__(
        self,
        cfg: ExperimentConfig,
        categorical_cols: List[str],
        profile: str = "tree",
        column_set: str = "full",
        missing_threshold: float = 0.90,
    ):
        assert profile in ("tree", "linear", "mlp")
        assert column_set in ("full", "drop_high_missing", "redundancy_pruned")
        self.cfg = cfg
        self.profile = profile
        self.column_set = column_set
        self.missing_threshold = missing_threshold
        self.all_categorical = set(categorical_cols)

        # learned state
        self.active_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.label_maps: Dict[str, Dict] = {}
        self.medians: Dict[str, float] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.indicator_cols: List[str] = []
        self.onehot_vocab: Dict[str, List] = {}
        self.freq_maps: Dict[str, Dict] = {}
        self.high_card_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self.output_cols_: List[str] = []

    # ------------------------------------------------------------------ #
    def _select_columns(self, X: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        cols = [c for c in feature_cols if c in X.columns]

        # drop constant (train-fit)
        nunq = X[cols].nunique(dropna=False)
        cols = [c for c in cols if nunq.get(c, 0) > 1]

        if self.column_set == "drop_high_missing":
            miss = X[cols].isna().mean()
            cols = [c for c in cols if miss[c] < self.missing_threshold]

        elif self.column_set == "redundancy_pruned":
            num = [c for c in cols if c not in self.all_categorical]
            if len(num) > 1:
                corr = X[num].corr().abs()
                drop = set()
                for i, a in enumerate(num):
                    if a in drop:
                        continue
                    for b in num[i + 1:]:
                        if b in drop:
                            continue
                        r = corr.loc[a, b]
                        if pd.notna(r) and r >= self.cfg.redundancy_corr_threshold:
                            drop.add(b)
                cols = [c for c in cols if c not in drop]
        return cols

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, feature_cols: List[str]) -> "FoldPreprocessor":
        self.active_cols = self._select_columns(X, feature_cols)
        self.cat_cols = [c for c in self.active_cols if c in self.all_categorical]
        self.num_cols = [c for c in self.active_cols if c not in self.all_categorical]

        if self.profile == "tree":
            for c in self.cat_cols:
                uniques = pd.Index(X[c].dropna().unique())
                # 0 reserved for missing / unseen
                self.label_maps[c] = {v: i + 1 for i, v in enumerate(uniques)}
            self.output_cols_ = list(self.num_cols) + list(self.cat_cols)

        else:  # linear / mlp
            # numeric: median impute + indicator (if train-missing) + standardize
            for c in self.num_cols:
                col = X[c]
                self.medians[c] = float(col.median())
                if col.isna().any():
                    self.indicator_cols.append(c)
                filled = col.fillna(self.medians[c])
                self.means[c] = float(filled.mean())
                std = float(filled.std())
                self.stds[c] = std if std > 0 else 1.0

            # categorical: low-card one-hot vs high-card frequency
            for c in self.cat_cols:
                card = X[c].nunique(dropna=True)
                if card <= self.cfg.high_card_threshold:
                    self.low_card_cols.append(c)
                    self.onehot_vocab[c] = list(pd.Index(X[c].fillna(_MISSING).unique()))
                else:
                    self.high_card_cols.append(c)
                    freq = X[c].fillna(_MISSING).value_counts(normalize=True)
                    self.freq_maps[c] = freq.to_dict()

            # establish output column order from a transform of the train head
            self.output_cols_ = list(self.transform(X.head(1)).columns)
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.profile == "tree":
            out = {}
            for c in self.num_cols:
                out[c] = X[c].to_numpy()
            for c in self.cat_cols:
                m = self.label_maps[c]
                out[c] = X[c].map(m).fillna(0).astype(np.int32).to_numpy()
            return pd.DataFrame(out, index=X.index)[self.output_cols_ if self.output_cols_ else list(out)]

        # linear / mlp
        frames = {}
        for c in self.num_cols:
            col = X[c]
            if c in self.indicator_cols:
                frames[f"{c}__ismissing"] = col.isna().astype(np.float32).to_numpy()
            filled = col.fillna(self.medians[c]).to_numpy(dtype="float64")
            frames[c] = ((filled - self.means[c]) / self.stds[c]).astype(np.float32)

        for c in self.low_card_cols:
            vals = X[c].fillna(_MISSING)
            for level in self.onehot_vocab[c]:
                frames[f"{c}={level}"] = (vals == level).astype(np.float32).to_numpy()

        for c in self.high_card_cols:
            vals = X[c].fillna(_MISSING)
            frames[f"{c}__freq"] = vals.map(self.freq_maps[c]).fillna(0.0).astype(np.float32).to_numpy()

        out = pd.DataFrame(frames, index=X.index)
        if self.output_cols_:
            for col in self.output_cols_:
                if col not in out.columns:
                    out[col] = np.float32(0.0)
            out = out[self.output_cols_]
        return out
