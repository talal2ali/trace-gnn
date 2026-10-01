"""
Stage-2 experiment configuration.

Loads the stage-1 prepared dataset (clean-but-raw CSV + manifest) and defines the
fold-aware, model-pluggable experiment: model profiles, column-set variants, the
high-missing threshold sweep, the rolling-origin folds, and operational evaluation.

Defaults are the Sparkov column names, which is the only corpus this paper uses. Every
entry point (`rerun_canonical.py`, `protocol_v2/runner.py`, `edge_k/runner.py` and
`notebooks/15_workstation_session.ipynb`) sets them explicitly anyway, so the defaults are
documentation rather than behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


FEATURE_VERSION = "agg_v3_uid_causal"


@dataclass
class ExperimentConfig:
    # --- stage-1 inputs ---
    prepared_dir: Path = field(default_factory=lambda: Path.cwd() / "prepared")
    clean_csv_name: str = "sparkov_clean_v1.csv"
    manifest_name: str = "column_manifest.json"

    # --- core columns (mirrors stage-1 manifest) ---
    id_col: str = "TransactionID"
    target_col: str = "is_fraud"
    time_col: str = "unix_time"
    week_col: str = "week_idx"
    amount_col: str = "amt"
    uid_col: str = "uid"
    # columns that are keys / intermediates, never fed to the model as features
    non_feature_cols: Tuple[str, ...] = ("uid",)

    # --- uid causal aggregation ---
    seconds_per_day: int = 24 * 60 * 60
    velocity_windows_days: Tuple[int, ...] = (1, 7)  # windowed prior-count features
    feature_version: str = FEATURE_VERSION

    # --- model profiles ---
    profiles: Tuple[str, ...] = ("tree", "linear", "mlp")

    # --- column-set variants ---
    column_sets: Tuple[str, ...] = ("full", "drop_high_missing", "redundancy_pruned")
    missing_thresholds: Tuple[float, ...] = (0.90, 0.80, 0.70)  # sweep for drop_high_missing
    redundancy_corr_threshold: float = 0.90
    high_card_threshold: int = 50  # categoricals above this are frequency-encoded, not one-hot

    # --- rolling-origin folds ---
    min_train_weeks: int = 4          # initial training window before first test week
    expanding_window: bool = True     # True = all prior weeks; False = sliding

    # --- operational evaluation ---
    alert_rate: float = 0.005         # Top-K screening budget (0.5% of test volume)

    # --- misc ---
    random_state: int = 42

    @property
    def clean_csv_path(self) -> Path:
        return self.prepared_dir / self.clean_csv_name

    @property
    def manifest_path(self) -> Path:
        return self.prepared_dir / self.manifest_name
