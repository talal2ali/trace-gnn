"""
Configuration and Sparkov column semantics for the preprocessing stage.

Mirrors an earlier pipeline on a different corpus: produces a clean-but-raw, time-sorted, typed dataset with
deterministic row-local derived columns and a uid, NaN preserved, then a manifest +
audit. Sparkov is the simpler case: a real account id (cc_num), a real timestamp,
and almost no missing data - so there is no card-key reconstruction and no V-block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PREP_VERSION = "sparkov_clean_v1"

# Raw Sparkov columns (Kaggle fraudTrain.csv / fraudTest.csv)
RAW_TIME = "trans_date_trans_time"
RAW_UNIX = "unix_time"
RAW_CC = "cc_num"

# Final feature columns kept (per agreed decision 4)
CATEGORICAL_FEATURES: Tuple[str, ...] = ("category", "merchant", "gender", "state", "job")
NUMERIC_FEATURES: Tuple[str, ...] = ("amt", "city_pop", "distance", "age")

# Columns dropped as PII / ids / leakage (not used as features)
DROP_COLS: Tuple[str, ...] = (
    "first", "last", "street", "trans_num", "city", "zip",
    "lat", "long", "merch_lat", "merch_long", "dob", RAW_TIME, "Unnamed: 0",
)


@dataclass
class SparkovPrepConfig:
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "prepared")
    figures_dir: Path = field(default_factory=lambda: Path.cwd() / "prepared" / "figures")

    train_file: str = "fraudTrain.csv"
    test_file: str = "fraudTest.csv"

    clean_csv_name: str = "sparkov_clean_v1.csv"
    manifest_name: str = "column_manifest.json"

    # core columns (final, post-prep)
    id_col: str = "TransactionID"
    target_col: str = "is_fraud"
    time_col: str = "unix_time"     # numeric seconds; used for sort / week / velocity
    week_col: str = "week_idx"
    amount_col: str = "amt"
    uid_col: str = "uid"            # = cc_num

    seconds_per_week: int = 7 * 24 * 60 * 60

    protected_from_downcast: Tuple[str, ...] = ("TransactionID", "unix_time", "cc_num")

    high_missing_flag_threshold: float = 0.90
    write_figures: bool = True

    @property
    def clean_csv_path(self) -> Path:
        return self.output_dir / self.clean_csv_name

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_name
