"""Load the stage-1 prepared dataset for the experiment, restoring dtypes."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd

from experiment_config import ExperimentConfig


def load_manifest(cfg: ExperimentConfig) -> Dict[str, object]:
    return json.loads(cfg.manifest_path.read_text(encoding="utf-8"))


def load_prepared(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    manifest = load_manifest(cfg)
    df = pd.read_csv(cfg.clean_csv_path, low_memory=False)
    for col, meta in manifest["columns"].items():
        if col not in df.columns:
            continue
        dtype, kind = meta["dtype"], meta["kind"]
        try:
            if kind == "categorical" and not dtype.startswith(("int", "float")):
                df[col] = df[col].astype("object")
            elif dtype.startswith(("int", "float")):
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    df[col] = df[col].astype("float32")
        except (ValueError, TypeError):
            pass
    return df, manifest


def categorical_columns(manifest: Dict[str, object], cfg: ExperimentConfig) -> List[str]:
    return [c for c, m in manifest["columns"].items() if m["kind"] == "categorical"]


def candidate_feature_columns(manifest: Dict[str, object], cfg: ExperimentConfig) -> List[str]:
    """Raw manifest predictors (excludes key/target/time and configured non-features).

    uid aggregation features are added separately by features_aggregation and are not
    part of the manifest.
    """
    exclude_kinds = {"key", "target", "time"}
    out = []
    for c, m in manifest["columns"].items():
        if m["kind"] in exclude_kinds:
            continue
        if c in cfg.non_feature_cols:
            continue
        out.append(c)
    return out
