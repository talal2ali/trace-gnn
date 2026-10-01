"""
Sparkov preprocessing: raw fraudTrain/fraudTest -> clean-but-raw dataset + manifest.

Pipeline (all row-local / deterministic / target-free):
    1. concat fraudTrain + fraudTest
    2. sort by unix_time (global timeline preserved; given split is re-folded weekly)
    3. derive week_idx (zero-based), hour, dayofweek, age, distance (haversine)
    4. uid = cc_num
    5. drop PII / id / leakage columns; keep agreed feature set
    6. memory downcast (value-preserving) + write clean CSV + manifest

NaN is left as-is (Sparkov has almost none). No encoding / selection here - those are
fold-aware in the experiment.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd

from config import (SparkovPrepConfig, PREP_VERSION, CATEGORICAL_FEATURES,
                    NUMERIC_FEATURES, DROP_COLS, RAW_TIME, RAW_UNIX, RAW_CC)


def load_and_concat(cfg: SparkovPrepConfig) -> pd.DataFrame:
    tr = cfg.data_dir / cfg.train_file
    te = cfg.data_dir / cfg.test_file
    missing = [p for p in (tr, te) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Sparkov files not found:\n" + "\n".join(str(p) for p in missing)
            + f"\n\nPlace fraudTrain.csv / fraudTest.csv in {cfg.data_dir}."
        )
    a = pd.read_csv(tr)
    b = pd.read_csv(te)
    df = pd.concat([a, b], ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0  # km
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return (2 * R * np.arcsin(np.sqrt(a))).astype(np.float32)


def add_derived(df: pd.DataFrame, cfg: SparkovPrepConfig) -> pd.DataFrame:
    dt = pd.to_datetime(df[RAW_TIME], errors="coerce")
    df[cfg.week_col] = ((df[RAW_UNIX] - df[RAW_UNIX].min()) // cfg.seconds_per_week).astype("int32")
    df["hour"] = dt.dt.hour.astype("int16")
    df["dayofweek"] = dt.dt.dayofweek.astype("int16")

    dob = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = ((dt - dob).dt.days / 365.25).astype("float32")

    df["distance"] = _haversine(df["lat"].to_numpy(), df["long"].to_numpy(),
                                df["merch_lat"].to_numpy(), df["merch_long"].to_numpy())

    df[cfg.uid_col] = df[RAW_CC].astype(str)
    return df


def select_columns(df: pd.DataFrame, cfg: SparkovPrepConfig) -> pd.DataFrame:
    df = df.sort_values(cfg.time_col, kind="mergesort").reset_index(drop=True)
    df[cfg.id_col] = np.arange(1, len(df) + 1, dtype=np.int64)
    keep = ([cfg.id_col, cfg.target_col, cfg.time_col, cfg.week_col, "hour", "dayofweek",
             cfg.uid_col] + list(CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES))
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def reduce_mem_usage(df: pd.DataFrame, cfg: SparkovPrepConfig) -> pd.DataFrame:
    protected = set(cfg.protected_from_downcast)
    for col in df.columns:
        if col in protected:
            continue
        ctype = df[col].dtype
        if pd.api.types.is_integer_dtype(ctype):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(ctype):
            df[col] = df[col].astype(np.float32)
    return df


def _kind(col: str, cfg: SparkovPrepConfig) -> str:
    if col == cfg.target_col:
        return "target"
    if col == cfg.id_col:
        return "key"
    if col == cfg.time_col:
        return "time"
    if col in (cfg.week_col, "hour", "dayofweek"):
        return "derived_time"
    if col == cfg.uid_col:
        return "derived_uid"
    if col in CATEGORICAL_FEATURES:
        return "categorical"
    return "numeric"


def build_manifest(df: pd.DataFrame, cfg: SparkovPrepConfig) -> Dict[str, object]:
    return {
        "prep_version": PREP_VERSION,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "id_col": cfg.id_col, "target_col": cfg.target_col, "time_col": cfg.time_col,
        "week_col": cfg.week_col, "uid_col": cfg.uid_col,
        "columns": {c: {"dtype": str(df[c].dtype), "kind": _kind(c, cfg)} for c in df.columns},
    }


def write_outputs(df: pd.DataFrame, cfg: SparkovPrepConfig) -> Dict[str, object]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.clean_csv_path, index=False)
    manifest = build_manifest(df, cfg)
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare(cfg: SparkovPrepConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SparkovPrepConfig()
    print(f"[prep] {PREP_VERSION}: loading + concatenating ...")
    df = load_and_concat(cfg)
    print(f"[prep] concatenated shape: {df.shape}")
    df = add_derived(df, cfg)
    df = select_columns(df, cfg)
    df = reduce_mem_usage(df, cfg)
    manifest = write_outputs(df, cfg)
    print(f"[prep] wrote {cfg.clean_csv_path}  ({manifest['n_rows']:,} rows x {manifest['n_columns']} cols)")
    print(f"[prep] wrote {cfg.manifest_path}")
    return df


if __name__ == "__main__":
    from dataset_audit import audit
    cfg = SparkovPrepConfig()
    df = prepare(cfg)
    audit(df, cfg)
