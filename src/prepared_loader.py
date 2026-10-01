"""Loader for the prepared Sparkov dataset (restores dtypes from the manifest)."""

from __future__ import annotations

import json
from typing import Dict

import pandas as pd

from config import SparkovPrepConfig


def load_manifest(cfg: SparkovPrepConfig | None = None) -> Dict[str, object]:
    cfg = cfg or SparkovPrepConfig()
    return json.loads(cfg.manifest_path.read_text(encoding="utf-8"))


def load_prepared(cfg: SparkovPrepConfig | None = None, restore_dtypes: bool = True) -> pd.DataFrame:
    cfg = cfg or SparkovPrepConfig()
    manifest = load_manifest(cfg)
    df = pd.read_csv(cfg.clean_csv_path, low_memory=False)
    if not restore_dtypes:
        return df
    for col, meta in manifest["columns"].items():
        if col not in df.columns:
            continue
        dtype, kind = meta["dtype"], meta["kind"]
        try:
            if kind == "derived_uid" or col == cfg.uid_col:
                df[col] = df[col].astype("object")
            elif kind == "categorical":
                df[col] = df[col].astype("object") if not dtype.startswith(("int", "float")) else df[col].astype(dtype)
            elif dtype.startswith(("int", "float")):
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    df[col] = df[col].astype("float32")
        except (ValueError, TypeError):
            pass
    return df


if __name__ == "__main__":
    cfg = SparkovPrepConfig()
    print(load_prepared(cfg).shape)
