"""
Experiment orchestrator.

Flow:
  1. load prepared data (stage 1)
  2. add causal uid aggregations ONCE on the full time-sorted frame
     (leakage-safe by construction - each row uses only prior rows)
  3. for each config (profile x column_set [x missing_threshold]):
        for each rolling-origin fold:
            fit FoldPreprocessor on TRAIN only -> transform train & test
            fit model on train -> score test
            record fold metrics (AP, precision/recall@k)
  4. paired Wilcoxon between column-set variants, per profile
  5. redundancy diagnostic (not shipped; see the note below)

Designed to run as a whole, or call run_single_config / build_configs piecewise from
the driver notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment_config import ExperimentConfig
import data_io
import temporal_folds
import features_aggregation
from fold_preprocessor import FoldPreprocessor
import model_factory
import evaluation
# redundancy_diagnostic is imported lazily inside run_all(), the only function that uses
# it. It is a diagnostic against pre-engineered columns of a different corpus and has
# nothing to do with this paper's data, so it is not shipped; run_all() will
# raise a clear ImportError if it is called without it.


@dataclass
class ConfigSpec:
    profile: str
    column_set: str
    missing_threshold: float = 0.90

    @property
    def name(self) -> str:
        if self.column_set == "drop_high_missing":
            return f"{self.profile}/{self.column_set}@{int(self.missing_threshold*100)}"
        return f"{self.profile}/{self.column_set}"


def build_configs(cfg: ExperimentConfig) -> List[ConfigSpec]:
    specs: List[ConfigSpec] = []
    for profile in cfg.profiles:
        for cs in cfg.column_sets:
            if cs == "drop_high_missing":
                for t in cfg.missing_thresholds:
                    specs.append(ConfigSpec(profile, cs, t))
            else:
                specs.append(ConfigSpec(profile, cs))
    return specs


def prepare_frame(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, Dict, List[str], List[str]]:
    df, manifest = data_io.load_prepared(cfg)
    cat_cols = data_io.categorical_columns(manifest, cfg)
    base_feats = data_io.candidate_feature_columns(manifest, cfg)
    uid_feats = features_aggregation.add_uid_aggregations(df, cfg)
    feature_cols = base_feats + uid_feats
    return df, manifest, feature_cols, uid_feats


def run_single_config(
    df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str],
    folds, spec: ConfigSpec, cfg: ExperimentConfig,
) -> pd.DataFrame:
    y = df[cfg.target_col].to_numpy().astype(int)
    rows = []
    for w, tr, te in folds:
        Xtr_raw, Xte_raw = df.iloc[tr], df.iloc[te]
        ytr, yte = y[tr], y[te]

        fp = FoldPreprocessor(cfg, cat_cols, profile=spec.profile,
                              column_set=spec.column_set,
                              missing_threshold=spec.missing_threshold)
        fp.fit(Xtr_raw, feature_cols)
        Xtr = fp.transform(Xtr_raw)
        Xte = fp.transform(Xte_raw)

        model = model_factory.make_model(spec.profile, cfg)
        model.fit(Xtr.to_numpy(), ytr)
        scores = model_factory.predict_scores(model, Xte.to_numpy())

        m = evaluation.fold_metrics(yte, scores, cfg.alert_rate)
        m.update({"config": spec.name, "test_week": w, "n_features": len(fp.output_cols_)})
        rows.append(m)
    return pd.DataFrame(rows)


def run_all(cfg: Optional[ExperimentConfig] = None, profiles: Optional[List[str]] = None) -> Dict[str, object]:
    cfg = cfg or ExperimentConfig()
    df, manifest, feature_cols, uid_feats = prepare_frame(cfg)
    cat_cols = data_io.categorical_columns(manifest, cfg)
    folds = temporal_folds.make_folds(df, cfg)

    specs = build_configs(cfg)
    if profiles is not None:
        specs = [s for s in specs if s.profile in profiles]

    per_fold = pd.concat(
        [run_single_config(df, feature_cols, cat_cols, folds, s, cfg) for s in specs],
        ignore_index=True,
    )

    # paired Wilcoxon: each column-set variant vs the profile's "full" baseline
    comparisons = []
    for profile in {s.profile for s in specs}:
        base = per_fold[per_fold.config == f"{profile}/full"].sort_values("test_week")
        for cfgname in per_fold.config.unique():
            if not cfgname.startswith(profile + "/") or cfgname == f"{profile}/full":
                continue
            cur = per_fold[per_fold.config == cfgname].sort_values("test_week")
            for metric in ("ap", "precision_at_k", "recall_at_k"):
                res = evaluation.paired_wilcoxon(cur[metric].to_numpy(), base[metric].to_numpy())
                comparisons.append({"profile": profile, "config": cfgname, "vs": "full",
                                    "metric": metric, **res})
    comparisons = pd.DataFrame(comparisons)

    # redundancy on the first train fold
    redundancy = pd.DataFrame()
    if folds:
        _, tr0, _ = folds[0]
        import redundancy_diagnostic          # not shipped with the paper code
        redundancy = redundancy_diagnostic.redundancy_table(df.iloc[tr0], uid_feats, cfg)

    return {"per_fold": per_fold, "comparisons": comparisons,
            "redundancy": redundancy, "folds": folds, "uid_features": uid_feats}


if __name__ == "__main__":
    out = run_all()
    print(out["per_fold"].groupby("config")[["ap", "precision_at_k", "recall_at_k"]].mean().round(4))
