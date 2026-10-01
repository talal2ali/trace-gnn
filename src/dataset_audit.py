"""Dataset audit for the prepared Sparkov file."""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from config import SparkovPrepConfig, CATEGORICAL_FEATURES

OKABE_ITO = {"blue": "#0072B2", "vermillion": "#D55E00", "sky": "#56B4E9"}


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"column": miss.index, "pct_missing": miss.values})


def cardinality_table(df: pd.DataFrame, cfg: SparkovPrepConfig) -> pd.DataFrame:
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns] + [cfg.uid_col]
    rows = [{"column": c, "n_unique": int(df[c].nunique(dropna=True)),
             "pct_missing": float(df[c].isna().mean())} for c in cats]
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False).reset_index(drop=True)


def weekly_distribution(df: pd.DataFrame, cfg: SparkovPrepConfig) -> pd.DataFrame:
    return (df.groupby(cfg.week_col)[cfg.target_col]
            .agg(n="count", frauds="sum", fraud_rate="mean").reset_index())


def uid_stats(df: pd.DataFrame, cfg: SparkovPrepConfig) -> Dict[str, float]:
    uid = df[cfg.uid_col]
    counts = uid.value_counts()
    return {"n_unique_uids": int(uid.nunique()),
            "median_txns_per_uid": float(counts.median()) if len(counts) else 0.0,
            "max_txns_per_uid": int(counts.max()) if len(counts) else 0,
            "pct_singleton_uids": float((counts == 1).mean()) if len(counts) else 0.0}


def flags(df: pd.DataFrame, miss: pd.DataFrame, cfg: SparkovPrepConfig) -> List[str]:
    out = []
    high = miss.loc[miss["pct_missing"] >= cfg.high_missing_flag_threshold, "column"].tolist()
    out.append(f"{len(high)} column(s) >= {cfg.high_missing_flag_threshold:.0%} missing")
    nun = df.nunique(dropna=False)
    const = nun[nun <= 1].index.tolist()
    if const:
        out.append(f"{len(const)} constant column(s): {const}")
    return out


def _figure(weekly: pd.DataFrame, cfg: SparkovPrepConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[audit] skipping figure ({e})"); return
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(weekly[cfg.week_col], weekly["n"], color=OKABE_ITO["sky"])
    ax1.set_xlabel("week_idx"); ax1.set_ylabel("transactions", color=OKABE_ITO["blue"])
    ax2 = ax1.twinx()
    ax2.plot(weekly[cfg.week_col], weekly["fraud_rate"], color=OKABE_ITO["vermillion"], marker="o")
    ax2.set_ylabel("fraud rate", color=OKABE_ITO["vermillion"])
    ax1.set_title("Sparkov: weekly volume and fraud rate")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "weekly_volume_fraud.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def audit(df: pd.DataFrame, cfg: SparkovPrepConfig | None = None) -> Dict[str, object]:
    cfg = cfg or SparkovPrepConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    miss = missingness_table(df)
    card = cardinality_table(df, cfg)
    weekly = weekly_distribution(df, cfg)
    miss.to_csv(cfg.output_dir / "missingness.csv", index=False)
    card.to_csv(cfg.output_dir / "cardinality.csv", index=False)
    weekly.to_csv(cfg.output_dir / "weekly_distribution.csv", index=False)

    report = {
        "n_transactions": int(df.shape[0]), "n_columns": int(df.shape[1]),
        "n_fraud": int(df[cfg.target_col].sum()), "fraud_rate": float(df[cfg.target_col].mean()),
        "week_min": int(df[cfg.week_col].min()), "week_max": int(df[cfg.week_col].max()),
        "n_weeks": int(df[cfg.week_col].nunique()),
        "uid": uid_stats(df, cfg), "flags": flags(df, miss, cfg),
    }
    (cfg.output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Sparkov prepared dataset - audit summary\n",
             f"- transactions: {report['n_transactions']:,}",
             f"- columns: {report['n_columns']}",
             f"- fraud cases: {report['n_fraud']:,} (rate {report['fraud_rate']:.4%})",
             f"- weeks: {report['week_min']}-{report['week_max']} ({report['n_weeks']} weeks)",
             "", "## uid"]
    u = report["uid"]
    lines += [f"- unique uids (cc_num): {u['n_unique_uids']:,}",
              f"- txns per uid: median {u['median_txns_per_uid']:.1f}, max {u['max_txns_per_uid']:,}",
              f"- singleton uids: {u['pct_singleton_uids']:.2%}", "", "## flags"]
    lines += [f"- {f}" for f in report["flags"]]
    lines += ["", "## missing columns (top 10)"]
    lines += [f"- {r['column']}: {r['pct_missing']:.2%}" for _, r in miss.head(10).iterrows()]
    (cfg.output_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")

    if cfg.write_figures:
        _figure(weekly, cfg)
    print(f"[audit] wrote summaries to {cfg.output_dir}")
    return report


if __name__ == "__main__":
    from prepared_loader import load_prepared
    cfg = SparkovPrepConfig()
    audit(load_prepared(cfg), cfg)
