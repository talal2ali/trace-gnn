"""
Operational evaluation under Top-K screening.

Metrics:
  average_precision      - threshold-free ranking quality
  precision_at_k / recall_at_k - at the operational alert budget (alert_rate)
Significance:
  paired_wilcoxon        - across matched rolling-origin folds (the unit of replication)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score


def top_k_from_rate(n: int, alert_rate: float) -> int:
    return max(1, math.ceil(n * alert_rate))


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> Tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    topk = order[:k]
    hits = float(y_true[topk].sum())
    total_pos = float(y_true.sum())
    precision = hits / k if k > 0 else 0.0
    recall = hits / total_pos if total_pos > 0 else 0.0
    return precision, recall


def fold_metrics(y_true: np.ndarray, scores: np.ndarray, alert_rate: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    k = top_k_from_rate(len(y_true), alert_rate)
    p, r = precision_recall_at_k(y_true, scores, k)
    ap = average_precision_score(y_true, scores) if y_true.sum() > 0 else float("nan")
    return {"ap": float(ap), "precision_at_k": p, "recall_at_k": r, "k": k, "n": len(y_true)}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired test between two configs' per-fold metric vectors (a vs b)."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    if len(a) < 1 or np.allclose(diff, 0):
        return {"n_folds": int(len(a)), "median_diff": float(np.median(diff) if len(a) else np.nan),
                "statistic": float("nan"), "p_value": float("nan")}
    # The rule the manuscript declares in Table 7's note and edge_k/analysis_settings.py
    # freezes: differences below 1e-12 are numerical ties and are DROPPED, and the test is
    # the exact Wilcoxon. SciPy's default method="auto" silently falls back to the normal
    # approximation as soon as one zero or tied rank is present, which is why Table 4's four
    # budget-metric rows once read 0.00043 where the exact test gives 0.00003 -- the floor
    # for a unanimous result on the 16 folds that are not ties. Conservative, but it made one
    # table mix two methods.
    kept = diff[np.abs(diff) > 1e-12]
    if len(kept) == 0:
        return {"n_folds": int(len(a)), "median_diff": float(np.median(diff)),
                "n_ties": int(len(diff)), "statistic": float("nan"), "p_value": float("nan")}
    try:
        stat, p = wilcoxon(kept, method="exact")
    except ValueError:
        stat, p = float("nan"), float("nan")
    return {"n_folds": int(len(a)), "median_diff": float(np.median(diff)),
            "n_ties": int(len(diff) - len(kept)),
            "statistic": float(stat), "p_value": float(p)}
