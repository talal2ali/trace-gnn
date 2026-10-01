"""
edge_k — the analysis, declared BEFORE any arm is trained and hashed.

Everything in this file is a decision taken with no results in hand. It is frozen by
provenance.write_analysis_lock(), which is write-once and runs from the first arm
notebook before that notebook trains anything, so the lock's timestamp precedes every
result file in the directory.

If this file is edited after a result is seen, settings_sha256() moves and
verify_analysis_lock() fails. That is the point.

The reasoning behind each choice is in paper/EDGE_K_WORK_PLAN.md sections 10 and 11.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# --------------------------------------------------------------------------- 1. quantity
PRIMARY_METRIC = {
    "name": "ap",
    "what": "mean average precision over the 17 retained test weeks",
    "estimator": "step-wise (Equation 3), as everywhere else in the paper",
}

# --------------------------------------------------------------------------- 2. baselines
# Fixed now, not chosen after seeing results. Values are read from the canonical run's own
# files at assembly time; the numbers here are recorded so a drift is visible.
BASELINES = {
    "K5_both":      {"primary": "four_seed_mean", "secondary": "seed42"},
    "K20_both":     {"primary": "four_seed_mean", "secondary": "seed42"},
    "no_edge_bias": {"primary": "four_seed_mean", "secondary": "seed42"},
    "uid_K20":      {"primary": "uid_only_seed42", "secondary": "seed42"},
    "no_time_terms":   {"primary": "four_seed_mean", "secondary": "seed42"},
    "no_amount_ratio": {"primary": "four_seed_mean", "secondary": "seed42"},
    "no_relation_ind": {"primary": "four_seed_mean", "secondary": "seed42"},
}

BASELINE_VALUES = {
    "seed42":         {"value": 0.8551, "source": "canonical/main/per_fold.csv, seed 42",
                       "note": "the HIGHEST of the four seeds; using it alone overstates "
                               "any loss by 0.0079"},
    "four_seed_mean": {"value": 0.8472, "source": "repeat_summary.csv, seeds 42-45",
                       "note": "sd 0.0102; this is the primary baseline for effect size"},
    "uid_only_seed42": {"value": 0.7559, "source": "canonical/uid_only/per_fold.csv",
                        "note": "isolates K within one relation for the uid_K20 arm"},
}

# uid_K20 is read against BOTH: against uid_only it is a pure K contrast, against main it
# is the budget-matched relation contrast (20 uid edges vs 10+10). Declared here so the
# second reading is not invented after the fact.
UID_K20_SECOND_READING = {
    "against": "seed42",
    "what": "budget-matched relation contrast: one relation at K=20 versus two at K=10, "
            "both giving 20 in-edges per node",
    "interpretation_if_below": "the merchant relation contributes beyond its edge budget; "
                               "Table 5's +0.099 survives budget matching",
    "interpretation_if_above": "the merchant gain in Table 5 is partly an edge-budget "
                               "effect; Section 5.2 needs revising",
}

# --------------------------------------------------------------------------- 3. the read
PRIMARY_READ = {
    "statistic": "difference in mean AP, arm minus baseline",
    "interval": "95% percentile bootstrap over 10,000 paired resamples of the 17 fold "
                "differences, seed 0",
    "n_resamples": 10_000,
    "bootstrap_seed": 0,
}

# --------------------------------------------------------------------------- 4. threshold
# 2 SD, not 1. The +/-0.014 band in Section 5.4 is ONE standard deviation of a
# two-execution difference; at 1 SD, 32% of true nulls fall outside it.
RUN_TO_RUN_SD = 0.0102              # repeat_summary.csv, seeds 42-45, ddof=1
THRESHOLDS = {
    "sd_single_run": RUN_TO_RUN_SD,
    "sd_diff_vs_seed42": round(RUN_TO_RUN_SD * (2 ** 0.5), 4),        # 0.0144
    "sd_diff_vs_four_seed_mean": round(RUN_TO_RUN_SD * (1.25 ** 0.5), 4),  # 0.0114
    "decision_threshold": 0.023,
    "decision_basis": "2 x the SD of a difference between one new run and the four-seed "
                      "mean (2 x 0.0114 = 0.023)",
    "report_as": "An effect smaller than 0.023 is reported as inside the resolution of a "
                 "single execution. It is neither a null nor an effect.",
}

# --------------------------------------------------------------------------- 5. the test
WILCOXON = {
    "alternative": "two-sided",
    "method": "exact",
    "zero_method": "wilcox",
    "tie_rule": "differences below 1e-12 are treated as numerical ties and dropped before "
                "the test, matching the convention declared in manuscript v6e Table 7",
    "unit": "the fold",
    "n_folds": 17,
    "STATUS": "DESCRIPTIVE ONLY — DOES NOT GATE ANY DECISION",
    "why": "The paired test removes fold-to-fold variation, but with one execution per "
           "condition the execution offset is common to all 17 folds and is perfectly "
           "confounded with the arm effect. The depth3 episode is the warning: Wilcoxon "
           "gave p = 0.196 and later p = 0.963 and was right both times, while the point "
           "estimate moved from +0.024 to -0.003 and changed sign. Report the p-value as "
           "fold consistency. Do not conclude from it.",
}

# --------------------------------------------------------------------------- 6. sign
SIGN_CONVENTION = {
    "rule": "every quantity is reported as arm minus baseline",
    "consequence": "every arm here is a removal or a reduction, so a loss of performance "
                   "is NEGATIVE",
    "derived_from": "nowhere else; this is the single declaration",
}

# --------------------------------------------------------------------------- 7. conduct
CONDUCT = {
    "no_dropping": "No arm is dropped, re-run or reinterpreted after its number is seen. "
                   "An arm that lands inside 0.023 is reported as landing inside 0.023.",
    "no_guard_edits": "No guard, expectation or threshold in this package is edited to "
                      "make a result pass. If a guard fires, that is the guard working.",
    "single_execution": "Every arm is one continuous run. A crashed arm is restarted "
                        "clean and the abandoned attempt is recorded, never resumed.",
    "gate": "The three feature-drop arms run only if |no_edge_bias effect vs the "
            "four-seed mean| >= 0.023. The decision and its measured basis are written to "
            "results/gate_decision.json BEFORE config.ARM_SPEC is edited.",
}

# --------------------------------------------------------------------------- 8. contrasts
# Declared in advance so the table cannot grow new rows after the fact.
CONTRASTS = (
    ("K5_both",      "four_seed_mean", "neighbour cap halved"),
    ("K20_both",     "four_seed_mean", "neighbour cap doubled"),
    ("no_edge_bias", "four_seed_mean", "edge encoding removed"),
    ("uid_K20",      "uid_only_seed42", "neighbour cap doubled within one relation"),
    ("uid_K20",      "seed42",          "budget-matched relation contrast"),
)


def settings_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def print_frozen() -> None:
    print("FROZEN ANALYSIS SETTINGS")
    print(f"  primary metric   {PRIMARY_METRIC['what']}")
    print(f"  primary read     {PRIMARY_READ['statistic']}")
    print(f"                   {PRIMARY_READ['interval']}")
    print(f"  threshold        {THRESHOLDS['decision_threshold']}  "
          f"({THRESHOLDS['decision_basis']})")
    print(f"  run-to-run SD    {RUN_TO_RUN_SD}  -> diff vs 4-seed mean "
          f"{THRESHOLDS['sd_diff_vs_four_seed_mean']}, vs seed42 "
          f"{THRESHOLDS['sd_diff_vs_seed42']}")
    print(f"  wilcoxon         {WILCOXON['method']}, {WILCOXON['alternative']}, "
          f"{WILCOXON['STATUS']}")
    print(f"  sign             {SIGN_CONVENTION['rule']}")
    print("  baselines")
    for k, v in BASELINE_VALUES.items():
        print(f"    {k:16} {v['value']:.4f}   {v['source']}")
    print(f"  settings sha256  {settings_sha256()[:16]}...")
