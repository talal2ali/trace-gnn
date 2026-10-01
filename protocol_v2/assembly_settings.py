"""
protocol_v2 — the analysis settings, FROZEN before any of the four arms is trained.

This file was written on 2026-09-01, before a single new arm existed. Everything the
assembly notebook (A5) does to the results is decided here, in advance: which arms enter
which contrast, which weeks each contrast uses, how the Wilcoxon test is configured, how
numbers are rounded. A5 imports these and may not override them.

The point is narrow and worth stating honestly. The contrasts themselves were already
published in Tables 6 and 7 of a manuscript under revision, so there was little room to
fish in the first place. What was genuinely undeclared, and is declared here, is:

  * the FOLD-SET RULE. Contrasts involving a P1 arm use 16 weeks; P2-versus-P3 uses 17.
    The difference is legitimate — P1's random 20% caught none of week 24's frauds — but
    it was never stated, and two rows of Table 7 quietly rest on different denominators.

  * the WILCOXON SETTINGS. The stored p = 0.36349 for P1_leaky - P1_causal is reproduced
    by none of the four zero-handling variants. The manuscript's printed 0.326 matches
    method="exact" with zero_method="wilcox". So the manuscript number reproduces and the
    shipped CSV is stale. Declaring the settings once, in advance, ends that.

    Note, because the audit got this wrong and this file repeated it: that contrast DOES
    contain zero differences — weeks 10 and 21, 2 of 16. See WILCOXON below.

Verify the freeze: settings_sha256() is recorded in the top-level manifest before A1
runs. If this file is edited after the results are seen, the hash moves and the record
shows it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# --------------------------------------------------------------------------- fold sets
# The five rungs of the ladder. P3_rolling is the canonical Section 5.1 run, read-only.
LADDER_ORDER = ("P1_leaky", "P1_causal", "P2_cutoff", "P3_rolling", "P3_acausal")

FOLD_SET_RULE = {
    "rule": "A contrast uses the weeks common to the arms it involves. In practice that "
            "is 16 weeks (8-23) whenever a P1 arm is on either side, and 17 weeks (8-24) "
            "otherwise.",
    "why_16_for_P1": "P1's random 20% test sample contains no fraud in week 24, so week "
                     "24 has no defined AP for either P1 arm and drops from any contrast "
                     "they enter.",
    "weeks_with_P1": tuple(range(8, 24)),         # 16 weeks
    "weeks_without_P1": tuple(range(8, 25)),      # 17 weeks
    "ladder_table_weeks": "16, the intersection across all five arms, so every row of "
                          "Table 6 is computed on the same weeks",
}

# --------------------------------------------------------------------------- contrasts
# (label, arm_a, arm_b) — reported as a - b. Fixed now, not chosen after looking.
# The four middle rows are Table 7's four cells; the manuscript's names are kept so the
# table can be checked line by line. Two of them are ALSO the single corrections, and
# carry a second name for that role — that duplication is deliberate, because using the
# wrong pair here is precisely the mistake this file now guards against.
CONTRASTS = (
    ("graph effect under rolling origin", "P3_acausal", "P3_rolling"),   # Table 7
    ("graph effect under random split",   "P1_leaky",   "P1_causal"),    # Table 7
    ("split effect, causal graph",        "P1_causal",  "P3_rolling"),   # Table 7
    ("split effect, unconstrained graph", "P1_leaky",   "P3_acausal"),   # Table 7
    ("total protocol effect",             "P1_leaky",   "P3_rolling"),
    ("cost of not retraining",            "P2_cutoff",  "P3_rolling"),
    ("single correction, graph",          "P1_leaky",   "P1_causal"),    # == row 2
    ("single correction, split",          "P1_leaky",   "P3_acausal"),   # == row 4
)

# ------------------------------------------------------------------ THE SIGN CONVENTION
# Declared once, here, and derived from nowhere else. An earlier draft of this package got
# this wrong in a way that was hard to see: it summed the wrong pair of contrasts, which
# produced the right MAGNITUDE with the opposite SIGN. 0.104 became -0.104 and looked
# almost right.
#
# The convention below is the manuscript's, and it reproduces the published +0.1040
# exactly from results/protocol/per_week_all_arms.csv.
#
#   Baseline for corrections is P1_leaky — the corner where BOTH problems are present.
#   A "correction" moves away from it, toward the honest protocol P3_rolling.
#
#     single correction, graph   P1_leaky - P1_causal    fix the graph, keep the random
#                                                        split            = +0.0028
#     single correction, split   P1_leaky - P3_acausal   fix the split, keep the
#                                                        unconstrained graph = +0.0193
#     sum of the two                                                       = +0.0220
#     total protocol effect      P1_leaky - P3_rolling   fix both          = +0.1261
#     non-additive excess        total - sum                              = +0.1040
#
#   Reduced to arms, the excess is
#
#     P1_causal + P3_acausal - P1_leaky - P3_rolling
#
#   which is what EXCESS_PER_WEEK computes, week by week. The aggregate is the mean of
#   that vector — not a separately-derived number — so the aggregate and the per-week
#   test cannot drift apart in sign again. assemble.non_additivity() asserts they agree
#   to 1e-12.
#
#   POSITIVE EXCESS MEANS: fixing either defect on its own recovers far less than fixing
#   both. That is the paper's claim, and the sign carries it.
#
# Note for anyone comparing against notebooks/08b_protocol_11feat.ipynb: that notebook is
# internally inconsistent here. Its aggregate path prints +0.1040, its per-week vector
# `(P1_leaky - P1_causal) - (P3_acausal - P3_rolling)` averages -0.1040. Both magnitudes
# are right and the two-sided p-value is identical either way, so no published number is
# wrong — but the inconsistency must not survive into this package, and does not.

SIGN_CONVENTION = {
    "baseline": "P1_leaky",
    "target": "P3_rolling",
    "direction": "a correction moves from P1_leaky toward P3_rolling; effects are "
                 "reported as positive when the correction improves average precision",
    "single_corrections": ("single correction, graph", "single correction, split"),
    "excess_formula": "P1_causal + P3_acausal - P1_leaky - P3_rolling",
    "excess_sign_means": "positive = fixing one defect alone recovers much less than "
                         "fixing both; the protocol effects are non-additive",
    "published_value": 0.1040,
    "reproduces_from": "results/protocol/per_week_all_arms.csv, 16 common weeks",
}

# The per-week excess, as (coefficient, arm) terms. assemble.py builds the vector from
# this list rather than from a hand-written expression, so the convention above and the
# arithmetic below cannot disagree.
EXCESS_PER_WEEK = ((+1, "P1_causal"), (+1, "P3_acausal"),
                   (-1, "P1_leaky"), (-1, "P3_rolling"))

DERIVED_QUANTITIES = {
    "sum of the two single corrections":
        "(P1_leaky - P1_causal) + (P1_leaky - P3_acausal), i.e. the two contrasts named "
        "in SIGN_CONVENTION['single_corrections']. NOT the two effects measured from "
        "P3_rolling — those sum to 0.2301 and give the excess the wrong sign.",
    "non-additive excess":
        "total protocol effect - sum of the two single corrections, which reduces to "
        "EXCESS_PER_WEEK. Reported as the mean of the per-week vector.",
    "interaction, per week":
        "the same EXCESS_PER_WEEK vector, with a two-sided Wilcoxon on it. The aggregate "
        "excess IS the mean of this vector; they are one quantity, not two.",
}

FACTORIAL_2X2 = {
    ("random split", "unconstrained graph"): "P1_leaky",
    ("random split", "causal graph"): "P1_causal",
    ("rolling origin", "unconstrained graph"): "P3_acausal",
    ("rolling origin", "causal graph"): "P3_rolling",
}

# --------------------------------------------------------------------------- statistics
WILCOXON = {
    "alternative": "two-sided",
    "zero_method": "wilcox",
    "method": "exact",
    "why_exact": "n is 16 or 17, well under scipy's exact-test limit of 25, so the exact "
                 "test is available for every contrast. It is also the setting that "
                 "reproduces the manuscript's printed 0.326 for P1_leaky - P1_causal.",
    "zero_differences": "THEY DO OCCUR, and an earlier version of this file said they did "
                        "not. On the shipped 75-epoch table, P1_leaky - P1_causal is "
                        "exactly zero in weeks 10 and 21 — 2 of 16. The audit's claim "
                        "that 'no zero differences occur' was wrong, and the fallback "
                        "written on top of it (switch to method='auto' when zeros "
                        "appear) had the effect of making the manuscript's own number "
                        "unreachable: auto gives 0.30029, exact gives 0.32581, and the "
                        "manuscript prints 0.326.",
    "how_zeros_are_handled": "zero_method='wilcox' drops zero differences before ranking, "
                             "which is scipy's default and what the published number "
                             "used. The exact test is then run on the reduced sample. "
                             "Both counts are reported: n_weeks is the number of paired "
                             "weeks, n_effective is what the test actually used, and "
                             "n_zero says how many were dropped. delta and the fold set "
                             "are always over n_weeks.",
    "no_silent_fallback": "method is 'exact' for every contrast, unconditionally. There "
                          "is no automatic switch. If n ever exceeds 25 the code raises "
                          "rather than quietly downgrading to the normal approximation.",
    "alpha": 0.05,
    "multiplicity": "none applied. Six contrasts, all pre-declared, all reported. The "
                    "effects of interest sit far below any reasonable corrected "
                    "threshold; the point is disclosure, not a corrected claim.",
    "inference_caveat": "rolling-origin folds have disjoint test weeks but nested, "
                        "overlapping training histories, so these tests are descriptive "
                        "under temporal dependence rather than exact.",
}

BOOTSTRAP = {"n_resamples": 10_000, "seed": 0, "interval": "percentile", "level": 0.95}

# --------------------------------------------------------------------------- metrics
AVERAGING = {
    "across_folds": "unweighted mean over weeks. Weekly test volume nearly doubles over "
                    "the period (15,766 to 31,849), so this is a choice, not a default, "
                    "and it is the choice the existing code makes.",
    "auroc": "computed in A5 from the retained per-transaction scores, per week, then "
             "averaged the same way. In the 75-epoch study it was computed inside the "
             "training loop; moving it out is what lets the training code stay identical "
             "to Section 5.1.",
    "top_k": "k = ceil(alert_rate * n) per week, alert_rate = 0.005",
    "ties": "np.argsort(-scores, kind='mergesort'), i.e. stable, first-come on ties",
    "f1_mcc": "computed from the top-k confusion matrix, same k and tie rule",
}

ROUNDING = {"tables": 4, "manuscript_text": 3,
            "rule": "round once, at the point of printing, from unrounded values. Never "
                    "compute a contrast from rounded inputs."}

REPORTED_METRICS = ("auroc", "ap", "precision_at_k", "recall_at_k", "f1_at_k", "mcc_at_k")

# --------------------------------------------------------------------------- exclusions
DO_NOT_USE = {
    "results/protocol/*": "the 75-epoch arms. Their scores may not enter any table built "
                          "from 200-epoch arms. Archived as superseded provenance only.",
    "experiment_results/protocol_11/*": "same, and the per-transaction scores were never "
                                        "retained, so F1 and MCC cannot be recovered from "
                                        "them at all.",
    "results/architecture/arch_11_*.csv": "summarise superseded architecture arms; never "
                                          "regenerated.",
}

PROVENANCE_SPLIT = {
    "note": "The five rungs come from two sessions, and the manifest records both. "
            "P3_rolling was trained 2026-08-26 on cuda:0; the other four are trained now "
            "on cuda:1. Every setting is matched. The wall-clock sitting is not, and that "
            "is a real difference that belongs in the paper. It is the price of not "
            "retraining P3, and it is smaller than the problem it avoids: a second "
            "stochastic draw of P3 would replace a known 0.0066 AP gap with an expected "
            "~0.010 one, on the arm every contrast is anchored to.",
    "scores_schema": "All four new arms write Section 5.1's schema, where row_pos is a "
                     "per-fold counter. The global row, and hence the week, is recovered "
                     "in A5 from the saved test-index arrays. The old "
                     "scores_P3_rolling.csv stored global positions directly; the two "
                     "files are not in the same format.",
}


def settings_sha256() -> str:
    """Hash of this file's bytes. Recorded before the first arm runs."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def print_frozen():
    print("FROZEN analysis settings   sha256", settings_sha256()[:32], "...")
    print(f"  contrasts        {len(CONTRASTS)}, pre-declared")
    print(f"  fold-set rule    16 weeks when a P1 arm is involved, 17 otherwise")
    print(f"  wilcoxon         {WILCOXON['alternative']}, zero_method="
          f"{WILCOXON['zero_method']!r}, method={WILCOXON['method']!r}, "
          f"alpha={WILCOXON['alpha']}")
    print(f"  averaging        {AVERAGING['top_k']}, unweighted mean over weeks")
    print(f"  rounding         {ROUNDING['tables']} dp in tables, from unrounded values")
