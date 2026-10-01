"""
protocol_v2 — one place where every setting lives.

Read this file and you know what the four new arms will do. The notebooks restate
nothing; they import from here. Where a value also exists in rerun_canonical.py, this
module IMPORTS it rather than copying it, and assert_matches_canonical() fails loudly if
the two ever drift apart.

Scope, stated once so it cannot be missed:

    TRAINABLE   P2_cutoff, P1_causal, P1_leaky, P3_acausal      four new arms, 200 epochs
    LOCKED      P3_rolling                                      the canonical Section 5.1
                                                                run, reused as-is, NEVER
                                                                trained or overwritten

P3_rolling has no training specification in this file. There is nothing here for a
trainer to consume. That is deliberate — see guards.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- locations
PROTOCOL_V2 = Path(__file__).resolve().parent
REPO = PROTOCOL_V2.parent
SRC = REPO / "src"
RESULTS = PROTOCOL_V2 / "results"
ARMS_DIR = RESULTS / "arms"
SPLITS_DIR = RESULTS / "splits"
DERIVED_DIR = RESULTS / "derived"
LOGS_DIR = RESULTS / "logs"
NOTEBOOKS = PROTOCOL_V2 / "notebooks"

# The prepared dataset. Resolved by search so the notebooks work from any cwd, but the
# resolved path and its SHA-256 are both recorded in the manifest and compared against
# the canonical run.
PREPARED_CANDIDATES = (
    REPO.parent.parent / "prepared",                    # .../Data Preprocessing Sparkov/prepared
    REPO.parent / "prepared",
    REPO / "prepared",
)

# ---------------------------------------------------------- the canonical run (READ ONLY)
# Section 5.1's main arm. This IS the P3_rolling arm of the protocol ladder. It is never
# retrained, never overwritten, never written to. guards.py enforces that.
CANONICAL_RUN = REPO / "results" / "rerun_20260826_114717_m5"
CANONICAL_P3_DIR = CANONICAL_RUN / "canonical" / "main"
CANONICAL_MANIFEST = CANONICAL_RUN / "run_manifest.json"
CANONICAL_LOCKFILE = RESULTS / "canonical_p3_lock.json"

# Every directory the protocol_v2 code is permitted to write into. Anything else is
# refused by guards.assert_writable, including all of REPO/results.
WRITE_ROOTS = (RESULTS,)

# --------------------------------------------------------------------------- arm registry
TRAINABLE_ARMS = ("P2_cutoff", "P1_causal", "P1_leaky", "P3_acausal")
LOCKED_ARMS = ("P3_rolling",)
ALL_ARMS = TRAINABLE_ARMS + LOCKED_ARMS

# Single-fold arms need an integer fold label, because run_date_gnn_fold casts the label
# with int() (writing test_week, and keying the resume set). The old protocol_runner used
# the strings "cutoff" and "random", which is defect B5. Negative sentinels cannot collide
# with a real week number, and the assembly step never reads them as weeks: it recovers
# the week of every scored row from the saved test-index arrays instead.
FOLD_LABEL_P1 = -1
FOLD_LABEL_P2 = -2

# arm -> (split, edges). "split" names the fold set, "edges" the graph builder.
ARM_SPEC = {
    "P2_cutoff":  {"split": "p2", "edges": "causal",
                   "what": "one model trained on everything before week 8, then used "
                           "unchanged to score weeks 8-24. Never refitted."},
    "P1_causal":  {"split": "p1", "edges": "causal",
                   "what": "random 80/20 split over the same rows, time-directed graph."},
    "P1_leaky":   {"split": "p1", "edges": "acausal",
                   "what": "the same random 80/20 split, graph free to reach forward."},
    "P3_acausal": {"split": "p3", "edges": "acausal",
                   "what": "proper rolling weekly folds, graph free to reach forward."},
}
assert set(ARM_SPEC) == set(TRAINABLE_ARMS)

# The order the notebooks are meant to be run in: cheapest first, so the pipeline is
# proved on a 20-minute arm rather than on the 8.8-hour one.
NOTEBOOK_ORDER = (
    ("A1", "P2_cutoff",  0.3),
    ("A2", "P1_causal",  0.8),
    ("A3", "P1_leaky",   0.8),
    ("A4", "P3_acausal", 8.8),
)

# A4 only. The smoke run trains these weeks, then the full run resumes from that
# checkpoint and trains the rest. See README, "A4 smoke then resume".
SMOKE_WEEKS = (8, 9)

# --------------------------------------------------------------------------- device
# Set to cuda:0 to MATCH the canonical run, which executed on cuda:0
# (run_manifest.json -> execution_segments). Matching it removes the one environment
# difference that was otherwise going to need disclosing.
#
# The original work plan pinned the new arms to cuda:1 instead, so the desktop stayed
# responsive and the card was empty. If cuda:0 is the card driving your display, or
# anything else is using it, change this ONE line to "cuda:1" and nothing else needs to
# move. Both are RTX 6000 Ada, so either choice is numerically equivalent; only the
# provenance record changes, and compare_environment() reports whichever you pick.
#
# The preflight in every arm notebook projects peak VRAM on that arm's largest fold and
# refuses to start unless there is 15% headroom, so an occupied card is caught before
# training rather than nine hours into A4.
DEVICE = "cuda:0"
CANONICAL_DEVICE = "cuda:0"

# --------------------------------------------------------------------------- expectations
# Preflight numbers. Every one is checked before any GPU time is spent. A wrong setting
# costs minutes now and ten hours later.
EXPECT = {
    "slice_rows": 500_000,
    "n_features": 11,
    "n_folds_p3": 17,
    "test_weeks": tuple(range(8, 25)),
    "edges_causal": 9_911_248,
    "edges_per_node_causal": 19.82,
    "contemporaneous_edges_causal": 54,
    "edges_acausal": 9_999_876,
    "edges_per_node_acausal": 20.00,
    "edge_dim": 5,                       # 3 + len(entity_cols)
    "p3_scored_rows": 338_456,
    "p3_positives": 1_276,
    "p2_train_rows": 148_946,            # rows before the first test row
    "p1_universe_rows": 487_402,         # p2_train + p3 test rows; week 25 is excluded
    "p1_train_rows": 389_921,            # int(0.8 * 487_402)
    "p1_test_rows": 97_481,
    "dataset_sha256": "24081f707c7d579ad529e9edb17cee0118cbbdd938e03b6cc56b9b9618f26728",
    "min_free_disk_gb": 2.0,
    "vram_headroom_frac": 0.15,          # require 15% spare over the projected peak
}

MAX_PRIOR_NEIGHBORS = 10
ENTITY_COLS = ("uid", "merchant")
NUMERIC = ("amt", "distance", "age", "city_pop", "hour", "dayofweek")
MIN_TRAIN_WEEKS = 8
ALERT_RATE = 0.005                       # ExperimentConfig default; asserted at load
SAVE_WEIGHTS = True                      # ~80 MB over four arms. Open item 2, answered yes.

# --------------------------------------------------------- settings taken from canonical
def ensure_import_path():
    """Put protocol_v2 FIRST on sys.path, then the repo root, then src.

    src/config.py and protocol_v2/config.py have the same module name — src's is the
    Sparkov preprocessing config, this one is the protocol-v2 settings. Whichever
    directory comes first wins, and getting it wrong makes `import config` silently return
    the wrong module.

    Existing entries are removed before being re-inserted, because a plain
    `if d not in sys.path: insert(0, d)` cannot fix an order that is already wrong.
    """
    for d in (str(SRC), str(REPO), str(PROTOCOL_V2)):
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)


def _load_canonical_expectations():
    """Import EXPECTED_TRAIN / EXPECTED_MODEL from rerun_canonical.py rather than copying
    them. If that file ever changes, these change with it and the assertions below fire.

    Deliberately does NOT touch sys.path. An earlier version pushed src/ to position 0
    here, which left src ahead of protocol_v2 for every later import — the exact ordering
    this package has to avoid. rerun_canonical.py imports nothing from src at module
    level, so nothing is needed.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rerun_canonical_cfg", REPO / "rerun_canonical.py")
    mod = importlib.util.module_from_spec(spec)
    # rerun_canonical.py runs argparse only under main(); importing it is side-effect free
    # apart from setting PYTORCH_CUDA_ALLOC_CONF, which is what we want anyway.
    spec.loader.exec_module(mod)
    return mod


_CANON = _load_canonical_expectations()

TRAIN_KWARGS = dict(_CANON.EXPECTED_TRAIN)          # epochs=200, lr=5e-4, seed=42, ...
MODEL_KWARGS = dict(_CANON.EXPECTED_MODEL)          # hidden_dim=128, n_heads=4, ...
CANONICAL_ARMS = dict(_CANON.ARMS)

# The whole point of protocol v2. If this ever reads 75 again, everything below is void.
assert TRAIN_KWARGS["epochs"] == 200, (
    f"epochs={TRAIN_KWARGS['epochs']} in rerun_canonical.EXPECTED_TRAIN. protocol_v2 "
    f"exists to run all four arms at the canonical 200-epoch budget. Stop.")
assert TRAIN_KWARGS["seed"] == 42
assert TRAIN_KWARGS["use_swa"] is False, "use_swa=True silently disables early stopping"
assert MAX_PRIOR_NEIGHBORS == _CANON.MAX_PRIOR_NEIGHBORS
assert NUMERIC == _CANON.NUMERIC
assert EXPECT["n_features"] == _CANON.EXPECTED_N_FEATURES
assert EXPECT["n_folds_p3"] == _CANON.EXPECTED_N_FOLDS
assert EXPECT["slice_rows"] == _CANON.EXPECTED_SLICE_ROWS
assert EXPECT["p3_scored_rows"] == _CANON.EXPECTED_SCORED_ROWS
assert EXPECT["p3_positives"] == _CANON.EXPECTED_POSITIVES
assert EXPECT["edge_dim"] == _CANON.EXPECTED_EDGE_DIM

# The model shape the four arms use is Section 5.1's "main" arm: uid+merchant, no
# max-agg, two layers. Taken from the canonical arm table, not restated.
_entity, _max_agg, _layers, _paper_ap = CANONICAL_ARMS["main"]
assert _entity == ENTITY_COLS and _max_agg is False and _layers == 2
N_LAYERS = _layers
USE_MAX_AGG = _max_agg


def summary_lines():
    """One block of text a human can check against the manuscript in thirty seconds."""
    t = TRAIN_KWARGS
    return [
        f"epochs           {t['epochs']}            <-- canonical Section 5.1 budget",
        f"lr               {t['lr']}         cosine, T_max = epochs",
        f"weight_decay     {t['weight_decay']}",
        f"grad_clip        {t['grad_clip']}",
        f"val_frac         {t['val_frac']}",
        f"patience         {t['patience']}",
        f"batch_size       {t['batch_size']}",
        f"loss             {t['loss']}  gamma {t['focal_gamma']}",
        f"early stop on    {t['early_stop_metric']}",
        f"use_swa          {t['use_swa']}",
        f"seed             {t['seed']}",
        f"layers/heads     {N_LAYERS} / {MODEL_KWARGS['n_heads']}",
        f"hidden/ffn/drop  {MODEL_KWARGS['hidden_dim']} / x{MODEL_KWARGS['ffn_mult']} / "
        f"{MODEL_KWARGS['dropout']}",
        f"entities         {ENTITY_COLS}  K={MAX_PRIOR_NEIGHBORS} per relation",
        f"alert rate       {ALERT_RATE}",
        f"device           {DEVICE}   (canonical ran on {CANONICAL_DEVICE})",
    ]


def print_summary():
    print("resolved configuration for every protocol_v2 arm")
    print("-" * 62)
    for line in summary_lines():
        print("  " + line)
    print("-" * 62)
    print(f"  trainable here : {', '.join(TRAINABLE_ARMS)}")
    print(f"  LOCKED         : {', '.join(LOCKED_ARMS)}  (read from {CANONICAL_P3_DIR.name}/)")
