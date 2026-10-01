"""
edge_k — one place where every setting lives.

Read this file and you know what the seven arms will do. The notebooks restate nothing;
they import from here. Where a value also exists in rerun_canonical.py, this module
IMPORTS it rather than copying it, and the assertions below fail loudly if the two ever
drift apart. That is the protocol_v2 pattern and it exists because the 75-epoch mistake
came from restating a setting instead of inheriting it.

Scope, stated once so it cannot be missed:

    TRAINABLE   K5_both, uid_K20, no_edge_bias, K20_both        the four authorised arms
    SCAFFOLDED  no_time_terms, no_amount_ratio, no_relation_ind three contingent arms,
                                                                built but NOT runnable
    LOCKED      every canonical arm and every protocol_v2 arm   read-only, never touched

ONE NAMED CHANGE PER ARM. Everything else is asserted equal to the canonical run in
runner.configs_for(). The one exception is uid_K20, which changes two things by
construction — it is a budget-matched relation arm, and that is recorded as two changes
rather than described as a clean K arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- locations
EDGE_K = Path(__file__).resolve().parent
REPO = EDGE_K.parent
SRC = REPO / "src"
RESULTS = EDGE_K / "results"
ARMS_DIR = RESULTS / "arms"
DERIVED_DIR = RESULTS / "derived"
NOTEBOOKS = EDGE_K / "notebooks"

PREPARED_CANDIDATES = (
    REPO.parent.parent / "prepared",
    REPO.parent / "prepared",
    REPO / "prepared",
)

# ---------------------------------------------------- the canonical run (READ ONLY)
CANONICAL_RUN = REPO / "results" / "rerun_20260826_114717_m5"
CANONICAL_MAIN_DIR = CANONICAL_RUN / "canonical" / "main"
CANONICAL_UID_DIR = CANONICAL_RUN / "canonical" / "uid_only"
CANONICAL_MANIFEST = CANONICAL_RUN / "run_manifest.json"
CANONICAL_SUMMARY = CANONICAL_RUN / "canonical_summary.csv"
CANONICAL_REPEATS = CANONICAL_RUN / "repeat_summary.csv"
CANONICAL_LOCKFILE = RESULTS / "canonical_lock.json"

# Every directory edge_k is permitted to write into. Anything else is refused by
# guards.assert_writable, which puts all of REPO/results and all of protocol_v2/results
# out of reach.
WRITE_ROOTS = (RESULTS,)

# --------------------------------------------------------------------------- arm registry
TRAINABLE_ARMS = ("K5_both", "uid_K20", "no_edge_bias", "K20_both")
SCAFFOLDED_ARMS = ("no_time_terms", "no_amount_ratio", "no_relation_ind")
LOCKED_NAMES = ("main", "uid_only", "maxagg", "depth3",
                "P3_rolling", "P3_acausal", "P1_causal", "P1_leaky", "P2_cutoff")

# Single-fold arms do not occur here: every edge_k arm uses the full 17 rolling folds.
FOLD_LABELS_P3 = tuple(range(8, 25))

# arm -> the ONE thing it changes, plus everything needed to build it.
#   k              neighbour cap per relation
#   entity_cols    the relations
#   use_edge_bias  the Equation 9 additive term
#   zero_cols      indices of build_edge_features columns forced to zero
ARM_SPEC = {
    "K5_both": {
        "k": 5, "entity_cols": ("uid", "merchant"), "use_edge_bias": True,
        "zero_cols": (), "runnable": True, "notebook": "E1_K5",
        "changes": ("max_prior_neighbors 10 -> 5",),
        "what": "both relations at half the neighbour cap. Quarters the receptive field.",
        "answers": "reviewer comment 2 — K is never varied",
    },
    "uid_K20": {
        "k": 20, "entity_cols": ("uid", "__none__"), "use_edge_bias": True,
        "zero_cols": (), "runnable": True, "notebook": "E2_uid_K20",
        "changes": ("entity_cols ('uid','merchant') -> ('uid','__none__')",
                    "max_prior_neighbors 10 -> 20"),
        "what": "cardholder relation alone, given the same 20-edge budget the two-relation "
                "arm receives. The budget-matched control Table 5 lacks.",
        "answers": "the edge-budget confound in Section 5.2",
    },
    "no_edge_bias": {
        "k": 10, "entity_cols": ("uid", "merchant"), "use_edge_bias": False,
        "zero_cols": (), "runnable": True, "notebook": "E3_no_edge_bias",
        "changes": ("ModelConfig.use_edge_bias True -> False",),
        "what": "the Equation 9 additive edge term removed, leaving plain scaled "
                "dot-product attention over the same causal neighbourhood.",
        "answers": "reviewer comment 1 — the edge features are never ablated",
    },
    "K20_both": {
        "k": 20, "entity_cols": ("uid", "merchant"), "use_edge_bias": True,
        "zero_cols": (), "runnable": True, "notebook": "E4_K20",
        "changes": ("max_prior_neighbors 10 -> 20",),
        "what": "both relations at double the neighbour cap. The expensive end of the "
                "K curve and the only arm with a real memory question.",
        "answers": "reviewer comment 2 — K is never varied",
    },
    # ---- contingent second stage. Built, tested, and NOT runnable until the gate opens.
    "no_time_terms": {
        "k": 10, "entity_cols": ("uid", "merchant"), "use_edge_bias": True,
        "zero_cols": (0, 1), "runnable": False, "notebook": "E5_no_time_terms",
        "changes": ("edge feature columns 0,1 (dt/30, log1p dt) zeroed",),
        "what": "the two elapsed-time terms zeroed; edge_dim stays 5 so the parameter "
                "count is identical.",
        "answers": "reviewer comment 1, second half — which component carries it",
    },
    "no_amount_ratio": {
        "k": 10, "entity_cols": ("uid", "merchant"), "use_edge_bias": True,
        "zero_cols": (4,), "runnable": False, "notebook": "E6_no_amount_ratio",
        "changes": ("edge feature column 4 (log amount ratio) zeroed",),
        "what": "the amount ratio zeroed; edge_dim stays 5.",
        "answers": "reviewer comment 1, second half",
    },
    "no_relation_ind": {
        "k": 10, "entity_cols": ("uid", "merchant"), "use_edge_bias": True,
        "zero_cols": (2, 3), "runnable": False, "notebook": "E7_no_relation_ind",
        "changes": ("edge feature columns 2,3 (relation indicators) zeroed",),
        "what": "the two relation indicators zeroed, so the model cannot tell a "
                "cardholder edge from a merchant edge; edge_dim stays 5.",
        "answers": "reviewer comment 1, second half. The most informative of the three.",
    },
}
assert set(ARM_SPEC) == set(TRAINABLE_ARMS) | set(SCAFFOLDED_ARMS)
assert all(ARM_SPEC[a]["runnable"] for a in TRAINABLE_ARMS)
assert not any(ARM_SPEC[a]["runnable"] for a in SCAFFOLDED_ARMS)
assert not (set(ARM_SPEC) & set(LOCKED_NAMES)), "an edge_k arm shadows a locked arm name"

# The order the notebooks are meant to be run in: cheapest and most certain first, the
# long uncertain one last, with a decision point before it.
NOTEBOOK_ORDER = (
    ("E1", "K5_both", 2.0),
    ("E2", "uid_K20", 7.0),
    ("E3", "no_edge_bias", 8.7),
    ("E4", "K20_both", 25.5),          # <- decision point in the plan sits BEFORE this
)
DECISION_POINT_BEFORE = "K20_both"

# --------------------------------------------------------------------------- device
# cuda:0 matches the canonical run (run_manifest.json -> execution_segments). Change this
# ONE line to cuda:1 if cuda:0 drives your display; provenance.compare_environment()
# reports whichever you pick rather than hiding it.
DEVICE = "cuda:1"   # arm D only: cuda:1 is genuinely empty (20 MiB);
                    # cuda:0 carries a stale idle kernel. Both are RTX 6000 Ada.
                    # compare_environment() reports this against the canonical cuda:0.
CANONICAL_DEVICE = "cuda:0"

# --------------------------------------------------------------- inherited, not restated
def ensure_import_path():
    """Put edge_k FIRST on sys.path, then the repo root, then src.

    src/config.py and edge_k/config.py have the same module name — src's is the Sparkov
    preprocessing config, this one is the edge_k settings. Whichever directory comes first
    wins. Existing entries are removed before re-insertion, because a plain
    `if d not in sys.path` cannot fix an order that is already wrong.
    """
    for d in (str(SRC), str(REPO), str(EDGE_K)):
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)


def _load_canonical_expectations():
    """Import EXPECTED_TRAIN / EXPECTED_MODEL from rerun_canonical.py rather than copying
    them. If that file changes, these change with it and the assertions below fire.

    Loaded by file path rather than by `import rerun_canonical`, so this does not disturb
    sys.path ordering for anything imported afterwards. rerun_canonical.py runs argparse
    only under main(), so importing it is side-effect free.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rerun_canonical_cfg_edge_k", REPO / "rerun_canonical.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CANON = _load_canonical_expectations()

TRAIN_KWARGS = dict(_CANON.EXPECTED_TRAIN)      # epochs=200, lr=5e-4, seed=42, ...
MODEL_KWARGS = dict(_CANON.EXPECTED_MODEL)      # hidden_dim=128, n_heads=4, ...
CANONICAL_ARMS = dict(_CANON.ARMS)

# The canonical two-hop main arm, which every edge_k arm is a one-change variant of.
_entity, _max_agg, _layers, _paper_ap = CANONICAL_ARMS["main"]
N_LAYERS = _layers
USE_MAX_AGG = _max_agg
CANONICAL_ENTITY_COLS = _entity
CANONICAL_K = _CANON.MAX_PRIOR_NEIGHBORS
NUMERIC = _CANON.NUMERIC
MIN_TRAIN_WEEKS = 8
ALERT_RATE = 0.005
SAVE_WEIGHTS = True             # the canonical run saved none, which is why the
                                # checkpoint-replay test could never be run. Do not repeat.

assert TRAIN_KWARGS["epochs"] == 200, (
    f"epochs={TRAIN_KWARGS['epochs']} in rerun_canonical.EXPECTED_TRAIN. Every edge_k arm "
    f"runs at the canonical 200-epoch budget; a different budget is a different "
    f"configuration, not a shorter run of the same one.")
assert TRAIN_KWARGS["seed"] == 42
assert TRAIN_KWARGS["use_swa"] is False
assert N_LAYERS == 2
assert USE_MAX_AGG is False
assert CANONICAL_K == 10
assert CANONICAL_ENTITY_COLS == ("uid", "merchant")

# --------------------------------------------------------------------------- expectations
EXPECT = {
    "slice_rows": 500_000,
    "n_features": 11,
    "n_folds": 17,
    "test_weeks": tuple(range(8, 25)),
    "edge_dim": 5,                        # 3 + len(entity_cols); must hold for EVERY arm
    "scored_rows": 338_456,
    "positives": 1_276,
    "dataset_sha256": "24081f707c7d579ad529e9edb17cee0118cbbdd938e03b6cc56b9b9618f26728",
    "min_free_disk_gb": 2.0,
    "vram_headroom_frac": 0.15,
    # canonical K=10 both-relation graph, from run_manifest.json
    "edges_k10_both": 9_911_248,
    "edges_per_node_k10_both": 19.82,
    "contemporaneous_k10_both": 54,
}

# Per-arm graph predictions, asserted BEFORE training by controls.check_graph(). A K
# change that silently did not take effect would otherwise produce a clean null that
# looks like a finding.
#
#   uid_K20 is a MEASURED figure: experiment_results/edge_relation/graph_stats.csv
#   recorded 9,807,302 edges at 19.61/node and 96.3% cap-binding on exactly this graph
#   in July. It is a hard check, not an estimate.
#   The others are projections from edges/node ~= K x bind_rate; tolerance is +/-5%.
GRAPH_EXPECT = {
    "K5_both":       {"edges_per_node": 9.9,  "tol": 0.05, "source": "projected"},
    "uid_K20":       {"edges_per_node": 19.61, "tol": 0.01,
                      "edges": 9_807_302, "source": "MEASURED July 2026 edge_relation"},
    "no_edge_bias":  {"edges_per_node": 19.82, "tol": 0.001,
                      "edges": 9_911_248, "source": "MEASURED canonical run_manifest"},
    "K20_both":      {"edges_per_node": 39.2, "tol": 0.05, "source": "projected"},
    "no_time_terms": {"edges_per_node": 19.82, "tol": 0.001, "edges": 9_911_248,
                      "source": "MEASURED canonical run_manifest"},
    "no_amount_ratio": {"edges_per_node": 19.82, "tol": 0.001, "edges": 9_911_248,
                        "source": "MEASURED canonical run_manifest"},
    "no_relation_ind": {"edges_per_node": 19.82, "tol": 0.001, "edges": 9_911_248,
                        "source": "MEASURED canonical run_manifest"},
}

# Receptive field per target at two layers, v6e's convention: 1 + d + d^2, d = relations*K.
# The leading 1 is the self-loop. Canonical is 421 and depth3 is 8,421, which is exactly
# what Sections 4.4 and 5.4 print.
def receptive_field(arm: str, n_layers: int = None) -> int:
    n_layers = N_LAYERS if n_layers is None else n_layers
    spec = ARM_SPEC[arm]
    relations = sum(1 for c in spec["entity_cols"] if c != "__none__")
    d = relations * spec["k"]
    return 1 + sum(d ** i for i in range(1, n_layers + 1))


# --------------------------------------------------------------------------- summary
def print_summary(arm: str = None) -> None:
    print("edge_k configuration, resolved from rerun_canonical.py")
    print(f"  epochs {TRAIN_KWARGS['epochs']}   lr {TRAIN_KWARGS['lr']}   "
          f"seed {TRAIN_KWARGS['seed']}   batch {TRAIN_KWARGS['batch_size']}")
    print(f"  loss {TRAIN_KWARGS['loss']}   patience {TRAIN_KWARGS['patience']}   "
          f"use_swa {TRAIN_KWARGS['use_swa']}")
    print(f"  hidden {MODEL_KWARGS['hidden_dim']}  heads {MODEL_KWARGS['n_heads']}  "
          f"layers {N_LAYERS}  max_agg {USE_MAX_AGG}")
    print(f"  canonical: K={CANONICAL_K}  entity_cols={CANONICAL_ENTITY_COLS}  "
          f"use_edge_bias=True  RF=421")
    print(f"  device {DEVICE}")
    if arm is None:
        print("\n  arms:")
        for a in TRAINABLE_ARMS + SCAFFOLDED_ARMS:
            s = ARM_SPEC[a]
            tag = "RUNNABLE  " if s["runnable"] else "scaffolded"
            print(f"    {tag} {a:16} RF {receptive_field(a):>5}   "
                  f"{'; '.join(s['changes'])}")
    else:
        s = ARM_SPEC[arm]
        print(f"\n  arm {arm}   ({'runnable' if s['runnable'] else 'SCAFFOLDED, not runnable'})")
        print(f"    {s['what']}")
        print(f"    answers      : {s['answers']}")
        print(f"    the change   : {'; '.join(s['changes'])}")
        print(f"    K            : {s['k']}")
        print(f"    entity_cols  : {s['entity_cols']}")
        print(f"    use_edge_bias: {s['use_edge_bias']}")
        print(f"    zeroed cols  : {s['zero_cols'] or 'none'}")
        print(f"    receptive fld: {receptive_field(arm)} nodes per target")
