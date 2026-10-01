#!/usr/bin/env python3
"""
edge_k — the configuration is inherited, not restated. No GPU, no data.

    python tests/test_config_matches_canonical.py

This is the test that would have caught the 75-epoch mistake. It checks that every
setting an edge_k arm uses is the canonical one, that each arm changes exactly the field
it declares and nothing else, and that the canonical values in analysis_settings.py still
match what the canonical result files actually say.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EK = HERE.parent
for d in (str(EK.parent / "src"), str(EK.parent), str(EK)):
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)

import config as C

FAILED = []


def canonical_results_present():
    """True when the canonical run's files are reachable.

    Two of the tests below compare this package's declared baselines against the canonical
    result files. In a code-only checkout those files are not present, and the tests used to
    fail with a bare FileNotFoundError traceback, which reads like a broken package rather
    than an absent input. They skip instead, and say so.
    """
    return ((C.CANONICAL_MAIN_DIR / "per_fold.csv").exists()
            and (C.CANONICAL_UID_DIR / "per_fold.csv").exists()
            and Path(C.CANONICAL_REPEATS).exists()
            and Path(C.CANONICAL_MANIFEST).exists())


def skip(name, why):
    print(f"skip      {name}   {why}")


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}      {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def test_imported_not_copied():
    src = (C.EDGE_K / "config.py").read_text()
    check("config.py imports the canonical settings rather than copying them",
          "_load_canonical_expectations" in src and "EXPECTED_TRAIN" in src)
    check("TRAIN_KWARGS came from rerun_canonical",
          C.TRAIN_KWARGS["epochs"] == 200 and C.TRAIN_KWARGS["lr"] == 5e-4,
          f"epochs={C.TRAIN_KWARGS['epochs']} lr={C.TRAIN_KWARGS['lr']}")
    check("MODEL_KWARGS came from rerun_canonical",
          C.MODEL_KWARGS["hidden_dim"] == 128 and C.MODEL_KWARGS["n_heads"] == 4)
    check("canonical anchors", C.CANONICAL_K == 10 and C.N_LAYERS == 2 and
          C.USE_MAX_AGG is False and C.CANONICAL_ENTITY_COLS == ("uid", "merchant"))


def test_each_arm_changes_only_what_it_declares():
    """Builds the three config objects for every arm without any data and diffs them
    against the canonical configuration field by field."""
    from model import ModelConfig
    from temporal_graph import GraphConfig
    from run_date_gnn import TrainConfig

    canon_t = TrainConfig(**C.TRAIN_KWARGS, device=C.DEVICE)
    canon_m = ModelConfig(n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG,
                          use_edge_bias=True, **C.MODEL_KWARGS)
    canon_g = GraphConfig(time_col="unix_time", entity_cols=C.CANONICAL_ENTITY_COLS,
                          max_prior_neighbors=C.CANONICAL_K)

    for arm, spec in C.ARM_SPEC.items():
        t = TrainConfig(**C.TRAIN_KWARGS, device=C.DEVICE)
        m = ModelConfig(n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG,
                        use_edge_bias=spec["use_edge_bias"], **C.MODEL_KWARGS)
        g = GraphConfig(time_col="unix_time", entity_cols=spec["entity_cols"],
                        max_prior_neighbors=spec["k"])

        tdiff = [k for k, v in asdict(canon_t).items() if asdict(t)[k] != v]
        mdiff = [k for k, v in asdict(canon_m).items() if asdict(m)[k] != v]
        gdiff = [k for k, v in asdict(canon_g).items() if asdict(g)[k] != v]

        check(f"{arm}: TrainConfig identical to canonical", not tdiff, str(tdiff))
        expect_m = ["use_edge_bias"] if not spec["use_edge_bias"] else []
        check(f"{arm}: ModelConfig differs only in {expect_m or 'nothing'}",
              mdiff == expect_m, str(mdiff))
        expect_g = []
        if spec["k"] != C.CANONICAL_K:
            expect_g.append("max_prior_neighbors")
        if spec["entity_cols"] != C.CANONICAL_ENTITY_COLS:
            expect_g.append("entity_cols")
        check(f"{arm}: GraphConfig differs only in {expect_g or 'nothing'}",
              sorted(gdiff) == sorted(expect_g), str(gdiff))
        check(f"{arm}: edge_dim stays 5", 3 + len(g.entity_cols) == 5)


def test_declared_baselines_match_the_result_files():
    """analysis_settings.py records the baseline values. They must still be what the
    canonical files say, or every threshold in this package is measured against nothing."""
    if not canonical_results_present():
        skip("declared baselines match the result files",
             "canonical results not in this checkout (code-only release); "
             "run rerun_canonical.py or point at the results archive")
        return
    import pandas as pd
    import analysis_settings as A

    main = pd.read_csv(C.CANONICAL_MAIN_DIR / "per_fold.csv")
    uid = pd.read_csv(C.CANONICAL_UID_DIR / "per_fold.csv")
    rep = pd.read_csv(C.CANONICAL_REPEATS)

    pairs = [("seed42", main["ap"].mean()),
             ("four_seed_mean", rep["mean_ap"].mean()),
             ("uid_only_seed42", uid["ap"].mean())]
    for name, got in pairs:
        want = A.BASELINE_VALUES[name]["value"]
        check(f"baseline {name} = {want}", abs(round(got, 4) - want) < 1e-9,
              f"file says {got:.4f}")

    sd = rep["mean_ap"].std(ddof=1)
    check(f"run-to-run SD = {A.RUN_TO_RUN_SD}", abs(round(sd, 4) - A.RUN_TO_RUN_SD) < 1e-9,
          f"file says {sd:.4f}")
    check("seed 42 is the highest of the four seeds",
          abs(rep["mean_ap"].max() - main["ap"].mean()) < 1e-9,
          "which is why the four-seed mean is the primary baseline")
    check("threshold is 2 x the SD of a difference against the four-seed mean",
          abs(A.THRESHOLDS["decision_threshold"] -
              round(2 * A.RUN_TO_RUN_SD * (1.25 ** 0.5), 3)) < 1e-9,
          f"2 x {A.THRESHOLDS['sd_diff_vs_four_seed_mean']} = "
          f"{2 * A.THRESHOLDS['sd_diff_vs_four_seed_mean']:.4f}")


def test_canonical_graph_expectation_matches_the_manifest():
    if not canonical_results_present():
        skip("canonical graph expectation matches the manifest",
             "canonical results not in this checkout (code-only release)")
        return
    import json
    man = json.loads(C.CANONICAL_MANIFEST.read_text())
    check("edges_k10_both matches run_manifest.json",
          C.EXPECT["edges_k10_both"] == man["experiment"]["edges"],
          f"{man['experiment']['edges']:,}")
    check("edges_per_node matches run_manifest.json",
          C.EXPECT["edges_per_node_k10_both"] == man["experiment"]["edges_per_node"])
    check("contemporaneous edges match run_manifest.json",
          C.EXPECT["contemporaneous_k10_both"] == man["experiment"]["contemporaneous_edges"])
    check("dataset sha256 matches run_manifest.json",
          C.EXPECT["dataset_sha256"] == man["dataset_sha256"])
    check("the 11 features match run_manifest.json",
          C.EXPECT["n_features"] == len(man["experiment"]["features"]))
    check("the 17 test weeks match run_manifest.json",
          list(C.EXPECT["test_weeks"]) == man["experiment"]["test_weeks"])


def main():
    print("edge_k — configuration inherited, not restated\n")
    test_imported_not_copied()
    print()
    test_each_arm_changes_only_what_it_declares()
    print()
    test_declared_baselines_match_the_result_files()
    print()
    test_canonical_graph_expectation_matches_the_manifest()
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("configuration checks passed")


if __name__ == "__main__":
    main()
