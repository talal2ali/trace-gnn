#!/usr/bin/env python3
"""
edge_k — static checks. No GPU, no data, no training.

    python tests/test_static.py

Checks the shape of the package and the notebooks: that no notebook restates a
hyperparameter, that every arm runs at 200 epochs, that the scaffolded arms really are
unrunnable, that the path-order collision is handled, and that no notebook ships with
stored output.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EK = HERE.parent
for d in (str(EK.parent / "src"), str(EK.parent), str(EK)):
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)

import config as C

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}      {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def notebooks():
    return sorted(Path(C.NOTEBOOKS).glob("*.ipynb"))


def cells(p):
    nb = json.loads(p.read_text())
    return ["".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"]


def test_every_arm_has_a_notebook():
    have = {p.stem for p in notebooks()}
    want = {s["notebook"] for s in C.ARM_SPEC.values()}
    check("every arm has a notebook", have == want, f"{sorted(have ^ want) or 'exact match'}")


def test_notebooks_are_unrun():
    bad = []
    for p in notebooks():
        nb = json.loads(p.read_text())
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and (c.get("outputs") or
                                             c.get("execution_count") is not None):
                bad.append(f"{p.name}#{i}")
    check("every code cell is unexecuted with no stored output", not bad, str(bad[:4]))


def test_no_restated_hyperparameters():
    """A notebook that writes `epochs=200` has stopped inheriting the configuration. This
    is the check that would have caught the 75-epoch mistake."""
    banned = ("epochs", "lr", "weight_decay", "hidden_dim", "n_heads", "ffn_mult",
              "dropout", "seed", "alert_rate", "max_prior_neighbors", "batch_size",
              "patience", "focal_gamma", "n_layers", "cat_emb_dim")
    bad = []
    for p in notebooks():
        for src in cells(p):
            for kw in banned:
                if re.search(rf"\b{kw}\s*=\s*[0-9\"']", src):
                    bad.append(f"{p.name}: {kw}=")
    check("no notebook restates a hyperparameter as a literal", not bad, str(bad[:4]))


def test_epoch_budget_is_200_everywhere():
    check("epochs come from rerun_canonical and equal 200",
          C.TRAIN_KWARGS["epochs"] == 200, f"epochs={C.TRAIN_KWARGS['epochs']}")
    bad = [p.name for p in notebooks() if re.search(r"\b75\b", "".join(cells(p)))]
    check("the number 75 appears in no notebook", not bad, str(bad))


def test_scaffolded_arms_are_not_runnable():
    import guards
    ok = True
    for arm in C.SCAFFOLDED_ARMS:
        try:
            guards.assert_trainable(arm)
            ok = False
        except guards.NotYetAuthorisedError:
            pass
    check("all three scaffolded arms refuse to train", ok)
    check("scaffolded arms are flagged runnable=False",
          not any(C.ARM_SPEC[a]["runnable"] for a in C.SCAFFOLDED_ARMS))


def test_locked_names_refused():
    import guards
    ok = True
    for name in C.LOCKED_NAMES:
        try:
            guards.assert_trainable(name)
            ok = False
        except guards.LockedArmError:
            pass
    check("every canonical and protocol_v2 arm name is refused", ok,
          f"{len(C.LOCKED_NAMES)} names")


def test_path_order_handled():
    bad = [p.name for p in notebooks()
           if "sys.path.insert(0, d)" not in "".join(cells(p))
           or "wrong config module" not in "".join(cells(p))]
    check("every notebook fixes the config path order and asserts it", not bad, str(bad))


def test_cleanup_is_manual():
    bad = []
    for p in notebooks():
        cs = cells(p)
        if "empty_cache" not in cs[-1]:
            bad.append(f"{p.name}: last cell is not the cleanup")
        if any("empty_cache" in c for c in cs[:-1]):
            bad.append(f"{p.name}: something frees the GPU before the end")
    check("cleanup is the last cell and nothing frees the GPU automatically", not bad,
          str(bad))


def test_one_change_per_arm():
    """uid_K20 is the one declared exception at two changes."""
    bad = []
    for arm, spec in C.ARM_SPEC.items():
        n = len(spec["changes"])
        limit = 2 if arm == "uid_K20" else 1
        if n > limit:
            bad.append(f"{arm}: {n} changes")
    check("one named change per arm (uid_K20 declared at two)", not bad, str(bad))


def test_edge_dim_is_five_everywhere():
    bad = [a for a, s in C.ARM_SPEC.items() if 3 + len(s["entity_cols"]) != 5]
    check("edge_dim is 5 in every arm (placeholder relation preserved)", not bad, str(bad))


def test_receptive_fields():
    want = {"K5_both": 111, "uid_K20": 421, "no_edge_bias": 421, "K20_both": 1641,
            "no_time_terms": 421, "no_amount_ratio": 421, "no_relation_ind": 421}
    got = {a: C.receptive_field(a) for a in C.ARM_SPEC}
    check("receptive fields match v6e's 1+d+d^2 convention", got == want, str(got))
    check("canonical RF is 421 and three layers is 8421",
          C.receptive_field("no_edge_bias") == 421 and
          C.receptive_field("no_edge_bias", 3) == 8421)


def test_write_fence():
    import guards
    outside = [C.REPO / "results" / "x.csv",
               C.REPO / "protocol_v2" / "results" / "x.csv",
               C.CANONICAL_MAIN_DIR / "per_fold.csv",
               C.REPO / "paper" / "x.docx",
               Path("/tmp/x.csv")]
    refused = 0
    for p in outside:
        try:
            guards.assert_writable(p)
        except guards.LockedArmError:
            refused += 1
    check("the write fence refuses every path outside edge_k/results",
          refused == len(outside), f"{refused}/{len(outside)} refused")
    inside = guards.assert_writable(C.RESULTS / "arms" / "x" / "y.csv")
    check("the write fence permits edge_k/results", inside is not None)


def test_analysis_declares_what_the_plan_says():
    import analysis_settings as A
    check("threshold is 2 SD, not 1", A.THRESHOLDS["decision_threshold"] == 0.023,
          str(A.THRESHOLDS["decision_threshold"]))
    check("wilcoxon is declared descriptive only",
          "DESCRIPTIVE ONLY" in A.WILCOXON["STATUS"])
    check("the four-seed mean is the primary baseline for A, B and D",
          all(A.BASELINES[a]["primary"] == "four_seed_mean"
              for a in ("K5_both", "K20_both", "no_edge_bias")))
    check("uid_K20 is read against uid_only first",
          A.BASELINES["uid_K20"]["primary"] == "uid_only_seed42")
    check("sign convention is declared once",
          "arm minus baseline" in A.SIGN_CONVENTION["rule"])


def main():
    print("edge_k static checks\n")
    for fn in (test_every_arm_has_a_notebook, test_notebooks_are_unrun,
               test_no_restated_hyperparameters, test_epoch_budget_is_200_everywhere,
               test_scaffolded_arms_are_not_runnable, test_locked_names_refused,
               test_path_order_handled, test_cleanup_is_manual,
               test_one_change_per_arm, test_edge_dim_is_five_everywhere,
               test_receptive_fields, test_write_fence,
               test_analysis_declares_what_the_plan_says):
        fn()
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("static checks passed")


if __name__ == "__main__":
    main()
