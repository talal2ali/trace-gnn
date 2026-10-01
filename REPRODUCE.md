# Reproducing the paper

Every number, table and figure, with the command that produces it.

> **This is the code-only release: no result files are shipped.** Tier 1 below rebuilds every
> table and figure from result files *without retraining* — but you must have those files, either
> from the results archive or from your own Tier 2 run. Every path in Tier 1 is given so that a
> reviewer holding the archive can follow it directly.

**Tier 1 needs no GPU and takes minutes**, given result files. **Tier 2 retrains** and takes
roughly 45 GPU-hours for the four canonical arms, 21 for the three repeats, 35 for the four
`edge_k` arms and a further day for the four retrained `protocol_v2` arms.

---

## Tier 1 — rebuild every table and figure, no GPU

    pip install -r requirements.txt

`CANON` below is the canonical results folder:

    CANON=results/rerun_20260826_114717_m5

### Figures

| Figure | Command |
|---|---|
| 4 | `python figures/make/fig_5_1.py results/ figures/ --canonical $CANON` |
| 5, 6 | `python protocol_v2/figures/make/fig_5_3_v2.py protocol_v2/results/ figures/` |
| 7, 8 | `python figures/make/fig_5_4.py results/ figures/ --canonical $CANON` |
| 9 | `python figures/make/fig_5_5.py results/ figures/ --canonical $CANON` |
| 1, 2, 3 | `python figures/make/fig_method.py figures/` |

Each script prints the arm means it read and cross-checks them against
`canonical_summary.csv` before drawing, so a wrong or truncated input aborts rather than
producing a plausible but incorrect figure.

Figures 5 and 6 take no `--canonical` flag: their arms live in `protocol_v2/results/`, which is
its own package. Four of the five were retrained at 200 epochs; the fifth, `P3_rolling`, is the
canonical run reused rather than retrained, which is what lets Section 5.3 hold the model fixed
while varying the protocol.

Figure 8 needs `--edge-arm` only if the edge-encoding arm is not at its default location,
`edge_k/results/arms/no_edge_bias/per_fold.csv`; the script searches there relative to both
ROOT and its own directory.

### Tables

| Table | Source |
|---|---|
| 1 | `$CANON/fold_composition.csv` |
| 2 | Section 4.3, no result file |
| 3 | Section 4.4, no result file |
| 4 | `$CANON/canonical/*/per_fold.csv` and `$CANON/derived/paired_tests.csv` |
| 5 | same two files |
| 6 | `protocol_v2/results/derived/protocol_ladder.csv` |
| 7 | `protocol_v2/results/derived/leak_decomposition.csv` and `factorial_2x2.csv` |
| 8 | `$CANON/canonical/{main,maxagg,depth3}/per_fold.csv` and `edge_k/results/arms/{no_edge_bias,K5_both,K20_both}/per_fold.csv`. Its Δ, fold counts and p-values are recomputed and printed by `figures/make/fig_5_4.py` |
| A.1 | Appendix A, no result file |
| B.1 | `results/per_fold/xgb_11_tuned_params.csv` |

### Checking the reported numbers

    python rerun_canonical.py --resume --out $CANON --check-only

Reports what each arm holds and confirms the three append-mode files agree, without training or
writing anything. It **needs neither the corpus nor a GPU nor PyTorch** *provided the folder it
is pointed at contains `fold_composition.csv`*: that file supplies the fold list, and the arms
are then inspected directly. If the folder is absent — as in the code-only release, which ships
no results — the command falls back to locating `prepared/` and aborts if it cannot. There is
nothing for it to check in that case anyway.

    python rerun_canonical.py --resume --out $CANON

Retrains nothing when every arm is complete, and rebuilds `canonical_summary.csv`, `derived/`
and `COMPARISON.md` from the shipped scores. Those regenerated files are byte-identical to the
ones in the repository. This command does load the corpus, because it rebuilds the fold set and
graph to re-run the assertions, so it needs `prepared/`.

---

## The baseline is shipped, not re-run

`results/per_fold/xgb_11_tuned.csv` and `results/budget/scores_xgb_11_tuned.csv` come with this
repository. The gradient-boosted tree is deterministic and was not re-fitted for the reported
run, so `rerun_canonical.py` reads it from a read-only search root (`--search-root`, default
`./results`) and will refuse to start if it is missing. You do not need to do anything for this
to work; it is noted so the two files are not mistaken for shipped results.

## Tier 2 — retrain

### Data

The corpus is not in this repository. It is public.

1. Download the Sparkov dataset from
   <https://www.kaggle.com/datasets/kartik2112/fraud-detection>.
2. `python src/prepare_sparkov.py --raw <download_dir> --out prepared/`

**Pass the location explicitly with `--prepared <path>`.** Every entry point accepts it:
`rerun_canonical.py`, `resume_canonical.py`, `protocol_v2/runner.py` and `edge_k/runner.py`.
If it is not passed, they fall back to searching `.`, `..` and `../../prepared` relative to
this directory, which is where `prepared/` happened to sit in the original working tree.
That fallback is a convenience, not a requirement, and it will not find your copy unless you
have reproduced that layout. Prefer `--prepared`.
`run_manifest.json` records the sha256 of the file used, so you can confirm you have the same one.

### Validate before spending the hours

    python rerun_canonical.py --dry-run

Loads the frame, builds the folds, builds the graph and runs every assertion, then stops. It
also samples one training batch from the largest fold and projects peak VRAM per arm. Expect
500,000 rows, 11 features, 17 folds, about 9.9 million edges at 19.8 per node, and 54
contemporaneous edges. If the projection exceeds free memory the run will very likely fail on
the last folds; move the deep arm to an emptier device.

### Train

    # the three two-hop arms, roughly 17 h
    python rerun_canonical.py --resume --out results/my_rerun --arms main,uid_only,maxagg --device cuda:0

    # the three-hop arm, roughly 33 h. Give it a device of its own if you have one
    python rerun_canonical.py --resume --out results/my_rerun --arms depth3 --device cuda:1

    # three repeats of the main arm for the variance estimate, roughly 21 h
    python rerun_canonical.py --resume --out results/my_rerun --arms main --repeats 3 --device cuda:0

Or run all of it from `notebooks/16_canonical_rerun.ipynb`, which is the same commands with the
progress streamed inline.

**Every command is idempotent.** `--resume` starts a fresh run if the folder is absent, trains
only the missing folds if it is partial, and returns immediately if it is complete. Locking is
per arm, so different arms may run concurrently on different devices. A crash costs one fold,
not the arm.

### Then

    python rerun_canonical.py --resume --out results/my_rerun          # rebuilds derived/ and COMPARISON.md

`COMPARISON.md` tabulates your run against the published values arm by arm and states which, if
any, moved by more than 0.001.

---

## What will not reproduce exactly

**Message passing is not deterministic.** Neighbour contributions are accumulated with scatter
operations that are not associative in floating point on GPU, so a fixed seed does not give
bit-identical results. The four repeated runs shipped here bound the consequence: mean average
precision varies with a standard deviation of 0.010 across seeds 42 to 45. Expect a re-run to
land within roughly 0.02 of 0.855, not on it.

**The seed is set once per run, not per fold**, so a run resumed from a checkpoint does not give
its remaining folds the generator state an uninterrupted run would. `depth3` in the shipped
results was resumed for its final fold after an out-of-memory failure; `run_manifest.json`
records the segments.

**The paired test is the exact Wilcoxon with numerical ties dropped**, the rule stated in Table
7's note and frozen in `edge_k/analysis_settings.py`. `src/evaluation.py`,
`src/budget_sweep_11.py`, `rerun_canonical.py` and `figures/make/fig_5_4.py` all implement it.
SciPy's default `method='auto'` silently falls back to the normal approximation when a tie is
present, and a tie is present in every budget-constrained metric because week 24's seven
positives sit inside every alert budget. If you recompute a p-value and get a larger one than
the paper prints, check which method you used before concluding anything.

**Tier 1 numbers are exact; Tier 1 image files are not.** Every value derived from the shipped
scores — every table, the budget sweep, the paired tests, `canonical_summary.csv`,
`COMPARISON.md` — reproduces byte-for-byte. The figures reproduce the same content, but the PNG
bytes depend on your Matplotlib and font versions, so `md5sum` on a regenerated `.png` will
differ from the shipped one even when the plotted data is identical. Compare the numbers each
script prints, not the image hashes.

---

## Verifying the causality guarantee

    PYTHONPATH=src python -m pytest tests/ -v

Ten tests covering the invariant that no edge points from a later transaction to an earlier one.
This is the property the paper's leakage argument rests on, and it is **the one check in this
repository that needs neither the corpus nor a result file**, so it runs immediately after
cloning.

The other two suites need `PYTHONPATH` set as well:

    PYTHONPATH=src:protocol_v2 python -m pytest protocol_v2/tests/ -q    # 30
    PYTHONPATH=src:edge_k     python -m pytest edge_k/tests/ -q          # 21

Two `protocol_v2` tests are environment-dependent and fail outside the machine that produced
the results: `test_lockfile_records_the_real_canonical_directory` compares a recorded absolute
path against the current tree, and `test_no_notebook_has_been_executed` is false in any shipped
archive. Both are expected failures, not regressions.
