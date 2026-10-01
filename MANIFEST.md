# Manifest

Every result file under `results/`, `protocol_v2/results/` and `edge_k/results/`, the number it
produces, and where that number appears in the paper. Each mean was recomputed from the file
itself, not copied from the manuscript. Last verified against manuscript v7a on 2026-09-08.

Average precision is the step-wise estimator of Equation 3. All per-fold files carry 17 rows,
one per retained test week, except the protocol arms noted below.

> **Read this first.** The paper draws on **three** designated sets of results, produced in
> three sittings. Nothing else in the tree is current.
>
> | sitting | directory | what the paper takes from it |
> |---|---|---|
> | August, canonical | `results/rerun_20260826_114717_m5/` | every reported TRACE-GNN number: Tables 4 and 5, Figures 4, 7, 8 and 9, Sections 5.1, 5.2, 5.4, 5.5 |
> | September, protocol | `protocol_v2/results/` | Tables 6 and 7, Figures 5 and 6, Sections 5.3 and 6.1, and the protocol figures in the Conclusion |
> | September, edge and K | `edge_k/results/` | the edge-encoding ablation and the neighbour-cap sweep in Section 5.4, Table 8, Figure 8's sixth row |
>
> The older files under `results/per_fold/`, `results/architecture/`, `results/convergence/`,
> `results/budget/` and **`results/protocol/`** are earlier executions, kept for provenance and
> **superseded**. The gradient-boosted baseline is the exception: it is deterministic under a
> fixed seed, was not re-run, and is still read from `results/per_fold/`.
>
> A per-number index, including how each figure is computed rather than only which file holds
> it, is in `paper/READING_MAP.md`.
>
> **Naming convention.** As of 2026-09-08 every superseded result file carries a
> `_SUPERSEDED` suffix, following `protocol_v2/results/arms/P3_acausal_SUPERSEDED_smoke_*`.
> Sixteen files were renamed. A file without that suffix under `results/per_fold/`,
> `results/budget/` or any current directory is current — the two directories that mix
> current and superseded files are `results/per_fold/`, where the four `xgb_*` files and
> `fold_fraud_report.csv` are current, and `results/budget/`, where
> `scores_xgb_11_tuned.csv` is current.
>
> **Two categories of reference were left naming the pre-rename paths, on purpose.** The
> analysis documents under `paper/` — `v6b_comment_responses_and_run_inventory.md`,
> `P3_substitution_audit.md`, `FINAL_RUN_PLAN.md`, `EDGE_K_WORK_PLAN.md` — and
> `protocol_v2/README_PACKAGE.md` record what the tree looked like when they were written;
> rewriting them would falsify a record. And the notebooks that **produced** these files
> (`08b`, `12`, `13`, `15`) still write the pre-rename names, so **re-running one would
> recreate an unmarked superseded file.** No script that *reads* a renamed file was left
> stale.

---

## results/rerun_20260826_114717_m5 — the reported results

Produced by `rerun_canonical.py`, driven from `notebooks/16_canonical_rerun.ipynb`. One
execution per arm, with per-transaction scores and per-epoch validation curves retained for
every fold, so Table 4, Figure 7(b) and Figure 8 all describe the same run.

### canonical/<arm>/

| Arm | `per_fold.csv` mean AP | Fold sd | Appears as |
|---|---|---|---|
| `main` | 0.8551 | 0.0898 | TRACE-GNN headline, Table 4, Table 8's reference row, Figures 4, 7(b), 8, 9 |
| `uid_only` | 0.7559 | 0.0825 | Cardholder-only arm, Table 5 |
| `maxagg` | 0.8361 | 0.0980 | Maximum-pool aggregation, Section 5.4, Table 8, Figure 8 |
| `depth3` | 0.8522 | 0.0997 | Third message-passing layer, Section 5.4, Table 8, Figure 8 |

Each arm folder holds `per_fold.csv` (17 rows), `scores.csv` (338,456 rows, 1,276 positives),
`val_curves.csv` (per-epoch validation AP) and `summary.json`.

The headline figures follow by subtraction against the tuned tree at 0.6361. **Each Δ is the
mean of the seventeen paired fold differences, not the difference of the two rounded means**,
so a hand subtraction of the values in this table may differ in the last digit.

    0.8551 - 0.6361 = 0.2191   total advantage over the tuned tree
    0.7559 - 0.6361 = 0.1198   cardholder history as structure
    0.8551 - 0.7559 = 0.0992   merchant relation
    0.8361 - 0.8551 = -0.0191  maximum-pool aggregation
    0.8522 - 0.8551 = -0.0029  third message-passing layer
    0.8177 - 0.8551 = -0.0374  edge encoding removed (edge_k, see below)

Ratios stated in Sections 5.4 and 6.2 are computed from these unrounded differences, never
from the rounded means:

    0.0992 / 0.0374 = 2.6   merchant relation against the largest architectural effect
    0.1198 / 0.0374 = 3.2   cardholder structure against the same

### repeats/repeat_NN/main/

Three further executions of the `main` arm at seeds 43, 44 and 45, identical in every other
respect. With the canonical run at seed 42 these give the run-to-run estimate quoted throughout
Section 5.4 and in the Abstract.

| Seed | Mean AP |
|---|---|
| 42 (canonical) | 0.8551 |
| 43 | 0.8328 |
| 44 | 0.8472 |
| 45 | 0.8537 |

    mean 0.8472, standard deviation 0.0102, range 0.8328 to 0.8551

`per_fold_variation.csv` gives the per-fold standard deviation across those four runs: median
0.028, maximum 0.080 on week 10.

**Every variance scale and threshold in the paper derives from that 0.0102.** All four are
stated in Section 5.4's repeats paragraph; this is the arithmetic:

| quantity | value | derivation |
|---|---|---|
| single-run SD | 0.0102 | `repeat_summary.csv`, four seeds, ddof=1 |
| SD of a difference between two single runs | 0.0145, printed as 0.014 | √2 × 0.0102. Figure 8's shaded band |
| SD of a difference against the four-run mean | 0.0114, printed as 0.011 | √1.25 × 0.0102 |
| resolution against the reported run | **0.029** | 2 × 0.0145 |
| resolution against the four-run mean | **0.023** | 2 × 0.0114. The pre-declared value in `edge_k/analysis_settings.py` |

The declared rule is *two standard deviations of a difference*; 0.023 and 0.029 are that one
rule instantiated against two different baselines. Table 8's Δ values are measured against the
reported run, so 0.029 is the resolution that applies to them. The edge-encoding arm clears
both: 0.0374 against the reported run and 0.0295 against the four-run mean.

**Repeats are never cited as results.** They exist only to bound variation. Their AUROC means
are the exception the paper does quote: 0.9863, 0.9879, 0.9841 and 0.9828, a standard deviation
of 0.0023, which is Section 5.1's "four repetitions vary by 0.002 in this statistic". Note that
the *range* of those four is 0.005, so that sentence should be read as a standard deviation.

### derived/

| File | Behind |
|---|---|
| `paired_tests.csv` | the p-values and fold counts in Tables 4 and 5. **Its two Section 5.4 rows are superseded — see below** |
| `budget_sweep_11feat_matched.csv` | Section 5.5 and all of Figure 9 |
| `scores_gnn_11_converged.csv`, `scores_xgb_11_tuned.csv` | inputs to the sweep, one row per scored transaction. Also the source of Table 4's ROC-AUC row, which is computed from the scores rather than stored in `per_fold.csv` |

**`paired_tests.csv`'s architecture rows are superseded and the file was not regenerated.**
It was written with `wilcoxon(a, b)` at SciPy's default `method='auto'`, which falls back to
the normal approximation when tied ranks are present, and with `wins=(d > 0).sum()`, which
counts a floating-point artefact as a win. The paper declares an exact test with differences
below 1e-12 treated as ties (Table 7's note). Under the declared rule:

| contrast | file says | correct | why |
|---|---|---|---|
| `maxagg - main` | wins 7, p 0.234 | 7 arm wins, **9** reference wins, **1 tie**, p **0.252** | normal approximation |
| `depth3 - main` | wins 9, p 0.963 | **8** arm wins, **8** reference wins, **1 tie**, p **0.980** | `depth3` and `main` differ by 2e-16 at week 24; `(d > 0)` counted it as a win |

The three information-channel rows are unaffected: `main - xgb_tuned` +0.2191 17/17,
`uid_only - xgb_tuned` +0.1198 16/17, `main - uid_only` +0.0992 17/17.

The generating code in `rerun_canonical.py` has been fixed and now emits `wins`, `losses`
and `ties` from the exact, tie-dropped test. **The file itself was not regenerated, because
that means re-running the canonical arms — roughly 35 GPU-hours — and it is a locked run
output.** Manuscript v7a carries the corrected values, which `figures/make/fig_5_4.py`
recomputes and prints on every run.

### run-level files

| File | Contents |
|---|---|
| `canonical_summary.csv` | one row per arm: mean AP, epochs, hours, segments |
| `COMPARISON.md` | this run against the previously published values, and what moved |
| `run_manifest.json` | versions, dataset sha256, graph facts, and the execution segments |
| `fold_composition.csv` | per-fold train/val/test sizes and fraud counts, Table 1 and Figure 7(a) |

**One provenance note, recorded in `run_manifest.json`.** The `depth3` arm was trained in two
segments. Training exhausted the memory of a 48 GB device on week 24, the largest fold, and was
resumed for that fold alone, which began from a fresh generator state. Same configuration, same
seed, drawn at a different point in the stream. Section 4.4 of the paper states this.

---

## results/per_fold — the baseline, and superseded arms

**Current.** The gradient-boosted baseline is not re-run.

| File | Mean AP | Appears as |
|---|---|---|
| `xgb_11_tuned.csv` | 0.6361 | Tuned tree baseline, Tables 4 and 5, Figures 4, 8, 9 |
| `xgb_11_defaults.csv` | 0.4936 | "At library defaults the tree attains 0.494", Section 4.3 |
| `xgb_11_fixed.csv` | 0.6153 | Fixed-configuration tree, Section 4.3 |
| `xgb_11_tuned_params.csv` | — | Per-fold hyperparameter selections, Appendix B |
| `fold_fraud_report.csv` | — | Fold composition. Identical to `fold_composition.csv` above |

**Superseded.** Earlier executions of the graph arms, retained for provenance only.

| File | Mean AP | Superseded by |
|---|---|---|
| `gnn_11_converged_SUPERSEDED.csv` | 0.8427 | `canonical/main/per_fold.csv` at 0.8551 |
| `gnn_11_uid_only_converged_SUPERSEDED.csv` | 0.7731 | `canonical/uid_only/per_fold.csv` at 0.7559 |
| `converged_wilcoxon_SUPERSEDED.csv` | — | `derived/paired_tests.csv` |

## protocol_v2/results — the reported protocol study

Produced 2026-09-02 by `protocol_v2/assemble.py`, driven from
`protocol_v2/notebooks/A5_assemble.ipynb`. **Four arms retrained at 200 epochs; the fifth,
`P3_rolling`, is the canonical Section 5.1 run reused rather than retrained**, which is why
Section 5.3 can say it holds the model configuration fixed. `protocol_v2/results/canonical_p3_lock.json`
records the reuse.

Why it exists: the earlier study under `results/protocol/` was trained at **75 epochs** while
the canonical run used 200. `CosineAnnealingLR` anneals over `T_max = epochs`, so a 75-epoch
run follows a different learning-rate path from the first step — two configurations, not two
lengths of one.

### derived/

| File | Behind |
|---|---|
| `per_week_all_arms.csv` | Table 6, Figure 5, and every protocol contrast. Per-week AP, AUROC, P@k, R@k, F1, MCC for all five arms |
| `paired_tests.csv` | Table 7. Also holds the AUROC, P@k and R@k contrasts, which the paper does not tabulate |
| `leak_decomposition.csv` | the same six AP contrasts, labelled by effect, behind Sections 5.3 and 6.1 |
| `factorial_2x2.csv` | Figure 6(a) and the non-additivity argument |
| `protocol_ladder.csv` | Table 6's rows as assembled, with the `source` column recording which arm was retrained and which was reused |
| `non_additive_excess_per_week.csv` | the per-week excess behind "positive in 14 of the 16 weeks" |

### Arm means, over the 16 weeks common to all five arms — Table 6

| Arm | Mean AP | Mean AUROC | Protocol |
|---|---|---|---|
| `P1_leaky` | 0.9742 | 0.9993 | Random split, unconstrained graph |
| `P1_causal` | 0.9708 | 0.9994 | Random split, causal graph |
| `P2_cutoff` | 0.7501 | 0.9746 | Single temporal cutoff, causal graph |
| `P3_acausal` | 0.9405 | 0.9983 | Rolling origin, unconstrained graph |
| `P3_rolling` | 0.8461 | 0.9854 | Rolling origin, causal graph. The protocol used everywhere else |

    AP range    0.9742 - 0.7501 = 0.224   Sections 5.3, 6.1 and the Conclusion
    AUROC range 0.9994 - 0.9746 = 0.025   the AUROC maximum is P1 causal, not P1 leaky

The AUROC range is 0.024727 unrounded. Subtracting Table 6's three-decimal 0.999 and 0.975
gives 0.024 and is the reason v6e printed that value; Section 5.3 now quotes the endpoints to
four decimals so the arithmetic on the page matches the stated range.

**Fold bases differ and the difference matters.** The randomly split arms cover weeks 8 to 23,
because a 20% sample of week 24 leaves too few positives to score. Contrasts involving a random
split therefore use the sixteen weeks common to all five arms. Contrasts between `P2_cutoff` and
`P3_rolling`, and between `P3_acausal` and `P3_rolling`, use all seventeen. Table 7's caption
records this. Consequences worth knowing before recomputing anything:

- `P3_acausal - P3_rolling` is **0.089 over seventeen weeks**, the figure Table 7 reports, and
  **0.094 over the matched sixteen**, the figure Sections 5.3 and 6.1 report. Both appear in the
  paper and they are not the same quantity.
- `P2_cutoff - P3_rolling` is **−0.090 over seventeen weeks**. Section 5.3 reports the
  seventeen-week means 0.765 and 0.855 alongside it.
- The non-additive excess, `(P1lk − P3ro) − [(P1lk − P1ca) + (P1lk − P3ac)]`, is **0.091** over
  the sixteen common weeks, positive in 14 of them, p = 0.00058.

**One difference in tie handling, which a verifier will hit.** `paired_tests.csv` and
`leak_decomposition.csv` drop only differences that are **exactly** zero. The manuscript
declares the threshold as 1e-12 (Table 7's note). Two rows differ under the two rules, and the
manuscript is correct under its own:

| contrast | stored | manuscript, at 1e-12 |
|---|---|---|
| `P1_leaky - P1_causal` | 1 tie, p 0.5614 | **2 ties, p 0.58301** — week 15 differs by 2.2e-16 |
| `P2_cutoff - P3_rolling` | 0 ties, p 0.00038 | **1 tie, p 0.00031** — week 24 differs by 2.2e-16 |

All four other rows are identical under both rules. The Δ values are unaffected, since they are
means over the full week set in either case.

**Graph facts for the unconstrained arm**, in `protocol_v2/results/arms/P3_acausal/arm_manifest.json`
under `graph_facts`: 9,999,876 edges, 20.0 per node, 5,010,950 of them pointing from a later
transaction to an earlier one, `future_frac` 0.5011. This is Section 5.3's "50.1% of the edges".

## results/protocol — superseded, and correctly described below

The **75-epoch** five-arm study. Retained for provenance; every number in it has been replaced
by `protocol_v2/results/`. It is still the eleven-feature study, and a sixteen-feature version
of the same filename exists in the wider project tree and is superseded twice over. Loading the
wrong one silently produces a plausible but incorrect Figure 5, which happened once during the
revision. **The current producer of Figures 5 and 6 is
`protocol_v2/figures/make/fig_5_3_v2.py`**, which asserts every arm against Table 6 before
drawing. The older `figures/make/fig_5_3_SUPERSEDED.py` asserts the 75-epoch study's values
instead — 0.966 / 0.963 / 0.741 / 0.839 / 0.946 — and prefers a directory name that no longer
exists, so it raises rather than drawing. An earlier version of this manifest credited it
with guarding against Table 6; it guards against the superseded table.

| Arm | Mean AP, 16 common weeks | Superseded by |
|---|---|---|
| `P1_leaky` | 0.9655 | `protocol_v2` at 0.9742 |
| `P1_causal` | 0.9628 | `protocol_v2` at 0.9708 |
| `P2_cutoff` | 0.7409 | `protocol_v2` at 0.7501 |
| `P3_acausal` | 0.9463 | `protocol_v2` at 0.9405 |
| `P3_rolling` | 0.8395 | `protocol_v2` at 0.8461 |

**If you have a copy of this manifest from before 2026-09-08, its protocol section described
these files as current.** The contrasts it quoted were correct for *this* run and are wrong for
the paper: 0.107 for `P3_acausal - P3_rolling` and 0.123 for `P1_causal - P3_rolling` on the
sixteen common weeks, and 0.103 for `P3_rolling - P2_cutoff` on all seventeen. The paper's
values are 0.094, 0.125 and 0.090. Both sets are reproducible; they describe different
executions.

The same applies to the note this section used to carry about Table 7's graph-effect p-value,
which read p = 0.326 against 0.363 in `leak_decomposition_perweek_SUPERSEDED.csv`. Both belong to the
75-epoch run. The reported contrast is now p = 0.58301 from `protocol_v2`, and the tie-handling
note above replaces it.

`factorial_2x2_SUPERSEDED.csv`, `leak_decomposition_perweek_SUPERSEDED.csv` and
`protocol_ladder_perweek_SUPERSEDED.csv` here are the 75-epoch derived summaries. Their
`protocol_v2` counterparts drop the `_perweek` suffix and carry no `_SUPERSEDED` marker.

## edge_k/results — the edge-encoding ablation and the neighbour-cap sweep

Produced 2026-09-05 to 2026-09-06, four arms, 68 folds, 35.0 GPU-hours, one continuous
execution each. Every arm holds `per_fold.csv` (17 rows), `scores.csv`, `val_curves.csv`,
`run_record.json`, `arm_manifest.json` and saved weights. The canonical arms were re-hashed at
the start and end of every arm and did not move; `canonical_lock.json` and `analysis_lock.json`
record it. **`analysis_lock.json` was written before the first arm trained**, which is what
makes the 0.023 threshold and the baseline choice pre-declared rather than chosen with results
in hand.

| Arm | Mean AP | Δ vs canonical seed 42 | Appears as |
|---|---|---|---|
| `no_edge_bias` | 0.8177 | −0.0374 | Section 5.4, Table 8, Figure 8's sixth row |
| `K5_both` | 0.8361 | −0.0191 | Section 5.4's neighbour cap, Table 8 |
| `K20_both` | 0.8549 | −0.0003 | Section 5.4's neighbour cap, Table 8 |
| `uid_K20` | 0.7317 | −0.1234 vs `main`; −0.0241 vs `uid_only` | the budget-matched control, Section 5.2 prose only |

`no_edge_bias` and `K5_both` are the pair to be careful with: `K5_both` at 0.836064 and the
canonical `maxagg` at 0.836064 coincide to six decimal places on seventeen folds each. They are
separately trained configurations and neither value is a transcription of the other. Table 8's
caption says so.

**`uid_K20` is reported in prose and must not enter Table 5.** Table 5's two increments sum to
0.219 by construction; the budget-matched contrast starts from `uid_K20` at 0.7317, not
`uid_only` at 0.7559, so inserting 0.1234 would break the sum (0.1198 + 0.1234 = 0.2432 against
a true 0.2190).

**Two files here carry a wrong wall-clock ratio.** `FINAL_ANALYSIS.md` and `SUMMARY_all_arms.md`
both state that K = 20 costs "6.7× the wall time". That is K = 20 against K = 5 (21.07 / 3.15).
Against the reported K = 10 configuration the ratio is **2.41×** (21.07 / 8.74), which is what
the paper uses. Everything else in both files reproduces.

## results/architecture — superseded

| File | Mean AP | Superseded by |
|---|---|---|
| `maxagg_11_SUPERSEDED.csv` | 0.8237 | `canonical/maxagg/per_fold.csv` at 0.8361 |
| `depth3_11_SUPERSEDED.csv` | 0.8568 | `canonical/depth3/per_fold.csv` at 0.8522 |
| `arch_11_summary_SUPERSEDED.csv` | reference 0.8331 | `canonical_summary.csv` |
| `arch_11_wilcoxon_SUPERSEDED.csv` | depth-3 **+0.0237** | `derived/paired_tests.csv` and Table 8 at −0.0029 |

The earlier depth arm read as a gain of +0.014. On the canonical run it reads −0.0029, and the
sign flip between two executions of one configuration is the clearest evidence in the project
for why a single execution cannot resolve differences of this size. `maxagg` is **not
parameter-matched**: fusing max-pool with the weighted sum widens the output projection and adds
roughly 12% more parameters.

Of the three architecture variants the paper now reports, only `no_edge_bias` produces a
difference beyond the resolution a single execution affords. `maxagg` and `depth3` do not.

## results/budget — superseded

`budget_sweep_11feat_matched_SUPERSEDED.csv` here is the earlier sweep; all six of its deltas
differ from the reported ones. The reported sweep is
`rerun_20260826_114717_m5/derived/budget_sweep_11feat_matched.csv`, regenerated from the
canonical scores. `scores_gnn_11_converged_SUPERSEDED.csv` is the 0.8427 run's
per-transaction output.

**`scores_xgb_11_tuned.csv` is NOT renamed.** It is the tree's per-transaction output, the
tree is deterministic and was not re-run, and this file is md5-identical to the copy in the
canonical `derived/` folder. It is current and the sweep reads it.

## results/convergence — superseded

`val_curves_converged_SUPERSEDED.csv` was the source of Figure 7(b) in earlier drafts.
`gnn_11_converged_run1_SUPERSEDED.csv` at 0.8331 was the first execution of the reported
configuration.

**The provenance defect these files record is now closed.** Earlier drafts plotted Figure 7(b)
from an execution other than the one Table 4 reported, because no validation curves had been
persisted for the reported run. The canonical run sets `val_curve_path` for every arm, so
`canonical/main/val_curves.csv` and `canonical/main/per_fold.csv` describe the same execution.

## Not included

- `data/` and `prepared/`, about 723 MB, derived from the public Kaggle corpus. The README gives
  the rebuild.
- The per-transaction score files for the five protocol arms, about 31 MB, whose aggregates are
  already in `per_week_all_arms.csv`.
- `foldckpt.csv` resume checkpoints, which carry no information beyond `per_fold.csv`.
- Superseded studies, including the sixteen-feature protocol run, the SWA-tuned arms and the
  fairness arms, none of which appear in the paper.
