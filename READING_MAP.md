# Reading map — where every number in TRACE-GNN v7a comes from

A provenance index for `paper/TRACE-GNN_manuscript_v7a.docx`. For each table, figure and stated
number: the value, the file it comes from, and **how it is computed** — the formula, not only
the filename.

Built 2026-09-08. Every entry was recomputed from the primary file at the time of writing, not
copied from the manuscript or from `MANIFEST.md`.

Companions: `MANIFEST.md` indexes the result files themselves; `paper/v7a_CHANGELOG.md` records
what changed from v6e and why; `figures/FIGURE_PROVENANCE.md` records which figures reproduce.

### How to maintain this

Rows are keyed by the manuscript location. **If a number changes, the row to update is the one
whose location matches** — and if that number appears in more than one place, §7 lists every
quantity the manuscript states twice, which is where a change has to be propagated. Ratios in §8
are all derived, so they change whenever their inputs do.

Two conventions hold throughout and explain most apparent mismatches:

1. **Δ values are means of paired per-fold differences, not differences of rounded means.** A
   hand subtraction of two displayed values may differ in the last digit. Table 4's caption
   states this.
2. **Paired tests are the exact Wilcoxon with differences below 1e-12 dropped as ties.** Table
   7's note states this. Two stored files use exactly-zero instead; see §9.

---

## 1. The three sittings

The paper's arms were not all trained together, and this is the provenance a reader will want in
one place.

| sitting | when | directory | arms | what the paper takes |
|---|---|---|---|---|
| **A — canonical** | 2026-08-26 | `results/rerun_20260826_114717_m5/` | `main`, `uid_only`, `maxagg`, `depth3`, plus three repeats of `main` at seeds 43–45 | Tables 4, 5, Table 8's reference and two architecture rows, Figures 4, 7, 8, 9, and §§5.1, 5.2, 5.4, 5.5 |
| **B — protocol** | 2026-09-02 | `protocol_v2/results/` | `P1_leaky`, `P1_causal`, `P2_cutoff`, `P3_acausal` retrained at 200 epochs; **`P3_rolling` is sitting A's `main` reused, not retrained** | Tables 6, 7, Figures 5, 6, §§5.3, 6.1, and the protocol figures in the Conclusion |
| **C — edge and K** | 2026-09-05/06 | `edge_k/results/` | `no_edge_bias`, `K5_both`, `K20_both`, `uid_K20` | Table 8's edge and neighbourhood rows, Figure 8's sixth row, §5.4's ablation and sweep, §5.2's budget-matched control |

**Why the reuse in sitting B matters.** `P3_rolling` being sitting A's `main` is what allows §5.3
to say it holds the model configuration fixed while varying the protocol. Recorded in
`protocol_v2/results/canonical_p3_lock.json`.

**Why sitting B exists at all.** The earlier protocol study, now `results/protocol/`, was trained
at 75 epochs while the canonical run used 200. `CosineAnnealingLR` anneals over `T_max = epochs`,
so a 75-epoch run follows a different learning-rate path from its first step. Two configurations,
not two lengths of one.

**Cross-sitting integrity.** Sitting C re-hashed sitting A's canonical arms at the start and end
of each of its four arms; the digests did not move. `edge_k/results/canonical_lock.json`.
Sitting C's `analysis_lock.json` was written *before* its first arm trained, which is what makes
the 0.023 threshold and the baseline choice pre-declared rather than chosen with results in hand.

**The one within-sitting irregularity.** Sitting A's `depth3` arm was trained in two segments:
week 24, the largest fold, exhausted a 48 GB device and was resumed for that fold alone, from a
fresh generator state. Same configuration and seed, drawn at a different point in the stream.
§4.4 discloses it; `run_manifest.json` records it under `execution_segments`.

---

## 2. Variance scales and thresholds

Everything here descends from one number, the standard deviation across the four repeats.

| quantity | value | derivation | where stated |
|---|---|---|---|
| four repeat means | 0.8551, 0.8328, 0.8472, 0.8537 | `repeat_summary.csv`, seeds 42–45 | §5.4 |
| four-run mean | 0.8472 | mean of the above | §5.4 |
| **single-run SD** | **0.0102** | sd of the four, ddof=1. Exactly 0.010219 | §5.4, Abstract, Figure 8's caption |
| SD of a difference between two single runs | 0.0145, printed **0.014** | √2 × 0.0102 = 0.014453 | §5.4, Figure 8's caption and shaded band |
| SD of a difference against the four-run mean | 0.0114, printed **0.011** | √1.25 × 0.0102 = 0.011425 | §5.4 |
| **resolution, against the reported run** | **0.029** | 2 × 0.0145 | §5.4 ×3 |
| **resolution, against the four-run mean** | **0.023** | 2 × 0.0114. Pre-declared in `edge_k/analysis_settings.py` `THRESHOLDS` | §5.4 ×2 |
| per-fold SD across the four runs | median 0.028, max 0.080 at week 10 | `per_fold_variation.csv` | §5.4 |
| AUROC across the four runs | sd 0.0023 | from each run's `scores.csv`, fold-wise `roc_auc_score`, then sd of the four means | §5.1, as "vary by 0.002" |

**The rule is two standard deviations of a difference.** 0.023 and 0.029 are that one rule
instantiated against two baselines. Table 8's Δ values are against the reported run, so 0.029 is
the resolution that applies to them. The edge-encoding arm clears both — 0.0374 against the
reported run, 0.0295 against the four-run mean — which is why the conclusion does not turn on
the baseline choice.

**Multiples of the band, as §5.4 states them:** 0.0029/0.014453 = 0.2, 0.0191/0.014453 = 1.3,
0.0374/0.014453 = 2.6, 0.0992/0.014453 = 6.9 ("seven"), 0.1198/0.014453 = 8.3 ("eight").

**The paired test is descriptive.** `edge_k/analysis_settings.py` records
`WILCOXON.STATUS = "DESCRIPTIVE ONLY — DOES NOT GATE ANY DECISION"`, because with one execution
per condition the execution offset is common to all seventeen folds and is confounded with the
arm effect. §5.4 and Figure 8's caption report it as fold consistency. The manuscript's
significance language in §§5.1, 5.3 and 5.5 is untouched, for the reason in
`v7a_CHANGELOG.md` §2.12: those effects are 7 to 15 times the run-to-run scale, where §5.4's are
0.2 to 2.6 times it.

---

## 3. Tables

| # | subject | result files | produced by | sitting | fold basis |
|---|---|---|---|---|---|
| 1 | Fold structure and protocol constants | `rerun_.../fold_composition.csv` | prose from the file | A | 17 retained of 18 candidate |
| 2 | Search space for the gradient-boosted baseline | none — it is the search grid | prose; the grid is in `src/` and `results/per_fold/xgb_11_tuned_params.csv` shows what it selected | — | — |
| 3 | TRACE-GNN configuration, 21 rows | `rerun_canonical.py` `EXPECTED_*` and `run_manifest.json` `expected` | prose from the frozen config | A | — . v7a corrects the decoder row, marks K as a tested value, and adds a row for the edge-bias term |
| 4 | Matched-feature comparison, TRACE-GNN vs XGBoost | `canonical/main/per_fold.csv`, `results/per_fold/xgb_11_tuned.csv`, `derived/paired_tests.csv`; the ROC-AUC row from `canonical/main/scores.csv` and `derived/scores_xgb_11_tuned.csv` | prose | A | 17 |
| 5 | Sequential ladder, tree → cardholder-only → two relations | `canonical/main/per_fold.csv`, `canonical/uid_only/per_fold.csv`, `results/per_fold/xgb_11_tuned.csv` | prose | A | 17 |
| 6 | Performance under five protocols | `protocol_v2/results/derived/per_week_all_arms.csv`, assembled in `protocol_ladder.csv`. **Not** `results/protocol/per_week_all_arms_SUPERSEDED.csv`, which is the 75-epoch study | `protocol_v2/assemble.py` | B (+A for `P3_rolling`) | **16 common weeks, 8–23** |
| 7 | Paired differences in average precision | `protocol_v2/results/derived/paired_tests.csv` and `leak_decomposition.csv` | `protocol_v2/assemble.py` | B | **mixed: 17 for the two P3/P2 contrasts, 16 for the four involving a random split.** The table's `n` column states which |
| **8** | Sensitivity to architecture and neighbour cap | `canonical/{main,maxagg,depth3}/per_fold.csv` and `edge_k/results/arms/{no_edge_bias,K5_both,K20_both}/per_fold.csv` | recomputed and printed by `figures/make/fig_5_4.py` | A + C | 17 |
| A.1 | The eleven features | none — definitions | prose | — | — |
| B.1 | Per-fold tree configuration | `results/per_fold/xgb_11_tuned_params.csv` | prose from the file | — | 17 |

### Table 8, cell by cell

Δ against `canonical/main`, mean of the seventeen paired fold differences. "Folds led" counts
folds on which `main` leads, ties separate. p is the exact Wilcoxon over non-tied folds.

| row | Mean AP | Fold sd | Δ AP | Folds led | p |
|---|---|---|---|---|---|
| reference, two relations, K = 10 | 0.855 | 0.090 | — | — | — |
| edge encoding removed | 0.818 | 0.096 | −0.0374 | 13 / 17, 1 tie | 0.005 |
| max-pool fusion | 0.836 | 0.098 | −0.0191 | 9 / 17, 1 tie | 0.252 |
| third message-passing layer | 0.852 | 0.100 | −0.0029 | 8 / 17, 1 tie | 0.980 |
| K = 5 | 0.836 | 0.064 | −0.0191 | 12 / 17 | 0.089 |
| K = 20 | 0.855 | 0.104 | −0.0003 | 7 / 17, 1 tie | 0.821 |

Unrounded means: 0.855123, 0.817674, 0.836064, 0.852194, 0.836064, 0.854859. **Max-pool and
K = 5 agree to six decimals** — 0.8360637 and 0.8360640 — from separate trainings in different
sittings. Table 8's caption says so. Fold sds: 0.089793, 0.096040, 0.098020, 0.099675, 0.063942,
0.103810.

---

## 4. Figures

MD5s are of the PNG embedded in v7a (`word/media/imageN.png`) against `figures/`.

| # | subject | data source | script | reproduces? | v7a MD5 | `figures/` MD5 |
|---|---|---|---|---|---|---|
| 1 | rolling-origin protocol schematic | none, schematic | `fig_method.py` | **no**, hand-finished | `bf91fba0` | `7b4a3c7b` |
| 2 | causal graph construction schematic | none, schematic | `fig_method.py` | **no**, hand-finished | `4aa8c9c1` | `8f6c15d1` |
| 3 | TRACE-GNN architecture schematic | none, schematic | `fig_method.py` | **no**, redrawn for v6e | `01a3de33` | `1022aaa3` |
| 4 | per-fold AP, TRACE-GNN vs tree | `canonical/main/per_fold.csv`, `xgb_11_tuned.csv` | `fig_5_1.py` | **yes** | `f79ee0a8` | `f79ee0a8` |
| 5 | AP and AUROC under five protocols | `protocol_v2/.../per_week_all_arms.csv` | `protocol_v2/figures/make/fig_5_3_v2.py` | **no**, hand-finished | `6f94862e` | `62c3f7a9` |
| 6 | AP across the four P1/P3 arms, non-additivity | same, plus `factorial_2x2.csv` | `protocol_v2/figures/make/fig_5_3_v2.py` | **no**, hand-finished | `9b8868aa` | `55e97262` |
| 7 | (a) fold volumes and fraud counts (b) validation curves | `fold_composition.csv`, `canonical/main/val_curves.csv` | `fig_5_4.py` | **yes** | `6a9873d4` | `6a9873d4` |
| 8 | mean AP differences, six rows | the six per-fold files in Table 8 | `fig_5_4.py` | **yes, as of v7a** | `0d7700b8` | `0d7700b8` |
| 9 | precision and recall across alert budgets | `derived/budget_sweep_11feat_matched.csv` | `fig_5_5.py` | **yes** | `7e94cc29` | `7e94cc29` |

v6e's hand-finished Figure 8 is kept as `figures/F8_effect_sizes_v6e_handfinished.png`
(`0768f0ca`) as a styling reference. Detail in `figures/FIGURE_PROVENANCE.md`.

**Figure 8's intervals** are 95% percentile bootstrap over 10,000 resamples of the paired fold
differences, one generator seeded at 0, consumed in row order. Bounds: tree
[+0.180, +0.264], cardholder [+0.075, +0.169], merchant [+0.075, +0.126], depth
[−0.034, +0.026], max-pool [−0.054, +0.010], edge encoding [−0.063, −0.015]. Across seeds 0–5
no bound moves by more than 0.0027, the worst being max-pool's lower bound; the caption says
"at most 0.003".

---

## 5. Numbers by section

### Front matter

| value | where | source and formula |
|---|---|---|
| 0.64 → 0.86 | Highlights, Abstract | rounded `xgb_11_tuned` 0.6361 and `canonical/main` 0.8551 |
| sd 0.010 across four repeats | Abstract | §2 above |
| ~12 additional frauds per week | Abstract, §5.5, §6.3, Conclusion | recall difference at the budget × mean fraud present: 0.157353 × 75.06 = 11.8 |
| 0.120, 0.099 | Abstract, §1, §5.2, §6.2, Conclusion | Table 5's two increments, 0.1198 and 0.0992 |
| **0.094**, **0.125** | Abstract, §6.1, Conclusion | 16-week `P3_acausal − P3_rolling` = 0.094418; 16-week `P1_causal − P3_rolling` = 0.124720. **v6e's abstract printed 0.09 and 0.13; corrected in v7a** |
| "roughly a tenth as much" | Highlights, Abstract | 0.025 / 0.224 = 0.11 |

### §3.3, graph construction

| value | source |
|---|---|
| 9,911,248 edges, 19.82 per node | `run_manifest.json` `experiment.edges`, `edges_per_node` |
| 54 contemporaneous pairs, **22 cardholder and 32 merchant** | `run_manifest.json` stores only the total. The per-relation split was **recomputed from the graph builder on 2026-09-08 and reproduces exactly** — derivation below |
| 4,949,363 cardholder edges and 4,961,885 merchant edges | same rebuild. Not stated in the paper; recorded here because it is the denominator for the 22 and the 32 |
| self-loop at every node, 10,411,248 with them | Equation 5. The 9,911,248 counts typed relation edges only; one loop per node takes it to 10,411,248. **Both are now stated in §3.3, which also records that the loops are injected per batch in `sample_subgraph` (`src/run_date_gnn.py:151`) rather than stored, which is why the built edge set carries none.** `add_self_loops` in `src/temporal_graph.py` is used only by `tests/test_causal_guard.py`, never in the training path |

**The rebuild, for anyone repeating it.** `prepared/sparkov_clean_v1.csv`, sha256
`24081f70...f26728`, which is the `dataset_sha256` in `run_manifest.json`; `df.tail(500_000)`
per `subsample_recent` in `src/run_date_gnn.py`; `build_temporal_edges` from
`src/temporal_graph.py` with `entity_cols=("uid","merchant")` and `max_prior_neighbors=10`.
A contemporaneous edge is one with `times[src] == times[dst]`, counted per `edge_rel`:
relation 0 is the cardholder, relation 1 the merchant. Total 9,911,248 edges at 19.82 per
node, 0 future edges, 54 contemporaneous, 22 and 32. Takes a few seconds and needs no GPU.

### §4.1, dataset and folds

| value | source |
|---|---|
| 1,852,394 transactions, 999 cardholders, 693 merchants, 14 categories, 0.521%, 22 fields | the public corpus, `dataset_sha256` in `run_manifest.json` |
| 500,000-transaction slice, 26 weeks | `run_manifest.json` `experiment.rows`, slice timestamps 2013-07-10 to 2013-12-31 |
| 921 cardholders active, 0.386% fraudulent in the slice | `experiment.positives` 1,929 / 500,000 = 0.3858% |
| fitting partition 126,605 to 387,572; 560 to 1,745 fraudulent | `fold_composition.csv` `n_train`, `train_fraud` |
| validation tail 93 to 270 fraudulent | same, `val_fraud` |
| test week 15,766 to 31,849; 7 to 113 fraudulent | same, `n_test`, `test_fraud` |
| total scored 338,456; 1,276 fraudulent; prevalence 0.377% | column sums of the 17 kept folds |
| k from 79 to 160, mean 100 | `per_fold.csv` `k` |

### §4.3, the tree

| value | source and formula |
|---|---|
| 0.494 at library defaults | `xgb_11_defaults.csv` mean AP 0.4936 |
| 0.615 at a fixed configuration | `xgb_11_fixed.csv` 0.6153 |
| 0.636 per-fold tuned | `xgb_11_tuned.csv` 0.6361 |
| 0.142 recovered by moving off defaults | 0.6361 − 0.4936 = 0.1425 |
| 0.021 for the per-fold search over fixed | 0.6361 − 0.6153 = 0.0208 |
| deepest trees on 10 of 17, smallest learning rate on 10 of 17 | `xgb_11_tuned_params.csv`: `max_depth` 8 ×10, `learning_rate` 0.03 ×10 |
| child weight selected 6, 6 and 5 times | same: `min_child_weight` 1 ×6, 5 ×6, 20 ×5 |
| class weighting on 4 of 17 | same: `scale_pos_weight` ≠ 1 on 4 folds, values 207.4 to 221.3 |
| rounds 20 to 975, median 126 | same, `rounds` |

### §4.4, configuration

| value | source |
|---|---|
| K = 10, cap binds on 98% of nodes | degree statistics at build time. `edge_k/config.py` records 96.3% for the one-relation graph at K = 20, the nearest stored analogue |
| receptive field 421 for two layers, 1,641 at K = 20, 8,421 for three layers | 1 + d + d² with d = relations × K: d = 20 → 421; d = 40 → 1,641; three layers d = 20 → 8,421 |
| batch expands to 70–95% of the fitting partition | measured during sitting A |
| library versions | `run_manifest.json` |

### §5.1

| value | source and formula |
|---|---|
| 0.855 vs 0.636, Δ +0.219 | mean of paired fold differences, 0.2191 |
| P@k 0.643/0.513, R@k 0.812/0.654 | `per_fold.csv` column means, both arms |
| Table 4's four budget p-values, all **0.00003** | 2/2¹⁶, the floor for a unanimous result on the 16 folds that are not ties. Week 24 is a tie for all four by construction: its budget of 158 alerts exceeds the seven positives available, so both models capture all seven. **v6e printed 0.00043 and 0.00044 from SciPy's `method='auto'` fallback; the source files are now patched** |
| F1 0.677/0.539 | **computed per fold and then averaged**, not from the displayed means. Per-fold harmonic means average to 0.6769 and 0.5393. Taking the harmonic mean *of* the mean P and R instead gives 0.717 and 0.575, which is a 0.04 discrepancy waiting for anyone who recomputes it the quick way |
| MCC 0.695/0.554 | same convention, from each fold's top-k confusion matrix. A reconstruction that thresholds at the k-th score gives 0.6963 and 0.5541; the printed 0.695 differs in the last digit through tie handling at the k boundary, where the paper breaks ties by order of arrival |
| ROC-AUC 0.986/0.985, Δ +0.001, 12/17, p 0.306 | fold-wise `roc_auc_score` on `scores.csv`: 0.9863 and 0.9849, Δ 0.0014 |
| fold sd 0.090 vs 0.134 | `per_fold.csv` `ap.std(ddof=1)`: 0.089793 and 0.134383 |
| per-fold range 0.644 to 1.000 | same, min and max |
| week 13: 0.644 vs 0.528; week 10: 0.875 vs 0.383 | per-fold rows; 0.6436/0.5280 and 0.8750/0.3835 |
| week 24 removed: 0.224 at p = 0.00003 | 16-fold paired mean 0.2241 |
| about 59 vs 47 frauds within the budget | fold-wise `recall_at_k` × `test_fraud`, averaged: 58.5 and 46.8 |
| four repeats vary by 0.002 in AUROC | §2 above. **This is the sd, 0.0023; the range is 0.005** |
| cardholder-only trails the tree by 0.007 in AUROC | 0.9781 − 0.9849 = −0.0067 |

### §5.2

| value | source and formula |
|---|---|
| ladder 0.636 → 0.756 → 0.855, +0.120 and +0.099 | 0.1198 and 0.0992, summing to 0.2191 **by construction** |
| p 0.00021 on 16/17, and 0.00002 on 17/17 | `derived/paired_tests.csv` |
| P@k and R@k 0.513/0.654 → 0.586/0.744 → 0.643/0.812 | column means of the three arms |
| fold sd 0.134 → 0.083 → 0.090 | `ap.std(ddof=1)` of `xgb_11_tuned`, `uid_only`, `main` |
| about 47, 54 and 59 frauds at the budget | 46.8, 53.7, 58.5 as above |
| "more than two fifths" of the gain from merchant | 0.0992 / 0.2191 = 0.45 |
| budget-matched control 0.732, contrast 0.123, 16/17, one tie, p 0.00003 | `edge_k/results/arms/uid_K20/per_fold.csv` against `canonical/main`, mean paired difference 0.1234. **Prose only — must not enter Table 5.** Replacing Table 5's +0.099 rung with it gives 0.1198 + 0.1234 = 0.2432 against a true 0.2190, because it starts from `uid_K20` at 0.7317, not `uid_only` at 0.7559 |
| in-degrees 9.90, 19.61 and 19.82 per node | 4,949,363 / 9,807,302 / 9,911,248 edges over 500,000 nodes. The first is from the 2026-09-08 rebuild, the others from the arm manifests |
| `uid_K20` vs `uid_only` −0.0241, p 0.145 | the pure K contrast within one relation, reported as unresolved |

### §5.3

| value | source and formula |
|---|---|
| Table 6's five arms | 16 common weeks, §3 above |
| AP range 0.750 to 0.974, difference 0.128 | 0.750084 to 0.974172; the paired 16-week contrast is 0.128105 |
| 0.974 → 0.971 under causal construction with a random split | `P1_leaky` − `P1_causal` = 0.003385 |
| `P3_acausal` 0.940 | 0.940485 |
| **AUROC range 0.025, from 0.9746 to 0.9994** | `P2_cutoff` 0.974640 to `P1_causal` 0.999367 = 0.024727. **The maximum is P1 causal, not P1 leaky.** v6e printed 0.024, the subtraction of Table 6's rounded values |
| graph effect 0.089 over 17 weeks, 15/17, p 0.00009 | `paired_tests.csv`; 0.088864 |
| 16-week version 0.094 | 0.094418 |
| 0.003 and 0.034 individually, 0.128 jointly, excess 0.091 | 0.003385, 0.033687, 0.128105; excess = 0.128105 − (0.003385 + 0.033687) = 0.091033 |
| excess positive in 14 of 16, p 0.00058 | `non_additive_excess_per_week.csv` |
| P2 0.765 vs 0.855 over 17 weeks, difference 0.090 | 0.764785, 0.855123, paired 0.090338 |
| 50.1% of edges point backwards | `protocol_v2/results/arms/P3_acausal/arm_manifest.json` `graph_facts.future_frac` 0.5011, being 5,010,950 of 9,999,876 |

### §5.4

Every value is in §2 and §3 above. The two provenance sentences: the four repeats are sitting A;
"the remaining arms are trained once each" covers `maxagg` and `depth3` from sitting A and the
three arms from sitting C.

| value | source |
|---|---|
| convergence: three quarters of the gain by epoch 10, 89% by epoch 15 | `canonical/main/val_curves.csv`, mean curve over epochs 0–53 where all 17 folds still train: 76.6% and 88.8% of the in-window gain from 0.5869 to 0.8488. 90% is first reached at epoch 18 |
| early stopping median 83, min 54, max 129 | `canonical_summary.csv` |
| edge arm epochs 39 to 142, median 63 | `edge_k/results/arms/no_edge_bias/run_record.json` |
| K wall times 3.15 h, 8.74 h, 21.07 h; **2.4×** | `summary.json` per arm and `canonical_summary.csv`. 21.07 / 8.74 = 2.41 |
| receptive field 111 / 421 / 1,641, a fifteenfold range | 1 + d + d², d = 2K. 1,641 / 111 = 14.78 |
| advantage over the tree 0.200 to 0.219 | `K5_both` 0.8361 − 0.6361 = 0.2000; `main` 0.2191; `K20_both` 0.2188 |
| fold sd 0.064 / 0.090 / 0.104 | the three arms' `sd_ap_across_folds` |

### §5.5

| value | source: `derived/budget_sweep_11feat_matched.csv` |
|---|---|
| six alert rates, mean k 20 / 40 / 100 / 200 / 399 / 996 | `mean_k` |
| at the budget: two thirds of alerts fraudulent, four fifths of fraud recovered, against half and two thirds | 0.6427/0.8115 against 0.5134/0.6542 |
| Δ precision 0.111 at 0.001, Δ recall 0.041 | 0.110619, 0.041448 |
| Δ recall 0.157 at the budget, at least 0.083 through 0.02 | 0.157353, 0.082966 |
| Δ precision 0.002 and Δ recall 0.023 at 0.05 | 0.002364, 0.023100 |
| p at or below 0.0005 from 0.001 to 0.02; 0.013 precision and 0.018 recall at 0.05 | exact Wilcoxon, ties dropped: 0.000488 / 0.0000305 ×3 / 0.000427 / 0.01309 and 0.01825. **v6e's "at or below 0.0013" was false as printed — the value at 0.02 is 0.0013462** |
| both models recover more than 93% at 0.05 | `gnn_r_at_k` 0.9561, `xgb11_r_at_k` 0.9330 |
| "a forty-seventh of the 0.111 precision advantage" | 0.110619 / 0.002364 = 46.8. **v6e said "a fifty-fifth", which is 0.111 / 0.002, rounded ÷ rounded; corrected in v7a** |

### §6 and §7

Every number restates one already indexed above. §6.3's "loses 0.090 average precision relative
to weekly refitting, almost exactly the 0.089 associated with causal graph construction" pairs
the 17-week P2 contrast with the 17-week graph contrast — both 17 weeks, so the comparison is on
one basis.

---

## 6. Ratios and multiples

Collected because every rounding error found in this revision was in a ratio. **All are computed
from unrounded inputs.**

| ratio | value | inputs | where |
|---|---|---|---|
| merchant ÷ largest architectural effect | **2.6** | 0.099235 / 0.037449 = 2.6499 | §5.4, §6.2 |
| cardholder ÷ largest architectural effect | **3.2** | 0.119830 / 0.037449 = 3.1998 | §5.4, §6.2 |
| edge effect ÷ two-execution SD | 2.6 standard deviations | 0.037449 / 0.014453 = 2.59 | §5.4 |
| max-pool, depth ÷ same | 1.3 and 0.2 | 0.0191 / 0.014453; 0.0029 / 0.014453 | §5.4 |
| cardholder, merchant ÷ same | "eight and seven" | 8.3 and 6.9 | §5.4 |
| AUROC range ÷ AP range | "roughly a tenth" | 0.024727 / 0.224089 = 0.110 | Abstract, Highlights |
| merchant share of the total gain | "more than two fifths" | 0.0992 / 0.2191 = 0.453 | §5.2 |
| receptive-field range | fifteenfold | 1,641 / 111 = 14.78 | §5.4 ×2 |
| K = 20 wall-clock | 2.4× | 21.07 / 8.74 = 2.41 | §5.4 |
| depth receptive-field cost | twentyfold | 8,421 / 421 = 20.0 | §5.4 |
| tightest ÷ widest precision advantage | **a forty-seventh**, 46.8 | 0.110619 / 0.002364 | §5.5 |

**The two unrelated 2.6s.** In §5.4 "2.6 standard deviations" is the edge effect over the noise
scale; in §5.4's closing sentence and §6.2, "2.6 times" is the merchant effect over the edge
effect. Numerically coincidental. §5.4 states the first in standard deviations precisely so the
two cannot be read as one quantity.

**The 3.2 mixes bases** — its numerator changes model family while its denominator is a
within-model ablation. §5.4 and §6.2 both say so at the number.

---

## 7. Quantities the manuscript states more than once

If one of these changes, every listed location changes with it.

| quantity | locations |
|---|---|
| 0.219 / 0.2191 | Abstract, §1, Table 4, §5.1 ×2, §5.2 ×2, §6.2, Conclusion |
| 0.120 and 0.099 | Abstract, §1, Table 5, §5.2 ×3, §5.4, §6.2 ×2, Conclusion |
| 0.037, always with the joint-bound caveat | §3.4, §5.4 ×3, Table 8, §6.2 ×2, Conclusion |
| 0.014 and 0.010 | §5.4 ×3, Figure 8's caption ×2, Abstract (0.010) |
| 0.023 and 0.029 | §5.4: 0.029 ×3, 0.023 ×2 |
| 0.094 and 0.125 | Abstract, §5.3, §6.1, Conclusion |
| **0.025** | §5.3, §6.1, Conclusion |
| 0.224 | §5.3, §6.1, Conclusion |
| 0.089 / 0.090 | §5.3 ×2, §6.1, §6.3 ×2, Conclusion |
| twelve extra frauds per week | Abstract, §5.5, §6.3, Conclusion |
| fold sd 0.090 | §5.1, §5.2, §5.4, Table 8 |
| fold sd 0.083 and 0.064 | §5.2 ×2 (0.064 quoted from §5.4), §5.4, Table 8 |
| 421 receptive field | §4.4, §5.4 ×2 |
| 0.019 for K = 5, and "less than 0.001" for K = 20 | §4.4, §5.4, Table 8 |
| 8.7 and 21.1 training hours | §4.4, §5.4 |
| the 2.6 / 3.2 ratios | §5.4, §6.2 |

---

## 8. Scripts and what they own

| script | produces | reads |
|---|---|---|
| `rerun_canonical.py` | sitting A, and `derived/paired_tests.csv` | the prepared corpus |
| `protocol_v2/assemble.py` | sitting B's derived files | sitting B's arms plus sitting A's `main` |
| `edge_k/runner.py` | sitting C's arms | the prepared corpus, with the canonical arms hashed as a guard |
| `figures/make/fig_5_1.py` | Figure 4 | sitting A |
| `protocol_v2/figures/make/fig_5_3_v2.py` | Figures 5 and 6 | sitting B |
| `figures/make/fig_5_3_SUPERSEDED.py` | nothing — it asserts the 75-epoch values and raises before drawing | renamed 2026-09-08 |
| `figures/make/fig_5_4.py` | Figures 7 and 8, **and Table 8's Δ, fold counts and p-values** | sittings A and C |
| `figures/make/fig_5_5.py` | Figure 9 | sitting A's budget sweep |
| `figures/make/fig_method.py` | Figures 1, 2, 3 | nothing; schematics |
| `src/evaluation.py` | AP, P@k, R@k, and the paired test used by the runners | — |

---

## 9. Known disagreements

Files in the repository that state something the manuscript does not, with what is correct and
why the file was not regenerated. **Found by searching every `.md`, `.csv`, `.json` and `.py`
under the repository root for the affected values, not from a list.**

| # | file | says | correct | why not regenerated |
|---|---|---|---|---|
| 1 | `results/rerun_20260826_114717_m5/derived/paired_tests.csv` | `maxagg - main` wins 7, p 0.234; `depth3 - main` wins 9, p 0.963 | 9 and 8 reference wins with one tie each, p 0.252 and 0.980 | regenerating means re-running sitting A, ~35 GPU-hours. The generating code in `rerun_canonical.py` **is** fixed |
| 2 | `edge_k/results/FINAL_ANALYSIS.md` | K = 20 costs "6.7× the wall time" | **2.41×**. 6.7 is K = 20 against K = 5 | `edge_k/results/` is checksummed; another session owns the correction |
| 3 | `edge_k/results/SUMMARY_all_arms.md` | the same 6.7× claim | as above | as above |
| 4 | `MANIFEST.md` | *(before 2026-09-08)* pointed the protocol numbers at `results/protocol/`, giving 0.107, 0.123 and 0.103 | 0.094, 0.125, 0.090 from `protocol_v2` | **fixed 2026-09-08.** The old values were correct for the 75-epoch run and are recorded there as such |
| 5 | `protocol_v2/results/derived/paired_tests.csv`, `leak_decomposition.csv` | `P1_leaky − P1_causal` 1 tie p 0.5614; `P2_cutoff − P3_rolling` 0 ties p 0.00038 | 2 ties p 0.58301; 1 tie p 0.00031 | the files drop only exactly-zero differences; the manuscript declares 1e-12. Two weeks differ by 2.2e-16. **The manuscript is right under its own rule.** The Δ values agree |
| 6 | `protocol_v2/README_PACKAGE.md` | lists §5.4's values to preserve as "p=0.234 / p=0.963" | 0.252 and 0.980 | a migration note from 2026-09-02, superseded by finding 1 |
| 7 | `protocol_v2/README_PACKAGE.md`, `RESULTS_SO_FAR.md` | "AUROC range 0.0248" | 0.024727 unrounded; 0.0248 is the subtraction of the ladder's four-decimal values | both round to the 0.025 the paper states; no consequence |
| 8 | `paper/REVIEWER_RESPONSE_DRAFT.md` | calls the edge arm "the only one that is statistically distinguishable"; says "fifteenfold" where it means the receptive-field range | the resolution framing of §5.4; fifteenfold is in fact right | **open, this session's final step** |

### Not disagreements, though they look like them

- `results/rerun_.../COMPARISON.md` quotes 0.8427, 0.8237 and 0.8568. Those are the superseded
  arms it exists to compare against.
- `results/architecture/` reads `maxagg` 0.8237 and `depth3` 0.8568, the latter a **+0.014
  gain** where the canonical run gives −0.0029. Two executions of one configuration, and the
  clearest evidence in the project for why a single execution cannot resolve differences of this
  size.
- `results/protocol/` reproduces its own numbers exactly. It is a different execution, not a
  wrong one.

### One provenance gap, now closed

§3.3 states that the 54 contemporaneous edges divide into **22 from the cardholder relation and
32 from the merchant relation**. No result file stored the split — `run_manifest.json` holds only
the total and `assert_causal` returns only a count — so it was **recomputed from the graph
builder on 2026-09-08 and reproduces exactly**, along with the 9,911,248 total, the 19.82 per
node and zero future edges. The derivation is in §5 under §3.3. Nothing in the text needed
changing.

---

## 10. Cross-check against the independent verifications

Three verification passes recomputed parts of this from primary sources without being told what
to expect. Where they overlap with this map they agree, with two exceptions worth recording:

| item | verifier | this map | resolution |
|---|---|---|---|
| Figure 8 bootstrap stability across seeds | "every bound moves by 0.002 or less" | worst movement **0.0027**, max-pool's lower bound | the caption says 0.003, the value that survives recomputation |
| receptive-field range | "14.8×, so say fourteenfold" | 14.78, which **rounds to fifteen** | fifteenfold, as v6e had it. The intermediate change to fourteenfold was reverted |

Everything else agreed: all six Figure 8 point estimates and twelve interval bounds, the band at
0.014453, the single-run SD at 0.010219, the fold counts and exact p-values for all five arms,
the receptive fields, wall times, epoch ranges, fold composition, the k = 158 precision cap at
week 24, and that no true statement in v6e was lost.

---

## 11. The `_SUPERSEDED` naming convention

As of 2026-09-08 every superseded result file carries a `_SUPERSEDED` suffix, following
`protocol_v2/results/arms/P3_acausal_SUPERSEDED_smoke_*`. **Sixteen files were renamed.** The
point is that a filename now tells you whether it can be cited.

| file, after renaming | disagrees with the manuscript how | replaced by |
|---|---|---|
| `results/per_fold/gnn_11_converged_SUPERSEDED.csv` | 0.8427 against 0.8551 | `canonical/main/per_fold.csv` |
| `results/per_fold/gnn_11_uid_only_converged_SUPERSEDED.csv` | 0.7731 against 0.7559 | `canonical/uid_only/per_fold.csv` |
| `results/per_fold/converged_wilcoxon_SUPERSEDED.csv` | built on a 0.8331 reference, and compares arms the paper does not | `derived/paired_tests.csv` |
| `results/architecture/maxagg_11_SUPERSEDED.csv` | 0.8237 against 0.8361 | `canonical/maxagg/per_fold.csv` |
| `results/architecture/depth3_11_SUPERSEDED.csv` | 0.8568 against 0.8522 | `canonical/depth3/per_fold.csv` |
| `results/architecture/arch_11_summary_SUPERSEDED.csv` | reference 0.8331 | `canonical_summary.csv` |
| `results/architecture/arch_11_wilcoxon_SUPERSEDED.csv` | depth-3 **+0.0237**, the opposite sign to Table 8's −0.0029 | `derived/paired_tests.csv`, Table 8 |
| `results/budget/budget_sweep_11feat_matched_SUPERSEDED.csv` | all six deltas differ | `derived/budget_sweep_11feat_matched.csv` |
| `results/budget/scores_gnn_11_converged_SUPERSEDED.csv` | the 0.8427 run's scores | `derived/scores_gnn_11_converged.csv` |
| `results/convergence/gnn_11_converged_run1_SUPERSEDED.csv` | 0.8331 against 0.8551 | `canonical/main/per_fold.csv` |
| `results/convergence/val_curves_converged_SUPERSEDED.csv` | curves from another execution than Table 4's | `canonical/main/val_curves.csv` |
| `results/protocol/per_week_all_arms_SUPERSEDED.csv` | 0.9655 / 0.9628 / 0.7409 / 0.9463 / 0.8395 against Table 6 | `protocol_v2/results/derived/per_week_all_arms.csv` |
| `results/protocol/factorial_2x2_SUPERSEDED.csv` | 75-epoch generation | `protocol_v2/.../factorial_2x2.csv` |
| `results/protocol/leak_decomposition_perweek_SUPERSEDED.csv` | 75-epoch generation | `protocol_v2/.../leak_decomposition.csv` |
| `results/protocol/protocol_ladder_perweek_SUPERSEDED.csv` | 75-epoch generation | `protocol_v2/.../protocol_ladder.csv` |
| `figures/make/fig_5_3_SUPERSEDED.py` | asserts 0.966 / 0.963 / 0.741 / 0.839 / 0.946, and raises before drawing | `protocol_v2/figures/make/fig_5_3_v2.py` |

### Deliberately not renamed

| file | why it is current |
|---|---|
| `results/per_fold/xgb_11_tuned.csv` | **the reported baseline at 0.6361.** The tree is deterministic under a fixed seed and was not re-run |
| `results/per_fold/xgb_11_defaults.csv` | 0.4936, which is §4.3's "0.494 at library defaults" |
| `results/per_fold/xgb_11_fixed.csv` | 0.6153, which is §4.3's fixed-configuration 0.615 |
| `results/per_fold/xgb_11_tuned_params.csv` | Table B.1 |
| `results/per_fold/fold_fraud_report.csv` | identical in content to `fold_composition.csv`; a duplicate, not a contradiction |
| `results/budget/scores_xgb_11_tuned.csv` | md5-identical to the canonical `derived/` copy, and the file the reported sweep reads |

`results/per_fold/` and `results/budget/` are therefore the two directories that hold both
current and superseded files, which is why the naming matters more there than anywhere else.
