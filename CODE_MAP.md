# Code map — which code produced which table and figure

**TRACE-GNN.** Every numbered table and figure in the manuscript, matched to the code in this
archive that produced it, with the command that reproduces it.

This archive contains **code**. No figure images, no dataset, no manuscript. The only data
files are the two gradient-boosted baseline files in `results/`, which the code **reads** and
does not produce — see `results/README.md`. Nothing here can be read off a shipped file — every number must be recomputed. That
is the point: the archive exists to show that the reported results come from this code, and to
let anyone run it and check.

| | |
|---|---|
| **Manuscript** | `TRACE-GNN_manuscript_v7a.docx` |
| **Draft** | v7a, written to disk 8 September 2026 |
| **Code exported** | 8 September 2026, after the manuscript |
| **Dataset** | Sparkov only — no other corpus is used anywhere in the paper |
| **Scope** | 9 figures, 8 numbered tables, 2 appendix tables |

---

## How to read this file

Each row names the **code that computes the numbers**, not the file they were stored in.
Paths are relative to this directory. Commands assume you have run the setup below.

Three kinds of entry appear:

- **Trained arm** — a model is fitted and scored. Expensive; needs a GPU.
- **Derived** — numbers computed from arms that have already run. Cheap, CPU only.
- **Schematic** — drawn from nothing but the code itself. No data required.

---

## Before anything runs

**1. Get the data.** Not redistributed; it is public.

```
# https://www.kaggle.com/datasets/kartik2112/fraud-detection
python src/prepare_sparkov.py --raw <download_dir> --out prepared/
```

This writes `prepared/sparkov_clean_v1.csv`. Pass its location to every entry point with
`--prepared <path>`.

**2. Install.** `torch` comes from pytorch.org, not from `requirements.txt`:

```
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

See `ENVIRONMENT.md` for the exact versions, hardware and the determinism caveat.

**3. Check before committing hours.**

```
python rerun_canonical.py --prepared <path> --dry-run
```

Loads the corpus, builds the folds and the graph, runs every assertion, projects peak VRAM,
then stops without training.

---

## 1. The three programmes

The code is in three parts because the paper's results were produced in three sittings. Each
has its own entry point, its own configuration and its own guards.

| programme | directory | what it produces |
|---|---|---|
| **Canonical run** | `rerun_canonical.py`, `src/` | Tables 1, 3, 4, 5, 8; Figures 4, 7, 8, 9 |
| **Protocol study** | `protocol_v2/` | Tables 6, 7; Figures 5, 6 |
| **Edge and K study** | `edge_k/` | Table 8 rows 2, 5, 6; Figure 8 rows 2, 5, 6 |

They share the model and the data pipeline in `src/`. They do not share runners, configs or
guards, deliberately — each fences its own writes so that one programme cannot overwrite
another's results.

---

## 2. Tables

| table | subject | code that produces it | kind |
|---|---|---|---|
| **1** | Fold structure and protocol constants | `src/temporal_folds.py` builds the folds; `rerun_canonical.py` writes the composition | trained arm |
| **2** | Search space for the gradient-boosted baseline | `notebooks/15_workstation_session.ipynb`, using `src/run_experiment.py` and `src/model_factory.py` | trained arm |
| **3** | TRACE-GNN configuration | `src/experiment_config.py` and `src/model.py` define it; `rerun_canonical.py` asserts every field before training (`EXPECTED_*`) | — |
| **4** | Matched-feature comparison, TRACE-GNN vs XGBoost | `rerun_canonical.py --arms main`, driven by `notebooks/16_canonical_rerun.ipynb`; metrics by `src/evaluation.py` | trained arm |
| **5** | Sequential ladder: tree → cardholder-only → two relations | `rerun_canonical.py --arms main,uid_only` | trained arm |
| **6** | Performance under five evaluation protocols | `protocol_v2/runner.py` trains four arms; `protocol_v2/assemble.py` assembles the ladder | trained arm |
| **7** | Paired differences in average precision | `protocol_v2/assemble.py` | derived |
| **8** | Sensitivity to architecture and neighbour cap | `rerun_canonical.py --arms main,maxagg,depth3` for rows 1, 3, 4; `edge_k/run_arm.py` for rows 2, 5, 6; differences recomputed by `figures/make/fig_5_4.py` | trained arm |
| **A.1** | The eleven features | `src/features_aggregation.py` and `src/config.py` define them; `src/fold_preprocessor.py` applies them per fold | — |
| **B.1** | Per-fold tree configuration | `notebooks/15_workstation_session.ipynb`, using `src/run_experiment.py` and `src/model_factory.py` | trained arm |

### Commands

```
# Tables 1, 4, 5 and rows 1,3,4 of Table 8 — the canonical run
python rerun_canonical.py --prepared <path> --repeats 3
#   or interactively, which streams progress:
#   notebooks/16_canonical_rerun.ipynb

# Tables 6 and 7 — the protocol study, cheapest arm first
#   notebooks A1..A5 in order, or equivalently:
python protocol_v2/runner.py P2_cutoff     # one model, trained once, never refitted
python protocol_v2/runner.py P1_causal     # random 80/20, time-directed graph
python protocol_v2/runner.py P1_leaky      # random 80/20, graph reaches forward
python protocol_v2/runner.py P3_acausal    # rolling folds, graph reaches forward
python protocol_v2/assemble.py             # Tables 6 and 7

# Table 8 rows 2, 5, 6 — the edge and K study, one process per arm
python edge_k/run_arm.py no_edge_bias      # edge encoding removed
python edge_k/run_arm.py K5_both           # neighbour cap 10 -> 5
python edge_k/run_arm.py K20_both          # neighbour cap 10 -> 20

# Table 2 and B.1 — the gradient-boosted baseline
#   notebooks/15_workstation_session.ipynb
```

The fifth protocol arm, `P3_rolling`, is **not** retrained: it is the canonical `main` arm from
the first programme, read as read-only input. `protocol_v2/guards.py` enforces that it can never
be trained or overwritten, which is what makes the claim in Section 5.3 — model held fixed,
protocol varied — checkable rather than asserted.

---

## 3. Figures

| figure | subject | code that produces it | kind |
|---|---|---|---|
| **1** | Rolling-origin protocol | `figures/make/fig_method.py` | schematic |
| **2** | Causal graph construction | `figures/make/fig_method.py` | schematic |
| **3** | TRACE-GNN architecture | `figures/make/fig_method.py` | schematic |
| **4** | Per-fold AP, TRACE-GNN vs tree | `figures/make/fig_5_1.py` | derived |
| **5** | AP and AUROC under five protocols | `protocol_v2/figures/make/fig_5_3_v2.py` | derived |
| **6** | Non-additivity across the four P1/P3 arms | `protocol_v2/figures/make/fig_5_3_v2.py` | derived |
| **7** | Fold volumes and validation curves | `figures/make/fig_5_4.py` | derived |
| **8** | Mean AP differences, six rows | `figures/make/fig_5_4.py` | derived |
| **9** | Precision and recall across alert budgets | `figures/make/fig_5_5.py`, sweep by `src/budget_sweep_11.py` | derived |

### Commands

Every figure script takes a search root and an output directory. All are CPU-only.

```
# Figures 1, 2, 3 — no data needed
python figures/make/fig_method.py figures/

# Figure 4
python figures/make/fig_5_1.py results/ figures/ --canonical results/<run>

# Figures 5 and 6
python protocol_v2/figures/make/fig_5_3_v2.py protocol_v2/results/ figures/

# Figures 7 and 8
python figures/make/fig_5_4.py results/ figures/ --canonical results/<run>
#   Table 8's rows 2, 5, 6 are added with:
#   --edge-arm edge_k/results/arms/<arm>/per_fold.csv

# Figure 9
python figures/make/fig_5_5.py results/ figures/ --canonical results/<run>
```

**`--canonical` is not optional in practice.** Without it the scripts locate per-fold files by
name anywhere under the root and take whichever is newest, so a figure can silently come from a
different execution than the table beside it. Naming the run fixes that.

**The figure scripts assert their inputs against the manuscript and refuse to draw if they
disagree.** This is deliberate. An earlier version silently loaded a superseded study and
produced a plausible but wrong Figure 5.

**Figures 1, 2, 3, 5 and 6 were hand-finished** after generation for the manuscript, so the
images in the paper are not byte-identical to what these scripts emit. Figures 4, 7, 8 and 9
reproduce exactly.

---

## 4. Every file in this archive, and what it is for

### `src/` — the model and the data pipeline, shared by all three programmes

| file | role | serves |
|---|---|---|
| `prepare_sparkov.py` | builds `sparkov_clean_v1.csv` from the Kaggle download | everything |
| `config.py` | Sparkov column semantics for the preprocessing stage | everything |
| `data_io.py` | loads the prepared frame and its manifest | everything |
| `prepared_loader.py` | thin loader around the prepared directory | everything |
| `dataset_audit.py` | audits the prepared file | preprocessing |
| `experiment_config.py` | the experiment configuration dataclass | Table 3 |
| `features_aggregation.py` | the five causal uid aggregations | Table A.1 |
| `fold_preprocessor.py` | per-fold encoding, fitted on train only | Table A.1, all arms |
| `temporal_folds.py` | rolling-origin fold construction | Table 1 |
| `temporal_graph.py` | causal graph: edges only from earlier transactions | Figures 2, 4–9 |
| `acausal_graph.py` | the forward-reaching graph used by the leaky arms | Tables 6, 7 |
| `model.py` | the TRACE-GNN architecture | Figure 3, all graph arms |
| `model_factory.py` | builds the model or the gradient-boosted baseline | Tables 2, 4, 5, B.1 |
| `run_date_gnn.py` | training loop, early stopping, exact mini-batch inference | all graph arms |
| `run_experiment.py` | fold loop, fit and score one configuration | Tables 2, 4, B.1 |
| `evaluation.py` | AP, AUROC, precision/recall at the alert budget | every reported metric |
| `budget_sweep_11.py` | the alert-budget sweep | Figure 9, Table 5's operating point |

### Top level

| file | role |
|---|---|
| `rerun_canonical.py` | the canonical run: trains `main`, `uid_only`, `maxagg`, `depth3` plus repeats, asserts the frozen configuration, writes the run manifest |
| `resume_canonical.py` | resumes an interrupted canonical run from its fold checkpoints |

### `notebooks/` — the documented interfaces

| file | role |
|---|---|
| `00_run_preprocessing.ipynb` | drives `src/prepare_sparkov.py`: builds `prepared/sparkov_clean_v1.csv` from the Kaggle download and audits it. Run this first |
| `16_canonical_rerun.ipynb` | drives `rerun_canonical.py` and streams its progress. Tables 1, 4, 5, 8; Figures 4, 7, 8, 9 |
| `15_workstation_session.ipynb` | the gradient-boosted baseline. Tables 2, 4, B.1 |

### `protocol_v2/` — the five-protocol study

| file | role |
|---|---|
| `config.py` | the four trainable arms and their split/graph combinations |
| `splits.py` | the three split schemes: rolling, random 80/20, single cutoff |
| `runner.py` | trains one arm |
| `assemble.py` | assembles Tables 6 and 7 from the arms |
| `assembly_settings.py` | the analysis settings, frozen before assembly |
| `guards.py` | locks the canonical arm, fences every write inside `protocol_v2/results/` |
| `provenance.py` | SHA-256 of every source file a run touches, plus the environment |
| `rejoin_scores.py` | rejoins row-split score files |
| `notebooks/A1`–`A5` | the documented interface: four arms then assembly |
| `tests/` | 30 tests, CPU only |

### `edge_k/` — the edge encoding and neighbour cap study

| file | role |
|---|---|
| `config.py` | the arm specifications, including the three built but never run |
| `runner.py` | trains one arm |
| `run_arm.py` | runs one arm start to finish as its own process |
| `controls.py` | positive controls — proves the ablation actually disconnected the edge features |
| `guards.py` | locks both canonical arms, fences every write inside `edge_k/results/` |
| `provenance.py` | as protocol_v2's, for this programme |
| `analysis_settings.py` | the resolution declared **before** training, not after |
| `monitor.py`, `status.py` | progress reporting |
| `notebooks/E1`–`E4` | the documented interface, one notebook per arm |
| `tests/` | 21 tests, CPU only |

### `figures/make/` and `legacy/`

| file | role |
|---|---|
| `fig_method.py` | Figures 1, 2, 3 |
| `fig_5_1.py` | Figure 4 |
| `fig_5_4.py` | Figures 7 and 8, and recomputes Table 8 |
| `fig_5_5.py` | Figure 9 |
| `legacy/fig_5_3_SUPERSEDED.py` | **do not run.** Asserts the retired 75-epoch values and refuses current data, by design. Kept so that someone holding an older copy can identify it |
| `legacy/protocol_runner.py` | the pre-`protocol_v2` runner. Superseded |

### `tests/`

`test_causal_guard.py` — 10 tests. The only artefact that demonstrates the leakage guarantee
independently of any result file. Runs after cloning, with no corpus and no GPU:

```
PYTHONPATH=src                pytest -q tests/
PYTHONPATH=src:protocol_v2    pytest -q protocol_v2/tests/
PYTHONPATH=src:edge_k         pytest -q edge_k/tests/
```

---

## 5. Order of execution

```
  src/prepare_sparkov.py                        once, builds prepared/
          |
          +-- rerun_canonical.py                Tables 1,3,4,5,8(rows 1,3,4)
          |         |                           Figures 4,7,8,9
          |         +-- figures/make/fig_5_1.py
          |         +-- figures/make/fig_5_4.py
          |         +-- src/budget_sweep_11.py --> figures/make/fig_5_5.py
          |
          +-- protocol_v2/runner.py  x4         Tables 6,7
          |         +-- protocol_v2/assemble.py
          |              +-- protocol_v2/figures/make/fig_5_3_v2.py   Figures 5,6
          |
          +-- edge_k/run_arm.py      x3         Table 8 rows 2,5,6
                    +-- figures/make/fig_5_4.py --edge-arm            Figure 8 rows 2,5,6

  figures/make/fig_method.py                    Figures 1,2,3 — independent of all of it
```

The canonical run must come first: both later programmes read it as a locked, read-only
reference and will refuse to start if its checksums do not match.

---

## 6. What a re-run will and will not reproduce

Message passing accumulates neighbour contributions with scatter operations that are not
associative in floating point on GPU, so **a fixed seed does not give bit-identical results**.
Four repeats of the reported configuration at seeds 42–45 give a standard deviation of 0.010 in
mean average precision. Expect a re-run to land within roughly 0.02 of 0.855, not on it.

The gradient-boosted baseline is deterministic and will reproduce exactly.

The figure scripts and all 61 tests are deterministic and CPU-only.
