# TRACE-GNN

**T**emporal **R**elational **A**ttention with **C**ausal **E**dges. A leakage-safe temporal
graph encoder for transaction-level fraud detection under a fixed alert budget, framed as node
classification. Pure PyTorch, with no PyTorch Geometric dependency in the model itself.

Reference implementation for the paper of the same name.

---

## Which manuscript this code corresponds to

This archive was verified against:

| | |
|---|---|
| **Manuscript** | `TRACE-GNN_manuscript_v7a.docx` |
| **Draft** | v7a (derived from v6e, md5 `34911384ab54f4921fab23fdbfa80bff`) |
| **Last saved in Word** | **2026-09-03 18:17 UTC** (revision 25) |
| **Written to disk** | **2026-09-08 13:59 +03:00** |
| **Code exported** | 2026-09-08 18:41–18:46 +03:00, from the working repository, after the manuscript |
| **Scope of v7a** | 258 paragraphs, 10 tables, 9 figures, 2 appendix tables |

Every figure, numbered table and appendix table in v7a has its producing code in this archive;
the mapping is the table under **"Reproducing the paper"** below. If you are holding a
different draft, check that table before assuming a script still applies.

---

> **Note on notebook outputs.** Executed output has been stripped from every shipped notebook,
> because output is not code and it accounted for 4.2 MB of the 5.2 MB this archive replaces.
> **One notebook is a deliberate exception: `notebooks/15_workstation_session.ipynb` retains its
> text output.** That notebook produced the gradient-boosted baseline reported at 0.636, which is
> deterministic, was never re-run, and is cited by Table 2, Table 4 and Table B.1. Its stored text
> is the only executed record of how that number was made, and this is an archive about
> provenance, so deleting it to gain 11 KB of uniformity would have been the wrong trade. What was
> removed from it are the nine embedded figure images in its final cell — 3.0 MB of pictures of
> output that the figure scripts regenerate. Its code cells and its text log are intact. The
> inconsistency is intentional, not an oversight.

> **Note on paths, 2026-09-21.** The project folder was moved on that date. It previously lived
> at `~/Desktop/Fraud Experiment/Second Paper/Start Over, Strick approach/…` and now lives
> under `~/Desktop/Research/Fraud Experiment/…`. Absolute paths visible in the *executed output
> cells* of the shipped notebooks refer to the old location and no longer resolve. They are
> retained deliberately: they are the record of how the reported runs were invoked, not live
> references. **No `.py` file in this archive contains an absolute path**, and nothing here
> depends on that layout — see `REPRODUCE.md` on `--prepared`.

> **This is the code-only release.** It contains no figures as images, no trained weights,
> no dataset and no manuscript.
>
> **Two exceptions, both required to run.** `results/per_fold/xgb_11_tuned.csv` (1.3 KB) and
> `results/budget/scores_xgb_11_tuned.csv` (7.6 MB) are the gradient-boosted baseline.
> `rerun_canonical.py` **reads** that baseline rather than re-running it — it is deterministic
> and was not re-fitted — and refuses to start without it. They are inputs, not results. Every number in the paper can be
> *recomputed* from this code plus the public corpus; none of it can be *read off* a shipped
> file, because no result file is shipped. `MANIFEST.md` and `READING_MAP.md` describe the
> result files by name, path and value so that a reviewer can check a number against the
> archive that holds them, or against their own re-run.

> **Naming.** The paper says **TRACE-GNN**. The code still says **DATE-GNN** (`date_gnn`,
> `DateGNN`, `config="date_gnn_v3"`). Same model, same weights, same results. The original name
> was dropped because *DATE* collides with Kim et al. (KDD 2020), a customs-fraud model that also
> ranks under a fixed inspection budget.

---

## What is here

    src/            model, causal graph construction, training loop, shared evaluation harness
    tests/          ten tests for the causality guarantee
    notebooks/      the three notebooks behind reported results — read the warning below
    figures/make/   the scripts that draw the figures
    protocol_v2/    the retrained 200-epoch protocol study, with its own runner, guards and tests
    edge_k/         the edge-encoding ablation and the neighbour-cap sweep, likewise
    legacy/         two superseded scripts, shipped so an older copy can be identified. Do not use
    ENVIRONMENT.md  exact versions, Python, CUDA, hardware
    MANIFEST.md     every result file, the number it produces, and where that number appears
    READING_MAP.md  every number in the paper, its source file, and the formula behind it

**Three sittings produced the reported results, and the code is organised that way.** The
canonical arms come from `rerun_canonical.py` (driven by `notebooks/16_canonical_rerun.ipynb`),
the protocol study from `protocol_v2/`, and the edge and neighbour-cap arms from `edge_k/`.
`MANIFEST.md` states which sitting every table and figure draws on; `READING_MAP.md` goes
down to the individual number.

Read `MANIFEST.md` before recomputing anything: several comparisons rest on different fold
sets, and it says which.

## Getting the data

The corpus is not in this repository. It is public.

1. Download the Sparkov credit card transactions dataset from
   <https://www.kaggle.com/datasets/kartik2112/fraud-detection> (Shenoy, 2020).
2. Build the prepared frame.

       python src/prepare_sparkov.py --raw <download_dir> --out prepared/

   This writes `prepared/sparkov_clean_v1.csv` with the derived `uid`, `week_idx` and
   `unix_time` columns the rest of the pipeline expects.

3. Every experiment then takes the most recent 500,000 transactions by time, which is done in
   code by `subsample_recent` rather than as a separate file.

### Dataset provenance

| | |
|---|---|
| source | Shenoy, K. (2020). *Credit card transactions fraud detection dataset* [Data set]. Kaggle. <https://www.kaggle.com/datasets/kartik2112/fraud-detection> |
| licence | as stated on the Kaggle page. Synthetic, generated by the Sparkov simulator |
| full corpus | **1,852,394** transactions over two calendar years, 999 cardholders, 693 merchants, 14 spending categories, 22 fields |
| full-corpus fraud rate | **0.521%** |
| slice used | the most recent **500,000** transactions by time, 26 consecutive weeks, taken in code by `subsample_recent` rather than as a separate file |
| slice fraud rate | **0.386%** (1,929 positives). 921 cardholders and all 693 merchants active |
| scored across the 17 retained folds | 338,456 transactions, 1,276 fraudulent, prevalence 0.377% |

**The digest the code expects.** After step 2 above, `prepared/sparkov_clean_v1.csv` must be:

    size    254,735,434 bytes
    sha256  24081f707c7d579ad529e9edb17cee0118cbbdd938e03b6cc56b9b9618f26728

    sha256sum prepared/sparkov_clean_v1.csv

Every run writes this digest into its own `run_manifest.json` under `dataset_sha256`, so a
mismatch is visible after the fact as well as before. A different digest does not necessarily
mean a wrong file — Kaggle has revised the archive before — but it does mean the numbers will
not match to the last digit, and it should be reported alongside any result.

## Installation

    pip install -r requirements.txt

See `ENVIRONMENT.md` for the exact versions, Python, CUDA and hardware. `torch` must be
installed from pytorch.org for your CUDA version rather than from `requirements.txt`.

Training needs a CUDA device. The figure scripts, the budget sweep and all three test suites
run on CPU — but in this code-only release they need result files you have generated yourself,
since none are shipped.

## Leakage safety, the core guarantee

Two independent channels can leak future information into a graph model. Both are closed.

**Causal topology** (`src/temporal_graph.py`). Every transaction links only to its K most recent
**prior** transactions sharing an entity. Edges point prior to current, so a node can never
aggregate a later transaction. `assert_causal()` runs on every build and raises on any edge
pointing backwards in time.

Neighbours are selected by position within an entity's time-sorted history, so two transactions
sharing an entity *and* a timestamp do produce an edge. These are contemporaneous rather than
future. `assert_causal()` returns how many there are, **54 of 9,911,248 edges** on the 500k
slice, and `strict=True` rejects them if you want the strict inequality.

**Fold-safe training** (`src/run_date_gnn.py`). One causal graph is built over the time-sorted
frame and shared across folds. Per fold the model trains on the fitting partition only.
Standardisation statistics and the categorical vocabulary are fitted over the whole history,
**including the validation tail**, whereas the class weight uses the fitting partition alone
(`run_date_gnn.py`, lines 219 and 226). The two are inconsistent, almost certainly by accident.
The history lies entirely before the test week either way, so no future information reaches a
test score, but the validation tail does influence its own standardisation. Section 3.2 of the
paper states this. Causal `uid` aggregations use strictly prior rows, so a record's features never
include its own outcome.

**This makes the protocol transductive in topology.** A test transaction may aggregate over
training transactions that preceded it. It never aggregates over anything that had not yet
occurred, which is the property the paper claims, but the setting is not strictly inductive and
the reported figures are not performance on entities seen for the first time.

## Architecture (`src/model.py`)

- **Input.** Linear projection of the numeric feature vector followed by GELU. Learned
  categorical embeddings exist but are **unused in the reported configuration**, which passes
  `cat_cols=[]` and supplies 11 numeric features.
- **Blocks.** `n_layers` pre-normalised transformer blocks, `nn.LayerNorm` on the *input* of each
  attention and feedforward sublayer with residual connections around both. **There is no
  BatchNorm anywhere**, so no batch-dependent quantity enters the forward pass and a
  transaction's score does not depend on which others are scored alongside it.
- **Attention.** Multi-head scaled dot-product over the causal neighbourhood, plus an **additive
  per-head bias from the 5-dim edge encoding** (`edge_bias = nn.Linear(edge_dim, n_heads)`),
  applied to the logit *outside* the `1/sqrt(d_head)` scaling. The edge term does not enter the
  query-key product.
- **Aggregation.** Attention-weighted sum of value vectors. **Edge features affect the attention
  weight only, never the message content.**
- **Decoder.** A final `nn.LayerNorm`, then a two-layer MLP to one logit per transaction.
- **`use_max_agg=True`** does not replace the weighted sum. It **fuses** it with a max-pool over
  the same neighbourhood and widens the output projection to `Linear(2*dim, dim)`. That arm
  carries roughly 12% more parameters and is **not parameter-matched**.

### Edge encoding (5-dim, `build_edge_features`)

    [ dt_days / 30 , log1p(dt_days) , 1[uid] , 1[merchant] , log((a_dst+1)/(a_src+1)) ]

`edge_dim = 3 + len(entity_cols)`, so single-relation arms must pass a **placeholder** entity
(`("uid", "__none__")`) to keep `edge_dim = 5`. `build_temporal_edges` skips any entity column
absent from the frame, so the placeholder contributes no edges and the network stays
parameter-identical across graph arms. Passing `("uid",)` gives `edge_dim = 4` and a different
model. That mistake invalidated one earlier round of experiments.

Self-loops are added at every node during subgraph materialisation, carrying a **zero** edge
vector, and are exempt from `assert_causal`.

## Training (`src/run_date_gnn.py`)

- **Loss.** Focal loss (gamma = 2) with class weighting, *not* plain weighted BCE. The weight is
  `pos_weight = N_neg / N_pos` on the fitting partition, roughly 225 at this prevalence, applied
  to the positive term only.
- **Optimiser.** AdamW (weight decay 1e-4), gradient-norm clipping at 1.0, cosine annealing with
  `T_max = epochs`. **The schedule is tied to the epoch budget**, so a 75-epoch run is not a
  prefix of a 200-epoch run and arms trained under different budgets are separate configurations.
- **Selection.** Best-validation-AP checkpoint on the temporal validation tail, the last 15% of
  the history by time, with early stopping on `patience`.

### Three things that will surprise you

- **`use_swa=True` silently disables early stopping.** The break is gated on `not tcfg.use_swa`.
  Every run reported in the paper uses `use_swa=False`.
- **The seed is set once, before the fold loop.** Fold *k*'s initialisation depends on RNG
  consumed by folds 1 to *k*−1, and checkpoint resume skips completed folds without consuming
  RNG, so a resumed run gives different state to its remaining folds than an uninterrupted one.
  Resume is crash-safety, not reproducibility. Seed per fold
  (`torch.manual_seed(tcfg.seed + int(w))`) if you need independence.
- **A fixed seed does not make a run reproducible.** Message passing accumulates neighbour
  contributions through non-deterministic scatter operations. Repeating the reported
  configuration end to end returned 0.843 against 0.833, with individual folds moving by as much
  as 0.12 in either direction. Treat 0.010 mean AP as the floor below which no single-seed
  difference can be interpreted.

## Mini-batch expansion is exact, not sampled

Despite the name, `sample_subgraph` **samples nothing**. It materialises every incoming edge
within exactly `n_layers` hops via CSR, so each target node's representation is identical to the
full-graph one, verified to zero absolute difference. What batching bounds is the extent of the
graph held with gradients for one update, and that extent is substantial. At batch 2,048 with
L=2 and K=10 per relation,

| Test week | Fitting partition | Mean subgraph | % of partition |
|---|---|---|---|
| 8  | 126,605 | 119,828 | 94.6% |
| 16 | 236,111 | 201,095 | 85.2% |
| 24 | 387,572 | 272,619 | 70.3% |

So the reduction against a full-partition pass is about fourfold on the earliest fold and under
twofold on the latest. It is enough to fit 48 GB. It is not independence from dataset size.

## The reported configuration

The numbers in the paper come from this exact call. `cat_cols` is empty and `num` holds 11
numeric features.

```python
from experiment_config import ExperimentConfig
import temporal_folds, evaluation, data_io
from run_experiment import prepare_frame
from temporal_graph import GraphConfig
from model import ModelConfig
from run_date_gnn import TrainConfig, run_date_gnn_fold, subsample_recent, filter_folds

cfg = ExperimentConfig(
    prepared_dir="prepared", clean_csv_name="sparkov_clean_v1.csv",
    id_col="TransactionID", target_col="is_fraud", time_col="unix_time",
    week_col="week_idx", amount_col="amt", uid_col="uid",
    non_feature_cols=("uid",), min_train_weeks=8,
)
df, manifest, feature_cols, uid_feats = prepare_frame(cfg)
df = subsample_recent(df, cfg.week_col, 500_000).reset_index(drop=True)

NUMERIC = ("amt", "distance", "age", "city_pop", "hour", "dayofweek")
num = [c for c in feature_cols if c in NUMERIC] + uid_feats      # 11 features

folds, report = filter_folds(df, temporal_folds.make_folds(df, cfg),
                             "is_fraud", 5, 5)                   # 18 -> 17 folds

res = run_date_gnn_fold(
    df, num, [], folds,                                          # cat_cols = []
    GraphConfig(time_col="unix_time", entity_cols=("uid", "merchant"),
                max_prior_neighbors=10),                         # PER RELATION
    ModelConfig(hidden_dim=128, n_heads=4, n_layers=2, ffn_mult=2,
                dropout=0.2, cat_emb_dim=16, use_max_agg=False),
    TrainConfig(epochs=200, lr=5e-4, batch_size=2048, loss="focal",
                use_swa=False, patience=25, device="cuda:0"),
    target_col="is_fraud", time_col="unix_time", alert_rate=cfg.alert_rate,
    evaluation=evaluation, amount_col="amt",
    checkpoint_path="foldckpt.csv", scores_path="scores.csv",
)
```

**`max_prior_neighbors` is per relation.** `("uid","merchant")` with `mpn=10` gives up to **20**
in-edges per node. `("uid","__none__")` gives 10. Any relation ablation must control for this,
which is why the paper includes an edge-budget-matched arm. Out-degree is uncapped.

## Re-running the reported arms

`rerun_canonical.py` produces one designated execution of every arm the paper reports, with the
validation curves, the per-transaction scores and the per-fold results all coming from that same
execution. `notebooks/16_canonical_rerun.ipynb` drives it and streams the progress.

    python rerun_canonical.py --prepared ../prepared --dry-run    # a minute, checks everything
    python rerun_canonical.py --prepared ../prepared --repeats 3  # ~56 h, four arms plus repeats

Output goes to a new timestamped folder under `results/`. Nothing existing is touched. Read
`COMPARISON.md` in that folder before changing anything in the manuscript, since it states whether
any reported value moved and, if so, which tables and figures go stale.

The runner asserts every configuration field before training. `TrainConfig()` defaults to
`lr=1e-3`, `batch_size=8192` and `use_swa=True`, none of which is the reported configuration, and
`use_swa=True` silently disables early stopping. An earlier session lost its results that way.

## Known gap: the per-fold baseline search is not in this archive

**Table 2** (the gradient-boosted search space) and **Table B.1** (the configuration selected on
each fold, with its validation and test score) were produced by an Optuna per-fold search that
ran in an earlier working session. **That search code is not in this archive, and it is not in
the working repository either** — `optuna` appears in `requirements.txt` and `ENVIRONMENT.md`,
but no source file in any TRACE-GNN tree imports it. `notebooks/15_workstation_session.ipynb`
does not perform the search; it *reads* `xgb_11_tuned.csv`, which its own cell describes as
having come from "the earlier session".

What this archive can and cannot do for the baseline:

| | |
|---|---|
| Train XGBoost at a **given** configuration, per fold | **yes** — `src/model_factory.py`, `src/run_experiment.py` |
| Score it and compute the reported metrics | **yes** — `src/evaluation.py` |
| Reproduce Table 2's **search space** as executed | **no** — search code absent |
| Reproduce Table B.1's **selection** of each fold's configuration | **no** — search code absent |

The selected per-fold configurations survive as data in `results/per_fold/xgb_11_tuned_params.csv`
in the results archive, so the baseline is fully **re-trainable and checkable** from this code —
it is the *search* that is not re-runnable. Section 4.3 states the space in prose, and Table B.1
states the outcome. A reviewer can verify the reported baseline; they cannot repeat the tuning
that chose it without reimplementing the search from Section 4.3's description.

This is recorded here rather than left to be discovered at run time.

## Reproducing the paper

| Output | How |
|---|---|
| Tables 4 and 5, Figure 4 | `notebooks/16_canonical_rerun.ipynb`, then `figures/make/fig_5_1.py --canonical <run>` |
| Table 6, Table 7, Figures 5 and 6 | `protocol_v2/notebooks/A1`–`A5`, then `protocol_v2/figures/make/fig_5_3_v2.py` |
| Sections 5.1/5.2/5.4, Figures 7 and 8 | `notebooks/16_canonical_rerun.ipynb`, then `figures/make/fig_5_4.py --canonical <run>` |
| Table 8, Figure 8's sixth row, §5.4's sweep | `edge_k/notebooks/E1`–`E4`, then `figures/make/fig_5_4.py --edge-arm <arm>/per_fold.csv` |
| Table 5's operating point, Figure 9 | `rerun_canonical.py` regenerates the sweep via `src/budget_sweep_11.py`, then `figures/make/fig_5_5.py --canonical <run>` |
| The tuned tree baseline | `notebooks/15_workstation_session.ipynb`. **The only current output of that notebook** — see the warning below |
| Figures 1, 2 and 3 | `figures/make/fig_method.py`, schematics, no data required |

`15_workstation_session.ipynb` is shipped for one reason: it produced the gradient-boosted
baseline, which is deterministic, was never re-run, and is still the reported one at 0.636.
It is the only executed record of how that number was made. Its other outputs are superseded.
The notebooks that produced *only* superseded results are not shipped — see the drop list.

Each figure script takes a search root and an output directory.

    python protocol_v2/figures/make/fig_5_3_v2.py protocol_v2/results/ figures/

The scripts assert their inputs against the values in the paper and refuse to draw if they
disagree, which is deliberate. An earlier version silently loaded a superseded study and produced
a plausible but wrong Figure 5.

### Warning: four notebooks write pre-rename filenames

Every superseded result file in the archive carries a `_SUPERSEDED` suffix, so that a filename
says whether it can be cited. **Four notebooks predate that convention and still write the
unsuffixed names.** Re-running one recreates a file that looks current and is not:

| notebook | writes, unsuffixed | which is superseded by |
|---|---|---|
| `15_workstation_session.ipynb` | `gnn_11_converged.csv`, `budget_sweep_11feat_matched.csv`, `depth3_11.csv` | the canonical run. **Its `xgb_*` outputs are current and are the reported baseline** |

Three other notebooks with the same problem — `08b`, `12`, `13` — produced only superseded
results and are not shipped with the paper code. See the drop list at the end of this file.

They are not rewritten because they are executed records of what was run, and editing their
source to claim they wrote `_SUPERSEDED` files would misrepresent that. If you re-run one,
rename its outputs yourself or write them to a scratch directory.

Figures 4 to 9 redraw from `results/` alone, with no GPU and no retraining. Only the numbers
themselves need the notebooks.

## Tests

    PYTHONPATH=src pytest -q tests/                      # 10 tests, the causality guarantee
    PYTHONPATH=src:protocol_v2 pytest -q protocol_v2/tests/   # 30 tests, the protocol package
    PYTHONPATH=src:edge_k pytest -q edge_k/tests/             # 21 tests, the edge and K package

`PYTHONPATH` is required: the packages import `src` modules by bare name. Two of the
`protocol_v2` tests are environment-dependent and fail outside the machine that produced the
results — one compares a recorded absolute path against the current tree, and one asserts that
no notebook has been executed, which is false in a shipped archive.

`tests/` covers construction causality, rejection of a reversed future edge, rejection of an
appended future edge, self-loop exemption, contemporaneous-edge counting, strict mode, absence of
ties under strictly increasing timestamps, the empty graph, the per-relation neighbour cap
reaching an in-degree of twenty, and the placeholder-entity behaviour.

## Hardware

Two RTX 6000 Ada (48 GB each), one fold per device. A full-graph forward pass with gradients does
**not** fit at 500k nodes, which is why training uses the exact L-hop expansion above.

## Known issues

Three issues listed in earlier versions of this file have since been **closed**, and are
recorded here so that an older copy of the README can be recognised:

- ~~No validation curves exist for the reported run.~~ **Closed.** `rerun_canonical.py` sets
  `val_curve_path` for every arm, so `canonical/main/val_curves.csv` and
  `canonical/main/per_fold.csv` describe the same execution. Figure 7(b) no longer plots a
  different run from Table 4.
- ~~Single seed throughout.~~ **Closed.** The reported configuration was trained four times at
  seeds 42 to 45. The standard deviation is 0.010, and every variance scale in the paper derives
  from it.
- ~~The neighbour cap and the edge feature set were never swept.~~ **Closed.** `edge_k/` sweeps
  the cap at K = 5 and K = 20 and ablates the edge encoding. Section 5.4 and Table 8 report all
  three.

Open:

- **Two copies of `per_week_all_arms.csv` exist in the wider project tree**, one per protocol
  generation. `protocol_v2/figures/make/fig_5_3_v2.py` asserts against Table 6 and refuses
  mismatched input. `legacy/fig_5_3_SUPERSEDED.py` asserts against the 75-epoch table instead
  and must not be used — see `legacy/README.md`.
- **The edge encoding is bounded jointly, not decomposed.** All five edge features reach the
  attention through one additive term, so the ablation bounds their contribution together and
  apportions nothing among them. Three further arms would separate them — `no_time_terms`,
  `no_amount_ratio` and `no_relation_ind`, fully specified in `edge_k/config.py`'s `ARM_SPEC`
  with their `zero_cols` and `runnable: False`. They are built and gated; they were not run,
  and their notebooks are not shipped with the paper code.
- **One arm was resumed.** `depth3` exhausted a 48 GB device on the largest fold and was
  resumed for that fold alone, from a fresh generator state. Same configuration and seed, drawn
  at a different point in the stream. Section 4.4 discloses it.

## Citation

Manuscript under review; citation details to follow. Machine-readable metadata for this
archive is in `CITATION.cff`. Please cite the manuscript, not this repository, once it appears.

## Data source

Shenoy, K. (2020). *Credit card transactions fraud detection dataset* [Data set]. Kaggle.
<https://www.kaggle.com/datasets/kartik2112/fraud-detection>

---

## What is not here, and what you lose

This is the **paper code**: the code that produced a number, table or figure in the
manuscript, plus what is needed to run it and to check that its provenance claims hold.
Thirteen files from the fuller code release were dropped.

| dropped | what it was | what you lose |
|---|---|---|
| `notebooks/08b_protocol_11feat.ipynb` | the 75-epoch protocol study | the executed record of a study fully replaced by `protocol_v2/`. Its four output files are all marked `_SUPERSEDED` in the results archive |
| `notebooks/09_alert_budget_sensitivity.ipynb` | the earlier budget sweep | nothing reported. The sweep behind Figure 9 is regenerated by `rerun_canonical.py` via `src/budget_sweep_11.py` |
| `notebooks/10_matched_feature_comparison.ipynb` | an earlier matched comparison | nothing reported; Table 4 comes from the canonical run |
| `notebooks/12_convergence_run.ipynb` | the earlier convergence run | nothing reported. Figure 7(b) plots the canonical run's own `val_curves.csv` |
| `notebooks/13_arch_ablation_11feat.ipynb` | the earlier architecture ablation | nothing reported. Its depth-3 arm read **+0.0237** where the canonical run reads −0.0029 — the sign flip that motivated the re-run |
| `edge_k/notebooks/E5`, `E6`, `E7` | the three feature-drop arms | nothing — they were never executed. Their full specification survives in `edge_k/config.py`'s `ARM_SPEC` |
| `edge_k/make_notebooks.py`, `protocol_v2/make_notebooks.py` | notebook generators | the ability to regenerate the arm notebooks identically. The notebooks themselves are shipped |

*(`edge_k/status.py` and `edge_k/monitor.py` were on the drop list as progress writers and
are in fact imported by `edge_k/runner.py`, `controls.py` and `run_arm.py`. They are shipped.)*
| `src/redundancy_diagnostic.py` | a diagnostic against pre-engineered columns of a different corpus | nothing. **This paper uses Sparkov only.** It was present only because `src/run_experiment.py` imported it at module level; that import is now lazy, inside the one function that uses it |

**Kept deliberately, though no number depends on them:** the eight guard, lock and
provenance modules, because §5.3's claim that the model was held fixed while the protocol
varied rests on `protocol_v2/guards.py` writing `canonical_p3_lock.json`, §5.4's
pre-declared 0.023 rests on `edge_k/analysis_settings.py` being hashed before training, and
`edge_k/controls.py` is what proved the ablation actually disconnected `edge_attr`. The eight
tests are kept for the same reason: `tests/test_causal_guard.py` is the only artefact that
demonstrates the leakage guarantee independently of any result file, and it runs after cloning
with no corpus and no GPU. And `legacy/`, three clearly labelled files that let someone
holding an older copy identify what they have.
