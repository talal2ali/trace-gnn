# TRACE-GNN

**T**emporal **R**elational **A**ttention with **C**ausal **E**dges — a leakage-safe temporal
graph encoder for transaction-level credit card fraud detection under a fixed alert budget.

Reference implementation and paper code. Pure PyTorch; no PyTorch Geometric.

---

## What the method does

Fraud detection research routinely reports performance that does not survive deployment:
random splits applied to a temporally ordered transaction stream, and threshold-free metrics
reported although screening can review only a small fixed fraction of transactions.

TRACE-GNN enforces leakage safety in **both** places it can be lost — the data split *and* the
graph topology. Each transaction is linked only to its most recent prior transactions on the
same card and, separately, at the same merchant, so no representation can depend on a future
event. The edge encoding enters the attention weight and never the message content.

Evaluated under a rolling-origin protocol with a fixed 0.5% alert budget, across seventeen
weekly folds of the Sparkov dataset:

| | mean average precision |
|---|---|
| per-fold tuned gradient-boosted baseline | 0.636 |
| **TRACE-GNN**, same eleven features | **0.855** |

TRACE-GNN leads on every fold. A sequential comparison attributes **0.120** to representing
cardholder history relationally and a further **0.099** to the merchant relation. Holding the
model fixed and varying only the evaluation protocol, correcting the split alone leaves average
precision inflated by 0.094 and correcting the graph alone by 0.125 — the two leakage routes are
substitutable, the safeguards against them are not. AUROC moves roughly a tenth as much, so it
barely registers the difference.

---

## Repository layout

```
.
├── src/                    the model and the data pipeline
├── rerun_canonical.py      the canonical run — Tables 1, 3, 4, 5, 8; Figures 4, 7, 8, 9
├── resume_canonical.py     resumes an interrupted canonical run
├── notebooks/              the documented interfaces to the above
├── protocol_v2/            the five-protocol study — Tables 6, 7; Figures 5, 6
├── edge_k/                 edge encoding and neighbour cap — Table 8 rows 2, 5, 6
├── figures/make/           the figure scripts
├── tests/                  61 tests, CPU-only, no corpus required
├── legacy/                 two superseded scripts, clearly labelled
└── results/                two baseline files the code reads (see below)
```

### Folders

| folder | what is in it |
|---|---|
| `src/` | 18 modules: the TRACE-GNN architecture (`model.py`), its training loop (`run_date_gnn.py`), causal and acausal graph construction, rolling-origin folds, the eleven features, per-fold preprocessing, evaluation at the alert budget, and the Sparkov preprocessing stage |
| `notebooks/` | `00_run_preprocessing` builds the corpus · `16_canonical_rerun` drives the canonical run · `15_workstation_session` produced the gradient-boosted baseline |
| `protocol_v2/` | its own runner, config, splits, guards and provenance, plus notebooks `A1`–`A5`: four arms then assembly |
| `edge_k/` | its own runner, config, guards, positive controls and provenance, plus notebooks `E1`–`E4`, one per arm |
| `figures/make/` | `fig_method.py` (Figures 1–3) · `fig_5_1.py` (4) · `fig_5_4.py` (7, 8) · `fig_5_5.py` (9). Figures 5 and 6 come from `protocol_v2/figures/make/fig_5_3_v2.py` |
| `tests/` | `test_causal_guard.py` demonstrates the leakage guarantee with no corpus and no GPU |
| `legacy/` | kept so anyone holding an older copy can identify it. Not for use |
| `results/` | **two input files, not an archive of results** — see `results/README.md` |

### Files at the top level

| file | what it is |
|---|---|
| `README.md` | this file |
| `CODE_MAP.md` | **every table and figure in the paper → the code that produced it**, with the command |
| `REPRODUCE.md` | step-by-step reproduction |
| `ENVIRONMENT.md` | exact versions, hardware, and the determinism caveat |
| `MANIFEST.md` · `READING_MAP.md` | provenance records — which result file backs which number |
| `VERSION.md` | which manuscript draft this snapshot corresponds to |
| `requirements.txt` · `LICENSE` · `CITATION.cff` | pinned versions, MIT licence, citation metadata |

---

## Quick start

**1. Install.** `torch` comes from pytorch.org, not from `requirements.txt`:

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

**2. Run the tests.** No data, no GPU, about a minute:

```bash
PYTHONPATH=src             pytest -q tests/
PYTHONPATH=src:protocol_v2 pytest -q protocol_v2/tests/
PYTHONPATH=src:edge_k      pytest -q edge_k/tests/
```

**3. Get the data.** Sparkov, from
[Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — the only dataset this
paper uses:

```bash
python src/prepare_sparkov.py --raw <download_dir> --out <path>/prepared
```

**4. Check before committing hours:**

```bash
python rerun_canonical.py --prepared <path>/prepared --dry-run
```

Loads the corpus, builds the folds and the graph, runs every assertion, projects peak VRAM,
then stops without training.

**5. Train:**

```bash
python rerun_canonical.py --prepared <path>/prepared --repeats 3
```

See `CODE_MAP.md` for the command behind each individual table and figure.

---

## What to expect from a re-run

Message passing accumulates neighbour contributions with scatter operations that are not
associative in floating point on GPU, so **a fixed seed does not give bit-identical results**.
Four repeats at seeds 42–45 give a standard deviation of 0.010 in mean average precision.
Expect a re-run to land within roughly 0.02 of 0.855, not on it. The gradient-boosted baseline
is deterministic and reproduces exactly.

The canonical run needs **2 × 48 GB GPUs** and roughly 56 hours for four arms plus three
repeats. `ENVIRONMENT.md` gives the exact environment and what does and does not need a GPU.

---

## Data

Not redistributed. Sparkov is public and synthetic, generated by a simulator — no real
cardholder data is involved.

Shenoy, K. (2020). *Credit card transactions fraud detection dataset* [Data set]. Kaggle.
<https://www.kaggle.com/datasets/kartik2112/fraud-detection>

---

## A note on naming

The paper says **TRACE-GNN**. The code still says **DATE-GNN** (`date_gnn`, `DateGNN`,
`config="date_gnn_v3"`). Same model, same weights, same results. The original name was dropped
because *DATE* collides with Kim et al. (KDD 2020), a customs-fraud model that also ranks under
a fixed inspection budget.

---

## Citation

Manuscript under review; citation details to follow. Machine-readable metadata is in
`CITATION.cff`. `VERSION.md` records which manuscript draft this code corresponds to.

## Licence

MIT — see `LICENSE`.
