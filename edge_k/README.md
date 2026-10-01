# edge_k — edge-encoding and neighbour-cap arms

**Built, tested, gated. Nothing has been trained. No arm has been executed.**

Implements `paper/EDGE_K_WORK_PLAN.md`. Four authorised arms and three scaffolded ones,
all 17 folds, seed 42, 200 epochs, 11 features, the canonical configuration with exactly
one named change each.

Nothing here trains, overwrites or writes to any canonical or protocol_v2 arm. Four
independent locks enforce that; see `guards.py`.

---

## The arms

| | Arm | Notebook | The one change | RF | Hours | Projected VRAM |
|---|---|---|---|---|---|---|
| B | `K5_both` | `E1_K5` | `max_prior_neighbors` 10 → 5 | 111 | 2.0 | 1.63 GB |
| C | `uid_K20` | `E2_uid_K20` | `entity_cols` → `("uid","__none__")` **and** K → 20 | 421 | 7.0 | 4.14 GB |
| A | `no_edge_bias` | `E3_no_edge_bias` | `ModelConfig.use_edge_bias` → False | 421 | 8.7 | 5.29 GB |
| D | `K20_both` | `E4_K20` | `max_prior_neighbors` 10 → 20 | 1,641 | 25.5 | 16.30 GB |

**Run order: E1, E2, E3, then the decision point, then E4.** One notebook at a time,
cleanup cell run by hand, kernel down between.

Scaffolded and **not runnable** — they refuse with `NotYetAuthorisedError`:

| Arm | Notebook | Zeroes | Gate |
|---|---|---|---|
| `no_time_terms` | `E5_no_time_terms` | edge cols 0, 1 | \|no_edge_bias effect vs 4-seed mean\| ≥ 0.023 |
| `no_amount_ratio` | `E6_no_amount_ratio` | edge col 4 | same |
| `no_relation_ind` | `E7_no_relation_ind` | edge cols 2, 3 | same |

---

## Layout

```
edge_k/
  config.py              every setting; imports from rerun_canonical.py, restates nothing
  guards.py              the four locks: arm registry, single call site, write fence, checksum
  provenance.py          source hashes, environment, comparison, the analysis lock
  analysis_settings.py   the frozen analysis — declared before any result exists
  controls.py            the positive controls, THE HARD GATE
  runner.py              data loading, preflight, the ONE call to run_date_gnn_fold
  status.py              STATUS.txt writer, so progress is readable with tail -f
  make_notebooks.py      regenerates all seven notebooks from one template
  notebooks/             E1-E7, all shipped unexecuted with no stored output
  tests/                 three files, all CPU, all fast
  results/               created by the runs; nothing here yet
```

---

## Before running anything

```bash
cd edge_k
python tests/test_static.py                     # package and notebook shape
python tests/test_config_matches_canonical.py   # the configuration is inherited, not restated
python tests/test_positive_controls.py          # the hard gate, and that it is not vacuous
```

All three are fast, need no GPU and train nothing. All three currently pass.

---

## The one change outside this folder

`src/model.py` gained one optional field, `ModelConfig.use_edge_bias`, defaulting to
`True`. At the default nothing changes: the model is bit-identical to the one every
reported arm used, which `controls.check_edge_bias_flag()` verifies by comparing against a
model built with no flag at all.

At `False` the `Linear(edge_dim, n_heads)` is **not constructed** and the additive term at
the old `model.py:68` is skipped, so the parameter count drops by exactly 48 across two
blocks and `edge_attr` becomes fully disconnected from the output.

The K arms needed no code change at all — `max_prior_neighbors` was already a
`GraphConfig` field.

`src/protocol_runner.py` is not used. The stale fork stays untouched and deprecated.

### Side-effect worth knowing

`ModelConfig` feeds `_rng_state_identity()` in `run_date_gnn.py`, so adding a field
changes the RNG-state configuration fingerprint. Any protocol_v2 resume from an existing
`rng_state.pt` would now be refused. **All four protocol_v2 arms are complete, so no
resume is pending** and nothing is affected in practice. If one ever needs resuming, the
state files predate this change and would have to be regenerated.

---

## Positive controls — measured, not asserted

These are the hard gate. A switch that is not wired produces a clean null that reads as a
finding. Every control reports numbers.

**Control 1 — the edge-bias flag**

| | |
|---|---|
| parameters, flag on | 283,441 |
| parameters, flag off | 283,393 |
| dropped | **48** (expected 48 = 2 blocks × (5×4 + 4)) |
| max abs output difference, on vs off | 0.948048 |
| mean abs output difference | 0.349863 |
| `use_edge_bias=True` reproduces the no-flag model | yes, exactly |
| ∂output/∂edge_attr with the flag **on** | 63.14 |
| ∂output/∂edge_attr with the flag **off** | **None** — autograd finds no path at all |
| `edge_bias` module when off | `None` |

The `None` gradient is stronger than a zero one: it means the edge encoding is not merely
contributing nothing, it is not in the computation graph.

**Control 2 — the zeroed columns.** For all three scaffolded arms: width preserved at 5
(so parameter-matched), exactly the named columns zero, every other column bit-identical,
input never modified. `apply_zero_cols` refuses an out-of-range column and returns a copy.

**Control 3 — K changes the neighbour count.** Built on the real corpus:

| Arm | K | Edges | Per node | Expected | Contemporaneous |
|---|---|---|---|---|---|
| `K5_both` | 5 | 4,975,790 | 9.95 | 9.9 projected | 54 |
| `uid_K20` | 20 | 9,807,302 | 19.61 | **19.61 measured** | 22 |
| `no_edge_bias` | 10 | 9,911,248 | 19.82 | **19.82 measured** | 54 |
| `K20_both` | 20 | 19,661,772 | 39.32 | 39.2 projected | 54 |

`uid_K20` reproduces the July 2026 `edge_relation` run's edge count **exactly**
(9,807,302), which is a hard check rather than a tolerance. `no_edge_bias` reproduces
`run_manifest.json` exactly. Every arm: 0 future edges, `assert_causal` passed.

---

## Dry-run memory projection

Largest fold, batch 2,048, two layers, `BYTES_PER_EDGE_PER_LAYER = 2400`:

| Arm | Edges in batch | Projected peak | +15% headroom |
|---|---|---|---|
| `K5_both` | 340,415 | 1.63 GB | 1.88 GB |
| `uid_K20` | 863,120 | 4.14 GB | 4.76 GB |
| `no_edge_bias` | 1,101,320 | 5.29 GB | 6.08 GB |
| `K20_both` | 3,395,943 | **16.30 GB** | 18.75 GB |

Card reports **48.7 GB free of 50.8 GB**.

**Read the K20_both number with the caveat the constant carries.** protocol_v2 recorded
that 2400 still under-projects — a ratio-matched value would be about 3900 — so the
realistic peak for `K20_both` is nearer **26 GB** than 16 GB. That is still comfortable
against 48.7 GB free, but it is half the card, not a third. Run D on an empty device,
watch `STATUS.txt`, and stop if any fold's peak exceeds 60% of the card.

---

## What is deliberately not built

**Edge-feature zeroing is not wired into training.** `run_date_gnn_fold` builds its edge
features internally and offers no hook. Wiring E5–E7 needs one more optional parameter on
that function — an `edge_feature_transform=None`, inert at its default, in the same style
as the three protocol_v2 added. That change is **not** made here, because the work plan
authorises four arms and these three are gated behind a result that does not yet exist.
The transform itself (`runner.apply_zero_cols`) is implemented and tested, so wiring it
later is a two-line change to `src/` plus clearing the `runnable` flag.

`runner.train_arm` raises `NotImplementedError` if a zeroing arm somehow reaches it, after
`assert_trainable` has already refused it. Two independent refusals, deliberately.

---

## Conventions inherited from protocol_v2

- **No resume, ever.** There is no `resume` parameter. depth3's crash-and-resume became a
  permanent provenance note in the paper. If an arm dies, move its directory to
  `<arm>_ABANDONED_<utc>`, record why in `STATUS.txt`, and start clean.
- **Weights are saved.** The canonical run saved none, which is why the checkpoint-replay
  test could never be run. `SAVE_WEIGHTS = True`.
- **The analysis is frozen first.** `analysis_settings.py` is hashed into
  `results/analysis_lock.json` by the first arm notebook, before it trains. Editing it
  afterwards makes `verify_analysis_lock()` fail.
- **The threshold is 2 SD, not 1.** 0.023 against the four-seed mean. The ±0.014 band in
  Section 5.4 is one SD, at which 32% of true nulls fall outside.
- **Wilcoxon is descriptive only** and gates no decision. See `analysis_settings.WILCOXON`
  for why, and the depth3 episode for the evidence.
- **Both baselines are reported.** Seed 42's 0.8551 is the highest of the four seeds; the
  mean is 0.8472. Using seed 42 alone overstates any loss by 0.0079.
