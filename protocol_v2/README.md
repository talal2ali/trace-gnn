# protocol_v2 — four arms, retrained at 200 epochs

**Nothing here has been run. No training has happened. No notebook has been executed.
No existing result folder has been touched — verified by a recursive checksum of
`results/protocol/`, `results/per_fold/`, `results/architecture/`, `results/convergence/`,
`results/budget/`, `results/rerun_20260826_114717_m5/` and `notebooks/` before and after
this work, all seven unchanged.**

Two files under `src/` *were* changed, deliberately and as instructed — see
"The one code change outside this folder" at the bottom.

---

## The problem, in four lines

Section 5.3's five-arm protocol study was trained at 75 epochs. Section 5.1's canonical
run was trained at 200. `CosineAnnealingLR` anneals over `T_max = epochs`, so a 75-epoch
run is not a shortened 200-epoch run — it follows a different learning-rate path from the
first step. They are two configurations, not two lengths of one.

## The fix

Keep the canonical Section 5.1 run and use it *as* the `P3_rolling` arm. Retrain only the
other four, at 200 epochs.

| arm | retrained here? | why |
|---|---|---|
| `P3_rolling` | **no** | it *is* the canonical Section 5.1 run — same settings, same folds, already trained |
| `P2_cutoff` | yes | ~0.3 h |
| `P1_causal` | yes | ~0.8 h |
| `P1_leaky` | yes | ~0.8 h |
| `P3_acausal` | yes | ~8.8 h |
| | | **~10.7 h** |

`P3_acausal` is retrained. `P3_rolling` is not. The names are kept distinct everywhere in
this package for exactly that reason.

---

## Everything is in this one folder

```
protocol_v2/
  README.md                  this file
  config.py                  every setting, in one place
  guards.py                  the P3_rolling lock and the write fence
  provenance.py              source hashes, environment, comparison against canonical
  splits.py                  the fold sets — built once, saved, reused
  runner.py                  data loading, preflight, THE single training call
  assembly_settings.py       analysis rules, frozen before any arm was trained
  assemble.py                the tables
  make_notebooks.py          regenerates the five notebooks
  notebooks/
    A1_P2_cutoff.ipynb       0.3 h   cheapest, proves the pipeline
    A2_P1_causal.ipynb       0.8 h   first use of the random-split folds
    A3_P1_leaky.ipynb        0.8 h   first use of the acausal graph
    A4_P3_acausal.ipynb      8.8 h   smoke run over weeks 8-9, then the rest
    A5_assemble.ipynb        no GPU
  tests/
    test_p3_rolling_locked.py        verification item (a)
    test_rng_resume.py               verification item (b)
    test_default_path_unchanged.py   the code-path claim
    test_static.py                   notebook shape, epochs, no restated settings
  results/                   created by the runs; nothing here yet but the lock file
```

**Not touched by anything here:** `results/protocol/`,
`results/rerun_20260826_114717_m5/`, `experiment_results/protocol_11/`, `notebooks/`,
`paper/`. Delete `protocol_v2/` and the project is exactly as it was.

---

## Before you run anything

```bash
cd protocol_v2
python tests/test_static.py
python tests/test_p3_rolling_locked.py
python tests/test_rng_resume.py
python tests/test_default_path_unchanged.py
```

All four are fast, need no GPU, and train nothing.

## Then, one notebook at a time

Open A1. Run it. **Run the cleanup cell at the bottom yourself.** Shut the kernel down.
Open A2. And so on. Nothing frees the GPU automatically — that was deliberate, so you can
still inspect the model after a run. The one automatic thing is the first cell, which
prints how much GPU memory was free at the start so the cleanup has a number to compare
against.

If you skip the cleanup, the next arm fails on a memory error partway through. That costs
an hour on A2 or A3 and up to nine on A4. A4 runs last so the habit forms on the cheap
arms.

---

## The four things to check before authorising a run

### (a) `P3_rolling` cannot be retrained or overwritten

Four independent locks. `python tests/test_p3_rolling_locked.py` checks all four.

| # | lock | where to look |
|---|---|---|
| 1 | no specification exists for it | `config.py`, `ARM_SPEC` — four entries, and the `assert` under it |
| 2 | one guarded training call site | `runner.py`, `train_arm` — first statement is `guards.assert_trainable(arm)`; it is the only call to `run_date_gnn_fold` in the package |
| 3 | writes are fenced | `guards.py`, `assert_writable` — refuses every path outside `protocol_v2/results/`, which puts the whole existing `results/` tree out of reach |
| 4 | the directory is checksummed | `guards.py`, `lock_canonical` / `assert_canonical_unchanged`; recorded in `results/canonical_p3_lock.json`, verified at the top **and** bottom of every notebook |

The recorded lock, taken before any of this ran:

```
results/rerun_20260826_114717_m5/canonical/main/
  5 files, combined SHA-256  4431eca212299cd5...
  see protocol_v2/results/canonical_p3_lock.json for the per-file hashes
```

**Nothing here chmods the canonical directory.** An earlier draft did, as belt-and-braces.
It was removed: chmod is a write to the canonical arm, it bypassed the fence (a mode
change is not a write `assert_writable` sees), it was invisible to the checksum, and
nothing would have undone it — a later legitimate `rerun_canonical.py --mode resume` on
that arm would then have failed. The reasoning is written out in `guards.py`, under
"deliberately NOT done: chmod", with the two shell commands if you ever want it done by
hand.

### (b) The A4 smoke/resume preserves the random state

**This needed a code change, and it is the one place I went beyond what the review asked
for. Read this bit properly.**

`run_date_gnn_fold` seeds once at the top and then loops over folds. On a resume it seeds
*again* and skips the completed folds — so folds trained after a resume see the stream a
*fresh* run would have given fold 1, not the stream a continuous run would have reached.
This is not hypothetical: `run_manifest.json` says exactly this about `depth3` week 24,
in its `provenance_note`.

For A4 that matters, because the smoke run over weeks 8–9 is followed by a resume for
10–24. Without a fix, those fifteen folds would be trained on a stream no uninterrupted
run ever produces.

So `run_date_gnn_fold` gained a third optional parameter, `rng_state_path`. With it set,
the generator state (numpy, torch CPU, torch CUDA) is saved at every fold boundary and
restored on resume.

**One entry per fold is kept, not just the newest.** `foldckpt.csv` and the state file are
two separate writes, so a crash between them leaves them one fold apart — and with only
the newest state kept, *neither* direction was recoverable. With every fold's state on
file, whatever the checkpoint says finished has a matching state to resume from. Extra
entries, from a fold whose state was written before the crash took the checkpoint write,
are ignored. It costs about 8 KB a fold.

* **the restore:** `src/run_date_gnn.py`, the block headed `RNG continuity across a
  resume`, just before the fold loop
* **the save:** `_save_rng_state`, defined below it, called once at the bottom of the
  fold loop — after the fold's scores, checkpoint and validation curve are written, and
  before the next fold's first draw
* **the guard:** `check_rng_state`, a module-level function so it can be tested directly.
  It raises — never warns and continues — on any of these:

  | it refuses when | what is compared |
  |---|---|
  | the state file does not cover every fold `foldckpt.csv` lists | `states` keys ⊇ checkpointed folds |
  | the configuration differs | a fingerprint over `TrainConfig`, `ModelConfig`, `GraphConfig`, the feature list, **the edge builder** and **the causality flag** |
  | the fold set differs | a **per-fold** SHA-256 of that fold's train and test index arrays |
  | the CUDA device count differs, or CUDA state is absent on a CUDA machine | `n_cuda_devices`, `torch_cuda` |
  | the file is an old version-1 state | `version` |

  The fingerprint deliberately does **not** include the fold list, and the fold check is
  per fold rather than over the whole list. That is what lets a 2-fold smoke state be a
  valid predecessor of the 17-fold run while still refusing a state file from a different
  arm. `P1_causal` and `P2_cutoff` share every config and every builder and are told
  apart here and only here.

  **This is where the first draft was wrong, and where the README was worse than the
  code.** Version 1 hashed only the configs and the feature list. Since `configs_for()`
  never reads the arm name, all four arms produced an identical fingerprint — the check
  could not tell one arm from another. The README nevertheless said it refused a state
  from "a different arm". It did not. That sentence is what a reviewer caught, and the
  wording was the worse half of the mistake.
* **the tests:** `tests/test_rng_resume.py` — split-then-resume produces the identical
  stream and differs from a naive reseed; the saved payload survives a save/load round
  trip; `check_rng_state` is called with each of ten bad states and must raise on
  every one

**Two things I want to be explicit about.**

1. The review said the code change should shrink from three parameters to two. I added a
   third anyway. `rng_state_path` is a different thing from the `row_index` parameter the
   review rejected — that one was about the results schema, this one is about the resume
   the review itself asked for — but it is still one more parameter than was agreed, and
   you should decide whether you want it. Default is `None`, which is the old behaviour
   exactly. Dropping it means the A4 resume restarts the stream, as the canonical run's
   `depth3` did.
2. This preserves the **stream**, not the **bits**. GPU scatter-add accumulates in
   nondeterministic order; that is why this project measures SD 0.0091 in mean AP across
   executions of one configuration. Nobody can promise bit-identity here and I am not.

### (c) A2 and A3 load the identical P1 split

The old notebook built `folds_p1` once and passed the same object to both arms. Separate
notebooks in separate processes cannot do that, so the guarantee is now on disk.

* `splits.py`, `build_splits` — construction, identical to `08b_protocol_11feat.ipynb`
  cell 3: `default_rng(42).permutation(universe)`, 80/20, both halves sorted
* `splits.py`, `ensure_splits` — builds and saves on the first notebook that runs, loads
  and hash-verifies on every call after
* `results/splits/p1_split.npz` — the actual row indices
* `results/splits/SPLITS_MANIFEST.json` — `train_sha256`, `test_sha256`, `pair_sha256`

**How to check it in ten seconds:** step 2 of A2 and step 2 of A3 each print

```
P1 split     pair sha256 <64 hex chars>
```

The two strings must be identical. The value is **recomputed from the arrays just loaded
off disk**, not read out of the manifest — an earlier draft printed the manifest's own
field, which would have been identical in both notebooks whatever the arrays contained,
and would have proved nothing. `load_splits` verifies all five hashes and raises before
any training if one fails.

`ensure_splits` also refuses to act on a half-present `splits/` directory: regenerating
over an existing `p1_split.npz` would be silent, and "it would probably have come out the
same" is not the guarantee this is for.

Also true, and worth knowing rather than checking: the validation rows follow
deterministically from the training rows (`tr[:-n_val], tr[-n_val:]` in
`run_date_gnn.py`), so the same split necessarily means the same validation set. No
randomness is involved.

### (d) All four arms use the canonical 200-epoch configuration

Nothing in this package restates a hyperparameter. `config.py` **imports**
`EXPECTED_TRAIN` and `EXPECTED_MODEL` from `rerun_canonical.py` and asserts
`epochs == 200` at import time, so a drift anywhere fails on the first line of every
notebook.

* `config.py` — `TRAIN_KWARGS = dict(_CANON.EXPECTED_TRAIN)`, then the assertions below it
* `config.print_summary()` — step 3 of every arm notebook prints the resolved block
* `runner.configs_for` — asserts every field against `TRAIN_KWARGS`, then
  `assert tcfg.epochs == 200`
* `tests/test_static.py` — `test_config_is_the_only_place_settings_live` fails if any
  notebook passes a hyperparameter as a literal; `test_epoch_budget_is_200_everywhere`
  fails if `75` reappears anywhere

---

## What the review asked for, and where it is

| review item | done | where |
|---|---|---|
| D1 drop "byte-identical output" | yes | narrower code-path claim, in `run_date_gnn.py`'s docstring and `tests/test_default_path_unchanged.py` |
| D2 A4 smoke run, then resume | yes | `A4`, steps 4b/4c/5; `config.SMOKE_WEEKS` |
| D3 explicit do-not-retrain-P3 + checksum | yes | `guards.py`; `results/canonical_p3_lock.json` |
| D4 SHA-256 of all source files | yes | `provenance.source_hashes()`, in every arm manifest |
| D5 compare the environment, don't just freeze it | yes | `provenance.compare_environment()`, printed in step 1 |
| D6 do not let 75-epoch scores into the new tables | yes | `assembly_settings.DO_NOT_USE` |
| D7 row identifiers in post-processing, not a parameter | yes | no `row_index` parameter; `splits.week_lookup` + `assemble.load_scores` |
| A1/A2 P1 split generated once and reused | yes | `splits.py`, and (c) above |
| A4 one shared `config.py` | yes | `config.py` |
| A5 state the fold-set rule | yes | `assembly_settings.FOLD_SET_RULE`, applied by `assemble.weeks_for` |
| A6 mark `protocol_runner.py` deprecated | yes | banner + `DeprecationWarning` in `src/protocol_runner.py` |
| C2 freeze and hash the analysis code | yes | `provenance.write_analysis_lock()`, called by the first arm notebook before it trains; `verify_analysis_lock()` in A5 |
| Part 6 predeclare the Wilcoxon settings | yes | `assembly_settings.WILCOXON` |

### One item could not be done, and I am not going to pretend otherwise

**C3, the checkpoint-replay equivalence test.** The review called this the single best
item in the whole document: load a canonical per-fold checkpoint, run inference through
the modified code, compare against the retained canonical scores. It cannot be done.
**The canonical run saved no weights.** `save_weights` was off for all four canonical arms
and all three repeats; there is not a single `.pt` file anywhere under `results/`. The
review assumed they existed. They do not.

What is there instead is weaker, and I want that on the record. All of it is static
analysis of the patched file — none of it runs the training function:

* every read of a new parameter is inside a construct that is inert at its default,
  checked on the syntax tree rather than by matching text, and printed line by line
* the two new lines that run *unconditionally* are named explicitly, so they are not a
  blind spot
* the resolution expression is **lifted out of the file** and evaluated in the module's
  own namespace: at `edge_builder=None` it must be `build_temporal_edges` itself
* the `if assert_causality:` branch must contain exactly `assert_causal(ei_np, times)`
  and nothing else

An earlier draft of that last part claimed "a numerical check that the default builder
builds byte-identical edges on real data". It did no such thing — it called
`build_temporal_edges` twice and compared the results, which shows only that the function
is deterministic. That claim is gone.

The four new arms **do** save weights (`config.SAVE_WEIGHTS = True`, ~80 MB total), so
the replay test will be available for any future change to the code.

---

## Things that are deliberately not consistent

1. **The five rungs come from two sessions.** Four are new; `P3_rolling` is the August
   run. Every setting is matched — epochs, schedule, folds, seed, features, slice, loss,
   top-k rule, tie-breaking, and the training function itself. The sitting is not, and the
   driver version may not be. Both sessions are recorded in the manifest. This is the
   price of not retraining P3, and it is smaller than the problem it avoids.

2. **The canonical run used `cuda:0`; these use `cuda:1`.** Both cards are RTX 6000 Ada.
   `config.DEVICE` is the one place to change it, and
   `provenance.compare_environment()` reports it as a difference rather than hiding it.

3. **One model versus seventeen.** A1, A2 and A3 train one model; A4 trains seventeen.
   That is the protocol difference those arms exist to test, so it is intended — but the
   arms are not comparable in how much training they received, and the random stream is a
   different length in each.

4. **The validation tail means different things in different arms.** For A4 the last 15%
   of the training rows is the latest 15% by time. For A2 and A3 the training set is a
   random sample of the whole period, so its "last 15%" overlaps in time with the test
   rows. Following the review, this is recorded as a **specification detail** of using a
   random split, not as a third leakage mechanism — the validation rows are a subset of
   training rows that are already contemporaneous with the test rows, so nothing new
   enters. It is written into `SPLITS_MANIFEST.json` so anyone reconstructing P1 knows.

5. **AUROC is computed afterwards**, in A5, from the saved scores. In the 75-epoch study
   it was computed inside the training loop. The number is the same; moving it out is what
   lets the training code stay identical to Section 5.1.

6. **`scores.csv` row positions.** All four new arms keep Section 5.1's schema, where
   `row_pos` is a per-fold counter. The old `scores_P3_rolling.csv` stored global
   positions. The two files are not in the same format. A5 recovers the global row, and
   hence the week, from the saved test-index arrays.

---

## If something crashes

Every arm writes a checkpoint after each fold, so a crash costs one fold, not the arm.
The single-fold arms (A1, A2, A3) cannot be resumed part-way, but they take under an
hour — just rerun them. Only A4 is long enough to matter, and it is two layers deep, not
three; the August crash was a three-layer arm.

To resume A4: rerun the notebook with `resume=True` already set in step 5. It will find
`foldckpt.csv`, skip the completed folds, and restore the RNG state. If the checkpoint and
the RNG state disagree, it refuses to start — which is the right answer, not a bug.

Nothing is ever deleted for you. A stale checkpoint raises `FileExistsError` with an
explanation; you decide what to do about it.

---

## The one code change outside this folder

`src/run_date_gnn.py`, `run_date_gnn_fold`, gained three optional keyword parameters at
the end of the signature:

| parameter | default | at the default |
|---|---|---|
| `edge_builder` | `None` | uses `build_temporal_edges`, exactly as before |
| `assert_causality` | `True` | runs `assert_causal`, exactly as before |
| `rng_state_path` | `None` | no state written or read, exactly as before |

Plus one existing line changed shape: the resume key is now cast with `int()`
(defect B5), which is what lets a non-integer fold label work at all.

`src/protocol_runner.py` is marked deprecated (module banner plus a `DeprecationWarning`
on call) and is not used by anything here.

---

## The sign convention, and the number it protects

Declared once, in `assembly_settings.SIGN_CONVENTION`, and derived from nowhere else.

Corrections start from **`P1_leaky`** — the corner where both problems are present — and
move toward `P3_rolling`:

```
single correction, graph   P1_leaky - P1_causal    = +0.0028
single correction, split   P1_leaky - P3_acausal   = +0.0193
sum of the two                                     = +0.0220
total protocol effect      P1_leaky - P3_rolling   = +0.1261
NON-ADDITIVE EXCESS        total - sum             = +0.1040
```

which reduces to `P1_causal + P3_acausal − P1_leaky − P3_rolling`. **Positive means
fixing either defect on its own recovers far less than fixing both** — the paper's claim.

`assemble.excess_per_week()` builds that vector week by week from the declared terms;
`non_additivity()` reports its mean as the aggregate and runs the Wilcoxon on the same
vector, asserting the two agree to 1e-12 and cross-checking against
`total − (the two single corrections)`. There is one signed quantity, not two.

**Two things this replaced.** The first draft summed the two effects measured *from*
`P3_rolling` — 0.2301 instead of 0.0220 — giving the right magnitude with the opposite
sign, −0.104 where the paper publishes +0.104. And `notebooks/08b_protocol_11feat.ipynb`
is itself inconsistent: its aggregate prints +0.1040 while its per-week vector averages
−0.1040. Two-sided p-values are unaffected either way, so **no published number is
wrong** — but neither problem survives into this package.

Fed the shipped 75-epoch table, the new code reproduces every published Table 7 value:

| effect | weeks | computed | manuscript |
|---|---|---|---|
| graph effect under rolling origin | 17 | +0.1005 | +0.101 |
| graph effect under random split | 16 | +0.0028 | +0.003 |
| split effect, causal graph | 16 | +0.1233 | +0.123 |
| split effect, unconstrained graph | 16 | +0.0193 | +0.019 |
| total protocol effect | 16 | +0.1261 | +0.126 |
| cost of not retraining | 17 | −0.1030 | −0.103 |
| **non-additive excess** | 16 | **+0.1040** | **+0.104** |

The declared fold-set rule — 16 weeks when a P1 arm is involved, 17 otherwise — is what
produces the 17-week figures, and it matches what the manuscript printed. This is a check
on the code, not a new result: those inputs are the superseded 75-epoch arms and may not
enter any new table.

---

## Defects found in this package by an independent audit, and fixed

Three adversarial reads found twenty-one real problems between them, two of which would
have stopped A5 running at all. Every one is fixed. They are listed because the fixes are
the interesting part of the diff, and because several were mistakes in this README rather
than in the code.

| what was wrong | fix |
|---|---|
| `freeze_canonical_permissions()` chmod'd the canonical arm — a write to it, bypassing the fence, invisible to the checksum, and irreversible | removed; the reasoning is kept in `guards.py` |
| a CUDA device-count mismatch on resume warned and carried on | `check_rng_state` raises |
| a state file with no CUDA state was accepted silently on a CUDA machine | raises |
| the state file recorded only its fold list, so one from another arm or seed could be accepted | a configuration fingerprint is stored and re-checked |
| the stale-file check on a fresh run missed `rng_state.pt` and `weights/` | both included |
| `ensure_splits` keyed only on the manifest, so a missing manifest silently overwrote an existing split | refuses to act on a half-present `splits/` directory |
| the P1 hash the notebooks printed came from the manifest, so it was identical by construction and proved nothing | recomputed from the loaded arrays |
| `week_lookup` silently defaulted an unknown arm name to the P3 fold set | raises |
| `array_sha256` folded the array dtype into the digest, so an int32 platform would fail verification | everything cast to int64 first |
| the peak-VRAM estimate failed silently into a bare `except` | loud, and recorded in the arm manifest as `vram_check_skipped` |
| two tests and three README passages overstated what they checked | rewritten; see (b) and (c) above |

### Second round

| what was wrong | fix |
|---|---|
| **the non-additive excess summed the wrong pair of contrasts, giving the right magnitude with the opposite sign** | one declared convention; aggregate and per-week vector derived from it and asserted equal. See the section above. |
| A5 would assemble whatever it found — a smoke-only arm gave a silent two-week ladder | `assemble.assert_arms_complete()` and `assert_week_coverage()`, both called before any table is built |
| a crash between the bulk writes and the checkpoint duplicated score rows, or lost validation curves for good | `foldckpt.csv` is now written last, and `runner.reconcile_after_crash()` drops orphaned rows on resume, keeping a `.pre_reconcile` backup |
| a row-count mismatch was logged as a "problem" that nothing read | `finalise` writes everything, then raises; `complete` in `summary.json` is now `not problems`, and A5 checks it |
| the RNG fingerprint could not tell one arm from another, and the README claimed it could | fingerprint covers the builder and causality flag; fold sets checked per fold; README rewritten |
| `assemble.py` — the analysis itself — was not in the source hashes | added, and `analysis_lock.json` is written by the first arm notebook *before* training and verified by A5 |
| `run_manifest.json` was hashed at lock time and never re-checked | `assert_canonical_unchanged` re-checks it |
| **`src/config.py` shadowed `protocol_v2/config.py`** — the notebooks put `src/` on `sys.path` first, so `import config` returned the Sparkov prep config and cell 1 would have died on the first run | path order reversed, plus an assertion naming the collision |

### Third round — two of these were blockers

| what was wrong | fix |
|---|---|
| **A5 could never have run.** `per_week()` grouped over every week in a scores file, but P1's universe includes the 148,946 rows before week 8, and those weeks contain fraud (13, 19, 12, 16, 34, 8, 11, 14 cases in weeks 0–7). So the P1 arms reported 24 weeks, the new `assert_week_coverage` gate fired on every run, and pre-week-8 rows would have leaked into `per_week_all_arms.csv`. Notebook 08b had this filter; the first draft dropped it. | `per_week()` restricts to the 17 evaluation weeks |
| **the frozen Wilcoxon block asserted something false, and its fallback made the published number unreachable.** It said "no zero differences occur". `P1_leaky − P1_causal` is exactly zero in weeks 10 and 21, 2 of 16 — so the declared fallback fired on the one contrast the manuscript quotes and gave p = 0.30029 where the paper prints 0.326. | `method="exact"` unconditionally, no fallback; `n_zero` and `n_effective` reported alongside `n_weeks`; raises rather than downgrading if n ever exceeds 25 |
| a crash between the checkpoint write and the RNG write left them one fold apart, which `check_rng_state` then refused — losing the arm | the state file keeps **one entry per fold**, so whatever the checkpoint says finished has a matching state. A state one fold *ahead* now resumes cleanly; one fold *behind* still raises, correctly |
| `.pre_reconcile` was written only if absent, so a second reconcile lost its dropped rows | numbered backups, `.pre_reconcile.1`, `.2`, … |
| an empty or header-only `foldckpt.csv` either crashed obscurely or would have deleted every row | both refused explicitly |
| `finalise` raising meant a re-run appended a second hours segment for the same wall clock | a segment is only appended when the call actually trained |
| `config.py` pushed `src/` to `sys.path[0]`, undoing the ordering rule for everything imported afterwards | removed; `config.ensure_import_path()` added and used by `runner.py` and all four test files |
| `guards.py` still imported `os` after its only users were deleted | removed |

---

## Version control

The repository was initialised and one commit made locally
(`738359f`, 144 files, 9.6 MB) before the push was called off. `.gitignore` excludes the
per-transaction score files (~80 MB across eleven existing files plus four to come),
weights, RNG state files and the saved index arrays; the small provenance records are all
tracked. Nothing has been pushed anywhere.

Two things to know if you use it: `paper/` is excluded by a pre-existing rule, so the
planning documents are not in the commit; and `.git/objects/` holds eleven stray
`tmp_obj_*` files from a first attempt in a sandbox that could create files but not delete
them. `git gc --prune=now` clears them.
