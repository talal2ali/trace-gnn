# legacy/ — shipped, superseded, must not be used

Two files that are kept for one reason: **a reader with an older copy of this project needs to
be able to tell what they have.** Neither is used by anything in this repository, and neither
produces a number that appears in the paper.

They are here rather than deleted because deleting them makes an older copy silently
unexplainable. They are here rather than in place because leaving them in place is how they got
used by mistake.

---

## `fig_5_3_SUPERSEDED.py` — was `figures/make/fig_5_3.py`

Drew Figures 5 and 6 from the **75-epoch** protocol study.

**It does not run.** It guards its input against the 75-epoch table:

    EXPECT = {'P1_leaky': 0.966, 'P1_causal': 0.963, 'P2_cutoff': 0.741,
              'P3_rolling': 0.839, 'P3_acausal': 0.946}

The paper's Table 6 reports **0.974 / 0.971 / 0.750 / 0.940 / 0.846**, from the retrained
200-epoch arms. The script also prefers a directory named `protocol_11`, which no longer
exists, so given more than one candidate input it raises `Refusing to guess` rather than
drawing anything. That guard was working correctly: it was refusing data it was not written for.

**Use instead:** `protocol_v2/figures/make/fig_5_3_v2.py`. Same plot code, colours, layout and
labels; two changes only, the guard values and the input directory preference.

Why two generations exist: the original study was trained at 75 epochs while the canonical run
used 200. `CosineAnnealingLR` anneals over `T_max = epochs`, so a 75-epoch run follows a
different learning-rate path from its first step. They are two configurations, not two lengths
of one, which is why the arms were retrained rather than extended.

---

## `protocol_runner.py` — was `src/protocol_runner.py`

A stale fork of the training loop, 213 lines against the 578 of `protocol_v2/runner.py`. It
diverged from the canonical loop and was never reconciled.

It has been marked deprecated for some time and says so at runtime:

    "protocol_runner.run_protocol_fold is DEPRECATED and diverged from ..."

The repository already enforces its disuse in three places, and those guards are worth knowing
about because they will fire if anyone reintroduces it:

- `edge_k/provenance.py` lists it in `DEPRECATED_FILES` and hashes it, so a change to it is
  visible in the provenance record.
- `protocol_v2/tests/test_p3_rolling_locked.py` fails any notebook whose source mentions
  `protocol_runner` or `run_protocol_fold`.
- `edge_k/runner.py` states in its module docstring that `src/protocol_runner.py` is not used.

**Use instead:** `protocol_v2/runner.py` for the protocol arms, `edge_k/runner.py` for the edge
and neighbour-cap arms, and `src/run_date_gnn.py` for the canonical training loop.

Note that the deprecation test looks for the string `protocol_runner` in **notebook source**,
so moving this file to `legacy/` does not weaken it.
