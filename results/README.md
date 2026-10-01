# `results/` — two inputs, not an archive of results

This repository ships **code**. The two files here are the only exception, and they are
**inputs** that the code reads rather than outputs it produced.

| file | size | what it is |
|---|---:|---|
| `per_fold/xgb_11_tuned.csv` | 1.3 KB | the per-fold scores of the tuned gradient-boosted baseline |
| `budget/scores_xgb_11_tuned.csv` | 7.6 MB | its per-transaction scores |

## Why they are here

`rerun_canonical.py` does not re-run the baseline. The tree is deterministic under a fixed
seed and was not re-fitted for the reported run, so the runner **reads** it from a read-only
search root and refuses to start if it is absent:

```python
# the tuned tree is not re-run, so fail now rather than after the training
for fn in ("xgb_11_tuned.csv", "scores_xgb_11_tuned.csv"):
    hits = [q for q in sorted(search_root.rglob(fn)) ...]
    check(bool(hits), f"{fn} not found under {search_root}. The paired tests and the "
                      f"budget sweep both need it, and it is not re-run")
```

Without them a fresh clone fails on its first command, before loading any data. They are the
minimum needed for the repository to run at all.

## What depends on them

- **Table 4** — the paired comparison of TRACE-GNN against the baseline
- **Table 5** — the first rung of the sequential ladder
- **Figure 9 and Table 5's operating point** — `src/budget_sweep_11.py` reads the
  per-transaction scores to build the alert-budget sweep

## They are never written to

`rerun_canonical.py` asserts that the search root is outside its own output folder, and
copies rather than modifies. Nothing in this repository writes to `results/`.
