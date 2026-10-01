# Version

| | |
|---|---|
| **Repository snapshot** | 2026-10-01 |
| **Manuscript** | `TRACE-GNN_manuscript_v7a.docx` |
| **Draft** | v7a, derived from v6e (md5 `34911384ab54f4921fab23fdbfa80bff`) |
| **Manuscript last saved** | 2026-09-03 18:17 UTC, revision 25 |
| **Manuscript written to disk** | 2026-09-08 13:59 +03:00 |
| **Code exported from the working repository** | 2026-09-08 18:41–18:46 +03:00, after the manuscript |
| **Dataset** | Sparkov only |
| **Licence** | MIT |

Every figure, numbered table and appendix table in v7a has its producing code here; the map is
`CODE_MAP.md`.

## What this snapshot changed against the 2026-09-08 export

- `src/experiment_config.py`: the dataclass defaults were the prepared-file name and column
  names of a different corpus, left over from shared tooling, plus an unused `redundancy_targets`
  field. All six are now Sparkov's; the unused field was removed.
  **Behaviour is unchanged** — all four entry points already set every one of these explicitly.
- Five docstrings and one notebook markdown cell reworded to stop naming a corpus this paper
  does not use.
- `.gitignore`: the `paper/` line removed, since no `paper/` directory is part of this release.
- Added `results/per_fold/xgb_11_tuned.csv` and `results/budget/scores_xgb_11_tuned.csv`, the
  two baseline files `rerun_canonical.py` reads. Without them a fresh clone cannot start.
- Added `CODE_MAP.md`, `LICENSE` (MIT), `CITATION.cff`, `VERSION.md`, `results/README.md`.
