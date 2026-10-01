"""Figure 4, Section 5.1.

Usage
-----
    # from a canonical re-run folder (preferred)
    python fig_5_1.py results/ figures/ --canonical results/rerun_20260826_114717_m5

    # legacy: locate per-fold files by name anywhere under ROOT
    python fig_5_1.py results/ figures/

`--canonical` takes the graph arm from one designated set of results rather than from
whichever `gnn_11_converged_SUPERSEDED.csv` happens to be newest under ROOT, so the figure and
Table 4 cannot come from different executions. The gradient-boosted baseline is not
re-run and is still located by name.

The hard-coded `EXPECT` check this script used to make asserted each arm against a value
copied from the manuscript, which went stale as soon as a run was repeated and then
aborted with "Wrong file" when the file was in fact correct. Under `--canonical` the
graph arm is instead cross-checked against `canonical_summary.csv`, which catches a
truncated or foreign file without encoding any particular result.
"""
import sys, os, argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

p_ = argparse.ArgumentParser()
p_.add_argument("root", nargs="?", default=".")
p_.add_argument("outdir", nargs="?", default=".")
p_.add_argument("--canonical", default=None,
                help="a results/rerun_* folder; the graph arm is read from it")
a_ = p_.parse_args()
ROOT, OUT, CANON = a_.root, a_.outdir, a_.canonical
os.makedirs(OUT, exist_ok=True)

GRAPH = "gnn_11_converged_SUPERSEDED.csv"   # renamed 2026-09-08; see MANIFEST.md
TREE  = "xgb_11_tuned.csv"
LEGACY_EXPECT = {"gnn": 0.843, "xgb": 0.636}      # the values the v5e manuscript printed
# The legacy graph arm is a SUPERSEDED execution (0.8427 against the reported 0.8551) and is
# named accordingly on disk. The tree is not superseded: it is deterministic and was not
# re-run. Pass --canonical for a figure that belongs in the manuscript.


def find(name):
    hits = [os.path.join(dp, name) for dp, _, fs in os.walk(ROOT) if name in fs]
    hits = [h for h in hits if "rerun_" not in h] or hits   # rerun folders use --canonical
    if not hits:
        raise FileNotFoundError(f"{name} not found under {os.path.abspath(ROOT)}")
    if len(hits) > 1:
        newest = max(hits, key=os.path.getmtime)
        print(f"  {len(hits)} copies of {name}, using {os.path.relpath(newest, ROOT)}")
        return newest
    return hits[0]


def load(name):
    p = find(name)
    d = pd.read_csv(p).sort_values("test_week").reset_index(drop=True)
    print(f"  {os.path.relpath(p, ROOT):55s} AP {d.ap.mean():.4f}  {len(d)} folds")
    return d


print("loading")
x = load(TREE)
if CANON:
    CANON = os.path.abspath(CANON)
    print(f"  canonical folder: {CANON}")
    f = os.path.join(CANON, "canonical", "main", "per_fold.csv")
    g = pd.read_csv(f).sort_values("test_week").reset_index(drop=True)
    want = float(pd.read_csv(os.path.join(CANON, "canonical_summary.csv"))
                 .set_index("arm").loc["main", "mean_ap"])
    assert abs(g.ap.mean() - want) < 1e-6, (
        f"main per_fold.csv means {g.ap.mean():.6f} but canonical_summary.csv records "
        f"{want:.6f}. The file is truncated or does not belong to this run.")
    assert len(g) == len(x), f"graph has {len(g)} folds, tree has {len(x)}"
    print(f"  {'canonical/main/per_fold.csv':55s} AP {g.ap.mean():.4f}  {len(g)} folds "
          f"(verified against summary)")
else:
    print("  legacy filename search. Pass --canonical for single-provenance figures.")
    print("  WARNING  the legacy graph arm is a superseded execution. This figure must not\n           be used for the manuscript.")
    g = load(GRAPH)
    for lbl, d, w_ in (("graph", g, LEGACY_EXPECT["gnn"]), ("tree", x, LEGACY_EXPECT["xgb"])):
        if abs(round(d.ap.mean(), 3) - w_) >= 0.0015:
            print(f"    NOTE  {lbl} arm reads {d.ap.mean():.3f}, the v5e manuscript printed "
                  f"{w_}. Expected if the run was repeated.")
print()

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '#666666'
fig, ax = plt.subplots(figsize=(9.8, 4.3))
w = g.test_week
ax.fill_between(w, g.ap, x.ap, color=BLUE, alpha=.12, label='TRACE-GNN margin', zorder=1)
ax.plot(w, g.ap, 'o-', color=BLUE, ms=7, lw=2, label='TRACE-GNN', zorder=3)
ax.plot(w, x.ap, 's--', color=ORANGE, ms=6, lw=1.8, label='XGBoost, tuned per fold', zorder=3)
ax.axhline(g.ap.mean(), color=BLUE, ls=':', lw=1.1, zorder=2)
ax.axhline(x.ap.mean(), color=ORANGE, ls=':', lw=1.1, zorder=2)
ax.annotate('7 fraud cases', xy=(24, g.ap.max()), xytext=(22.1, 1.075), fontsize=10,
            color='black', ha='center', arrowprops=dict(arrowstyle='-', color='0.6', lw=.9))
ax.set_xticks(w); ax.set_xlabel('Test week'); ax.set_ylabel('Average precision')
ax.set_ylim(0.28, 1.13)
ax.legend(frameon=False, loc='lower left', fontsize=10, ncol=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F4_per_fold_matched.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'F4_per_fold_matched.pdf'), bbox_inches='tight')
print(f"GNN {g.ap.mean():.4f} | XGB {x.ap.mean():.4f} | "
      f"GNN above on {(g.ap.values > x.ap.values).sum()}/{len(g)} folds")
