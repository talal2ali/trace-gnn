"""
Alert-budget sweep for Section 5.5.

  graph = scores_gnn_11_converged.csv   (the reported converged model)
  tree  = scores_xgb_11_tuned.csv       (the per-fold tuned eleven-feature baseline)

Both are written by 15_workstation_session.ipynb into experiment_results/reruns/.

Usage
    python budget_sweep_11.py ROOT OUTDIR
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import top_k_from_rate          # one budget rule for the whole codebase

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)

RATES = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]
GRAPH_FILE = "scores_gnn_11_converged.csv"
TREE_FILE = "scores_xgb_11_tuned.csv"


def find(name):
    hits = []
    for dirpath, _, files in os.walk(ROOT):
        if name in files:
            hits.append(os.path.join(dirpath, name))
    if not hits:
        raise FileNotFoundError(
            f"{name} not found under {os.path.abspath(ROOT)}.\n"
            f"Run cells 3 and 4 of 15_workstation_session.ipynb first."
        )
    if len(hits) > 1:
        print(f"  NOTE {len(hits)} copies of {name}:")
        for h in hits:
            print("      ", os.path.relpath(h, ROOT))
        print("       using the first")
    return hits[0]


def load(fname):
    p = find(fname)
    d = pd.read_csv(p)
    print(f"  {os.path.relpath(p, ROOT)}")
    print(f"    {len(d):>8,} rows  cols={list(d.columns)}")
    return d


def col(d, *cands):
    for c in cands:
        if c in d.columns:
            return c
    raise KeyError(f"none of {cands} in {list(d.columns)}")


def per_fold(d):
    w = col(d, "fold", "test_week", "week")
    s = col(d, "score", "y_score", "prob", "pred", "p")
    yc = col(d, "y", "label", "y_true", "is_fraud")
    out = {int(k): (g[s].to_numpy(float), g[yc].to_numpy(int)) for k, g in d.groupby(w)}
    print(f"    -> {len(out)} folds, weeks {min(out)}-{max(out)}, "
          f"positives {sum(int(v[1].sum()) for v in out.values()):,}")
    return out


def exact_wilcoxon(a, b):
    """Paired exact Wilcoxon with numerical ties dropped, matching evaluation.paired_wilcoxon.

    SciPy's default method="auto" falls back to the normal approximation the moment a zero
    difference appears, and a zero appears at every rate here because week 24's seven
    positives sit inside every budget. That fallback is what printed 0.00217 at the tightest
    rate where the exact test gives 0.00049.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    kept = d[np.abs(d) > 1e-12]
    return float(wilcoxon(kept, method="exact")[1]) if len(kept) else float("nan")


def at_k(s, y, rate):
    """The budget must match the one every other result in the paper uses.

    An alert rate of 0.5% of a 16,859-transaction week is 84.30 alerts, which has to become a
    whole number. `evaluation.top_k_from_rate` rounds up; this function previously rounded to
    nearest, and the two disagreed on 7 of the 17 folds, 1,700 alerts against 1,693. That is
    why the recall advantage at the operating point read 0.1585 here and 0.1574 in Table 4 for
    the same models on the same weeks. It now defers to the shared definition so there is one
    budget rule in the codebase.
    """
    n = len(s)
    k = top_k_from_rate(n, rate)
    tp = int(y[np.argsort(-s)[:k]].sum())
    pos = int(y.sum())
    return k, tp / k, (tp / pos if pos else np.nan)


print("loading")
G = per_fold(load(GRAPH_FILE))
X = per_fold(load(TREE_FILE))
weeks = sorted(set(G) & set(X))
print(f"  {len(weeks)} common folds: {weeks[0]}-{weeks[-1]}\n")

for w in weeks:
    assert len(G[w][0]) == len(X[w][0]), f"week {w}: row counts differ between arms"
    assert G[w][1].sum() == X[w][1].sum(), f"week {w}: positive counts differ between arms"

rows = []
for r in RATES:
    gp, gr, xp, xr, ks = [], [], [], [], []
    for w in weeks:
        k, p, rc = at_k(*G[w], r)
        ks.append(k); gp.append(p); gr.append(rc)
        _, p2, rc2 = at_k(*X[w], r)
        xp.append(p2); xr.append(rc2)
    gp, gr, xp, xr = map(np.array, (gp, gr, xp, xr))
    rows.append(dict(
        alert_rate=r, mean_k=int(round(np.mean(ks))),
        gnn_p_at_k=gp.mean(), gnn_r_at_k=gr.mean(),
        xgb11_p_at_k=xp.mean(), xgb11_r_at_k=xr.mean(),
        delta_p=gp.mean() - xp.mean(), delta_r=gr.mean() - xr.mean(),
        p_precision=exact_wilcoxon(gp, xp), p_recall=exact_wilcoxon(gr, xr),
        folds_won_p=int((gp > xp).sum()), n_folds=len(weeks),
    ))

out = pd.DataFrame(rows)
dest = os.path.join(OUT, "budget_sweep_11feat_matched.csv")
out.to_csv(dest, index=False)
print(out.round(4).to_string(index=False))
print("\nwrote", dest)
