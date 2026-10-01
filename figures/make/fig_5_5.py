"""Figure 9, Section 5.5.

Usage
-----
    # from a canonical re-run folder (preferred)
    python fig_5_5.py results/ figures/ --canonical results/rerun_20260826_114717_m5

    # legacy: locate the sweep by name anywhere under ROOT
    python fig_5_5.py results/ figures/

`--canonical` names the sweep explicitly. The legacy search takes whichever
`budget_sweep_11feat_matched.csv` was modified most recently, which silently changes
meaning as soon as a re-run writes one, so the figure's provenance depends on file
timestamps rather than on anything stated.

This script used to assert the sweep against precision values copied from Table 4. That
check went stale the moment the run was repeated and then aborted on correct files. It
now cross-checks the sweep against the arm's own per-fold output when `--canonical` is
given, which catches a mismatched score file without encoding a particular result.
"""
import sys, os, argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

p_ = argparse.ArgumentParser()
p_.add_argument("root", nargs="?", default=".")
p_.add_argument("outdir", nargs="?", default=".")
p_.add_argument("--canonical", default=None,
                help="a results/rerun_* folder; the sweep is read from its derived/ folder")
a_ = p_.parse_args()
ROOT, OUT, CANON = a_.root, a_.outdir, a_.canonical
os.makedirs(OUT, exist_ok=True)


def find(n):
    hits = [os.path.join(dp, n) for dp, _, fs in os.walk(ROOT) if n in fs]
    hits = [h for h in hits if "rerun_" not in h] or hits
    if not hits:
        raise FileNotFoundError(f"{n} not found under {os.path.abspath(ROOT)}")
    return max(hits, key=os.path.getmtime) if len(hits) > 1 else hits[0]


plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif',
                     'axes.spines.top': False, 'axes.spines.right': False})

if CANON:
    CANON = os.path.abspath(CANON)
    sweep = os.path.join(CANON, "derived", "budget_sweep_11feat_matched.csv")
    if not os.path.exists(sweep):
        raise FileNotFoundError(f"{sweep} missing. Re-run rerun_canonical.py --resume to "
                                f"regenerate the budget sweep for this folder.")
    b = pd.read_csv(sweep).sort_values('alert_rate')
    row = b[b.alert_rate == 0.005].iloc[0]
    pf = pd.read_csv(os.path.join(CANON, "canonical", "main", "per_fold.csv"))
    want = pf.precision_at_k.mean()
    assert abs(row.gnn_p_at_k - want) < 0.005, (
        f"sweep reads {row.gnn_p_at_k:.4f} at the operating budget but the arm's per-fold "
        f"output means {want:.4f}. The sweep was built from different scores.")
    print(f"  canonical folder: {CANON}")
    print(f"  operating budget: gnn {row.gnn_p_at_k:.3f} xgb {row.xgb11_p_at_k:.3f} "
          f"(verified against canonical/main/per_fold.csv)")
else:
    # renamed 2026-09-08: the copy under results/budget/ is a superseded sweep
    b = pd.read_csv(find('budget_sweep_11feat_matched_SUPERSEDED.csv')).sort_values('alert_rate')
    row = b[b.alert_rate == 0.005].iloc[0]
    print("  legacy filename search. Pass --canonical for single-provenance figures.")
    print("  WARNING  the legacy sweep is superseded. This figure must not be used for the\n           manuscript.")
    print(f"  operating budget: gnn {row.gnn_p_at_k:.3f} xgb {row.xgb11_p_at_k:.3f}")

nsig = int((b.p_precision > 0.05).sum())
print(f"  rates where the precision advantage is NOT significant at 5%: {nsig}"
      + ("" if nsig else "  (none, so panel (b) carries no p-value annotations)"))

def panel(ax, tag):
    ax.text(-0.09, 1.03, tag, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')

BLUE, GREEN, GREY = '#0072B2', '#009E73', '#666666'
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.5), gridspec_kw={'width_ratios': [1.1, 1]})
r = b.alert_rate
a1.axvline(0.005, color=GREY, ls=':', lw=1.4, zorder=1)
a1.plot(r, b.gnn_p_at_k, 'o-', color=BLUE, ms=7, lw=2, label='TRACE-GNN, precision', zorder=3)
a1.plot(r, b.xgb11_p_at_k, 'o--', color=BLUE, ms=6, lw=1.6, alpha=.55, label='XGBoost, precision', zorder=3)
a1.plot(r, b.gnn_r_at_k, 's-', color=GREEN, ms=7, lw=2, label='TRACE-GNN, recall', zorder=3)
a1.plot(r, b.xgb11_r_at_k, 's--', color=GREEN, ms=6, lw=1.6, alpha=.55, label='XGBoost, recall', zorder=3)
a1.fill_between(r, b.gnn_p_at_k, b.xgb11_p_at_k, color=BLUE, alpha=.10, zorder=0)
a1.fill_between(r, b.gnn_r_at_k, b.xgb11_r_at_k, color=GREEN, alpha=.10, zorder=0)
a1.set_xscale('log'); a1.set_xlabel('Alert budget (fraction of transactions)')
a1.set_ylabel('Score'); a1.set_ylim(0, 1.03)
a1.text(0.0053, 0.045, 'operating\nbudget 0.5%', fontsize=10, color='black')
a1.legend(frameon=False, fontsize=10, loc='center right', bbox_to_anchor=(1.0, 0.52))
panel(a1, '(a)')

w = 0.36; idx = np.arange(len(b))
a2.bar(idx - w / 2, b.delta_p, w, color=BLUE, label='Precision')
a2.bar(idx + w / 2, b.delta_r, w, color=GREEN, label='Recall')
for i, (dp, dr, pp) in enumerate(zip(b.delta_p, b.delta_r, b.p_precision)):
    a2.annotate(f'{dp:.3f}', (i - w / 2, dp), xytext=(0, 3), textcoords='offset points',
                ha='center', fontsize=10, color='black')
    a2.annotate(f'{dr:.3f}', (i + w / 2, dr), xytext=(0, 3), textcoords='offset points',
                ha='center', fontsize=10, color='black')
    if pp > 0.05:
        a2.annotate(f'p = {pp:.3f}', (i, max(dp, dr)), xytext=(0, 17), textcoords='offset points',
                    ha='center', fontsize=10, color='black')
a2.axhline(0, color='black', lw=1)
a2.set_xticks(idx); a2.set_xticklabels([f'{v:g}' for v in r]); a2.set_xlabel('Alert budget')
a2.set_ylabel('TRACE-GNN advantage'); a2.set_ylim(0, 0.215)
a2.legend(frameon=False, fontsize=10, loc='upper right')
panel(a2, '(b)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F9_budget_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'F9_budget_sensitivity.pdf'), bbox_inches='tight')
print('wrote F9_budget_sensitivity to', os.path.abspath(OUT))
