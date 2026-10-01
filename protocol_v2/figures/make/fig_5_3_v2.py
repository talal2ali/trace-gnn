"""
Figures 5 and 6, Section 5.3 of the TRACE-GNN manuscript.

REGENERATED from the retrained 200-epoch arms (protocol_v2). **This is the current
producer of Figures 5 and 6.** Identical to the older
figures/make/fig_5_3_SUPERSEDED.py except for the Table 6 guard values and the
input directory preference. Plot code, colours, layout and labels are unchanged.

The older script asserts the 75-epoch study's values (0.966 / 0.963 / 0.741 /
0.839 / 0.946) and prefers a directory that no longer exists, so it raises
before drawing. It was renamed on 2026-09-08 and must not be used.

Usage
    python fig_5_3_v2.py                 # search ./ for the CSV, write here
    python fig_5_3_v2.py ROOT            # search ROOT recursively
    python fig_5_3_v2.py ROOT OUTDIR     # also set the output directory

Only input is per_week_all_arms.csv, written by the protocol notebook.
"""
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT  = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)

def find(name, prefer='derived'):
    """Return the wanted file. Several superseded copies exist under ROOT, so a
    directory preference is required rather than taking whatever os.walk hits first."""
    hits = [os.path.join(dp, name) for dp, _, fs in os.walk(ROOT) if name in fs]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {os.path.abspath(ROOT)}")
    if len(hits) == 1:
        return hits[0]                     # unambiguous, the Table 6 assertion below still guards it
    chosen = next((h for h in hits if os.sep + prefer + os.sep in h), None)
    if chosen is None:
        raise FileNotFoundError(
            f"{name} found in {[os.path.relpath(h, ROOT) for h in hits]} "
            f"but none under '{prefer}'. Refusing to guess.")
    if len(hits) > 1:
        print(f"  {len(hits)} copies of {name}, using the one under {prefer}")
    return chosen

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif',
                     'axes.spines.top': False, 'axes.spines.right': False})

src = find('per_week_all_arms.csv')
print("reading", src)
d = pd.read_csv(src)
P = {a: g.set_index('test_week').sort_index() for a, g in d.groupby('arm')}
c = sorted(set.intersection(*[set(g.index) for g in P.values()]))
print(f"{len(P)} arms, {len(c)} common weeks: {c[0]}-{c[-1]}")

# Retrained 200-epoch values (protocol_v2, 2026-09-02). The previous guard held the
# 75-epoch numbers 0.966/0.963/0.741/0.839/0.946 and correctly REFUSES this data.
EXPECT = {'P1_leaky': 0.974, 'P1_causal': 0.971, 'P2_cutoff': 0.750,
          'P3_rolling': 0.846, 'P3_acausal': 0.941}
for a, want in EXPECT.items():
    got = P[a].loc[c, 'ap'].mean()
    assert abs(got - want) < 0.0015, (
        f"{a} reads {got:.3f}, Table 6 says {want}. Wrong protocol study, do not use these figures.")
print("  all five arms match Table 6")

BLUE, PINK, GREEN, ORANGE, GREY = '#0072B2', '#CC79A7', '#009E73', '#D55E00', '#666666'
order = ['P1_leaky', 'P1_causal', 'P2_cutoff', 'P3_rolling', 'P3_acausal']
lab = ['Random split\nunconstrained', 'Random split\ncausal', 'Single cutoff\ncausal',
       'Rolling origin\ncausal', 'Rolling origin\nunconstrained']
ap = [P[a].loc[c, 'ap'].mean() for a in order]
sd = [P[a].loc[c, 'ap'].std() for a in order]
au = [P[a].loc[c, 'auroc'].mean() for a in order]
x = np.arange(5)

def panel(ax, tag):
    ax.text(-0.09, 1.03, tag, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')

# ---------------- Figure 5: protocol ladder ----------------
fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.axvspan(2.55, 3.45, color='0.92', zorder=0)
ax.errorbar(x, au, fmt='s--', color=PINK, ms=8, lw=1.8, label='AUROC', zorder=3)
ax.errorbar(x, ap, yerr=sd, fmt='o-', color=BLUE, ms=8, lw=2, capsize=4,
            label='Average precision', zorder=3)
for i, (a, u, e) in enumerate(zip(ap, au, sd)):
    ax.annotate(f'{u:.3f}', (i, u), textcoords='offset points', xytext=(0, 11),
                ha='center', fontsize=10, color='black')
    ax.annotate(f'{a:.3f}', (i, a - e), textcoords='offset points', xytext=(0, -15),
                ha='center', fontsize=10, color='black')
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=10)
ax.set_ylabel('Score'); ax.set_ylim(0.52, 1.06)
ax.legend(frameon=False, loc='lower left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F5_protocol_ladder.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'F5_protocol_ladder.pdf'), bbox_inches='tight')
plt.close()

# ---------------- Figure 6: non-additivity of the two corrections ----------------
# unrounded, so the recovered-AP bars match Table 7 rather than differences of rounded means
m = {a: P[a].loc[c, 'ap'].mean() for a in P}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={'width_ratios': [1, 1.05]})
a1.plot([0, 1], [m['P1_leaky'], m['P1_causal']], 'o-', color=ORANGE, ms=9, lw=2.2, label='Random split')
a1.plot([0, 1], [m['P3_acausal'], m['P3_rolling']], 's--', color=BLUE, ms=9, lw=2.2, label='Rolling origin')
for xx, yy, dx, dy, ha in [(0, m['P1_leaky'], 0, 9, 'center'), (1, m['P1_causal'], 0, 9, 'center'),
                           (0, m['P3_acausal'], 0, -18, 'center'), (1, m['P3_rolling'], -8, 6, 'right')]:
    a1.annotate(f'{yy:.3f}', (xx, yy), textcoords='offset points', xytext=(dx, dy), ha=ha, fontsize=10)
a1.annotate('', xy=(1.06, m['P1_causal']), xytext=(1.06, m['P3_rolling']),
            arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.3))
a1.text(1.10, (m['P1_causal'] + m['P3_rolling']) / 2, 'both\nsafeguards\nclosed',
        fontsize=10, color='black', va='center')
a1.set_xticks([0, 1]); a1.set_xticklabels(['Unconstrained\ngraph', 'Causal\ngraph'])
a1.set_xlim(-0.25, 1.6); a1.set_ylabel('Average precision')
a1.legend(frameon=False, fontsize=10, loc='lower left')
panel(a1, '(a)')

bars = ['Constrain\nsplit only', 'Constrain\ngraph only', 'Sum of\nthe two', 'Constrain\nboth']
vals = [m['P1_leaky'] - m['P3_acausal'], m['P1_leaky'] - m['P1_causal'],
        (m['P1_leaky'] - m['P3_acausal']) + (m['P1_leaky'] - m['P1_causal']),
        m['P1_leaky'] - m['P3_rolling']]
b = a2.bar(range(4), vals, color=['0.75', '0.75', '0.5', GREEN], width=.62)
for r, v in zip(b, vals):
    a2.annotate(f'{v:.3f}', (r.get_x() + r.get_width() / 2, v), textcoords='offset points',
                xytext=(0, 4), ha='center', fontsize=10)
a2.axhline(vals[2], color=GREY, ls=':', lw=1.2)
# "non-additive excess", not "interaction": the four arms differ in test composition, training
# history and refitting schedule as well as in the two named factors, so this is not a clean
# factorial effect. Section 5.3 and the caption use the same term.
a2.annotate(f'non-additive\nexcess +{vals[3]-vals[2]:.3f}', xy=(3, (vals[2] + vals[3]) / 2),
            xytext=(1.40, 0.088), fontsize=10, color='black',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))
a2.set_xticks(range(4)); a2.set_xticklabels(bars, fontsize=10)
a2.set_ylabel('AP recovered'); a2.set_ylim(0, 0.152)
panel(a2, '(b)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F6_non_additivity.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'F6_non_additivity.pdf'), bbox_inches='tight')
plt.close()

print("wrote F5_protocol_ladder and F6_non_additivity (png + pdf) to", os.path.abspath(OUT))
