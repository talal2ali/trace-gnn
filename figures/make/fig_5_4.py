"""Figures 7 and 8, Section 5.4.

Usage
-----
    # from a canonical re-run folder (preferred: every arm from one set of results)
    python fig_5_4.py results/ figures/ --canonical results/rerun_20260826_114717_m5

    # legacy: locate per-fold files by name anywhere under ROOT
    python fig_5_4.py results/ figures/

Why --canonical exists
----------------------
The legacy path finds per-fold data by filename (`gnn_11_converged_SUPERSEDED.csv`,
`maxagg_11_SUPERSEDED.csv`, ...) anywhere under ROOT. Those files belong to the older executions in
`results/per_fold/` and `results/architecture/`. Drawing panel 7(b) from a newer run's
validation curves while the per-fold means still came from the older files is exactly the
provenance mismatch notebook 16 was written to remove, so prefer --canonical, which takes
every arm and the curves from one folder.

Validation instead of hard-coded constants
------------------------------------------
This script used to assert each arm's mean against a number copied from the manuscript,
which went stale the moment the run was repeated and then failed with "Wrong file" when
the files were in fact correct. Under --canonical it instead cross-checks each
`per_fold.csv` against the `mean_ap` recorded in `canonical_summary.csv`. That catches a
genuinely wrong or truncated file without encoding any particular result.

The noise band
--------------
Each row of Figure 8 is a difference between two single executions. If one execution has
standard deviation s, that difference has standard deviation s*sqrt(2). The band is drawn
at that width, read from `repeat_summary.csv` when it exists, so an effect inside the band
is one that repeated runs of the same configuration could have produced on their own. The
band is a scale, not a decision rule: it is one standard deviation, so a third of true
nulls fall outside it. The paired test is what is reported alongside it.

The paired test
---------------
`paired()` implements the rule the manuscript declares in Table 7's note and
`edge_k/analysis_settings.py` freezes: differences below 1e-12 are numerical ties and are
dropped, and the test is the exact Wilcoxon rather than the normal approximation SciPy
falls back to when tied ranks are present. Fold counts follow the same rule, so a fold
whose two arms differ by a float artefact of 2e-16 is a tie and not a win. Counting it as
a win is what put "9 of 17" for the depth variant into manuscript v6e.

The edge-encoding arm
---------------------
The sixth row comes from `edge_k/results/arms/no_edge_bias/per_fold.csv`, which sits
outside any canonical re-run folder because it was trained later, against the same locked
canonical arms. Pass `--edge-arm` to point elsewhere; the default location is searched
relative to ROOT and to this file. Its mean is cross-checked against the `mean_ap` in the
arm's own `run_record.json`, in the same spirit as the canonical checks above. Nothing
under `edge_k/results/` is written to.
"""
import sys, os, json, argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from scipy.stats import wilcoxon

p = argparse.ArgumentParser()
p.add_argument("root", nargs="?", default=".")
p.add_argument("outdir", nargs="?", default=".")
p.add_argument("--canonical", default=None,
               help="a results/rerun_* folder. Every arm and the validation curves are "
                    "read from it, so the figures and the tables share one provenance")
p.add_argument("--default-sd", type=float, default=0.0102,
               help="per-run SD used for the Figure 8 band when repeat_summary.csv is absent")
p.add_argument("--edge-arm", default=None,
               help="per_fold.csv for the edge-encoding ablation. Defaults to "
                    "edge_k/results/arms/no_edge_bias/per_fold.csv, searched relative to "
                    "ROOT and to this file. Pass 'none' to draw the figure without it")
a = p.parse_args()
ROOT, OUT, CANON = a.root, a.outdir, a.canonical
os.makedirs(OUT, exist_ok=True)

# arm -> (canonical subfolder, legacy filename)
# Legacy filenames carry the _SUPERSEDED suffix as of 2026-09-08: every one of these four is
# an earlier execution that disagrees with the manuscript. See MANIFEST.md.
ARMS = {"main":     ("main",     "gnn_11_converged_SUPERSEDED.csv"),
        "uid_only": ("uid_only", "gnn_11_uid_only_converged_SUPERSEDED.csv"),
        "maxagg":   ("maxagg",   "maxagg_11_SUPERSEDED.csv"),
        "depth3":   ("depth3",   "depth3_11_SUPERSEDED.csv")}
LEGACY_EXPECT = {"gnn_11_converged_SUPERSEDED.csv": 0.843, "xgb_11_tuned.csv": 0.636,
                 "gnn_11_uid_only_converged_SUPERSEDED.csv": 0.773,
                 "maxagg_11_SUPERSEDED.csv": 0.824, "depth3_11_SUPERSEDED.csv": 0.857}


def find(name):
    hits = [os.path.join(dp, name) for dp, _, fs in os.walk(ROOT) if name in fs]
    # never pick a file out of a rerun folder by accident; those are addressed by --canonical
    hits = [h for h in hits if "rerun_" not in h] or hits
    if not hits:
        raise FileNotFoundError(f"{name} not found under {os.path.abspath(ROOT)}")
    if len(hits) > 1:
        newest = max(hits, key=os.path.getmtime)
        print(f"  {len(hits)} copies of {name}, using {os.path.relpath(newest, ROOT)}")
        return newest
    return hits[0]


def load_legacy(name):
    d = pd.read_csv(find(name)).sort_values("test_week").reset_index(drop=True)
    got = round(d.ap.mean(), 3)
    print(f"  {name:34s} AP {d.ap.mean():.4f}")
    if name in LEGACY_EXPECT and abs(got - LEGACY_EXPECT[name]) >= 0.0015:
        print(f"    NOTE  reads {got}, the manuscript value is {LEGACY_EXPECT[name]}. "
              f"Expected if the run was repeated; check it is the file you meant.")
    return d


print("loading")
if CANON:
    CANON = os.path.abspath(CANON)
    print(f"  canonical folder: {CANON}")
    summ = pd.read_csv(os.path.join(CANON, "canonical_summary.csv")).set_index("arm")
    arms = {}
    for arm, (sub, _) in ARMS.items():
        f = os.path.join(CANON, "canonical", sub, "per_fold.csv")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} missing. Finish that arm before drawing figures.")
        d = pd.read_csv(f).sort_values("test_week").reset_index(drop=True)
        want = float(summ.loc[arm, "mean_ap"])
        assert abs(d.ap.mean() - want) < 1e-6, (
            f"{arm}: per_fold.csv means {d.ap.mean():.6f} but canonical_summary.csv "
            f"records {want:.6f}. The file is truncated or does not belong to this run.")
        assert len(d) == 17, f"{arm}: {len(d)} folds, expected 17"
        print(f"  {arm:34s} AP {d.ap.mean():.4f}  ({len(d)} folds, verified against summary)")
        arms[arm] = d
    g, u, mx, d3 = arms["main"], arms["uid_only"], arms["maxagg"], arms["depth3"]
    x  = load_legacy("xgb_11_tuned.csv")          # tree is not re-run, deterministic
    fr = pd.read_csv(os.path.join(CANON, "fold_composition.csv")).sort_values("test_week")
    vc = pd.read_csv(os.path.join(CANON, "canonical", "main", "val_curves.csv"))
else:
    print("  legacy filename search. Pass --canonical for single-provenance figures.")
    print("  WARNING  every legacy graph arm is a superseded execution. This figure must not\n           be used for the manuscript.")
    g  = load_legacy(ARMS["main"][1])
    x  = load_legacy("xgb_11_tuned.csv")          # not superseded; the tree was not re-run
    u  = load_legacy(ARMS["uid_only"][1])
    mx = load_legacy(ARMS["maxagg"][1])
    d3 = load_legacy(ARMS["depth3"][1])
    fr = pd.read_csv(find("fold_fraud_report.csv")).sort_values("test_week")
    vc = pd.read_csv(find("val_curves_converged_SUPERSEDED.csv"))

assert vc.test_week.nunique() == len(g), (
    f"validation curves cover {vc.test_week.nunique()} folds but the main arm has {len(g)}. "
    f"They come from different executions, which is the provenance bug this avoids.")

# ---- the edge-encoding ablation, trained outside the canonical folder ------------------
EDGE_DEFAULT = os.path.join("edge_k", "results", "arms", "no_edge_bias", "per_fold.csv")


def edge_arm_path(given):
    if given and given.lower() == "none":
        return None
    if given:
        return given
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (ROOT, here, os.path.join(here, "..", ".."), os.path.join(ROOT, "..")):
        c = os.path.normpath(os.path.join(base, EDGE_DEFAULT))
        if os.path.exists(c):
            return c
    return None


eb = None
_ebp = edge_arm_path(a.edge_arm)
if _ebp is None:
    print("  NOTE  edge-encoding arm not found; Figure 8 will be drawn with five rows. "
          "Pass --edge-arm to supply it.")
else:
    eb = pd.read_csv(_ebp).sort_values("test_week").reset_index(drop=True)
    rec = os.path.join(os.path.dirname(_ebp), "run_record.json")
    if os.path.exists(rec):
        want = float(json.load(open(rec))["mean_ap"])
        assert abs(eb.ap.mean() - want) < 1e-6, (
            f"no_edge_bias: per_fold.csv means {eb.ap.mean():.6f} but run_record.json "
            f"records {want:.6f}. The file is truncated or belongs to another arm.")
    assert len(eb) == len(g), f"no_edge_bias: {len(eb)} folds, expected {len(g)}"
    assert list(eb.test_week) == list(g.test_week), (
        "no_edge_bias covers different weeks from the main arm; the rows are not paired.")
    print(f"  {'no_edge_bias':34s} AP {eb.ap.mean():.4f}  ({len(eb)} folds, "
          f"verified against run_record.json)")


# ---- the paired test, as the manuscript declares it -----------------------------------
def paired(a_, base):
    """Exact Wilcoxon with numerical ties dropped. Returns p, arm wins, base wins, ties."""
    d = a_.ap.values - base.ap.values
    keep = np.abs(d) > 1e-12
    dd = d[keep]
    pv = float(wilcoxon(dd, method="exact")[1]) if len(dd) else float("nan")
    return pv, int((dd > 0).sum()), int((dd < 0).sum()), int((~keep).sum())

# ---- run-to-run noise, measured if available -----------------------------------------
SD, SD_SRC = a.default_sd, "assumed"
if CANON and os.path.exists(os.path.join(CANON, "repeat_summary.csv")):
    r = pd.read_csv(os.path.join(CANON, "repeat_summary.csv"))
    if len(r) > 1:
        SD = float(r.mean_ap.std(ddof=1))
        SD_SRC = f"{len(r)} seeds {list(r.seed)}"
BAND = SD * np.sqrt(2)          # SD of a difference between two single runs
print(f"\n  run-to-run SD {SD:.4f} ({SD_SRC}); Figure 8 band +/-{BAND:.4f} = sqrt(2)*SD\n")

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, GREEN, ORANGE, GREY = '#0072B2', '#009E73', '#D55E00', '#666666'


def panel(ax, tag):
    ax.text(-0.09, 1.14, tag, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


# ---------------- Figure 7 ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.4), gridspec_kw={'width_ratios': [1.15, 1]})
k, drop = fr[fr.kept], fr[~fr.kept]
a1.bar(k.test_week, k.n_test, color='0.80', width=.72, label='Test transactions')
a1.bar(drop.test_week, drop.n_test, color='white', edgecolor='0.55', hatch='///',
       width=.72, label='Excluded, no fraud')
a1.set_ylabel('Test transactions'); a1.set_xlabel('Test week')
a1.set_xticks(fr.test_week); a1.set_ylim(0, 38000)
b = a1.twinx(); b.spines['top'].set_visible(False)
b.plot(fr.test_week, fr.test_fraud, 'o-', color=ORANGE, ms=5.5, lw=1.8, label='Fraud cases')
b.set_ylabel('Fraud cases', color=ORANGE); b.tick_params(axis='y', colors=ORANGE)
b.set_ylim(0, 125)
# annotate the sparsest kept fold and the excluded one, read from the data rather than fixed
kept_min = k.loc[k.test_fraud.idxmin()]
b.annotate(f"{int(kept_min.test_fraud)}", xy=(kept_min.test_week, kept_min.test_fraud),
           xytext=(kept_min.test_week - 1.0, 26), fontsize=10, color=ORANGE,
           arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))
if len(drop):
    dr = drop.iloc[0]
    b.annotate(f"{int(dr.test_fraud)}", xy=(dr.test_week, dr.test_fraud),
               xytext=(dr.test_week, 18), fontsize=10, color=ORANGE, ha='center',
               arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=10, ncol=3,
          loc='lower left', bbox_to_anchor=(-0.02, 1.01))
panel(a1, '(a)')

for _, gg in vc.groupby('test_week'):
    a2.plot(gg.epoch, gg.val_metric, color='0.82', lw=.8, zorder=1)
m = vc.groupby('epoch').val_metric.mean(); cnt = vc.groupby('epoch').size()
full = m[cnt >= vc.test_week.nunique()]
a2.plot(m.index, m.values, color=BLUE, lw=2.4, zorder=3, label='Mean across folds')
a2.axvspan(0, full.index.max(), color=BLUE, alpha=.06, zorder=0)
med_stop = int(np.median(vc.groupby('test_week').epoch.max() + 1))    # from THIS run
a2.axvline(med_stop, color=GREY, ls=':', lw=1.4, zorder=2)
a2.annotate(f'median stop\n{med_stop} epochs', xy=(med_stop, 0.62),
            xytext=(med_stop + 12, 0.50), fontsize=10, color='black',
            arrowprops=dict(arrowstyle='->', color='0.4', lw=1))
a2.text(full.index.max() / 2, 0.32, f'all {vc.test_week.nunique()} folds',
        fontsize=10, color='black', ha='center')
a2.set_xlabel('Epoch'); a2.set_ylabel('Validation average precision')
a2.legend(frameon=False, fontsize=10, loc='lower right')
a2.set_ylim(0.28, 0.96)
panel(a2, '(b)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F7_folds_convergence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'F7_folds_convergence.pdf'), bbox_inches='tight'); plt.close()

# ---------------- Figure 8 ----------------
rng = np.random.default_rng(0)


def ci(a_, base):
    d = a_.ap.values - base.ap.values
    bt = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    return d.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5)


# Rows are ordered by signed difference, descending, within each group. The edge-encoding
# arm is APPENDED rather than inserted: ci() draws from one seeded generator in row order,
# so appending leaves the five existing intervals bit-for-bit unchanged.
rows = [('TRACE-GNN vs XGBoost, same feature columns', g,  x, 'design'),
        ('Cardholder history as structure',     u,  x, 'design'),
        ('Merchant relation added',             g,  u, 'design'),
        ('Third message-passing layer',         d3, g, 'arch'),
        ('Maximum-pool aggregation',            mx, g, 'arch')]
if eb is not None:
    rows.append(('Edge encoding removed',       eb, g, 'arch'))

fig, ax = plt.subplots(figsize=(9.8, 4.2 + 0.7 * (len(rows) - 5)))
ys = np.arange(len(rows))[::-1]
# group geometry, derived from the rows rather than hard-coded, so a new row cannot
# silently leave the divider and the group labels behind
n_design = sum(1 for r in rows if r[3] == 'design')
top_design, top_arch = ys[0], ys[n_design]
divider = top_arch + 0.5
print("  effect sizes")
for y, (lab, a_, base, kind) in zip(ys, rows):
    mm, lo, hi = ci(a_, base)
    col = GREEN if kind == 'design' else GREY
    if lo < 0 < hi or abs(mm) < BAND:
        col = GREY
    ax.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle='round', zorder=2)
    ax.scatter([mm], [y], s=80, color=col, zorder=3, edgecolor='white', linewidth=.9)
    ax.text(hi + 0.008, y, f'{mm:+.3f}', va='center', fontsize=10, color='black')
    pval, aw, bw, ties = paired(a_, base)
    verdict = "inside the noise band" if abs(mm) < BAND else ""
    print(f"    {lab:38s} {mm:+.3f}  [{lo:+.3f},{hi:+.3f}]  p {pval:.4f}  "
          f"reference leads {bw}/{len(a_)} (arm {aw}, ties {ties})  "
          f"{abs(mm)/BAND:.1f}x band  {verdict}")
ax.axvline(0, color='black', lw=1)
ax.axhline(divider, color='0.85', lw=1, ls='--')
ax.axvspan(-BAND, BAND, color=GREY, alpha=.10, zorder=0)
ax.text(BAND + 0.004, divider + 1.05,
        f'run-to-run noise\non a difference\n±{BAND:.3f}', fontsize=10,
        color='black', va='center')
ax.text(-0.045, top_design - 0.38, 'Design choices', fontsize=10, color='black')
ax.text(-0.045, top_arch + 0.12, 'Architecture variants', fontsize=10, color='black')
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlabel('Change in average precision (95% bootstrap CI)'); ax.set_xlim(-0.05, 0.31)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F8_effect_sizes.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'F8_effect_sizes.pdf'), bbox_inches='tight')
print('\nwrote F7_folds_convergence and F8_effect_sizes to', os.path.abspath(OUT))
