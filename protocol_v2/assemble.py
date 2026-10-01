"""
protocol_v2 — building the tables. No GPU, no training, no writing outside protocol_v2.

Reads five arms:

    P2_cutoff, P1_causal, P1_leaky, P3_acausal    trained here, from results/arms/
    P3_rolling                                    NOT trained here. Read read-only from
                                                  the canonical Section 5.1 run.

Every setting used below comes from assembly_settings.py, which was frozen before the
first arm was trained.

Recovering the week of a scored row
-----------------------------------
run_date_gnn_fold writes row_pos as a per-fold counter (Section 5.1's schema; review item
D7 says to leave it that way). _infer scores its target index in order, so for a fold
whose test index array is `te`, scored row i is the global row te[i]. The week follows
from the frame. For P3 arms the fold label is already the week, so no lookup is needed —
but the same mapping is applied anyway, and the two answers are cross-checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

import assembly_settings as A
import config as C
import guards
import splits as S


# --------------------------------------------------------------------------- loading
def load_scores(arm: str, sp: dict, week_of_row: np.ndarray) -> pd.DataFrame:
    """Per-transaction scores for one arm, with a `week` column attached."""
    if arm in C.LOCKED_ARMS:
        path = guards.read_canonical("scores.csv")          # read-only, canonical run
        s = pd.read_csv(path)
        s["week"] = s["fold"].astype(int)                   # fold label IS the week here
        s["source"] = "canonical Section 5.1 run (NOT retrained)"
        return s

    guards.assert_trainable(arm)
    path = C.ARMS_DIR / arm / "scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} — has arm {arm} been run?")
    s = pd.read_csv(path)
    lut = S.week_lookup(arm, sp, week_of_row)
    glob = np.empty(len(s), np.int64)
    for label, te in lut.items():
        m = s["fold"].astype(int) == int(label)
        if not m.any():
            continue
        rp = s.loc[m, "row_pos"].to_numpy()
        if rp.max() >= len(te) or not np.array_equal(np.sort(rp), np.arange(len(te))):
            raise AssertionError(
                f"{arm} fold {label}: row_pos is not a complete 0..{len(te)-1} range. "
                f"The per-fold counter assumption behind the week mapping does not hold.")
        glob[m.to_numpy()] = np.asarray(te)[rp]
    s["global_row"] = glob
    s["week"] = week_of_row[glob]
    s["source"] = f"protocol_v2 arms/{arm}"

    # cross-check on the rolling arm, where the fold label is independently the week
    if C.ARM_SPEC[arm]["split"] == "p3":
        bad = int((s["week"].to_numpy() != s["fold"].astype(int).to_numpy()).sum())
        if bad:
            raise AssertionError(f"{arm}: {bad} rows where the recovered week disagrees "
                                 f"with the fold label")
    return s


# --------------------------------------------------------------------------- metrics
def _at_k(y, sc, rate):
    k = max(1, int(np.ceil(len(y) * rate)))
    top = np.argsort(-sc, kind="mergesort")[:k]
    tp = float(y[top].sum())
    fp = k - tp
    pos = float(y.sum())
    fn = pos - tp
    tn = len(y) - tp - fp - fn
    prec = tp / k if k else 0.0
    rec = tp / pos if pos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else 0.0
    return dict(k=k, precision_at_k=prec, recall_at_k=rec, f1_at_k=f1, mcc_at_k=mcc)


def per_week(arm: str, s: pd.DataFrame, rate: float = C.ALERT_RATE) -> pd.DataFrame:
    """Weekly metrics, restricted to the 17 evaluation weeks.

    THE FILTER IS NOT COSMETIC. P1's universe is `p2_train UNION p3_test`, and p2_train is
    every row before week 8 — 148,946 of them. So a random 20% of that universe
    necessarily samples weeks 0 to 7, and those weeks contain fraud: 13, 19, 12, 16, 34,
    8, 11 and 14 cases respectively. Without this filter the P1 arms report 24 weeks, the
    declared fold-set rule is violated on every run, and pre-week-8 rows leak into
    per_week_all_arms.csv.

    notebooks/08b_protocol_11feat.ipynb cell 6 had this filter (`for w in locked_weeks`).
    An earlier draft of this file dropped it.
    """
    evaluation_weeks = set(C.EXPECT["test_weeks"])
    s = s[s["week"].isin(evaluation_weeks)]
    rows = []
    for w, g in s.groupby("week", sort=True):
        y = g.y.to_numpy().astype(int)
        sc = g.score.to_numpy(dtype="float64")
        if y.sum() == 0:
            continue                       # AP and AUROC undefined; this is what drops
        m = {"arm": arm, "test_week": int(w), "n": len(y), "n_fraud": int(y.sum()),
             "ap": float(average_precision_score(y, sc)),
             "auroc": (float(roc_auc_score(y, sc)) if 0 < y.sum() < len(y) else np.nan)}
        m.update(_at_k(y, sc, rate))
        rows.append(m)
    return pd.DataFrame(rows)


def all_per_week(sp: dict, week_of_row: np.ndarray, verbose: bool = True) -> pd.DataFrame:
    frames = []
    for arm in A.LADDER_ORDER:
        s = load_scores(arm, sp, week_of_row)
        pw = per_week(arm, s)
        if verbose:
            tag = "LOCKED, canonical" if arm in C.LOCKED_ARMS else "new, 200 epochs"
            print(f"  {arm:11} {len(s):>7,} scored rows -> {len(pw):>2} weeks with fraud "
                  f"({tag})")
        frames.append(pw)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- fold sets
def weeks_for(a: str, b: str) -> list:
    """The pre-declared fold-set rule, applied. 16 weeks if a P1 arm is involved, 17
    otherwise — and always intersected with what each arm actually has."""
    involves_p1 = a.startswith("P1") or b.startswith("P1")
    return list(A.FOLD_SET_RULE["weeks_with_P1"] if involves_p1
                else A.FOLD_SET_RULE["weeks_without_P1"])


def paired(pw: pd.DataFrame, a: str, b: str, metric: str = "ap") -> dict:
    declared = set(weeks_for(a, b))
    A_ = pw[pw.arm == a].set_index("test_week")
    B_ = pw[pw.arm == b].set_index("test_week")
    common = sorted(declared & set(A_.index) & set(B_.index))
    av = A_.loc[common, metric].to_numpy(dtype="float64")
    bv = B_.loc[common, metric].to_numpy(dtype="float64")
    ok = ~(np.isnan(av) | np.isnan(bv))
    av, bv = av[ok], bv[ok]
    diff = av - bv
    n_zero = int(np.count_nonzero(diff == 0))

    # method is 'exact', unconditionally. There is no fallback. An earlier version
    # switched to 'auto' whenever a zero difference appeared, on the strength of an audit
    # claim that none did — but P1_leaky - P1_causal is exactly zero in weeks 10 and 21,
    # so that switch fired on the one contrast the manuscript quotes and produced 0.30029
    # where the paper prints 0.326. zero_method='wilcox' drops the zeros and the exact
    # test runs on what remains, which is what the published number did.
    method = A.WILCOXON["method"]
    n_effective = int(len(av) - n_zero)
    if n_effective > 25:
        raise AssertionError(
            f"{a} - {b}: n_effective={n_effective} exceeds scipy's exact-test limit of "
            f"25. The declared method is 'exact' and there is deliberately no automatic "
            f"downgrade. Decide explicitly and record the decision.")

    if len(av) == 0 or np.allclose(diff, 0):
        stat, p = np.nan, np.nan
    else:
        r = wilcoxon(av, bv, alternative=A.WILCOXON["alternative"],
                     zero_method=A.WILCOXON["zero_method"], method=method)
        stat, p = float(r.statistic), float(r.pvalue)

    note = ""
    if n_zero:
        note = (f"{n_zero} of {len(av)} weekly differences are exactly zero (weeks "
                f"{[int(w) for w, d in zip(common, diff) if d == 0]}); zero_method="
                f"'wilcox' drops them, so the test used n={n_effective} while delta is "
                f"over n={len(av)}")
    return {"contrast": f"{a} - {b}", "metric": metric, "n_weeks": int(len(av)),
            "n_zero": n_zero, "n_effective": n_effective,
            "weeks": common, "delta": float(diff.mean()) if len(diff) else np.nan,
            "median_delta": float(np.median(diff)) if len(diff) else np.nan,
            "consistent": int((diff > 0).sum()),
            "statistic": stat, "p_value": p, "wilcoxon_method": method, "note": note}


# --------------------------------------------------------------------------- tables
def ladder(pw: pd.DataFrame) -> pd.DataFrame:
    """Table 6. Every row on the same weeks — the intersection across all five arms."""
    common = sorted(set.intersection(*[set(pw[pw.arm == a].test_week)
                                       for a in A.LADDER_ORDER]))
    sub = pw[pw.test_week.isin(common)]
    t = (sub.groupby("arm")[list(A.REPORTED_METRICS)].mean()
            .reindex(list(A.LADDER_ORDER)))
    t["n_weeks"] = sub.groupby("arm").size().reindex(list(A.LADDER_ORDER))
    t["source"] = ["canonical Section 5.1 (not retrained)" if a in C.LOCKED_ARMS
                   else "protocol_v2, 200 epochs" for a in t.index]
    return t.round(A.ROUNDING["tables"]), common


def decomposition(pw: pd.DataFrame, metric: str = "ap",
                  include_corrections: bool = False) -> pd.DataFrame:
    """Table 7. The two 'single correction, ...' entries in CONTRASTS are aliases of two
    rows already present, so they are left out by default to keep the table one row per
    contrast; pass include_corrections=True to see them named in their second role."""
    wanted = [(n, a, b) for n, a, b in A.CONTRASTS
              if include_corrections or not n.startswith("single correction")]
    rows = []
    for name, a, b in wanted:
        r = paired(pw, a, b, metric)
        r["effect"] = name
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(0, "effect", out.pop("effect"))
    out["weeks"] = out["weeks"].apply(lambda w: f"{min(w)}-{max(w)}" if w else "")
    return out


# --------------------------------------------------------------- completeness gate (A5)
def assert_arms_complete(verbose: bool = True) -> dict:
    """Refuse to assemble anything until all four new arms are finished and the canonical
    arm is intact. Without this, a P3_acausal that had only run its two smoke folds would
    quietly produce a two-week ladder, because ladder() intersects whatever weeks it
    finds.
    """
    e, problems, report = C.EXPECT, [], {}

    for arm in C.TRAINABLE_ARMS:
        d = C.ARMS_DIR / arm
        info = {"dir": str(d)}
        sp_kind = C.ARM_SPEC[arm]["split"]
        want_folds = 17 if sp_kind == "p3" else 1
        want_rows = e["p3_scored_rows"] if sp_kind in ("p2", "p3") else e["p1_test_rows"]

        s_path = d / "summary.json"
        if not s_path.exists():
            problems.append(f"{arm}: no summary.json — the arm has not finished. A smoke "
                            f"run leaves per_fold_PARTIAL.csv and no summary.")
            report[arm] = info
            continue
        s = json.loads(s_path.read_text())
        info["summary"] = s
        if not s.get("complete"):
            problems.append(f"{arm}: summary.json says complete={s.get('complete')!r}")
        if s.get("problems"):
            problems.append(f"{arm}: finalise recorded {s['problems']}")
        if s.get("n_folds") != want_folds:
            problems.append(f"{arm}: {s.get('n_folds')} folds, expected {want_folds}")
        if s.get("scored_rows") != want_rows:
            problems.append(f"{arm}: {s.get('scored_rows'):,} scored rows, expected "
                            f"{want_rows:,}")
        if s.get("epoch_budget") != C.TRAIN_KWARGS["epochs"]:
            problems.append(f"{arm}: epoch_budget {s.get('epoch_budget')}, expected "
                            f"{C.TRAIN_KWARGS['epochs']}")
        if (d / "per_fold_PARTIAL.csv").exists() and not (d / "per_fold.csv").exists():
            problems.append(f"{arm}: only per_fold_PARTIAL.csv is present")
        for f in ("per_fold.csv", "scores.csv", "val_curves.csv", "arm_manifest.json"):
            if not (d / f).exists():
                problems.append(f"{arm}: missing {f}")
        report[arm] = info

    # the canonical arm, read-only
    try:
        cs = pd.read_csv(guards.read_canonical("scores.csv"))
        cf = pd.read_csv(guards.read_canonical("per_fold.csv"))
        report["P3_rolling"] = {"scored_rows": int(len(cs)),
                                "positives": int(cs.y.sum()), "n_folds": int(len(cf)),
                                "source": str(C.CANONICAL_P3_DIR)}
        if len(cs) != e["p3_scored_rows"]:
            problems.append(f"P3_rolling: {len(cs):,} scored rows, expected "
                            f"{e['p3_scored_rows']:,}")
        if int(cs.y.sum()) != e["p3_positives"]:
            problems.append(f"P3_rolling: {int(cs.y.sum()):,} positives, expected "
                            f"{e['p3_positives']:,}")
        if len(cf) != e["n_folds_p3"]:
            problems.append(f"P3_rolling: {len(cf)} folds, expected {e['n_folds_p3']}")
    except FileNotFoundError as exc:
        problems.append(f"P3_rolling: {exc}")

    if problems:
        raise AssertionError(
            "REFUSING TO ASSEMBLE — the inputs are not all complete:\n  - " +
            "\n  - ".join(problems) +
            "\n\nFinish the missing arms first. A partial ladder is worse than no ladder, "
            "because it looks like a result.")
    if verbose:
        print("all five arms present and complete")
        for arm in C.TRAINABLE_ARMS:
            s = report[arm]["summary"]
            print(f"  {arm:11} {s['n_folds']:>2} folds  {s['scored_rows']:>7,} rows  "
                  f"{s['epoch_budget']} epochs  mean AP {s['mean_ap']:.4f}")
        r = report["P3_rolling"]
        print(f"  {'P3_rolling':11} {r['n_folds']:>2} folds  {r['scored_rows']:>7,} rows  "
              f"read-only from the canonical run")
    return report


def assert_week_coverage(pw: pd.DataFrame, verbose: bool = True) -> None:
    """The declared fold-set rule, checked rather than assumed: 16 weeks for the P1 arms,
    17 for the rest."""
    problems = []
    for arm in A.LADDER_ORDER:
        got = sorted(pw[pw.arm == arm].test_week)
        want = list(A.FOLD_SET_RULE["weeks_with_P1"] if arm.startswith("P1")
                    else A.FOLD_SET_RULE["weeks_without_P1"])
        if got != want:
            problems.append(f"{arm}: weeks {got}, expected {want}")
    if problems:
        raise AssertionError(
            "week coverage does not match the declared fold-set rule:\n  - " +
            "\n  - ".join(problems) +
            "\n\nThe rule says 16 weeks whenever a P1 arm is involved (week 24 has no "
            "fraud in P1's random 20%) and 17 otherwise. A different pattern means "
            "something other than that is going on.")
    if verbose:
        print("week coverage matches the declared rule: 16 for the P1 arms, 17 for the "
              "rest")


def factorial(pw: pd.DataFrame) -> pd.DataFrame:
    weeks = set(A.FOLD_SET_RULE["weeks_with_P1"])       # the 2x2 involves both P1 arms
    sub = pw[pw.test_week.isin(weeks)]
    g = pd.DataFrame(index=["random split", "rolling origin"],
                     columns=["unconstrained graph", "causal graph"], dtype=float)
    for (r, c), arm in A.FACTORIAL_2X2.items():
        g.loc[r, c] = float(sub[sub.arm == arm].ap.mean())
    g["graph effect"] = g["unconstrained graph"] - g["causal graph"]
    g.loc["split effect"] = g.loc["random split"] - g.loc["rolling origin"]
    return g.round(A.ROUNDING["tables"])


def excess_per_week(pw: pd.DataFrame, metric: str = "ap") -> pd.Series:
    """The non-additive excess, week by week, built term by term from
    assembly_settings.EXCESS_PER_WEEK rather than from a hand-written expression.

    There is one signed quantity here, not two. The aggregate excess is the MEAN of this
    vector, and the Wilcoxon test is run on this vector. An earlier draft derived the
    aggregate separately and got the opposite sign from the per-week test — the same
    inconsistency notebook 08b has. Deriving both from this one function removes the
    possibility.
    """
    weeks = sorted(set(A.FOLD_SET_RULE["weeks_with_P1"]))
    piv = (pw[pw.test_week.isin(weeks)]
           .pivot(index="test_week", columns="arm", values=metric))
    missing = [arm for _, arm in A.EXCESS_PER_WEEK if arm not in piv.columns]
    if missing:
        raise AssertionError(f"cannot compute the excess: {missing} absent from the "
                             f"per-week table")
    v = sum(coef * piv[arm] for coef, arm in A.EXCESS_PER_WEEK)
    return v.dropna()


def non_additivity(pw: pd.DataFrame, metric: str = "ap") -> dict:
    """Aggregate excess, its per-week vector, and the paired test — all one quantity.

    Cross-checked two ways before returning:
      * the aggregate equals the mean of the per-week vector, to 1e-12
      * the aggregate equals total - (the two single corrections named in the sign
        convention), to 1e-12
    Either check failing means the convention and the arithmetic have come apart, which
    is exactly the bug this replaced.
    """
    v = excess_per_week(pw, metric)
    mean_from_vector = float(v.mean()) if len(v) else float("nan")

    a, b = A.SIGN_CONVENTION["single_corrections"]
    singles = [paired(pw, x, y, metric) for name, x, y in A.CONTRASTS if name in (a, b)]
    if len(singles) != 2:
        raise AssertionError(f"expected two single corrections, matched {len(singles)}")
    total = paired(pw, "P1_leaky", "P3_rolling", metric)
    sum_singles = sum(s["delta"] for s in singles)
    mean_from_contrasts = total["delta"] - sum_singles

    if len(v) and abs(mean_from_vector - mean_from_contrasts) > 1e-12:
        raise AssertionError(
            f"the non-additive excess disagrees with itself:\n"
            f"  mean of the per-week vector      {mean_from_vector:+.10f}\n"
            f"  total minus the two corrections  {mean_from_contrasts:+.10f}\n"
            f"The sign convention in assembly_settings.py and the arithmetic here have "
            f"come apart. Do not report either number.")

    out = {"n": int(len(v)), "weeks": [int(w) for w in v.index],
           "excess": mean_from_vector,
           "sum_of_single_corrections": float(sum_singles),
           "total_protocol_effect": float(total["delta"]),
           "single_corrections": {s["contrast"]: float(s["delta"]) for s in singles},
           "formula": A.SIGN_CONVENTION["excess_formula"],
           "sign_means": A.SIGN_CONVENTION["excess_sign_means"],
           "consistent_weeks": int((v > 0).sum())}

    if len(v) == 0 or np.allclose(v, 0):
        out.update(p_value=float("nan"), wilcoxon_method=None, n_zero=0)
        return out
    n_zero = int(np.count_nonzero(v.to_numpy() == 0))
    r = wilcoxon(v.to_numpy(), alternative=A.WILCOXON["alternative"],
                 zero_method=A.WILCOXON["zero_method"], method=A.WILCOXON["method"])
    out.update(p_value=float(r.pvalue), statistic=float(r.statistic),
               wilcoxon_method=A.WILCOXON["method"], n_zero=n_zero,
               n_effective=int(len(v) - n_zero))
    return out


def interaction(pw: pd.DataFrame, metric: str = "ap") -> dict:
    """Kept as a name, delegating to non_additivity, so nothing can reintroduce a second
    expression for the same quantity."""
    return non_additivity(pw, metric)


def write_all(pw, lad, common, dec, fac, inter) -> dict:
    d = guards.assert_writable(C.DERIVED_DIR)
    d.mkdir(parents=True, exist_ok=True)
    pw.to_csv(guards.assert_writable(d / "per_week_all_arms.csv"), index=False)
    lad.to_csv(guards.assert_writable(d / "protocol_ladder.csv"))
    dec.to_csv(guards.assert_writable(d / "leak_decomposition.csv"), index=False)
    fac.to_csv(guards.assert_writable(d / "factorial_2x2.csv"))

    ex = excess_per_week(pw)
    ex.rename("excess").to_frame().to_csv(
        guards.assert_writable(d / "non_additive_excess_per_week.csv"))

    tests = pd.concat([pd.DataFrame([paired(pw, a, b, m) for _, a, b in A.CONTRASTS])
                       for m in ("ap", "auroc", "precision_at_k", "recall_at_k")],
                      ignore_index=True)
    tests["weeks"] = tests["weeks"].apply(lambda w: f"{min(w)}-{max(w)}" if w else "")
    tests.to_csv(guards.assert_writable(d / "paired_tests.csv"), index=False)

    json.dump({"ladder_weeks": common,
               "non_additivity": inter,
               "sign_convention": A.SIGN_CONVENTION,
               "settings_sha256": A.settings_sha256()},
              open(guards.assert_writable(d / "assembly_notes.json"), "w"),
              indent=1, default=str)
    return {"dir": str(d), "files": sorted(p.name for p in d.glob("*"))}
