"""
protocol_v2 — the fold sets, built once, saved, then only ever loaded.

WHY THIS FILE EXISTS

The P1 arms (P1_causal and P1_leaky) must be trained on exactly the same 80/20 split.
In the old notebook that was true by construction: one `folds_p1` object was built in
cell 3 and handed to both arms in cell 4. Here the two arms run in separate notebooks,
in separate processes, hours apart, so "the same object" is no longer available as a
guarantee. This module replaces it with a stronger one:

    the split is generated ONCE, written to results/splits/p1_split.npz with its row
    indices, hashed, and every later use LOADS that file and verifies the hash.

A2 and A3 therefore do not each draw a split that ought to match. They read the same
bytes off disk and both print the hash. Two printed hashes agreeing is the check.

CONSTRUCTION - identical to notebooks/08b_protocol_11feat.ipynb cell 3, which is what
the published arms used. Only the epoch budget changes in protocol v2; the fold sets do
not.

    folds_p3   filter_folds(make_folds(df, cfg), 'is_fraud', 5, 5)   -> 17 weekly folds
    p3_test    every P3 test row, sorted
    p2_train   every row before the first P3 test row
    folds_p2   one fold: train on p2_train, score all of p3_test
    universe   p2_train + p3_test  (note: week 25 is NOT in it, because week 25 was
               filtered out of the fold set for having too few frauds)
    folds_p1   rng = default_rng(42); shuffled = rng.permutation(universe)
               train = sort(shuffled[:int(0.8*len)]),  test = sort(shuffled[int(0.8*len):])

A note worth keeping, from the review: P1's validation rows are `train[-15%:]` after
sorting, so they are the temporally latest 15% of a sample drawn from the whole period,
and are therefore contemporaneous with P1's test rows. That is a specification detail of
using a random split at all, not a separate leakage mechanism. It is recorded in the
splits manifest so anyone reconstructing P1 knows.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

import config as C
from guards import assert_writable, sha256_file

P1_FILE = C.SPLITS_DIR / "p1_split.npz"
P2_FILE = C.SPLITS_DIR / "p2_cutoff.npz"
P3_FILE = C.SPLITS_DIR / "p3_folds.npz"
MANIFEST = C.SPLITS_DIR / "SPLITS_MANIFEST.json"

P1_SEED = 42            # == TrainConfig seed; the old notebook used SEED for both
P1_TRAIN_FRAC = 0.8


def array_sha256(*arrays) -> str:
    """Hash of the array CONTENTS, independent of file container or compression.

    Everything is cast to int64 first. These are row indices, so the values are identical
    either way, but np.arange gives int32 on some platforms and the dtype would otherwise
    enter the digest — making a hash computed at save time fail to match the same indices
    loaded back on a different machine.
    """
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- build
def build_splits(df, cfg, temporal_folds, filter_folds) -> dict:
    """Derive all three fold sets from the loaded frame. Pure and deterministic."""
    folds_all = temporal_folds.make_folds(df, cfg)
    folds_p3, fold_report = filter_folds(df, folds_all, cfg.target_col, 5, 5)
    weeks = [int(w) for w, _, _ in folds_p3]
    if len(folds_p3) != C.EXPECT["n_folds_p3"]:
        raise AssertionError(f"{len(folds_p3)} P3 folds, expected {C.EXPECT['n_folds_p3']}")
    if tuple(weeks) != C.EXPECT["test_weeks"]:
        raise AssertionError(f"P3 weeks {weeks}, expected {list(C.EXPECT['test_weeks'])}")

    p3_test = np.sort(np.concatenate([te for _, _, te in folds_p3]))
    first_test = int(p3_test.min())
    pos_all = np.arange(len(df))
    p2_train = pos_all[pos_all < first_test]
    p2_test = p3_test

    universe = np.sort(np.concatenate([p2_train, p3_test]))
    rng = np.random.default_rng(P1_SEED)
    shuffled = rng.permutation(universe)
    cut = int(P1_TRAIN_FRAC * len(shuffled))
    p1_train = np.sort(shuffled[:cut])
    p1_test = np.sort(shuffled[cut:])

    e = C.EXPECT
    for name, got, want in (("p2_train", len(p2_train), e["p2_train_rows"]),
                            ("p3_test", len(p3_test), e["p3_scored_rows"]),
                            ("universe", len(universe), e["p1_universe_rows"]),
                            ("p1_train", len(p1_train), e["p1_train_rows"]),
                            ("p1_test", len(p1_test), e["p1_test_rows"])):
        if got != want:
            raise AssertionError(f"{name} has {got:,} rows, expected {want:,}")

    return {
        "p3": {"weeks": weeks,
               "train": [np.asarray(tr) for _, tr, _ in folds_p3],
               "test": [np.asarray(te) for _, _, te in folds_p3]},
        "p2": {"train": p2_train, "test": p2_test, "first_test_row": first_test},
        "p1": {"train": p1_train, "test": p1_test},
        "fold_report": fold_report,
    }


def _pack(arrays):
    """Ragged list of index arrays -> one flat array plus offsets."""
    flat = np.concatenate(arrays) if arrays else np.empty(0, np.int64)
    off = np.cumsum([0] + [len(a) for a in arrays]).astype(np.int64)
    return flat.astype(np.int64), off


def _unpack(flat, off):
    return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def save_splits(s: dict) -> dict:
    d = assert_writable(C.SPLITS_DIR)
    d.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(assert_writable(P1_FILE),
                        train_idx=s["p1"]["train"], test_idx=s["p1"]["test"])
    np.savez_compressed(assert_writable(P2_FILE),
                        train_idx=s["p2"]["train"], test_idx=s["p2"]["test"],
                        first_test_row=np.array([s["p2"]["first_test_row"]]))
    tr_flat, tr_off = _pack(s["p3"]["train"])
    te_flat, te_off = _pack(s["p3"]["test"])
    np.savez_compressed(assert_writable(P3_FILE),
                        weeks=np.asarray(s["p3"]["weeks"], np.int64),
                        train_flat=tr_flat, train_off=tr_off,
                        test_flat=te_flat, test_off=te_off)

    man = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Generated ONCE and reused. P1_causal and P1_leaky must load the same "
                   "p1_split.npz; that is what makes them a matched pair.",
        "construction": "identical to notebooks/08b_protocol_11feat.ipynb cell 3",
        "p1": {
            "file": P1_FILE.name,
            "file_sha256": sha256_file(P1_FILE),
            "seed": P1_SEED, "train_frac": P1_TRAIN_FRAC,
            "n_train": int(len(s["p1"]["train"])), "n_test": int(len(s["p1"]["test"])),
            "train_sha256": array_sha256(s["p1"]["train"]),
            "test_sha256": array_sha256(s["p1"]["test"]),
            "pair_sha256": array_sha256(s["p1"]["train"], s["p1"]["test"]),
            "used_by": ["P1_causal", "P1_leaky"],
            "validation_note": "run_date_gnn_fold takes val = train[-15%:] after the "
                               "train indices are sorted, so P1's validation rows are the "
                               "temporally latest 15% of a sample spanning the whole "
                               "period and are contemporaneous with P1's test rows. This "
                               "follows from using a random split; it is a specification "
                               "detail, not a third leakage mechanism.",
        },
        "p2": {
            "file": P2_FILE.name, "file_sha256": sha256_file(P2_FILE),
            "first_test_row": int(s["p2"]["first_test_row"]),
            "n_train": int(len(s["p2"]["train"])), "n_test": int(len(s["p2"]["test"])),
            "train_sha256": array_sha256(s["p2"]["train"]),
            "test_sha256": array_sha256(s["p2"]["test"]),
            "used_by": ["P2_cutoff"],
        },
        "p3": {
            "file": P3_FILE.name, "file_sha256": sha256_file(P3_FILE),
            "weeks": s["p3"]["weeks"],
            "n_test_total": int(sum(len(a) for a in s["p3"]["test"])),
            "test_sha256": array_sha256(*s["p3"]["test"]),
            "train_sha256": array_sha256(*s["p3"]["train"]),
            "used_by": ["P3_acausal (trained here)",
                        "P3_rolling (canonical, NOT trained here — same fold set, "
                        "verified in A5 by comparing per-fold n against per_fold.csv)"],
        },
        "fold_labels": {"P1": C.FOLD_LABEL_P1, "P2": C.FOLD_LABEL_P2,
                        "note": "single-fold arms need an integer label because "
                                "run_date_gnn_fold calls int() on it. These sentinels "
                                "cannot collide with a week number. The assembly step "
                                "recovers each scored row's week from the test-index "
                                "arrays above, never from the label."},
    }
    p = assert_writable(MANIFEST)
    p.write_text(json.dumps(man, indent=1))
    return man


# --------------------------------------------------------------------------- load
def _verify(name, arrays, expect_sha, man_path):
    got = array_sha256(*arrays)
    if got != expect_sha:
        raise AssertionError(
            f"\n\n  {name} DOES NOT MATCH THE SAVED SPLIT.\n"
            f"    manifest {man_path}\n"
            f"    expected {expect_sha}\n"
            f"    loaded   {got}\n"
            f"  The two P1 arms would no longer be a matched pair. Stop and find out why "
            f"before training anything.\n")
    return got


ALL_SPLIT_FILES = (P1_FILE, P2_FILE, P3_FILE, MANIFEST)


def load_splits(verbose: bool = True) -> dict:
    """Load the saved fold sets and verify every hash against SPLITS_MANIFEST.json."""
    missing = [p.name for p in ALL_SPLIT_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing from {C.SPLITS_DIR}: {missing}. The saved splits are incomplete. "
            f"Run notebook A1 first — it builds and saves the splits that every later arm "
            f"reuses. If some files are present and some are not, move the whole splits/ "
            f"directory aside and let A1 rebuild it; do not repair it by hand.")
    man = json.loads(MANIFEST.read_text())

    p1 = np.load(P1_FILE)
    p1_train, p1_test = p1["train_idx"], p1["test_idx"]
    _verify("p1_split.npz train_idx", [p1_train], man["p1"]["train_sha256"], MANIFEST)
    _verify("p1_split.npz test_idx", [p1_test], man["p1"]["test_sha256"], MANIFEST)

    p2 = np.load(P2_FILE)
    p2_train, p2_test = p2["train_idx"], p2["test_idx"]
    _verify("p2_cutoff.npz train_idx", [p2_train], man["p2"]["train_sha256"], MANIFEST)
    _verify("p2_cutoff.npz test_idx", [p2_test], man["p2"]["test_sha256"], MANIFEST)

    p3 = np.load(P3_FILE)
    weeks = [int(w) for w in p3["weeks"]]
    p3_train = _unpack(p3["train_flat"], p3["train_off"])
    p3_test = _unpack(p3["test_flat"], p3["test_off"])
    _verify("p3_folds.npz test", p3_test, man["p3"]["test_sha256"], MANIFEST)
    _verify("p3_folds.npz train", p3_train, man["p3"]["train_sha256"], MANIFEST)

    # The number A2 and A3 print. RECOMPUTED from the arrays that were just loaded off
    # disk, not copied out of the manifest — otherwise both notebooks would be printing
    # the same string from the same JSON file and the comparison would prove nothing.
    pair = _verify("p1_split.npz train+test pair", [p1_train, p1_test],
                   man["p1"]["pair_sha256"], MANIFEST)

    if verbose:
        print("saved fold sets, every hash recomputed from the loaded arrays and checked "
              "against SPLITS_MANIFEST.json")
        print(f"  P1  train {len(p1_train):>7,}  test {len(p1_test):>7,}")
        print(f"  P2  train {len(p2_train):>7,}  test {len(p2_test):>7,}")
        print(f"  P3  {len(weeks)} folds, weeks {weeks[0]}-{weeks[-1]}, "
              f"test rows {sum(len(a) for a in p3_test):,}")

    return {
        "manifest": man,
        "p1_pair_sha256_recomputed": pair,
        "p1": {"train": p1_train, "test": p1_test},
        "p2": {"train": p2_train, "test": p2_test,
               "first_test_row": int(p2["first_test_row"][0])},
        "p3": {"weeks": weeks, "train": p3_train, "test": p3_test},
    }


def ensure_splits(df, cfg, temporal_folds, filter_folds, verbose: bool = True) -> dict:
    """Build and save on first call, load and verify on every call after. Whichever
    notebook runs first creates them; the rest reuse them.

    Refuses to act on a half-present splits/ directory. Regenerating over existing npz
    files would be silent, and although generation is deterministic, "it would probably
    have come out the same" is not the guarantee this file is for.
    """
    present = [p for p in ALL_SPLIT_FILES if p.exists()]
    if len(present) == len(ALL_SPLIT_FILES):
        if verbose:
            print("  [splits] already generated; loading and verifying")
        return load_splits(verbose=verbose)
    if present:
        raise FileExistsError(
            f"{C.SPLITS_DIR} is incomplete: {[p.name for p in present]} present, "
            f"{[p.name for p in ALL_SPLIT_FILES if not p.exists()]} missing. Refusing to "
            f"regenerate over the files that are there. Move the whole splits/ directory "
            f"aside and rerun A1. Nothing is deleted for you.")
    if verbose:
        print("  [splits] first run — generating and saving")
    s = build_splits(df, cfg, temporal_folds, filter_folds)
    man = save_splits(s)
    if verbose:
        print(f"  [splits] wrote {C.SPLITS_DIR.name}/ and SPLITS_MANIFEST.json")
        print(f"  [splits] P1 pair sha256 {man['p1']['pair_sha256']}")
    return load_splits(verbose=verbose)


# --------------------------------------------------------------------------- for training
def folds_for(arm: str, sp: dict):
    """The (label, train_idx, test_idx) list run_date_gnn_fold consumes, for one arm."""
    kind = C.ARM_SPEC[arm]["split"]
    if kind == "p1":
        return [(C.FOLD_LABEL_P1, sp["p1"]["train"], sp["p1"]["test"])]
    if kind == "p2":
        return [(C.FOLD_LABEL_P2, sp["p2"]["train"], sp["p2"]["test"])]
    if kind == "p3":
        return [(w, tr, te) for w, tr, te in
                zip(sp["p3"]["weeks"], sp["p3"]["train"], sp["p3"]["test"])]
    raise ValueError(kind)


def week_lookup(arm: str, sp: dict, week_of_row: np.ndarray) -> dict:
    """Maps (fold label, fold-local row_pos) -> global row -> week.

    This is review item D7: instead of adding a row_index parameter to the training
    function, the score files keep Section 5.1's schema (row_pos is a per-fold counter)
    and the global position is recovered here, from the saved test-index arrays. Order is
    preserved because _infer scores target_idx in order.
    """
    if arm in C.LOCKED_ARMS:
        # The canonical arm's scores already carry the week in the fold column, so this
        # lookup is not used for it. assemble.load_scores handles it separately.
        raise ValueError(f"{arm} is read from the canonical run; its scores carry the "
                         f"week directly and need no lookup")
    if arm not in C.ARM_SPEC:
        raise ValueError(f"unknown arm {arm!r}. Known: {', '.join(C.ARM_SPEC)}")
    kind = C.ARM_SPEC[arm]["split"]
    if kind == "p1":
        return {C.FOLD_LABEL_P1: sp["p1"]["test"]}
    if kind == "p2":
        return {C.FOLD_LABEL_P2: sp["p2"]["test"]}
    if kind == "p3":
        return {int(w): te for w, te in zip(sp["p3"]["weeks"], sp["p3"]["test"])}
    raise ValueError(f"unknown split kind {kind!r} for arm {arm!r}")
