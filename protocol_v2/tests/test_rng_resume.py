"""
Does a smoke run followed by a resume draw the same random stream as one uninterrupted
run? This is verification item (b), and it is the reason run_date_gnn_fold gained
rng_state_path.

WHAT WAS WRONG BEFORE
    run_date_gnn_fold seeds once, at the top, then loops over folds. On a resume it seeds
    again and skips the completed folds. So the folds trained after a resume see the
    stream a FRESH run would have given fold 1, not the stream a continuous run would
    have reached by that point. The canonical run manifest says exactly this about
    depth3 week 24: "a resumed fold starts from a fresh RNG state rather than the state a
    continuous run would have reached."

    For A4 that matters, because the review asks for a smoke run over weeks 8-9 and then
    a resume for 10-24. Without a fix, weeks 10-24 would be trained on a stream no
    uninterrupted run ever produces, and the arm would not be the arm we meant to run.

WHAT THE FIX DOES
    With rng_state_path set, the generator state (numpy, torch CPU, torch CUDA) is saved
    at every fold boundary and restored on resume. Split then resume == uninterrupted.

    ONE ENTRY PER FOLD is kept, not just the newest. foldckpt.csv and the state file are
    two separate writes, so a crash between them leaves them one fold apart. With every
    fold's state on file, whatever the checkpoint says finished has a matching state.

WHAT THIS TEST PROVES, AND WHAT IT DOES NOT
    Proves: the random STREAM is identical. Every draw, in order.
    Does not prove: bitwise-identical training results. It cannot, and nothing could.
    Scatter-add on the GPU accumulates in nondeterministic order, which is why this
    project measures SD 0.0091 in mean AP across executions of one configuration. The
    claim is "same stream", not "same bits".

Run it:  cd protocol_v2 && python -m pytest tests/test_rng_resume.py -v
     or: cd protocol_v2 && python tests/test_rng_resume.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_R = Path(__file__).resolve().parents[2]
for _d in (str(_R / "src"), str(_R), str(_R / "protocol_v2")):   # protocol_v2 LAST -> first
    while _d in sys.path:
        sys.path.remove(_d)
    sys.path.insert(0, _d)

SEED = 42
N_FOLDS = 5
SPLIT_AFTER = 2           # stands in for the A4 smoke run over weeks 8 and 9
DRAWS_PER_FOLD = 7


# --------------------------------------------------------------------- numpy only
def _numpy_fold(i):
    """Stand-in for one fold's numpy consumption (np.random.permutation of the train
    rows, once per epoch). The count differs per fold on purpose: a real fold's epoch
    count is data-dependent, so the stream position after fold k is not predictable."""
    return [float(np.random.rand()) for _ in range(DRAWS_PER_FOLD + i)]


def test_numpy_stream_survives_a_resume():
    # (1) one uninterrupted run
    np.random.seed(SEED)
    uninterrupted = [_numpy_fold(i) for i in range(N_FOLDS)]

    # (2) smoke run over the first SPLIT_AFTER folds, saving the state at the boundary
    np.random.seed(SEED)
    part_a = [_numpy_fold(i) for i in range(SPLIT_AFTER)]
    saved = np.random.get_state()

    # (3) a brand new process would reseed. Simulate that, then restore.
    np.random.seed(SEED)                      # what the old code did, and only that
    np.random.set_state(saved)                # what rng_state_path adds
    part_b = [_numpy_fold(i) for i in range(SPLIT_AFTER, N_FOLDS)]

    assert part_a + part_b == uninterrupted, "numpy stream diverged across the resume"

    # and show the failure the fix prevents: reseeding without restoring
    np.random.seed(SEED)
    naive = [_numpy_fold(i) for i in range(SPLIT_AFTER, N_FOLDS)]
    assert naive != part_b, ("reseeding produced the same numbers as restoring, which "
                             "means this test is not actually exercising anything")
    print("numpy   : split+resume == uninterrupted, and != naive reseed   OK")


# --------------------------------------------------------------------- torch too
def test_torch_stream_survives_a_resume():
    try:
        import torch
    except ImportError:                       # pragma: no cover
        print("torch   : not installed here, skipped")
        return

    def fold(i):
        return [torch.rand(3).tolist() for _ in range(DRAWS_PER_FOLD + i)]

    torch.manual_seed(SEED)
    uninterrupted = [fold(i) for i in range(N_FOLDS)]

    torch.manual_seed(SEED)
    part_a = [fold(i) for i in range(SPLIT_AFTER)]
    saved_cpu = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.manual_seed(SEED)
    torch.set_rng_state(saved_cpu)
    if saved_cuda is not None:
        torch.cuda.set_rng_state_all(saved_cuda)
    part_b = [fold(i) for i in range(SPLIT_AFTER, N_FOLDS)]

    assert part_a + part_b == uninterrupted, "torch stream diverged across the resume"
    print("torch   : split+resume == uninterrupted   OK")


# ------------------------------------------------ the real save/load round trip
def test_saved_state_file_round_trips():
    """Exercise the exact payload run_date_gnn_fold writes — one entry per fold — through
    torch.save/torch.load, and restore via the real pick_rng_state."""
    try:
        import torch
        from run_date_gnn import pick_rng_state, RNG_STATE_VERSION
    except ImportError:                       # pragma: no cover
        print("payload : torch not installed here, skipped")
        return

    def draw():
        return (float(np.random.rand()), torch.rand(1).item())

    np.random.seed(SEED); torch.manual_seed(SEED)
    [draw() for _ in range(11)]
    expect = [draw() for _ in range(5)]

    np.random.seed(SEED); torch.manual_seed(SEED)
    [draw() for _ in range(11)]
    snap = {"numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None)}
    payload = {"version": RNG_STATE_VERSION, "completed_weeks": [8, 9],
               "identity": "abc123", "fold_digests": {8: "aaaa", 9: "bbbb"},
               "n_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
               "states": {"8": {"numpy": None}, "9": snap}}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rng_state.pt"
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(p)                                   # the atomic write the code does

        np.random.seed(0); torch.manual_seed(0)          # scramble both generators
        st = torch.load(p, map_location="cpu", weights_only=False)
        entry = pick_rng_state(st, 9)                    # the real selector
        np.random.set_state(entry["numpy"])
        torch.set_rng_state(entry["torch_cpu"])
        if entry["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(entry["torch_cuda"])
        got = [draw() for _ in range(5)]

        # a fold with no saved state must raise, not fall back to something plausible
        try:
            pick_rng_state(st, 12)
        except RuntimeError:
            pass
        else:
            raise AssertionError("pick_rng_state invented a state for an unsaved fold")

    assert got == expect, "state did not survive the save/load round trip"
    print("payload : per-fold state file round-trips exactly, and an unsaved fold raises "
          "  OK")


def test_a_crash_between_the_two_writes_is_recoverable():
    """The checkpoint and the state file are two separate writes, so a crash between them
    leaves them one fold apart. Keeping ONE state per fold makes both directions
    recoverable; keeping only the newest made neither."""
    try:
        from run_date_gnn import check_rng_state, pick_rng_state, RNG_STATE_VERSION
    except ImportError:                       # pragma: no cover
        print("crashwin: torch not installed here, skipped")
        return

    FD = {8: "a", 9: "b", 10: "c"}
    base = {"version": RNG_STATE_VERSION, "identity": "id", "fold_digests": FD,
            "n_cuda_devices": 0, "torch_cuda": None}

    # crash AFTER the state write, BEFORE the checkpoint write: the state file is one
    # fold ahead. The extra entry is ignored and fold 9 restores cleanly.
    ahead = {**base, "completed_weeks": [8, 9, 10],
             "states": {"8": {}, "9": {"marker": "nine"}, "10": {}}}
    check_rng_state(ahead, {8, 9}, "id", FD, n_cuda_devices=0)
    assert pick_rng_state(ahead, 9)["marker"] == "nine"

    # crash AFTER the checkpoint write, BEFORE the state write: the state file is one
    # fold behind. That IS unrecoverable, and it must raise rather than restore the
    # wrong point in the stream.
    behind = {**base, "completed_weeks": [8], "states": {"8": {}}}
    try:
        check_rng_state(behind, {8, 9}, "id", FD, n_cuda_devices=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a state file missing a checkpointed fold was accepted")
    print("crashwin: a state file one fold AHEAD resumes cleanly; one fold BEHIND "
          "raises   OK")


# --------------------------------------------- the guard, exercised for real
def test_the_real_guard_refuses_every_bad_state():
    """Calls run_date_gnn.check_rng_state — the actual function run_date_gnn_fold calls —
    with each way a saved state can be wrong, and requires every one to raise.

    An earlier version of this test asserted `{8,9} != {8,9,10}` on two literals and
    imported nothing. It proved that two Python sets are unequal. This one runs the code.
    """
    try:
        from run_date_gnn import check_rng_state, RNG_STATE_VERSION
    except ImportError as exc:                        # pragma: no cover
        print(f"guard   : cannot import run_date_gnn here ({exc}); run this on the "
              f"workstation")
        return

    FD = {8: "aaaa", 9: "bbbb", 10: "cccc"}           # this run's per-fold digests
    good = {"version": RNG_STATE_VERSION, "completed_weeks": [8, 9], "identity": "abc123",
            # The generator payload is PER FOLD, under "states" — this mirrors what
            # _save_rng_state actually writes. The old fixture carried a top-level
            # "torch_cuda" that no real state file has, which is exactly why the
            # top-level-read bug in check_rng_state survived the test suite.
            "states": {"8": {"numpy": None, "torch_cpu": None, "torch_cuda": ["x", "y"]},
                       "9": {"numpy": None, "torch_cpu": None, "torch_cuda": ["x", "y"]}},
            "fold_digests": {8: "aaaa", 9: "bbbb"}, "n_cuda_devices": 2}

    # the happy path must NOT raise
    check_rng_state(good, {8, 9}, "abc123", FD, n_cuda_devices=2)

    # the SMOKE-THEN-RESUME path must not raise either: the state was saved after a
    # 2-fold call, the resume passes all 17 folds. The shared folds must match; the
    # extra ones are simply not yet done.
    check_rng_state(good, {8, 9}, "abc123", {**FD, **{w: f"f{w}" for w in range(11, 25)}},
                    n_cuda_devices=2)

    # torch.save/load round-trips dict keys as str in some paths; both must be accepted
    check_rng_state({**good, "fold_digests": {"8": "aaaa", "9": "bbbb"}},
                    {8, 9}, "abc123", FD, n_cuda_devices=2)

    cases = [
        ("fold list disagrees with the checkpoint",
         good, {8, 9, 10}, "abc123", FD, 2),
        # DELIBERATELY ABSENT: "fold list disagrees the other way" — a state file holding
        # MORE folds than foldckpt.csv records. That case used to live here, from before
        # the state file kept one entry per fold. It is not a bad state now: it is exactly
        # the crash-after-the-RNG-write-before-the-checkpoint-write window, which the
        # package accepts on purpose. test_crash_window above asserts that a state one
        # fold AHEAD resumes cleanly, and the README documents the extra entries as
        # ignored. Do not reinstate it — it would only pass if check_rng_state were
        # changed to refuse a recoverable resume.
        ("different config, builder or causality flag",
         {**good, "identity": "deadbeef"}, {8, 9}, "abc123", FD, 2),
        ("same configs and builder, DIFFERENT fold set (P1_causal vs P2_cutoff)",
         good, {8, 9}, "abc123", {8: "zzzz", 9: "bbbb"}, 2),
        ("a completed fold does not exist in this run's fold set",
         good, {8, 9}, "abc123", {8: "aaaa"}, 2),
        ("saved on a machine with a different CUDA device count",
         {**good, "n_cuda_devices": 1}, {8, 9}, "abc123", FD, 2),
        # A fold entry with no CUDA payload, on a CUDA machine. Checked per entry, so a
        # missing payload on ANY restorable fold must raise, not just the newest.
        ("no CUDA state saved, but this run uses CUDA",
         {**good, "states": {"8": {"torch_cuda": ["x", "y"]}, "9": {"torch_cuda": None}}},
         {8, 9}, "abc123", FD, 2),
        ("no CUDA state on an EARLIER fold, which pick_rng_state may still return",
         {**good, "states": {"8": {"torch_cuda": None}, "9": {"torch_cuda": ["x", "y"]}}},
         {8, 9}, "abc123", FD, 2),
        ("no identity field at all",
         {k: v for k, v in good.items() if k != "identity"}, {8, 9}, "abc123", FD, 2),
        ("no fold_digests field at all",
         {k: v for k, v in good.items() if k != "fold_digests"}, {8, 9}, "abc123", FD, 2),
        ("an old version-1 state file",
         {**good, "version": 1}, {8, 9}, "abc123", FD, 2),
    ]
    for name, st, done, ident, fd, ncuda in cases:
        try:
            check_rng_state(st, done, ident, fd, n_cuda_devices=ncuda)
        except RuntimeError:
            continue
        raise AssertionError(f"check_rng_state accepted a bad state: {name}")

    print(f"guard   : accepts the good state, accepts a 2-fold smoke state resuming into "
          f"17 folds, and raises on all {len(cases)} bad states   OK")
    print("          (it never warns and continues — a partially restored stream is not "
          "the run the manifest would claim)")


def test_the_four_arms_get_four_different_fingerprints():
    """The version-1 fingerprint hashed only the configs and features, and protocol_v2
    builds all four arms from identical configs — so all four collided. This checks that
    the arm-distinguishing inputs are now actually covered."""
    try:
        from run_date_gnn import _rng_state_identity, _fold_digests, TrainConfig
        from model import ModelConfig
        from temporal_graph import GraphConfig, build_temporal_edges
        from acausal_graph import build_acausal_edges
    except ImportError as exc:                        # pragma: no cover
        print(f"identity: cannot import here ({exc}); run this on the workstation")
        return

    t, m = TrainConfig(), ModelConfig()
    g = GraphConfig()
    feats = [f"f{i}" for i in range(11)]

    causal = _rng_state_identity(t, m, g, feats, None, True)
    acausal = _rng_state_identity(t, m, g, feats, build_acausal_edges, False)
    assert causal != acausal, ("P1_causal and P1_leaky share every config and differ only "
                               "in the builder — they must not share a fingerprint")

    # P1_causal vs P2_cutoff: identical configs AND identical builder. Told apart only by
    # the fold digests, which is why those are checked separately and per fold.
    p1 = _fold_digests([(-1, np.arange(0, 80), np.arange(80, 100))])
    p2 = _fold_digests([(-2, np.arange(0, 60), np.arange(60, 100))])
    assert p1 != p2 and set(p1) != set(p2)

    # same label, different rows, must still differ
    a = _fold_digests([(-1, np.arange(0, 80), np.arange(80, 100))])
    b = _fold_digests([(-1, np.arange(0, 80), np.arange(80, 99))])
    assert a[-1] != b[-1]

    # a smoke subset must reproduce the same digests for the folds it shares
    full = _fold_digests([(w, np.arange(0, 10 * w), np.arange(10 * w, 10 * w + 5))
                          for w in range(8, 25)])
    smoke = _fold_digests([(w, np.arange(0, 10 * w), np.arange(10 * w, 10 * w + 5))
                           for w in (8, 9)])
    assert all(full[w] == smoke[w] for w in smoke), (
        "the smoke run's fold digests must match the full run's, or the resume the smoke "
        "run exists to enable would be refused")
    print("identity: causal and acausal arms differ; fold sets differ; a smoke subset "
          "still matches the full run on shared folds   OK")


if __name__ == "__main__":
    test_numpy_stream_survives_a_resume()
    test_torch_stream_survives_a_resume()
    test_saved_state_file_round_trips()
    test_a_crash_between_the_two_writes_is_recoverable()
    test_the_real_guard_refuses_every_bad_state()
    test_the_four_arms_get_four_different_fingerprints()
    print("\nall RNG resume checks passed")
