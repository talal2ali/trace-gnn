"""Tests for the causality guarantee of Equation 6 (Section 3.3).

These exist because the manuscript claims the guard "is verified programmatically on
every graph that is built". That claim should be reproducible by anyone using the
released code, not merely recorded in a lab notebook.

Run either way:

    python test_causal_guard.py
    pytest -q test_causal_guard.py

Requires `temporal_graph.py` on the import path, with the patched `assert_causal`
from `temporal_graph_patched.py` (the tie-counting tests need its return value; the
future-edge tests pass against the unpatched version too).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from temporal_graph import (
    GraphConfig,
    build_temporal_edges,
    assert_causal,
    add_self_loops,
)

GCFG = GraphConfig(time_col="unix_time", entity_cols=("uid", "merchant"),
                   max_prior_neighbors=10)


def _frame() -> pd.DataFrame:
    """Twelve transactions, three cards, two merchants, time-sorted.

    Rows 6 and 7 share both a timestamp and a card, which is the contemporaneous
    case Equation 6 admits and Equation 5's original strict form did not.
    """
    return pd.DataFrame({
        "unix_time": [10, 20, 30, 40, 50, 60, 70, 70, 80, 90, 100, 110],
        "uid":       ["a", "b", "a", "c", "b", "a", "c", "c", "b", "a", "c", "b"],
        "merchant":  ["m1", "m1", "m2", "m1", "m2", "m1", "m2", "m2", "m1", "m2", "m1", "m2"],
        "amt":       [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.],
    })


def _build(df):
    ei, dt, rel = build_temporal_edges(df.reset_index(drop=True), GCFG)
    return ei, dt, rel, df["unix_time"].to_numpy().astype(np.float64)


# --------------------------------------------------------------------------- 1
def test_construction_is_causal():
    """Every edge the builder emits points from earlier to later."""
    ei, _, _, t = _build(_frame())
    assert ei.shape[1] > 0, "fixture produced no edges"
    assert_causal(ei, t)                                  # must not raise
    assert np.all(t[ei[0]] <= t[ei[1]]), "independent re-check failed"


# --------------------------------------------------------------------------- 2
def test_injected_future_edge_is_rejected():
    """The guard catches a deliberately acausal edge.

    This is the test the manuscript's claim rests on. Reverse one edge so it runs
    from a later transaction to an earlier one, and confirm the guard fails.
    """
    ei, _, _, t = _build(_frame())

    # find an edge with a genuine time gap and reverse it
    gap = np.flatnonzero(t[ei[1]] > t[ei[0]])
    assert len(gap) > 0
    bad = ei.copy()
    k = int(gap[0])
    bad[0, k], bad[1, k] = ei[1, k], ei[0, k]

    try:
        assert_causal(bad, t)
    except AssertionError as exc:
        assert "FUTURE EDGE" in str(exc)
        return
    raise AssertionError("guard did NOT reject an injected future edge")


# --------------------------------------------------------------------------- 3
def test_appended_future_edge_is_rejected():
    """Same, for an edge appended rather than reversed in place."""
    ei, _, _, t = _build(_frame())
    late, early = int(np.argmax(t)), int(np.argmin(t))
    bad = np.concatenate([ei, np.array([[late], [early]], dtype=ei.dtype)], axis=1)

    try:
        assert_causal(bad, t)
    except AssertionError as exc:
        assert "FUTURE EDGE" in str(exc)
        return
    raise AssertionError("guard did NOT reject an appended future edge")


# --------------------------------------------------------------------------- 4
def test_self_loops_are_exempt():
    """Self-loops carry zero elapsed time and must not trip the guard."""
    ei, dt, rel, t = _build(_frame())
    ei2, _, _ = add_self_loops(ei, dt, rel, len(t))
    assert ei2.shape[1] == ei.shape[1] + len(t)
    assert_causal(ei2, t)                                 # must not raise


# --------------------------------------------------------------------------- 5
def test_contemporaneous_edges_are_counted_not_hidden():
    """Equation 6 admits t_src == t_dst; the guard should report how many.

    Rows 6 and 7 share a timestamp and a card, so exactly one tied edge is
    expected from the cardholder relation (and one more from the merchant
    relation, since they also share a merchant).
    """
    ei, _, _, t = _build(_frame())
    n_tied = assert_causal(ei, t)
    assert n_tied == 2, f"expected 2 contemporaneous edges, got {n_tied}"


# --------------------------------------------------------------------------- 6
def test_strict_mode_rejects_contemporaneous_edges():
    """strict=True enforces the strict inequality of the original Equation 5."""
    ei, _, _, t = _build(_frame())
    try:
        assert_causal(ei, t, strict=True)
    except AssertionError as exc:
        assert "TIED EDGE" in str(exc)
        return
    raise AssertionError("strict=True did NOT reject contemporaneous edges")


# --------------------------------------------------------------------------- 7
def test_strictly_increasing_times_have_no_ties():
    """With distinct timestamps the strict inequality holds, so strict=True passes."""
    df = _frame()
    df["unix_time"] = np.arange(10, 10 + 10 * len(df), 10)
    ei, _, _, t = _build(df)
    assert assert_causal(ei, t) == 0
    assert_causal(ei, t, strict=True)                     # must not raise


# --------------------------------------------------------------------------- 8
def test_empty_graph_is_accepted():
    """A frame with no repeated entity yields no edges and must not raise."""
    df = pd.DataFrame({
        "unix_time": [10, 20, 30],
        "uid": ["a", "b", "c"],
        "merchant": ["m1", "m2", "m3"],
        "amt": [1., 2., 3.],
    })
    ei, _, _, t = _build(df)
    assert ei.shape[1] == 0
    assert assert_causal(ei, t) == 0


# --------------------------------------------------------------------------- 9
def test_neighbour_cap_is_per_relation():
    """K applies within each relation, so in-degree may reach 2K.

    Stated in the manuscript at Equation 5 and in methodology record 11.1. Included
    here because the edge-relation ablation of Section 5.3 depends on it.
    """
    n = 30
    df = pd.DataFrame({
        "unix_time": np.arange(1, n + 1) * 10,
        "uid": ["a"] * n,
        "merchant": ["m1"] * n,
        "amt": np.arange(1., n + 1.),
    })
    gcfg = GraphConfig(time_col="unix_time", entity_cols=("uid", "merchant"),
                       max_prior_neighbors=10)
    ei, _, _, t = build_temporal_edges(df, gcfg) + (df["unix_time"].to_numpy(),)
    in_deg = np.bincount(ei[1], minlength=n)
    assert in_deg.max() == 20, f"expected max in-degree 20 (10 per relation), got {in_deg.max()}"
    assert_causal(ei, t)


# --------------------------------------------------------------------------- 10
def test_dummy_relation_contributes_no_edges():
    """The placeholder entity of Section 5.2 keeps edge_dim at 5 but adds no edges.

    `build_temporal_edges` skips any entity column absent from the frame, so a
    cardholder-only arm is topological rather than architectural.
    """
    df = _frame()
    both = build_temporal_edges(df, GraphConfig(
        time_col="unix_time", entity_cols=("uid", "merchant"), max_prior_neighbors=10))[0]
    uid_only = build_temporal_edges(df, GraphConfig(
        time_col="unix_time", entity_cols=("uid", "__none__"), max_prior_neighbors=10))[0]

    assert uid_only.shape[1] < both.shape[1]
    # every uid-only edge shares a cardholder
    u = df["uid"].to_numpy()
    assert np.all(u[uid_only[0]] == u[uid_only[1]])


# --------------------------------------------------------------------------- run
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
