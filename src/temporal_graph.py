"""
Leakage-safe MULTI-RELATIONAL temporal graph construction for DATE-GNN v3.

Each transaction is a node. For every entity relation (uid/card, merchant, ...) a
transaction links to its K most recent PRIOR transactions sharing that entity. Edges are
prior -> current only, so message passing is causal. Each edge is tagged with its
relation id so the model can weight relations differently (edge-aware attention).

Returns edge_index (2,E), edge_dt seconds (E,), and edge_rel (E,) relation index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class GraphConfig:
    time_col: str = "unix_time"
    entity_cols: Tuple[str, ...] = ("uid", "merchant")     # multi-relational by default
    max_prior_neighbors: int = 10                           # per relation
    time_window_seconds: Optional[int] = None


def build_temporal_edges(df: pd.DataFrame, gcfg: GraphConfig):
    """Return (edge_index (2,E)[src=prior,dst=current], edge_dt (E,) sec, edge_rel (E,))."""
    assert df[gcfg.time_col].is_monotonic_increasing, "df must be time-sorted"
    n = len(df)
    times = df[gcfg.time_col].to_numpy().astype(np.float64)
    srcs, dsts, rels = [], [], []

    for r, ent in enumerate(gcfg.entity_cols):
        if ent not in df.columns:
            continue
        vals = df[ent].to_numpy(); notna = df[ent].notna().to_numpy()
        groups: Dict[object, List[int]] = {}
        for i in range(n):
            if notna[i]:
                groups.setdefault(vals[i], []).append(i)
        for _, idx in groups.items():
            if len(idx) < 2:
                continue
            idx_arr = np.asarray(idx); t_arr = times[idx_arr]
            for pos in range(1, len(idx_arr)):
                cur = idx_arr[pos]; lo = 0
                if gcfg.time_window_seconds is not None:
                    lo = int(np.searchsorted(t_arr[:pos], t_arr[pos] - gcfg.time_window_seconds, "left"))
                prior = idx_arr[lo:pos]
                if gcfg.max_prior_neighbors and len(prior) > gcfg.max_prior_neighbors:
                    prior = prior[-gcfg.max_prior_neighbors:]
                if len(prior) == 0:
                    continue
                srcs.append(prior); dsts.append(np.full(len(prior), cur)); rels.append(np.full(len(prior), r))

    if not srcs:
        return (np.zeros((2, 0), np.int64), np.zeros((0,), np.float64), np.zeros((0,), np.int64))
    src = np.concatenate(srcs); dst = np.concatenate(dsts); rel = np.concatenate(rels)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_dt = (times[dst] - times[src])
    return edge_index, edge_dt, rel.astype(np.int64)


def add_self_loops(edge_index, edge_dt, edge_rel, n):
    loops = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
    if edge_index.shape[1] == 0:
        return loops, np.zeros(n), np.full(n, -1, np.int64)
    return (np.concatenate([edge_index, loops], axis=1),
            np.concatenate([edge_dt, np.zeros(n)]),
            np.concatenate([edge_rel, np.full(n, -1, np.int64)]))


def assert_causal(edge_index, times, strict: bool = False) -> int:            # PATCH 3
    """Verify that no edge carries information backwards in time.

    Self-loops are exempt, since they connect a transaction to itself and cannot
    transport information from another event.

    Parameters
    ----------
    edge_index : (2, E) int array
        Row 0 is the source (prior) node, row 1 the destination (current) node.
    times : (N,) array
        Timestamps indexed by node id.
    strict : bool
        If True, edges joining two distinct transactions that share a timestamp
        are treated as violations.

    Returns
    -------
    int
        The number of contemporaneous non-self-loop edges, i.e. those with
        t_src == t_dst. Zero means the strict inequality t_j < t_i holds.

    Raises
    ------
    AssertionError
        If any non-self-loop edge has t_src > t_dst, or if strict=True and any
        has t_src == t_dst.
    """
    if edge_index.shape[1] == 0:
        return 0

    s, d = edge_index[0], edge_index[1]
    non_loop = s != d
    t_src, t_dst = times[s[non_loop]], times[d[non_loop]]

    n_future = int(np.count_nonzero(t_src > t_dst))                           # PATCH 1
    assert n_future == 0, (
        f"FUTURE EDGE: leakage in graph! {n_future} of {int(non_loop.sum())} "
        f"non-self-loop edges point from a later transaction to an earlier one."
    )

    n_tied = int(np.count_nonzero(t_src == t_dst))                            # PATCH 2
    if strict:
        assert n_tied == 0, (
            f"TIED EDGE: {n_tied} of {int(non_loop.sum())} non-self-loop edges join "
            f"two distinct transactions sharing a timestamp. These are contemporaneous "
            f"rather than future; pass strict=False to admit them."
        )
    return n_tied
