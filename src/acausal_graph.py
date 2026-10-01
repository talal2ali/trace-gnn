"""
ACAUSAL temporal graph construction — the honest strawman for the protocol study (08).

Mirror of temporal_graph.build_temporal_edges, with ONE deliberate difference: a
transaction links to its K temporally-NEAREST same-entity transactions in BOTH
directions (past and future), instead of its K most-recent PRIOR ones.

This is what a competent-but-not-leakage-aware practitioner would build: a symmetric
k-NN-in-time graph over each entity. Under message passing it lets FUTURE information
flow into a node's representation — exactly the graph-level leak that TRACE-GNN's causal
edges prevent. It is used ONLY for the P1-leaky arm of notebook 08 to quantify that leak.

Edge features are built by the SAME run_date_gnn.build_edge_features, so edge_dim is
identical to the causal arms (the leak is topological, not architectural). Δt here is
signed |t_dst - t_src| magnitude via the same formula; direction is not enforced.

NOTE: assert_causal MUST NOT be called on these edges — they intentionally violate it.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from temporal_graph import GraphConfig


def build_acausal_edges(df: pd.DataFrame, gcfg: GraphConfig):
    """Symmetric K-nearest-in-time same-entity edges (past AND future).

    Returns (edge_index (2,E)[src,dst], edge_dt (E,) seconds |Δt|, edge_rel (E,)),
    matching build_temporal_edges' signature so the same downstream code consumes it.

    For each transaction i in an entity group, connect it to the K other transactions
    in that group whose time is closest to t_i (either side). Edges are emitted as
    src=neighbour, dst=i (so messages flow neighbour -> i, same as the causal builder),
    but neighbours may lie in i's FUTURE. That is the point.
    """
    assert df[gcfg.time_col].is_monotonic_increasing, "df must be time-sorted"
    n = len(df)
    times = df[gcfg.time_col].to_numpy().astype(np.float64)
    K = gcfg.max_prior_neighbors
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
            idx_arr = np.asarray(idx)
            t_arr = times[idx_arr]                      # already sorted (df is time-sorted)
            for pos in range(len(idx_arr)):
                cur = idx_arr[pos]
                # K nearest in time on EITHER side, excluding self, via a two-pointer
                # expansion outward from pos over the (sorted) group timeline.
                lo, hi = pos - 1, pos + 1
                picked = []
                while len(picked) < K and (lo >= 0 or hi < len(idx_arr)):
                    take_lo = False
                    if lo >= 0 and hi < len(idx_arr):
                        take_lo = (t_arr[pos] - t_arr[lo]) <= (t_arr[hi] - t_arr[pos])
                    elif lo >= 0:
                        take_lo = True
                    if take_lo:
                        picked.append(idx_arr[lo]); lo -= 1
                    else:
                        picked.append(idx_arr[hi]); hi += 1
                if not picked:
                    continue
                picked = np.asarray(picked)
                srcs.append(picked)
                dsts.append(np.full(len(picked), cur))
                rels.append(np.full(len(picked), r))

    if not srcs:
        return (np.zeros((2, 0), np.int64), np.zeros((0,), np.float64), np.zeros((0,), np.int64))
    src = np.concatenate(srcs); dst = np.concatenate(dsts); rel = np.concatenate(rels)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_dt = np.abs(times[dst] - times[src])            # |Δt| — direction not enforced
    return edge_index, edge_dt, rel.astype(np.int64)


def count_future_edges(edge_index, times) -> dict:
    """Diagnostic: how many edges point FUTURE->current (src later than dst)?

    For causal graphs this is 0 by construction. For the acausal graph it should be
    ~half the edges — that fraction IS the leak, quantified.
    """
    if edge_index.shape[1] == 0:
        return {"n_edges": 0, "future_edges": 0, "future_frac": 0.0}
    s, d = edge_index[0], edge_index[1]
    non_loop = s != d
    future = int(np.sum(times[s[non_loop]] > times[d[non_loop]]))
    total = int(np.sum(non_loop))
    return {"n_edges": total, "future_edges": future,
            "future_frac": round(future / total, 4) if total else 0.0}
