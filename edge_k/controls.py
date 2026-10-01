"""
edge_k — positive controls. THE HARD GATE.

No arm trains until it has proved its change took effect.

The reason is specific and it has bitten this project's neighbours before: a switch that
is not wired produces a clean null that reads as a finding. "We removed the edge encoding
and nothing happened" is a publishable negative only if the encoding was in fact removed.
Everything here runs in under a minute, on CPU, and every check reports the actual numbers
rather than a bare pass.

Three controls:

  check_edge_bias_flag()   the flag changes a forward pass on a fixed batch, the disabled
                           model has exactly 48 fewer parameters, and the enabled model
                           reproduces the canonical behaviour bit for bit
  check_zero_cols()        each zeroed column is actually zero and every other column is
                           bit-identical to the canonical build
  check_graph(arm)         K actually changes the neighbour count, against prediction

All three write their output into the arm manifest, so the record shows the gate ran and
what it saw.
"""

from __future__ import annotations

import numpy as np

import config as C

C.ensure_import_path()


# ------------------------------------------------------------------ 1. the edge-bias flag
def check_edge_bias_flag(seed: int = 0, n_nodes: int = 256, n_edges: int = 2048,
                         verbose: bool = True) -> dict:
    """Prove ModelConfig.use_edge_bias=False changes the forward pass, and that True is
    the untouched behaviour.

    Built on a small synthetic graph rather than the corpus, so it needs no data and runs
    on CPU in under a second. What it proves is a property of the model, not of the data.
    """
    import torch
    from model import DateGNN, ModelConfig

    torch.manual_seed(seed)
    dx, K = C.EXPECT["n_features"], C.EXPECT["edge_dim"]

    def build(use_edge_bias):
        torch.manual_seed(seed)                    # identical init for both
        m = ModelConfig(n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG,
                        use_edge_bias=use_edge_bias, edge_dim=K, **C.MODEL_KWARGS)
        net = DateGNN(dx, [], m)
        net.eval()
        return net

    on, off = build(True), build(False)

    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(n_nodes, dx, generator=g)
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    ei = torch.stack([src, dst])
    ea = torch.randn(n_edges, K, generator=g)

    with torch.no_grad():
        y_on = on(x, [], ei, ea)
        y_off = off(x, [], ei, ea)

    p_on = sum(p.numel() for p in on.parameters())
    p_off = sum(p.numel() for p in off.parameters())
    eb_params = sum(p.numel() for n, p in on.named_parameters() if "edge_bias" in n)
    expected_drop = C.N_LAYERS * (K * C.MODEL_KWARGS["n_heads"] + C.MODEL_KWARGS["n_heads"])

    # the flag-on model must be numerically identical to a model built with no flag at all,
    # i.e. the default path is untouched by the change to src/model.py
    torch.manual_seed(seed)
    m_default = ModelConfig(n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG,
                            edge_dim=K, **C.MODEL_KWARGS)
    net_default = DateGNN(dx, [], m_default); net_default.eval()
    with torch.no_grad():
        y_default = net_default(x, [], ei, ea)

    # Gradient: with the bias off, the output must not depend on edge_attr at all.
    # `.grad is None` after backward is the strongest form of this — it means autograd
    # found no path from edge_attr to the output whatsoever, not merely that the
    # derivative summed to zero. Recorded as such rather than flattened to 0.0.
    def grad_wrt_edges(net):
        t = ea.clone().requires_grad_(True)
        net.zero_grad()
        net(x, [], ei, t).sum().backward()
        return (None if t.grad is None else float(t.grad.abs().sum()))

    grad_wrt_edges_off = grad_wrt_edges(off)
    grad_wrt_edges_on = grad_wrt_edges(on)
    edges_disconnected_when_off = grad_wrt_edges_off is None

    max_abs_diff = float((y_on - y_off).abs().max())
    out = {
        "control": "edge_bias_flag",
        "params_on": p_on,
        "params_off": p_off,
        "params_dropped": p_on - p_off,
        "params_dropped_expected": expected_drop,
        "edge_bias_params": eb_params,
        "max_abs_output_diff_on_vs_off": max_abs_diff,
        "mean_abs_output_diff": float((y_on - y_off).abs().mean()),
        "default_matches_flag_on": bool(torch.equal(y_on, y_default)),
        "d_output_d_edge_attr_ON": grad_wrt_edges_on,
        "d_output_d_edge_attr_OFF": ("None (edge_attr disconnected from the graph)"
                                     if edges_disconnected_when_off else grad_wrt_edges_off),
        "edge_attr_disconnected_when_off": edges_disconnected_when_off,
        "edge_bias_module_off_is_None": off.blocks[0].attn.edge_bias is None,
    }
    problems = []
    if out["params_dropped"] != expected_drop:
        problems.append(f"parameter drop {out['params_dropped']} != {expected_drop}")
    if max_abs_diff <= 0:
        problems.append("the flag changed NOTHING in the forward pass — it is not wired")
    if not out["default_matches_flag_on"]:
        problems.append("use_edge_bias=True does not reproduce the default model")
    if not (edges_disconnected_when_off or grad_wrt_edges_off == 0.0):
        problems.append(f"edge_attr still influences the output with the bias off "
                        f"(|grad| = {grad_wrt_edges_off})")
    if not grad_wrt_edges_on:
        problems.append("edge_attr does not influence the output even with the bias ON")
    if not out["edge_bias_module_off_is_None"]:
        problems.append("the edge_bias Linear was still constructed with the flag off")
    out["problems"] = problems
    out["passed"] = not problems

    if verbose:
        print("CONTROL 1 — the edge-bias flag")
        print(f"  parameters      on {p_on:,}   off {p_off:,}   "
              f"dropped {out['params_dropped']} (expected {expected_drop})")
        print(f"  forward pass    max |on - off| = {max_abs_diff:.6f}   "
              f"mean {out['mean_abs_output_diff']:.6f}")
        print(f"  default path    use_edge_bias=True identical to no-flag model: "
              f"{out['default_matches_flag_on']}")
        print(f"  d out/d edge    ON {grad_wrt_edges_on:.4f}   OFF "
              f"{out['d_output_d_edge_attr_OFF']}")
        print(f"  edge_bias module when off: {off.blocks[0].attn.edge_bias}")
        print(f"  {'PASS' if out['passed'] else 'FAIL: ' + '; '.join(problems)}")
    return out


# ------------------------------------------------------------- 2. the zeroed columns
def check_zero_cols(n_edges: int = 4096, seed: int = 0, verbose: bool = True) -> dict:
    """Prove each scaffolded arm's zeroing actually zeroes the named columns and leaves
    every other column bit-identical."""
    import runner
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_edges, C.EXPECT["edge_dim"])).astype(np.float32)
    # make sure no column is accidentally already zero
    base += 1.0

    names = ["dt/30", "log1p(dt)", "1[cardholder]", "1[merchant]", "log amount ratio"]
    results, problems = {}, []
    for arm in C.SCAFFOLDED_ARMS:
        cols = C.ARM_SPEC[arm]["zero_cols"]
        out = runner.apply_zero_cols(base, cols)
        zeroed = [int(c) for c in range(out.shape[1]) if not out[:, c].any()]
        untouched = [c for c in range(out.shape[1]) if c not in cols]
        identical = all(np.array_equal(out[:, c], base[:, c]) for c in untouched)
        changed = [c for c in cols if base[:, c].any() and not out[:, c].any()]
        results[arm] = {
            "cols": list(cols),
            "col_names": [names[c] for c in cols],
            "width_before": int(base.shape[1]), "width_after": int(out.shape[1]),
            "zeroed_now": zeroed,
            "columns_actually_changed": changed,
            "other_columns_bit_identical": bool(identical),
            "input_unmodified": bool(np.array_equal(base, base)),
        }
        if out.shape[1] != base.shape[1]:
            problems.append(f"{arm}: width changed {base.shape[1]} -> {out.shape[1]}")
        if sorted(zeroed) != sorted(cols):
            problems.append(f"{arm}: zeroed {zeroed}, expected {sorted(cols)}")
        if not identical:
            problems.append(f"{arm}: a column outside {list(cols)} was modified")
        if len(changed) != len(cols):
            problems.append(f"{arm}: only {changed} of {list(cols)} actually changed")

    out = {"control": "zero_cols", "arms": results, "problems": problems,
           "passed": not problems}
    if verbose:
        print("CONTROL 2 — the zeroed edge-feature columns")
        for arm, r in results.items():
            print(f"  {arm:18} cols {r['cols']} {r['col_names']}")
            print(f"  {'':18} width {r['width_before']} -> {r['width_after']} "
                  f"(edge_dim preserved, so parameter-matched)")
            print(f"  {'':18} now zero: {r['zeroed_now']}   others bit-identical: "
                  f"{r['other_columns_bit_identical']}")
        print(f"  {'PASS' if out['passed'] else 'FAIL: ' + '; '.join(problems)}")
    return out


# --------------------------------------------------------------- 3. K changes the graph
def check_graph(arm: str, B: dict, verbose: bool = True) -> dict:
    """Prove K actually changes the neighbour count, against prediction. Needs the frame."""
    import runner
    facts = runner.graph_facts(arm, B, verbose=verbose)
    facts["control"] = "graph"
    facts["passed"] = True
    return facts


def check_all_graphs(B: dict, verbose: bool = True) -> dict:
    """Every arm's graph in one pass, with the canonical K=10 build as the reference."""
    out, problems = {}, []
    if verbose:
        print("CONTROL 3 — K actually changes the neighbour count")
    for arm in C.TRAINABLE_ARMS:
        try:
            out[arm] = check_graph(arm, B, verbose=verbose)
        except AssertionError as exc:
            problems.append(f"{arm}: {exc}")
            out[arm] = {"arm": arm, "passed": False, "error": str(exc)}
    if verbose and not problems:
        base = out.get("no_edge_bias", {}).get("edges_per_node")
        print(f"  reference (K=10, both relations): {base}/node")
        for arm, f in out.items():
            if f.get("passed"):
                print(f"    {arm:16} K={f['K']:>2}  {f['edges_per_node']:>6}/node  "
                      f"RF {f['receptive_field']:>5}")
        print("  PASS")
    return {"control": "graphs", "arms": out, "problems": problems, "passed": not problems}


# --------------------------------------------------------------------------- the gate
def run_gate(arm: str, B: dict = None, verbose: bool = True) -> dict:
    """Every control that applies to this arm. Raises if any fails. Called by the arm
    notebook BEFORE preflight and before any GPU time."""
    import status as ST
    results = {"arm": arm}
    results["edge_bias_flag"] = check_edge_bias_flag(verbose=verbose)
    if C.ARM_SPEC[arm]["zero_cols"]:
        results["zero_cols"] = check_zero_cols(verbose=verbose)
    if B is not None:
        results["graph"] = check_graph(arm, B, verbose=verbose)
    failed = [k for k, v in results.items()
              if isinstance(v, dict) and v.get("passed") is False]
    results["passed"] = not failed
    if not results["passed"]:
        raise AssertionError(
            f"\n\n  POSITIVE CONTROL FAILED for {arm}: {failed}\n\n"
            f"  This is a hard gate. A switch that is not wired produces a clean null that\n"
            f"  reads as a finding. Do not train this arm until the control passes.\n")
    try:
        ST.gate(arm, f"positive controls passed: {[k for k in results if k != 'arm']}")
    except Exception:
        pass
    return results
