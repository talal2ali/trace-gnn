#!/usr/bin/env python3
"""
edge_k — the positive controls, as a test. No GPU, no data, a few seconds.

    python tests/test_positive_controls.py

The graph control needs the corpus and so lives in the arm notebooks; the two that do not
run here. This file also checks the controls themselves are not vacuous — a control that
passes on a deliberately broken model would be worse than no control at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EK = HERE.parent
for d in (str(EK.parent / "src"), str(EK.parent), str(EK)):
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)

import warnings; warnings.filterwarnings("ignore")
import numpy as np

import config as C
import controls
import runner

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}      {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def test_edge_bias_control():
    r = controls.check_edge_bias_flag(verbose=False)
    check("control 1 passes", r["passed"], str(r["problems"]))
    check("the flag drops exactly 48 parameters",
          r["params_dropped"] == 48 and r["params_dropped_expected"] == 48,
          f"{r['params_on']:,} -> {r['params_off']:,}")
    check("the flag changes the forward pass",
          r["max_abs_output_diff_on_vs_off"] > 0,
          f"max |on-off| = {r['max_abs_output_diff_on_vs_off']:.6f}")
    check("use_edge_bias=True reproduces the untouched default model",
          r["default_matches_flag_on"])
    check("edge_attr is fully disconnected when the flag is off",
          r["edge_attr_disconnected_when_off"],
          "autograd finds no path from edge_attr to the output")
    check("edge_attr does influence the output when the flag is on",
          r["d_output_d_edge_attr_ON"] > 0, f"{r['d_output_d_edge_attr_ON']:.2f}")
    check("the edge_bias module is not constructed when off",
          r["edge_bias_module_off_is_None"])


def test_zero_cols_control():
    r = controls.check_zero_cols(verbose=False)
    check("control 2 passes", r["passed"], str(r["problems"]))
    for arm, d in r["arms"].items():
        check(f"{arm}: width preserved at 5",
              d["width_before"] == d["width_after"] == 5)
        check(f"{arm}: exactly {d['cols']} zeroed", d["zeroed_now"] == d["cols"])
        check(f"{arm}: every other column bit-identical",
              d["other_columns_bit_identical"])


def test_the_controls_are_not_vacuous():
    """A control that cannot fail is decoration. Feed each one something broken."""
    base = np.ones((16, 5), dtype=np.float32)
    out = runner.apply_zero_cols(base, ())
    check("apply_zero_cols with no columns changes nothing", np.array_equal(out, base))
    check("apply_zero_cols returns a copy, never a view",
          out is not base and np.array_equal(base, np.ones((16, 5), dtype=np.float32)))
    raised = False
    try:
        runner.apply_zero_cols(base, (9,))
    except IndexError:
        raised = True
    check("apply_zero_cols refuses an out-of-range column", raised)

    import torch
    from model import DateGNN, ModelConfig
    torch.manual_seed(0)
    m = ModelConfig(n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG, use_edge_bias=True,
                    edge_dim=5, **C.MODEL_KWARGS)
    a = DateGNN(11, [], m); a.eval()
    torch.manual_seed(0)
    b = DateGNN(11, [], m); b.eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 11, generator=g)
    ei = torch.stack([torch.randint(0, 64, (256,), generator=g),
                      torch.randint(0, 64, (256,), generator=g)])
    ea = torch.randn(256, 5, generator=g)
    with torch.no_grad():
        same = torch.equal(a(x, [], ei, ea), b(x, [], ei, ea))
    check("two identically-seeded models agree, so a zero diff really means no change",
          same, "this is what the control would see if the flag were not wired")


def test_runner_refuses_a_scaffolded_arm():
    import guards
    raised = False
    try:
        runner.train_arm("no_time_terms", {})
    except (guards.NotYetAuthorisedError, guards.LockedArmError):
        raised = True
    check("train_arm refuses a scaffolded arm before touching anything", raised)

    refused_locked = 0
    for name in ("main", "P3_rolling", "depth3"):
        try:
            runner.train_arm(name, {})
        except guards.LockedArmError:
            refused_locked += 1
        except Exception:
            pass
    check("train_arm refuses canonical and protocol_v2 arm names",
          refused_locked == 3, f"{refused_locked}/3")


def main():
    print("edge_k positive controls\n")
    test_edge_bias_control()
    print()
    test_zero_cols_control()
    print()
    test_the_controls_are_not_vacuous()
    print()
    test_runner_refuses_a_scaffolded_arm()
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("positive controls passed")


if __name__ == "__main__":
    main()
