#!/usr/bin/env python3
"""
edge_k — run ONE arm, start to finish, as its own process.

    python run_arm.py K5_both

One process per arm is stronger than a kernel restart between notebooks: nothing from the
previous arm can survive into this one. The notebooks remain the documented interface and
are byte-identical in what they do; this is the same sequence without a kernel.

Order, and it does not vary:

    1  canonical checksum          before anything else
    2  frozen analysis             written or verified
    3  POSITIVE CONTROLS           the hard gate; raises rather than training
    4  load the frame
    5  graph control + preflight   raises rather than training
    6  TRAIN                       heartbeat rewriting STATUS.txt throughout
    7  canonical checksum again    at the end, as the notebooks do

The CLEAN step is NOT here. Freeing the GPU stays a manual decision, exactly as in the
notebooks. This process exiting releases the card anyway, which is the point of one
process per arm.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for d in (str(HERE.parent / "src"), str(HERE.parent), str(HERE)):
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)

import warnings; warnings.filterwarnings("ignore")

import config as C
assert Path(C.__file__).parent == HERE, f"wrong config module: {C.__file__}"

import guards
import provenance as prov
import controls
import runner
import status as ST
import monitor


def main(arm: str) -> int:
    print(f"\n{'=' * 70}\n edge_k — {arm}\n{'=' * 70}")
    guards.assert_trainable(arm)                     # refuses early and loudly

    d = C.ARMS_DIR / arm
    if (d / "summary.json").exists():
        print(f"  {arm} already has a summary.json. Refusing to touch a finished arm.")
        return 0

    monitor.write("PENDING", arm, note="starting: controls and preflight")

    # ---- 1 canonical, before anything else
    print("\n[1] canonical checksum")
    guards.assert_canonical_unchanged()

    # ---- 2 the frozen analysis
    print("\n[2] frozen analysis")
    prov.write_analysis_lock()
    prov.freeze_requirements()

    # ---- 3 the hard gate
    print("\n[3] positive controls — the hard gate")
    gate = controls.run_gate(arm)
    print(f"  gate passed: {gate['passed']}")

    # ---- 4 data
    print("\n[4] loading the frame")
    B = runner.load_frame()

    # ---- 5 graph control and preflight
    print("\n[5] graph control")
    graph = controls.check_graph(arm, B)
    print("\n[5] preflight")
    pf = runner.preflight(arm, B)

    ST.open_arm(arm)
    ST.gate(arm, f"controls passed; graph {graph['edges']:,} edges "
                 f"({graph['edges_per_node']}/node); projected peak "
                 f"{pf['projected_peak_vram_gb']} GB")

    # ---- 6 train
    print(f"\n[6] training {arm}")
    t0 = time.time()
    try:
        with monitor.Heartbeat(arm) as hb:
            summary = runner.train_arm(arm, B)
    except Exception as exc:
        hours = round((time.time() - t0) / 3600, 2)
        print(f"\n  {arm} DIED after {hours}h: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        monitor.write("DIED", arm, t0, note=f"{type(exc).__name__}: {exc}")
        return 1

    # ---- 7 canonical again
    print("\n[7] canonical checksum, after")
    guards.assert_canonical_unchanged()

    monitor.write("FINISHED", arm, t0, result=summary)

    # a compact record for the reply, alongside the manifest
    peak = monitor._peak_vram_gb()
    rec = dict(summary)
    rec["peak_vram_gb"] = round(peak, 2) if peak else None
    rec["projected_peak_vram_gb"] = pf["projected_peak_vram_gb"]
    rec["graph"] = graph
    guards.assert_writable(d / "run_record.json").write_text(json.dumps(rec, indent=1))

    print(f"\n{'=' * 70}")
    print(f" {arm} COMPLETE   mean AP {summary['mean_ap']:.4f}   "
          f"{summary['n_folds']} folds   {summary['hours']}h   peak {rec['peak_vram_gb']} GB")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"arms: {', '.join(C.TRAINABLE_ARMS)}")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
