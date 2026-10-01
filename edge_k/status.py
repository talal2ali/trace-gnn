"""
edge_k — STATUS.txt, so progress is readable from outside the process.

protocol_v2 wrote a STATUS.txt by hand for its long arm. This makes it a function, because
the one arm here that matters for this (K20_both, up to 40 hours) is exactly the case where
you want `tail -f` rather than a notebook you dare not touch.

One line per event, append-only, flushed immediately. Nothing here is read back by the
code; it exists for a human with a terminal.

    tail -f edge_k/results/arms/K20_both/STATUS.txt
"""

from __future__ import annotations

import time
from pathlib import Path

import config as C
from guards import assert_writable


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def path_for(arm: str) -> Path:
    return assert_writable(C.ARMS_DIR / arm / "STATUS.txt")


def write(arm: str, line: str, echo: bool = False) -> None:
    """Append one timestamped line. Creates the file and its directory if absent."""
    p = path_for(arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"{_stamp()}  {line}\n")
        f.flush()
    if echo:
        print(f"  [status] {line}")


def open_arm(arm: str, note: str = "") -> None:
    spec = C.ARM_SPEC[arm]
    write(arm, "=" * 78)
    write(arm, f"ARM {arm}   package edge_k   RF {C.receptive_field(arm)}")
    write(arm, f"  the one change : {'; '.join(spec['changes'])}")
    write(arm, f"  what           : {spec['what']}")
    write(arm, f"  answers        : {spec['answers']}")
    write(arm, f"  device         : {C.DEVICE}   epochs {C.TRAIN_KWARGS['epochs']}   "
               f"seed {C.TRAIN_KWARGS['seed']}")
    if note:
        write(arm, f"  note           : {note}")
    write(arm, "=" * 78)


def fold_done(arm: str, fold, epochs: int, ap: float, hours: float,
              peak_vram_gb: float = None) -> None:
    """One line per completed fold. peak_vram_gb is the running maximum on the device,
    which is the trend that matters for K20_both."""
    v = f"  peakVRAM {peak_vram_gb:5.1f}G" if peak_vram_gb is not None else ""
    write(arm, f"fold {str(fold):>4}  epochs {epochs:>4}  AP {ap:.4f}  "
               f"{hours:6.2f}h{v}")


def note(arm: str, line: str) -> None:
    write(arm, f"note  {line}")


def gate(arm: str, line: str) -> None:
    """A control or preflight gate, pass or fail."""
    write(arm, f"gate  {line}")


def abandoned(arm: str, why: str) -> None:
    write(arm, "!" * 78)
    write(arm, f"ABANDONED: {why}")
    write(arm, "This arm will be restarted clean, not resumed. The directory should be "
               "moved to <arm>_ABANDONED_<utc> and a fresh one started.")
    write(arm, "!" * 78)


def close_arm(arm: str, summary: dict) -> None:
    write(arm, "-" * 78)
    write(arm, f"COMPLETE  mean AP {summary.get('mean_ap')}  "
               f"folds {summary.get('n_folds')}  hours {summary.get('hours')}")
    write(arm, f"  problems: {summary.get('problems') or 'none'}")
    write(arm, "-" * 78)


def tail(arm: str, n: int = 20) -> str:
    p = path_for(arm)
    if not p.exists():
        return f"(no STATUS.txt for {arm} yet)"
    return "".join(p.read_text().splitlines(keepends=True)[-n:])
