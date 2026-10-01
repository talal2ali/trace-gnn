"""
edge_k — the standalone run status file.

Writes edge_k/results/STATUS.txt, which is meant to be read on its own by someone who is
not watching the process. It answers, without needing anything else:

    what state is the run in      RUNNING / FINISHED / DIED / ABANDONED / PENDING
    which arm, and where in the queue
    how many folds are done out of how many
    how long it has been going and roughly how much is left
    peak GPU memory so far
    what the last completed fold did

The per-fold numbers come from foldckpt.csv, which run_date_gnn_fold appends to after
every fold (src/run_date_gnn.py, the line after _save_rng_state). Nothing here reaches
into the training loop; a background thread polls that file. That keeps the training call
exactly as it was.

    tail -f edge_k/results/STATUS.txt        # or just open it, it is rewritten in place
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

import config as C
from guards import assert_writable

RUN_STATUS = C.RESULTS / "STATUS.txt"
_LOCK = threading.Lock()

STATES = ("PENDING", "RUNNING", "FINISHED", "DIED", "ABANDONED")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hm(seconds) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def _folds_done(arm: str) -> tuple[int, dict | None]:
    """(folds completed, last row) read from the arm's foldckpt.csv."""
    p = C.ARMS_DIR / arm / "foldckpt.csv"
    if not p.exists() or p.stat().st_size == 0:
        return 0, None
    try:
        import pandas as pd
        d = pd.read_csv(p)
        if len(d) == 0:
            return 0, None
        return len(d), d.iloc[-1].to_dict()
    except Exception:
        return 0, None


def _peak_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated(torch.device(C.DEVICE)) / 1e9
    except Exception:
        return None


def write(state: str, arm: str | None = None, started: float | None = None,
          note: str = "", result: dict | None = None) -> Path:
    """Rewrite STATUS.txt in place. Safe to call from a thread."""
    assert state in STATES, f"unknown state {state!r}"
    p = assert_writable(RUN_STATUS)
    p.parent.mkdir(parents=True, exist_ok=True)

    order = [a for _, a, _ in C.NOTEBOOK_ORDER]
    hours = {a: h for _, a, h in C.NOTEBOOK_ORDER}
    idx = order.index(arm) + 1 if arm in order else 0

    done, last = (_folds_done(arm) if arm else (0, None))
    total = C.EXPECT["n_folds"]
    elapsed = (time.time() - started) if started else None

    remaining = None
    if elapsed and done > 0 and state == "RUNNING":
        remaining = elapsed / done * (total - done)
    elif elapsed is not None and state == "RUNNING" and arm:
        remaining = hours.get(arm, 0) * 3600 - elapsed

    peak = _peak_vram_gb()

    L = []
    L.append("=" * 70)
    L.append(" edge_k RUN STATUS")
    L.append("=" * 70)
    L.append(f" updated      {_utc()}")
    L.append(f" state        {state}")
    if arm:
        L.append(f" arm          {arm}   ({idx} of {len(order)})")
        L.append(f" folds        {done} / {total}")
        L.append(f" elapsed      {_hm(elapsed)}")
        if state == "RUNNING":
            L.append(f" remaining    ~{_hm(remaining)}  (estimate)")
            L.append(f" nominal      ~{hours.get(arm, '?')} h for this arm")
        if peak is not None:
            L.append(f" peak VRAM    {peak:.2f} GB   (device {C.DEVICE})")
        if last:
            L.append(f" last fold    week {int(last.get('test_week', -1))}   "
                     f"AP {float(last.get('ap', float('nan'))):.4f}   "
                     f"epochs {int(last.get('epochs_used', 0))}")
    if note:
        L.append(f" note         {note}")
    if result:
        L.append("-" * 70)
        L.append(f" RESULT       mean AP {result.get('mean_ap'):.4f}   "
                 f"folds {result.get('n_folds')}   hours {result.get('hours')}")
        L.append(f"              epochs {result.get('epochs_min')}-"
                 f"{result.get('epochs_max')} (median {result.get('epochs_median')})")
        L.append(f"              problems: {result.get('problems') or 'none'}")

    L.append("-" * 70)
    L.append(" queue")
    for _, a, h in C.NOTEBOOK_ORDER:
        d = C.ARMS_DIR / a
        if (d / "summary.json").exists():
            mark, extra = "done", ""
            try:
                import json
                s = json.loads((d / "summary.json").read_text())
                extra = f"  mean AP {s['mean_ap']:.4f}  {s['hours']}h"
            except Exception:
                pass
        elif a == arm and state == "RUNNING":
            mark, extra = "RUN ", f"  {done}/{total} folds"
        elif a == arm and state == "DIED":
            mark, extra = "DIED", ""
        else:
            mark, extra = "    ", f"  ~{h} h"
        L.append(f"   [{mark}] {a:16}{extra}")
    L.append("=" * 70)

    with _LOCK:
        p.write_text("\n".join(L) + "\n")
    return p


class Heartbeat:
    """Background thread that rewrites STATUS.txt while an arm trains."""

    def __init__(self, arm: str, every: float = 30.0):
        self.arm, self.every = arm, every
        self.started = time.time()
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        write("RUNNING", self.arm, self.started, note="training started")
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                write("RUNNING", self.arm, self.started)
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        if exc_type is not None:
            import status as ST
            msg = f"{exc_type.__name__}: {exc}"
            write("DIED", self.arm, self.started, note=msg)
            try:
                ST.write(self.arm, "!" * 70)
                ST.write(self.arm, f"DIED  {msg}")
                for line in traceback.format_exception(exc_type, exc, tb)[-6:]:
                    ST.write(self.arm, "  " + line.rstrip())
                ST.write(self.arm, "edge_k never resumes. Move this directory to "
                                   f"{self.arm}_ABANDONED_<utc> and start clean.")
                ST.write(self.arm, "!" * 70)
            except Exception:
                pass
        return False
