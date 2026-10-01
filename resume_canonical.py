#!/usr/bin/env python3
"""
Resume an interrupted canonical re-run, finishing only the folds that are missing.

Why this exists
---------------
`rerun_canonical.py` deliberately refuses to write into a folder that already holds
append-mode files, because a stale `foldckpt.csv` would make the runner silently skip
folds. That guard is right for a fresh run and wrong for a crash.

`run_date_gnn_fold` already has a correct resume path: it reads `foldckpt.csv`, prints
which weeks are done, skips them, and appends only the new ones to `scores.csv`,
`val_curves.csv` and `foldckpt.csv`. This script reaches that path safely by

  1. reporting exactly which weeks are missing, per arm, and refusing to start if the
     three append-mode files disagree with each other about how far the run got, which
     is the one situation where resuming would corrupt the output,
  2. neutralising only the stale-file guard, never deleting anything,
  3. re-running every assertion `one_arm` makes once the arm is complete,
  4. rebuilding `canonical_summary.csv`, `derived/` and `COMPARISON.md` from all four
     arms so the folder ends up identical to an uninterrupted run.

The training configuration is untouched. This resumes; it does not reconfigure.

Usage
-----
    # --prepared is optional. Left out, the folder holding sparkov_clean_v1.csv is
    # found by searching upward from the repo, exactly as notebook 16 does.
    python resume_canonical.py \
        --out results/rerun_20260826_114717_m5 \
        --arms depth3 --device cuda:0

    # see what is missing and stop
    python resume_canonical.py --out results/rerun_... --check-only

Memory
------
Defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is a pure allocator
change with no effect on any number produced. It reclaims the reserved-but-unallocated
fragmentation that a long multi-arm run accumulates. Override by setting the variable
yourself before calling.
"""
from __future__ import annotations

import os

# must be set before torch initialises CUDA. rerun_canonical imports torch lazily,
# inside functions, so importing it below does not defeat this.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import atexit
import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rerun_canonical as rc  # noqa: E402


ARM_ORDER = ["main", "uid_only", "maxagg", "depth3"]

# every file prepare_frame() needs. Missing any of them must abort with a readable
# message, not a traceback from inside data_io.
PREPARED_FILES = ("sparkov_clean_v1.csv", "column_manifest.json")


def find_prepared(explicit: Path | None) -> Path:
    """Locate the prepared/ folder. It sits OUTSIDE the repo, two levels up, so the
    default is a search rather than a hard-coded relative path."""
    if explicit is not None:
        p = Path(explicit).resolve()
        rc.check(p.exists(), f"--prepared {p} does not exist. Note that prepared/ is TWO "
                             f"levels above the repo, so from here it is ../../prepared")
        missing = [f for f in PREPARED_FILES if not (p / f).exists()]
        rc.check(not missing, f"{p} is missing {missing}. That is not the prepared folder. "
                              f"From the repo it is normally ../../prepared")
        return p

    for base in [HERE, *HERE.parents]:
        cand = base / "prepared"
        if all((cand / f).exists() for f in PREPARED_FILES):
            return cand.resolve()
    rc.die(f"no prepared/ folder containing {list(PREPARED_FILES)} found above {HERE}. "
           f"Pass --prepared explicitly")


def acquire_lock(out: Path) -> None:
    """Refuse to start if another resume is already working on this folder.

    Two processes here is worse than either one failing: both read foldckpt.csv, both
    see the same folds done, both train the same missing fold, and both APPEND their
    results. The output ends up with duplicated folds and fails its own assertions
    after the GPU time has already been spent.
    """
    lock = out / ".resume.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        try:
            holder = lock.read_text().strip()
        except Exception:
            holder = "unknown"
        pid = holder.split()[0] if holder else ""
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        if alive:
            rc.die(f"another resume is already running on {out}\n"
                   f"         lock held by: {holder}\n"
                   f"         Wait for it, or stop it with:  kill {pid}")
        rc.die(f"a stale lock file exists at {lock}\n"
               f"         it names process {pid}, which is no longer running, so a "
               f"previous resume was killed or crashed.\n"
               f"         Re-run with --check-only FIRST to confirm the fold files are "
               f"still consistent, then delete the lock:\n"
               f"         rm {lock}")
    os.write(fd, f"{os.getpid()} started {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n".encode())
    os.close(fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))


def arm_state(armdir: Path, all_weeks: list[int]) -> dict:
    """What this arm has on disk, and whether its three append files agree."""
    st = {"dir": armdir, "complete": False, "missing": list(all_weeks), "consistent": True,
          "detail": {}}
    ck = armdir / "foldckpt.csv"
    if not ck.exists():
        st["detail"]["foldckpt.csv"] = "absent, arm has not started"
        return st

    done = sorted(int(w) for w in pd.read_csv(ck)["test_week"].unique())
    st["detail"]["foldckpt.csv"] = f"{len(done)} folds {done}"

    vc = armdir / "val_curves.csv"
    if vc.exists():
        vw = sorted(int(w) for w in pd.read_csv(vc)["test_week"].unique())
        st["detail"]["val_curves.csv"] = f"{len(vw)} folds"
        if vw != done:
            st["consistent"] = False
            st["detail"]["val_curves.csv"] += f"  MISMATCH vs foldckpt: {sorted(set(vw) ^ set(done))}"

    sc = armdir / "scores.csv"
    if sc.exists():
        sw = sorted(int(w) for w in pd.read_csv(sc)["fold"].unique())
        st["detail"]["scores.csv"] = f"{len(sw)} folds"
        if sw != done:
            st["consistent"] = False
            st["detail"]["scores.csv"] += f"  MISMATCH vs foldckpt: {sorted(set(sw) ^ set(done))}"

    st["missing"] = [w for w in all_weeks if w not in done]
    st["complete"] = not st["missing"]
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, default=None,
                    help="folder holding sparkov_clean_v1.csv and column_manifest.json. "
                         "Found by searching upward from the repo if omitted")
    ap.add_argument("--out", type=Path, required=True,
                    help="the EXISTING rerun_* folder to finish")
    ap.add_argument("--arms", default=",".join(ARM_ORDER),
                    help="arms to consider. Complete arms are left alone")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--search-root", type=Path, default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="report what is missing and exit without training")
    ap.add_argument("--skip-derive", action="store_true",
                    help="do not rebuild derived/ and COMPARISON.md")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    rc.check(out.exists(), f"--out {out} does not exist. This script finishes an existing run")
    rc.check((out / "canonical").exists(), f"{out} holds no canonical/ folder")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        rc.check(a in rc.ARMS, f"unknown arm {a!r}. Choose from {list(rc.ARMS)}")
    arms = [a for a in ARM_ORDER if a in arms]

    search_root = (Path(args.search_root).resolve() if args.search_root
                   else (HERE / "results"))
    rc.check(search_root.exists(), f"search root {search_root} does not exist")

    prepared = find_prepared(args.prepared)

    # --check-only writes nothing, so it never takes the lock and is always safe to run.
    if not args.check_only:
        acquire_lock(out)

    print("TRACE-GNN canonical re-run, RESUME\n" + "=" * 74)
    print(f"  folder        : {out}")
    print(f"  prepared      : {prepared}")
    print(f"  arms          : {', '.join(arms)}")
    print(f"  alloc conf    : {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    # rebuild the frame, folds and graph exactly as the original run did. This is the
    # only way to know the canonical week list, and it re-runs every dry-run assertion.
    B = rc.build(SimpleNamespace(prepared=prepared, device=args.device))
    B["search_root"] = search_root
    all_weeks = sorted(int(w) for w, _, _ in B["folds"])
    print(f"\n  canonical folds: {len(all_weeks)} weeks {all_weeks}")

    states = {a: arm_state(out / "canonical" / a, all_weeks) for a in arms}
    print("\n  state on disk")
    bad = []
    for a in arms:
        st = states[a]
        flag = "COMPLETE" if st["complete"] else f"missing {st['missing']}"
        print(f"    {a:9} {flag}")
        for k, v in st["detail"].items():
            print(f"                {k:16} {v}")
        if not st["consistent"]:
            bad.append(a)

    if bad:
        rc.die(f"append-mode files disagree for {bad}. foldckpt.csv is the resume "
               f"authority, so scores.csv or val_curves.csv holding a different fold set "
               f"means a fold was half-written. Resuming would duplicate or lose rows. "
               f"Re-run those arms from scratch into a new folder instead")

    todo = [a for a in arms if not states[a]["complete"]]
    if args.check_only:
        print(f"\n  check only. Would resume: {todo or 'nothing, all arms complete'}\n")
        return
    if not todo:
        print("\n  every arm is already complete. Nothing to train.")
    else:
        est = sum(len(states[a]["missing"]) for a in todo)
        print(f"\n  resuming {len(todo)} arm(s), {est} fold(s) of training")

        # The ONLY guard we lift. one_arm() dies on, or with --force deletes, the
        # append-mode files. We want neither: we want run_date_gnn_fold to find
        # foldckpt.csv and resume from it. Nothing is deleted.
        rc.APPEND_FILES = ()
        for a in todo:
            _, s = rc.one_arm(a, out / "canonical", B, force=False)
            print(f"    note: 'hours' for {a} covers only the resumed folds")

    # ---- rebuild the summary across all four arms, from what is on disk ----------
    summaries = []
    for a in ARM_ORDER:
        sj = out / "canonical" / a / "summary.json"
        if sj.exists():
            summaries.append(json.load(open(sj)))
    if not summaries:
        rc.die("no summary.json found for any arm")
    pd.DataFrame(summaries).to_csv(out / "canonical_summary.csv", index=False)
    print(f"\n  wrote canonical_summary.csv over {len(summaries)} arm(s)")

    arms_done = [s["arm"] for s in summaries]
    if args.skip_derive:
        print("  derive skipped by request")
    else:
        pf, xgb_pf = rc.derive(out, arms_done, B)
        rc.comparison_report(out, summaries, pf, xgb_pf, None)

    rs = out / "repeat_summary.csv"
    print("\n" + "=" * 74)
    print(f"  Folder finished: {out}")
    if not rs.exists():
        print("  NOTE  no repeats were run, so run-to-run variation is still NOT estimated.")
        print("        Section 5.4 has no noise baseline until you run them.")
    print(f"  Read {out / 'COMPARISON.md'} before touching the manuscript.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
