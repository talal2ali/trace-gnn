"""
edge_k — the locks.

ONE RULE ABOVE ALL OTHERS

    Nothing in edge_k trains, overwrites, or writes to any canonical arm, any
    protocol_v2 arm, or anything at all outside edge_k/results/.

    Only the four arms in config.TRAINABLE_ARMS may ever be trained. The three in
    SCAFFOLDED_ARMS are built and tested but refuse to run until the gate in the work
    plan opens.

Four independent things enforce that, the same four protocol_v2 uses:

  1. NO SPECIFICATION FOR A LOCKED ARM.  config.ARM_SPEC contains seven entries and none
     of them is a canonical or protocol_v2 arm name; an assertion in config.py checks
     that. A trainer has nothing to consume.

  2. ONE GUARDED CALL SITE.  runner.train_arm is the only place in edge_k that calls
     run_date_gnn_fold, and its first statement is assert_trainable(arm). The notebooks
     never call the training function directly.

  3. WRITES ARE FENCED.  assert_writable refuses every path outside edge_k/results/.
     The whole of results/ and the whole of protocol_v2/results/ are unreachable from
     this code even by an explicit path.

  4. THE CANONICAL DIRECTORIES ARE CHECKSUMMED.  lock_canonical() records a SHA-256 per
     file across canonical/main, canonical/uid_only and the run manifest.
     assert_canonical_unchanged() re-checks it. Every notebook calls it at the start AND
     at the end. If a byte moves, the notebook stops.

Why two canonical directories rather than one: edge_k compares against canonical/main
(arms A, B, D) and against canonical/uid_only (arm C), so both are inputs and both must
be provably untouched.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import config as C


class LockedArmError(RuntimeError):
    """Raised on any attempt to train, or write to, something edge_k must not touch."""


class NotYetAuthorisedError(RuntimeError):
    """Raised when a scaffolded arm is asked to run before its gate has opened."""


class CanonicalTouchedError(RuntimeError):
    """Raised when a canonical directory no longer matches its recorded checksum."""


# --------------------------------------------------------------------------- arm lock
def assert_trainable(arm: str) -> str:
    """Gate every training call. Returns the arm name so it can be used inline."""
    if arm in C.LOCKED_NAMES:
        raise LockedArmError(
            f"\n\n"
            f"  REFUSED: {arm!r} is a canonical or protocol_v2 arm. edge_k never trains,\n"
            f"  overwrites or writes to any of them. They are read-only inputs.\n\n"
            f"  Locked names : {', '.join(C.LOCKED_NAMES)}\n"
            f"  Trainable    : {', '.join(C.TRAINABLE_ARMS)}\n")
    if arm in C.SCAFFOLDED_ARMS:
        raise NotYetAuthorisedError(
            f"\n\n"
            f"  REFUSED: {arm!r} is scaffolded but NOT authorised to run.\n\n"
            f"  The three feature-drop arms are contingent. Per the work plan they run\n"
            f"  only if the no_edge_bias arm's effect against the four-seed mean is at\n"
            f"  least 0.023 in magnitude. Below that, the whole edge encoding is worth\n"
            f"  less than the resolution of a single run and decomposing it into three\n"
            f"  parts measured with the same instrument cannot produce an interpretable\n"
            f"  answer.\n\n"
            f"  To authorise: set ARM_SPEC[{arm!r}]['runnable'] = True in config.py, and\n"
            f"  record the measured no_edge_bias effect in results/gate_decision.json\n"
            f"  BEFORE doing so.\n")
    if arm not in C.TRAINABLE_ARMS:
        raise LockedArmError(
            f"{arm!r} is not an edge_k arm. Trainable: {', '.join(C.TRAINABLE_ARMS)}.")
    if not C.ARM_SPEC[arm]["runnable"]:
        raise NotYetAuthorisedError(f"{arm!r} is marked runnable=False in config.ARM_SPEC.")
    return arm


# --------------------------------------------------------------------------- write fence
def assert_writable(path) -> Path:
    """Every output path in edge_k goes through here. Anything outside edge_k/results/ is
    refused, which puts the entire existing results/ tree and the whole of
    protocol_v2/results/ out of reach."""
    p = Path(path).resolve()
    for root in C.WRITE_ROOTS:
        try:
            p.relative_to(Path(root).resolve())
            return p
        except ValueError:
            continue
    raise LockedArmError(
        f"\n\n  REFUSED: edge_k may not write to\n    {p}\n"
        f"  Permitted root:\n    {Path(C.RESULTS).resolve()}\n"
        f"  Every existing result folder — canonical and protocol_v2 alike — is read-only\n"
        f"  reference material.\n")


# --------------------------------------------------------------------------- checksums
def sha256_file(path, _blk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_blk), b""):
            h.update(b)
    return h.hexdigest()


def checksum_tree(root) -> dict:
    """SHA-256 of every file under root, plus a combined digest over the sorted
    (relative path, hash) pairs. Order-stable and path-sensitive, so a renamed or added
    file changes the combined digest just as an edited one does."""
    root = Path(root)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(root))] = {"sha256": sha256_file(p),
                                               "bytes": p.stat().st_size}
    combined = hashlib.sha256(
        "\n".join(f"{k} {v['sha256']}" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    return {"root": str(root), "n_files": len(files), "combined_sha256": combined,
            "files": files}


# The canonical inputs edge_k reads. Both are compared against, so both are locked.
def _canonical_targets() -> dict:
    return {"canonical/main": C.CANONICAL_MAIN_DIR,
            "canonical/uid_only": C.CANONICAL_UID_DIR}


def lock_canonical(force: bool = False) -> dict:
    """Record the canonical directories' checksums. Written once, then only read.
    Run this before any GPU time is spent."""
    lockfile = assert_writable(C.CANONICAL_LOCKFILE)
    if lockfile.exists() and not force:
        return json.loads(lockfile.read_text())
    trees = {}
    for name, d in _canonical_targets().items():
        if not Path(d).is_dir():
            raise FileNotFoundError(f"canonical directory not found: {d}")
        trees[name] = checksum_tree(d)
    rec = {
        "trees": trees,
        "status": "LOCKED - read-only inputs; edge_k never trains or writes to these",
        "role": "canonical/main supplies the baseline for K5_both, no_edge_bias and "
                "K20_both; canonical/uid_only supplies the baseline for uid_K20",
        "manifest": str(C.CANONICAL_MANIFEST),
        "manifest_sha256": sha256_file(C.CANONICAL_MANIFEST),
        "summary_sha256": sha256_file(C.CANONICAL_SUMMARY),
        "repeats_sha256": sha256_file(C.CANONICAL_REPEATS),
        "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(json.dumps(rec, indent=1))
    return rec


def assert_canonical_unchanged(verbose: bool = True) -> dict:
    """Re-hash the canonical inputs and compare against the recorded lock. Called at the
    start and the end of every notebook."""
    lockfile = Path(C.CANONICAL_LOCKFILE)
    if not lockfile.exists():
        rec = lock_canonical()
        if verbose:
            print(f"  [lock] created  {lockfile.name}")
            for name, t in rec["trees"].items():
                print(f"  [lock] {name:20} {t['n_files']} files  "
                      f"combined {t['combined_sha256'][:16]}...")
        return rec
    rec = json.loads(lockfile.read_text())

    # The three loose files sit outside both trees, so checksum_tree does not cover them.
    # run_manifest.json is what compare_environment measures against; canonical_summary
    # and repeat_summary carry the baseline means and the run-to-run SD this whole package
    # reads its thresholds from. If any drifts, every comparison silently moves with it.
    for label, key, path in (("run_manifest.json", "manifest_sha256", C.CANONICAL_MANIFEST),
                             ("canonical_summary.csv", "summary_sha256", C.CANONICAL_SUMMARY),
                             ("repeat_summary.csv", "repeats_sha256", C.CANONICAL_REPEATS)):
        now = sha256_file(path)
        if now != rec.get(key):
            raise CanonicalTouchedError(
                f"\n\n  {label.upper()} HAS CHANGED. STOP.\n\n"
                f"    {path}\n"
                f"    locked   {rec.get(key)}   ({rec.get('locked_utc')})\n"
                f"    now      {now}\n\n"
                f"  edge_k reads its baselines and thresholds from this file. Restore it "
                f"from backup before doing anything else. Do not re-lock over the change.\n")

    for name, d in _canonical_targets().items():
        now = checksum_tree(d)
        was = rec["trees"][name]
        if now["combined_sha256"] != was["combined_sha256"]:
            a, b = was["files"], now["files"]
            changed = sorted(k for k in set(a) & set(b) if a[k]["sha256"] != b[k]["sha256"])
            raise CanonicalTouchedError(
                f"\n\n  THE CANONICAL DIRECTORY {name} HAS CHANGED. STOP.\n\n"
                f"    {d}\n"
                f"    locked   {was['combined_sha256']}   ({rec['locked_utc']})\n"
                f"    now      {now['combined_sha256']}\n"
                f"    edited   {changed or '(none)'}\n"
                f"    added    {sorted(set(b) - set(a)) or '(none)'}\n"
                f"    removed  {sorted(set(a) - set(b)) or '(none)'}\n\n"
                f"  This is a read-only input to every edge_k comparison. Restore it from\n"
                f"  backup before doing anything else. Do not re-lock over the change.\n")
    if verbose:
        for name, t in rec["trees"].items():
            print(f"  [lock] {name:20} unchanged  {t['n_files']} files  "
                  f"{t['combined_sha256'][:16]}...")
        print(f"  [lock] manifest / summary / repeats unchanged")
    return rec


# --------------------------------------------------------------------------- read-only IO
def read_canonical(name: str, arm: str = "main") -> Path:
    """The only sanctioned way to reach into a canonical run. Returns a path for reading.
    Anything that opens one of these for writing is a bug."""
    base = C.CANONICAL_MAIN_DIR if arm == "main" else C.CANONICAL_UID_DIR
    p = Path(base) / name
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist in the canonical {arm} arm")
    return p


# --------------------------------------------------- deliberately NOT done: chmod
# protocol_v2/guards.py explains at length why the canonical directory is not chmod'd:
# it is a write to the thing being protected, it bypasses the fence, it is invisible to
# the checksum, and it would break a later legitimate resume. The same reasoning applies
# here and the same decision is taken. The checksum lock is the guarantee.
