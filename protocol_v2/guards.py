"""
protocol_v2 — the locks.

ONE RULE ABOVE ALL OTHERS

    P3_rolling is the canonical Section 5.1 run. It is NEVER trained and NEVER
    overwritten. The protocol ladder reads it from
    results/rerun_20260826_114717_m5/canonical/main/ as read-only input.

    Only P2_cutoff, P1_causal, P1_leaky and P3_acausal may ever be trained.

Four independent things enforce that, so no single mistake is enough to break it:

  1. NO SPECIFICATION.  config.ARM_SPEC contains four arms. P3_rolling is not one of
     them. There is no fold set, no graph choice and no output directory for it. A
     trainer has nothing to consume.

  2. ONE GUARDED CALL SITE.  runner.train_arm is the only place in protocol_v2 that
     calls run_date_gnn_fold, and its first statement is assert_trainable(arm). The
     notebooks never call the training function directly.

  3. WRITES ARE FENCED.  assert_writable refuses every path outside
     protocol_v2/results/. The whole of the existing results/ tree, including the
     canonical run, is unreachable from this code even by an explicit path.

  4. THE CANONICAL DIRECTORY IS CHECKSUMMED.  lock_canonical() records a SHA-256 per
     file plus a combined digest. assert_canonical_unchanged() re-checks it. Every
     notebook calls it at the start AND at the end. If a byte moves, the notebook stops.

Naming, kept explicit everywhere so the two are never confused:

    P3_acausal   a NEW arm, retrained here, rolling folds + forward-reaching graph
    P3_rolling   the CANONICAL arm, not retrained, rolling folds + causal graph
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import config as C


class LockedArmError(RuntimeError):
    """Raised on any attempt to train, or write to, a locked arm."""


class CanonicalTouchedError(RuntimeError):
    """Raised when the canonical Section 5.1 directory no longer matches its checksum."""


# --------------------------------------------------------------------------- arm lock
def assert_trainable(arm: str) -> str:
    """Gate every training call. Returns the arm name so it can be used inline."""
    if arm in C.LOCKED_ARMS:
        raise LockedArmError(
            f"\n\n"
            f"  REFUSED: {arm} is LOCKED and must never be trained.\n\n"
            f"  {arm} is the canonical Section 5.1 run. The protocol-v2 ladder reuses it\n"
            f"  exactly as it was trained in August. Retraining it would replace a known\n"
            f"  0.0066 AP gap with a fresh stochastic draw of about 0.010, and would\n"
            f"  destroy the one arm every contrast in Table 7 is anchored to.\n\n"
            f"  Read it instead:  {C.CANONICAL_P3_DIR}\n"
            f"  Trainable arms :  {', '.join(C.TRAINABLE_ARMS)}\n"
            f"  Do not confuse it with P3_acausal, which IS retrained here.\n")
    if arm not in C.TRAINABLE_ARMS:
        raise LockedArmError(
            f"{arm!r} is not a protocol_v2 arm. Trainable: {', '.join(C.TRAINABLE_ARMS)}.")
    return arm


# --------------------------------------------------------------------------- write fence
def assert_writable(path) -> Path:
    """Every output path in protocol_v2 goes through here. Anything outside
    protocol_v2/results/ is refused, which puts the entire existing results/ tree —
    canonical run included — out of reach."""
    p = Path(path).resolve()
    for root in C.WRITE_ROOTS:
        try:
            p.relative_to(Path(root).resolve())
            return p
        except ValueError:
            continue
    raise LockedArmError(
        f"\n\n  REFUSED: protocol_v2 may not write to\n    {p}\n"
        f"  Permitted root:\n    {Path(C.RESULTS).resolve()}\n"
        f"  Every existing result folder is read-only reference material.\n")


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


def lock_canonical(force: bool = False) -> dict:
    """Record the canonical P3_rolling directory's checksum. Written once, then only
    read. Run this before any GPU time is spent."""
    lockfile = assert_writable(C.CANONICAL_LOCKFILE)
    if lockfile.exists() and not force:
        return json.loads(lockfile.read_text())
    if not Path(C.CANONICAL_P3_DIR).is_dir():
        raise FileNotFoundError(
            f"canonical P3_rolling directory not found: {C.CANONICAL_P3_DIR}")
    rec = checksum_tree(C.CANONICAL_P3_DIR)
    rec.update({
        "arm": "P3_rolling",
        "status": "LOCKED - never trained, never overwritten by protocol_v2",
        "role": "supplies the P3_rolling rung of the protocol ladder, as trained in "
                "August under the canonical 200-epoch configuration",
        "manifest": str(C.CANONICAL_MANIFEST),
        "manifest_sha256": sha256_file(C.CANONICAL_MANIFEST),
        "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(json.dumps(rec, indent=1))
    return rec


def assert_canonical_unchanged(verbose: bool = True) -> dict:
    """Re-hash the canonical directory and compare against the recorded lock. Called at
    the start and the end of every notebook."""
    lockfile = Path(C.CANONICAL_LOCKFILE)
    if not lockfile.exists():
        rec = lock_canonical()
        if verbose:
            print(f"  [lock] created  {lockfile.name}")
            print(f"  [lock] P3_rolling {rec['n_files']} files  "
                  f"combined {rec['combined_sha256'][:16]}...")
        return rec
    rec = json.loads(lockfile.read_text())

    # run_manifest.json sits one level ABOVE canonical/main/, so checksum_tree does not
    # cover it. It was recorded at lock time and must be re-checked separately:
    # provenance.compare_environment reads it to decide what the canonical environment
    # was, so if it drifts, every environment comparison is silently against new ground
    # truth.
    man_now = sha256_file(C.CANONICAL_MANIFEST)
    if man_now != rec.get("manifest_sha256"):
        raise CanonicalTouchedError(
            f"\n\n  THE CANONICAL RUN MANIFEST HAS CHANGED. STOP.\n\n"
            f"    {C.CANONICAL_MANIFEST}\n"
            f"    locked   {rec.get('manifest_sha256')}   ({rec.get('locked_utc')})\n"
            f"    now      {man_now}\n\n"
            f"  This file is what the environment comparison is measured against, and it "
            f"records the configuration the canonical arm was trained under. Restore it "
            f"from backup before doing anything else. Do not re-lock over the change.\n")

    now = checksum_tree(C.CANONICAL_P3_DIR)
    if now["combined_sha256"] != rec["combined_sha256"]:
        was, iss = rec["files"], now["files"]
        changed = sorted(k for k in set(was) & set(iss)
                         if was[k]["sha256"] != iss[k]["sha256"])
        raise CanonicalTouchedError(
            f"\n\n  THE CANONICAL P3_rolling DIRECTORY HAS CHANGED. STOP.\n\n"
            f"    {C.CANONICAL_P3_DIR}\n"
            f"    locked   {rec['combined_sha256']}   ({rec['locked_utc']})\n"
            f"    now      {now['combined_sha256']}\n"
            f"    edited   {changed or '(none)'}\n"
            f"    added    {sorted(set(iss) - set(was)) or '(none)'}\n"
            f"    removed  {sorted(set(was) - set(iss)) or '(none)'}\n\n"
            f"  This arm anchors every contrast in Table 7. Restore it from backup before\n"
            f"  doing anything else. Do not re-lock over the change.\n")
    if verbose:
        print(f"  [lock] P3_rolling unchanged  {rec['n_files']} files  "
              f"combined {rec['combined_sha256'][:16]}...")
        print(f"  [lock] run_manifest.json unchanged  {man_now[:16]}...")
    return rec


# --------------------------------------------------------------------------- read-only IO
def read_canonical(name: str) -> Path:
    """The only sanctioned way to reach into the canonical run. Returns a path for
    reading. Anything that opens one of these for writing is a bug."""
    p = Path(C.CANONICAL_P3_DIR) / name
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist in the canonical P3_rolling arm")
    return p


# --------------------------------------------------------- deliberately NOT done: chmod
# An earlier version of this file had a freeze_canonical_permissions() helper that
# chmod'd the canonical arm to 0444/0555. It was removed, for three reasons:
#
#   1. It writes to the canonical directory. That contradicts the rule this module
#      exists to enforce, and it did it by bypassing assert_writable, because chmod is
#      not a write the fence sees.
#   2. It is not reversible by anything here. Deleting protocol_v2/ would have left the
#      canonical arm permanently read-only, and rerun_canonical.py --mode resume appends
#      to scores.csv and foldckpt.csv, so a later legitimate resume would have failed.
#   3. checksum_tree hashes content and size, not mode, so the change would have been
#      invisible to assert_canonical_unchanged. The notebook would still have printed
#      "unchanged" over a directory this code had in fact altered.
#
# The checksum lock is the guarantee. If you want the filesystem to enforce it as well,
# do it yourself, outside this package, and remember to undo it before any resume:
#
#     chmod -R a-w  results/rerun_20260826_114717_m5/canonical/main/     # lock
#     chmod -R u+w  results/rerun_20260826_114717_m5/canonical/main/     # unlock
#
# Nothing in protocol_v2 depends on either state.
