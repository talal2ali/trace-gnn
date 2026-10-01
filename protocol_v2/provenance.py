"""
protocol_v2 — what ran, on what, built from which bytes.

Three jobs:

  source_hashes()        SHA-256 of every source file the run touches. The canonical run
                         predates version control — its run_manifest.json records
                         "git_commit": null — so for THAT run these hashes are the only
                         thing pinning the code state. The repository now exists, so for
                         the new arms both are recorded: the commit says which version,
                         the hashes say whether the working tree matched it.

  environment_manifest() the same fields rerun_canonical.environment_manifest() records,
                         so the two are directly comparable.

  compare_environment()  the part the plan was missing. Freezing the current environment
                         records it; it does not check it against the environment the
                         canonical arm was trained in. This does, field by field, and
                         prints the differences rather than hiding them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from guards import sha256_file, assert_writable

# Files whose bytes determine what the four arms compute. Hashing more than this is
# noise; hashing less leaves a gap.
SOURCE_FILES = (
    "src/run_date_gnn.py",
    "src/temporal_graph.py",
    "src/acausal_graph.py",
    "src/model.py",
    "src/evaluation.py",
    "src/temporal_folds.py",
    "src/experiment_config.py",
    "src/features_aggregation.py",
    "src/run_experiment.py",
    "src/data_io.py",
    "src/prepared_loader.py",
    "rerun_canonical.py",
    "protocol_v2/config.py",
    "protocol_v2/guards.py",
    "protocol_v2/provenance.py",
    "protocol_v2/splits.py",
    "protocol_v2/runner.py",
    "protocol_v2/assembly_settings.py",
    "protocol_v2/assemble.py",          # the analysis ITSELF; omitting it was a real gap
    "protocol_v2/make_notebooks.py",
)

# The analysis. assembly_settings.py declares the rules, assemble.py implements them —
# hashing only the first would freeze the declaration and leave the arithmetic free to
# move, which is backwards. Both are locked before the first arm is trained.
ANALYSIS_FILES = ("protocol_v2/assembly_settings.py", "protocol_v2/assemble.py")

# Deliberately NOT hashed as an input, because nothing here may use it:
DEPRECATED_FILES = ("src/protocol_runner.py",)


def source_hashes(include_notebooks: bool = True) -> dict:
    out = {}
    for rel in SOURCE_FILES:
        p = C.REPO / rel
        out[rel] = sha256_file(p) if p.exists() else None
    for rel in DEPRECATED_FILES:
        p = C.REPO / rel
        out[f"{rel}  [DEPRECATED - not used]"] = sha256_file(p) if p.exists() else None
    if include_notebooks:
        for p in sorted(Path(C.NOTEBOOKS).glob("*.ipynb")):
            out[f"protocol_v2/notebooks/{p.name}"] = sha256_file(p)
    return out


def print_source_hashes(h: dict | None = None) -> dict:
    h = h or source_hashes()
    print("source file hashes (SHA-256, first 16)")
    for k, v in h.items():
        print(f"  {v[:16] if v else '-- MISSING --':16}  {k}")
    return h


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=C.REPO,
                                             stderr=subprocess.DEVNULL).decode().strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def environment_manifest(dataset_path=None) -> dict:
    import torch
    man = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    for mod in ("sklearn", "xgboost", "scipy"):
        try:
            man[mod] = __import__(mod).__version__
        except Exception:
            pass
    man.update(git_state())
    if dataset_path is not None:
        src = Path(dataset_path)
        if src.exists():
            man.update(dataset_file=str(src), dataset_bytes=src.stat().st_size,
                       dataset_sha256=sha256_file(src))
    return man


# ------------------------------------------------------------------ the comparison (D5)
COMPARE_FIELDS = ("python", "platform", "numpy", "pandas", "torch", "cuda", "cudnn",
                  "sklearn", "xgboost", "scipy", "gpus", "dataset_sha256", "dataset_bytes")

# Differences that cannot change a number, and are expected. Anything not listed here is
# reported as a difference that must be disclosed.
BENIGN = {
    "platform": "kernel version only; not a numerical dependency",
}


def compare_environment(current: dict, canonical_manifest=None, verbose: bool = True) -> dict:
    """Field-by-field against the manifest the canonical P3_rolling arm was trained under.
    Returns {'same': [...], 'differs': {...}, 'missing': [...]}. Never raises: a
    difference is something to disclose, not necessarily something to stop for. The one
    exception is the dataset hash, which is flagged FATAL because it means the two arms
    would not be scoring the same rows."""
    path = Path(canonical_manifest or C.CANONICAL_MANIFEST)
    canon = json.loads(path.read_text())

    same, differs, missing = [], {}, []
    for f in COMPARE_FIELDS:
        a, b = canon.get(f, "<absent>"), current.get(f, "<absent>")
        if a == "<absent>" or b == "<absent>":
            missing.append(f)
        elif a == b:
            same.append(f)
        else:
            differs[f] = {"canonical": a, "now": b, "benign": BENIGN.get(f)}

    # device is not a manifest field, but it is a real difference and belongs in the list
    if C.DEVICE == C.CANONICAL_DEVICE:
        same.append("device")
    else:
        differs["device"] = {
            "canonical": C.CANONICAL_DEVICE, "now": C.DEVICE,
            "benign": "both cards are RTX 6000 Ada; a different card of the same model "
                      "changes float accumulation order, not the configuration",
        }

    fatal = "dataset_sha256" in differs
    out = {"canonical_manifest": str(path), "same": same, "differs": differs,
           "missing": missing, "fatal": fatal}

    if verbose:
        print(f"environment vs the canonical run ({canon.get('utc')})")
        print(f"  matches ({len(same)}): {', '.join(same) or '-'}")
        if missing:
            print(f"  not recorded on one side: {', '.join(missing)}")
        if not differs:
            print("  no differences")
        for k, v in differs.items():
            tag = "FATAL " if k == "dataset_sha256" else ("note  " if v["benign"] else "DIFFERS")
            print(f"  {tag} {k}")
            print(f"          canonical : {v['canonical']}")
            print(f"          now       : {v['now']}")
            if v["benign"]:
                print(f"          -> {v['benign']}")
        if fatal:
            print("\n  The dataset hash does not match. The new arms would not be scoring "
                  "the same rows as P3_rolling. STOP.")
        elif differs:
            print("\n  These differences are real and must be recorded in the manifest and "
                  "disclosed in the paper. They are not a reason to stop.")
    return out


def freeze_requirements() -> Path:
    """pip freeze, written once, before arm 1."""
    out = assert_writable(C.RESULTS / "requirements.lock.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  [env] requirements.lock.txt already frozen ({out.stat().st_size} bytes); "
              f"left alone")
        return out
    txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    out.write_text(txt)
    print(f"  [env] froze {len(txt.splitlines())} packages -> {out.name}")
    return out


# ------------------------------------------------------------------ the analysis lock (C2)
ANALYSIS_LOCKFILE = C.RESULTS / "analysis_lock.json"


def write_analysis_lock(verbose: bool = True) -> dict:
    """Freeze the analysis BEFORE the first arm is trained.

    The point of pre-declaring the analysis is lost if the only durable record of it is
    written by the assembly step, after the results are in hand. An earlier version did
    exactly that: settings_sha256() was printed by the arm notebooks and first written to
    disk by A5. This writes the lock from the arm notebooks instead — whichever runs
    first — so the timestamp precedes every result file in the directory.

    Write-once. If the lock exists it is verified, not overwritten.
    """
    p = assert_writable(ANALYSIS_LOCKFILE)
    hashes = {rel: (sha256_file(C.REPO / rel) if (C.REPO / rel).exists() else None)
              for rel in ANALYSIS_FILES}
    if p.exists():
        return verify_analysis_lock(verbose=verbose)
    rec = {
        "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": "The contrasts, fold-set rule, Wilcoxon settings, averaging and rounding "
               "were fixed before any of the four arms was trained. This file is the "
               "evidence: it is written by the first arm notebook, before that arm runs.",
        "files": hashes,
        "note": "assembly_settings.py declares the rules; assemble.py implements them. "
                "Both are locked. Freezing only the declaration would leave the "
                "arithmetic free to move.",
        "scope": "This does not stop you changing the analysis. It stops the change being "
                 "silent: A5 compares against these hashes and refuses to run if either "
                 "file has moved.",
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    if verbose:
        print(f"  [analysis] froze the analysis before any training: {p.name}")
        for rel, h in hashes.items():
            print(f"             {h[:16] if h else '-- MISSING --'}  {rel}")
    return rec


def verify_analysis_lock(verbose: bool = True, strict: bool = True) -> dict:
    """Compare the analysis files against the lock written before training started."""
    p = Path(ANALYSIS_LOCKFILE)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. It is written by the first arm notebook, before any "
            f"training. Its absence means the analysis was never frozen, so there is "
            f"nothing to check the current analysis code against.")
    rec = json.loads(p.read_text())
    moved = {}
    for rel, was in rec["files"].items():
        now = sha256_file(C.REPO / rel) if (C.REPO / rel).exists() else None
        if now != was:
            moved[rel] = {"locked": was, "now": now}
    if moved and strict:
        raise AssertionError(
            f"THE ANALYSIS CHANGED AFTER IT WAS FROZEN.\n"
            f"  locked {rec['locked_utc']}\n" +
            "".join(f"    {rel}\n      locked {v['locked']}\n      now    {v['now']}\n"
                    for rel, v in moved.items()) +
            "\nThe results have been seen since the lock was written. Either restore the "
            "analysis files, or re-lock deliberately and say in the paper that you did.")
    if verbose:
        print(f"  [analysis] unchanged since it was frozen at {rec['locked_utc']}")
    return rec


def write_manifest(path, payload: dict) -> Path:
    p = assert_writable(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, default=str))
    return p
