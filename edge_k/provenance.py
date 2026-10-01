"""
edge_k — what ran, on what, built from which bytes.

Same three jobs as protocol_v2/provenance.py, and deliberately the same shape so the two
packages' manifests are directly comparable:

  source_hashes()        SHA-256 of every source file the run touches.
  environment_manifest() the fields rerun_canonical.environment_manifest() records.
  compare_environment()  field by field against the canonical run, printing differences
                         rather than hiding them.

Plus the analysis lock, which is written before the first arm trains rather than by the
assembly step afterwards.
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

# Files whose bytes determine what the arms compute.
SOURCE_FILES = (
    "src/run_date_gnn.py",
    "src/temporal_graph.py",
    "src/acausal_graph.py",
    "src/model.py",                    # <- carries the use_edge_bias switch
    "src/evaluation.py",
    "src/temporal_folds.py",
    "src/experiment_config.py",
    "src/features_aggregation.py",
    "src/run_experiment.py",
    "src/data_io.py",
    "src/prepared_loader.py",
    "rerun_canonical.py",
    "edge_k/config.py",
    "edge_k/guards.py",
    "edge_k/provenance.py",
    "edge_k/runner.py",
    "edge_k/controls.py",
    "edge_k/status.py",
    "edge_k/analysis_settings.py",
    "edge_k/make_notebooks.py",
)

ANALYSIS_FILES = ("edge_k/analysis_settings.py",)

DEPRECATED_FILES = ("src/protocol_runner.py",)


def source_hashes(include_notebooks: bool = True) -> dict:
    out = {}
    for rel in SOURCE_FILES:
        p = C.REPO / rel
        out[rel] = sha256_file(p) if p.exists() else None
    for rel in DEPRECATED_FILES:
        p = C.REPO / rel
        out[f"{rel}  [DEPRECATED - not used]"] = sha256_file(p) if p.exists() else None
    if include_notebooks and Path(C.NOTEBOOKS).is_dir():
        for p in sorted(Path(C.NOTEBOOKS).glob("*.ipynb")):
            out[f"edge_k/notebooks/{p.name}"] = sha256_file(p)
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


COMPARE_FIELDS = ("python", "platform", "numpy", "pandas", "torch", "cuda", "cudnn",
                  "sklearn", "xgboost", "scipy", "gpus", "dataset_sha256", "dataset_bytes")

BENIGN = {
    "platform": "kernel version only; not a numerical dependency",
    "xgboost": "no GNN arm imports xgboost. src/model_factory.py is the only module that "
               "does, and no edge_k arm reaches it. This difference cannot touch a number "
               "here, but it is recorded because it is real.",
}


def compare_environment(current: dict, canonical_manifest=None, verbose: bool = True) -> dict:
    """Field by field against the manifest the canonical arm was trained under. Never
    raises: a difference is something to disclose, not necessarily something to stop for.
    The exception is the dataset hash, flagged FATAL, because it would mean the arms are
    not scoring the same rows as the baseline."""
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
            print("\n  The dataset hash does not match. These arms would not be scoring the "
                  "same rows as the baseline. STOP.")
        elif differs:
            print("\n  These differences are real and must be recorded in the manifest and "
                  "disclosed in the paper. They are not a reason to stop.")
    return out


def freeze_requirements() -> Path:
    """pip freeze, written once, before arm 1."""
    out = assert_writable(C.RESULTS / "requirements.lock.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  [env] requirements.lock.txt already frozen "
              f"({out.stat().st_size} bytes); left alone")
        return out
    txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    out.write_text(txt)
    print(f"  [env] froze {len(txt.splitlines())} packages -> {out.name}")
    return out


# ------------------------------------------------------------------ the analysis lock
ANALYSIS_LOCKFILE = C.RESULTS / "analysis_lock.json"


def _analysis_hashes() -> dict:
    return {rel: (sha256_file(C.REPO / rel) if (C.REPO / rel).exists() else None)
            for rel in ANALYSIS_FILES}


def write_analysis_lock(verbose: bool = True) -> dict:
    """Freeze the analysis BEFORE the first arm is trained. Write-once: if the lock
    exists it is verified, not overwritten."""
    p = assert_writable(ANALYSIS_LOCKFILE)
    if p.exists():
        return verify_analysis_lock(verbose=verbose)
    import analysis_settings as A
    rec = {
        "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": "The analysis is declared before any result exists. If analysis_settings.py "
               "is edited after a number is seen, these hashes move and "
               "verify_analysis_lock() fails.",
        "files": _analysis_hashes(),
        "settings_sha256": A.settings_sha256(),
        "declared": {
            "primary_metric": A.PRIMARY_METRIC,
            "baselines": A.BASELINES,
            "baseline_values": A.BASELINE_VALUES,
            "primary_read": A.PRIMARY_READ,
            "thresholds": A.THRESHOLDS,
            "wilcoxon": A.WILCOXON,
            "sign_convention": A.SIGN_CONVENTION,
            "conduct": A.CONDUCT,
            "contrasts": [list(c) for c in A.CONTRASTS],
        },
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    if verbose:
        print(f"  [analysis] frozen at {rec['locked_utc']}  "
              f"settings {rec['settings_sha256'][:16]}...")
    return rec


def verify_analysis_lock(verbose: bool = True) -> dict:
    p = Path(ANALYSIS_LOCKFILE)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. The analysis must be frozen before any arm is trained. "
            f"Call provenance.write_analysis_lock() from the first arm notebook.")
    rec = json.loads(p.read_text())
    now = _analysis_hashes()
    moved = [k for k, v in rec["files"].items() if now.get(k) != v]
    if moved:
        raise RuntimeError(
            f"\n\n  THE FROZEN ANALYSIS HAS BEEN EDITED SINCE IT WAS LOCKED. STOP.\n\n"
            f"    moved: {moved}\n"
            f"    locked {rec['locked_utc']}\n\n"
            f"  The whole point of pre-declaring the analysis is that it cannot move after\n"
            f"  the results are in hand. Restore the file, or record openly that the\n"
            f"  analysis changed after seeing results and why.\n")
    if verbose:
        print(f"  [analysis] unchanged since {rec['locked_utc']}  "
              f"{rec['settings_sha256'][:16]}...")
    return rec


# ------------------------------------------------------------------ the arm manifest
def arm_manifest(arm: str, summary: dict, tcfg, mcfg, gcfg, extra: dict = None) -> dict:
    """Everything needed to say what this arm was, on what, built from which bytes."""
    from dataclasses import asdict
    import runner
    man = {
        "arm": arm,
        "package": "edge_k",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "the_one_change": C.ARM_SPEC[arm]["changes"],
        "what": C.ARM_SPEC[arm]["what"],
        "answers": C.ARM_SPEC[arm]["answers"],
        "receptive_field": C.receptive_field(arm),
        "summary": summary,
        "train_config": asdict(tcfg),
        "model_config": asdict(mcfg),
        "graph_config": asdict(gcfg),
        "zeroed_edge_columns": list(C.ARM_SPEC[arm]["zero_cols"]),
        "baseline": {k: __import__("analysis_settings").BASELINE_VALUES[v]
                     for k, v in __import__("analysis_settings").BASELINES[arm].items()},
        "source_hashes": source_hashes(),
        "analysis_lock": verify_analysis_lock(verbose=False),
        "segments": 1,
        "note_single_execution": "edge_k arms are never resumed. A crashed arm is "
                                 "restarted clean and the abandoned attempt is recorded.",
        "note_not_canonical": f"{arm} is a NEW arm trained by edge_k. It is not, and must "
                              f"not be confused with, any canonical or protocol_v2 arm.",
    }
    if extra:
        man.update(extra)
    return man
