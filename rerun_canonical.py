#!/usr/bin/env python3
"""
Canonical re-run of every arm the paper reports from the TRACE-GNN model.

Produces ONE designated execution per arm, fully instrumented, plus optional repeats of
the main arm used only to state run-to-run variation. Repeats never overwrite canonical
output.

Why this exists
---------------
Three problems in the current results, all fixed by one session.

1. The reported number comes from an execution that its own notebook flagged DIVERGENT
   against a 0.002 tolerance, at a gap of 0.0096. It was used anyway.
2. Notebook 15 passed no val_curve_path, so no validation curves exist for the reported
   run and Figure 7(b) plots a different execution.
3. Two executions cannot support the run-to-run claim in Section 5.4.

Why all four arms and not just the main one
-------------------------------------------
Table 5, Figure 8 and Section 5.4 pair the main arm against uid_only, maxagg and depth3.
If only the main arm is re-run and its mean moves, every one of those comparisons is
computed across two different executions. So either all four move together or none do.
XGBoost is not re-run, being deterministic under a fixed seed, but its saved scores are
required for the budget sweep.

Where output goes
-----------------
A NEW timestamped folder every time, `results/rerun_YYYYmmdd_HHMMSS/` by default, so no
existing result is ever touched. Nothing outside that folder is written. The existing
results tree is read only, and only to locate the XGBoost baseline, which is not re-run.

    results/rerun_20260824_0930/
      run_manifest.json          versions, dataset sha256, slice timestamps, graph facts
      fold_composition.csv       per-fold train/val/test sizes and fraud counts
      canonical_summary.csv      one row per arm
      COMPARISON.md              paper vs re-run, and what goes stale. Read this first
      canonical/<arm>/           per_fold.csv scores.csv val_curves.csv foldckpt.csv summary.json
      repeats/repeat_NN/main/    same again, variance only, never cited
      derived/                   paired_tests.csv and the regenerated budget sweep
      per_fold_variation.csv     only when --repeats is used

Usage
-----
    # validate data, folds, graph and every assertion, then stop. About a minute.
    python rerun_canonical.py --prepared ../prepared --dry-run

    # all four arms, no repeats. Roughly 32 h on one RTX 6000 Ada.
    python rerun_canonical.py --prepared ../prepared

    # all four arms plus three repeats of the main arm. Roughly 56 h.
    python rerun_canonical.py --prepared ../prepared --repeats 3

    # main arm only, if you accept that downstream comparisons go stale
    python rerun_canonical.py --prepared ../prepared --arms main

    # pin the folder name yourself
    python rerun_canonical.py --prepared ../prepared --out results/my_rerun

Resuming after a crash
----------------------
Training can die part way through: out of memory on the largest fold, a power cut, or
someone else needing the GPU. `--resume` finishes an existing folder instead of starting
a new one. It is idempotent, so it is safe to run repeatedly:

    folder absent      -> starts a fresh run
    folder incomplete  -> trains only the missing folds
    folder complete    -> trains nothing, just rebuilds the summary

    python rerun_canonical.py --resume --out results/rerun_20260826_114717_m5

`run_date_gnn_fold` already knows how to resume: it reads `foldckpt.csv`, skips the
weeks listed there and appends only new ones. Without `--resume` that path is blocked
on purpose, because a stale checkpoint would otherwise make a *fresh* run silently skip
folds. `--resume` says you mean it, and records the fact in `run_manifest.json` so the
provenance stays auditable.

Before resuming, the three append-mode files are cross-checked against each other. If
they disagree about how far the run got, a fold was half-written and resuming would
duplicate or lose rows, so the run aborts rather than corrupting the output.

A resumed fold begins with a fresh RNG state rather than the state a continuous run
would have reached, so a resumed arm is not bit-identical to an uninterrupted one. It
is the same configuration and the same seed, drawn at a different point in the stream.
`run_manifest.json` records every segment so this is visible rather than hidden.

Locking is per arm, so different arms may safely run at the same time on different GPUs:

    python rerun_canonical.py --resume --arms main,uid_only,maxagg --device cuda:0 &
    python rerun_canonical.py --resume --arms depth3               --device cuda:1 &

Run --dry-run first. A configuration error then costs a minute rather than eight hours.
It also prints a projected peak VRAM per arm, which is the failure this script hits most.
"""
from __future__ import annotations

import os

# Reduces allocator fragmentation, which is what turns a tight fit into an OOM on the
# largest folds. Purely an allocator change: it affects no number this script produces.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse, atexit, errno, hashlib, json, platform, shutil, subprocess, sys, time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# What the paper reports. Nothing here is a library default. Drift aborts the run.
# --------------------------------------------------------------------------------------
EXPECTED_TRAIN = dict(
    epochs=200, lr=5e-4, weight_decay=1e-4, grad_clip=1.0, val_frac=0.15,
    patience=25, batch_size=2048, loss="focal", focal_gamma=2.0,
    early_stop_metric="ap", use_swa=False, seed=42,
)
EXPECTED_MODEL = dict(hidden_dim=128, n_heads=4, ffn_mult=2, dropout=0.2, cat_emb_dim=16)

# arm -> (entity_cols, use_max_agg, n_layers, mean AP currently in the paper)
ARMS = {
    "main":     (("uid", "merchant"),  False, 2, 0.8427),
    "uid_only": (("uid", "__none__"),  False, 2, 0.7731),
    "maxagg":   (("uid", "merchant"),  True,  2, 0.8237),
    "depth3":   (("uid", "merchant"),  False, 3, 0.8568),
}
XGB_BASELINE_AP = 0.6361          # not re-run, deterministic under a fixed seed

EXPECTED_EDGE_DIM = 5             # 3 + len(entity_cols). 4 means the placeholder was lost.
EXPECTED_N_FEATURES = 11
EXPECTED_N_FOLDS = 17
EXPECTED_SCORED_ROWS = 338_456
EXPECTED_POSITIVES = 1_276
EXPECTED_SLICE_ROWS = 500_000
MAX_PRIOR_NEIGHBORS = 10
NUMERIC = ("amt", "distance", "age", "city_pop", "hour", "dayofweek")

# every file run_date_gnn_fold opens in append mode. All must be absent before a run.
APPEND_FILES = ("scores.csv", "val_curves.csv", "foldckpt.csv")

# files prepare_frame() needs, checked up front so a wrong --prepared gives one readable
# line instead of a traceback from inside data_io
PREPARED_FILES = ("sparkov_clean_v1.csv", "column_manifest.json")

# Measured on an RTX 6000 Ada: depth3 week 24 held 41.5 GiB of activations across
# 3 layers at roughly 5.8M edges in the batch. Each layer saves q[dst], k[src], v[src]
# and msg, all (E, hidden), so cost is linear in edges x layers.
BYTES_PER_EDGE_PER_LAYER = 2400


def die(msg): print(f"\n  ABORT  {msg}\n", file=sys.stderr); sys.exit(1)
def check(cond, msg):
    if not cond: die(msg)


def find_prepared(explicit) -> Path:
    """Locate the prepared/ folder. It sits OUTSIDE the repo, two levels up, so the
    default is an upward search rather than a hard-coded relative path."""
    here = Path(__file__).resolve().parent
    if explicit is not None:
        p = Path(explicit).resolve()
        check(p.exists(), f"--prepared {p} does not exist. prepared/ is TWO levels above "
                          f"the repo, so from here it is normally ../../prepared")
        missing = [f for f in PREPARED_FILES if not (p / f).exists()]
        check(not missing, f"{p} is missing {missing}, so it is not the prepared folder")
        return p
    for base in [here, *here.parents]:
        cand = base / "prepared"
        if all((cand / f).exists() for f in PREPARED_FILES):
            return cand.resolve()
    die(f"no prepared/ folder containing {list(PREPARED_FILES)} found above {here}. "
        f"Pass --prepared explicitly")


def acquire_arm_lock(armdir: Path):
    """One writer per arm. Two processes on the same arm both resume from the same
    checkpoint, both train the same fold and both APPEND, which duplicates folds and
    fails the assertions after the GPU time is already spent. Locking per arm rather
    than per folder lets different arms run concurrently on different GPUs."""
    armdir.mkdir(parents=True, exist_ok=True)
    lock = armdir / ".arm.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        holder = lock.read_text().strip() if lock.exists() else "unknown"
        pid = holder.split()[0] if holder else ""
        if pid.isdigit() and Path(f"/proc/{pid}").exists():
            die(f"arm {armdir.name} is already being trained by another process\n"
                f"         {holder}\n"
                f"         Wait for it, or stop it with:  kill {pid}")
        # The named process is gone, so the lock is stale: a previous run was killed or
        # crashed. Reclaim it rather than demanding a manual rm, since the fold files are
        # separately cross-checked for consistency before anything is appended.
        print(f"    reclaiming stale lock from dead process {pid or '?'} ({holder})")
        lock.unlink(missing_ok=True)
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{os.getpid()} on {platform.node()} since "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
    os.close(fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))
    return lock


def arm_state(armdir: Path, all_weeks) -> dict:
    """What this arm has on disk, and whether its append-mode files agree with each
    other. They must, or a fold was half-written and resuming would corrupt it."""
    st = {"complete": False, "missing": list(all_weeks), "consistent": True, "detail": {},
          "started": False, "resumable": True}
    ck = armdir / "foldckpt.csv"
    if not ck.exists():
        # foldckpt.csv is the resume checkpoint and is gitignored, so a distributed copy of a
        # finished run may not carry it. per_fold.csv is the arm's actual result and is always
        # written on completion, so fall back to it for reporting. Resume is not possible
        # without the checkpoint: run_date_gnn_fold would retrain every fold.
        pf = armdir / "per_fold.csv"
        if not pf.exists():
            st["detail"]["foldckpt.csv"] = "absent, arm not started"
            return st
        done = sorted(int(w) for w in pd.read_csv(pf)["test_week"].unique())
        st["started"] = True
        st["resumable"] = False
        st["detail"]["per_fold.csv"] = f"{len(done)} folds (foldckpt.csv absent, report only)"
    else:
        st["started"] = True
        done = sorted(int(w) for w in pd.read_csv(ck)["test_week"].unique())
        st["detail"]["foldckpt.csv"] = f"{len(done)} folds"
    for fname, col in (("val_curves.csv", "test_week"), ("scores.csv", "fold")):
        p = armdir / fname
        if not p.exists():
            continue
        got = sorted(int(w) for w in pd.read_csv(p)[col].unique())
        st["detail"][fname] = f"{len(got)} folds"
        if got != done:
            st["consistent"] = False
            st["detail"][fname] += f"  MISMATCH vs foldckpt on {sorted(set(got) ^ set(done))}"
    st["missing"] = [w for w in all_weeks if w not in done]
    st["complete"] = not st["missing"]
    return st


# --------------------------------------------------------------------------------------
def environment_manifest(prepared: Path, clean_csv: str) -> dict:
    import torch
    man = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    for mod in ("sklearn", "xgboost", "scipy"):
        try: man[mod] = __import__(mod).__version__
        except Exception: pass
    try:
        here = Path(__file__).parent
        man["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, stderr=subprocess.DEVNULL).decode().strip()
        man["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=here, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        man["git_commit"] = None
    src = prepared / clean_csv
    if src.exists():
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
        man.update(dataset_file=str(src), dataset_bytes=src.stat().st_size,
                   dataset_sha256=h.hexdigest())
    return man


def layersof(arm) -> int:
    return ARMS[arm][2]


def free_vram_gb(device):
    """Free VRAM on the target device, or None if it cannot be determined."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        free, _total = torch.cuda.mem_get_info(idx)
        return free / 1e9
    except Exception:
        return None


def fingerprint(tcfg, mcfg, gcfg) -> str:
    p = json.dumps({k: {kk: str(vv) for kk, vv in sorted(asdict(c).items())}
                    for k, c in (("t", tcfg), ("m", mcfg), ("g", gcfg))}, sort_keys=True)
    return hashlib.sha256(p.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
def build(args):
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from experiment_config import ExperimentConfig
    import temporal_folds, evaluation
    from run_experiment import prepare_frame
    from temporal_graph import GraphConfig, build_temporal_edges, assert_causal
    from model import ModelConfig
    from run_date_gnn import TrainConfig, subsample_recent, filter_folds

    cfg = ExperimentConfig(
        prepared_dir=Path(args.prepared), clean_csv_name="sparkov_clean_v1.csv",
        id_col="TransactionID", target_col="is_fraud", time_col="unix_time",
        week_col="week_idx", amount_col="amt", uid_col="uid",
        non_feature_cols=("uid",), min_train_weeks=8,
    )
    print("  loading prepared frame ...")
    df, _, feature_cols, uid_feats = prepare_frame(cfg)
    df = subsample_recent(df, cfg.week_col, EXPECTED_SLICE_ROWS).reset_index(drop=True)
    check(len(df) == EXPECTED_SLICE_ROWS, f"slice has {len(df):,} rows, expected {EXPECTED_SLICE_ROWS:,}")
    check(df[cfg.time_col].is_monotonic_increasing,
          "frame is not time-sorted. prepare_sparkov.py must sort with kind='mergesort'")

    num = [c for c in feature_cols if c in NUMERIC] + uid_feats
    check(len(num) == EXPECTED_N_FEATURES, f"{len(num)} features, expected {EXPECTED_N_FEATURES}: {num}")
    folds, report = filter_folds(df, temporal_folds.make_folds(df, cfg), cfg.target_col, 5, 5)
    check(len(folds) == EXPECTED_N_FOLDS, f"{len(folds)} folds, expected {EXPECTED_N_FOLDS}")

    tcfg = TrainConfig(**EXPECTED_TRAIN, device=args.device)
    for k, v in EXPECTED_TRAIN.items():
        check(getattr(tcfg, k) == v, f"TrainConfig.{k}={getattr(tcfg,k)!r}, expected {v!r}")
    check(tcfg.use_swa is False,
          "use_swa=True silently disables early stopping. Every reported run uses False")

    print("  building the causal graph ...")
    t0 = time.time()
    gmain = GraphConfig(time_col=cfg.time_col, entity_cols=ARMS["main"][0],
                        max_prior_neighbors=MAX_PRIOR_NEIGHBORS)
    ei, dt, rel = build_temporal_edges(df.reset_index(drop=True), gmain)
    ties = assert_causal(ei, df[cfg.time_col].to_numpy())
    build_s = time.time() - t0

    # ---- preflight: project peak VRAM on the WORST fold of each arm -------------------
    # This is the failure mode that actually bites, and it bites at hour 27 on the last
    # fold. Sampling one batch costs a few seconds of numpy and no GPU at all.
    # Only the arms this invocation will actually train. Profiling all four costs a few
    # minutes of graph building on every resume, for arms that are already finished.
    want = [a.strip() for a in getattr(args, "arms", ",".join(ARMS)).split(",") if a.strip()]
    want = [a for a in ARMS if a in want] or list(ARMS)
    mem = {}
    big_w, big_tr, _ = max(folds, key=lambda f: len(f[1]))
    try:
        from run_date_gnn import GraphCSR, sample_subgraph, build_edge_features
        rng = np.random.default_rng(0)
        for arm in want:
            entity, _, layers, _ = ARMS[arm]
            g = GraphConfig(time_col=cfg.time_col, entity_cols=entity,
                            max_prior_neighbors=MAX_PRIOR_NEIGHBORS)
            aei, adt, arel = build_temporal_edges(df.reset_index(drop=True), g)
            aea = build_edge_features(df, aei, adt, arel, len(entity), cfg.amount_col)
            csr = GraphCSR(aei, aea, len(df))
            seeds = rng.choice(np.asarray(big_tr), size=min(tcfg.batch_size, len(big_tr)),
                               replace=False)
            _, sei, _, _ = sample_subgraph(seeds, layers, csr)
            e = int(sei.shape[1])
            gb = e * layers * BYTES_PER_EDGE_PER_LAYER / 1e9
            mem[arm] = (e, gb)
            del csr, aei, aea
    except Exception as exc:                       # never let an estimate block a run
        print(f"    (peak-memory estimate unavailable: {exc})")

    if mem:
        print(f"  projected peak VRAM on the largest fold (week {int(big_w)}, "
              f"{len(big_tr):,} train rows, batch {tcfg.batch_size}):")
        for arm, (e, gb) in mem.items():
            print(f"    {arm:9} {layersof(arm)} hops  {e:>10,} edges/batch  ~{gb:5.1f} GB")
        free = free_vram_gb(args.device)
        worst = max(mem.values(), key=lambda t: t[1])[1]
        if free is not None:
            print(f"  {args.device} reports {free:.1f} GB free")
            if worst > free:
                print(f"\n  WARNING  the largest arm needs about {worst:.0f} GB and only "
                      f"{free:.0f} GB is free.\n"
                      f"           It will very likely OOM on the last folds. Options:\n"
                      f"           - use a GPU with no desktop on it (--device cuda:1)\n"
                      f"           - run fewer arms per process\n"
                      f"           - the run is resumable, so a crash costs one fold, "
                      f"not the arm\n")
            elif worst > 0.85 * free:
                print(f"\n  NOTE  the largest arm needs about {worst:.0f} GB of the "
                      f"{free:.0f} GB free. That is tight; anything else using the GPU "
                      f"could push it over.\n")

    s0, s1 = int(df[cfg.time_col].iloc[0]), int(df[cfg.time_col].iloc[-1])
    facts = dict(
        rows=len(df), features=num, n_folds=len(folds),
        test_weeks=[int(w) for w, _, _ in folds],
        edges=int(ei.shape[1]), edges_per_node=round(ei.shape[1] / len(df), 2),
        contemporaneous_edges=int(ties) if ties is not None else None,
        graph_build_seconds=round(build_s, 1),
        slice_unix_start=s0, slice_unix_end=s1,
        slice_utc_start=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(s0)),
        slice_utc_end=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(s1)),
        positives=int(df[cfg.target_col].sum()),
        prevalence_pct=round(100 * df[cfg.target_col].mean(), 4),
    )
    print(f"    {facts['edges']:,} edges, {facts['edges_per_node']}/node, "
          f"{facts['contemporaneous_edges']} contemporaneous, {build_s:.0f}s")
    print(f"    slice {facts['slice_utc_start']} to {facts['slice_utc_end']}, "
          f"{facts['positives']:,} positives ({facts['prevalence_pct']}%)")
    return dict(cfg=cfg, df=df, num=num, folds=folds, report=report, tcfg=tcfg,
                evaluation=evaluation, facts=facts, ModelConfig=ModelConfig,
                GraphConfig=GraphConfig)


# --------------------------------------------------------------------------------------
def one_arm(arm, outdir, B, force, seed=None, resume=False, save_weights=False):
    """seed=None uses the reported seed. Repeats pass a different one so that the
    variance estimate reflects initialisation and batch order, not only GPU atomics.

    resume=True lets run_date_gnn_fold find an existing foldckpt.csv and continue from
    it, training only the missing folds. Without it, an existing checkpoint is an error
    (or, with --force, is deleted), because a fresh run must never silently skip folds.
    """
    from run_date_gnn import run_date_gnn_fold
    entity, max_agg, layers, paper_ap = ARMS[arm]
    cfg, df, num, folds, tcfg = B["cfg"], B["df"], B["num"], B["folds"], B["tcfg"]
    if seed is not None and seed != tcfg.seed:
        tcfg = replace(tcfg, seed=seed)
        print(f"    seed {seed} (reported run uses {B['tcfg'].seed}); variance estimate only")

    mcfg = B["ModelConfig"](n_layers=layers, use_max_agg=max_agg, **EXPECTED_MODEL)
    gcfg = B["GraphConfig"](time_col=cfg.time_col, entity_cols=entity,
                            max_prior_neighbors=MAX_PRIOR_NEIGHBORS)
    check(3 + len(gcfg.entity_cols) == EXPECTED_EDGE_DIM,
          f"{arm}: edge_dim would be {3+len(gcfg.entity_cols)}, expected {EXPECTED_EDGE_DIM}. "
          f"Single-relation arms need the ('uid','__none__') placeholder")

    d = outdir / arm
    # every one of these is opened in append mode by the runner, and an existing
    # foldckpt.csv additionally triggers the resume path and SKIPS folds.
    if d.exists() and not resume:
        stale = [f for f in APPEND_FILES if (d / f).exists()]
        if stale and not force:
            die(f"{d} already holds {stale}. These are append-mode files and foldckpt.csv "
                f"would make the runner skip folds. Pass --resume to continue this run, "
                f"or --force to clear them and start the arm over")
        for f in stale:
            (d / f).unlink()
            print(f"    cleared stale {f}")
    d.mkdir(parents=True, exist_ok=True)
    lock = acquire_arm_lock(d)

    print(f"\n  === {arm} ===  entity={entity} max_agg={max_agg} layers={layers}")
    t0 = time.time()
    res = run_date_gnn_fold(
        df, num, [], folds, gcfg, mcfg, tcfg,
        target_col=cfg.target_col, time_col=cfg.time_col, alert_rate=cfg.alert_rate,
        evaluation=B["evaluation"], amount_col=cfg.amount_col,
        checkpoint_path=str(d / "foldckpt.csv"),
        val_curve_path=str(d / "val_curves.csv"),
        scores_path=str(d / "scores.csv"),
        weights_dir=str(d / "weights") if save_weights else None,
    )
    hours = (time.time() - t0) / 3600
    fp = fingerprint(tcfg, mcfg, gcfg)
    res["config"] = arm; res["config_fingerprint"] = fp
    res.to_csv(d / "per_fold.csv", index=False)

    check(len(res) == EXPECTED_N_FOLDS,
          f"{arm}: {len(res)} folds in results, expected {EXPECTED_N_FOLDS}. Resume may have skipped some")
    check(set(res.n_features) == {EXPECTED_N_FEATURES}, f"{arm}: wrong feature count")
    check((res.epochs_used < tcfg.epochs).all(),
          f"{arm}: a fold hit the {tcfg.epochs} epoch ceiling, so early stopping never bound")
    sc = pd.read_csv(d / "scores.csv")
    check(len(sc) == EXPECTED_SCORED_ROWS, f"{arm}: {len(sc):,} scored rows, expected {EXPECTED_SCORED_ROWS:,}")
    check(int(sc.y.sum()) == EXPECTED_POSITIVES, f"{arm}: {int(sc.y.sum()):,} positives, expected {EXPECTED_POSITIVES:,}")
    vc = pd.read_csv(d / "val_curves.csv")
    check(vc.test_week.nunique() == EXPECTED_N_FOLDS,
          f"{arm}: val_curves covers {vc.test_week.nunique()} folds, expected {EXPECTED_N_FOLDS}")

    # Hours must accumulate across segments, or a resumed arm reports only the time of
    # its final fold and looks 10x cheaper than it was.
    prev = {}
    if (d / "summary.json").exists():
        try: prev = json.load(open(d / "summary.json"))
        except Exception: prev = {}
    segments = list(prev.get("hours_segments", []))
    if prev and not segments and prev.get("hours"):
        segments = [prev["hours"]]                  # summary written before segments existed
    segments.append(round(hours, 2))

    e = res.epochs_used
    s = dict(arm=arm, mean_ap=float(res.ap.mean()), sd_ap_across_folds=float(res.ap.std(ddof=1)),
             min_ap=float(res.ap.min()), max_ap=float(res.ap.max()),
             mean_p_at_k=float(res.precision_at_k.mean()),
             mean_r_at_k=float(res.recall_at_k.mean()),
             epochs_min=int(e.min()), epochs_median=int(e.median()), epochs_max=int(e.max()),
             hours=round(sum(segments), 2), hours_segments=segments,
             segments=len(segments), resumed=bool(resume and len(segments) > 1),
             paper_ap=paper_ap, seed=int(tcfg.seed),
             delta_vs_paper=round(float(res.ap.mean()) - paper_ap, 4),
             config_fingerprint=fp, val_curve_rows=len(vc),
             weights_saved=bool(save_weights))
    print(f"    mean AP {s['mean_ap']:.4f}   paper {paper_ap:.4f}   delta {s['delta_vs_paper']:+.4f}")
    print(f"    epochs {s['epochs_min']}/{s['epochs_median']}/{s['epochs_max']}   "
          f"curves {len(vc):,} rows over {vc.test_week.nunique()} folds   {hours:.1f}h")
    if len(segments) > 1:
        print(f"    {len(segments)} segments, {sum(segments):.1f}h total: {segments}")
    json.dump(s, open(d / "summary.json", "w"), indent=1)
    lock.unlink(missing_ok=True)
    return res, s


# --------------------------------------------------------------------------------------
def derive(out, arms_run, B):
    """Budget sweep and paired tests, from the newly written scores."""
    from scipy.stats import wilcoxon
    dv = out / "derived"; dv.mkdir(parents=True, exist_ok=True)
    pf = {a: pd.read_csv(out / "canonical" / a / "per_fold.csv").sort_values("test_week")
          for a in arms_run}

    # the tuned tree is not re-run. Locate its saved per-fold and per-transaction scores.
    xgb_pf = xgb_scores = None
    # skip anything inside an output folder: derive() copies the tree's scores into
    # derived/, so a previous rerun would otherwise shadow the original.
    for p in sorted(Path(B["search_root"]).rglob("xgb_11_tuned.csv")):
        if out in p.parents or "rerun_" in str(p): continue
        xgb_pf = pd.read_csv(p).sort_values("test_week"); print(f"  tree per-fold: {p}"); break
    for p in sorted(Path(B["search_root"]).rglob("scores_xgb_11_tuned.csv")):
        if out in p.parents or "rerun_" in str(p): continue
        xgb_scores = p; print(f"  tree scores  : {p}"); break

    rows = []
    def pair(a_name, a, b_name, b):
        # The tie rule and the exact test are the ones the manuscript declares in Table 7's
        # note: differences below 1e-12 are numerical ties, dropped before the test.
        # Counting them as wins, and letting SciPy fall back to the normal approximation
        # when tied ranks are present, is what put "9 of 17" and p = 0.963 for the depth
        # variant into manuscript v6e. The depth arm ties with main at week 24 by 2e-16.
        d = a.ap.values - b.ap.values
        kept = d[np.abs(d) > 1e-12]
        st, p = wilcoxon(kept, method="exact")
        rows.append(dict(contrast=f"{a_name} - {b_name}", delta=round(d.mean(), 4),
                         wins=int((kept > 0).sum()), losses=int((kept < 0).sum()),
                         ties=int(len(d) - len(kept)), n=len(d), p=float(p)))

    if xgb_pf is not None and "main" in pf: pair("main", pf["main"], "xgb_tuned", xgb_pf)
    if xgb_pf is not None and "uid_only" in pf: pair("uid_only", pf["uid_only"], "xgb_tuned", xgb_pf)
    if {"main", "uid_only"} <= pf.keys(): pair("main", pf["main"], "uid_only", pf["uid_only"])
    for v in ("maxagg", "depth3"):
        if {"main", v} <= pf.keys(): pair(v, pf[v], "main", pf["main"])
    if rows:
        pd.DataFrame(rows).to_csv(dv / "paired_tests.csv", index=False)
        print("\n  paired tests -> derived/paired_tests.csv")
        print(pd.DataFrame(rows).to_string(index=False))

    # budget sweep expects these two filenames under a search root
    if "main" in arms_run and xgb_scores is not None:
        shutil.copy(out / "canonical" / "main" / "scores.csv", dv / "scores_gnn_11_converged.csv")
        shutil.copy(xgb_scores, dv / "scores_xgb_11_tuned.csv")
        script = Path(__file__).parent / "src" / "budget_sweep_11.py"
        print("\n  regenerating the alert-budget sweep ...")
        r = subprocess.run([sys.executable, str(script), str(dv), str(dv)],
                           capture_output=True, text=True)
        print("   ", (r.stdout or r.stderr).strip().replace("\n", "\n    ")[:900])
        if r.returncode != 0:
            print(f"\n  WARNING  budget sweep exited {r.returncode}. Table 5 and Figure 9 were "
                  f"NOT regenerated. Re-run src/budget_sweep_11.py against {dv} by hand.")
    else:
        print("\n  budget sweep skipped, needs the main arm and scores_xgb_11_tuned.csv")
    return pf, xgb_pf


def comparison_report(out, summaries, pf, xgb_pf, repeats_sm):
    """Old versus new for every number the paper reports, plus what goes stale."""
    L = ["# Canonical re-run, comparison with the published values", "",
         "Every figure below is recomputed from this session's outputs.", "",
         "## Arm means", "", "| Arm | Paper | This run | Delta |", "|---|---|---|---|"]
    for s in summaries:
        L.append(f"| {s['arm']} | {s['paper_ap']:.4f} | {s['mean_ap']:.4f} | {s['delta_vs_paper']:+.4f} |")
    L.append(f"| xgb_tuned (not re-run) | {XGB_BASELINE_AP:.4f} | {XGB_BASELINE_AP:.4f} | 0.0000 |")

    g = {s["arm"]: s["mean_ap"] for s in summaries}
    L += ["", "## Derived quantities the paper states", "",
          "| Quantity | Paper | This run |", "|---|---|---|"]
    if "main" in g:
        L.append(f"| Advantage over tuned tree | 0.207 | {g['main']-XGB_BASELINE_AP:.3f} |")
    if "uid_only" in g:
        L.append(f"| Cardholder structure | 0.137 | {g['uid_only']-XGB_BASELINE_AP:.3f} |")
    if {"main", "uid_only"} <= g.keys():
        L.append(f"| Merchant relation | 0.070 | {g['main']-g['uid_only']:.3f} |")
    if {"main", "maxagg"} <= g.keys():
        L.append(f"| Max-pool variant | -0.019 | {g['maxagg']-g['main']:+.3f} |")
    if {"main", "depth3"} <= g.keys():
        L.append(f"| Third layer | +0.014 | {g['depth3']-g['main']:+.3f} |")
    m = next((s for s in summaries if s["arm"] == "main"), None)
    if m:
        L.append(f"| Epochs, min/median/max | 52/77/144 | "
                 f"{m['epochs_min']}/{m['epochs_median']}/{m['epochs_max']} |")

    if repeats_sm is not None and len(repeats_sm) > 1:
        a = repeats_sm.mean_ap.values
        L += ["", "## Run-to-run variation", "",
              f"{len(a)} executions of the main arm, identical configuration except the "
              f"seed ({', '.join(str(x) for x in repeats_sm.seed)}), so this bounds "
              f"initialisation and batch-order noise rather than GPU jitter alone.", "",
              f"- mean {a.mean():.4f}, sd **{a.std(ddof=1):.4f}**",
              f"- range {a.min():.4f} to {a.max():.4f}, spread {a.max()-a.min():.4f}", "",
              "Quote the standard deviation. This replaces the two-run 0.010 figure, which "
              "Section 5.4 could only describe as a difference rather than an estimate."]
    else:
        L += ["", "## Run-to-run variation", "",
              "**Not estimated.** No repeats were requested, so Section 5.4 still has no "
              "basis for comparing architecture effects against noise."]

    L += ["", "## What must be updated in the manuscript", ""]
    moved = [s for s in summaries if abs(s["delta_vs_paper"]) >= 0.001]
    if not moved:
        L.append("No arm moved by 0.001 or more. Every reported value stands, and the only "
                 "change is provenance, since all artefacts now come from one execution.")
    else:
        L.append("These arms moved, so every number derived from them is stale:")
        L.append("")
        for s in moved:
            L.append(f"- **{s['arm']}** {s['paper_ap']:.4f} to {s['mean_ap']:.4f}")
        L += ["", "Affected: Table 4, Table 5, Figure 4, Figure 8, Section 5.4, the "
              "alert-budget sweep in Table 5 and Figure 9, the twelve-transactions-per-week "
              "figure, and the Abstract, Highlights and Conclusion.",
              "", "The protocol study of Section 5.3 is a separate set of executions on 16 "
              "matched weeks and is **not** affected."]
    L += ["", "Figure 7(b) should now be redrawn from `canonical/main/val_curves.csv`, which "
          "belongs to the same execution as Table 4. That closes the provenance gap.", ""]
    (out / "COMPARISON.md").write_text("\n".join(L), encoding="utf8")
    print(f"\n  wrote COMPARISON.md")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--prepared", type=Path, default=None,
                    help="folder holding sparkov_clean_v1.csv and column_manifest.json. "
                         "Found by searching upward from the repo if omitted")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder. Default is a NEW timestamped folder under results/, "
                         "so nothing existing is ever overwritten")
    ap.add_argument("--tag", default="", help="readable suffix on the default folder name")
    ap.add_argument("--arms", default="main,uid_only,maxagg,depth3",
                    help="comma-separated subset of " + ",".join(ARMS))
    ap.add_argument("--repeats", type=int, default=0,
                    help="extra executions of the MAIN arm only, for the variance estimate")
    ap.add_argument("--search-root", type=Path, default=None,
                    help="READ-ONLY tree searched for xgb_11_tuned.csv and its scores, which "
                         "are not re-run. Defaults to this repository's results/ folder")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="finish an EXISTING --out folder, training only missing folds. "
                         "Idempotent: absent folder starts fresh, complete folder just "
                         "rebuilds the summary")
    ap.add_argument("--check-only", action="store_true",
                    help="with --resume, report what is missing and exit. Writes nothing")
    ap.add_argument("--save-weights", action="store_true",
                    help="write the selected weights for every fold to <arm>/weights/. Adds "
                         "roughly 20 MB per arm. Off by default: the reported results need only "
                         "the scores, and weights cannot answer a question that changes the "
                         "feature set, which requires retraining")
    ap.add_argument("--force", action="store_true",
                    help="permit writing into a folder that already holds a run, clearing "
                         "the arm's append-mode files. Use --resume to continue instead")
    args = ap.parse_args()

    # --check-only inspects an existing output folder and writes nothing. It must work for
    # someone who has cloned the repository but not downloaded the 255 MB corpus, so the
    # canonical week list is taken from the folder's own fold_composition.csv when possible
    # and the prepared frame is never touched.
    offline_check = False
    if args.check_only and args.out is not None:
        fc = Path(args.out) / "fold_composition.csv"
        if fc.exists():
            offline_check = True
    args.prepared = None if offline_check else find_prepared(args.prepared)

    # a new folder every time unless the caller names one
    if args.out is None:
        check(not args.resume, "--resume needs --out naming the folder to finish")
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        name = f"rerun_{stamp}" + (f"_{args.tag}" if args.tag else "")
        args.out = here / "results" / name
    args.out = Path(args.out).resolve()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms: check(a in ARMS, f"unknown arm {a!r}. Choose from {list(ARMS)}")
    arms = [a for a in ARMS if a in arms]           # canonical order, whatever was typed

    # --resume on an absent folder is simply a fresh run. That is what makes the notebook
    # cell idempotent: one command covers start, continue and no-op.
    fresh = not (args.out / "canonical").exists()
    if args.resume and fresh:
        print("  (--resume: no existing run here, starting a fresh one)")
        args.resume = False

    print("TRACE-GNN canonical re-run" + (", RESUME" if args.resume else "") + "\n" + "=" * 74)
    print(f"  output folder : {args.out}")
    print(f"  arms          : {', '.join(arms)}   repeats of main: {args.repeats}")
    print(f"  alloc conf    : {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    if (args.out / "canonical").exists() and not args.force and not args.dry_run \
            and not args.resume and not args.check_only:
        die(f"{args.out / 'canonical'} already holds a run. The canonical run is what the "
            f"paper cites, so it is not overwritten by accident. Pass --resume to finish "
            f"it, choose a different --out, or pass --force if you really mean to redo it")

    if offline_check:
        # inspect the folder only: no corpus, no torch, no graph build
        rep = pd.read_csv(args.out / "fold_composition.csv")
        all_weeks = sorted(int(w) for w in rep[rep.kept]["test_week"])
        print(f"  mode          : offline check, reading {args.out.name}/fold_composition.csv")
        print(f"\n  canonical folds: {len(all_weeks)} weeks {all_weeks}\n\n  state on disk")
        broken = []
        for a in arms:
            st = arm_state(args.out / "canonical" / a, all_weeks)
            print(f"    {a:9} {'COMPLETE' if st['complete'] else 'missing ' + str(st['missing'])}")
            for k, v in st["detail"].items():
                print(f"                {k:16} {v}")
            if not st["consistent"]:
                broken.append(a)
        if broken:
            die(f"append-mode files disagree for {broken}. foldckpt.csv is the resume "
                f"authority, so a differing fold set means a fold was half-written")
        todo = [a for a in arms
                if not arm_state(args.out / "canonical" / a, all_weeks)["complete"]]
        print(f"\n  check only, nothing written. Would train: "
              f"{todo or 'nothing, all arms complete'}"
              + (f", plus {args.repeats} repeats of main" if args.repeats else "") + "\n")
        return

    # validate inputs before creating anything, so a bad path leaves no empty folder behind
    check(args.prepared.exists(), f"--prepared {args.prepared} does not exist")
    check((args.prepared / "sparkov_clean_v1.csv").exists(),
          f"sparkov_clean_v1.csv not found in {args.prepared}. Build it with src/prepare_sparkov.py")
    search_root = Path(args.search_root).resolve() if args.search_root else (here / "results")
    check(search_root.exists(), f"search root {search_root} does not exist. Pass --search-root")
    check(search_root != args.out and args.out not in search_root.parents,
          "the read-only search root must not sit inside the output folder")
    print(f"  reading (never writing) the baseline from: {search_root}")

    B = build(args)
    B["search_root"] = search_root

    # the tuned tree is not re-run, so fail now rather than after the training
    for fn in ("xgb_11_tuned.csv", "scores_xgb_11_tuned.csv"):
        hits = [q for q in sorted(search_root.rglob(fn))
                if args.out not in q.parents and "rerun_" not in str(q)]
        check(bool(hits), f"{fn} not found under {search_root}. The paired tests and the "
                          f"budget sweep both need it, and it is not re-run")
        print(f"  baseline {fn}: {hits[0]}")

    if args.dry_run:
        print("\n  DRY RUN OK. Every assertion passed. No folder was created.")
        print("  Re-run without --dry-run to train.\n")
        return

    args.out.mkdir(parents=True, exist_ok=True)   # only once the inputs are known good
    canon = args.out / "canonical"
    all_weeks = sorted(int(w) for w, _, _ in B["folds"])

    # ---- what is already on disk -----------------------------------------------------
    states = {a: arm_state(canon / a, all_weeks) for a in arms}
    if args.resume or args.check_only:
        print("\n  state on disk")
        broken = []
        for a in arms:
            st = states[a]
            print(f"    {a:9} {'COMPLETE' if st['complete'] else 'missing ' + str(st['missing'])}")
            for k, v in st["detail"].items():
                print(f"                {k:16} {v}")
            if not st["consistent"]:
                broken.append(a)
        if broken:
            die(f"append-mode files disagree for {broken}. foldckpt.csv is the resume "
                f"authority, so scores.csv or val_curves.csv holding a different fold set "
                f"means a fold was half-written. Resuming would duplicate or lose rows. "
                f"Re-run those arms with --force, or into a new folder")

    # honoured with or without --resume: someone asking to check must never start training
    if args.check_only:
        todo = [a for a in arms if not states[a]["complete"]]
        print(f"\n  check only, nothing written. Would train: "
              f"{todo or 'nothing, all arms complete'}"
              + (f", plus {args.repeats} repeats of main" if args.repeats else "") + "\n")
        return

    # ---- manifest, extended with one entry per execution segment ---------------------
    man_path = args.out / "run_manifest.json"
    man = json.load(open(man_path)) if (args.resume and man_path.exists()) \
        else environment_manifest(args.prepared, B["cfg"].clean_csv_name)
    man["experiment"] = B["facts"]
    man.setdefault("arms", {}).update({a: str(ARMS[a]) for a in arms})
    man["expected"] = {"train": EXPECTED_TRAIN, "model": EXPECTED_MODEL}
    seg = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "mode": "resume" if args.resume else "fresh", "device": args.device,
           "arms": arms, "repeats": args.repeats,
           "already_complete": [a for a in arms if states[a]["complete"]],
           "to_train": {a: states[a]["missing"] for a in arms if not states[a]["complete"]}}
    man.setdefault("execution_segments", []).append(seg)
    if len(man["execution_segments"]) > 1:
        man["provenance_note"] = (
            "This folder was produced in more than one execution. A resumed fold starts "
            "from a fresh RNG state rather than the state a continuous run would have "
            "reached, so it is the same configuration and seed drawn at a different point "
            "in the stream. See execution_segments for what each segment trained.")
    json.dump(man, open(man_path, "w"), indent=1, default=str)
    B["report"].to_csv(args.out / "fold_composition.csv", index=False)
    print("\n  wrote run_manifest.json and fold_composition.csv")

    summaries = []
    for a in arms:
        if args.resume and states[a]["complete"]:
            print(f"\n  === {a} === already complete, skipping")
            continue
        _, s = one_arm(a, canon, B, args.force, resume=args.resume,
                       save_weights=args.save_weights)
        summaries.append(s)

    # ---- canonical_summary.csv is rebuilt from every summary.json on disk, not just
    # the arms this process trained, so a resumed run still writes a complete table ----
    summaries = []
    for a in ARMS:
        sj = canon / a / "summary.json"
        if sj.exists():
            summaries.append(json.load(open(sj)))
    check(bool(summaries), "no summary.json written for any arm")
    pd.DataFrame(summaries).to_csv(args.out / "canonical_summary.csv", index=False)
    print(f"\n  wrote canonical_summary.csv over {len(summaries)} arm(s)")

    repeats_sm = None
    if args.repeats:
        main_sum = next((s for s in summaries if s["arm"] == "main"), None)
        check(main_sum is not None, "repeats compare against the main arm, which is not done")
        rs = [main_sum]
        for i in range(1, args.repeats + 1):
            rdir = args.out / "repeats" / f"repeat_{i:02d}"
            st = arm_state(rdir / "main", all_weeks)
            check(st["consistent"], f"repeat_{i:02d} has inconsistent append files")
            if st["complete"] and (rdir / "main" / "summary.json").exists():
                print(f"\n  === repeat_{i:02d} === already complete, skipping")
                rs.append(json.load(open(rdir / "main" / "summary.json")))
                continue
            _, s = one_arm("main", rdir, B, args.force,
                           seed=EXPECTED_TRAIN["seed"] + i,
                           resume=args.resume or st["started"],
                           save_weights=args.save_weights)
            rs.append(s)
        repeats_sm = pd.DataFrame(rs)
        repeats_sm.to_csv(args.out / "repeat_summary.csv", index=False)
        # Bookkeeping only, and it runs after many hours of training. A failure here must
        # not abort before derive() and COMPARISON.md, which are the point of the run.
        try:
            pfs = [pd.read_csv(args.out / ("canonical" if i == 0 else f"repeats/repeat_{i:02d}")
                               / "main" / "per_fold.csv").sort_values("test_week").ap.values
                   for i in range(len(rs))]
            v = np.std(np.vstack(pfs), axis=0, ddof=1)
            pd.DataFrame({"test_week": sorted(int(w) for w, _, _ in B["folds"]),
                          "sd_across_runs": v}).to_csv(args.out / "per_fold_variation.csv",
                                                       index=False)
            print(f"\n  per-fold sd across {len(rs)} runs: median {np.median(v):.4f}, "
                  f"max {v.max():.4f}")
        except Exception as exc:
            print(f"\n  WARNING  per_fold_variation.csv not written ({exc}). The training is "
                  f"safe on disk and repeat_summary.csv holds the arm means; re-run with "
                  f"--resume to regenerate it.")

    # derive() pairs arms fold by fold, so it needs every arm complete. Running it on a
    # partial set would silently drop contrasts from paired_tests.csv.
    done_arms = [s["arm"] for s in summaries]
    incomplete = [a for a in ARMS if a not in done_arms]
    if incomplete:
        print(f"\n  derived/ and COMPARISON.md NOT rebuilt: {incomplete} still missing.")
        print(f"  Finish them, then:  python {Path(__file__).name} --resume --out {args.out}")
    else:
        pf, xgb_pf = derive(args.out, done_arms, B)
        comparison_report(args.out, summaries, pf, xgb_pf, repeats_sm)

    print("\n" + "=" * 74)
    print(f"  Everything written to: {args.out}")
    print("  No file outside that folder was modified.")
    nseg = len(man.get("execution_segments", []))
    if nseg > 1:
        print(f"  Produced over {nseg} execution segments, recorded in run_manifest.json.")
    else:
        print("  Every artefact for each arm comes from one execution.")
    if not (args.out / "repeat_summary.csv").exists():
        print("  NOTE  no repeats, so run-to-run variation is NOT estimated. Section 5.4")
        print("        has no noise baseline to compare architecture effects against.")
    if not incomplete:
        print(f"  Read {args.out / 'COMPARISON.md'} before touching the manuscript.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
