"""
protocol_v2 — data loading, preflight, and the ONE training call.

There is exactly one call to run_date_gnn_fold in this whole package, in train_arm()
below, and its first line is guards.assert_trainable(arm). Nothing else in protocol_v2
imports the training function. That is how P3_rolling stays untrainable.

src/protocol_runner.py is NOT used. It is a stale copy of the training loop and is now
marked deprecated. The acausal arms reach the same behaviour through the real function's
edge_builder / assert_causality parameters instead.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import guards
import provenance as P
import splits as S

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# protocol_v2 first, then the repo, then src — see config.ensure_import_path for why the
# order matters (src/config.py shadows protocol_v2/config.py otherwise). The earlier
# version of this loop skipped directories already on the path, so it could not correct
# an order that was already wrong.
C.ensure_import_path()


# --------------------------------------------------------------------------- memory
def free_gpu(device=None):
    """(free_bytes, total_bytes) on the target device, or (None, None)."""
    import torch
    if not torch.cuda.is_available():
        return None, None
    dev = torch.device(device or C.DEVICE)
    return torch.cuda.mem_get_info(dev)


def print_free_gpu(label="free at start"):
    """The one thing every notebook does automatically. Gives a number to compare the
    end-of-run cleanup against."""
    free, total = free_gpu()
    if free is None:
        print("  no CUDA device visible")
        return None, None
    print(f"  {label}: {free/1e9:.1f} GB of {total/1e9:.1f} GB on {C.DEVICE}")
    return free, total


# --------------------------------------------------------------------------- data
def find_prepared() -> Path:
    for c in C.PREPARED_CANDIDATES:
        if (Path(c) / "sparkov_clean_v1.csv").exists():
            return Path(c).resolve()
    for base in (C.REPO, C.REPO.parent, C.REPO.parent.parent):
        hits = list(Path(base).glob("**/sparkov_clean_v1.csv"))
        if hits:
            return hits[0].parent.resolve()
    raise FileNotFoundError("sparkov_clean_v1.csv not found. Build it with "
                            "src/prepare_sparkov.py, or set config.PREPARED_CANDIDATES.")


def load_frame(verbose: bool = True) -> dict:
    """Load the prepared frame, cut the 500,000-row slice, build the fold sets, and check
    every preflight number. Identical construction to rerun_canonical.build()."""
    from experiment_config import ExperimentConfig
    import temporal_folds, evaluation
    from run_experiment import prepare_frame
    from temporal_graph import GraphConfig
    from model import ModelConfig
    from run_date_gnn import TrainConfig, subsample_recent, filter_folds

    prepared = find_prepared()
    cfg = ExperimentConfig(
        prepared_dir=prepared, clean_csv_name="sparkov_clean_v1.csv",
        id_col="TransactionID", target_col="is_fraud", time_col="unix_time",
        week_col="week_idx", amount_col="amt", uid_col="uid",
        non_feature_cols=("uid",), min_train_weeks=C.MIN_TRAIN_WEEKS,
    )
    assert cfg.alert_rate == C.ALERT_RATE, (
        f"alert_rate {cfg.alert_rate} != {C.ALERT_RATE}")

    if verbose:
        print(f"  loading {prepared/'sparkov_clean_v1.csv'}")
    df, _manifest, feature_cols, uid_feats = prepare_frame(cfg)
    df = subsample_recent(df, cfg.week_col, C.EXPECT["slice_rows"]).reset_index(drop=True)

    e = C.EXPECT
    assert len(df) == e["slice_rows"], f"slice has {len(df):,}, expected {e['slice_rows']:,}"
    assert df[cfg.time_col].is_monotonic_increasing, "frame is not time-sorted"
    num = [c for c in feature_cols if c in C.NUMERIC] + uid_feats
    assert len(num) == e["n_features"], f"{len(num)} features, expected {e['n_features']}: {num}"

    sp = S.ensure_splits(df, cfg, temporal_folds, filter_folds, verbose=verbose)
    week_of_row = df[cfg.week_col].to_numpy()

    return dict(cfg=cfg, df=df, num=num, cat=[], splits=sp, week_of_row=week_of_row,
                evaluation=evaluation, prepared=prepared,
                TrainConfig=TrainConfig, ModelConfig=ModelConfig, GraphConfig=GraphConfig)


def configs_for(arm: str, B: dict):
    """The three config objects, built from config.py, which reads them from
    rerun_canonical.EXPECTED_TRAIN / EXPECTED_MODEL. Nothing is restated here."""
    tcfg = B["TrainConfig"](**C.TRAIN_KWARGS, device=C.DEVICE)
    for k, v in C.TRAIN_KWARGS.items():
        assert getattr(tcfg, k) == v, f"TrainConfig.{k}={getattr(tcfg,k)!r}, expected {v!r}"
    assert tcfg.epochs == 200, f"epochs={tcfg.epochs}; every protocol_v2 arm runs at 200"
    mcfg = B["ModelConfig"](n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG, **C.MODEL_KWARGS)
    gcfg = B["GraphConfig"](time_col=B["cfg"].time_col, entity_cols=C.ENTITY_COLS,
                            max_prior_neighbors=C.MAX_PRIOR_NEIGHBORS)
    assert 3 + len(gcfg.entity_cols) == C.EXPECT["edge_dim"]
    return tcfg, mcfg, gcfg


def edge_builder_for(arm: str):
    """Causal arms get None, which means run_date_gnn_fold uses build_temporal_edges
    exactly as Section 5.1 did. Acausal arms get the acausal builder and switch the
    causality assertion off, because assert_causal is designed to fail on their edges."""
    from acausal_graph import build_acausal_edges
    if C.ARM_SPEC[arm]["edges"] == "causal":
        return None, True
    return build_acausal_edges, False


# --------------------------------------------------------------------------- preflight
def graph_facts(B: dict, kind: str, verbose: bool = True) -> dict:
    """Build the graph once and print the facts the plan says to check before spending
    GPU time. Costs a few seconds of numpy and no GPU."""
    from temporal_graph import build_temporal_edges, assert_causal
    from acausal_graph import build_acausal_edges, count_future_edges
    df, cfg = B["df"], B["cfg"]
    gcfg = B["GraphConfig"](time_col=cfg.time_col, entity_cols=C.ENTITY_COLS,
                            max_prior_neighbors=C.MAX_PRIOR_NEIGHBORS)
    t0 = time.time()
    if kind == "causal":
        ei, dt, rel = build_temporal_edges(df.reset_index(drop=True), gcfg)
        ties = assert_causal(ei, df[cfg.time_col].to_numpy())
        facts = {"kind": "causal", "edges": int(ei.shape[1]),
                 "edges_per_node": round(ei.shape[1] / len(df), 2),
                 "contemporaneous_edges": int(ties), "future_edges": 0}
        want_e, want_pn = C.EXPECT["edges_causal"], C.EXPECT["edges_per_node_causal"]
        extra = [("contemporaneous_edges", facts["contemporaneous_edges"],
                  C.EXPECT["contemporaneous_edges_causal"])]
    else:
        ei, dt, rel = build_acausal_edges(df.reset_index(drop=True), gcfg)
        diag = count_future_edges(ei, df[cfg.time_col].to_numpy())
        facts = {"kind": "acausal", "edges": int(ei.shape[1]),
                 "edges_per_node": round(ei.shape[1] / len(df), 2),
                 "future_edges": diag["future_edges"],
                 "future_frac": diag["future_frac"]}
        want_e, want_pn = C.EXPECT["edges_acausal"], C.EXPECT["edges_per_node_acausal"]
        extra = []
    facts["build_seconds"] = round(time.time() - t0, 1)

    bad = []
    if facts["edges"] != want_e:
        bad.append(f"edges {facts['edges']:,} != expected {want_e:,}")
    if abs(facts["edges_per_node"] - want_pn) > 0.01:
        bad.append(f"edges/node {facts['edges_per_node']} != expected {want_pn}")
    for name, got, want in extra:
        if got != want:
            bad.append(f"{name} {got} != expected {want}")

    if verbose:
        print(f"  {kind} graph: {facts['edges']:,} edges, {facts['edges_per_node']}/node, "
              f"built in {facts['build_seconds']}s")
        if kind == "causal":
            print(f"    {facts['contemporaneous_edges']} contemporaneous, 0 future "
                  f"(assert_causal passed)")
        else:
            print(f"    {facts['future_edges']:,} edges point FUTURE->current "
                  f"({facts['future_frac']*100:.1f}%) — this IS the graph leak, on purpose")
    if bad:
        raise AssertionError("graph facts do not match the canonical run:\n    " +
                             "\n    ".join(bad))
    del ei, dt, rel
    return facts


def preflight(arm: str, B: dict, verbose: bool = True) -> dict:
    """Everything that must be true before any GPU time is spent. Raises on anything
    wrong. Fixing a setting costs minutes now and ten hours later."""
    import torch
    guards.assert_trainable(arm)
    tcfg, mcfg, gcfg = configs_for(arm, B)
    sp, e = B["splits"], C.EXPECT
    folds = S.folds_for(arm, sp)

    checks = [
        ("rows in the slice", len(B["df"]), e["slice_rows"]),
        ("features", len(B["num"]), e["n_features"]),
        ("folds for this arm", len(folds), 17 if C.ARM_SPEC[arm]["split"] == "p3" else 1),
        ("epochs", tcfg.epochs, 200),
        ("seed", tcfg.seed, 42),
        ("use_swa", tcfg.use_swa, False),
        ("layers", mcfg.n_layers, C.N_LAYERS),
        ("neighbour cap K", gcfg.max_prior_neighbors, C.MAX_PRIOR_NEIGHBORS),
    ]
    if verbose:
        print(f"preflight — {arm}")
        for name, got, want in checks:
            print(f"  {'OK ' if got == want else 'BAD'}  {name:22} {got!r:>12}  "
                  f"expected {want!r}")
    bad = [f"{n}={g!r} expected {w!r}" for n, g, w in checks if g != w]

    # disk
    free_gb = shutil.disk_usage(C.PROTOCOL_V2).free / 1e9
    if verbose:
        print(f"  {'OK ' if free_gb >= e['min_free_disk_gb'] else 'BAD'}  "
              f"{'free disk':22} {free_gb:11.1f}G  need {e['min_free_disk_gb']}G")
    if free_gb < e["min_free_disk_gb"]:
        bad.append(f"only {free_gb:.1f} GB free on disk")

    # projected peak VRAM on this arm's largest fold, by sampling one batch on the CPU
    proj = None
    proj_error = None
    try:
        from run_date_gnn import GraphCSR, sample_subgraph, build_edge_features
        builder, _ = edge_builder_for(arm)
        if builder is None:
            from temporal_graph import build_temporal_edges as builder
        ei, dt, rel = builder(B["df"].reset_index(drop=True), gcfg)
        ea = build_edge_features(B["df"], ei, dt, rel, len(gcfg.entity_cols),
                                 B["cfg"].amount_col)
        csr = GraphCSR(ei, ea, len(B["df"]))
        big = max(folds, key=lambda f: len(f[1]))
        rng = np.random.default_rng(0)
        seeds = rng.choice(np.asarray(big[1]),
                           size=min(tcfg.batch_size, len(big[1])), replace=False)
        _, sei, _, _ = sample_subgraph(seeds, mcfg.n_layers, csr)
        # Raised from 900 to 2400 before A4, on measurement rather than theory. The 900
        # figure under-projected every arm actually run:
        #     A1 P2_cutoff  projected 1.5 GB   peak 5.9 GB (card-level)   ~3.9x
        #     A2 P1_causal  projected 2.0 GB   peak 8.60 GB (torch)       ~4.3x
        #     A3 P1_leaky   projected 2.0 GB   peak 8.63 GB (torch)       ~4.3x
        # 2400 restores most of that margin; it does NOT fully close it (a ratio-matched
        # value would be ~3900), so the projection still reads low. It is a floor, not a
        # forecast. A4 is the seventeen-fold arm where a bad margin costs nine hours.
        BYTES_PER_EDGE_PER_LAYER = 2400         # was 900, measured on an RTX 6000 Ada
        proj = int(sei.shape[1]) * mcfg.n_layers * BYTES_PER_EDGE_PER_LAYER / 1e9
        del csr, ei, ea, dt, rel
    except Exception as exc:
        # An estimate failing must not block a run — that is how rerun_canonical.py
        # treats it too. But it must be loud, and it must end up in the manifest, because
        # this is the one check standing between A4 and an out-of-memory death nine hours
        # in. Silence here would have made preflight's promise untrue without saying so.
        proj_error = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  !!  PEAK-VRAM CHECK DID NOT RUN: {proj_error}")
            print(f"  !!  This arm is starting WITHOUT the memory headroom check. Confirm "
                  f"by eye that {C.DEVICE} has room, or stop and fix the estimator.")

    free_b, total_b = free_gpu()
    if proj is not None and free_b is not None:
        free_gb_gpu = free_b / 1e9
        need = proj * (1 + e["vram_headroom_frac"])
        ok = need <= free_gb_gpu
        if verbose:
            print(f"  {'OK ' if ok else 'BAD'}  {'projected peak VRAM':22} "
                  f"{proj:10.1f}G  +{int(e['vram_headroom_frac']*100)}% = {need:.1f}G, "
                  f"{free_gb_gpu:.1f}G free on {C.DEVICE}")
        if not ok:
            bad.append(f"projected peak {proj:.0f} GB (+15% = {need:.0f} GB) exceeds the "
                       f"{free_gb_gpu:.0f} GB free on {C.DEVICE}. Free the card, or run "
                       f"the arm on the other GPU.")

    if bad:
        raise AssertionError("PREFLIGHT FAILED — do not start this arm:\n  - " +
                             "\n  - ".join(bad))
    if verbose:
        print(f"  preflight clean for {arm}")
    return {"checks": dict((n, g) for n, g, _ in checks), "free_disk_gb": round(free_gb, 1),
            "projected_peak_vram_gb": round(proj, 1) if proj else None,
            "vram_check_skipped": proj_error,
            "free_vram_gb": round(free_b / 1e9, 1) if free_b else None}


# --------------------------------------------------------------------------- the run
def arm_dir(arm: str) -> Path:
    return guards.assert_writable(C.ARMS_DIR / guards.assert_trainable(arm))


def reconcile_after_crash(arm: str, verbose: bool = True) -> dict:
    """Bring scores.csv and val_curves.csv back into agreement with foldckpt.csv.

    run_date_gnn_fold writes the checkpoint LAST, so the checkpoint is the definition of
    which folds finished. A crash between the bulk writes and the checkpoint write leaves
    rows for a fold that is not checkpointed; on resume that fold is retrained and its
    rows appended a second time. This drops the orphans first, so the resumed arm ends
    with exactly one set of rows per fold.

    Called automatically by train_arm(resume=True). Rewrites via a temp file and an
    atomic replace, so an interruption here cannot leave a half-written CSV. The originals
    are copied to <name>.pre_reconcile once, so nothing is destroyed.
    """
    d = arm_dir(arm)
    ck = d / "foldckpt.csv"
    if not ck.exists():
        return {"arm": arm, "reconciled": False, "reason": "no checkpoint yet"}

    if ck.stat().st_size == 0:
        raise FileExistsError(
            f"{ck} is empty (0 bytes). A checkpoint with no header cannot say which folds "
            f"finished, so nothing can be reconciled against it and a resume would be a "
            f"guess. Move the arm directory aside and start over.")
    done_df = pd.read_csv(ck)
    if len(done_df) == 0:
        raise FileExistsError(
            f"{ck} has a header but no rows, so no fold ever completed — yet the arm "
            f"directory holds output. Reconciling against it would delete every row in "
            f"scores.csv and val_curves.csv. Refusing. Move the arm directory aside and "
            f"start over.")
    done = {int(w) for w in done_df["test_week"].tolist()}

    report = {"arm": arm, "reconciled": True, "checkpointed_folds": sorted(done),
              "last_fold": int(done_df["test_week"].iloc[-1]), "dropped": {}}
    for name, col in (("scores.csv", "fold"), ("val_curves.csv", "test_week")):
        p = d / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        keep = df[col].astype(int).isin(done)
        n_drop = int((~keep).sum())
        report["dropped"][name] = n_drop
        if n_drop == 0:
            continue
        orphan_folds = sorted(set(df.loc[~keep, col].astype(int)))
        # numbered backups, not a single one. A second crash-and-reconcile would
        # otherwise find the backup already there, skip it, and lose the second batch of
        # dropped rows for good.
        n = 1
        while (d / f"{name}.pre_reconcile.{n}").exists():
            n += 1
        backup = guards.assert_writable(d / f"{name}.pre_reconcile.{n}")
        shutil.copy2(p, backup)
        tmp = guards.assert_writable(d / f"{name}.tmp")
        df[keep].to_csv(tmp, index=False)
        tmp.replace(guards.assert_writable(p))
        if verbose:
            print(f"  [reconcile] {name}: dropped {n_drop:,} row(s) for fold(s) "
                  f"{orphan_folds}, which the checkpoint does not list. Full copy kept at "
                  f"{backup.name}.")

    # weights/ needs no reconciling: fold_NN.pt is written per fold under a fixed name, so
    # retraining an orphaned fold overwrites its file rather than adding one. Said here so
    # the omission is a decision rather than an oversight.
    # rng_state.pt needs no reconciling either: it keeps one entry per fold, so entries
    # for uncheckpointed folds are simply ignored on restore. That is why it keeps all of
    # them instead of only the newest.
    if verbose and not any(report["dropped"].values()):
        print(f"  [reconcile] scores.csv and val_curves.csv already agree with the "
              f"{len(done)} checkpointed fold(s); nothing dropped")
    return report


def train_arm(arm: str, B: dict, only_weeks=None, resume: bool = False,
              save_weights=None, verbose: bool = True):
    """THE ONLY CALL TO run_date_gnn_fold IN protocol_v2.

    arm         must be one of config.TRAINABLE_ARMS. P3_rolling raises LockedArmError.
    only_weeks  restrict to these fold labels. Used by the A4 smoke run (weeks 8 and 9).
    resume      allow an existing foldckpt.csv to be continued. Without it, an existing
                checkpoint is an error, because a fresh run must never silently skip
                folds. Resuming also restores the saved RNG state, so a smoke run
                followed by a resume draws the same random stream as one uninterrupted
                run would have drawn.
    """
    guards.assert_trainable(arm)                       # <-- the lock, first line
    from run_date_gnn import run_date_gnn_fold

    tcfg, mcfg, gcfg = configs_for(arm, B)
    builder, assert_causality = edge_builder_for(arm)
    folds = S.folds_for(arm, B["splits"])
    if only_weeks is not None:
        want = {int(w) for w in only_weeks}
        folds = [f for f in folds if int(f[0]) in want]
        if not folds:
            raise ValueError(f"only_weeks={sorted(want)} matched no fold of {arm}")

    d = arm_dir(arm)
    d.mkdir(parents=True, exist_ok=True)
    ck = guards.assert_writable(d / "foldckpt.csv")
    rng_path = guards.assert_writable(d / "rng_state.pt")
    carryover = [ck,
                 guards.assert_writable(d / "scores.csv"),
                 guards.assert_writable(d / "val_curves.csv"),
                 rng_path,
                 guards.assert_writable(d / "weights")]

    if not resume:
        stale = [p.name for p in carryover if p.exists()]
        if stale:
            raise FileExistsError(
                f"{d} already holds {stale}. The CSVs are opened in append mode, "
                f"foldckpt.csv would make the runner skip folds, and rng_state.pt would be "
                f"read as the point to continue the random stream from. Pass resume=True to "
                f"continue this arm, or move the directory aside and start over. Nothing is "
                f"deleted for you.")

    reconcile = None
    if resume:
        reconcile = reconcile_after_crash(arm, verbose=verbose)

    if save_weights is None:
        save_weights = C.SAVE_WEIGHTS

    if verbose:
        which = sorted(int(f[0]) for f in folds)
        scope = f"SMOKE, folds {which}" if only_weeks else f"{len(folds)} fold(s)"
        builder_name = ("build_acausal_edges" if builder is not None
                        else "None -> build_temporal_edges, exactly as Section 5.1")
        print(f"\n=== {arm} ===   (NOT P3_rolling; that arm is locked and never trained)")
        print(f"  {C.ARM_SPEC[arm]['what']}")
        print(f"  scope        {scope}")
        print(f"  epochs       {tcfg.epochs}   seed {tcfg.seed}   device {tcfg.device}")
        print(f"  edge_builder {builder_name}")
        print(f"  assert_causality={assert_causality}   resume={resume}   "
              f"rng_state={rng_path.name}")
        print(f"  out          {d}")

    t0 = time.time()
    res = run_date_gnn_fold(
        B["df"], B["num"], B["cat"], folds, gcfg, mcfg, tcfg,
        target_col=B["cfg"].target_col, time_col=B["cfg"].time_col,
        alert_rate=B["cfg"].alert_rate, evaluation=B["evaluation"],
        amount_col=B["cfg"].amount_col,
        checkpoint_path=str(ck),
        val_curve_path=str(guards.assert_writable(d / "val_curves.csv")),
        scores_path=str(guards.assert_writable(d / "scores.csv")),
        weights_dir=str(guards.assert_writable(d / "weights")) if save_weights else None,
        edge_builder=builder,
        assert_causality=assert_causality,
        rng_state_path=str(rng_path),
    )
    hours = (time.time() - t0) / 3600
    if verbose:
        print(f"  {arm}: {len(res)} folds in the checkpoint after {hours:.2f} h")
    return res, hours


def finalise(arm: str, res, hours: float, B: dict, preflight_rec=None,
             graph_rec=None, verbose: bool = True) -> dict:
    """Write per_fold.csv, summary.json and the per-arm manifest — but only once the arm
    is complete. A smoke run leaves a partial file so nothing downstream mistakes it for
    a finished arm."""
    guards.assert_trainable(arm)
    d = arm_dir(arm)
    expected_folds = 17 if C.ARM_SPEC[arm]["split"] == "p3" else 1
    complete = len(res) == expected_folds

    res = res.copy()
    res["arm"] = arm
    out = d / ("per_fold.csv" if complete else "per_fold_PARTIAL.csv")
    res.to_csv(guards.assert_writable(out), index=False)

    if not complete:
        if verbose:
            print(f"  [partial] {len(res)}/{expected_folds} folds. Wrote {out.name}. "
                  f"per_fold.csv and summary.json are written only when the arm finishes.")
        return {"arm": arm, "complete": False, "folds_done": int(len(res))}

    sc = pd.read_csv(d / "scores.csv")
    vc = pd.read_csv(d / "val_curves.csv")
    e = C.EXPECT
    expect_rows = (e["p3_scored_rows"] if C.ARM_SPEC[arm]["split"] in ("p2", "p3")
                   else e["p1_test_rows"])
    problems = []
    if len(sc) != expect_rows:
        extra = len(sc) - expect_rows
        problems.append(
            f"{len(sc):,} scored rows, expected {expect_rows:,} ({extra:+,}). "
            + ("More rows than folds can account for usually means a fold's scores were "
               "appended twice after a crash. runner.reconcile_after_crash() drops "
               "orphaned rows on resume; if this arm was resumed some other way, rerun it."
               if extra > 0 else
               "Fewer rows than expected means a fold never wrote its scores."))
    if (res.epochs_used >= C.TRAIN_KWARGS["epochs"]).any():
        hit = sorted(res.loc[res.epochs_used >= C.TRAIN_KWARGS["epochs"], "test_week"])
        problems.append(f"fold(s) {hit} hit the {C.TRAIN_KWARGS['epochs']}-epoch ceiling, "
                        f"so early stopping never bound and the arm is truncated")
    if set(res.n_features) != {e["n_features"]}:
        problems.append(f"feature counts {sorted(set(res.n_features))}, expected "
                        f"{e['n_features']}")
    if vc.test_week.nunique() != expected_folds:
        problems.append(f"val_curves.csv covers {vc.test_week.nunique()} fold(s), expected "
                        f"{expected_folds}. A checkpointed fold whose curve was never "
                        f"written stays missing across a resume.")

    # Hours accumulate across segments, or a resumed arm reports only the time of its
    # final fold. But finalise can now RAISE after writing summary.json, so it may be run
    # again on the same results once the problem is fixed — and that second run must not
    # append a second segment for wall clock already counted. A segment is only appended
    # if this call actually trained something (hours > 0) or no summary exists yet.
    prev = {}
    if (d / "summary.json").exists():
        try:
            prev = json.loads((d / "summary.json").read_text())
        except Exception:
            prev = {}
    segments = list(prev.get("hours_segments", []))
    if hours > 0 or not segments:
        segments.append(round(hours, 2))

    s = {
        # complete means "all folds ran AND everything validated". A5 refuses to assemble
        # an arm whose summary says otherwise.
        "arm": arm, "complete": not problems,
        "what": C.ARM_SPEC[arm]["what"],
        "split": C.ARM_SPEC[arm]["split"], "edges": C.ARM_SPEC[arm]["edges"],
        "n_folds": int(len(res)),
        "mean_ap": float(res.ap.mean()),
        "sd_ap_across_folds": float(res.ap.std(ddof=1)) if len(res) > 1 else None,
        "mean_p_at_k": float(res.precision_at_k.mean()),
        "mean_r_at_k": float(res.recall_at_k.mean()),
        "epochs_min": int(res.epochs_used.min()),
        "epochs_median": int(res.epochs_used.median()),
        "epochs_max": int(res.epochs_used.max()),
        "epoch_budget": C.TRAIN_KWARGS["epochs"],
        "seed": C.TRAIN_KWARGS["seed"],
        "device": C.DEVICE,
        "hours": round(sum(segments), 2), "hours_segments": segments,
        "scored_rows": int(len(sc)), "positives": int(sc.y.sum()),
        "val_curve_rows": int(len(vc)),
        "weights_saved": (d / "weights").is_dir(),
        "problems": problems,
        "note_not_p3_rolling": "This is a NEW arm trained by protocol_v2. It is not, and "
                               "must not be confused with, the canonical P3_rolling run.",
    }
    json.dump(s, open(guards.assert_writable(d / "summary.json"), "w"), indent=1)

    man = {
        "arm": arm, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": s,
        "train_config": dict(C.TRAIN_KWARGS, device=C.DEVICE),
        "model_config": dict(C.MODEL_KWARGS, n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG),
        "graph_config": {"entity_cols": list(C.ENTITY_COLS),
                         "max_prior_neighbors": C.MAX_PRIOR_NEIGHBORS,
                         "builder": ("build_temporal_edges"
                                     if C.ARM_SPEC[arm]["edges"] == "causal"
                                     else "build_acausal_edges"),
                         "assert_causality": C.ARM_SPEC[arm]["edges"] == "causal"},
        "splits_manifest": json.loads(Path(S.MANIFEST).read_text()),
        "environment": P.environment_manifest(B["prepared"] / "sparkov_clean_v1.csv"),
        "environment_vs_canonical": P.compare_environment(
            P.environment_manifest(B["prepared"] / "sparkov_clean_v1.csv"), verbose=False),
        "source_hashes": P.source_hashes(),
        "preflight": preflight_rec, "graph_facts": graph_rec,
        "canonical_p3_lock": guards.assert_canonical_unchanged(verbose=False),
    }
    P.write_manifest(d / "arm_manifest.json", man)

    if verbose:
        print(f"  mean AP {s['mean_ap']:.4f}   epochs {s['epochs_min']}/"
              f"{s['epochs_median']}/{s['epochs_max']} of {s['epoch_budget']}   "
              f"{s['hours']:.2f} h")
        print(f"  wrote per_fold.csv, summary.json, arm_manifest.json in {d}")

    # Everything above is written FIRST, then this raises. An 8.8-hour arm must not be
    # thrown away because its validation failed — the results stay on disk, complete=False
    # in summary.json, and A5 will refuse to use them until the problem is dealt with.
    # This used to be a logged "problem" that A5 never read.
    if problems:
        raise AssertionError(
            f"{arm} FINISHED BUT DID NOT VALIDATE. Results are on disk in {d} and "
            f"summary.json records complete=False. Do not assemble with this arm.\n  - "
            + "\n  - ".join(problems))
    return s
