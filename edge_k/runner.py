"""
edge_k — data loading, preflight, and the ONE training call.

There is exactly one call to run_date_gnn_fold in this whole package, in train_arm()
below, and its first line is guards.assert_trainable(arm). Nothing else in edge_k imports
the training function. That is how the canonical and protocol_v2 arms stay untrainable,
and how the three scaffolded arms stay unrunnable.

src/protocol_runner.py is NOT used. It is a stale copy of the training loop, marked
deprecated. The edge-bias arm reaches its behaviour through the real function's model
config instead, and the K arms through GraphConfig, which already had the field.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import guards
import provenance as P
import status as ST

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    """Load the prepared frame, cut the 500,000-row slice, build the 17 rolling folds and
    check every preflight number. Identical construction to rerun_canonical.build(): every
    edge_k arm uses the canonical fold set, so there is no split to generate or reuse."""
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
    assert cfg.alert_rate == C.ALERT_RATE, f"alert_rate {cfg.alert_rate} != {C.ALERT_RATE}"

    if verbose:
        print(f"  loading {prepared/'sparkov_clean_v1.csv'}")
    df, _manifest, feature_cols, uid_feats = prepare_frame(cfg)
    df = subsample_recent(df, cfg.week_col, C.EXPECT["slice_rows"]).reset_index(drop=True)

    e = C.EXPECT
    assert len(df) == e["slice_rows"], f"slice has {len(df):,}, expected {e['slice_rows']:,}"
    assert df[cfg.time_col].is_monotonic_increasing, "frame is not time-sorted"
    num = [c for c in feature_cols if c in C.NUMERIC] + uid_feats
    assert len(num) == e["n_features"], f"{len(num)} features, expected {e['n_features']}: {num}"

    folds, fold_report = filter_folds(df, temporal_folds.make_folds(df, cfg),
                                      cfg.target_col, 5, 5)
    weeks = [int(w) for w, _, _ in folds]
    assert len(folds) == e["n_folds"], f"{len(folds)} folds, expected {e['n_folds']}"
    assert tuple(weeks) == e["test_weeks"], f"weeks {weeks} != {list(e['test_weeks'])}"
    scored = sum(len(te) for _, _, te in folds)
    assert scored == e["scored_rows"], f"{scored:,} scored rows, expected {e['scored_rows']:,}"

    if verbose:
        print(f"  {len(df):,} rows, {len(num)} features, {len(folds)} folds "
              f"(weeks {weeks[0]}-{weeks[-1]}), {scored:,} scored rows")

    return dict(cfg=cfg, df=df, num=num, cat=[], folds=folds, weeks=weeks,
                evaluation=evaluation, prepared=prepared,
                TrainConfig=TrainConfig, ModelConfig=ModelConfig, GraphConfig=GraphConfig)


# --------------------------------------------------------------------------- configs
def configs_for(arm: str, B: dict):
    """The three config objects. Everything comes from config.py, which reads it from
    rerun_canonical.EXPECTED_TRAIN / EXPECTED_MODEL. Nothing is restated here.

    ONE named change per arm. Every other field is asserted equal to the canonical run.
    """
    spec = C.ARM_SPEC[arm]

    # ---- training: identical for every arm, no exceptions
    tcfg = B["TrainConfig"](**C.TRAIN_KWARGS, device=C.DEVICE)
    for k, v in C.TRAIN_KWARGS.items():
        assert getattr(tcfg, k) == v, f"TrainConfig.{k}={getattr(tcfg,k)!r}, expected {v!r}"
    assert tcfg.epochs == 200, f"epochs={tcfg.epochs}; every edge_k arm runs at 200"
    assert tcfg.seed == 42
    assert tcfg.use_swa is False

    # ---- model: only no_edge_bias may differ, and only in use_edge_bias
    mcfg = B["ModelConfig"](n_layers=C.N_LAYERS, use_max_agg=C.USE_MAX_AGG,
                            use_edge_bias=spec["use_edge_bias"], **C.MODEL_KWARGS)
    for k, v in C.MODEL_KWARGS.items():
        assert getattr(mcfg, k) == v, f"ModelConfig.{k}={getattr(mcfg,k)!r}, expected {v!r}"
    assert mcfg.n_layers == C.N_LAYERS
    assert mcfg.use_max_agg is C.USE_MAX_AGG
    assert mcfg.use_edge_bias is spec["use_edge_bias"]
    if arm != "no_edge_bias":
        assert mcfg.use_edge_bias is True, (
            f"{arm} must keep the edge bias; only no_edge_bias removes it")

    # ---- graph: only the K arms and uid_K20 may differ
    gcfg = B["GraphConfig"](time_col=B["cfg"].time_col, entity_cols=spec["entity_cols"],
                            max_prior_neighbors=spec["k"])
    assert gcfg.max_prior_neighbors == spec["k"]
    assert gcfg.entity_cols == spec["entity_cols"]
    # edge_dim must be 5 in EVERY arm. uid_K20 keeps it at 5 through the "__none__"
    # placeholder, exactly as Section 5.2's uid_only arm does. A drop to 4 is a different
    # model and has invalidated a round of experiments before.
    assert 3 + len(gcfg.entity_cols) == C.EXPECT["edge_dim"], (
        f"edge_dim would be {3 + len(gcfg.entity_cols)}, must be {C.EXPECT['edge_dim']}")

    # ---- the one-change audit: exactly what this arm changes, and nothing else
    diffs = []
    if gcfg.max_prior_neighbors != C.CANONICAL_K:
        diffs.append(f"max_prior_neighbors {C.CANONICAL_K} -> {gcfg.max_prior_neighbors}")
    if gcfg.entity_cols != C.CANONICAL_ENTITY_COLS:
        diffs.append(f"entity_cols {C.CANONICAL_ENTITY_COLS} -> {gcfg.entity_cols}")
    if mcfg.use_edge_bias is not True:
        diffs.append("ModelConfig.use_edge_bias True -> False")
    if spec["zero_cols"]:
        diffs.append(f"edge feature columns {list(spec['zero_cols'])} zeroed")
    expected = list(spec["changes"])
    assert len(diffs) == len(expected), (
        f"{arm} changes {len(diffs)} things {diffs} but ARM_SPEC declares "
        f"{len(expected)} {expected}. One named change per arm; uid_K20 is the one "
        f"declared exception at two.")
    return tcfg, mcfg, gcfg


# ------------------------------------------------------- edge-feature zeroing (E5-E7)
def apply_zero_cols(ea: np.ndarray, cols) -> np.ndarray:
    """Zero the named columns of an edge-feature matrix, leaving its width alone.

    ZERO, do not delete. Keeping edge_dim at 5 keeps the parameter count byte-identical,
    so the arm is exactly parameter-matched to the canonical one. Deleting a column would
    change edge_dim and therefore the model. This is the same device Section 5.2 uses with
    the "__none__" placeholder.

    Returns a copy; the input is not modified.
    """
    out = np.array(ea, copy=True)
    for c in cols:
        if not 0 <= c < out.shape[1]:
            raise IndexError(f"edge feature column {c} out of range for width {out.shape[1]}")
        out[:, c] = 0.0
    return out


# NOTE ON WIRING E5-E7.  run_date_gnn_fold builds its edge features internally
# (src/run_date_gnn.py, the `ea_np = build_edge_features(...)` line) and offers no hook to
# transform them. Wiring the three feature-drop arms therefore needs ONE more optional
# parameter on that function — an `edge_feature_transform=None`, inert at its default, in
# the same style as the edge_builder / assert_causality / rng_state_path parameters added
# for protocol_v2. That change is NOT made here, because the work plan authorises the four
# arms only and these three are gated behind a result that does not yet exist. The
# transform itself is implemented and tested above, so wiring it later is a two-line
# change to src/ plus removing the runnable=False flag.


# --------------------------------------------------------------------------- preflight
def graph_facts(arm: str, B: dict, verbose: bool = True) -> dict:
    """Build this arm's graph once and check it against prediction BEFORE any GPU time.
    A K change that silently did not take effect would otherwise produce a clean null
    that reads as a finding. Costs a few seconds of numpy and no GPU."""
    from temporal_graph import build_temporal_edges, assert_causal
    df, cfg = B["df"], B["cfg"]
    _, _, gcfg = configs_for(arm, B)
    t0 = time.time()
    ei, dt, rel = build_temporal_edges(df.reset_index(drop=True), gcfg)
    ties = assert_causal(ei, df[cfg.time_col].to_numpy())
    facts = {
        "arm": arm,
        "K": gcfg.max_prior_neighbors,
        "entity_cols": list(gcfg.entity_cols),
        "edges": int(ei.shape[1]),
        "edges_per_node": round(ei.shape[1] / len(df), 2),
        "contemporaneous_edges": int(ties),
        "future_edges": 0,
        "build_seconds": round(time.time() - t0, 1),
        "receptive_field": C.receptive_field(arm),
    }
    want = C.GRAPH_EXPECT[arm]
    bad = []
    got, exp, tol = facts["edges_per_node"], want["edges_per_node"], want["tol"]
    if abs(got - exp) > exp * tol:
        bad.append(f"edges/node {got} vs expected {exp} (+/-{tol*100:.0f}%)")
    if "edges" in want and facts["edges"] != want["edges"]:
        bad.append(f"edges {facts['edges']:,} != expected {want['edges']:,}")
    if verbose:
        print(f"  graph for {arm}: K={facts['K']}, {facts['edges']:,} edges, "
              f"{facts['edges_per_node']}/node, {facts['contemporaneous_edges']} "
              f"contemporaneous, 0 future  ({facts['build_seconds']}s)")
        print(f"    expected {exp}/node ({want['source']}), receptive field "
              f"{facts['receptive_field']}")
    del ei, dt, rel
    if bad:
        raise AssertionError(f"graph facts for {arm} do not match prediction:\n    " +
                             "\n    ".join(bad))
    return facts


BYTES_PER_EDGE_PER_LAYER = 2400   # calibrated on the real depth3 OOM; see rerun_canonical


def project_vram(arm: str, B: dict, verbose: bool = True) -> dict:
    """Sample one batch from the largest fold and project peak VRAM. CPU only.

    This is a FLOOR, not a forecast. protocol_v2 measured the 900-byte version
    under-projecting every arm by ~4x and raised it to 2400, which restores most but not
    all of the margin. Treat the number as the minimum the arm will need.
    """
    from run_date_gnn import GraphCSR, sample_subgraph, build_edge_features
    from temporal_graph import build_temporal_edges
    tcfg, mcfg, gcfg = configs_for(arm, B)
    ei, dt, rel = build_temporal_edges(B["df"].reset_index(drop=True), gcfg)
    ea = build_edge_features(B["df"], ei, dt, rel, len(gcfg.entity_cols),
                             B["cfg"].amount_col)
    csr = GraphCSR(ei, ea, len(B["df"]))
    big = max(B["folds"], key=lambda f: len(f[1]))
    rng = np.random.default_rng(0)
    seeds = rng.choice(np.asarray(big[1]), size=min(tcfg.batch_size, len(big[1])),
                       replace=False)
    _, sei, _, _ = sample_subgraph(seeds, mcfg.n_layers, csr)
    e_batch = int(sei.shape[1])
    gb = e_batch * mcfg.n_layers * BYTES_PER_EDGE_PER_LAYER / 1e9
    out = {"arm": arm, "K": gcfg.max_prior_neighbors,
           "largest_fold_week": int(big[0]), "largest_fold_train_rows": len(big[1]),
           "batch_size": tcfg.batch_size, "layers": mcfg.n_layers,
           "edges_in_batch": e_batch, "projected_peak_gb": round(gb, 2),
           "with_headroom_gb": round(gb * (1 + C.EXPECT["vram_headroom_frac"]), 2),
           "bytes_per_edge_per_layer": BYTES_PER_EDGE_PER_LAYER,
           "caveat": "floor, not forecast; the constant has read low before"}
    if verbose:
        print(f"  {arm:16} K={out['K']:>2}  {e_batch:>9,} edges/batch  "
              f"~{out['projected_peak_gb']:6.2f} GB  "
              f"(+15% = {out['with_headroom_gb']:.2f} GB)")
    del csr, ei, ea, dt, rel
    return out


def preflight(arm: str, B: dict, verbose: bool = True) -> dict:
    """Everything that must be true before any GPU time is spent. Raises on anything
    wrong. Fixing a setting costs minutes now and up to forty hours later."""
    guards.assert_trainable(arm)
    tcfg, mcfg, gcfg = configs_for(arm, B)
    spec, e = C.ARM_SPEC[arm], C.EXPECT

    checks = [
        ("rows in the slice", len(B["df"]), e["slice_rows"]),
        ("features", len(B["num"]), e["n_features"]),
        ("folds", len(B["folds"]), e["n_folds"]),
        ("epochs", tcfg.epochs, 200),
        ("seed", tcfg.seed, 42),
        ("use_swa", tcfg.use_swa, False),
        ("layers", mcfg.n_layers, C.N_LAYERS),
        ("use_max_agg", mcfg.use_max_agg, C.USE_MAX_AGG),
        ("use_edge_bias", mcfg.use_edge_bias, spec["use_edge_bias"]),
        ("neighbour cap K", gcfg.max_prior_neighbors, spec["k"]),
        ("edge_dim", 3 + len(gcfg.entity_cols), e["edge_dim"]),
    ]
    if verbose:
        print(f"preflight — {arm}")
        for name, got, want in checks:
            print(f"  {'OK ' if got == want else 'BAD'}  {name:22} {got!r:>14}  "
                  f"expected {want!r}")
    bad = [f"{n}={g!r} expected {w!r}" for n, g, w in checks if g != w]

    free_gb = shutil.disk_usage(C.EDGE_K).free / 1e9
    if verbose:
        print(f"  {'OK ' if free_gb >= e['min_free_disk_gb'] else 'BAD'}  "
              f"{'free disk':22} {free_gb:13.1f}G  need {e['min_free_disk_gb']}G")
    if free_gb < e["min_free_disk_gb"]:
        bad.append(f"only {free_gb:.1f} GB free on disk")

    proj, proj_error = None, None
    try:
        proj = project_vram(arm, B, verbose=verbose)["projected_peak_gb"]
    except Exception as exc:
        proj_error = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  !!  PEAK-VRAM CHECK DID NOT RUN: {proj_error}")
            print(f"  !!  This arm is starting WITHOUT the memory headroom check.")

    free_b, _ = free_gpu()
    if proj is not None and free_b is not None:
        need = proj * (1 + e["vram_headroom_frac"])
        ok = need <= free_b / 1e9
        if verbose:
            print(f"  {'OK ' if ok else 'BAD'}  {'projected peak VRAM':22} "
                  f"{proj:12.1f}G  +15% = {need:.1f}G, {free_b/1e9:.1f}G free on {C.DEVICE}")
        if not ok:
            bad.append(f"projected peak {proj:.0f} GB (+15% = {need:.0f} GB) exceeds the "
                       f"{free_b/1e9:.0f} GB free on {C.DEVICE}. Free the card, or use the "
                       f"other GPU.")

    if bad:
        raise AssertionError("PREFLIGHT FAILED — do not start this arm:\n  - " +
                             "\n  - ".join(bad))
    if verbose:
        print(f"  preflight clean for {arm}")
    return {"checks": dict((n, g) for n, g, _ in checks),
            "free_disk_gb": round(free_gb, 1),
            "projected_peak_vram_gb": proj,
            "vram_check_skipped": proj_error,
            "free_vram_gb": round(free_b / 1e9, 1) if free_b else None}


# --------------------------------------------------------------------------- the run
def arm_dir(arm: str) -> Path:
    return guards.assert_writable(C.ARMS_DIR / guards.assert_trainable(arm))


def train_arm(arm: str, B: dict, save_weights=None, verbose: bool = True):
    """THE ONLY CALL TO run_date_gnn_fold IN edge_k.

    There is no resume parameter, deliberately. Every edge_k arm is one continuous
    execution: depth3's crash-and-resume became a permanent provenance note in the paper
    and none of these arms is long enough to justify repeating that. If an arm dies, move
    its directory to <arm>_ABANDONED_<utc>, record why in STATUS.txt, and start clean.
    """
    guards.assert_trainable(arm)                       # <-- the lock, first line
    from run_date_gnn import run_date_gnn_fold

    spec = C.ARM_SPEC[arm]
    if spec["zero_cols"]:
        raise NotImplementedError(
            f"{arm} needs edge-feature zeroing, which is not wired into "
            f"run_date_gnn_fold. See the note beside apply_zero_cols() in this file. "
            f"This arm is scaffolded, not runnable.")

    tcfg, mcfg, gcfg = configs_for(arm, B)
    folds = B["folds"]

    d = arm_dir(arm)
    d.mkdir(parents=True, exist_ok=True)
    ck = guards.assert_writable(d / "foldckpt.csv")
    carryover = [ck,
                 guards.assert_writable(d / "scores.csv"),
                 guards.assert_writable(d / "val_curves.csv"),
                 guards.assert_writable(d / "weights")]
    stale = [p.name for p in carryover if p.exists()]
    if stale:
        raise FileExistsError(
            f"{d} already holds {stale}. The CSVs are opened in append mode and "
            f"foldckpt.csv would make the runner skip folds. edge_k never resumes: move "
            f"the directory to {arm}_ABANDONED_<utc> and start clean. Nothing is deleted "
            f"for you.")

    if save_weights is None:
        save_weights = C.SAVE_WEIGHTS

    if verbose:
        print(f"\n=== {arm} ===   (a NEW arm; no canonical or protocol_v2 arm is touched)")
        print(f"  {spec['what']}")
        print(f"  the one change  {'; '.join(spec['changes'])}")
        print(f"  folds           {len(folds)}   weeks {B['weeks'][0]}-{B['weeks'][-1]}")
        print(f"  epochs          {tcfg.epochs}   seed {tcfg.seed}   device {tcfg.device}")
        print(f"  K               {gcfg.max_prior_neighbors}   "
              f"entity_cols {gcfg.entity_cols}   use_edge_bias {mcfg.use_edge_bias}")
        print(f"  receptive field {C.receptive_field(arm)} nodes per target")
        print(f"  out             {d}")

    ST.open_arm(arm)
    ST.gate(arm, "preflight and controls passed; training starts now")

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
        edge_builder=None,               # causal builder, exactly as Section 5.1
        assert_causality=True,           # every edge_k arm is causal
        rng_state_path=None,             # no resume, so no state to carry
    )
    hours = round((time.time() - t0) / 3600, 2)
    return finalise(arm, res, B, tcfg, mcfg, gcfg, hours, save_weights, verbose=verbose)


def finalise(arm, res, B, tcfg, mcfg, gcfg, hours, save_weights, verbose=True):
    """Write per_fold.csv, summary.json and arm_manifest.json. Everything is written
    before any problem is raised, so a failed check never costs the results."""
    d = arm_dir(arm)
    per_fold = guards.assert_writable(d / "per_fold.csv")
    res.to_csv(per_fold, index=False)

    problems = []
    if len(res) != C.EXPECT["n_folds"]:
        problems.append(f"{len(res)} folds, expected {C.EXPECT['n_folds']}")
    scores_p = d / "scores.csv"
    if scores_p.exists():
        n = sum(1 for _ in open(scores_p)) - 1
        if n != C.EXPECT["scored_rows"]:
            problems.append(f"{n:,} scored rows, expected {C.EXPECT['scored_rows']:,}")

    summary = {
        "arm": arm,
        "complete": not problems,
        "what": C.ARM_SPEC[arm]["what"],
        "n_folds": int(len(res)),
        "mean_ap": float(res["ap"].mean()),
        "sd_ap_across_folds": float(res["ap"].std(ddof=1)) if len(res) > 1 else None,
        "mean_p_at_k": float(res["precision_at_k"].mean()),
        "mean_r_at_k": float(res["recall_at_k"].mean()),
        "epochs_min": int(res["epochs_used"].min()),
        "epochs_median": int(res["epochs_used"].median()),
        "epochs_max": int(res["epochs_used"].max()),
        "epoch_budget": tcfg.epochs,
        "seed": tcfg.seed,
        "device": tcfg.device,
        "hours": hours,
        "segments": 1,
        "weights_saved": bool(save_weights),
        "problems": problems,
    }
    guards.assert_writable(d / "summary.json").write_text(json.dumps(summary, indent=1))

    man = P.arm_manifest(arm, summary, tcfg, mcfg, gcfg, extra={
        "environment": P.environment_manifest(B["prepared"] / "sparkov_clean_v1.csv"),
        "environment_vs_canonical": P.compare_environment(
            P.environment_manifest(B["prepared"] / "sparkov_clean_v1.csv"), verbose=False),
        "canonical_lock": guards.assert_canonical_unchanged(verbose=False),
    })
    guards.assert_writable(d / "arm_manifest.json").write_text(json.dumps(man, indent=1))

    ST.close_arm(arm, summary)
    if verbose:
        print(f"\n  {arm}: mean AP {summary['mean_ap']:.4f} over {summary['n_folds']} folds "
              f"in {hours}h")
        print(f"  wrote per_fold.csv, summary.json, arm_manifest.json to {d}")
    if problems:
        raise AssertionError(f"{arm} finished with problems: {problems}")
    return summary
