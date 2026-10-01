"""
DATE-GNN v3 per-fold runner: multi-relational neighbor-sampling, AdamW + SWA,
focal/budget-aligned loss, richer edge features. Leakage-safe and memory-scalable.

Enhancements over v2 (driven by what v2's results showed):
  - multi-relational edges (uid + merchant) with relation-aware edge features
  - richer edge features: [Δt/30, log1p(Δt days), relation one-hot, log amount ratio]
  - AdamW + cosine LR + optional SWA (stochastic weight averaging) to cut the high
    fold-to-fold variance v2 showed; LayerNorm-only model => no BN recalibration needed
  - focal loss option (targets hard fraud cases -> precision@k) + weighted BCE
  - budget-aligned model selection: early-stop / SWA-select on val AP or val precision@k
  - higher epoch ceiling

Memory scales with batch size (neighbor-sampling expands exactly n_layers causal hops,
identical to full-graph forward). Edges are prior->current only (leakage-safe).
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from temporal_graph import GraphConfig, build_temporal_edges, assert_causal
from model import DateGNN, ModelConfig


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4              # AdamW decoupled
    grad_clip: float = 1.0
    val_frac: float = 0.15
    patience: int = 25
    batch_size: Optional[int] = 8192
    infer_batch_size: int = 16384
    loss: str = "focal"                    # "focal" | "bce"
    focal_gamma: float = 2.0
    early_stop_metric: str = "ap"          # "ap" | "precision_at_k" (budget-aligned)
    use_swa: bool = True
    swa_start_frac: float = 0.6            # begin weight averaging after this fraction
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# --------------------------------------------------------------------------- utilities
def subsample_recent(df, week_col, n):
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    df = df.tail(int(n)).reset_index(drop=True)
    df[week_col] = (df[week_col] - df[week_col].min()).astype(int)
    return df


def frauds_per_fold(df, folds, target_col, val_frac=0.15):
    """Report train/val/test sizes and fraud counts per fold (transparency for sparse folds)."""
    y = df[target_col].to_numpy()
    rows = []
    for w, tr, te in folds:
        tr = np.asarray(tr); te = np.asarray(te)
        nv = max(1, int(len(tr) * val_frac)); tr_core, val = tr[:-nv], tr[-nv:]
        rows.append({"test_week": int(w), "n_train": len(tr_core), "train_fraud": int(y[tr_core].sum()),
                     "n_val": len(val), "val_fraud": int(y[val].sum()),
                     "n_test": len(te), "test_fraud": int(y[te].sum())})
    return pd.DataFrame(rows)


def filter_folds(df, folds, target_col, min_test_fraud=5, min_train_fraud=5, val_frac=0.15):
    """Drop folds too sparse to score reliably, BEFORE any model runs, so every model uses
    an identical fold set by construction. Returns (kept_folds, report_df)."""
    rep = frauds_per_fold(df, folds, target_col, val_frac)
    keep_weeks = rep[(rep.test_fraud >= min_test_fraud) & (rep.train_fraud >= min_train_fraud)]["test_week"].tolist()
    kept = [(w, tr, te) for (w, tr, te) in folds if int(w) in keep_weeks]
    rep["kept"] = rep.test_week.isin(keep_weeks)
    return kept, rep


def _standardize(df, feature_cols, train_pos):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    mu = np.nanmean(X[train_pos], axis=0); sd = np.nanstd(X[train_pos], axis=0); sd[sd == 0] = 1.0
    return (np.where(np.isnan(X), mu, X) - mu).__truediv__(sd).astype(np.float32)


def _encode_categoricals(df, cat_cols, train_pos):
    idx_arrays, vocab_sizes = [], []
    for c in cat_cols:
        train_vals = pd.Index(df[c].iloc[train_pos].dropna().unique())
        mapping = {v: i + 1 for i, v in enumerate(train_vals)}
        idx_arrays.append(df[c].map(mapping).fillna(0).astype(np.int64).to_numpy())
        vocab_sizes.append(len(train_vals))
    return idx_arrays, vocab_sizes


def build_edge_features(df, edge_index, edge_dt, edge_rel, n_relations, amount_col):
    """[Δt/30, log1p(Δt days), relation one-hot (R), log amount ratio]."""
    dt_days = edge_dt / 86400.0
    feats = [dt_days / 30.0, np.log1p(np.clip(dt_days, 0, None))]
    onehot = np.zeros((len(edge_dt), n_relations), np.float32)
    valid = edge_rel >= 0
    onehot[np.arange(len(edge_dt))[valid], edge_rel[valid]] = 1.0
    if amount_col in df.columns and edge_index.shape[1] > 0:
        amt = df[amount_col].to_numpy().astype(np.float64)
        ratio = np.log((amt[edge_index[1]] + 1.0) / (amt[edge_index[0]] + 1.0))
    else:
        ratio = np.zeros(len(edge_dt))
    return np.column_stack([*feats, onehot, ratio]).astype(np.float32)


class GraphCSR:
    def __init__(self, edge_index, edge_attr, n):
        dst = edge_index[1]; order = np.argsort(dst, kind="stable")
        self.src = edge_index[0][order].astype(np.int64)
        self.attr = edge_attr[order].astype(np.float32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=n))]).astype(np.int64)
        self.edge_dim = edge_attr.shape[1]

    def in_edges(self, dst_nodes):
        starts = self.indptr[dst_nodes]; ends = self.indptr[dst_nodes + 1]; lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            z = np.empty(0, np.int64); return z, z, np.empty((0, self.edge_dim), np.float32)
        offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        epos = np.repeat(starts, lens) + offs
        return self.src[epos], np.repeat(dst_nodes, lens), self.attr[epos]


def sample_subgraph(seeds, n_hops, csr):
    Ss, Ds, As = [], [], []; expanded = set(); cur = np.unique(seeds)
    for _ in range(n_hops):
        to_exp = cur[~np.isin(cur, list(expanded))] if expanded else cur
        if len(to_exp) == 0:
            break
        s, d, a = csr.in_edges(to_exp); expanded.update(to_exp.tolist())
        if len(s):
            Ss.append(s); Ds.append(d); As.append(a); cur = np.unique(s)
        else:
            break
    if Ss:
        S = np.concatenate(Ss); D = np.concatenate(Ds); A = np.concatenate(As)
        nodes = np.unique(np.concatenate([seeds, S, D]))
        sub_src = np.searchsorted(nodes, S); sub_dst = np.searchsorted(nodes, D)
    else:
        nodes = np.unique(seeds); sub_src = np.empty(0, np.int64); sub_dst = np.empty(0, np.int64)
        A = np.empty((0, csr.edge_dim), np.float32)
    m = len(nodes); loops = np.arange(m)
    ei = np.vstack([np.concatenate([sub_src, loops]), np.concatenate([sub_dst, loops])]).astype(np.int64)
    ea = np.concatenate([A, np.zeros((m, csr.edge_dim), np.float32)]) if len(A) else np.zeros((m, csr.edge_dim), np.float32)
    return nodes, ei, ea, np.searchsorted(nodes, seeds)


# --------------------------------------------------------------------------- forward / loss
def _forward_batch(model, nodes, ei, ea, x_np, cat_np, dev):
    x_sub = torch.from_numpy(x_np[nodes]).to(dev)
    cat_sub = [torch.from_numpy(c[nodes]).to(dev) for c in cat_np]
    return model(x_sub, cat_sub, torch.from_numpy(ei).to(dev), torch.from_numpy(ea).to(dev))


def _loss_fn(logits, targets, pos_weight, tcfg):
    if tcfg.loss == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits); p_t = p * targets + (1 - p) * (1 - targets)
    return (ce * (1 - p_t).clamp(min=1e-6) ** tcfg.focal_gamma).mean()


def _infer(model, target_idx, n_hops, csr, x_np, cat_np, dev, bs):
    model.eval(); out = np.empty(len(target_idx), np.float32)
    with torch.no_grad():
        for i in range(0, len(target_idx), bs):
            seeds = target_idx[i:i + bs]
            nodes, ei, ea, seed_local = sample_subgraph(seeds, n_hops, csr)
            out[i:i + bs] = torch.sigmoid(_forward_batch(model, nodes, ei, ea, x_np, cat_np, dev)[seed_local]).cpu().numpy()
    return out


def _val_metric(y_val, scores, alert_rate, metric, evaluation):
    if y_val.sum() == 0:
        return 0.0
    if metric == "precision_at_k":
        return evaluation.fold_metrics(y_val, scores, alert_rate)["precision_at_k"]
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_val, scores)


# ------------------------------------------------------- RNG state across a resume (v2)
RNG_STATE_VERSION = 2


def _rng_state_identity(tcfg, mcfg, gcfg, feature_cols,
                        edge_builder, assert_causality) -> str:
    """Fingerprint of the run an RNG state belongs to.

    Version 1 hashed only the three configs and the feature list. That was not enough:
    protocol_v2 builds all four arms from the SAME TrainConfig, ModelConfig and
    GraphConfig — configs_for() does not even read the arm name — so all four produced an
    identical fingerprint. Two of the three things that actually distinguish them were
    missing and are now included:

        the graph builder      P1_causal vs P1_leaky differ ONLY here
        the causality flag     same

    The third distinguishing thing is the fold set, and it deliberately does NOT go in
    here. A smoke run over weeks 8-9 and the full run over 8-24 are the same arm with
    different fold LISTS, so folding the list into this hash would make the resume the
    smoke run exists to enable fail. Folds are checked separately, per fold, by
    _fold_digests below — which is subset-safe.
    """
    import hashlib
    from dataclasses import asdict

    if edge_builder is None:
        builder_name = "temporal_graph.build_temporal_edges (default)"
    else:
        builder_name = (f"{getattr(edge_builder, '__module__', '?')}."
                        f"{getattr(edge_builder, '__qualname__', repr(edge_builder))}")

    payload = json.dumps({
        "version": RNG_STATE_VERSION,
        "t": {k: str(v) for k, v in sorted(asdict(tcfg).items())},
        "m": {k: str(v) for k, v in sorted(asdict(mcfg).items())},
        "g": {k: str(v) for k, v in sorted(asdict(gcfg).items())},
        "features": list(feature_cols),
        "edge_builder": builder_name,
        "assert_causality": bool(assert_causality),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _fold_digests(folds) -> dict:
    """{fold label -> SHA-256 of that fold's train and test index arrays}.

    Held per fold rather than as one digest over the whole list, so a resume can check
    only the folds it shares with the saved state. That is what lets the A4 smoke run
    (2 folds) hand over to the full run (17 folds) while still catching a state file
    that belongs to a different arm: P1_causal and P2_cutoff have identical configs and
    identical builders, and are told apart here and only here.
    """
    import hashlib
    out = {}
    for label, tr, te in folds:
        tr = np.ascontiguousarray(np.asarray(tr, dtype=np.int64))
        te = np.ascontiguousarray(np.asarray(te, dtype=np.int64))
        h = hashlib.sha256()
        h.update(str((tr.size, te.size)).encode())
        h.update(tr.tobytes()); h.update(te.tobytes())
        out[int(label)] = h.hexdigest()[:16]
    return out


def pick_rng_state(state: dict, last_done):
    """Return the generator state saved immediately after fold `last_done`.

    The file keeps ONE ENTRY PER FOLD, not just the newest. That is not tidiness, it
    closes a hole. foldckpt.csv and the state file are two separate writes, so a crash
    between them leaves them one fold apart in whichever direction the writes are
    ordered — and neither direction is recoverable if only the newest state is kept: the
    checkpoint says N folds are done and the state is at N-1, or the reverse. Keeping
    every fold's state means whatever the checkpoint says finished, the matching state is
    there. Extra entries are ignored; a missing one raises.

    Costs about 8 KB per fold, so roughly 140 KB for a seventeen-fold arm.
    """
    states = state.get("states") or {}
    entry = states.get(str(int(last_done)), states.get(int(last_done)))
    if entry is None:
        raise RuntimeError(
            f"the RNG state file has no entry for fold {last_done}, which foldckpt.csv "
            f"records as the last fold completed. It holds entries for "
            f"{sorted(int(k) for k in states)}. The stream cannot be continued from the "
            f"right point; restart the arm.")
    return entry


def check_rng_state(state: dict, done_weeks, identity: str, fold_digests: dict,
                    n_cuda_devices: int) -> None:
    """Everything that must hold before a saved generator state may be restored.

    Split out of run_date_gnn_fold so it can be tested directly — see
    protocol_v2/tests/test_rng_resume.py. Every failure raises. None of them warns and
    continues, because a run that carries on with a partially restored stream is not the
    run its manifest will claim it is.
    """
    if state.get("version") != RNG_STATE_VERSION:
        raise RuntimeError(
            f"RNG state file is version {state.get('version')!r}; this code writes and "
            f"reads version {RNG_STATE_VERSION}. Version 1 files recorded too little to "
            f"be trusted — they could not tell one arm from another. Delete the "
            f"checkpoint and the state file and restart the arm.")

    # The checkpoint is the authority on what finished; the state file must simply COVER
    # it. Extra entries are fine and expected — they are the folds whose state was written
    # before a crash took the checkpoint write with it. Missing ones are not.
    saved = {int(w) for w in (state.get("states") or {})}
    done = {int(w) for w in done_weeks}
    if not done <= saved:
        raise RuntimeError(
            f"the RNG state file does not cover every checkpointed fold. The checkpoint "
            f"lists {sorted(done)}; the state file holds {sorted(saved)}; missing "
            f"{sorted(done - saved)}. The stream cannot be continued from the right "
            f"point. Restart the arm rather than guessing.")

    got = state.get("identity")
    if got != identity:
        raise RuntimeError(
            f"RNG state file belongs to a different run. Its configuration fingerprint is "
            f"{got!r}; this run's is {identity!r}. That covers the training, model and "
            f"graph configs, the feature list, the edge builder and the causality flag. A "
            f"state file from another arm, seed or configuration must not be restored "
            f"into this one.")

    # Fold identity, checked only on the folds the two runs share. This is what makes a
    # 2-fold smoke run a valid predecessor of the 17-fold run, while still refusing a
    # state file from an arm that happens to share every config (P1_causal vs P2_cutoff).
    saved_digests = state.get("fold_digests") or {}
    for w in sorted(saved):
        a = saved_digests.get(str(w), saved_digests.get(w))
        b = fold_digests.get(int(w))
        if b is None:
            raise RuntimeError(
                f"RNG state was saved after fold {w}, but this run has no fold {w}. The "
                f"state belongs to a different fold set.")
        if a != b:
            raise RuntimeError(
                f"fold {w} does not hold the same rows it did when the RNG state was "
                f"saved (train/test digest {a!r} then, {b!r} now). The state belongs to a "
                f"different arm or a different split. Restart the arm.")

    if state.get("n_cuda_devices", 0) != n_cuda_devices:
        raise RuntimeError(
            f"RNG state was saved on a machine with {state.get('n_cuda_devices')} CUDA "
            f"device(s); this one has {n_cuda_devices}. The CUDA generator state cannot be "
            f"restored faithfully, so the resumed stream would not be the stream an "
            f"uninterrupted run draws. Restart the arm instead.")

    # The generator payload lives in the PER-FOLD entries under "states", not at the top
    # level -- see _save_rng_state, which is the only writer. An earlier version of this
    # check read state["torch_cuda"], a key no real state file has ever carried, so on any
    # CUDA machine it raised unconditionally and no resume could ever succeed. The unit
    # test did not catch it because its fixture carried the top-level key. Every entry that
    # the resume might restore is checked, not just the newest, because pick_rng_state may
    # legitimately return an earlier fold's entry.
    if n_cuda_devices:
        for _w, _entry in sorted((state.get("states") or {}).items(),
                                 key=lambda kv: int(kv[0])):
            if _entry.get("torch_cuda") is None:
                raise RuntimeError(
                    f"the RNG state entry for fold {_w} carries no CUDA generator state, "
                    f"but this run uses CUDA. Restoring it would leave the CUDA stream at "
                    f"the seed while the CPU stream is mid-run. Restart the arm instead.")


# --------------------------------------------------------------------------- main runner
def run_date_gnn_fold(df, feature_cols, cat_cols, folds, gcfg, mcfg, tcfg,
                      target_col, time_col, alert_rate, evaluation, amount_col="amt",
                      checkpoint_path=None, val_curve_path=None,
                      scores_path=None, weights_dir=None,
                      edge_builder=None, assert_causality=True,
                      rng_state_path=None) -> pd.DataFrame:
    """weights_dir, if given, receives one checkpoint per fold holding the selected weights
    together with everything needed to rebuild the model: both configs, the feature names in
    order, and the categorical vocabulary sizes. A bare state_dict is close to useless later,
    because the input width and the edge dimension are not recoverable from it.

    Three optional parameters were added for the protocol-v2 study (protocol_v2/). All three
    default to the behaviour this function had before they existed, so leaving them alone
    executes the identical code path Section 5.1 used. This is a code-path claim, verifiable
    by reading the diff. It is NOT a claim that a re-run reproduces earlier output
    bit-for-bit: this pipeline is nondeterministic across executions (SD 0.0091 mean AP).

        edge_builder        None  -> temporal_graph.build_temporal_edges (causal, as before).
                                     Pass acausal_graph.build_acausal_edges for the arms that
                                     deliberately admit future edges.
        assert_causality    True  -> run assert_causal on the built edges, as before.
                                     The acausal arms must pass False; assert_causal is
                                     designed to fail on their edges.
        rng_state_path      None  -> no RNG state is written or read, as before. A resumed
                                     run therefore restarts the random stream from the seed,
                                     which is what produced the "resumed fold starts from a
                                     fresh RNG state" note in the canonical run manifest.
                                     Give a path and the generator state is saved at every
                                     fold boundary and restored on resume, so that
                                     (smoke run over weeks 8-9) + (resume over 10-24) draws
                                     exactly the same random stream as one uninterrupted run.
                                     See protocol_v2/README.md, "A4 smoke then resume".
    """
    torch.manual_seed(tcfg.seed); np.random.seed(tcfg.seed)
    dev = torch.device(tcfg.device)
    y = df[target_col].to_numpy().astype(np.float32); times = df[time_col].to_numpy()

    _build_edges = build_temporal_edges if edge_builder is None else edge_builder
    ei_np, dt_np, rel_np = _build_edges(df.reset_index(drop=True), gcfg)
    if assert_causality:
        assert_causal(ei_np, times)
    n_rel = len(gcfg.entity_cols)
    ea_np = build_edge_features(df, ei_np, dt_np, rel_np, n_rel, amount_col)
    mcfg.edge_dim = ea_np.shape[1]                      # keep model edge_dim consistent
    csr = GraphCSR(ei_np, ea_np, len(df)); n_hops = mcfg.n_layers

    # fold-level checkpoint: resume from completed folds
    done_weeks, prior_rows = set(), []
    if checkpoint_path and Path(checkpoint_path).exists():
        prev = pd.read_csv(checkpoint_path)
        prior_rows = prev.to_dict("records")
        done_weeks = {int(w) for w in prev["test_week"].tolist()}     # B5: cast to int
        print(f"  [resume] {len(done_weeks)} folds already done: {sorted(done_weeks)}")

    # --- RNG continuity across a resume (opt-in; default None = old behaviour) -----------
    # Without this, the seed above is the only thing setting the stream, so folds trained
    # after a resume see a stream that a continuous run would never have produced.
    _rng_identity = _rng_state_identity(tcfg, mcfg, gcfg, feature_cols,
                                        edge_builder, assert_causality)
    _fold_sha = _fold_digests(folds)
    _rng_states = {}
    if rng_state_path and Path(rng_state_path).exists() and not done_weeks:
        raise RuntimeError(
            f"{rng_state_path} exists but {checkpoint_path} records no completed folds. "
            f"That pairing is not resumable: the state belongs to work the checkpoint "
            f"does not account for. Move the arm directory aside and start over.")
    if rng_state_path and Path(rng_state_path).exists():
        _st = torch.load(rng_state_path, map_location="cpu", weights_only=False)
        _rng_states.update(_st.get("states") or {})
        check_rng_state(_st, done_weeks, _rng_identity, _fold_sha,
                        n_cuda_devices=(torch.cuda.device_count()
                                        if torch.cuda.is_available() else 0))
        # restore from the state saved after the LAST fold the checkpoint records, taken
        # in checkpoint order rather than by sorting, because folds are trained in list
        # order and that is where the stream actually stands
        _last = int(pd.read_csv(checkpoint_path)["test_week"].iloc[-1])
        _entry = pick_rng_state(_st, _last)
        np.random.set_state(_entry["numpy"])
        torch.set_rng_state(_entry["torch_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_entry["torch_cuda"])
        _extra = sorted({int(w) for w in _st["states"]} - done_weeks)
        print(f"  [rng] restored the state saved after fold {_last} "
              f"({len(done_weeks)} fold(s) done)"
              + (f"; ignored {len(_extra)} state(s) for uncheckpointed fold(s) {_extra}"
                 if _extra else ""))
    elif rng_state_path and done_weeks:
        raise RuntimeError(
            f"Resuming from {checkpoint_path} with {len(done_weeks)} folds already done, but "
            f"no RNG state at {rng_state_path}. The stream cannot be continued. Either delete "
            f"the checkpoint and restart the arm, or pass rng_state_path=None and accept the "
            f"fresh-stream resume the canonical run documents.")

    def _save_rng_state(just_finished, completed):
        """Append this fold's generator state. Every fold's state is kept — see
        pick_rng_state for why. Written to a temp file and moved into place, so an
        interruption mid-write cannot corrupt the file."""
        if not rng_state_path:
            return
        _rng_states[str(int(just_finished))] = {
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        }
        p = Path(rng_state_path); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save({"version": RNG_STATE_VERSION,
                    "completed_weeks": sorted(int(w) for w in completed),
                    "identity": _rng_identity,
                    "fold_digests": dict(_fold_sha),
                    "n_cuda_devices": (torch.cuda.device_count()
                                       if torch.cuda.is_available() else 0),
                    "states": _rng_states}, tmp)
        tmp.replace(p)                       # atomic: a crash mid-write cannot corrupt it

    rows = []
    for w, tr, te in folds:
        if int(w) in done_weeks:
            print(f"  [skip] week {w} (checkpointed)"); continue
        tr = np.asarray(tr); te = np.asarray(te)
        n_val = max(1, int(len(tr) * tcfg.val_frac)); tr_core, val = tr[:-n_val], tr[-n_val:]
        x_np = _standardize(df, feature_cols, tr)
        cat_np, vocab = _encode_categoricals(df, cat_cols, tr)

        model = DateGNN(num_dim=x_np.shape[1], cat_vocab_sizes=vocab, mcfg=mcfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
        pos = float(y[tr_core].sum()); neg = float(len(tr_core) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=dev)
        bs = tcfg.batch_size or len(tr_core)
        swa_start = int(tcfg.epochs * tcfg.swa_start_frac)

        best_val, best_state, since = -1.0, None, 0
        swa_sum, swa_n = None, 0
        val_curve = []
        for ep in range(tcfg.epochs):
            model.train()
            perm = np.random.permutation(tr_core)
            for i in range(0, len(perm), bs):
                seeds = perm[i:i + bs]
                nodes, ei, ea, seed_local = sample_subgraph(seeds, n_hops, csr)
                opt.zero_grad()
                logits = _forward_batch(model, nodes, ei, ea, x_np, cat_np, dev)
                loss = _loss_fn(logits[seed_local], torch.from_numpy(y[seeds]).to(dev), pos_weight, tcfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
            sched.step()

            # SWA accumulation
            if tcfg.use_swa and ep >= swa_start:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if swa_sum is None:
                    swa_sum = sd; swa_n = 1
                else:
                    for k in swa_sum:
                        if swa_sum[k].is_floating_point():
                            swa_sum[k] += sd[k]
                    swa_n += 1

            vs = _infer(model, val, n_hops, csr, x_np, cat_np, dev, tcfg.infer_batch_size)
            vmet = _val_metric(y[val], vs, alert_rate, tcfg.early_stop_metric, evaluation)
            val_curve.append({"test_week": int(w), "epoch": ep, "val_metric": float(vmet)})
            if vmet > best_val:
                best_val, since = vmet, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if (not tcfg.use_swa) and since >= tcfg.patience:
                    break

        if dev.type == "cuda":
            torch.cuda.empty_cache()          # release training-peak activations before selection

        # choose final weights: best-early-stopped vs SWA-average (whichever wins on val)
        candidates = []
        if best_state is not None:
            candidates.append(("best", best_state))
        if swa_sum is not None and swa_n > 0:
            swa_state = {k: (v / swa_n if v.is_floating_point() else v) for k, v in swa_sum.items()}
            candidates.append(("swa", swa_state))
        chosen = "best"
        if len(candidates) > 1:
            scored = []
            for name, st in candidates:
                model.load_state_dict(st)
                vs = _infer(model, val, n_hops, csr, x_np, cat_np, dev, tcfg.infer_batch_size)
                scored.append((_val_metric(y[val], vs, alert_rate, tcfg.early_stop_metric, evaluation), name, st))
            scored.sort(key=lambda t: t[0], reverse=True)
            chosen = scored[0][1]; model.load_state_dict(scored[0][2])
        elif candidates:
            model.load_state_dict(candidates[0][1])

        # persist the selected weights before scoring, while the chosen state is loaded
        if weights_dir:
            from dataclasses import asdict
            wd = Path(weights_dir); wd.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": asdict(mcfg),
                "train_config": asdict(tcfg),
                "graph_config": asdict(gcfg),
                "feature_cols": list(feature_cols),
                "cat_cols": list(cat_cols),
                "cat_vocab_sizes": list(vocab),
                "num_dim": int(x_np.shape[1]),
                "test_week": int(w),
                "selected": chosen,
                "epochs_used": int(ep + 1),
                "note": ("Standardisation is fitted on this fold's training rows and is NOT "
                         "stored here; rebuild it with _standardize on the same fold to score "
                         "new data consistently."),
            }, wd / f"fold_{int(w):02d}.pt")

        scores = _infer(model, te, n_hops, csr, x_np, cat_np, dev, tcfg.infer_batch_size)

        m = evaluation.fold_metrics(y[te], scores, alert_rate)
        m.update({"config": "date_gnn_v3", "test_week": int(w),
                  "n_features": len(feature_cols) + len(cat_cols), "epochs_used": ep + 1, "final": chosen})
        rows.append(m)
        print(f"  [date_gnn_v3] week {w}: AP={m['ap']:.4f} P@k={m['precision_at_k']:.4f} "
              f"(train={len(tr_core)}, test={len(te)}, ep={ep+1}, final={chosen})")

        # ---------------------------------------------------------------- write order
        # THE CHECKPOINT IS WRITTEN LAST. Everything else for this fold — scores,
        # validation curve, RNG state — is on disk before foldckpt.csv names the fold as
        # done. That makes the checkpoint the single source of truth: anything it does
        # not list is unfinished, and every crash window is recoverable.
        #
        #   crash before the scores write      nothing written, fold simply retrains
        #   crash between scores and ckpt      orphan score/val rows exist for a fold the
        #                                      checkpoint does not list; the RNG state may
        #                                      hold an extra entry. On resume,
        #                                      runner.reconcile_after_crash drops the
        #                                      orphan rows and the extra RNG entry is
        #                                      ignored, so the fold retrains cleanly.
        #   crash after the ckpt write         the fold is complete; nothing to undo.
        #
        # The RNG state MUST be saved before the checkpoint, not after. The other way
        # round leaves a window where the checkpoint claims a fold the state file has no
        # entry for, and that is the one combination nothing can recover: the stream
        # cannot be rewound, so the arm has to be restarted. Saving it first only ever
        # leaves a harmless extra entry.
        #
        # pandas.to_csv consumes no randomness, so moving the save earlier captures the
        # identical generator state.
        if scores_path:
            hdr = not Path(scores_path).exists()
            pd.DataFrame({"row_pos": np.arange(len(te)), "score": scores,
                          "y": y[te].astype(int), "fold": int(w)}).to_csv(
                scores_path, mode="a", header=hdr, index=False)
        if val_curve_path:
            hdr = not Path(val_curve_path).exists()
            pd.DataFrame(val_curve).to_csv(val_curve_path, mode="a", header=hdr, index=False)

        # fold boundary: this is exactly the point a resumed run must pick the stream up at
        done_weeks.add(int(w))
        _save_rng_state(int(w), done_weeks)

        if checkpoint_path:
            hdr = not Path(checkpoint_path).exists()
            pd.DataFrame([m]).to_csv(checkpoint_path, mode="a", header=hdr, index=False)

        del model, opt, sched, best_state, swa_sum, x_np, cat_np
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(prior_rows + rows)
