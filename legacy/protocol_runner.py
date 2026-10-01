"""
=======================================================================================
DEPRECATED - DO NOT USE FOR NEW WORK.  Superseded 2026-09-01 by protocol_v2/.
=======================================================================================

This module is a hand-made COPY of run_date_gnn_fold's body, taken once and never
updated. The original then gained three things this copy never got, and every one of
them turned into a defect in the results this file produced:

    validation curves are not recorded        (run_date_gnn_fold writes val_curves.csv)
    selected weights are not saved            (run_date_gnn_fold writes weights_dir)
    the "config" column is missing            (run_date_gnn_fold adds it)
    the resume key is not cast to int         (breaks on non-integer fold labels)
    scores.csv uses a different row_pos rule  (global positions, not a per-fold counter)

Its own docstring below already warned "if frozen internals change, re-derive from
run_date_gnn.py". Nobody did, for three months.

The replacement is run_date_gnn.run_date_gnn_fold itself, which now takes
`edge_builder` and `assert_causality` so an acausal arm needs no separate copy of the
training loop. protocol_v2/runner.py is the only caller. Use that.

This file is kept solely so the 75-epoch results in results/protocol/ remain traceable
to the code that produced them. Nothing new should import it.

---------------------------------------------------------------------------------------
Original docstring follows.
---------------------------------------------------------------------------------------

Protocol-study runner shim for notebook 08 (four-arm evaluation-protocol ablation).

Wraps the frozen run_date_gnn machinery so we can, WITHOUT editing frozen code:
  1. build EITHER causal (temporal_graph) OR acausal (acausal_graph) edges
  2. skip assert_causal for the deliberately-leaky arm
  3. report AUROC in addition to AP / P@k / R@k (the literature's inflated metric)

Everything else — training, SWA, neighbour sampling, val-selection, focal loss — is
the identical code path used for the locked result, so any performance difference
across arms is attributable to the PROTOCOL, not to a reimplementation.

This is intentionally a thin re-expression of run_date_gnn_fold's body. If frozen
internals change, re-derive from run_date_gnn.py.
"""

from __future__ import annotations

import warnings

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import run_date_gnn as R
from model import DateGNN, ModelConfig
from temporal_graph import GraphConfig, build_temporal_edges, assert_causal
from acausal_graph import build_acausal_edges, count_future_edges


def _metrics_with_auroc(y_true, scores, alert_rate, evaluation):
    m = evaluation.fold_metrics(y_true, scores, alert_rate)          # ap, p@k, r@k, k, n
    yt = np.asarray(y_true).astype(int)
    m["auroc"] = float(roc_auc_score(yt, scores)) if yt.sum() > 0 and yt.sum() < len(yt) else float("nan")
    return m


def run_protocol_fold(df, feature_cols, cat_cols, folds, gcfg, mcfg, tcfg,
                      target_col, time_col, alert_rate, evaluation,
                      amount_col="amt", causal=True, checkpoint_path=None, scores_path=None):
    """Same contract as run_date_gnn_fold, plus:
        causal=True  -> temporal_graph.build_temporal_edges + assert_causal (safe arms)
        causal=False -> acausal_graph.build_acausal_edges, NO assert (leaky arm)
    Emits an extra 'auroc' column and (for the leaky arm) prints the future-edge fraction.

    scores_path: if given, persist per-row test scores (row_pos, score, y, fold_label) so
    every arm can later be RE-EVALUATED on identical weekly folds. Without this, P1/P2
    (single big test pool, k=0.5% of 97k-338k rows) are not comparable to P3 (17 pools,
    k=0.5% of ~16k rows) — different k, different base rates, different estimators.

    DEPRECATED. Use run_date_gnn.run_date_gnn_fold with edge_builder= and
    assert_causality=. See the module banner.
    """
    warnings.warn(
        "protocol_runner.run_protocol_fold is DEPRECATED and diverged from "
        "run_date_gnn_fold. Use run_date_gnn.run_date_gnn_fold with edge_builder= and "
        "assert_causality= instead (protocol_v2/runner.py does this).",
        DeprecationWarning, stacklevel=2)
    torch.manual_seed(tcfg.seed); np.random.seed(tcfg.seed)
    dev = torch.device(tcfg.device)
    y = df[target_col].to_numpy().astype(np.float32)
    times = df[time_col].to_numpy()

    if causal:
        ei_np, dt_np, rel_np = build_temporal_edges(df.reset_index(drop=True), gcfg)
        assert_causal(ei_np, times)
    else:
        ei_np, dt_np, rel_np = build_acausal_edges(df.reset_index(drop=True), gcfg)
        diag = count_future_edges(ei_np, times)
        print(f"  [acausal] {diag['future_edges']:,}/{diag['n_edges']:,} edges point "
              f"FUTURE->current  ({diag['future_frac']*100:.1f}%) — this is the graph leak")

    n_rel = len(gcfg.entity_cols)
    ea_np = R.build_edge_features(df, ei_np, dt_np, rel_np, n_rel, amount_col)
    mcfg.edge_dim = ea_np.shape[1]
    csr = R.GraphCSR(ei_np, ea_np, len(df)); n_hops = mcfg.n_layers

    done_weeks, prior_rows = set(), []
    if checkpoint_path and Path(checkpoint_path).exists():
        prev = pd.read_csv(checkpoint_path)
        prior_rows = prev.to_dict("records"); done_weeks = set(prev["test_week"].tolist())
        print(f"  [resume] {len(done_weeks)} folds done: {sorted(done_weeks)}")

    rows = []
    for w, tr, te in folds:
        if w in done_weeks:
            print(f"  [skip] {w}"); continue
        tr = np.asarray(tr); te = np.asarray(te)
        n_val = max(1, int(len(tr) * tcfg.val_frac)); tr_core, val = tr[:-n_val], tr[-n_val:]
        x_np = R._standardize(df, feature_cols, tr)
        cat_np, vocab = R._encode_categoricals(df, cat_cols, tr)

        model = DateGNN(num_dim=x_np.shape[1], cat_vocab_sizes=vocab, mcfg=mcfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
        pos = float(y[tr_core].sum()); neg = float(len(tr_core) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=dev)
        bs = tcfg.batch_size or len(tr_core)
        swa_start = int(tcfg.epochs * tcfg.swa_start_frac)

        best_val, best_state, since = -1.0, None, 0
        swa_sum, swa_n = None, 0
        for ep in range(tcfg.epochs):
            model.train()
            perm = np.random.permutation(tr_core)
            for i in range(0, len(perm), bs):
                seeds = perm[i:i + bs]
                nodes, ei, ea, seed_local = R.sample_subgraph(seeds, n_hops, csr)
                opt.zero_grad()
                logits = R._forward_batch(model, nodes, ei, ea, x_np, cat_np, dev)
                loss = R._loss_fn(logits[seed_local], torch.from_numpy(y[seeds]).to(dev), pos_weight, tcfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
            sched.step()

            if tcfg.use_swa and ep >= swa_start:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if swa_sum is None:
                    swa_sum = sd; swa_n = 1
                else:
                    for k in swa_sum:
                        if swa_sum[k].is_floating_point():
                            swa_sum[k] += sd[k]
                    swa_n += 1

            vs = R._infer(model, val, n_hops, csr, x_np, cat_np, dev, tcfg.infer_batch_size)
            vmet = R._val_metric(y[val], vs, alert_rate, tcfg.early_stop_metric, evaluation)
            if vmet > best_val:
                best_val, since = vmet, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if (not tcfg.use_swa) and since >= tcfg.patience:
                    break

        if dev.type == "cuda":
            torch.cuda.empty_cache()

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
                vs = R._infer(model, val, n_hops, csr, x_np, cat_np, dev, tcfg.infer_batch_size)
                scored.append((R._val_metric(y[val], vs, alert_rate, tcfg.early_stop_metric, evaluation), name, st))
            scored.sort(key=lambda t: t[0], reverse=True)
            chosen = scored[0][1]; model.load_state_dict(scored[0][2])
        elif candidates:
            model.load_state_dict(candidates[0][1])

        scores = R._infer(model, te, n_hops, csr, x_np, cat_np, dev, tcfg.infer_batch_size)

        if scores_path:                      # persist for per-week re-evaluation
            hdr_s = not Path(scores_path).exists()
            pd.DataFrame({"row_pos": np.asarray(te), "score": np.asarray(scores),
                          "y": y[te].astype(int), "fold": w}).to_csv(
                scores_path, mode="a", header=hdr_s, index=False)

        m = _metrics_with_auroc(y[te], scores, alert_rate, evaluation)
        m.update({"test_week": w, "epochs_used": ep + 1, "final": chosen,
                  "n_features": len(feature_cols) + len(cat_cols)})
        rows.append(m)
        print(f"  week {w}: AP={m['ap']:.4f} AUROC={m['auroc']:.4f} "
              f"P@k={m['precision_at_k']:.4f} R@k={m['recall_at_k']:.4f} (ep={ep+1})")

        if checkpoint_path:
            hdr = not Path(checkpoint_path).exists()
            pd.DataFrame([m]).to_csv(checkpoint_path, mode="a", header=hdr, index=False)

        del model, opt, sched, best_state, swa_sum, x_np, cat_np
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(prior_rows + rows)
