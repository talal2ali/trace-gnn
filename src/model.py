"""
DATE-GNN v2 (advanced) — pure PyTorch.

Bundle:
  1. learned categorical embeddings (merchant/category/state/job/gender)
  2. full pre-norm transformer blocks (edge-aware attention + FFN + residual + LayerNorm)
  3. edge-aware attention: attention logit conditioned on edge features
  4. time-aware weighting: edge features include the prior->current time gap (Δt),
     so the model can learn to weight recent neighbors more (seed of the drift arm)
  5. (training enhancements live in run_date_gnn: cosine LR, early stopping, grad clip)

Message flows from edge_index[0] (prior) to edge_index[1] (current); causal by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn_mult: int = 2
    dropout: float = 0.2
    cat_emb_dim: int = 16
    edge_dim: int = 2          # [Δt/30days, log1p(Δt days)]
    use_max_agg: bool = False  # False = LOCKED behaviour (attention-weighted sum only).
                               # True  = fuse attention-sum with a max-pool over neighbour
                               #         messages ("was ANY prior txn sharply anomalous?",
                               #         GraphSAGE max-pool intuition). Ablatable.
    use_edge_bias: bool = True # True  = LOCKED behaviour: the additive per-head edge term
                               #         of Equation 9 is applied, exactly as in every
                               #         reported arm. This is the default, so nothing
                               #         changes for existing callers.
                               # False = the edge encoding is not consulted at all and the
                               #         Linear(edge_dim, n_heads) is not constructed, so
                               #         attention reduces to plain scaled dot-product over
                               #         the same causal neighbourhood. Used only by the
                               #         edge_k/ no_edge_bias arm. Removes 24 parameters
                               #         per block.


def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
    max_per_dst = scores.new_full((n_nodes, scores.size(1)), float("-inf"))
    max_per_dst = max_per_dst.index_reduce(0, dst, scores, "amax", include_self=True)
    scores = scores - max_per_dst[dst]
    exp = scores.exp()
    denom = torch.zeros(n_nodes, scores.size(1), device=scores.device).index_add(0, dst, exp)
    return exp / (denom[dst] + 1e-16)


class EdgeAwareAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, edge_dim: int, dropout: float,
                 use_max_agg: bool = False, use_edge_bias: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.use_max_agg = use_max_agg
        self.use_edge_bias = use_edge_bias
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        # Not constructed when the arm is edge-bias-free, so the parameter count drops by
        # exactly (edge_dim * n_heads + n_heads) per block rather than carrying dead weight.
        self.edge_bias = nn.Linear(edge_dim, n_heads) if use_edge_bias else None
        # locked behaviour: Linear(dim,dim). max-agg: fuse [sum ; max] -> Linear(2*dim, dim)
        self.out = nn.Linear(dim * (2 if use_max_agg else 1), dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        n = x.size(0); src, dst = edge_index[0], edge_index[1]
        q = self.q(x).view(n, self.h, self.dh)
        k = self.k(x).view(n, self.h, self.dh)
        v = self.v(x).view(n, self.h, self.dh)
        score = (q[dst] * k[src]).sum(-1) / (self.dh ** 0.5)      # (E,H)
        if self.edge_bias is not None:
            score = score + self.edge_bias(edge_attr)              # edge-aware + time-aware
        alpha = self.drop(_scatter_softmax(score, dst, n))         # (E,H)
        msg = v[src] * alpha.unsqueeze(-1)
        agg = torch.zeros(n, self.h, self.dh, device=x.device).index_add_(0, dst, msg)
        agg = agg.reshape(n, self.h * self.dh)

        if self.use_max_agg:
            # max over the SAME causal neighbour set (prior->current edges only), so the
            # leakage guarantee is unchanged. Isolated nodes (no in-edges) -> 0.
            neg_inf = torch.finfo(v.dtype).min
            agg_max = torch.full((n, self.h, self.dh), neg_inf, device=x.device, dtype=v.dtype)
            agg_max = agg_max.index_reduce_(0, dst, v[src], "amax", include_self=True)
            agg_max = torch.where(agg_max <= neg_inf, torch.zeros_like(agg_max), agg_max)
            agg = torch.cat([agg, agg_max.reshape(n, self.h * self.dh)], dim=-1)

        return self.out(agg)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, edge_dim, dropout, use_max_agg=False,
                 use_edge_bias=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = EdgeAwareAttention(dim, n_heads, edge_dim, dropout,
                                       use_max_agg=use_max_agg,
                                       use_edge_bias=use_edge_bias)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        h = h + self.drop(self.attn(self.ln1(h), edge_index, edge_attr))   # pre-norm residual
        h = h + self.drop(self.ffn(self.ln2(h)))
        return h


class DateGNN(nn.Module):
    def __init__(self, num_dim: int, cat_vocab_sizes: List[int], mcfg: ModelConfig):
        super().__init__()
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(v + 1, mcfg.cat_emb_dim) for v in cat_vocab_sizes]   # idx 0 = unknown/missing
        )
        in_dim = num_dim + len(cat_vocab_sizes) * mcfg.cat_emb_dim
        self.in_proj = nn.Sequential(nn.Linear(in_dim, mcfg.hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            TransformerBlock(mcfg.hidden_dim, mcfg.n_heads, mcfg.ffn_mult, mcfg.edge_dim,
                             mcfg.dropout, use_max_agg=mcfg.use_max_agg,
                             use_edge_bias=getattr(mcfg, "use_edge_bias", True))
            for _ in range(mcfg.n_layers)
        ])
        self.norm = nn.LayerNorm(mcfg.hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(mcfg.hidden_dim, mcfg.hidden_dim), nn.GELU(),
            nn.Dropout(mcfg.dropout), nn.Linear(mcfg.hidden_dim, 1),
        )

    def forward(self, x_num, cat_idx, edge_index, edge_attr):
        parts = [x_num]
        for emb, idx in zip(self.cat_embs, cat_idx):
            parts.append(emb(idx))
        h = self.in_proj(torch.cat(parts, dim=1))
        for blk in self.blocks:
            h = blk(h, edge_index, edge_attr)
        return self.decoder(self.norm(h)).squeeze(-1)
