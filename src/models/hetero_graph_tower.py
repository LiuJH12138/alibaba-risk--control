"""Stage 3a Heterogeneous GraphSAGE tower.

Wraps PyG `HeteroConv` over 9 `SAGEConv` instances (one per relation/direction).
Transaction nodes carry `mixer_out_dim`-dimensional embeddings produced upstream
by the shared `EmbeddingMixer` (so the cat+num tower sees the same transaction
representation as the sequence tower). Entity nodes carry pre-computed 5-dim
aggregated statistics projected per-type to `d_graph`.
"""
from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv


class EntityProjector(nn.Module):
    """Per-entity-type Linear projecting 5-dim aggregates to d_graph."""

    def __init__(self, entity_types: Iterable[str], in_dim: int = 5, d_graph: int = 64):
        super().__init__()
        self.proj = nn.ModuleDict({t: nn.Linear(in_dim, d_graph) for t in entity_types})

    def forward(self, x_dict: dict) -> dict:
        # transaction nodes are projected separately by the tower; entity types only here
        return {t: self.proj[t](x) for t, x in x_dict.items() if t != "transaction"}


# Edge schema is fixed: 4 forward + 4 reverse relations + 1 directed transaction-transaction edge.
EDGE_SPEC: list[tuple[str, str, str]] = [
    ("transaction", "paid_with", "card1"),
    ("card1", "rev_paid_with", "transaction"),
    ("transaction", "shipped_to", "addr1"),
    ("addr1", "rev_shipped_to", "transaction"),
    ("transaction", "sent_to_email", "P_emaildomain"),
    ("P_emaildomain", "rev_sent_to_email", "transaction"),
    ("transaction", "on_device", "DeviceInfo"),
    ("DeviceInfo", "rev_on_device", "transaction"),
    ("transaction", "next_by_uid", "transaction"),
]


class HeteroGraphTower(nn.Module):
    """2-layer HeteroConv with mean-aggregation SAGEConv per relation.

    `mean` aggregator is chosen because card1/addr1 entity-degree distribution is
    long-tailed (top 1% holds ~30% of edges); `sum` would let head entities
    dominate the message, drowning out cold/medium ones.
    """

    def __init__(self, mixer_out_dim: int, d_graph: int = 64, n_layers: int = 2,
                 entity_types: Iterable[str] = ("card1", "addr1", "P_emaildomain", "DeviceInfo"),
                 dropout: float = 0.2):
        super().__init__()
        self.entity_types = tuple(entity_types)
        self.entity_proj = EntityProjector(self.entity_types, in_dim=5, d_graph=d_graph)
        self.txn_proj = nn.Linear(mixer_out_dim, d_graph)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(HeteroConv(
                {edge: SAGEConv(d_graph, d_graph, aggr="mean") for edge in EDGE_SPEC},
                aggr="mean",
            ))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hetero_data, txn_mixed_emb: torch.Tensor,
                seed_local: torch.Tensor) -> torch.Tensor:
        x_dict = self.entity_proj(hetero_data.x_dict)
        x_dict["transaction"] = self.txn_proj(txn_mixed_emb)
        for conv in self.convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {t: self.dropout(self.act(x)) for t, x in x_dict.items()}
        return x_dict["transaction"][seed_local]
