"""Stage 3a Heterogeneous Graph tower.

Wraps PyG `HeteroConv` over 9 conv instances (one per relation/direction).
Transaction nodes carry `mixer_out_dim`-dimensional embeddings produced upstream
by the shared `EmbeddingMixer` (so the cat+num tower sees the same transaction
representation as the sequence tower). Entity nodes carry pre-computed 5-dim
aggregated statistics projected per-type to `d_graph`.

conv_type options (Stage 3a v3.3):
  sage  - SAGEConv(aggr='mean'). Default and Stage 3a v2 baseline.
  gatv2 - GATv2Conv(heads=2, concat=False, add_self_loops=False). Brody 2022.
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
    """2-layer HeteroConv with per-relation mean-aggregation conv.

    conv_type:
      'sage'  - SAGEConv(aggr='mean'). Stage 3a v2 default.
      'gatv2' - GATv2Conv(heads=2, concat=False, add_self_loops=False).
                Brody 2022; strictly more expressive than GAT, same cost.
                add_self_loops=False required for heterogeneous edges
                (src_type != dst_type breaks the default GAT self-loop logic).
    """

    def __init__(self, mixer_out_dim: int, d_graph: int = 64, n_layers: int = 2,
                 entity_types: Iterable[str] = ("card1", "addr1", "P_emaildomain", "DeviceInfo"),
                 dropout: float = 0.2, conv_type: str = "sage"):
        super().__init__()
        self.entity_types = tuple(entity_types)
        self.conv_type = conv_type
        self.entity_proj = EntityProjector(self.entity_types, in_dim=5, d_graph=d_graph)
        self.txn_proj = nn.Linear(mixer_out_dim, d_graph)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            if conv_type == "sage":
                edge_to_conv = {edge: SAGEConv(d_graph, d_graph, aggr="mean") for edge in EDGE_SPEC}
            elif conv_type == "gatv2":
                from torch_geometric.nn import GATv2Conv
                edge_to_conv = {
                    edge: GATv2Conv(d_graph, d_graph, heads=2, concat=False,
                                    dropout=dropout, add_self_loops=False)
                    for edge in EDGE_SPEC
                }
            else:
                raise ValueError(f"unknown conv_type: {conv_type}")
            self.convs.append(HeteroConv(edge_to_conv, aggr="mean"))
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
