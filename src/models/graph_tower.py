import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphTower(nn.Module):
    """GraphSAGE 图塔。在交易图上聚合邻居信息,输出每节点 graph_emb。
    n_layers 层 SAGEConv,层间 ReLU + dropout。无边时退化为逐节点 MLP 行为。"""

    def __init__(self, feat_dim: int, d_graph: int, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        in_dim = feat_dim
        for _ in range(n_layers):
            self.convs.append(SAGEConv(in_dim, d_graph))
            in_dim = d_graph
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
