import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """融合序列与图 embedding 并输出欺诈 logit。
    mode: seq_only / graph_only / concat / gated(消融用,共用代码路径)。
    gated: gate = sigmoid(W[s;g]);fused = gate*s + (1-gate)*g(逐维门控)。"""

    def __init__(self, d_seq: int, d_graph: int, d_fuse: int,
                 mlp_hidden: int, mode: str = "gated", dropout: float = 0.1):
        super().__init__()
        assert mode in {"seq_only", "graph_only", "concat", "gated"}
        self.mode = mode
        self.seq_proj = nn.Linear(d_seq, d_fuse)
        self.graph_proj = nn.Linear(d_graph, d_fuse)
        if mode == "gated":
            self.gate = nn.Linear(2 * d_fuse, d_fuse)
        head_in = 2 * d_fuse if mode == "concat" else d_fuse
        self.mlp = nn.Sequential(
            nn.Linear(head_in, mlp_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1),
        )

    def forward(self, seq_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        s = self.seq_proj(seq_emb)
        g = self.graph_proj(graph_emb)
        if self.mode == "seq_only":
            fused = s
        elif self.mode == "graph_only":
            fused = g
        elif self.mode == "concat":
            fused = torch.cat([s, g], dim=-1)
        else:  # gated
            gate = torch.sigmoid(self.gate(torch.cat([s, g], dim=-1)))
            fused = gate * s + (1 - gate) * g
        return self.mlp(fused).squeeze(-1)
