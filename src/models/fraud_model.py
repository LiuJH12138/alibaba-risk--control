import torch
import torch.nn as nn
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.fusion import FusionHead

class FraudModel(nn.Module):
    """双塔欺诈检测模型。
    训练:forward(seq, mask, x, edge_index, seed_idx) —— 图塔在子图上算,取 seed 节点 emb。
    部署:forward_online(seq, mask, graph_emb) —— 图 emb 离线预计算后查表传入。"""

    def __init__(self, feat_dim: int, model_cfg: dict, fusion_mode: str = "gated"):
        super().__init__()
        c = model_cfg
        self.seq_tower = SequenceTower(
            feat_dim=feat_dim, d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_transformer_layers"], d_seq=c["d_seq"], dropout=c["dropout"])
        self.graph_tower = GraphTower(
            feat_dim=feat_dim, d_graph=c["d_graph"],
            n_layers=c["graphsage_layers"], dropout=c["dropout"])
        self.fusion = FusionHead(
            d_seq=c["d_seq"], d_graph=c["d_graph"], d_fuse=c["d_fuse"],
            mlp_hidden=c["mlp_hidden"], mode=fusion_mode, dropout=c["dropout"])

    def forward(self, seq, mask, x, edge_index, seed_idx):
        seq_emb = self.seq_tower(seq, mask)
        graph_emb_all = self.graph_tower(x, edge_index)
        graph_emb = graph_emb_all[seed_idx]
        return self.fusion(seq_emb, graph_emb)

    def forward_online(self, seq, mask, graph_emb):
        seq_emb = self.seq_tower(seq, mask)
        return self.fusion(seq_emb, graph_emb)
