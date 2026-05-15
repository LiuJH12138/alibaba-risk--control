import torch
import torch.nn as nn
from src.models.embedding_mixer import EmbeddingMixer
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.fusion import FusionHead


class FraudModel(nn.Module):
    """双塔欺诈检测模型(Stage 2:per-field cat embedding + 共享 mixer)。
    训练 forward(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx)
    部署 forward_online(seq_cat, seq_num, mask, graph_emb)"""

    def __init__(self, cat_cardinalities, n_num_total: int, model_cfg: dict,
                 fusion_mode: str = "gated"):
        super().__init__()
        c = model_cfg
        self.mixer = EmbeddingMixer(cat_cardinalities, c["cat_emb_dim"], n_num_total)
        feat_dim = self.mixer.out_dim
        self.seq_tower = SequenceTower(
            feat_dim=feat_dim, d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_transformer_layers"], d_seq=c["d_seq"], dropout=c["dropout"])
        self.graph_tower = GraphTower(
            feat_dim=feat_dim, d_graph=c["d_graph"],
            n_layers=c["graphsage_layers"], dropout=c["dropout"])
        self.fusion = FusionHead(
            d_seq=c["d_seq"], d_graph=c["d_graph"], d_fuse=c["d_fuse"],
            mlp_hidden=c["mlp_hidden"], mode=fusion_mode, dropout=c["dropout"])

    def forward(self, seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx):
        seq = self.mixer(seq_cat, seq_num)
        x = self.mixer(x_cat, x_num)
        seq_emb = self.seq_tower(seq, mask)
        graph_emb_all = self.graph_tower(x, edge_index)
        return self.fusion(seq_emb, graph_emb_all[seed_idx])

    def forward_online(self, seq_cat, seq_num, mask, graph_emb):
        seq = self.mixer(seq_cat, seq_num)
        return self.fusion(self.seq_tower(seq, mask), graph_emb)
