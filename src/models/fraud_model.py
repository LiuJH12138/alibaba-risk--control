import torch
import torch.nn as nn
from src.models.embedding_mixer import EmbeddingMixer
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.hetero_graph_tower import HeteroGraphTower
from src.models.fusion import FusionHead


class FraudModel(nn.Module):
    """Two-tower fraud detection (Stage 3a: optional heterogeneous graph backbone).

    forward signatures by graph_backbone:
        homo  : forward(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx)
        hetero: forward_hetero(seq_cat, seq_num, mask, hetero_data, seed_local)
    Plus the unchanged deployment path:
        forward_online(seq_cat, seq_num, mask, graph_emb)
    """

    def __init__(self, cat_cardinalities, n_num_total: int, model_cfg: dict,
                 fusion_mode: str = "gated", graph_backbone: str = "homo"):
        super().__init__()
        c = model_cfg
        self.graph_backbone = graph_backbone
        self.mixer = EmbeddingMixer(cat_cardinalities, c["cat_emb_dim"], n_num_total)
        feat_dim = self.mixer.out_dim
        self.seq_tower = SequenceTower(
            feat_dim=feat_dim, d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_transformer_layers"], d_seq=c["d_seq"], dropout=c["dropout"])
        if graph_backbone == "homo":
            self.graph_tower = GraphTower(
                feat_dim=feat_dim, d_graph=c["d_graph"],
                n_layers=c["graphsage_layers"], dropout=c["dropout"])
            graph_out_dim = c["d_graph"]
        elif graph_backbone == "hetero":
            self.graph_tower = HeteroGraphTower(
                mixer_out_dim=feat_dim,
                d_graph=c["hetero_d_graph"],
                n_layers=c["hetero_n_layers"],
                dropout=c.get("hetero_dropout", c["dropout"]),
                conv_type=c.get("hetero_conv_type", "sage"),
            )
            graph_out_dim = c["hetero_d_graph"]
        else:
            raise ValueError(f"unknown graph_backbone: {graph_backbone}")
        self.fusion = FusionHead(
            d_seq=c["d_seq"], d_graph=graph_out_dim, d_fuse=c["d_fuse"],
            mlp_hidden=c["mlp_hidden"], mode=fusion_mode, dropout=c["dropout"])

    # --- homogeneous backbone forward (Stage 1/2 path, unchanged signature) ---
    def forward(self, seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx):
        seq = self.mixer(seq_cat, seq_num)
        x = self.mixer(x_cat, x_num)
        seq_emb = self.seq_tower(seq, mask)
        graph_emb_all = self.graph_tower(x, edge_index)
        return self.fusion(seq_emb, graph_emb_all[seed_idx])

    # --- heterogeneous backbone forward (Stage 3a) ---
    def forward_hetero(self, seq_cat, seq_num, mask, hetero_data, seed_local):
        seq = self.mixer(seq_cat, seq_num)
        # Mix transaction node features (cat_x, num_x) once and pass into the tower
        txn_mixed = self.mixer(hetero_data["transaction"].cat_x,
                               hetero_data["transaction"].num_x)
        seq_emb = self.seq_tower(seq, mask)
        graph_emb = self.graph_tower(hetero_data, txn_mixed, seed_local)
        return self.fusion(seq_emb, graph_emb)

    def forward_online(self, seq_cat, seq_num, mask, graph_emb):
        seq = self.mixer(seq_cat, seq_num)
        return self.fusion(self.seq_tower(seq, mask), graph_emb)
