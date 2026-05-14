import torch
from src.models.sequence_tower import SequenceTower

def test_sequence_tower_output_shape():
    tower = SequenceTower(feat_dim=16, d_model=32, n_heads=4,
                          n_layers=2, d_seq=24, dropout=0.0)
    seq = torch.randn(8, 10, 16)
    mask = torch.ones(8, 10, dtype=torch.bool)
    out = tower(seq, mask)
    assert out.shape == (8, 24)

def test_sequence_tower_respects_padding_mask():
    tower = SequenceTower(feat_dim=4, d_model=16, n_heads=2,
                          n_layers=1, d_seq=8, dropout=0.0).eval()
    seq = torch.randn(1, 6, 4)
    mask = torch.tensor([[False, False, True, True, True, True]])
    out_a = tower(seq, mask)
    # 改动被 mask 掉的 padding 位置,输出应不变
    seq2 = seq.clone(); seq2[0, 0] = torch.randn(4)
    out_b = tower(seq2, mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)

from src.models.graph_tower import GraphTower
from torch_geometric.data import Data

def test_graph_tower_output_shape():
    tower = GraphTower(feat_dim=16, d_graph=24, n_layers=2, dropout=0.0)
    x = torch.randn(20, 16)
    edge_index = torch.randint(0, 20, (2, 50))
    out = tower(x, edge_index)
    assert out.shape == (20, 24)

def test_graph_tower_handles_no_edges():
    tower = GraphTower(feat_dim=8, d_graph=12, n_layers=2, dropout=0.0)
    x = torch.randn(5, 8)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    out = tower(x, edge_index)
    assert out.shape == (5, 12)
    assert torch.isfinite(out).all()
