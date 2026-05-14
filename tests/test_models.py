import torch
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.fusion import FusionHead

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

def test_gated_fusion_output_shape():
    head = FusionHead(d_seq=24, d_graph=24, d_fuse=16, mlp_hidden=8, mode="gated")
    logit = head(torch.randn(8, 24), torch.randn(8, 24))
    assert logit.shape == (8,)

def test_seq_only_mode_ignores_graph():
    head = FusionHead(d_seq=12, d_graph=12, d_fuse=8, mlp_hidden=4, mode="seq_only").eval()
    s = torch.randn(4, 12)
    a = head(s, torch.randn(4, 12))
    b = head(s, torch.randn(4, 12))      # 不同 graph 输入
    assert torch.allclose(a, b, atol=1e-6)

def test_all_modes_run():
    for mode in ["seq_only", "graph_only", "concat", "gated"]:
        head = FusionHead(d_seq=12, d_graph=12, d_fuse=8, mlp_hidden=4, mode=mode)
        out = head(torch.randn(3, 12), torch.randn(3, 12))
        assert out.shape == (3,)
