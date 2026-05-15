import torch
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.fusion import FusionHead
from src.models.fraud_model import FraudModel
from src.models.embedding_mixer import EmbeddingMixer
from src.dataset import make_loader
from torch_geometric.data import Data

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

def test_fraud_model_train_forward():
    model = FraudModel(cat_cardinalities=[5, 7, 4], n_num_total=8, model_cfg={
        "d_model": 32, "n_heads": 4, "n_transformer_layers": 1, "d_seq": 24,
        "d_graph": 24, "graphsage_layers": 2, "d_fuse": 16, "mlp_hidden": 8,
        "dropout": 0.0, "cat_emb_dim": 4}, fusion_mode="gated")
    seq_cat = torch.randint(0, 4, (6, 10, 3))
    seq_num = torch.randn(6, 10, 8)
    mask = torch.ones(6, 10, dtype=torch.bool)
    x_cat = torch.randint(0, 4, (30, 3))
    x_num = torch.randn(30, 8)
    edge_index = torch.randint(0, 30, (2, 60))
    seed = torch.arange(6)
    logit = model(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed)
    assert logit.shape == (6,)

def test_fraud_model_online_forward_uses_precomputed_graph_emb():
    model = FraudModel(cat_cardinalities=[5, 7], n_num_total=4, model_cfg={
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1, "d_seq": 12,
        "d_graph": 12, "graphsage_layers": 2, "d_fuse": 8, "mlp_hidden": 4,
        "dropout": 0.0, "cat_emb_dim": 4}, fusion_mode="gated").eval()
    seq_cat = torch.randint(0, 4, (3, 5, 2))
    seq_num = torch.randn(3, 5, 4)
    mask = torch.ones(3, 5, dtype=torch.bool)
    graph_emb = torch.randn(3, 12)
    logit = model.forward_online(seq_cat, seq_num, mask, graph_emb)
    assert logit.shape == (3,)
    # 梯度流:train forward 下 mixer + 两塔都有梯度
    model.train()
    x_cat = torch.randint(0, 4, (10, 2))
    x_num = torch.randn(10, 4)
    edge_index = torch.randint(0, 10, (2, 20))
    out = model(seq_cat, seq_num, mask, x_cat, x_num, edge_index, torch.arange(3))
    out.sum().backward()
    assert model.mixer.embeddings[0].weight.grad is not None
    assert model.seq_tower.input_proj.weight.grad is not None
    assert model.graph_tower.convs[0].lin_l.weight.grad is not None

def test_loader_yields_aligned_seq_and_seeds():
    n = 40
    graph = Data(cat_x=torch.randint(0, 3, (n, 5)),       # 5 个 cat 字段
                 num_x=torch.randn(n, 8),                 # 8 num 维(已含 isna)
                 edge_index=torch.randint(0, n, (2, 120)),
                 y=(torch.rand(n) > 0.9).float(),
                 t=torch.arange(n))
    seq_all = {"cat": torch.randint(0, 3, (n, 6, 5)),
               "num": torch.randn(n, 6, 8),
               "mask": torch.ones(n, 6, dtype=torch.bool)}
    idx = torch.arange(0, 20)
    loader = make_loader(graph, seq_all, idx, batch_size=8,
                         neighbor_sample=[10, 5], shuffle=False)
    batch = next(iter(loader))
    # dict 接口:含 x_cat/x_num/seq_cat/seq_num/mask/seed_local/label/edge_index
    expected_keys = {"x_cat", "x_num", "edge_index", "seed_local",
                     "seq_cat", "seq_num", "mask", "label"}
    assert set(batch.keys()) == expected_keys
    # 形状一致性
    assert batch["seq_cat"].shape[0] == batch["label"].shape[0]
    assert batch["seq_cat"].shape[0] <= 8
    assert batch["seed_local"].max() < batch["x_cat"].shape[0]

def test_embedding_mixer_output_shape_2d_and_3d():
    mixer = EmbeddingMixer(cat_cardinalities=[5, 10, 7], cat_emb_dim=4, n_num_total=8)
    # 2D input: [B, n_cat] / [B, n_num_total]
    cat = torch.tensor([[1, 5, 3], [4, 0, 6], [2, 8, 0]])
    num = torch.randn(3, 8)
    out = mixer(cat, num)
    assert out.shape == (3, 3 * 4 + 8)   # 12 + 8 = 20

    # 3D input: [B, L, n_cat] / [B, L, n_num_total]
    cat3 = torch.tensor([[[1, 5, 3], [2, 0, 6]]])
    num3 = torch.randn(1, 2, 8)
    out3 = mixer(cat3, num3)
    assert out3.shape == (1, 2, 20)
    assert mixer.out_dim == 20

def test_embedding_mixer_handles_unknown_index_zero():
    mixer = EmbeddingMixer(cat_cardinalities=[5, 10], cat_emb_dim=4, n_num_total=2)
    cat = torch.tensor([[0, 0]])     # 双 unknown 桶
    num = torch.zeros(1, 2)
    out = mixer(cat, num)
    assert out.shape == (1, 2 * 4 + 2)
    assert torch.isfinite(out).all()
