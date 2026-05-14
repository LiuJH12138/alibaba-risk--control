import torch
from torch_geometric.data import Data
from src.train import train_one_config

def test_train_one_config_runs_and_returns_metrics(tmp_path):
    n = 200
    graph = Data(x=torch.randn(n, 12),
                 edge_index=torch.randint(0, n, (2, 600)),
                 y=(torch.rand(n) > 0.85).float(),
                 t=torch.arange(n))
    seq_all = {"seq": torch.randn(n, 8, 12), "mask": torch.ones(n, 8, dtype=torch.bool)}
    split = {"train_idx": torch.arange(0, 150), "val_idx": torch.arange(150, n)}
    result = train_one_config(
        graph, seq_all, split, fusion_mode="gated", use_hnm=True,
        model_cfg={"d_model": 16, "n_heads": 2, "n_transformer_layers": 1,
                   "d_seq": 12, "d_graph": 12, "graphsage_layers": 2,
                   "d_fuse": 8, "mlp_hidden": 4, "dropout": 0.0},
        train_cfg={"batch_size": 32, "lr": 1e-3, "weight_decay": 0.0, "epochs": 2,
                   "warmup_steps": 5, "grad_clip": 1.0, "seed": 42,
                   "neighbor_sample": [5, 5], "focal_gamma_pos": 1.0,
                   "focal_gamma_neg": 4.0, "focal_alpha": 0.25,
                   "hnm_neg_pos_ratio": 3.0, "early_stop_patience": 5},
        device="cpu")
    assert "roc_auc" in result and "pr_auc" in result
    assert 0.0 <= result["roc_auc"] <= 1.0
