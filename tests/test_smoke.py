import torch
from torch_geometric.data import Data
from src.train import train_one_config
from src.baseline_lgbm import train_lgbm_baseline

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

def test_lgbm_baseline_runs(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(300, 10)); y_train = (x_train[:, 0] > 0).astype(float)
    x_val = rng.normal(size=(100, 10)); y_val = (x_val[:, 0] > 0).astype(float)
    metrics, model = train_lgbm_baseline(x_train, y_train, x_val, y_val)
    assert metrics["roc_auc"] > 0.9        # x[:,0] 强信号,应学得到

from src.deploy.export_onnx import export_online_path, verify_onnx_parity

def test_onnx_export_and_parity(tmp_path):
    from src.models.fraud_model import FraudModel
    model = FraudModel(feat_dim=12, model_cfg={
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1, "d_seq": 12,
        "d_graph": 12, "graphsage_layers": 2, "d_fuse": 8, "mlp_hidden": 4,
        "dropout": 0.0}, fusion_mode="gated").eval()
    onnx_path = str(tmp_path / "model.onnx")
    export_online_path(model, feat_dim=12, seq_len=8, d_graph=12, path=onnx_path)
    assert verify_onnx_parity(model, onnx_path, feat_dim=12, seq_len=8, d_graph=12)

from src.deploy.benchmark import benchmark_torch

def test_benchmark_torch_returns_latency_stats():
    from src.models.fraud_model import FraudModel
    model = FraudModel(feat_dim=8, model_cfg={
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1, "d_seq": 8,
        "d_graph": 8, "graphsage_layers": 2, "d_fuse": 8, "mlp_hidden": 4,
        "dropout": 0.0}, fusion_mode="gated").eval()
    stats = benchmark_torch(model, feat_dim=8, seq_len=6, d_graph=8,
                            device="cpu", n_runs=20, warmup=5)
    assert "p50_ms" in stats and "p95_ms" in stats and "p99_ms" in stats
    assert stats["p50_ms"] > 0

def test_end_to_end_pipeline(tiny_raw_df, tmp_path, monkeypatch):
    """微数据集端到端:特征 → 序列 → 图 → 训练 → 评估,全程跑通。"""
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from src.data.uid import synthesize_uid
    from src.data.features import FeatureProcessor
    from src.data.sequence import build_sequences
    from src.data.graph import build_edges
    from src.train import train_one_config

    df = tiny_raw_df.sort_values("TransactionDT").reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy()
    y = df["isFraud"].to_numpy().astype("float32")
    df["uid"] = synthesize_uid(df)

    fp = FeatureProcessor(cat_cols=["ProductCD", "card1", "P_emaildomain", "DeviceInfo"],
                          num_cols=["TransactionAmt", "D1"])
    cut = int(len(df) * 0.8)
    fp.fit(df.iloc[:cut])
    feat = fp.transform(df).to_numpy().astype("float32")

    seq, mask = build_sequences(feat, df["uid"].to_numpy(), dt, seq_len=8)
    src, dst = build_edges(df, ["card1", "P_emaildomain"], max_degree=20, max_per_entity=10)
    graph = Data(x=torch.from_numpy(feat),
                 edge_index=torch.from_numpy(np.stack([src, dst])),
                 y=torch.from_numpy(y), t=torch.from_numpy(dt))
    seq_all = {"seq": torch.from_numpy(seq), "mask": torch.from_numpy(mask)}
    split = {"train_idx": torch.arange(0, cut), "val_idx": torch.arange(cut, len(df))}

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
    assert "roc_auc" in result
