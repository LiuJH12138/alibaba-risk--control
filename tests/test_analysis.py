"""Stage 3a analysis-layer unit tests (curve plotting + centrality)."""
import json
from pathlib import Path
import pytest


def test_plot_curves_generates_png(tmp_path):
    from src.analysis.plot_curves import plot_curves
    history = [
        {"epoch": e, "lr": 1e-3, "train_loss": 1.0 / e, "epoch_seconds": 5.0,
         "val_roc_auc": 0.8 + 0.01 * e, "val_pr_auc": 0.5 + 0.02 * e,
         "val_ks": 0.6 + 0.01 * e, "val_recall_at_fpr_0.01": 0.4 + 0.01 * e}
        for e in range(1, 11)
    ]
    h_path = tmp_path / "training_history_dummy.json"
    h_path.write_text(json.dumps(history))
    out_png = tmp_path / "curves_dummy.png"
    plot_curves(str(h_path), str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 5000,         f"expected PNG > 5KB, got {out_png.stat().st_size if out_png.exists() else 'missing'}"


def test_centrality_topk_count():
    """identify_fraud_rings on a tiny hetero subgraph must return at most top_k
    entities per type and never NaN scores."""
    import torch
    from torch_geometric.data import HeteroData
    from src.analysis.centrality import identify_fraud_rings

    hg = HeteroData()
    n = 20
    hg["transaction"].num_nodes = n
    hg["transaction"].y = torch.zeros(n, dtype=torch.float32)
    hg["transaction"].y[:6] = 1.0   # 6 frauds
    for col, n_nodes in [("card1", 5), ("addr1", 4), ("P_emaildomain", 3), ("DeviceInfo", 3)]:
        hg[col].num_nodes = n_nodes
    # Each transaction connects to entity (i % n_nodes) of each type
    for col, fwd, rev in [
        ("card1", "paid_with", "rev_paid_with"),
        ("addr1", "shipped_to", "rev_shipped_to"),
        ("P_emaildomain", "sent_to_email", "rev_sent_to_email"),
        ("DeviceInfo", "on_device", "rev_on_device"),
    ]:
        n_e = hg[col].num_nodes
        src = torch.arange(n)
        dst = torch.arange(n) % n_e
        hg["transaction", fwd, col].edge_index = torch.stack([src, dst])
        hg[col, rev, "transaction"].edge_index = torch.stack([dst, src])

    # Use predicted scores directly (skip model loading by passing them in)
    scores = torch.zeros(n)
    scores[:6] = 0.95     # all 6 frauds get high score
    fraud_idx = torch.arange(6)

    out = identify_fraud_rings(
        hetero_graph=hg, fraud_seed_idx=fraud_idx, top_k=2,
        entity_types=("card1", "addr1", "P_emaildomain", "DeviceInfo"),
    )
    for col in ("card1", "addr1", "P_emaildomain", "DeviceInfo"):
        assert col in out
        assert len(out[col]) <= 2
        for entry in out[col]:
            assert "node_idx" in entry and "degree" in entry and "pagerank" in entry
            assert entry["pagerank"] == entry["pagerank"], "no NaN allowed"   # pagerank != NaN
            assert entry["degree"] >= 0
