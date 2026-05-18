"""Stage 3a fraud-ring identification (post-training analysis).

Workflow used by `run_centrality_for_config(...)` (called by Task 19):
    1. Load trained hetero FraudModel checkpoint
    2. Score val-set transactions
    3. Take prob > prob_threshold as the fraud seed set
    4. Build a NetworkX heterograph projection limited to those transactions
       and their connected entity nodes
    5. Compute degree + PageRank per entity-type node block
    6. Persist top_k per type into experiments/core_entities_<config>.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import networkx as nx
import torch
from torch_geometric.data import HeteroData


def _build_fraud_subgraph_nx(hetero_graph: HeteroData,
                             fraud_seed_idx: torch.Tensor,
                             entity_types: Iterable[str]) -> nx.DiGraph:
    """Project the fraud-only subgraph into a NetworkX DiGraph with typed node ids."""
    g = nx.DiGraph()
    fraud_set = set(int(i) for i in fraud_seed_idx.tolist())
    for tx in fraud_set:
        g.add_node(("transaction", tx))
    for col in entity_types:
        fwd_name = {
            "card1": "paid_with", "addr1": "shipped_to",
            "P_emaildomain": "sent_to_email", "DeviceInfo": "on_device",
        }[col]
        et = ("transaction", fwd_name, col)
        if et not in hetero_graph.edge_types:
            continue
        src, dst = hetero_graph[et].edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            if s in fraud_set:
                g.add_node((col, int(d)))
                g.add_edge(("transaction", int(s)), (col, int(d)))
                g.add_edge((col, int(d)), ("transaction", int(s)))
    return g


def identify_fraud_rings(hetero_graph: HeteroData,
                         fraud_seed_idx: torch.Tensor,
                         top_k: int = 20,
                         entity_types: Iterable[str] = ("card1", "addr1",
                                                        "P_emaildomain", "DeviceInfo"),
                         ) -> dict:
    """Returns {entity_type: list[ {node_idx, degree, pagerank} ]}, sorted by
    PageRank descending, length <= top_k per type. Empty list if no edges of
    that type connect to any fraud seed."""
    g = _build_fraud_subgraph_nx(hetero_graph, fraud_seed_idx, entity_types)
    if g.number_of_nodes() == 0:
        return {col: [] for col in entity_types}

    pr = nx.pagerank(g, alpha=0.85, max_iter=100, tol=1e-6)
    out: dict[str, list[dict]] = {}
    for col in entity_types:
        candidates = [(node, pr.get(node, 0.0), g.degree(node))
                      for node in g.nodes if node[0] == col]
        candidates.sort(key=lambda x: x[1], reverse=True)
        out[col] = [
            {"node_idx": int(n[1]), "pagerank": float(p), "degree": int(d)}
            for n, p, d in candidates[:top_k]
        ]
    return out


@torch.no_grad()
def run_centrality_for_config(checkpoint_path: str, config_name: str,
                              v_strategy: str = "pruned_v",
                              prob_threshold: float = 0.9,
                              top_k: int = 20,
                              device: str = "cuda") -> dict:
    """End-to-end: load checkpoint -> score val -> run centrality -> persist JSON.
    Returns the same dict that gets written to experiments/core_entities_<name>.json."""
    from src.config import load_config
    from src.dataset import make_hetero_loader
    from src.models.fraud_model import FraudModel

    proc_dir = Path("data/processed") / v_strategy
    hetero_graph = torch.load(proc_dir / "hetero_graph.pt", weights_only=False)
    seq_all = torch.load(proc_dir / "seq_all.pt", weights_only=False)
    split = torch.load(proc_dir / "split.pt", weights_only=False)
    manifest = json.loads((proc_dir / "manifest.json").read_text())
    meta = json.loads((proc_dir / "feature_meta.json").read_text())
    cat_cardinalities = [meta["cat_cardinalities"][c] for c in meta["cat_cols"]]
    n_num_total = manifest["n_num_total"]

    model_cfg = load_config("model")
    train_cfg = load_config("train")
    model = FraudModel(cat_cardinalities, n_num_total, model_cfg,
                       fusion_mode="gated", graph_backbone="hetero").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    val_loader = make_hetero_loader(hetero_graph, seq_all, split["val_idx"],
                                    train_cfg["batch_size"], train_cfg["neighbor_sample"],
                                    shuffle=False)

    # Score val and collect global indices of high-prob transactions
    high_prob_global: list[int] = []
    val_idx = split["val_idx"]
    cursor = 0
    for b in val_loader:
        bs = b["seed_local"].shape[0]
        logit = model.forward_hetero(
            b["seq_cat"].to(device), b["seq_num"].to(device), b["mask"].to(device),
            b["hetero_data"].to(device), b["seed_local"].to(device))
        probs = torch.sigmoid(logit).cpu()
        global_ids = val_idx[cursor:cursor + bs]
        cursor += bs
        for p, gid in zip(probs.tolist(), global_ids.tolist()):
            if p > prob_threshold:
                high_prob_global.append(int(gid))

    fraud_seed_idx = torch.tensor(high_prob_global, dtype=torch.int64)
    rings = identify_fraud_rings(hetero_graph, fraud_seed_idx, top_k=top_k)

    out = {
        "config": config_name,
        "checkpoint": checkpoint_path,
        "prob_threshold": prob_threshold,
        "n_high_prob_fraud_seeds": len(high_prob_global),
        "rings_per_type": rings,
    }
    out_path = Path(f"experiments/core_entities_{config_name}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"centrality done [{config_name}]: {len(high_prob_global)} seeds; "
          f"top entities saved to {out_path}")
    return out
