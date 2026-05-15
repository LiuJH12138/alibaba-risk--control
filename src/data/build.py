import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.config import load_config
from src.data.load import load_raw
from src.data.uid import synthesize_uid
from src.data.features import FeatureProcessor, NUM_COLS as DEFAULT_NUM_COLS
from src.data.sequence import build_sequences
from src.data.graph import build_edges
from src.data.v_pruning import compute_pruned_v_cols

V_PRUNED_CACHE = "data/processed/v_pruned_cols.json"


def time_split(dt: np.ndarray, ratio: float):
    order = np.argsort(dt, kind="stable")
    cut = int(len(dt) * ratio)
    return order[:cut].astype("int64"), order[cut:].astype("int64")


def validate_split(dt, train_idx, val_idx):
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, "split overlap"
    assert dt[train_idx].max() <= dt[val_idx].min(), "time leak: train overlaps val"


def _resolve_num_cols(df_train, v_strategy: str) -> list[str]:
    """根据 v_strategy 决定使用哪些 num cols。pruned 结果缓存。"""
    non_v_num = [c for c in DEFAULT_NUM_COLS if not (c.startswith("V") and c[1:].isdigit())]
    if v_strategy == "full_v":
        v_cols = [f"V{i}" for i in range(1, 340)]
    elif v_strategy == "pruned_v":
        cache = Path(V_PRUNED_CACHE)
        if cache.exists():
            v_cols = json.loads(cache.read_text())
        else:
            v_cols = compute_pruned_v_cols(df_train, threshold=0.95)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(v_cols))
        print(f"pruned_v: kept {len(v_cols)} of 339 V cols")
    else:
        raise ValueError(f"unknown v_strategy: {v_strategy}")
    return non_v_num + v_cols


def build_all(v_strategy: str = "full_v"):
    """端到端构建。v_strategy ∈ {full_v, pruned_v}。落 data/processed/{v_strategy}/。"""
    cfg = load_config("data")
    out = Path(cfg["processed_dir"]) / v_strategy
    out.mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg["raw_dir"])
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy()
    y = df["isFraud"].to_numpy().astype("float32")
    df["uid"] = synthesize_uid(df)

    train_idx, val_idx = time_split(dt, cfg["split_ratio"])
    validate_split(dt, train_idx, val_idx)

    num_cols = _resolve_num_cols(df.iloc[train_idx], v_strategy)
    fp = FeatureProcessor(num_cols=num_cols)         # 仍用默认 cat_cols
    fp.fit(df.iloc[train_idx])
    feat = fp.transform(df)                          # dict: {cat_idx, num}

    cat_x = torch.from_numpy(feat["cat_idx"])        # int64 [N, n_cat]
    num_x = torch.from_numpy(feat["num"])            # float32 [N, n_num*2]

    # 序列:对 cat 和 num 分别构造,共享 mask
    seq_cat, mask = build_sequences(feat["cat_idx"].astype("float32"),
                                    df["uid"].to_numpy(), dt, cfg["seq_len"])
    seq_num, _   = build_sequences(feat["num"],
                                    df["uid"].to_numpy(), dt, cfg["seq_len"])
    seq_cat = seq_cat.astype("int64")                # 还原 dtype

    src, dst = build_edges(df, cfg["graph_entity_cols"],
                           cfg["graph_max_degree"], cfg["graph_max_neighbors_per_entity"])

    graph = Data(
        cat_x=cat_x,
        num_x=num_x,
        edge_index=torch.from_numpy(np.stack([src, dst])),
        y=torch.from_numpy(y),
        t=torch.from_numpy(dt.astype("int64")),
    )
    torch.save(graph, out / "graph.pt")
    torch.save({"cat": torch.from_numpy(seq_cat),
                "num": torch.from_numpy(seq_num),
                "mask": torch.from_numpy(mask)}, out / "seq_all.pt")
    torch.save({"train_idx": torch.from_numpy(train_idx),
                "val_idx": torch.from_numpy(val_idx)}, out / "split.pt")
    with open(out / "feature_meta.json", "w") as f:
        json.dump(fp.meta, f, indent=2)

    fr = float(y.mean())
    assert abs(fr - cfg["expected_fraud_rate"]) < cfg["fraud_rate_tol"]
    assert graph.edge_index.shape[1] == 0 or (dt[src] <= dt[dst]).all(), \
        "graph has non-time-respecting edges"
    manifest = {
        "v_strategy": v_strategy,
        "n_transactions": int(len(df)), "fraud_rate": fr,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_edges": int(graph.edge_index.shape[1]),
        "n_cat": int(cat_x.shape[1]),
        "n_num_total": int(num_x.shape[1]),
        "seq_len": cfg["seq_len"],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"build done [{v_strategy}]:", manifest)
    return manifest


if __name__ == "__main__":
    for s in ["full_v", "pruned_v"]:
        build_all(s)
