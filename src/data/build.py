import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.config import load_config
from src.data.load import load_raw
from src.data.uid import synthesize_uid
from src.data.features import FeatureProcessor
from src.data.sequence import build_sequences
from src.data.graph import build_edges


def time_split(dt: np.ndarray, ratio: float):
    """按时间戳排序,前 ratio 作 train,其余作 val。返回 (train_idx, val_idx) int64。"""
    order = np.argsort(dt, kind="stable")
    cut = int(len(dt) * ratio)
    return order[:cut].astype("int64"), order[cut:].astype("int64")


def validate_split(dt, train_idx, val_idx):
    """断言无时间泄漏:train 的最大时间 <= val 的最小时间。"""
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, "split overlap"
    assert dt[train_idx].max() <= dt[val_idx].min(), "time leak: train overlaps val"


def build_all():
    """端到端:加载 → uid → 特征(train fit)→ 序列 → 图 → 切分 → 落盘 → 校验。"""
    cfg = load_config("data")
    out = Path(cfg["processed_dir"]); out.mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg["raw_dir"])
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy()
    y = df["isFraud"].to_numpy().astype("float32")
    df["uid"] = synthesize_uid(df)

    train_idx, val_idx = time_split(dt, cfg["split_ratio"])
    validate_split(dt, train_idx, val_idx)

    # Stage 1 MVP disk constraint: /root/autodl-tmp is ~50 GB.
    # Full V1-V339 + __isna columns (~791 dims) with seq_len=32 and 590K rows
    # would produce a ~60 GB seq_all.pt — exceeding available space.
    # We keep only V1-V50 (plus the non-V numerics), giving feat_dim ≈ 213
    # and seq_all.pt ≈ 16 GB (float32), which fits comfortably.
    reduced_num_cols = (
        ["TransactionAmt", "dist1", "dist2"]
        + [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 16)]
        + [f"V{i}" for i in range(1, 51)]   # V1-V50 only (disk constraint)
    )
    fp = FeatureProcessor(num_cols=reduced_num_cols)   # cat_cols stays default

    fp.fit(df.iloc[train_idx])             # 仅 train fit
    feat_df = fp.transform(df)             # 全量 transform
    feat = feat_df.to_numpy().astype("float32")

    seq, mask = build_sequences(feat, df["uid"].to_numpy(), dt, cfg["seq_len"])
    src, dst = build_edges(df, cfg["graph_entity_cols"],
                           cfg["graph_max_degree"], cfg["graph_max_neighbors_per_entity"])

    graph = Data(
        x=torch.from_numpy(feat),
        edge_index=torch.from_numpy(np.stack([src, dst])),
        y=torch.from_numpy(y),
        t=torch.from_numpy(dt.astype("int64")),
    )
    torch.save(graph, out / "graph.pt")
    torch.save({"seq": torch.from_numpy(seq), "mask": torch.from_numpy(mask)},
               out / "seq_all.pt")
    torch.save({"train_idx": torch.from_numpy(train_idx),
                "val_idx": torch.from_numpy(val_idx)}, out / "split.pt")
    with open(out / "feature_meta.json", "w") as f:
        json.dump(fp.meta, f, indent=2)

    # 校验
    fr = float(y.mean())
    assert abs(fr - cfg["expected_fraud_rate"]) < cfg["fraud_rate_tol"], \
        f"fraud rate {fr} off expected"
    assert graph.edge_index.shape[1] == 0 or \
        (dt[src] <= dt[dst]).all(), "graph has non-time-respecting edges"
    manifest = {
        "n_transactions": int(len(df)), "fraud_rate": fr,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_edges": int(graph.edge_index.shape[1]),
        "feat_dim": int(feat.shape[1]), "seq_len": cfg["seq_len"],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("build done:", manifest)
    return manifest
