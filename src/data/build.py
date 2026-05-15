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


def time_split(dt: np.ndarray, ratio: float):
    order = np.argsort(dt, kind="stable")
    cut = int(len(dt) * ratio)
    return order[:cut].astype("int64"), order[cut:].astype("int64")


def validate_split(dt, train_idx, val_idx):
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, "split overlap"
    assert dt[train_idx].max() <= dt[val_idx].min(), "time leak: train overlaps val"


def _resolve_num_cols(df_train, v_strategy: str, processed_dir: str) -> list[str]:
    """根据 v_strategy 决定使用哪些 num cols。pruned 结果缓存。"""
    non_v_num = [c for c in DEFAULT_NUM_COLS if not (c.startswith("V") and c[1:].isdigit())]
    if v_strategy == "full_v":
        v_cols = [f"V{i}" for i in range(1, 340)]
    elif v_strategy == "pruned_v":
        cache = Path(processed_dir) / "v_pruned_cols.json"
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

    num_cols = _resolve_num_cols(df.iloc[train_idx], v_strategy, cfg["processed_dir"])
    fp = FeatureProcessor(num_cols=num_cols)         # 仍用默认 cat_cols
    fp.fit(df.iloc[train_idx])
    feat = fp.transform(df)                          # dict: {cat_idx, num}

    cat_x = torch.from_numpy(feat["cat_idx"])        # int64 [N, n_cat]
    num_x = torch.from_numpy(feat["num"])            # float32 [N, n_num*2]

    # 序列:对 cat 和 num 分别构造,共享 mask
    seq_cat, mask = build_sequences(feat["cat_idx"].astype("float32"),
                                    df["uid"].to_numpy(), dt, cfg["seq_len"])
    seq_num, mask2 = build_sequences(feat["num"],
                                      df["uid"].to_numpy(), dt, cfg["seq_len"])
    # 两次 build_sequences 用同 uid/dt,mask 应严格一致 —— 显式断言防 build_sequences
    # 未来加 feat-shape-依赖逻辑时静默引入 bug
    assert np.array_equal(mask, mask2), "masks diverged between cat and num seq builds"
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
    assert abs(fr - cfg["expected_fraud_rate"]) < cfg["fraud_rate_tol"], \
        f"fraud rate {fr} off expected"
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

from torch_geometric.data import HeteroData
from src.data.entity_stats import compute_all_entity_features, ENTITY_NA_SENTINEL, COLD_START_SENTINEL


def _entity_id_to_idx(entity_block: dict) -> dict:
    """ids list -> {value: row index}; cold-start sentinel maps to last row."""
    return {eid: i for i, eid in enumerate(entity_block["ids"])}


def build_hetero_graph(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    txn_cat_x: torch.Tensor,
    txn_num_x: torch.Tensor,
    entity_cols: list[str],
    amt_col: str = "TransactionAmt",
    dt_col: str = "TransactionDT",
    label_col: str = "isFraud",
) -> HeteroData:
    """Build PyG HeteroData with 5 node types and 9 edge_index groups.

    Node types:
        transaction (N rows)         feats = (cat_x, num_x) — Mixer-ready
        card1, addr1, P_emaildomain, DeviceInfo
                                     feats = float32 [n_unique+1, 5] (last row = cold-start)

    Edge types (bidirectional unless marked):
        transaction -> entity (paid_with / shipped_to / sent_to_email / on_device) + reverse
        transaction -> transaction next_by_uid (time-respecting, single direction)
    """
    n = len(df)
    assert txn_cat_x.shape[0] == n and txn_num_x.shape[0] == n, \
        "txn feature row count must equal df row count"
    hg = HeteroData()

    # --- transaction nodes (features stored as Mixer-ready cat/num pair) ---
    hg["transaction"].cat_x = txn_cat_x
    hg["transaction"].num_x = txn_num_x
    hg["transaction"].num_nodes = n
    hg["transaction"].t = torch.from_numpy(df[dt_col].to_numpy().astype("int64"))
    hg["transaction"].y = torch.from_numpy(df[label_col].to_numpy().astype("float32"))

    # --- entity nodes ---
    feats = compute_all_entity_features(
        df=df, train_idx=train_idx, val_idx=val_idx,
        entity_cols=entity_cols, amt_col=amt_col, dt_col=dt_col, label_col=label_col,
    )
    id2idx = {col: _entity_id_to_idx(feats[col]) for col in entity_cols}
    cold_idx = {col: id2idx[col][COLD_START_SENTINEL] for col in entity_cols}
    for col in entity_cols:
        hg[col].x = torch.from_numpy(feats[col]["x"])
        hg[col].num_nodes = len(feats[col]["ids"])

    # --- transaction -> entity edges (4 relations, bidirectional) ---
    rel_names = {
        "card1":         ("paid_with",     "rev_paid_with"),
        "addr1":         ("shipped_to",    "rev_shipped_to"),
        "P_emaildomain": ("sent_to_email", "rev_sent_to_email"),
        "DeviceInfo":    ("on_device",     "rev_on_device"),
    }
    for col, (fwd_name, rev_name) in rel_names.items():
        vals = df[col].fillna(ENTITY_NA_SENTINEL).astype(str).to_numpy()
        # Map each transaction's entity value -> entity-node row index;
        # unseen entities (not in train) fall back to cold-start row.
        col_idx = id2idx[col]
        cold = cold_idx[col]
        dst_arr = np.fromiter((col_idx.get(v, cold) for v in vals),
                              dtype=np.int64, count=n)
        src_arr = np.arange(n, dtype=np.int64)
        ei_fwd = torch.from_numpy(np.stack([src_arr, dst_arr]))
        ei_rev = torch.from_numpy(np.stack([dst_arr, src_arr]))
        hg["transaction", fwd_name, col].edge_index = ei_fwd
        hg[col, rev_name, "transaction"].edge_index = ei_rev

    # --- next_by_uid (transaction -> transaction, time-respecting) ---
    if "uid" in df.columns:
        order = np.argsort(df[dt_col].to_numpy(), kind="stable")
        sorted_uids = df["uid"].to_numpy()[order]
        sorted_pos = order
        # Within each uid group consecutive (i -> i+1) edges by sort position
        src_list, dst_list = [], []
        prev_uid, prev_pos = None, None
        for uid_val, pos in zip(sorted_uids, sorted_pos):
            if uid_val == prev_uid and prev_pos is not None:
                src_list.append(prev_pos)
                dst_list.append(pos)
            prev_uid, prev_pos = uid_val, pos
        if src_list:
            ei = torch.from_numpy(np.stack([np.array(src_list, dtype=np.int64),
                                            np.array(dst_list, dtype=np.int64)]))
        else:
            ei = torch.empty((2, 0), dtype=torch.int64)
        hg["transaction", "next_by_uid", "transaction"].edge_index = ei
    else:
        # If caller did not synthesize uid, emit empty edge group (still a valid edge type)
        hg["transaction", "next_by_uid", "transaction"].edge_index = torch.empty((2, 0), dtype=torch.int64)

    return hg
