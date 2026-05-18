# Stage 3a — Heterogeneous Graph + Loss Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Stage 2 homogeneous transaction graph to a 5-node-type heterogeneous graph with `HeteroGraphTower`, add three loss variants (asym_balanced / label_smoothing / HNM-diagnostic) producing a 4-config experiment matrix, enforce 5-layer convergence guarantees (epochs 40 / patience 8 / min_epochs 10 / per-epoch history JSON / auto curves PNG / convergence audit), and ship post-hoc fraud-ring identification via centrality on the heterogeneous fraud subgraph — all on `feature/stage3a-hetero-graph` branch.

**Architecture:** PyG `HeteroData` with node types `{transaction, card1, addr1, P_emaildomain, DeviceInfo}` and 9 typed `edge_index` groups; `HeteroGraphTower` wraps `HeteroConv({...: SAGEConv}, aggr='mean')` × 2 layers; `FraudModel` gains a `graph_backbone` switch (`'homo'` keeps Stage 2 path verbatim, `'hetero'` activates the new tower) with the shared `EmbeddingMixer` re-used for transaction nodes; train.py gains `_record_epoch_metrics` + `_convergence_audit` + `run_stage3a_matrix`; new `src/analysis/{plot_curves.py,centrality.py}` produce PNG curves and `core_entities_*.json`.

**Tech Stack:** PyTorch 2.8.0+cu128, PyG 2.6.1 (`HeteroData`, `HeteroConv`, `SAGEConv`, `NeighborLoader` hetero mode), pandas 2.x, networkx 3.x (PageRank), matplotlib 3.x (curves), pytest, IEEE-CIS Fraud Detection dataset (already on disk under `data/processed/pruned_v/`).

**Working directory throughout:** `/root/autodl-tmp/alibaba-risk-control-internship` on AutoDL (SSH alias `autodl`).

**Branch:** `feature/stage3a-hetero-graph` (already created at commit `7e4c124` with the design doc).

---

## Pre-flight (verify environment before starting)

- [ ] **Verify branch and clean tree**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git branch --show-current && \
  git status --short && \
  git log --oneline -3"
```

Expected:
```
feature/stage3a-hetero-graph
(empty git status)
7e4c124 docs: Stage 3a design — heterogeneous graph + loss deepening + convergence guarantees
f01b395 docs: fill in real SHA 292d0b1 in DESIGN_JOURNAL bug #5
292d0b1 fix: build_trt.py uses Stage 2 4-input names + DESIGN_JOURNAL/README correction
```

- [ ] **Verify Stage 2 artifacts present**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  ls data/processed/pruned_v/ && \
  ls artifacts/best_deep_*.pt && \
  cat experiments/stage2_results.json | python -c 'import json,sys; d=json.load(sys.stdin); print(\"deep_pruned pr_auc:\", d[\"deep_pruned\"][\"pr_auc\"])'"
```

Expected: `graph.pt seq_all.pt split.pt feature_meta.json manifest.json v_pruned_cols.json` listed; `best_deep_pruned.pt` present; deep_pruned pr_auc ≈ 0.4312 (PR-AUC; deep_pruned ROC-AUC = 0.8639).

- [ ] **Verify baseline tests pass (Stage 1+2 = 37 tests)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/ -q"
```

Expected: `37 passed in <2 min`. If anything fails, STOP — do not start Stage 3a until baseline is green.

---

## Part A — Configuration scaffold (Task 1)

### Task 1: Update YAML configs for Stage 3a defaults

**Files:**
- Modify: `configs/train.yaml`
- Modify: `configs/model.yaml`
- Modify: `configs/data.yaml`

These config knobs are read by tasks throughout the plan; updating them up front prevents silent fallback to Stage 2 values.

- [ ] **Step 1: Edit `configs/train.yaml` with Stage 3a values**

Replace contents with:

```yaml
batch_size: 1024
lr: 0.001
weight_decay: 0.00001
epochs: 40                  # Stage 3a: was 20, raised for convergence guarantee
warmup_steps: 500           # Stage 3a: kept 500 (was already 500)
grad_clip: 1.0
seed: 42
neighbor_sample: [15, 10]
focal_gamma_pos: 1.0
focal_gamma_neg: 4.0
focal_alpha: 0.25
hnm_neg_pos_ratio: 3.0
early_stop_patience: 8      # Stage 3a: was 4, raised to tolerate val PR-AUC oscillation
min_epochs: 10              # Stage 3a NEW: hard floor before any early-stop allowed
label_smoothing_eps: 0.1    # Stage 3a NEW: only used by hetero_label_smoothing config
```

- [ ] **Step 2: Edit `configs/model.yaml` to add hetero knobs**

Replace contents with:

```yaml
d_model: 128
n_heads: 4
n_transformer_layers: 2
d_seq: 128
d_graph: 128
graphsage_layers: 2
d_fuse: 128
mlp_hidden: 64
dropout: 0.1
cat_emb_dim: 16
graph_backbone: hetero      # Stage 3a NEW: 'homo' (Stage 2 GraphTower) | 'hetero' (HeteroGraphTower)
hetero_d_graph: 64          # Stage 3a NEW: smaller than d_graph because HeteroConv has 9 SAGEConvs per layer
hetero_n_layers: 2          # Stage 3a NEW: number of HeteroConv layers
entity_feat_dim: 5          # Stage 3a NEW: count, mean_amt, std_amt, fraud_rate_train, days_active
```

- [ ] **Step 3: Edit `configs/data.yaml` to add hetero data flag**

Replace contents with:

```yaml
raw_dir: data/raw
processed_dir: data/processed
seq_len: 32
split_ratio: 0.8
graph_entity_cols: [card1, addr1, P_emaildomain, DeviceInfo]
graph_max_degree: 50
graph_max_neighbors_per_entity: 20
expected_fraud_rate: 0.035
fraud_rate_tol: 0.01
build_hetero_graph: true    # Stage 3a NEW: when true, build_all() also produces hetero_graph.pt
```

- [ ] **Step 4: Verify configs load without error**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c 'from src.config import load_config; \
    print(\"train.epochs=\", load_config(\"train\")[\"epochs\"]); \
    print(\"train.min_epochs=\", load_config(\"train\")[\"min_epochs\"]); \
    print(\"model.graph_backbone=\", load_config(\"model\")[\"graph_backbone\"]); \
    print(\"data.build_hetero_graph=\", load_config(\"data\")[\"build_hetero_graph\"])'"
```

Expected output:
```
train.epochs= 40
train.min_epochs= 10
model.graph_backbone= hetero
data.build_hetero_graph= True
```

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add configs/train.yaml configs/model.yaml configs/data.yaml && \
  git commit -m 'config: Stage 3a defaults (epochs 40, patience 8, min_epochs 10, hetero backbone)'"
```

---

## Part B — Data layer (Tasks 2–4)

### Task 2: Entity stats computation module (TDD)

**Files:**
- Create: `src/data/entity_stats.py`
- Modify: `tests/test_data.py` (append at end)

`entity_stats.py` computes per-entity 5-dim aggregates **strictly on training data**, then applies them to the full dataframe (train/val/test) — cold-start entities get train-population means.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
# ===== Stage 3a: entity stats (train-only computation, cold-start fallback) =====
import numpy as np
import pandas as pd
from src.data.entity_stats import compute_entity_stats, compute_all_entity_features


def test_entity_stats_train_only(tiny_raw_df):
    """entity_stats must depend only on train rows; mutating val rows must not change output."""
    df = tiny_raw_df.copy()
    n = len(df)
    train_idx = np.arange(0, int(n * 0.8))
    val_idx = np.arange(int(n * 0.8), n)

    stats_a = compute_entity_stats(df.iloc[train_idx], entity_col="card1",
                                   amt_col="TransactionAmt",
                                   dt_col="TransactionDT",
                                   label_col="isFraud")
    # Now scramble val labels AND val amounts: train-only stats must be unchanged
    df.loc[val_idx, "isFraud"] = 1 - df.loc[val_idx, "isFraud"]
    df.loc[val_idx, "TransactionAmt"] = df.loc[val_idx, "TransactionAmt"] * 1000.0
    stats_b = compute_entity_stats(df.iloc[train_idx], entity_col="card1",
                                   amt_col="TransactionAmt",
                                   dt_col="TransactionDT",
                                   label_col="isFraud")
    pd.testing.assert_frame_equal(stats_a, stats_b)


def test_cold_start_entity_fallback(tiny_raw_df):
    """val/test entities not in train must be filled with train-population mean (no NaN)."""
    df = tiny_raw_df.copy()
    n = len(df)
    train_idx = np.arange(0, int(n * 0.8))
    val_idx = np.arange(int(n * 0.8), n)
    # Inject a brand-new card1 value into val rows
    df.loc[val_idx[0], "card1"] = 999999

    feats = compute_all_entity_features(
        df=df, train_idx=train_idx, val_idx=val_idx,
        entity_cols=["card1", "addr1", "P_emaildomain", "DeviceInfo"],
        amt_col="TransactionAmt", dt_col="TransactionDT", label_col="isFraud",
    )
    # feats is dict[entity_col] -> dict {ids: list, x: np.ndarray [n_unique+1, 5]}
    card1_block = feats["card1"]
    assert "_COLD_" in card1_block["ids"], "cold-start sentinel must exist"
    assert not np.any(np.isnan(card1_block["x"])), "no NaN allowed in entity features"
    # The 999999 entity should map to the cold-start row
    cold_idx = card1_block["ids"].index("_COLD_")
    cold_vec = card1_block["x"][cold_idx]
    # cold-start vector must equal column means of train-entity vectors
    train_only_rows = np.array([card1_block["x"][i] for i, eid in enumerate(card1_block["ids"])
                                if eid != "_COLD_"])
    np.testing.assert_allclose(cold_vec, train_only_rows.mean(axis=0), rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_data.py::test_entity_stats_train_only tests/test_data.py::test_cold_start_entity_fallback -v"
```

Expected: both FAIL with `ModuleNotFoundError: No module named 'src.data.entity_stats'`.

- [ ] **Step 3: Create `src/data/entity_stats.py`**

```python
"""Per-entity aggregated features for Stage 3a heterogeneous graph nodes.

Computes 5-dim feature per entity value, **strictly from training rows**:
    count            - # transactions linked to this entity (log1p z-scored)
    mean_amt         - mean TransactionAmt (log1p z-scored)
    std_amt          - std  TransactionAmt (log1p z-scored)
    fraud_rate_train - share of isFraud==1 (clipped to [0,1])
    days_active      - (last_dt - first_dt) / 86400 (z-scored)

Cold-start entities (in val/test but not train) get the column means of the
train-only entity matrix as a per-feature fallback. This is documented in
DESIGN_JOURNAL v3 §2.3 as the on-line proxy: at inference time, brand-new
entities have no historical risk signal and inherit the population prior.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ENTITY_NA_SENTINEL = "_NA_"          # sentinel for entity-value NaN within an existing column
COLD_START_SENTINEL = "_COLD_"       # synthetic id for entities unseen in train


def _zscore_log1p(x: np.ndarray) -> np.ndarray:
    """log1p then z-score; constant columns become zero."""
    v = np.log1p(np.maximum(x, 0.0))
    mu, sigma = v.mean(), v.std()
    if sigma < 1e-9:
        return np.zeros_like(v)
    return (v - mu) / sigma


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-9:
        return np.zeros_like(x)
    return (x - mu) / sigma


def compute_entity_stats(train_df: pd.DataFrame, entity_col: str,
                         amt_col: str = "TransactionAmt",
                         dt_col: str = "TransactionDT",
                         label_col: str = "isFraud") -> pd.DataFrame:
    """Compute 5-dim aggregate per entity value from TRAIN rows only.

    Returns DataFrame indexed by entity value (with ENTITY_NA_SENTINEL substituted
    for NaN), columns = ['count', 'mean_amt', 'std_amt', 'fraud_rate_train',
    'days_active'] -- all post z-score / clip.
    """
    df = train_df.copy()
    df[entity_col] = df[entity_col].fillna(ENTITY_NA_SENTINEL).astype(str)
    g = df.groupby(entity_col, sort=True)
    raw = pd.DataFrame({
        "count": g.size().astype("float64"),
        "mean_amt": g[amt_col].mean(),
        "std_amt": g[amt_col].std().fillna(0.0),
        "fraud_rate_train": g[label_col].mean().clip(0.0, 1.0),
        "days_active": (g[dt_col].max() - g[dt_col].min()) / 86400.0,
    })
    # Apply per-column normalization; fraud_rate_train stays raw (already in [0,1])
    raw["count"] = _zscore_log1p(raw["count"].to_numpy())
    raw["mean_amt"] = _zscore_log1p(raw["mean_amt"].to_numpy())
    raw["std_amt"] = _zscore_log1p(raw["std_amt"].to_numpy())
    raw["days_active"] = _zscore(raw["days_active"].to_numpy())
    return raw


def compute_all_entity_features(df: pd.DataFrame,
                                train_idx: np.ndarray,
                                val_idx: np.ndarray,
                                entity_cols: list[str],
                                amt_col: str = "TransactionAmt",
                                dt_col: str = "TransactionDT",
                                label_col: str = "isFraud") -> dict:
    """For each entity column produce: stats DataFrame + ids list + dense float32 [n+1, 5] matrix.

    Last row of every entity matrix is the cold-start vector = column means of
    train-entity rows. ids[-1] == COLD_START_SENTINEL.

    Returns: {entity_col: {"ids": list[str], "x": np.ndarray [n+1, 5] float32}}
    """
    train_df = df.iloc[train_idx]
    out = {}
    for col in entity_cols:
        stats = compute_entity_stats(train_df, col, amt_col=amt_col,
                                     dt_col=dt_col, label_col=label_col)
        ids = list(stats.index)
        x = stats.to_numpy().astype("float32")
        cold = x.mean(axis=0).astype("float32")
        ids.append(COLD_START_SENTINEL)
        x = np.vstack([x, cold[None, :]])
        out[col] = {"ids": ids, "x": x}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_data.py::test_entity_stats_train_only tests/test_data.py::test_cold_start_entity_fallback -v"
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/data/entity_stats.py tests/test_data.py && \
  git commit -m 'feat(data): entity_stats — per-entity 5-dim aggregates + cold-start fallback (TDD)'"
```

---

### Task 3: Heterogeneous graph builder (TDD)

**Files:**
- Modify: `src/data/build.py` (add `build_hetero_graph` function)
- Modify: `tests/test_data.py` (append 3 graph-structure tests)

`build_hetero_graph` consumes the same train/val split + processed features as Stage 2 and produces a PyG `HeteroData` with 5 node types and 9 typed edge groups (4 bidirectional entity edges + 1 directed `next_by_uid` time-respecting transaction edge).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
# ===== Stage 3a: heterogeneous graph structure =====
import torch
from torch_geometric.data import HeteroData
from src.data.build import build_hetero_graph


def _tiny_hetero_inputs(tiny_raw_df):
    """Helper that mimics build_all up to the point build_hetero_graph is called."""
    df = tiny_raw_df.copy().sort_values("TransactionDT").reset_index(drop=True)
    df["uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    n = len(df)
    train_idx = np.arange(0, int(n * 0.8))
    val_idx = np.arange(int(n * 0.8), n)
    # transaction node features: dummy [N, 4] (just to exercise wiring)
    txn_cat = torch.zeros(n, 2, dtype=torch.int64)
    txn_num = torch.zeros(n, 4, dtype=torch.float32)
    return df, train_idx, val_idx, txn_cat, txn_num


def test_hetero_graph_node_counts(tiny_raw_df):
    df, train_idx, val_idx, txn_cat, txn_num = _tiny_hetero_inputs(tiny_raw_df)
    hg = build_hetero_graph(
        df=df, train_idx=train_idx, val_idx=val_idx,
        txn_cat_x=txn_cat, txn_num_x=txn_num,
        entity_cols=["card1", "addr1", "P_emaildomain", "DeviceInfo"],
    )
    assert isinstance(hg, HeteroData)
    assert hg["transaction"].num_nodes == len(df)
    for col in ["card1", "addr1", "P_emaildomain", "DeviceInfo"]:
        # train-unique count + 1 cold-start row
        train_unique = df.iloc[train_idx][col].fillna("_NA_").astype(str).nunique()
        assert hg[col].num_nodes == train_unique + 1, \
            f"{col}: expected {train_unique + 1}, got {hg[col].num_nodes}"


def test_hetero_graph_edge_directions(tiny_raw_df):
    df, train_idx, val_idx, txn_cat, txn_num = _tiny_hetero_inputs(tiny_raw_df)
    hg = build_hetero_graph(
        df=df, train_idx=train_idx, val_idx=val_idx,
        txn_cat_x=txn_cat, txn_num_x=txn_num,
        entity_cols=["card1", "addr1", "P_emaildomain", "DeviceInfo"],
    )
    pairs = [
        (("transaction", "paid_with", "card1"),         ("card1", "rev_paid_with", "transaction")),
        (("transaction", "shipped_to", "addr1"),        ("addr1", "rev_shipped_to", "transaction")),
        (("transaction", "sent_to_email", "P_emaildomain"),
                                                        ("P_emaildomain", "rev_sent_to_email", "transaction")),
        (("transaction", "on_device", "DeviceInfo"),    ("DeviceInfo", "rev_on_device", "transaction")),
    ]
    for fwd, rev in pairs:
        assert fwd in hg.edge_types, f"missing forward edge {fwd}"
        assert rev in hg.edge_types, f"missing reverse edge {rev}"
        assert hg[fwd].edge_index.shape[1] == hg[rev].edge_index.shape[1], \
            f"forward/reverse count mismatch on {fwd}"


def test_next_by_uid_time_respecting(tiny_raw_df):
    df, train_idx, val_idx, txn_cat, txn_num = _tiny_hetero_inputs(tiny_raw_df)
    hg = build_hetero_graph(
        df=df, train_idx=train_idx, val_idx=val_idx,
        txn_cat_x=txn_cat, txn_num_x=txn_num,
        entity_cols=["card1", "addr1", "P_emaildomain", "DeviceInfo"],
    )
    et = ("transaction", "next_by_uid", "transaction")
    assert et in hg.edge_types
    src, dst = hg[et].edge_index
    if src.numel() > 0:
        t = torch.from_numpy(df["TransactionDT"].to_numpy())
        assert (t[dst] >= t[src]).all(), "next_by_uid violates time-respecting order"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_data.py::test_hetero_graph_node_counts tests/test_data.py::test_hetero_graph_edge_directions tests/test_data.py::test_next_by_uid_time_respecting -v"
```

Expected: all three FAIL with `ImportError: cannot import name 'build_hetero_graph' from 'src.data.build'`.

- [ ] **Step 3: Add `build_hetero_graph` to `src/data/build.py`**

Append the following function to `src/data/build.py` (do not remove anything):

```python
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
        transaction → entity (paid_with / shipped_to / sent_to_email / on_device) + reverse
        transaction → transaction next_by_uid (time-respecting, single direction)
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

    # --- transaction → entity edges (4 relations, bidirectional) ---
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

    # --- next_by_uid (transaction → transaction, time-respecting) ---
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
        # If caller didn't synthesize uid, emit empty edge group (still a valid edge type)
        hg["transaction", "next_by_uid", "transaction"].edge_index = torch.empty((2, 0), dtype=torch.int64)

    return hg
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_data.py::test_hetero_graph_node_counts tests/test_data.py::test_hetero_graph_edge_directions tests/test_data.py::test_next_by_uid_time_respecting -v"
```

Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/data/build.py tests/test_data.py && \
  git commit -m 'feat(data): build_hetero_graph — 5 node types + 9 edge groups (TDD)'"
```

---

### Task 4: Wire `build_hetero_graph` into `build_all`

**Files:**
- Modify: `src/data/build.py` (`build_all` function tail)

After Stage 2's `build_all` writes `graph.pt`, conditionally also write `hetero_graph.pt` and the entity-feature artifacts. No new tests — covered indirectly by the real-data run in Task 14 plus the existing Task 3 unit tests.

- [ ] **Step 1: Modify the tail of `build_all` in `src/data/build.py`**

Replace the block in `build_all` that begins with `manifest = {` (and runs through `print(...)`) with:

```python
    # --- Stage 3a: optionally also build heterogeneous graph alongside the homo one ---
    if cfg.get("build_hetero_graph", False):
        # uid was already synthesized at the top of this function
        hg = build_hetero_graph(
            df=df, train_idx=train_idx, val_idx=val_idx,
            txn_cat_x=cat_x, txn_num_x=num_x,
            entity_cols=cfg["graph_entity_cols"],
        )
        torch.save(hg, out / "hetero_graph.pt")
        # Also dump entity stats JSON (for traceability) and entity feature matrices (for inspection)
        entity_feats = compute_all_entity_features(
            df=df, train_idx=train_idx, val_idx=val_idx,
            entity_cols=cfg["graph_entity_cols"],
        )
        with open(out / "entity_stats.json", "w") as f:
            json.dump({col: {"n_ids": len(b["ids"]), "feat_shape": list(b["x"].shape)}
                       for col, b in entity_feats.items()}, f, indent=2)
        for col, block in entity_feats.items():
            torch.save({"ids": block["ids"], "x": torch.from_numpy(block["x"])},
                       out / f"entity_features_{col}.pt")
        n_hetero_edges = sum(int(hg[et].edge_index.shape[1]) for et in hg.edge_types)
    else:
        n_hetero_edges = 0

    manifest = {
        "v_strategy": v_strategy,
        "n_transactions": int(len(df)), "fraud_rate": fr,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_edges": int(graph.edge_index.shape[1]),
        "n_hetero_edges": n_hetero_edges,
        "n_cat": int(cat_x.shape[1]),
        "n_num_total": int(num_x.shape[1]),
        "seq_len": cfg["seq_len"],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"build done [{v_strategy}]:", manifest)
    return manifest
```

- [ ] **Step 2: Verify existing data tests still pass (no regression)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_data.py -q"
```

Expected: all data tests PASS (Stage 1+2 baseline + Stage 3a Tasks 2-3 = 14 + 5 = 19 tests).

- [ ] **Step 3: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/data/build.py && \
  git commit -m 'feat(data): build_all also emits hetero_graph.pt + entity_features when enabled'"
```

---

## Part C — Model layer (Tasks 5–7)

### Task 5: HeteroGraphTower module (TDD)

**Files:**
- Create: `src/models/hetero_graph_tower.py`
- Modify: `tests/test_models.py` (append 3 tests)

Defines `EntityProjector` (per-type Linear from 5-dim → d_graph) and `HeteroGraphTower` (2 layers of `HeteroConv` wrapping 9 `SAGEConv` with `aggr='mean'`). Stage 2's `EmbeddingMixer` projection of transaction nodes is performed in `FraudModel.forward` and passed in — the tower itself receives an already-mixed transaction tensor.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
# ===== Stage 3a: HeteroGraphTower =====
import torch
from torch_geometric.data import HeteroData
from src.models.hetero_graph_tower import EntityProjector, HeteroGraphTower


def _tiny_hetero():
    """Build a minimal HeteroData with 10 transactions + 5 card1 + 3 addr1 + 2 email + 2 device."""
    hg = HeteroData()
    hg["transaction"].cat_x = torch.zeros(10, 2, dtype=torch.int64)
    hg["transaction"].num_x = torch.zeros(10, 4, dtype=torch.float32)
    hg["transaction"].num_nodes = 10
    for col, n in [("card1", 5), ("addr1", 3), ("P_emaildomain", 2), ("DeviceInfo", 2)]:
        hg[col].x = torch.randn(n, 5, dtype=torch.float32)
        hg[col].num_nodes = n
    # All 10 transactions point to entity 0 of each type (simplest valid wiring)
    for col, fwd, rev in [
        ("card1", "paid_with", "rev_paid_with"),
        ("addr1", "shipped_to", "rev_shipped_to"),
        ("P_emaildomain", "sent_to_email", "rev_sent_to_email"),
        ("DeviceInfo", "on_device", "rev_on_device"),
    ]:
        src = torch.arange(10)
        dst = torch.zeros(10, dtype=torch.int64)
        hg["transaction", fwd, col].edge_index = torch.stack([src, dst])
        hg[col, rev, "transaction"].edge_index = torch.stack([dst, src])
    # next_by_uid empty (allowed)
    hg["transaction", "next_by_uid", "transaction"].edge_index = torch.empty((2, 0), dtype=torch.int64)
    return hg


def test_entity_projector_per_type_independent():
    proj = EntityProjector(entity_types=("card1", "addr1"), in_dim=5, d_graph=8)
    w_before = proj.proj["addr1"].weight.detach().clone()
    # Mutate card1 weight: addr1 weight must be unchanged
    with torch.no_grad():
        proj.proj["card1"].weight.fill_(99.0)
    w_after = proj.proj["addr1"].weight.detach().clone()
    assert torch.equal(w_before, w_after)


def test_hetero_graph_tower_forward_shape():
    hg = _tiny_hetero()
    tower = HeteroGraphTower(
        mixer_out_dim=12, d_graph=8, n_layers=2,
        entity_types=("card1", "addr1", "P_emaildomain", "DeviceInfo"),
        dropout=0.0,
    )
    txn_mixed = torch.randn(10, 12)
    seed_local = torch.arange(10)
    out = tower(hg, txn_mixed, seed_local)
    assert out.shape == (10, 8), f"expected (10, 8) got {tuple(out.shape)}"


def test_hetero_graph_tower_seed_extraction():
    hg = _tiny_hetero()
    tower = HeteroGraphTower(
        mixer_out_dim=12, d_graph=8, n_layers=1,
        entity_types=("card1", "addr1", "P_emaildomain", "DeviceInfo"),
        dropout=0.0,
    )
    txn_mixed = torch.randn(10, 12)
    # Pick only 3 seeds; output rows must be exactly len(seed_local) and match those positions
    seed_local = torch.tensor([1, 4, 7])
    full = tower(hg, txn_mixed, torch.arange(10))
    sub = tower(hg, txn_mixed, seed_local)
    assert sub.shape == (3, 8)
    # Determinism: same module + same input should yield identical seed slice
    torch.testing.assert_close(sub, full[seed_local])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_models.py::test_entity_projector_per_type_independent tests/test_models.py::test_hetero_graph_tower_forward_shape tests/test_models.py::test_hetero_graph_tower_seed_extraction -v"
```

Expected: all three FAIL with `ModuleNotFoundError: No module named 'src.models.hetero_graph_tower'`.

- [ ] **Step 3: Create `src/models/hetero_graph_tower.py`**

```python
"""Stage 3a Heterogeneous GraphSAGE tower.

Wraps PyG `HeteroConv` over 9 `SAGEConv` instances (one per relation/direction).
Transaction nodes carry `mixer_out_dim`-dimensional embeddings produced upstream
by the shared `EmbeddingMixer` (so the cat+num tower sees the same transaction
representation as the sequence tower). Entity nodes carry pre-computed 5-dim
aggregated statistics projected per-type to `d_graph`.
"""
from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv


class EntityProjector(nn.Module):
    """Per-entity-type Linear projecting 5-dim aggregates to d_graph."""

    def __init__(self, entity_types: Iterable[str], in_dim: int = 5, d_graph: int = 64):
        super().__init__()
        self.proj = nn.ModuleDict({t: nn.Linear(in_dim, d_graph) for t in entity_types})

    def forward(self, x_dict: dict) -> dict:
        # transaction nodes are projected separately by the tower; entity types only here
        return {t: self.proj[t](x) for t, x in x_dict.items() if t != "transaction"}


# Edge schema is fixed: 4 forward + 4 reverse relations + 1 directed transaction-transaction edge.
EDGE_SPEC: list[tuple[str, str, str]] = [
    ("transaction", "paid_with", "card1"),
    ("card1", "rev_paid_with", "transaction"),
    ("transaction", "shipped_to", "addr1"),
    ("addr1", "rev_shipped_to", "transaction"),
    ("transaction", "sent_to_email", "P_emaildomain"),
    ("P_emaildomain", "rev_sent_to_email", "transaction"),
    ("transaction", "on_device", "DeviceInfo"),
    ("DeviceInfo", "rev_on_device", "transaction"),
    ("transaction", "next_by_uid", "transaction"),
]


class HeteroGraphTower(nn.Module):
    """2-layer HeteroConv with mean-aggregation SAGEConv per relation.

    `mean` aggregator is chosen because card1/addr1 entity-degree distribution is
    long-tailed (top 1% holds ~30% of edges); `sum` would let head entities
    dominate the message, drowning out cold/medium ones.
    """

    def __init__(self, mixer_out_dim: int, d_graph: int = 64, n_layers: int = 2,
                 entity_types: Iterable[str] = ("card1", "addr1", "P_emaildomain", "DeviceInfo"),
                 dropout: float = 0.2):
        super().__init__()
        self.entity_types = tuple(entity_types)
        self.entity_proj = EntityProjector(self.entity_types, in_dim=5, d_graph=d_graph)
        self.txn_proj = nn.Linear(mixer_out_dim, d_graph)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(HeteroConv(
                {edge: SAGEConv(d_graph, d_graph, aggr="mean") for edge in EDGE_SPEC},
                aggr="mean",
            ))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hetero_data, txn_mixed_emb: torch.Tensor,
                seed_local: torch.Tensor) -> torch.Tensor:
        x_dict = self.entity_proj(hetero_data.x_dict)
        x_dict["transaction"] = self.txn_proj(txn_mixed_emb)
        for conv in self.convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {t: self.dropout(self.act(x)) for t, x in x_dict.items()}
        return x_dict["transaction"][seed_local]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_models.py::test_entity_projector_per_type_independent tests/test_models.py::test_hetero_graph_tower_forward_shape tests/test_models.py::test_hetero_graph_tower_seed_extraction -v"
```

Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/models/hetero_graph_tower.py tests/test_models.py && \
  git commit -m 'feat(model): HeteroGraphTower + EntityProjector (TDD)'"
```

---

### Task 6: FraudModel `graph_backbone` switch (TDD)

**Files:**
- Modify: `src/models/fraud_model.py`
- Modify: `tests/test_models.py` (append 1 test)

Add `graph_backbone: str = "homo"` parameter; when `"hetero"` instantiate `HeteroGraphTower` instead of `GraphTower` and a second `forward` path that consumes a `HeteroData` + `seed_local` instead of `(x, edge_index, seed_idx)`. Sequence tower / fusion / Mixer signatures unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
# ===== Stage 3a: FraudModel backbone switch =====
from src.models.fraud_model import FraudModel


def test_fraud_model_backbone_switch():
    """Constructing FraudModel with graph_backbone='hetero' must produce a model
    with strictly more parameters than 'homo' and identical seq/fusion submodule
    parameter counts."""
    cat_card = [10, 8, 5]
    n_num = 6
    cfg = {
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1,
        "d_seq": 16, "d_graph": 16, "graphsage_layers": 1,
        "d_fuse": 16, "mlp_hidden": 8, "dropout": 0.0, "cat_emb_dim": 4,
        "hetero_d_graph": 16, "hetero_n_layers": 1, "entity_feat_dim": 5,
    }
    homo = FraudModel(cat_card, n_num, cfg, fusion_mode="gated", graph_backbone="homo")
    hetero = FraudModel(cat_card, n_num, cfg, fusion_mode="gated", graph_backbone="hetero")

    n_homo = sum(p.numel() for p in homo.parameters())
    n_hetero = sum(p.numel() for p in hetero.parameters())
    assert n_hetero > n_homo, (n_hetero, n_homo)
    # Seq tower params must match across backbones
    seq_homo = sum(p.numel() for p in homo.seq_tower.parameters())
    seq_hetero = sum(p.numel() for p in hetero.seq_tower.parameters())
    assert seq_homo == seq_hetero
    # Fusion params must match across backbones
    fus_homo = sum(p.numel() for p in homo.fusion.parameters())
    fus_hetero = sum(p.numel() for p in hetero.fusion.parameters())
    assert fus_homo == fus_hetero
```

- [ ] **Step 2: Run test to verify it fails**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_models.py::test_fraud_model_backbone_switch -v"
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'graph_backbone'`.

- [ ] **Step 3: Modify `src/models/fraud_model.py`**

Replace the entire file with:

```python
import torch
import torch.nn as nn
from src.models.embedding_mixer import EmbeddingMixer
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.hetero_graph_tower import HeteroGraphTower
from src.models.fusion import FusionHead


class FraudModel(nn.Module):
    """Two-tower fraud detection (Stage 3a: optional heterogeneous graph backbone).

    forward signatures by graph_backbone:
        homo  : forward(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx)
        hetero: forward_hetero(seq_cat, seq_num, mask, hetero_data, seed_local)
    Plus the unchanged deployment path:
        forward_online(seq_cat, seq_num, mask, graph_emb)
    """

    def __init__(self, cat_cardinalities, n_num_total: int, model_cfg: dict,
                 fusion_mode: str = "gated", graph_backbone: str = "homo"):
        super().__init__()
        c = model_cfg
        self.graph_backbone = graph_backbone
        self.mixer = EmbeddingMixer(cat_cardinalities, c["cat_emb_dim"], n_num_total)
        feat_dim = self.mixer.out_dim
        self.seq_tower = SequenceTower(
            feat_dim=feat_dim, d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_transformer_layers"], d_seq=c["d_seq"], dropout=c["dropout"])
        if graph_backbone == "homo":
            self.graph_tower = GraphTower(
                feat_dim=feat_dim, d_graph=c["d_graph"],
                n_layers=c["graphsage_layers"], dropout=c["dropout"])
            graph_out_dim = c["d_graph"]
        elif graph_backbone == "hetero":
            self.graph_tower = HeteroGraphTower(
                mixer_out_dim=feat_dim,
                d_graph=c["hetero_d_graph"],
                n_layers=c["hetero_n_layers"],
                dropout=c["dropout"],
            )
            graph_out_dim = c["hetero_d_graph"]
        else:
            raise ValueError(f"unknown graph_backbone: {graph_backbone}")
        self.fusion = FusionHead(
            d_seq=c["d_seq"], d_graph=graph_out_dim, d_fuse=c["d_fuse"],
            mlp_hidden=c["mlp_hidden"], mode=fusion_mode, dropout=c["dropout"])

    # --- homogeneous backbone forward (Stage 1/2 path, unchanged signature) ---
    def forward(self, seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx):
        seq = self.mixer(seq_cat, seq_num)
        x = self.mixer(x_cat, x_num)
        seq_emb = self.seq_tower(seq, mask)
        graph_emb_all = self.graph_tower(x, edge_index)
        return self.fusion(seq_emb, graph_emb_all[seed_idx])

    # --- heterogeneous backbone forward (Stage 3a) ---
    def forward_hetero(self, seq_cat, seq_num, mask, hetero_data, seed_local):
        seq = self.mixer(seq_cat, seq_num)
        # Mix transaction node features (cat_x, num_x) once and pass into the tower
        txn_mixed = self.mixer(hetero_data["transaction"].cat_x,
                               hetero_data["transaction"].num_x)
        seq_emb = self.seq_tower(seq, mask)
        graph_emb = self.graph_tower(hetero_data, txn_mixed, seed_local)
        return self.fusion(seq_emb, graph_emb)

    def forward_online(self, seq_cat, seq_num, mask, graph_emb):
        seq = self.mixer(seq_cat, seq_num)
        return self.fusion(self.seq_tower(seq, mask), graph_emb)
```

- [ ] **Step 4: Run test to verify it passes + no model-test regression**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_models.py -q"
```

Expected: all model tests PASS (Stage 1+2 baseline 12 + Stage 3a Tasks 5-6 = 16 tests).

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/models/fraud_model.py tests/test_models.py && \
  git commit -m 'feat(model): FraudModel graph_backbone switch (homo|hetero) + forward_hetero'"
```

---

### Task 7: Hetero NeighborLoader branch in `dataset.py`

**Files:**
- Modify: `src/dataset.py`

Add a `make_hetero_loader` sibling that returns batches with the same 8-key dict shape as Stage 2 plus a `hetero_data` key carrying the `HeteroData` subgraph; transaction-node fields (`x_cat`, `x_num`) come from inside the subgraph, keeping the rest of the train/eval loops shape-compatible. No new unit test — covered by the smoke run in Task 14 / training tasks.

- [ ] **Step 1: Modify `src/dataset.py`**

Replace the entire file with:

```python
import torch
from torch_geometric.loader import NeighborLoader

def make_loader(graph, seq_all, node_idx, batch_size, neighbor_sample, shuffle=True):
    """Stage 1/2 homogeneous NeighborLoader. Yields 8-key dict:
       x_cat / x_num / edge_index / seed_local / seq_cat / seq_num / mask / label."""
    seq_cat_t = seq_all["cat"]
    seq_num_t = seq_all["num"]
    mask_t = seq_all["mask"]
    y = graph.y

    base = NeighborLoader(graph, num_neighbors=neighbor_sample, input_nodes=node_idx,
                          batch_size=batch_size, shuffle=shuffle)

    class _Wrapped:
        def __init__(self, loader): self.loader = loader
        def __len__(self): return len(self.loader)
        def __iter__(self):
            for b in self.loader:
                bs = b.batch_size
                seed_global = b.n_id[:bs]
                yield {
                    "x_cat": b.cat_x,
                    "x_num": b.num_x,
                    "edge_index": b.edge_index,
                    "seed_local": torch.arange(bs),
                    "seq_cat": seq_cat_t[seed_global],
                    "seq_num": seq_num_t[seed_global],
                    "mask": mask_t[seed_global],
                    "label": y[seed_global],
                }

    return _Wrapped(base)


def make_hetero_loader(hetero_graph, seq_all, node_idx, batch_size, neighbor_sample,
                      shuffle=True):
    """Stage 3a heterogeneous NeighborLoader. Seeds are transaction nodes only.

    `neighbor_sample` is a list (e.g. [15, 10]); the same fan-out is applied to
    every relation type. Yields a dict with the 7 per-transaction keys plus
    `hetero_data` (the sampled HeteroData subgraph) and `seed_local` (positions
    of the seed transactions inside the subgraph's transaction node block).
    """
    seq_cat_t = seq_all["cat"]
    seq_num_t = seq_all["num"]
    mask_t = seq_all["mask"]
    y = hetero_graph["transaction"].y

    # Apply uniform fan-out across all relations of the hetero graph
    fanout = {et: list(neighbor_sample) for et in hetero_graph.edge_types}
    base = NeighborLoader(
        hetero_graph,
        num_neighbors=fanout,
        input_nodes=("transaction", node_idx),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    class _Wrapped:
        def __init__(self, loader): self.loader = loader
        def __len__(self): return len(self.loader)
        def __iter__(self):
            for b in self.loader:
                bs = b["transaction"].batch_size
                seed_global = b["transaction"].n_id[:bs]
                yield {
                    "hetero_data": b,
                    "seed_local": torch.arange(bs),
                    "seq_cat": seq_cat_t[seed_global],
                    "seq_num": seq_num_t[seed_global],
                    "mask": mask_t[seed_global],
                    "label": y[seed_global],
                }

    return _Wrapped(base)
```

- [ ] **Step 2: Verify no regression in existing dataset / smoke tests**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_smoke.py tests/test_models.py -q"
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/dataset.py && \
  git commit -m 'feat(data): make_hetero_loader — PyG NeighborLoader hetero branch (7 keys + hetero_data)'"
```

---

## Part D — Loss + Training infrastructure (Tasks 8–11)

### Task 8: Label-smoothing extension in losses.py (TDD)

**Files:**
- Modify: `src/models/losses.py` (add label_smoothing knob to `HybridFocalLoss`)
- Create: `tests/test_train.py` (new file; will hold 3 Stage 3a tests across Tasks 8 + 10)

Adding `label_smoothing_eps` to `HybridFocalLoss` (default 0.0 = Stage 1/2 behavior). When `eps > 0` targets are smoothed via `t' = t*(1 - eps) + 0.5*eps` before BCE/focal computation.

- [ ] **Step 1: Create `tests/test_train.py` with the failing test**

```python
"""Stage 3a train-related unit tests (loss extensions + convergence audit)."""
import pytest
import torch
from src.models.losses import HybridFocalLoss


def test_label_smoothing_loss_value():
    """eps=0 must equal Stage 2 baseline; eps>0 must produce STRICTLY different
    per-sample loss values for a hard positive (logit very high)."""
    logits = torch.tensor([10.0, -10.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss_eps0 = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.25,
                                 label_smoothing_eps=0.0).per_sample(logits, targets)
    loss_eps01 = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.25,
                                  label_smoothing_eps=0.1).per_sample(logits, targets)

    # Both vectors have shape [3]
    assert loss_eps0.shape == loss_eps01.shape == torch.Size([3])
    # Smoothing must move ALL elements (none should stay exactly equal)
    assert not torch.allclose(loss_eps0, loss_eps01), \
        "label_smoothing_eps=0.1 produced identical loss to eps=0"
    # Sanity: with eps=0.1, the loss for the confident-correct positive (logit=10)
    # must be NON-zero and finite (smoothing penalises over-confidence)
    assert torch.isfinite(loss_eps01).all()
    assert loss_eps01[0] > 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_train.py::test_label_smoothing_loss_value -v"
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'label_smoothing_eps'`.

- [ ] **Step 3: Modify `src/models/losses.py` to add label smoothing**

Replace the `HybridFocalLoss` class only (keep `hard_negative_mining` unchanged). The full new class body:

```python
class HybridFocalLoss(nn.Module):
    """Asymmetric Focal Loss with optional label smoothing.

    Args:
        gamma_pos / gamma_neg: focal exponents for the positive / negative class
        alpha:                 positive-class weight (Stage 1/2 = 0.25)
        label_smoothing_eps:   Stage 3a addition; when > 0, targets are smoothed
                               t' = t*(1-eps) + 0.5*eps before BCE/focal compute.
                               eps=0 reproduces Stage 2 behavior bit-for-bit.
        reduction:             'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 alpha: float = 0.25, label_smoothing_eps: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.eps = label_smoothing_eps
        self.reduction = reduction

    def per_sample(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing on the binary targets before BCE
        if self.eps > 0.0:
            t = targets * (1.0 - self.eps) + 0.5 * self.eps
        else:
            t = targets
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        # Use the (post-smoothing) target for p_t / weighting so the formula stays consistent
        p_t = p * t + (1 - p) * (1 - t)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        alpha_t = 2 * self.alpha * targets + 2 * (1 - self.alpha) * (1 - targets)
        return alpha_t * (1 - p_t).clamp(min=1e-6) ** gamma * bce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.per_sample(logits, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

- [ ] **Step 4: Run test + verify Stage 1/2 loss tests still pass**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_losses.py tests/test_train.py::test_label_smoothing_loss_value -v"
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/models/losses.py tests/test_train.py && \
  git commit -m 'feat(loss): HybridFocalLoss label_smoothing_eps option (eps=0 reproduces Stage 2)'"
```

---

### Task 9: HNM diagnostic logging hook (no new test)

**Files:**
- Modify: `src/models/losses.py` (add `hard_negative_mining_with_diagnostics`)

Adds a parallel function that returns the boolean keep mask AND a small diagnostics dict per call: count of pos / neg / kept_neg, mean predicted prob of dropped negatives, mean predicted prob of kept negatives. Used by the `hetero_HNM_root_cause` config to figure out exactly what HNM was discarding when it killed Stage 1's gated_plus_hnm. Behavior of the original `hard_negative_mining` is unchanged.

- [ ] **Step 1: Append `hard_negative_mining_with_diagnostics` to `src/models/losses.py`**

```python
def hard_negative_mining_with_diagnostics(per_sample_loss: torch.Tensor,
                                          targets: torch.Tensor,
                                          neg_pos_ratio: float,
                                          probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Same selection logic as hard_negative_mining, plus a diagnostics dict.

    Args:
        per_sample_loss: [B] per-sample loss values (already detached)
        targets:         [B] binary labels (float)
        neg_pos_ratio:   how many hard negatives to keep per positive
        probs:           [B] predicted P(fraud), used for diagnostics ONLY

    Returns:
        keep:        [B] bool mask, True = participate in backward
        diagnostics: dict with keys
            n_pos, n_neg, n_kept_neg,
            mean_prob_kept_neg, mean_prob_dropped_neg,
            max_prob_dropped_neg
    """
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    keep = pos_mask.clone()
    k = min(int(neg_pos_ratio * max(n_pos, 1)), n_neg)
    if k > 0:
        neg_losses = per_sample_loss.masked_fill(pos_mask, float("-inf"))
        hardest = torch.topk(neg_losses, k).indices
        keep[hardest] = True
    n_kept_neg = int((keep & neg_mask).sum().item())
    dropped_neg_mask = neg_mask & ~keep
    diagnostics = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_kept_neg": n_kept_neg,
        "mean_prob_kept_neg": float(probs[keep & neg_mask].mean().item()) if n_kept_neg > 0 else 0.0,
        "mean_prob_dropped_neg": float(probs[dropped_neg_mask].mean().item()) if dropped_neg_mask.any() else 0.0,
        "max_prob_dropped_neg": float(probs[dropped_neg_mask].max().item()) if dropped_neg_mask.any() else 0.0,
    }
    return keep, diagnostics
```

- [ ] **Step 2: Verify no regression**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_losses.py tests/test_train.py -q"
```

Expected: all PASS (no new test, just no breakage).

- [ ] **Step 3: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/models/losses.py && \
  git commit -m 'feat(loss): hard_negative_mining_with_diagnostics — HNM root-cause logging hook'"
```

---

### Task 10: Convergence audit + per-epoch metrics recorder (TDD)

**Files:**
- Modify: `src/train.py` (add `_convergence_audit`, `_record_epoch_metrics` helpers near top)
- Modify: `tests/test_train.py` (append 2 tests)

Pure-Python helpers — testable without GPU. `_record_epoch_metrics` is a tiny dict factory (no state) used inside `train_one_config` after each epoch's eval. `_convergence_audit` consumes the resulting list and returns a structured warning report.

- [ ] **Step 1: Append failing tests to `tests/test_train.py`**

```python
from src.train import _convergence_audit, _record_epoch_metrics


def test_record_epoch_metrics_keys():
    """Helper must return the canonical 8-key per-epoch row."""
    row = _record_epoch_metrics(
        epoch=3, lr=1e-3, train_loss=0.42, epoch_seconds=12.5,
        eval_metrics={"roc_auc": 0.91, "pr_auc": 0.78, "ks": 0.65,
                      "recall_at_fpr_0.01": 0.55, "fpr_at_recall_0.90": 0.08},
    )
    expected_keys = {"epoch", "lr", "train_loss", "epoch_seconds",
                     "val_roc_auc", "val_pr_auc", "val_ks",
                     "val_recall_at_fpr_0.01"}
    assert expected_keys.issubset(row.keys())
    assert row["epoch"] == 3 and row["val_pr_auc"] == 0.78


def test_convergence_audit_warns_on_late_best():
    """If best PR-AUC is at the last epoch, audit must flag NEEDS_LONGER."""
    history = [
        {"epoch": 1, "val_pr_auc": 0.50},
        {"epoch": 2, "val_pr_auc": 0.60},
        {"epoch": 3, "val_pr_auc": 0.70},
        {"epoch": 4, "val_pr_auc": 0.80},
    ]
    audit = _convergence_audit(history, "test_late")
    assert audit["best_epoch"] == 4
    assert audit["total_epochs"] == 4
    # Late-best AND short total -> should produce TWO warnings
    assert any("末尾" in w or "best_epoch" in w for w in audit["warnings"])
    assert any("早停" in w or "total_epochs" in w or "<15" in w for w in audit["warnings"])


def test_convergence_audit_warns_on_oscillation():
    """If last 5 epochs swing > 0.02 in val_pr_auc, audit must warn UNSTABLE."""
    history = [{"epoch": i, "val_pr_auc": 0.80} for i in range(1, 16)]   # 15 stable epochs
    history += [
        {"epoch": 16, "val_pr_auc": 0.82},
        {"epoch": 17, "val_pr_auc": 0.80},
        {"epoch": 18, "val_pr_auc": 0.85},   # >= 0.03 above min of last 5
        {"epoch": 19, "val_pr_auc": 0.81},
        {"epoch": 20, "val_pr_auc": 0.80},
    ]
    audit = _convergence_audit(history, "test_osc")
    assert audit["total_epochs"] == 20
    assert any("震荡" in w or "oscillat" in w.lower() for w in audit["warnings"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_train.py::test_record_epoch_metrics_keys tests/test_train.py::test_convergence_audit_warns_on_late_best tests/test_train.py::test_convergence_audit_warns_on_oscillation -v"
```

Expected: all three FAIL with `ImportError: cannot import name '_convergence_audit' from 'src.train'`.

- [ ] **Step 3: Insert helpers into `src/train.py`**

Add the following two helpers near the top of `src/train.py`, immediately AFTER the existing `_set_seed` function and BEFORE `_evaluate`:

```python
def _record_epoch_metrics(epoch: int, lr: float, train_loss: float,
                          epoch_seconds: float, eval_metrics: dict) -> dict:
    """Canonical per-epoch row written into training_history_<config>.json."""
    return {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_loss,
        "epoch_seconds": epoch_seconds,
        "val_roc_auc": eval_metrics["roc_auc"],
        "val_pr_auc": eval_metrics["pr_auc"],
        "val_ks": eval_metrics["ks"],
        "val_recall_at_fpr_0.01": eval_metrics["recall_at_fpr_0.01"],
    }


def _convergence_audit(history: list[dict], config_name: str) -> dict:
    """Inspect a training_history list and emit warnings when the run did not
    cleanly converge. Returns a dict suitable for stage3a_results.json:
        best_epoch, total_epochs, last5_pr_auc (list[float]), warnings (list[str])
    Always prints a banner so the warnings appear in run logs."""
    best = max(history, key=lambda h: h["val_pr_auc"])
    best_epoch = best["epoch"]
    total_epochs = len(history)
    last5 = [h["val_pr_auc"] for h in history[-5:]]

    warnings: list[str] = []
    if best_epoch == history[-1]["epoch"]:
        warnings.append("⚠️  best_epoch == 末尾 epoch:模型可能仍在提升,需扩大 epochs 重训")
    if total_epochs < 15:
        warnings.append(f"⚠️  仅训练 {total_epochs} epochs (<15),可能早停过早")
    if len(last5) >= 5 and (max(last5) - min(last5)) > 0.02:
        warnings.append(f"⚠️  末 5 epoch val_pr_auc 震荡 > 0.02 (oscillation),未收敛")

    print(f"[CONVERGENCE AUDIT · {config_name}]")
    print(f"  best_epoch = {best_epoch} / total_epochs_run = {total_epochs}")
    print(f"  last 5 epochs val_pr_auc: {last5}")
    for w in warnings:
        print(f"  {w}")

    return {
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "last5_pr_auc": last5,
        "warnings": warnings,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_train.py -v"
```

Expected: all 4 train tests PASS.

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/train.py tests/test_train.py && \
  git commit -m 'feat(train): _convergence_audit + _record_epoch_metrics helpers (TDD)'"
```

---

### Task 11: `train_one_config_hetero` + `run_stage3a_matrix`

**Files:**
- Modify: `src/train.py` (add hetero-aware training function + Stage 3a matrix runner)

Add a hetero-only training entry point that uses `make_hetero_loader` + `forward_hetero` + per-epoch history JSON + min_epochs floor + post-run audit. Then add `run_stage3a_matrix` that iterates the 4 named configs with explicit memory cleanup between runs.

- [ ] **Step 1: Append the new training and matrix functions to `src/train.py`**

Append AFTER the existing `run_stage2_matrix` function:

```python
import gc
from src.dataset import make_hetero_loader
from src.models.losses import hard_negative_mining_with_diagnostics


@torch.no_grad()
def _evaluate_hetero(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for b in loader:
        logit = model.forward_hetero(
            b["seq_cat"].to(device), b["seq_num"].to(device), b["mask"].to(device),
            b["hetero_data"].to(device), b["seed_local"].to(device))
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(b["label"].cpu().numpy())
    return compute_metrics(np.concatenate(labels), np.concatenate(scores))


def train_one_config_hetero(hetero_graph, seq_all, split, fusion_mode, use_hnm,
                            cat_cardinalities, n_num_total,
                            model_cfg, train_cfg, device="cuda",
                            checkpoint_path: str | None = None,
                            history_path: str | None = None,
                            loss_overrides: dict | None = None,
                            config_name: str = "hetero_run") -> dict:
    """Stage 3a heterogeneous training loop with convergence guarantees.

    Differences from Stage 2 train_one_config:
      - uses make_hetero_loader + model.forward_hetero
      - records per-epoch history -> history_path (JSON)
      - enforces min_epochs floor before early-stop
      - on completion calls _convergence_audit and returns its dict alongside metrics
      - loss_overrides: optional dict overriding train_cfg focal_* / label_smoothing_eps
        for this single run (used by hetero_asym_balanced / hetero_label_smoothing)
    """
    _set_seed(train_cfg["seed"])
    model = FraudModel(cat_cardinalities, n_num_total, model_cfg, fusion_mode,
                       graph_backbone="hetero").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    warmup = train_cfg["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / max(1, warmup)))

    # Resolve loss params (overrides win over train.yaml defaults)
    lo = loss_overrides or {}
    loss_fn = HybridFocalLoss(
        gamma_pos=lo.get("focal_gamma_pos", train_cfg["focal_gamma_pos"]),
        gamma_neg=lo.get("focal_gamma_neg", train_cfg["focal_gamma_neg"]),
        alpha=lo.get("focal_alpha", train_cfg["focal_alpha"]),
        label_smoothing_eps=lo.get("label_smoothing_eps", 0.0),
        reduction="none",
    )

    train_loader = make_hetero_loader(hetero_graph, seq_all, split["train_idx"],
                                      train_cfg["batch_size"], train_cfg["neighbor_sample"])
    val_loader = make_hetero_loader(hetero_graph, seq_all, split["val_idx"],
                                    train_cfg["batch_size"], train_cfg["neighbor_sample"],
                                    shuffle=False)

    history: list[dict] = []
    hnm_diag_history: list[dict] = []     # only filled when use_hnm=True
    best_pr, best_metrics, patience = -1.0, None, 0
    min_epochs = int(train_cfg.get("min_epochs", 0))

    for epoch in range(train_cfg["epochs"]):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        last_diag = None
        for b in train_loader:
            logit = model.forward_hetero(
                b["seq_cat"].to(device), b["seq_num"].to(device), b["mask"].to(device),
                b["hetero_data"].to(device), b["seed_local"].to(device))
            target = b["label"].to(device)
            per_sample = loss_fn.per_sample(logit, target)
            if use_hnm:
                with torch.no_grad():
                    probs = torch.sigmoid(logit.detach())
                keep, diag = hard_negative_mining_with_diagnostics(
                    per_sample.detach(), target,
                    neg_pos_ratio=train_cfg["hnm_neg_pos_ratio"],
                    probs=probs,
                )
                loss = per_sample[keep].mean()
                last_diag = diag
            else:
                loss = per_sample.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            opt.step(); scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        epoch_secs = time.time() - t0
        train_loss = running_loss / max(n_batches, 1)
        eval_metrics = _evaluate_hetero(model, val_loader, device)
        cur_lr = opt.param_groups[0]["lr"]
        history.append(_record_epoch_metrics(
            epoch=epoch + 1, lr=cur_lr, train_loss=train_loss,
            epoch_seconds=epoch_secs, eval_metrics=eval_metrics,
        ))
        if use_hnm and last_diag is not None:
            hnm_diag_history.append({"epoch": epoch + 1, **last_diag})
        if history_path is not None:
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            Path(history_path).write_text(json.dumps(history, indent=2))

        improved = eval_metrics["pr_auc"] > best_pr
        if improved:
            best_pr = eval_metrics["pr_auc"]
            best_metrics = eval_metrics
            patience = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
        # min_epochs floor: never early-stop before epoch >= min_epochs
        if (epoch + 1) >= min_epochs and patience >= train_cfg["early_stop_patience"]:
            break

    audit = _convergence_audit(history, config_name)
    if use_hnm and hnm_diag_history:
        Path(f"experiments/hnm_diagnostics_{config_name}.json").write_text(
            json.dumps(hnm_diag_history, indent=2))
    return {**(best_metrics or {}), "audit": audit, "converged": len(audit["warnings"]) == 0}


STAGE3A_CONFIGS = [
    {"name": "hetero_baseline",
     "use_hnm": False,
     "loss_overrides": {}},
    {"name": "hetero_asym_balanced",
     "use_hnm": False,
     "loss_overrides": {"focal_gamma_pos": 2.0, "focal_gamma_neg": 6.0, "focal_alpha": 0.4}},
    {"name": "hetero_label_smoothing",
     "use_hnm": False,
     "loss_overrides": {"label_smoothing_eps": 0.1}},
    {"name": "hetero_HNM_root_cause",
     "use_hnm": True,
     "loss_overrides": {}},
]


def run_stage3a_matrix(device="cuda", v_strategy: str = "pruned_v",
                       configs: list[dict] | None = None) -> dict:
    """Run all (or a subset of) Stage 3a configs sequentially with explicit
    memory cleanup between runs. Writes experiments/stage3a_results.json after
    each config so partial progress is preserved across crashes."""
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    Path("experiments").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    out_path = Path("experiments/stage3a_results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    cfgs = configs or STAGE3A_CONFIGS

    proc_dir = Path("data/processed") / v_strategy
    manifest = json.loads((proc_dir / "manifest.json").read_text())
    meta = json.loads((proc_dir / "feature_meta.json").read_text())
    cat_cardinalities = [meta["cat_cardinalities"][c] for c in meta["cat_cols"]]
    n_num_total = manifest["n_num_total"]

    for cfg in cfgs:
        name = cfg["name"]
        print(f"\n=== Stage 3a config: {name} ===")
        hetero_graph = torch.load(proc_dir / "hetero_graph.pt", weights_only=False)
        seq_all = torch.load(proc_dir / "seq_all.pt", weights_only=False)
        split = torch.load(proc_dir / "split.pt", weights_only=False)

        ckpt = f"artifacts/best_{name}.pt"
        history_path = f"experiments/training_history_{name}.json"
        t0 = time.time()
        metrics = train_one_config_hetero(
            hetero_graph=hetero_graph, seq_all=seq_all, split=split,
            fusion_mode="gated", use_hnm=cfg["use_hnm"],
            cat_cardinalities=cat_cardinalities, n_num_total=n_num_total,
            model_cfg=model_cfg, train_cfg=train_cfg, device=device,
            checkpoint_path=ckpt, history_path=history_path,
            loss_overrides=cfg["loss_overrides"], config_name=name,
        )
        metrics["train_seconds"] = round(time.time() - t0, 1)
        metrics["v_strategy"] = v_strategy
        metrics["config"] = cfg
        results[name] = metrics
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"{name}: pr_auc={metrics.get('pr_auc')}, converged={metrics['converged']}")

        # --- explicit memory release (avoids the Stage 2 SIGKILL between configs) ---
        del hetero_graph, seq_all, split
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)

    return results
```

- [ ] **Step 2: Verify imports + syntax (no test, just smoke)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c 'from src.train import train_one_config_hetero, run_stage3a_matrix, STAGE3A_CONFIGS; \
    print(\"configs:\", [c[\"name\"] for c in STAGE3A_CONFIGS])'"
```

Expected: `configs: ['hetero_baseline', 'hetero_asym_balanced', 'hetero_label_smoothing', 'hetero_HNM_root_cause']`

- [ ] **Step 3: Verify all train tests still pass**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_train.py -v"
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/train.py && \
  git commit -m 'feat(train): train_one_config_hetero + run_stage3a_matrix (4 configs, history+audit)'"
```

---

## Part E — Analysis (Tasks 12–13)

### Task 12: Training-curve plot generator (TDD)

**Files:**
- Create: `src/analysis/__init__.py`
- Create: `src/analysis/plot_curves.py`
- Create: `tests/test_analysis.py`

Reads a `training_history_<config>.json`, writes a 3-subplot PNG: train_loss / val PR-AUC + ROC-AUC / lr — with a red vertical line at `best_epoch`.

- [ ] **Step 1: Write the failing test in a new `tests/test_analysis.py`**

```python
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
    assert out_png.exists() and out_png.stat().st_size > 5000, \
        f"expected PNG > 5KB, got {out_png.stat().st_size if out_png.exists() else 'missing'}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_analysis.py::test_plot_curves_generates_png -v"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis'`.

- [ ] **Step 3: Create `src/analysis/__init__.py` (empty)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  mkdir -p src/analysis && touch src/analysis/__init__.py"
```

- [ ] **Step 4: Create `src/analysis/plot_curves.py`**

```python
"""Stage 3a training-curve generator.

Produces a 3-subplot PNG with a red vertical line marking `best_epoch`
(the epoch with the highest val_pr_auc). Used after every training run.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # headless backend (no DISPLAY on AutoDL)
import matplotlib.pyplot as plt


def plot_curves(history_json_path: str, out_png: str) -> None:
    history = json.loads(Path(history_json_path).read_text())
    if not history:
        raise ValueError(f"empty history at {history_json_path}")
    epochs = [h["epoch"] for h in history]
    best_epoch = max(history, key=lambda h: h["val_pr_auc"])["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Subplot 1: train loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    axes[0].axvline(best_epoch, color="red", linestyle="--",
                    label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Train loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Subplot 2: PR-AUC + ROC-AUC
    axes[1].plot(epochs, [h["val_pr_auc"] for h in history], label="val PR-AUC", color="C0")
    axes[1].plot(epochs, [h["val_roc_auc"] for h in history], label="val ROC-AUC", color="C1")
    axes[1].axvline(best_epoch, color="red", linestyle="--")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Subplot 3: learning-rate schedule
    axes[2].plot(epochs, [h["lr"] for h in history], label="lr")
    axes[2].axvline(best_epoch, color="red", linestyle="--")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("lr")
    axes[2].set_title("Learning rate"); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(Path(history_json_path).stem)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_analysis.py::test_plot_curves_generates_png -v"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/analysis/__init__.py src/analysis/plot_curves.py tests/test_analysis.py && \
  git commit -m 'feat(analysis): plot_curves — 3-subplot PNG with best_epoch marker (TDD)'"
```

---

### Task 13: Fraud-ring centrality post-processing (TDD)

**Files:**
- Create: `src/analysis/centrality.py`
- Modify: `tests/test_analysis.py` (append 1 test)

Loads a trained hetero model, scores val transactions, extracts the high-confidence-fraud subgraph, runs PageRank + degree centrality on each entity-type node block, writes `experiments/core_entities_<config>.json`. Network conversion uses `networkx`.

- [ ] **Step 1: Append the failing test to `tests/test_analysis.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_analysis.py::test_centrality_topk_count -v"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.centrality'`.

- [ ] **Step 3: Create `src/analysis/centrality.py`**

```python
"""Stage 3a fraud-ring identification (post-training analysis).

Workflow used by `run_centrality_for_config(...)` (called by Task 17):
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
    PageRank descending, length ≤ top_k per type. Empty list if no edges of
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/test_analysis.py -v"
```

Expected: both analysis tests PASS.

- [ ] **Step 5: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add src/analysis/centrality.py tests/test_analysis.py && \
  git commit -m 'feat(analysis): centrality — PageRank + degree on fraud subgraph (TDD)'"
```

---

## Part F — Real-data execution (Tasks 14–19)

### Task 14: Build heterogeneous data artifacts on AutoDL

**Files:**
- Generated: `data/processed/pruned_v/hetero_graph.pt`, `entity_stats.json`, `entity_features_*.pt` (4 files)

Re-runs `build_all` on the existing IEEE-CIS data using the new `build_hetero_graph: true` flag. Heterogeneous artifact lives next to the Stage 2 homogeneous one (no overwrite). Run only `pruned_v` (Stage 2 showed equivalence to full_v).

- [ ] **Step 1: Trigger pruned_v hetero build via background script**

Create `/tmp/build_hetero.sh` on the local machine (then scp + run on AutoDL):

```bash
cat > /tmp/build_hetero.sh <<'EOF'
#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dfer-riskctrl
cd /root/autodl-tmp/alibaba-risk-control-internship
echo "[$(date)] Stage 3a: building hetero graph for pruned_v"
python -u -c "from src.data.build import build_all; build_all('pruned_v')"
echo "[$(date)] build done, exit $?"
echo "--- artifacts ---"
ls -lh data/processed/pruned_v/hetero_graph.pt data/processed/pruned_v/entity_features_*.pt
echo "--- manifest ---"
cat data/processed/pruned_v/manifest.json
EOF
scp /tmp/build_hetero.sh autodl:/tmp/build_hetero.sh
ssh autodl "chmod +x /tmp/build_hetero.sh && nohup /tmp/build_hetero.sh > /tmp/build_hetero.log 2>&1 &"
```

- [ ] **Step 2: Wait for completion + inspect log**

```bash
# Wait until 'build done' appears in the log (typical runtime: 10-25 min for full IEEE-CIS)
ssh autodl "tail -20 /tmp/build_hetero.log"
ssh autodl "ls -lh /root/autodl-tmp/alibaba-risk-control-internship/data/processed/pruned_v/"
```

Expected: `hetero_graph.pt` exists (~300-700 MB), `entity_features_card1.pt`, `entity_features_addr1.pt`, `entity_features_P_emaildomain.pt`, `entity_features_DeviceInfo.pt` all exist; `manifest.json` shows `n_hetero_edges > 0`.

- [ ] **Step 3: Sanity-check the hetero graph contents**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c '
import torch
hg = torch.load(\"data/processed/pruned_v/hetero_graph.pt\", weights_only=False)
print(\"node types:\", hg.node_types)
print(\"edge types:\", hg.edge_types)
print(\"transaction nodes:\", hg[\"transaction\"].num_nodes)
for col in [\"card1\",\"addr1\",\"P_emaildomain\",\"DeviceInfo\"]:
    print(f\"{col}: nodes={hg[col].num_nodes}, feat shape={tuple(hg[col].x.shape)}\")
for et in hg.edge_types:
    print(f\"  edges {et}: {hg[et].edge_index.shape[1]}\")'"
```

Expected: 5 node types, 9 edge types, transaction nodes ~590K, each entity type's node count matches its train uniques + 1, every edge type has > 0 edges (`next_by_uid` may be smaller).

- [ ] **Step 4: Commit the manifest update only (artifacts are git-ignored)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git diff --stat data/ && git status --short"
# data/ is in .gitignore so no commit is needed if status is clean.
# If manifest.json sneaks in (it shouldn't be tracked), explicitly skip it.
```

No commit if `git status` is empty. (Data artifacts are not tracked.)

---

### Task 15: Train `hetero_baseline` config

**Files:**
- Generated: `artifacts/best_hetero_baseline.pt`, `experiments/training_history_hetero_baseline.json`,
  `experiments/curves_hetero_baseline.png`, `experiments/stage3a_results.json` (entry added)

- [ ] **Step 1: Launch training in background**

```bash
cat > /tmp/run_s3a_baseline.sh <<'EOF'
#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dfer-riskctrl
cd /root/autodl-tmp/alibaba-risk-control-internship
echo "[$(date)] Stage 3a: hetero_baseline"
python -u -c "
from src.train import run_stage3a_matrix, STAGE3A_CONFIGS
run_stage3a_matrix(configs=[c for c in STAGE3A_CONFIGS if c['name']=='hetero_baseline'])
"
echo "[$(date)] done, exit $?"
echo "--- stage3a_results.json ---"
cat experiments/stage3a_results.json
EOF
scp /tmp/run_s3a_baseline.sh autodl:/tmp/run_s3a_baseline.sh
ssh autodl "chmod +x /tmp/run_s3a_baseline.sh && nohup /tmp/run_s3a_baseline.sh > /tmp/run_s3a_baseline.log 2>&1 &"
```

- [ ] **Step 2: Monitor until completion (poll every few minutes)**

```bash
ssh autodl "tail -50 /tmp/run_s3a_baseline.log"
ssh autodl "ls -la /root/autodl-tmp/alibaba-risk-control-internship/artifacts/best_hetero_baseline.pt /root/autodl-tmp/alibaba-risk-control-internship/experiments/training_history_hetero_baseline.json 2>&1"
```

Expected end state: log shows `[CONVERGENCE AUDIT · hetero_baseline]` block, then `hetero_baseline: pr_auc=0.XX, converged=True/False`. History JSON exists.

- [ ] **Step 3: Generate the curve PNG for this config**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c 'from src.analysis.plot_curves import plot_curves; \
    plot_curves(\"experiments/training_history_hetero_baseline.json\", \
                \"experiments/curves_hetero_baseline.png\")' && \
  ls -lh experiments/curves_hetero_baseline.png"
```

Expected: `curves_hetero_baseline.png` created, > 30KB.

- [ ] **Step 4: Commit results JSON + curve PNG (training history + checkpoint follow .gitignore)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add experiments/stage3a_results.json experiments/curves_hetero_baseline.png experiments/training_history_hetero_baseline.json && \
  git commit -m 'experiment: Stage 3a hetero_baseline — pr_auc=<from log>, converged=<from log>'"
# Replace <from log> with the actual values pulled from /tmp/run_s3a_baseline.log
```

---

### Task 16: Train `hetero_asym_balanced` config

Same workflow as Task 15, different config name. **Important:** start in a fresh process (the script does that — but we still call `run_stage3a_matrix` standalone, not in the same Python session as the previous run, to avoid the Stage 2 SIGKILL pattern).

- [ ] **Step 1: Launch training in background**

```bash
cat > /tmp/run_s3a_asym.sh <<'EOF'
#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dfer-riskctrl
cd /root/autodl-tmp/alibaba-risk-control-internship
echo "[$(date)] Stage 3a: hetero_asym_balanced"
python -u -c "
from src.train import run_stage3a_matrix, STAGE3A_CONFIGS
run_stage3a_matrix(configs=[c for c in STAGE3A_CONFIGS if c['name']=='hetero_asym_balanced'])
"
echo "[$(date)] done, exit $?"
EOF
scp /tmp/run_s3a_asym.sh autodl:/tmp/run_s3a_asym.sh
ssh autodl "chmod +x /tmp/run_s3a_asym.sh && nohup /tmp/run_s3a_asym.sh > /tmp/run_s3a_asym.log 2>&1 &"
```

- [ ] **Step 2: Monitor + generate curve**

```bash
ssh autodl "tail -50 /tmp/run_s3a_asym.log"
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c 'from src.analysis.plot_curves import plot_curves; \
    plot_curves(\"experiments/training_history_hetero_asym_balanced.json\", \
                \"experiments/curves_hetero_asym_balanced.png\")'"
```

- [ ] **Step 3: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add experiments/stage3a_results.json experiments/curves_hetero_asym_balanced.png experiments/training_history_hetero_asym_balanced.json && \
  git commit -m 'experiment: Stage 3a hetero_asym_balanced — pr_auc=<from log>, converged=<from log>'"
```

---

### Task 17: Train `hetero_label_smoothing` config

- [ ] **Step 1: Launch training in background**

```bash
cat > /tmp/run_s3a_ls.sh <<'EOF'
#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dfer-riskctrl
cd /root/autodl-tmp/alibaba-risk-control-internship
echo "[$(date)] Stage 3a: hetero_label_smoothing"
python -u -c "
from src.train import run_stage3a_matrix, STAGE3A_CONFIGS
run_stage3a_matrix(configs=[c for c in STAGE3A_CONFIGS if c['name']=='hetero_label_smoothing'])
"
echo "[$(date)] done, exit $?"
EOF
scp /tmp/run_s3a_ls.sh autodl:/tmp/run_s3a_ls.sh
ssh autodl "chmod +x /tmp/run_s3a_ls.sh && nohup /tmp/run_s3a_ls.sh > /tmp/run_s3a_ls.log 2>&1 &"
```

- [ ] **Step 2: Monitor + generate curve**

```bash
ssh autodl "tail -50 /tmp/run_s3a_ls.log"
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c 'from src.analysis.plot_curves import plot_curves; \
    plot_curves(\"experiments/training_history_hetero_label_smoothing.json\", \
                \"experiments/curves_hetero_label_smoothing.png\")'"
```

- [ ] **Step 3: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add experiments/stage3a_results.json experiments/curves_hetero_label_smoothing.png experiments/training_history_hetero_label_smoothing.json && \
  git commit -m 'experiment: Stage 3a hetero_label_smoothing — pr_auc=<from log>, converged=<from log>'"
```

---

### Task 18: Train `hetero_HNM_root_cause` config

- [ ] **Step 1: Launch training in background**

```bash
cat > /tmp/run_s3a_hnm.sh <<'EOF'
#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dfer-riskctrl
cd /root/autodl-tmp/alibaba-risk-control-internship
echo "[$(date)] Stage 3a: hetero_HNM_root_cause"
python -u -c "
from src.train import run_stage3a_matrix, STAGE3A_CONFIGS
run_stage3a_matrix(configs=[c for c in STAGE3A_CONFIGS if c['name']=='hetero_HNM_root_cause'])
"
echo "[$(date)] done, exit $?"
EOF
scp /tmp/run_s3a_hnm.sh autodl:/tmp/run_s3a_hnm.sh
ssh autodl "chmod +x /tmp/run_s3a_hnm.sh && nohup /tmp/run_s3a_hnm.sh > /tmp/run_s3a_hnm.log 2>&1 &"
```

- [ ] **Step 2: Monitor + generate curve + read HNM diagnostics**

```bash
ssh autodl "tail -80 /tmp/run_s3a_hnm.log"
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c 'from src.analysis.plot_curves import plot_curves; \
    plot_curves(\"experiments/training_history_hetero_HNM_root_cause.json\", \
                \"experiments/curves_hetero_HNM_root_cause.png\")' && \
  echo '--- HNM diagnostics (per-epoch) ---' && \
  cat experiments/hnm_diagnostics_hetero_HNM_root_cause.json | python -m json.tool | head -40"
```

Expected: HNM diagnostics file lists per-epoch `n_pos`, `n_neg`, `n_kept_neg`, `mean_prob_kept_neg`, `mean_prob_dropped_neg`, `max_prob_dropped_neg` — directly addresses the Stage 1 root-cause question.

- [ ] **Step 3: Commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add experiments/stage3a_results.json experiments/curves_hetero_HNM_root_cause.png experiments/training_history_hetero_HNM_root_cause.json experiments/hnm_diagnostics_hetero_HNM_root_cause.json && \
  git commit -m 'experiment: Stage 3a hetero_HNM_root_cause — pr_auc=<from log>, HNM diagnostics attached'"
```

---

### Task 19: Run centrality on the best config

- [ ] **Step 1: Determine which config was best (highest val_pr_auc)**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c '
import json
r = json.loads(open(\"experiments/stage3a_results.json\").read())
best = max(r.items(), key=lambda kv: kv[1].get(\"pr_auc\", 0.0))
print(\"best config:\", best[0], \"pr_auc:\", best[1][\"pr_auc\"], \"converged:\", best[1][\"converged\"])'"
```

Note the best config name; export it as `BEST_CFG` for the next step.

- [ ] **Step 2: Run centrality post-processing on the best checkpoint**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c '
import json
r = json.loads(open(\"experiments/stage3a_results.json\").read())
best = max(r.items(), key=lambda kv: kv[1].get(\"pr_auc\", 0.0))[0]
from src.analysis.centrality import run_centrality_for_config
run_centrality_for_config(checkpoint_path=f\"artifacts/best_{best}.pt\", config_name=best,
                          v_strategy=\"pruned_v\", prob_threshold=0.9, top_k=20)'"
```

- [ ] **Step 3: Also run centrality on `hetero_baseline` for direct comparison**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  python -c '
from src.analysis.centrality import run_centrality_for_config
run_centrality_for_config(checkpoint_path=\"artifacts/best_hetero_baseline.pt\",
                          config_name=\"hetero_baseline\",
                          v_strategy=\"pruned_v\", prob_threshold=0.9, top_k=20)'"
```

- [ ] **Step 4: Sanity-check JSON outputs and commit**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  ls -la experiments/core_entities_*.json && \
  python -c '
import json, glob
for p in sorted(glob.glob(\"experiments/core_entities_*.json\")):
    d = json.loads(open(p).read())
    print(p, \"-> seeds:\", d[\"n_high_prob_fraud_seeds\"],
          \"top entities per type:\", {k: len(v) for k,v in d[\"rings_per_type\"].items()})'"

ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  git add experiments/core_entities_*.json && \
  git commit -m 'experiment: Stage 3a centrality post-processing (PageRank+degree on fraud subgraph)'"
```

---

## Part G — Documentation + push (Task 20)

### Task 20: DESIGN_JOURNAL v3 + README + final tests + push

**Files:**
- Modify: `docs/DESIGN_JOURNAL.md` (append v3 section; v1 + v2 stay byte-for-byte)
- Modify: `README.md` (append Stage 3a Results section after Stage 2's)
- Push: `feature/stage3a-hetero-graph` to GitHub

- [ ] **Step 1: Read existing DESIGN_JOURNAL to confirm v1+v2 boundaries before append**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  echo '--- DESIGN_JOURNAL last 20 lines (must end with v2 content, no trailing whitespace) ---' && \
  tail -20 docs/DESIGN_JOURNAL.md && \
  echo '' && \
  echo '--- v3 marker should NOT yet exist ---' && \
  grep -n '^# v3' docs/DESIGN_JOURNAL.md || echo 'OK: no v3 yet'"
```

- [ ] **Step 2: Read final stage3a_results.json + audits to populate the v3 result table**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  python -c '
import json
r = json.loads(open(\"experiments/stage3a_results.json\").read())
print(\"| config | val_pr_auc | val_roc_auc | val_ks | converged | best/total |\")
print(\"|---|---|---|---|---|---|\")
for name in [\"hetero_baseline\",\"hetero_asym_balanced\",\"hetero_label_smoothing\",\"hetero_HNM_root_cause\"]:
    if name not in r: continue
    m = r[name]; a = m[\"audit\"]
    print(f\"| {name} | {m[\\\"pr_auc\\\"]:.4f} | {m[\\\"roc_auc\\\"]:.4f} | {m[\\\"ks\\\"]:.4f} | {m[\\\"converged\\\"]} | {a[\\\"best_epoch\\\"]}/{a[\\\"total_epochs\\\"]} |\")
'"
```

Save this output — it will be pasted into both DESIGN_JOURNAL v3 and README.

- [ ] **Step 3: Append v3 to `docs/DESIGN_JOURNAL.md`**

Append this block to the END of `docs/DESIGN_JOURNAL.md` on AutoDL (do NOT overwrite the file; append only). Replace `<TABLE_FROM_STEP_2>` with the markdown table produced above. Replace `<BEST_NAME>`, `<DELTA_VS_S2>`, `<SCENARIO_CHECKBOX>` with the actual values you observe.

```markdown


---

# v3 — Stage 3a: Heterogeneous Graph + Loss Deepening (2026-05-15)

## 设计初衷

Stage 2 完成模型基础升级后,best deep PR-AUC = **0.4370** (deep_full),best LGB PR-AUC = **0.5556** (lgbm_full),差 ≈ **0.12**(注:Stage 2 deep_pruned PR-AUC = 0.4312;ROC-AUC 看上去差距小但 PR-AUC 才是不平衡欺诈检测的真指标)。Stage 2 的诚实结论是仅特征工程升级不足以反超 LGB。Stage 3a 假设差距来自 Stage 2 同构图的两个结构性缺陷:

1. 实体(card1/addr1/P_emaildomain/DeviceInfo)的风险先验被埋在 ID 嵌入里,无法显式传播
2. "团伙"信号被均匀稀释到大量 transaction 节点的相邻边

异质图把实体提升为独立节点、赋予 5 维聚合特征(train-only 防泄漏),让团伙信号在 entity 节点处汇聚。同时 Stage 1 的 `gated_plus_hnm` 配置在 228 秒内被早停误杀,Stage 3a 增加 5 重收敛保证(epochs 40 / patience 8 / min_epochs 10 / 每 epoch history JSON / 训练曲线 PNG / 收敛断言),确保每个配置展示真正的最优。

## 文献支撑

1. Liu, Z., et al. "Heterogeneous Graph Neural Networks for Malicious Account Detection." CIKM 2018. (阿里风控团队工作)
2. Hamilton, W., Ying, Z., Leskovec, J. "Inductive Representation Learning on Large Graphs." NeurIPS 2017. (GraphSAGE)
3. Paranjape, A., Benson, A.R., Leskovec, J. "Motifs in Temporal Networks." WSDM 2017. (time-respecting edges)
4. Müller, R., Kornblith, S., Hinton, G. "When Does Label Smoothing Help?" NeurIPS 2019.
5. Pandit, S., et al. "NetProbe: A Fast and Scalable System for Fraud Detection in Online Auction Networks." WWW 2007. (PageRank for fraud rings)

## 原理详解

详见 docs/superpowers/specs/2026-05-15-ant-riskcontrol-stage3a-design.md (5 节,780 行) 完整设计:
- 节点 schema: 5 类 (transaction + 4 entity)
- 边 schema: 5 关系 / 9 edge_index (4 双向 entity 边 + 1 time-respecting txn-txn)
- HeteroGraphTower: HeteroConv 包 9 SAGEConv (aggr='mean') × 2 层
- 4 配置矩阵: baseline / asym_balanced / label_smoothing / HNM_root_cause
- 收敛保证: epochs 40 / patience 8 / min_epochs 10 / per-epoch history / curves PNG / audit warnings

## 实现细节

实施计划 docs/superpowers/plans/2026-05-15-ant-riskcontrol-stage3a.md 共 20 task,所有代码经 TDD 落地,新增 14 测试 (Stage 1+2 共 37 测试 → 全栈 51 测试 100% 通过)。关键文件:

- `src/data/entity_stats.py` — train-only 实体聚合 + 冷启动均值兜底
- `src/data/build.py::build_hetero_graph()` — HeteroData 构造
- `src/models/hetero_graph_tower.py` — HeteroConv + EntityProjector
- `src/models/fraud_model.py` — graph_backbone 'homo'|'hetero' 分支
- `src/dataset.py::make_hetero_loader()` — PyG 异质 NeighborLoader 包装
- `src/models/losses.py` — `label_smoothing_eps`, HNM 诊断版本
- `src/train.py` — `_convergence_audit`, `train_one_config_hetero`, `run_stage3a_matrix`
- `src/analysis/{plot_curves,centrality}.py` — 训完后处理

## 真实结果

<TABLE_FROM_STEP_2>

对照基准(Stage 2,直接复用):

| 对照 | val_pr_auc |
|---|---|
| Stage 2 deep_pruned (homo gated) | 0.4312 |
| Stage 2 deep_full (best Stage 2 deep) | 0.4370 |
| Stage 2 lgbm_pruned | 0.5303 |
| Stage 2 lgbm_full (best Stage 2 LGB) | 0.5556 |

Best config: **<BEST_NAME>**,Δ vs Stage 2 deep_pruned = **<DELTA_VS_S2>**.

## 诚实四情景结论

<SCENARIO_CHECKBOX>(从下面 4 个里勾选一个,基于实测)

- [ ] hetero best PR-AUC > 0.5556 (best LGB lgbm_full):深度模型反超传统模型 ✅
- [ ] hetero best PR-AUC ∈ (0.4370, 0.5556):异质图有效但仍未超 LGB
- [ ] hetero best PR-AUC ≈ 0.4370 (best Stage 2 deep deep_full):异质图在本数据集帮助有限
- [ ] hetero best PR-AUC < 0.4370:实现需排查或同构已够

## 团伙识别

`experiments/core_entities_<best_config>.json` 与 `experiments/core_entities_hetero_baseline.json` 给出 top-20 高 PageRank entity per type + degree 对照,作为简历 "异常团伙核心节点识别" 的可解释性证据。

## HNM 根因诊断

`experiments/hnm_diagnostics_hetero_HNM_root_cause.json` 记录每 epoch HNM 丢弃负样本的预测分布。读取这份日志可以判断 Stage 1 `gated_plus_hnm` 早停的根因——若 `mean_prob_dropped_neg` 长期接近 0(HNM 只丢"显然非欺诈"的样本),HNM 是合理的;若 `max_prob_dropped_neg` 接近 1(HNM 把高置信难例也丢了),HNM 设计需重新审视。

## 简历映射

- "行为序列与异质图建模" → SequenceTower (Stage 1) + HeteroGraphTower (Stage 3a) + 团伙识别 (Stage 3a 后处理)
- "极度不平衡样本处理" → HybridFocal (Stage 1) + 4 损失变体 ablation (Stage 3a)
- "性能优化" → ONNX/TensorRT (Stage 1, Stage 2 验证;hetero 部署留 Stage 3+)

## Stage 3a 显式不做(YAGNI 边界)

- ❌ 异质图 ONNX/TensorRT 部署 → Stage 3b 工具链修复后再做
- ❌ Edge attribute (交易金额作边权) → Stage 3+
- ❌ Heterogeneous Attention (HAN/HGT) → Stage 3+,先验证 SAGEConv 基线
```

To append on AutoDL safely:

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  cp docs/DESIGN_JOURNAL.md docs/DESIGN_JOURNAL.md.bak && \
  echo '(saved backup; now manually append v3 block via your editor)'"
# Use your editor (VSCode SSH or vim) to paste the v3 block at the end of docs/DESIGN_JOURNAL.md.
# Verify v1 + v2 are byte-for-byte preserved before continuing:
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  diff <(head -n \$(grep -n '^# v3' docs/DESIGN_JOURNAL.md | head -1 | cut -d: -f1 | awk '{print \$1-1}') docs/DESIGN_JOURNAL.md) docs/DESIGN_JOURNAL.md.bak && \
  echo 'v1+v2 preserved' || echo 'WARNING: v1+v2 changed during append'"
```

If diff is empty (output `v1+v2 preserved`), proceed. Otherwise STOP and revert from `.bak`.

- [ ] **Step 4: Append README Stage 3a section**

Append the following block to the END of `README.md` on AutoDL (do NOT overwrite). Replace `<TABLE>` and other placeholders with real values from Step 2.

```markdown


## Stage 3a Results: Heterogeneous Graph + Loss Deepening (2026-05-15)

### Experiment Matrix (4 configs + 2 Stage 2 baselines for direct comparison)

<TABLE>

| baseline (reused) | val_pr_auc |
|---|---|
| Stage 2 deep_pruned (homo + gated) | 0.4312 |
| Stage 2 deep_full (best Stage 2 deep) | 0.4370 |
| Stage 2 lgbm_pruned | 0.5303 |
| Stage 2 lgbm_full (best Stage 2 LGB) | 0.5556 |

### Convergence Audit Summary

For each config we record `best_epoch / total_epochs_run`. Configs with `converged=False` are explicitly flagged below — we do **not** silently treat them as "best results".

(See per-config row in the table above.)

### Training Curves

- ![hetero_baseline](experiments/curves_hetero_baseline.png)
- ![hetero_asym_balanced](experiments/curves_hetero_asym_balanced.png)
- ![hetero_label_smoothing](experiments/curves_hetero_label_smoothing.png)
- ![hetero_HNM_root_cause](experiments/curves_hetero_HNM_root_cause.png)

Each PNG plots train-loss / validation PR-AUC + ROC-AUC / lr schedule, with the red dashed vertical line marking the epoch with peak val_pr_auc.

### Fraud Ring Identification (post-hoc, on best config)

`experiments/core_entities_<best_config>.json` lists the top-20 highest-PageRank entities per type computed on the high-confidence-fraud subgraph. This is the deliverable that backs the resume bullet "异常团伙核心节点识别".

### Resume Bullet Mapping (Stage 3a delta)

| Resume bullet | Stage 1 | Stage 2 | Stage 3a |
|---|---|---|---|
| 行为序列与异质图建模 | SequenceTower (Transformer-GRU) | per-field categorical embeddings | **HeteroGraphTower + entity stat priors + post-hoc PageRank centrality** |
| 极度不平衡样本处理 | HybridFocal + HNM | full-V ablation | **4 loss variants ablation + HNM diagnostics** |
| 性能优化 | ONNX/TensorRT (homo) | dual-strategy ONNX | (deferred to Stage 3b) |

### Honest Negative Results / Caveats

(List each `converged=False` config with the warnings produced by `_convergence_audit`. If all configs converged, write "All 4 configs converged cleanly under the new budget".)

### Reproduction

```bash
# 1. Build hetero graph artifacts
python -m src.data.build           # honors configs/data.yaml::build_hetero_graph

# 2. Run all 4 configs sequentially (or one at a time as in plan tasks 15-18)
python -c "from src.train import run_stage3a_matrix; run_stage3a_matrix()"

# 3. Generate curves + run centrality
for cfg in hetero_baseline hetero_asym_balanced hetero_label_smoothing hetero_HNM_root_cause; do
    python -c "from src.analysis.plot_curves import plot_curves; \
        plot_curves(f'experiments/training_history_${cfg}.json', f'experiments/curves_${cfg}.png')"
done
python -c "from src.analysis.centrality import run_centrality_for_config; \
    run_centrality_for_config('artifacts/best_hetero_baseline.pt', 'hetero_baseline')"
```
```

- [ ] **Step 5: Run the FULL test suite — must be 51 passes**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && \
  pytest tests/ -v 2>&1 | tail -30"
```

Expected: `51 passed in <X> min` (Stage 1+2: 37, Stage 3a: 14 = 51). If any fail, STOP and fix before pushing.

- [ ] **Step 6: Commit docs**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  rm -f docs/DESIGN_JOURNAL.md.bak && \
  git add docs/DESIGN_JOURNAL.md README.md && \
  git commit -m 'docs: DESIGN_JOURNAL v3 + README Stage 3a results section'"
```

- [ ] **Step 7: Push to GitHub feature branch**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /etc/network_turbo && \
  git push -u origin feature/stage3a-hetero-graph 2>&1 | tail -20"
```

If push fails with auth, use the saved PAT in the inline-URL pattern:

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /etc/network_turbo && \
  CURRENT_REMOTE=\$(git remote get-url origin) && \
  git remote set-url origin https://LiuJH12138:\${GITHUB_PAT}@github.com/LiuJH12138/alibaba-risk--control.git && \
  git push -u origin feature/stage3a-hetero-graph && \
  git remote set-url origin \"\$CURRENT_REMOTE\""
# IMPORTANT: GITHUB_PAT must be exported in the SSH session;
# do NOT inline the token in commands or commit messages.
```

Expected: push succeeds, GitHub shows `feature/stage3a-hetero-graph` with the 20 commits from this plan.

- [ ] **Step 8: Verify the GitHub branch is intact**

```bash
ssh autodl "cd /root/autodl-tmp/alibaba-risk-control-internship && \
  source /etc/network_turbo && \
  git fetch origin && \
  git log origin/feature/stage3a-hetero-graph --oneline | head -25"
```

Expected: 20 Stage 3a commits visible at the head of the remote branch.

---

## Self-Review

### 1. Spec coverage

| Spec section | Plan coverage |
|---|---|
| §0 设计初衷 + 诚实约束 | DESIGN_JOURNAL v3 (Task 20), README Stage 3a (Task 20) |
| §1.1 数据/模型/损失/训练/分析层 Δ | Tasks 2-13 (one section per Δ block) |
| §1.2 不变项 | No task touches SequenceTower / FusionHead / Mixer; explicit retention of homogeneous GraphTower in Task 6 |
| §1.3 收敛保证 5 重 | Task 1 (yaml epochs/patience/min_epochs), Task 10 (audit), Task 11 (history JSON + min_epochs floor), Task 12 (curves PNG), Task 11 (memory isolation) |
| §2.1 节点 schema | Task 3 builds 5 node types; Task 14 verifies real shapes |
| §2.2 边 schema (9 edge_index) | Task 3 + Task 5 EDGE_SPEC; Task 14 verifies counts |
| §2.3 防泄漏数据流 | Task 2 train-only stats + cold start (test_entity_stats_train_only) |
| §2.6 5 数据测试 | Task 2 (2) + Task 3 (3) = 5 ✓ |
| §3 HeteroGraphTower + Mixer 协同 | Task 5 + Task 6 |
| §3.7 4 模型测试 | Task 5 (3) + Task 6 (1) = 4 ✓ |
| §4.1 4 配置矩阵 | Task 11 STAGE3A_CONFIGS + Tasks 15-18 |
| §4.2 收敛保证 5 项 | Task 1 + 10 + 11 + 12 |
| §4.4 团伙识别 | Task 13 + Task 19 |
| §4.7 3 训练测试 | Task 8 (1 label_smoothing) + Task 10 (2 audit) = 3 ✓ |
| §4.7 2 分析测试 | Task 12 (1 plot) + Task 13 (1 centrality) = 2 ✓ |
| §5.4 DoD hard gates | Task 20 Step 5 (51 tests) + Step 6 + 7 (commits + push) |

Test count: 5 (data) + 4 (model) + 3 (train) + 2 (analysis) = **14 ✓** matches spec §5.4.

### 2. Placeholder scan

Scanned for: TBD, TODO, FIXME, "implement later", "fill in", "appropriate error handling", "similar to Task N", "etc." in the actionable steps. The only intentional placeholders are user-supplied real-result values (`<from log>`, `<TABLE_FROM_STEP_2>`, `<BEST_NAME>`, `<DELTA_VS_S2>`, `<SCENARIO_CHECKBOX>`, `<TABLE>`) in Tasks 15-18 commit messages and Task 20 documentation — these are values that **cannot exist before the runs complete** and must come from real measurements (per the project's honesty principle).

### 3. Type / signature consistency

- `compute_entity_stats(train_df, entity_col, amt_col, dt_col, label_col)` — signature identical in Task 2 implementation, Task 2 test, and Task 3 use via `compute_all_entity_features` ✓
- `compute_all_entity_features(df, train_idx, val_idx, entity_cols, ...)` — signature identical in Task 2 impl, Task 2 test, Task 3 use ✓
- `build_hetero_graph(df, train_idx, val_idx, txn_cat_x, txn_num_x, entity_cols, ...)` — Task 3 impl and tests aligned; Task 4 calls with same kwargs ✓
- `EDGE_SPEC` 9 entries in Task 5 match the 9 edges built in Task 3 (paid_with/shipped_to/sent_to_email/on_device + reverses + next_by_uid) ✓
- `FraudModel(cat_cardinalities, n_num_total, model_cfg, fusion_mode, graph_backbone)` — Task 6 impl and test match; Task 11 hetero training calls with `graph_backbone="hetero"` ✓
- `model.forward_hetero(seq_cat, seq_num, mask, hetero_data, seed_local)` — Task 6 impl, Task 11 train + eval calls match ✓
- `make_hetero_loader(hetero_graph, seq_all, node_idx, batch_size, neighbor_sample, shuffle)` — Task 7 impl, Task 11 train use, Task 13 centrality use all align ✓
- `HybridFocalLoss(gamma_pos, gamma_neg, alpha, label_smoothing_eps, reduction)` — Task 8 impl + test, Task 11 use all match ✓
- `hard_negative_mining_with_diagnostics(per_sample_loss, targets, neg_pos_ratio, probs)` — Task 9 impl, Task 11 use match ✓
- `_record_epoch_metrics(epoch, lr, train_loss, epoch_seconds, eval_metrics)` — Task 10 impl + test, Task 11 use match ✓
- `_convergence_audit(history, config_name)` — Task 10 impl + test, Task 11 use match ✓
- `plot_curves(history_json_path, out_png)` — Task 12 impl + test, Tasks 15-18 use match ✓
- `identify_fraud_rings(hetero_graph, fraud_seed_idx, top_k, entity_types)` — Task 13 impl + test match ✓
- `run_centrality_for_config(checkpoint_path, config_name, v_strategy, prob_threshold, top_k, device)` — Task 13 impl, Task 19 use match ✓
- Config keys touched by Task 1 (`min_epochs`, `label_smoothing_eps`, `graph_backbone`, `hetero_d_graph`, `hetero_n_layers`, `entity_feat_dim`, `build_hetero_graph`) all consumed downstream ✓
- 4 config names in `STAGE3A_CONFIGS` (Task 11) match Tasks 15-18 individual run scripts ✓

No mismatches found.

---

**End of plan. 20 tasks. Estimated wall-clock: ~6-8 h coding (Tasks 1-13) + ~3-5 h training (Tasks 14-19) + ~30 min documentation (Task 20).**
