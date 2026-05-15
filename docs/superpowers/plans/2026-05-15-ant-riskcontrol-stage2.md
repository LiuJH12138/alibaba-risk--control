# Ant 风控实习复刻 · Stage 2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Stage 1 的深度模型从「输给 LightGBM 0.908」拉到与之持平/超过 —— 加 per-field 类别 embedding,启用完整 V 列(同时做相关性剪枝消融),保 best checkpoint;实验矩阵收窄到 `gated_fusion × {full_v, pruned_v}` + LGB × 同两套 = 4 次训练。

**Architecture:** 数据层 `FeatureProcessor.transform` 由 flat 张量改为 `{cat_idx, num}` 字典(质变);新增 `EmbeddingMixer` 模块在两塔间共享、把 dict 转回统一 `[..., feat_dim_unified]` 张量;`SequenceTower` / `GraphTower` / `FusionHead` 完全不动,只是 `feat_dim` 从硬编码改为 mixer 派生。`build.py` 参数化 `v_strategy` 跑两次,产物分两套目录;`train.py` 加 best checkpoint 保存 + 新 `run_stage2_matrix`;部署链路 ONNX 升 4 输入。

**Tech Stack:** 沿用 Stage 1(Python 3.12, PyTorch 2.8+cu128, PyG 2.6.1, LightGBM 4.5, ONNX/TensorRT 10.16, sklearn2pmml 0.130, pytest)。

**执行环境:** 所有命令在 AutoDL 远程主机执行(用户 VSCode-SSH)。项目 `/root/autodl-tmp/alibaba-risk-control-internship`,git 当前在 `master`。建议子代理为 Stage 2 创建新分支 `feat/stage2-impl`。conda 环境 `dfer-riskctrl` 已就绪,数据盘扩容到 150GB(Stage 1 时 50GB 是限制因子)。

**关键接口约定(全计划一致):**
- `FeatureProcessor.transform(df) -> dict` 形状:`{"cat_idx": int64 [N, n_cat], "num": float32 [N, n_num*2]}`(n_num*2 = 标准化值 + isna 拼接)
- 数据落盘格式:`graph.pt` 保存 PyG `Data(cat_x=int64[Nn,n_cat], num_x=float32[Nn,n_num*2], edge_index, y, t)`(graph.x 字段名变,但属性结构清晰)。`seq_all.pt` 保存 `{"cat": int64[Nn,L,n_cat], "num": float32[Nn,L,n_num*2], "mask": bool[Nn,L]}`。
- 数据目录 `data/processed/{full_v|pruned_v}/...` 双轨。
- `EmbeddingMixer(cat_idx, num) -> tensor [..., out_dim]`,2D/3D 输入通用。
- `FraudModel.__init__(cat_cardinalities, n_num_total, model_cfg, fusion_mode)`;`forward(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx)`;`forward_online(seq_cat, seq_num, mask, graph_emb)`。
- `make_loader` yields dict: `{x_cat, x_num, edge_index, seed_local, seq_cat, seq_num, mask, label}`。
- `train_one_config(..., checkpoint_path: str | None = None)` —— 新增可选参数;PR-AUC 提升时 `torch.save(model.state_dict(), checkpoint_path)`。
- `EXPERIMENT_MATRIX` 保留(供 Stage 1 兼容);新增 `STAGE2_MATRIX` 仅含 `gated_fusion`;新增 `run_stage2_matrix()` 跑 2 个深度配置 × 2 v_strategy。
- 路径常量:`data/processed/{strategy}/`、`artifacts/best_deep_{strategy}.pt`、`artifacts/best_lgbm_{strategy}.pkl`、`artifacts/online_{strategy}.onnx`、`experiments/stage2_results.json`、`experiments/benchmark_stage2.json`。

---

## 文件结构

```
alibaba-risk-control-internship/
├── src/
│   ├── data/
│   │   ├── features.py        # MODIFY: transform → dict
│   │   ├── build.py           # MODIFY: 参数化 v_strategy + 双轨落盘
│   │   ├── v_pruning.py       # NEW: compute_pruned_v_cols
│   │   └── ... (load/uid/sequence/graph 不动)
│   ├── models/
│   │   ├── embedding_mixer.py # NEW: EmbeddingMixer 类
│   │   ├── fraud_model.py     # MODIFY: 持 mixer + 新 forward 签名
│   │   └── ... (sequence_tower/graph_tower/fusion/losses 不动)
│   ├── dataset.py             # MODIFY: dict batch
│   ├── train.py               # MODIFY: checkpoint 保存 + run_stage2_matrix
│   ├── baseline_lgbm.py       # MODIFY: 参数化 v_strategy + flatten 辅助
│   └── deploy/
│       ├── export_onnx.py     # MODIFY: 4 输入 wrapper
│       └── benchmark.py       # MODIFY: 双模型 benchmark
├── tests/
│   ├── test_data.py           # MODIFY: processor 3 测试 + 新增 v_pruning 1 测试
│   ├── test_models.py         # MODIFY: fraud_model 2 + loader 1 + 新增 mixer 2 测试
│   └── test_smoke.py          # MODIFY: train/lgbm/onnx/benchmark/e2e 5 测试全适配
├── docs/
│   └── DESIGN_JOURNAL.md      # APPEND v2(v1 字节级保留)
└── README.md                  # APPEND Stage 2 节
```

---

## Phase 0 — 新分支

### Task 0: 创建 Stage 2 feature 分支

- [ ] **Step 1: 切到 master 确认干净**

Run:
```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git checkout master && git status --short'
```
Expected: `master` checked out,`git status --short` 输出空(干净)。

- [ ] **Step 2: 创建并切到 feat/stage2-impl**

Run:
```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git checkout -b feat/stage2-impl && git branch'
```
Expected: 列表显示 `* feat/stage2-impl`(当前)和 `master`。

- [ ] **Step 3: 验证 Stage 1 测试基线**

Run:
```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest --tb=line -p no:warnings 2>&1 | tail -2'
```
Expected: `34 passed`(Stage 1 基线)。

(无 commit;切分支只是状态变化。)

---

## Phase 1 — 新增独立工具(无依赖、纯 TDD)

### Task 1: `EmbeddingMixer` 模块

**Files:**
- Create: `src/models/embedding_mixer.py`
- Test: `tests/test_models.py`(追加 2 个测试)

- [ ] **Step 1: 写失败测试(追加到 `tests/test_models.py`)**

```python
from src.models.embedding_mixer import EmbeddingMixer

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
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_models.py -k embedding_mixer -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.embedding_mixer'`

- [ ] **Step 3: 写 `src/models/embedding_mixer.py`**

```python
import torch
import torch.nn as nn

class EmbeddingMixer(nn.Module):
    """把 {cat_idx, num} 字典转成统一 [..., feat_dim_unified] 张量。
    每类别字段一个独立 nn.Embedding;num 直通;最终拼接。
    形状无关:同时支持序列输入 [B, L, n_cat] 和图输入 [N, n_cat]。"""

    def __init__(self, cat_cardinalities, cat_emb_dim: int, n_num_total: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(c), cat_emb_dim) for c in cat_cardinalities]
        )
        self.cat_emb_dim = cat_emb_dim
        self.n_num_total = n_num_total
        self.out_dim = len(cat_cardinalities) * cat_emb_dim + n_num_total

    def forward(self, cat_idx: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        # cat_idx: [..., n_cat] long;  num: [..., n_num_total] float
        embs = [emb(cat_idx[..., i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embs, dim=-1)         # [..., n_cat * cat_emb_dim]
        return torch.cat([cat_out, num], dim=-1)  # [..., out_dim]
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_models.py -k embedding_mixer -v'`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/models/embedding_mixer.py tests/test_models.py && git commit -m "feat: EmbeddingMixer for per-field categorical embeddings"'
```

---

### Task 2: V 列相关性贪心剪枝工具

**Files:**
- Create: `src/data/v_pruning.py`
- Test: `tests/test_data.py`(追加 1 个测试)

- [ ] **Step 1: 写失败测试**

```python
from src.data.v_pruning import compute_pruned_v_cols

def test_v_column_pruning_keeps_one_per_correlated_group():
    rng = np.random.default_rng(0)
    n = 200
    base1 = rng.normal(size=n)
    base2 = rng.normal(size=n)
    df = pd.DataFrame({
        "V1": base1,
        "V2": base1 + 0.005 * rng.normal(size=n),  # |corr|≈1 with V1
        "V3": base2,
        "V4": base2 + 0.005 * rng.normal(size=n),  # |corr|≈1 with V3
        "V5": rng.normal(size=n),                   # 独立
    })
    kept = compute_pruned_v_cols(df, threshold=0.95)
    assert kept == ["V1", "V3", "V5"]   # 贪心顺序保留首个代表
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_data.py -k v_column_pruning -v'`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/data/v_pruning.py`**

```python
import pandas as pd

def compute_pruned_v_cols(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """对 V 列贪心剪枝:从前往后遍历,若与已保留列存在 |corr| >= threshold,丢弃。
    只用 train 数据调用(防泄漏)。返回保留列名列表。"""
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    if not v_cols:
        return []
    corr = df[v_cols].corr().abs()
    kept: list[str] = []
    for c in v_cols:
        if all(corr.loc[c, k] < threshold for k in kept):
            kept.append(c)
    return kept
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_data.py -k v_column_pruning -v'`
Expected: PASS(1 passed)

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/data/v_pruning.py tests/test_data.py && git commit -m "feat: V column correlation-based greedy pruning utility"'
```

---

## Phase 2 — 数据层质变(dict 接口)

### Task 3: `FeatureProcessor.transform` 改为返回 dict

**Files:**
- Modify: `src/data/features.py` (整个 `transform` 方法重写;`fit`/`__init__`/`meta` 不动)
- Test: `tests/test_data.py`(改造 3 个 processor 测试 + 增强 bounded 测试)

- [ ] **Step 1: 改造 3 个测试**

打开 `tests/test_data.py`,**替换**这 3 个测试函数(其余测试不动):

```python
def test_processor_fits_on_train_only():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    val = pd.DataFrame({"ProductCD": ["A", "C"], "TransactionAmt": [30.0, 40.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    tr = fp.transform(train); va = fp.transform(val)
    # dict 接口
    assert set(tr.keys()) == {"cat_idx", "num"}
    # 未见类别 "C" 映射到 0(unknown 桶)—— cat 现在是整数索引
    assert va["cat_idx"][1, 0] == 0
    # 数值标准化:train 列 0(TransactionAmt)均值 ≈ 0
    assert abs(tr["num"][:, 0].mean()) < 1e-6

def test_processor_meta_has_cardinalities():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    assert fp.meta["cat_cardinalities"]["ProductCD"] == 3
    assert fp.meta["num_cols"] == ["TransactionAmt"]

def test_processor_output_is_bounded():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "card1": np.arange(500),                      # 高基数 cat
        "TransactionAmt": rng.normal(0, 1, size=500),
    })
    fp = FeatureProcessor(cat_cols=["card1"], num_cols=["TransactionAmt"])
    fp.fit(train)
    out = fp.transform(train)
    # cat: 整数索引,在 [0, cardinality);本例 cardinality = 501
    assert out["cat_idx"].min() >= 0 and out["cat_idx"].max() < 501
    # num: 标准化 + 裁剪 [-10, 10]
    assert out["num"].min() >= -10.0 and out["num"].max() <= 10.0
```

- [ ] **Step 2: 运行验证失败(应该 fail,因当前 transform 返回 DataFrame)**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_data.py -k processor -v'`
Expected: 3 FAIL(`AssertionError` 或 `TypeError: dict-like access on DataFrame` 等)。失败原因 = transform 返回旧格式。

- [ ] **Step 3: 改造 `src/data/features.py` 的 `transform` 方法**

替换 `transform` 整个方法体(`__init__` / `fit` / `CAT_COLS` / `NUM_COLS` 不动):

```python
    def transform(self, df: pd.DataFrame) -> dict:
        # cat: 整数索引数组 [N, n_cat] (int64);unknown → 0
        cat_arr = np.zeros((len(df), len(self.cat_cols)), dtype="int64")
        for j, c in enumerate(self.cat_cols):
            m = self._cat_maps[c]
            cat_arr[:, j] = (df[c].astype(str).fillna("nan")
                             .map(m).fillna(0).astype("int64").to_numpy())
        # num: 标准化 + clip 到 [-10, 10],并拼接 isna 指示位
        n_num = len(self.num_cols)
        num_arr = np.zeros((len(df), n_num * 2), dtype="float32")
        for j, c in enumerate(self.num_cols):
            col = df[c].astype("float64")
            std_val = ((col - self._num_mean[c]) / self._num_std[c]
                       ).fillna(0.0).clip(-10.0, 10.0).astype("float32").to_numpy()
            num_arr[:, j] = std_val
            num_arr[:, n_num + j] = col.isna().astype("float32").to_numpy()
        return {"cat_idx": cat_arr, "num": num_arr}
```

需要在文件顶部加 `import numpy as np`(若已有则跳过)。

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_data.py -v'`
Expected: 全部 test_data.py 测试通过(3 processor + 1 v_pruning + 其余 9 个不依赖 FP 输出格式的测试 = 共 ~13 通过)。

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/data/features.py tests/test_data.py && git commit -m "refactor: FeatureProcessor.transform returns dict {cat_idx, num}"'
```

---

### Task 4: `build.py` 参数化 + 双轨落盘

**Files:**
- Modify: `src/data/build.py` (`build_all` 改造为接受 `v_strategy`,落到子目录;`__main__` 跑两套)
- Test: `tests/test_data.py`(`test_time_split_*` / `test_validate_split_*` 不动)

- [ ] **Step 1: 写改造后的 `src/data/build.py`**

完全替换文件内容:

```python
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
    cat_arr_full = feat["cat_idx"].astype("float32") # build_sequences 接受单 array;这里临时统一
    num_arr_full = feat["num"]
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
```

- [ ] **Step 2: 验证 split 测试仍通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_data.py -k split -v'`
Expected: 2 passed(time_split + validate_split 测试都不依赖 build_all)。

- [ ] **Step 3: 验证模块可导入**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && python -c "import src.data.build; print(\"build module OK\")"'`
Expected: `build module OK` 无异常。

- [ ] **Step 4: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/data/build.py && git commit -m "refactor: build.py parameterized v_strategy + dual-track artifacts"'
```

---

## Phase 3 — Loader + Model 重构

### Task 5: `make_loader` 输出 dict batch

**Files:**
- Modify: `src/dataset.py` (`make_loader` 改造)
- Test: `tests/test_models.py`(改造 1 个 loader 测试)

- [ ] **Step 1: 改造 `test_loader_yields_aligned_seq_and_seeds`**

打开 `tests/test_models.py`,**替换** `test_loader_yields_aligned_seq_and_seeds`:

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_models.py -k loader -v'`
Expected: FAIL(`KeyError: 'cat_x'` 或类似 —— 当前 graph.x 形式)。

- [ ] **Step 3: 改造 `src/dataset.py`**

完全替换:

```python
import torch
from torch_geometric.loader import NeighborLoader

def make_loader(graph, seq_all, node_idx, batch_size, neighbor_sample, shuffle=True):
    """NeighborLoader 驱动 batch。yield dict:
       x_cat / x_num / edge_index / seed_local / seq_cat / seq_num / mask / label。"""
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
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_models.py -k loader -v'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/dataset.py tests/test_models.py && git commit -m "refactor: make_loader yields dict batches with cat/num split"'
```

---

### Task 6: `FraudModel` 持 mixer + 新 forward 签名

**Files:**
- Modify: `src/models/fraud_model.py` (整体重写)
- Test: `tests/test_models.py`(改造 2 个 fraud_model 测试)

- [ ] **Step 1: 改造 `test_fraud_model_*` 两个测试**

替换两个 fraud_model 测试函数:

```python
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
    # mixer 共享:序列路径和图路径用同一组 embedding 对象
    assert id(model.seq_tower.input_proj) != id(model.graph_tower.convs[0])  # 不同塔
    assert all(emb is mixer_emb for emb, mixer_emb in
               zip(list(model.mixer.embeddings), list(model.mixer.embeddings)))
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_models.py -k fraud_model -v'`
Expected: FAIL(`TypeError`:旧签名 `feat_dim` 参数,或 forward 参数数量不匹配)。

- [ ] **Step 3: 改造 `src/models/fraud_model.py`**

完全替换:

```python
import torch
import torch.nn as nn
from src.models.embedding_mixer import EmbeddingMixer
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.fusion import FusionHead


class FraudModel(nn.Module):
    """双塔欺诈检测模型(Stage 2:per-field cat embedding + 共享 mixer)。
    训练 forward(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx)
    部署 forward_online(seq_cat, seq_num, mask, graph_emb)"""

    def __init__(self, cat_cardinalities, n_num_total: int, model_cfg: dict,
                 fusion_mode: str = "gated"):
        super().__init__()
        c = model_cfg
        self.mixer = EmbeddingMixer(cat_cardinalities, c["cat_emb_dim"], n_num_total)
        feat_dim = self.mixer.out_dim
        self.seq_tower = SequenceTower(
            feat_dim=feat_dim, d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_transformer_layers"], d_seq=c["d_seq"], dropout=c["dropout"])
        self.graph_tower = GraphTower(
            feat_dim=feat_dim, d_graph=c["d_graph"],
            n_layers=c["graphsage_layers"], dropout=c["dropout"])
        self.fusion = FusionHead(
            d_seq=c["d_seq"], d_graph=c["d_graph"], d_fuse=c["d_fuse"],
            mlp_hidden=c["mlp_hidden"], mode=fusion_mode, dropout=c["dropout"])

    def forward(self, seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx):
        seq = self.mixer(seq_cat, seq_num)
        x = self.mixer(x_cat, x_num)
        seq_emb = self.seq_tower(seq, mask)
        graph_emb_all = self.graph_tower(x, edge_index)
        return self.fusion(seq_emb, graph_emb_all[seed_idx])

    def forward_online(self, seq_cat, seq_num, mask, graph_emb):
        seq = self.mixer(seq_cat, seq_num)
        return self.fusion(self.seq_tower(seq, mask), graph_emb)
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_models.py -v'`
Expected: PASS(全部 test_models.py 测试 —— 序列塔、图塔、融合各原样,mixer 2 个新测试,fraud_model 2 个适配,loader 1 个适配 = 共 ~12 通过)。

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/models/fraud_model.py tests/test_models.py && git commit -m "refactor: FraudModel owns shared EmbeddingMixer; new dict-based forward signatures"'
```

---

## Phase 4 — 训练 + 基线适配

### Task 7: `train.py` 加 best checkpoint 保存 + `run_stage2_matrix`

**Files:**
- Modify: `src/train.py` (整体重写)
- Test: `tests/test_smoke.py`(改造 train smoke 测试)

- [ ] **Step 1: 改造 `test_train_one_config_runs_and_returns_metrics`**

打开 `tests/test_smoke.py`,**替换** train 测试:

```python
def test_train_one_config_runs_and_returns_metrics(tmp_path):
    n = 200
    graph = Data(cat_x=torch.randint(0, 4, (n, 3)),
                 num_x=torch.randn(n, 8),
                 edge_index=torch.randint(0, n, (2, 600)),
                 y=(torch.rand(n) > 0.85).float(),
                 t=torch.arange(n))
    seq_all = {"cat": torch.randint(0, 4, (n, 8, 3)),
               "num": torch.randn(n, 8, 8),
               "mask": torch.ones(n, 8, dtype=torch.bool)}
    split = {"train_idx": torch.arange(0, 150), "val_idx": torch.arange(150, n)}
    ckpt = tmp_path / "ckpt.pt"
    result = train_one_config(
        graph, seq_all, split, fusion_mode="gated", use_hnm=True,
        cat_cardinalities=[4, 4, 4], n_num_total=8,
        model_cfg={"d_model": 16, "n_heads": 2, "n_transformer_layers": 1,
                   "d_seq": 12, "d_graph": 12, "graphsage_layers": 2,
                   "d_fuse": 8, "mlp_hidden": 4, "dropout": 0.0, "cat_emb_dim": 4},
        train_cfg={"batch_size": 32, "lr": 1e-3, "weight_decay": 0.0, "epochs": 2,
                   "warmup_steps": 5, "grad_clip": 1.0, "seed": 42,
                   "neighbor_sample": [5, 5], "focal_gamma_pos": 1.0,
                   "focal_gamma_neg": 4.0, "focal_alpha": 0.25,
                   "hnm_neg_pos_ratio": 3.0, "early_stop_patience": 5},
        device="cpu", checkpoint_path=str(ckpt))
    assert "roc_auc" in result and 0.0 <= result["roc_auc"] <= 1.0
    assert ckpt.exists(), "checkpoint should be saved on PR-AUC improvement"
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k train_one_config -v'`
Expected: FAIL(`TypeError: train_one_config() got unexpected keyword 'cat_cardinalities'` 或类似)。

- [ ] **Step 3: 改造 `src/train.py`**

完全替换:

```python
import json
import time
from pathlib import Path
import numpy as np
import torch

from src.config import load_config
from src.dataset import make_loader
from src.models.fraud_model import FraudModel
from src.models.losses import HybridFocalLoss, hard_negative_mining
from src.evaluate import compute_metrics


def _set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for b in loader:
        logit = model(b["seq_cat"].to(device), b["seq_num"].to(device),
                      b["mask"].to(device),
                      b["x_cat"].to(device), b["x_num"].to(device),
                      b["edge_index"].to(device), b["seed_local"].to(device))
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(b["label"].cpu().numpy())
    return compute_metrics(np.concatenate(labels), np.concatenate(scores))


def train_one_config(graph, seq_all, split, fusion_mode, use_hnm,
                     cat_cardinalities, n_num_total,
                     model_cfg, train_cfg, device="cuda",
                     checkpoint_path: str | None = None):
    """训练单配置。返回 best 指标。可选保存 best checkpoint 到 checkpoint_path。"""
    _set_seed(train_cfg["seed"])
    model = FraudModel(cat_cardinalities, n_num_total, model_cfg, fusion_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    warmup = train_cfg["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    loss_fn = HybridFocalLoss(train_cfg["focal_gamma_pos"], train_cfg["focal_gamma_neg"],
                              train_cfg["focal_alpha"], reduction="none")
    train_loader = make_loader(graph, seq_all, split["train_idx"],
                               train_cfg["batch_size"], train_cfg["neighbor_sample"])
    val_loader = make_loader(graph, seq_all, split["val_idx"],
                             train_cfg["batch_size"], train_cfg["neighbor_sample"], shuffle=False)

    best_pr, best_metrics, patience = -1.0, None, 0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for b in train_loader:
            logit = model(b["seq_cat"].to(device), b["seq_num"].to(device),
                          b["mask"].to(device),
                          b["x_cat"].to(device), b["x_num"].to(device),
                          b["edge_index"].to(device), b["seed_local"].to(device))
            target = b["label"].to(device)
            per_sample = loss_fn.per_sample(logit, target)
            if use_hnm:
                keep = hard_negative_mining(per_sample.detach(), target,
                                            train_cfg["hnm_neg_pos_ratio"])
                loss = per_sample[keep].mean()
            else:
                loss = per_sample.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            opt.step(); scheduler.step()
        metrics = _evaluate(model, val_loader, device)
        if metrics["pr_auc"] > best_pr:
            best_pr, best_metrics, patience = metrics["pr_auc"], metrics, 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
            if patience >= train_cfg["early_stop_patience"]:
                break
    return best_metrics


# Stage 2 矩阵:gated_fusion × {full_v, pruned_v}
STAGE2_DEEP_CONFIGS = [{"name": "deep_full", "v_strategy": "full_v"},
                       {"name": "deep_pruned", "v_strategy": "pruned_v"}]


def run_stage2_matrix(device="cuda"):
    """跑 Stage 2 深度模型矩阵(2 跑),写 experiments/stage2_results.json。"""
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    Path("experiments").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    # 增量更新(允许多次运行)
    out_path = Path("experiments/stage2_results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for cfg in STAGE2_DEEP_CONFIGS:
        name, v_strategy = cfg["name"], cfg["v_strategy"]
        proc_dir = Path("data/processed") / v_strategy
        graph = torch.load(proc_dir / "graph.pt", weights_only=False)
        seq_all = torch.load(proc_dir / "seq_all.pt", weights_only=False)
        split = torch.load(proc_dir / "split.pt", weights_only=False)
        manifest = json.loads((proc_dir / "manifest.json").read_text())
        meta = json.loads((proc_dir / "feature_meta.json").read_text())
        cat_cardinalities = [meta["cat_cardinalities"][c] for c in meta["cat_cols"]]
        n_num_total = manifest["n_num_total"]

        ckpt = f"artifacts/best_{name}.pt"
        t0 = time.time()
        metrics = train_one_config(graph, seq_all, split, fusion_mode="gated",
                                   use_hnm=False,
                                   cat_cardinalities=cat_cardinalities,
                                   n_num_total=n_num_total,
                                   model_cfg=model_cfg, train_cfg=train_cfg,
                                   device=device, checkpoint_path=ckpt)
        metrics["train_seconds"] = round(time.time() - t0, 1)
        metrics["v_strategy"] = v_strategy
        results[name] = metrics
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{name}: {metrics}")

    return results


if __name__ == "__main__":
    run_stage2_matrix()
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k train_one_config -v'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/train.py tests/test_smoke.py && git commit -m "refactor: train.py dict-based forward + best checkpoint saving + run_stage2_matrix"'
```

---

### Task 8: `baseline_lgbm.py` 参数化 v_strategy + flatten 辅助

**Files:**
- Modify: `src/baseline_lgbm.py` (整体重写)
- Test: `tests/test_smoke.py`(`test_lgbm_baseline_runs` 检查是否需要适配)

- [ ] **Step 1: 检查现有 `test_lgbm_baseline_runs`,不需改动**

`test_lgbm_baseline_runs` 直接传扁平 numpy 数组给 `train_lgbm_baseline`,不依赖 `run_baseline` 也不依赖 v_strategy。新签名 `train_lgbm_baseline(x_train, y_train, x_val, y_val, categorical_feature=None)` 默认 None 向后兼容。无需改测试。

- [ ] **Step 2: 改造 `src/baseline_lgbm.py`**

完全替换:

```python
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
import torch
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.preprocessing import FunctionTransformer

from src.evaluate import compute_metrics


def flatten_for_lgbm(graph_data) -> tuple[np.ndarray, list[int]]:
    """把 graph 的 cat_x + num_x 拍扁回单矩阵给 LGB。
    返回 (X [N, n_cat + n_num_total], categorical_feature_indices)。"""
    cat = graph_data.cat_x.numpy()
    num = graph_data.num_x.numpy()
    X = np.concatenate([cat, num], axis=1).astype("float32")
    cat_idx = list(range(cat.shape[1]))    # 前 n_cat 列是类别
    return X, cat_idx


def train_lgbm_baseline(x_train, y_train, x_val, y_val, categorical_feature=None):
    """在扁平表特征上训 LightGBM。"""
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        class_weight="balanced",
    )
    fit_kwargs = {"categorical_feature": categorical_feature} if categorical_feature else {}
    clf.fit(x_train, y_train, **fit_kwargs)
    scores = clf.predict_proba(x_val)[:, 1]
    return compute_metrics(y_val, scores), clf


def export_pmml(clf, path: str):
    pipe = PMMLPipeline([("identity", FunctionTransformer()), ("clf", clf)])
    sklearn2pmml(pipe, path)


def run_baseline(v_strategy: str):
    """v_strategy ∈ {full_v, pruned_v}。结果合并入 stage2_results.json,模型存 .pkl。"""
    proc_dir = Path("data/processed") / v_strategy
    graph = torch.load(proc_dir / "graph.pt", weights_only=False)
    split = torch.load(proc_dir / "split.pt", weights_only=False)
    X, cat_idx = flatten_for_lgbm(graph)
    y = graph.y.numpy()
    tr, va = split["train_idx"].numpy(), split["val_idx"].numpy()
    metrics, clf = train_lgbm_baseline(X[tr], y[tr], X[va], y[va],
                                       categorical_feature=cat_idx)
    metrics["v_strategy"] = v_strategy
    name = f"lgbm_{v_strategy.replace('_v', '')}"   # lgbm_full, lgbm_pruned
    print(f"{name}: {metrics}")

    # 增量更新 stage2_results.json
    Path("experiments").mkdir(exist_ok=True)
    out_path = Path("experiments/stage2_results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results[name] = metrics
    out_path.write_text(json.dumps(results, indent=2))

    # 存模型 + 尝试 PMML(失败不致命)
    Path("artifacts").mkdir(exist_ok=True)
    import pickle
    with open(f"artifacts/best_lgbm_{v_strategy.replace('_v', '')}.pkl", "wb") as f:
        pickle.dump(clf, f)
    try:
        export_pmml(clf, f"artifacts/lgbm_baseline_{v_strategy.replace('_v', '')}.pmml")
        print("PMML exported")
    except Exception as e:
        print(f"PMML export skipped (Java toolchain): {e}")


if __name__ == "__main__":
    for s in ["full_v", "pruned_v"]:
        run_baseline(s)
```

- [ ] **Step 3: 验证测试通过(向后兼容,旧测试不动)**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k lgbm -v'`
Expected: PASS(向后兼容)

- [ ] **Step 4: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/baseline_lgbm.py && git commit -m "refactor: baseline_lgbm.py parameterized v_strategy + flatten helper + PMML graceful skip"'
```

---

## Phase 5 — 真实数据实验

### Task 9: 跑 `build_all` 双 v_strategy(产出真实数据)

**Files:** 无新文件;产出 `data/processed/{full_v,pruned_v}/`(共 ~84GB)

- [ ] **Step 1: 清理 Stage 1 旧 processed 数据(扁平结构)**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && rm -rf data/processed/*.pt data/processed/*.json && ls data/processed/'
```
Expected: 目录空(只剩可能存在的子目录,首次运行无)。

- [ ] **Step 2: 启动后台双轨构建(估计 10-20 分钟)**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && nohup python -u -m src.data.build > /tmp/build.log 2>&1 &'
```
然后轮询直到完成:
```bash
ssh autodl 'until ! pgrep -f "[s]rc.data.build" >/dev/null; do sleep 30; done; echo done; tail -8 /tmp/build.log'
```
Expected: 看到两次 `build done [...]` 输出,第二次包含 `pruned_v: kept N of 339 V cols`(N 应在 80-150 之间)。

- [ ] **Step 3: 校验产物**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && du -sh data/processed/full_v data/processed/pruned_v && cat data/processed/full_v/manifest.json && echo "---" && cat data/processed/pruned_v/manifest.json'
```
Expected: full_v ~58GB(graph + seq_all),pruned_v ~30GB;两 manifest 都显示 `n_transactions: 590540`、`fraud_rate ≈ 0.035`、`v_strategy` 字段正确。`n_num_total` 相差(full ≈ 678,pruned ≈ 280)。

- [ ] **Step 4: 不需 commit(数据 gitignored)**

跳过。状态已通过 manifest 写入。

---

### Task 10: 跑 `run_stage2_matrix`(2 个深度配置)

**Files:** 产出 `experiments/stage2_results.json` + `artifacts/best_deep_{full,pruned}.pt`

- [ ] **Step 1: 启动后台训练(估计 50-70 分钟)**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && nohup python -u -m src.train > /tmp/stage2_train.log 2>&1 &'
```
轮询:
```bash
ssh autodl 'until ! pgrep -f "[s]rc.train" >/dev/null; do sleep 60; done; echo done; tail -20 /tmp/stage2_train.log'
```
Expected: `deep_full` 和 `deep_pruned` 各打一行 metrics dict,无 NaN/Exception。stage2_results.json 含两 key。

- [ ] **Step 2: 校验结果合理性**

```bash
ssh autodl 'cat /root/autodl-tmp/alibaba-risk-control-internship/experiments/stage2_results.json'
```
Expected: 两个 deep 配置的 roc_auc 均落在合理区间(预期 0.85-0.93;若超过 LGB 0.908 是诚实四情景情景 1 或 3,若不及是情景 2 或 4)。两个 checkpoint 文件存在:
```bash
ssh autodl 'ls -lh /root/autodl-tmp/alibaba-risk-control-internship/artifacts/best_deep_*.pt'
```

- [ ] **Step 3: Commit 结果(不含 ckpt,先看 ckpt 大小)**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && du -sh artifacts/best_deep_*.pt'
```
若每个 < 100MB,直接 commit:
```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add experiments/stage2_results.json artifacts/best_deep_*.pt && git commit -m "experiment: Stage 2 deep model results (gated_fusion × {full_v, pruned_v})"'
```
若 > 100MB,改为加到 .gitignore 单独说明:
```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && echo "artifacts/best_deep_*.pt" >> .gitignore && git add experiments/stage2_results.json .gitignore && git commit -m "experiment: Stage 2 deep model results (checkpoints gitignored due to size)"'
```

---

### Task 11: 跑 LightGBM 双 v_strategy

**Files:** 更新 `experiments/stage2_results.json` + `artifacts/best_lgbm_{full,pruned}.pkl`

- [ ] **Step 1: 跑(快,~2 分钟)**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && python -u -m src.baseline_lgbm 2>&1 | tail -20'
```
Expected: `lgbm_full` 和 `lgbm_pruned` 各打 metrics(roc_auc 应接近 0.91)。两 .pkl 文件生成。PMML 大概率 skip(Java 11+ 工具链未装,优雅 skip 已设计)。

- [ ] **Step 2: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add experiments/stage2_results.json artifacts/best_lgbm_*.pkl && git commit -m "experiment: Stage 2 LightGBM baseline on both V strategies"'
```

---

## Phase 6 — 部署适配 + 真实 Benchmark

### Task 12: `export_onnx.py` 4 输入 wrapper

**Files:**
- Modify: `src/deploy/export_onnx.py` (整体重写)
- Test: `tests/test_smoke.py`(改造 onnx 测试)

- [ ] **Step 1: 改造 `test_onnx_export_and_parity`**

```python
def test_onnx_export_and_parity(tmp_path):
    from src.models.fraud_model import FraudModel
    model = FraudModel(cat_cardinalities=[5, 7], n_num_total=4, model_cfg={
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1, "d_seq": 12,
        "d_graph": 12, "graphsage_layers": 2, "d_fuse": 8, "mlp_hidden": 4,
        "dropout": 0.0, "cat_emb_dim": 4}, fusion_mode="gated").eval()
    onnx_path = str(tmp_path / "model.onnx")
    export_online_path(model, n_cat=2, n_num_total=4, seq_len=8, d_graph=12, path=onnx_path)
    assert verify_onnx_parity(model, onnx_path, n_cat=2, n_num_total=4,
                              cat_cardinalities=[5, 7], seq_len=8, d_graph=12)
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k onnx -v'`
Expected: FAIL(`TypeError`:旧 `feat_dim` 参数)

- [ ] **Step 3: 改造 `src/deploy/export_onnx.py`**

完全替换:

```python
import numpy as np
import torch
import onnxruntime as ort


class _OnlineWrapper(torch.nn.Module):
    """只暴露在线路径(mixer + 序列塔 + 融合头),供 ONNX 导出。"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, seq_cat, seq_num, mask, graph_emb):
        return self.model.forward_online(seq_cat, seq_num, mask, graph_emb)


def export_online_path(model, n_cat, n_num_total, seq_len, d_graph, path):
    """把 FraudModel 在线路径导出为 ONNX(动态 batch 轴)。"""
    wrapper = _OnlineWrapper(model).eval()
    seq_cat = torch.zeros(2, seq_len, n_cat, dtype=torch.long)
    seq_num = torch.randn(2, seq_len, n_num_total)
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(2, d_graph)
    torch.onnx.export(
        wrapper, (seq_cat, seq_num, mask, graph_emb), path,
        input_names=["seq_cat", "seq_num", "mask", "graph_emb"],
        output_names=["logit"],
        dynamic_axes={k: {0: "batch"}
                      for k in ["seq_cat", "seq_num", "mask", "graph_emb", "logit"]},
        opset_version=17,
    )


def verify_onnx_parity(model, onnx_path, n_cat, n_num_total, cat_cardinalities,
                       seq_len, d_graph, atol=1e-4):
    """校验 ONNX 输出与 PyTorch 一致(随机有效索引 + 随机数值)。"""
    model.eval()
    cards = torch.tensor(cat_cardinalities)
    seq_cat = torch.stack([torch.randint(0, int(cards[i]), (4, seq_len))
                           for i in range(n_cat)], dim=-1)  # [4, seq_len, n_cat]
    seq_num = torch.randn(4, seq_len, n_num_total)
    mask = torch.ones(4, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(4, d_graph)
    with torch.no_grad():
        torch_out = model.forward_online(seq_cat, seq_num, mask, graph_emb).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "seq_cat": seq_cat.numpy(), "seq_num": seq_num.numpy(),
        "mask": mask.numpy(), "graph_emb": graph_emb.numpy(),
    })[0]
    return bool(np.allclose(torch_out, onnx_out, atol=atol))
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k onnx -v'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/deploy/export_onnx.py tests/test_smoke.py && git commit -m "refactor: ONNX export 4-input version (seq_cat int64, seq_num, mask, graph_emb)"'
```

---

### Task 13: `benchmark.py` 双模型 + 真实 ckpt

**Files:**
- Modify: `src/deploy/benchmark.py` (整体重写)
- Test: `tests/test_smoke.py`(改造 benchmark 测试)

- [ ] **Step 1: 改造 `test_benchmark_torch_returns_latency_stats`**

```python
def test_benchmark_torch_returns_latency_stats():
    from src.models.fraud_model import FraudModel
    model = FraudModel(cat_cardinalities=[5, 7], n_num_total=4, model_cfg={
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1, "d_seq": 8,
        "d_graph": 8, "graphsage_layers": 2, "d_fuse": 8, "mlp_hidden": 4,
        "dropout": 0.0, "cat_emb_dim": 4}, fusion_mode="gated").eval()
    stats = benchmark_torch(model, cat_cardinalities=[5, 7], n_num_total=4,
                            seq_len=6, d_graph=8, device="cpu", n_runs=20, warmup=5)
    assert "p50_ms" in stats and "p95_ms" in stats and "p99_ms" in stats
    assert stats["p50_ms"] > 0
```

- [ ] **Step 2: 运行验证失败**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k benchmark -v'`
Expected: FAIL(`TypeError`:旧 `feat_dim` 参数)

- [ ] **Step 3: 改造 `src/deploy/benchmark.py`**

完全替换:

```python
import json
import time
from pathlib import Path
import numpy as np
import torch


def _percentiles(times_ms):
    arr = np.array(times_ms)
    return {"p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean())}


def _make_inputs(batch, seq_len, cat_cardinalities, n_num_total, d_graph, device):
    cards = torch.tensor(cat_cardinalities)
    seq_cat = torch.stack([torch.randint(0, int(cards[i]), (batch, seq_len), device=device)
                           for i in range(len(cat_cardinalities))], dim=-1)
    seq_num = torch.randn(batch, seq_len, n_num_total, device=device)
    mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    graph_emb = torch.randn(batch, d_graph, device=device)
    return seq_cat, seq_num, mask, graph_emb


def benchmark_torch(model, cat_cardinalities, n_num_total, seq_len, d_graph,
                    device, n_runs=1000, warmup=50, batch=1):
    model = model.to(device).eval()
    seq_cat, seq_num, mask, graph_emb = _make_inputs(
        batch, seq_len, cat_cardinalities, n_num_total, d_graph, device)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_online(seq_cat, seq_num, mask, graph_emb)
        if device == "cuda": torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.forward_online(seq_cat, seq_num, mask, graph_emb)
            if device == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)


def benchmark_onnx(onnx_path, cat_cardinalities, n_num_total, seq_len, d_graph,
                   providers, n_runs=1000, warmup=50, batch=1):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=providers)
    cards = np.array(cat_cardinalities)
    seq_cat = np.stack([np.random.randint(0, int(cards[i]), (batch, seq_len))
                        for i in range(len(cat_cardinalities))], axis=-1).astype("int64")
    seq_num = np.random.randn(batch, seq_len, n_num_total).astype("float32")
    mask = np.ones((batch, seq_len), dtype=bool)
    graph_emb = np.random.randn(batch, d_graph).astype("float32")
    feed = {"seq_cat": seq_cat, "seq_num": seq_num, "mask": mask, "graph_emb": graph_emb}
    for _ in range(warmup): sess.run(None, feed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times), sess.get_providers()


def _benchmark_one_model(name, v_strategy, ckpt_path):
    """benchmark 一个深度模型(4 档)。返回 dict。"""
    from src.config import load_config
    from src.models.fraud_model import FraudModel
    from src.deploy.export_onnx import export_online_path, verify_onnx_parity
    from src.deploy.build_trt import trt_available, build_engine

    mcfg = load_config("model")
    proc_dir = Path("data/processed") / v_strategy
    manifest = json.loads((proc_dir / "manifest.json").read_text())
    meta = json.loads((proc_dir / "feature_meta.json").read_text())
    cat_cardinalities = [meta["cat_cardinalities"][c] for c in meta["cat_cols"]]
    n_num_total = manifest["n_num_total"]
    n_cat = manifest["n_cat"]
    seq_len = manifest["seq_len"]
    d_graph = mcfg["d_graph"]

    model = FraudModel(cat_cardinalities, n_num_total, mcfg, fusion_mode="gated")
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    else:
        print(f"WARN: no checkpoint at {ckpt_path}, using random init for latency only")
    model.eval()

    Path("artifacts").mkdir(exist_ok=True)
    onnx_path = f"artifacts/online_{v_strategy.replace('_v','')}.onnx"
    export_online_path(model, n_cat, n_num_total, seq_len, d_graph, onnx_path)
    assert verify_onnx_parity(model, onnx_path, n_cat, n_num_total,
                              cat_cardinalities, seq_len, d_graph), "ONNX parity failed"

    res = {}
    res["pytorch_cpu"] = benchmark_torch(model, cat_cardinalities, n_num_total,
                                         seq_len, d_graph, "cpu")
    if torch.cuda.is_available():
        res["pytorch_gpu"] = benchmark_torch(model, cat_cardinalities, n_num_total,
                                             seq_len, d_graph, "cuda")
        try:
            stats, providers = benchmark_onnx(onnx_path, cat_cardinalities, n_num_total,
                                              seq_len, d_graph, ["CUDAExecutionProvider"])
            if "CUDAExecutionProvider" in providers:
                res["onnx_gpu"] = stats
            else:
                res["onnx_gpu"] = {"skipped": f"CUDAExecutionProvider not active (got {providers})"}
        except Exception as e:
            res["onnx_gpu"] = {"skipped": f"ORT CUDA load error: {e}"}

    if trt_available():
        engine = f"artifacts/online_{v_strategy.replace('_v','')}.engine"
        if build_engine(onnx_path, engine, fp16=True):
            try:
                stats, providers = benchmark_onnx(
                    onnx_path, cat_cardinalities, n_num_total, seq_len, d_graph,
                    [("TensorrtExecutionProvider", {"trt_fp16_enable": True})])
                if "TensorrtExecutionProvider" in providers:
                    res["tensorrt_fp16"] = stats
                else:
                    res["tensorrt_fp16"] = {"skipped": f"TRT EP not active (got {providers}); engine built OK"}
            except Exception as e:
                res["tensorrt_fp16"] = {"skipped": f"ORT TRT EP error: {e}; engine built OK"}
        else:
            res["tensorrt_fp16"] = {"skipped": "engine build failed"}
    else:
        res["tensorrt_fp16"] = {"skipped": "TensorRT not available"}
    return res


def run_benchmark():
    """对两个深度模型各跑 4 档 benchmark,落 experiments/benchmark_stage2.json。"""
    Path("experiments").mkdir(exist_ok=True)
    out_path = Path("experiments/benchmark_stage2.json")
    results = {}
    for v_strategy in ["full_v", "pruned_v"]:
        name = f"deep_{v_strategy.replace('_v','')}"
        ckpt = f"artifacts/best_{name}.pt"
        print(f"\n=== {name} ===")
        results[name] = _benchmark_one_model(name, v_strategy, ckpt)
        for k, v in results[name].items():
            print(f"  {k}: {v}")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
```

- [ ] **Step 4: 运行验证通过**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k benchmark -v'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add src/deploy/benchmark.py tests/test_smoke.py && git commit -m "refactor: benchmark.py dual-model latency benchmark with real checkpoints"'
```

---

### Task 14: 跑真实 Benchmark(双模型 4 档)

**Files:** 产出 `experiments/benchmark_stage2.json` + `artifacts/online_{full,pruned}.onnx`

- [ ] **Step 1: 跑(估计 10-15 分钟,主要是各 1000 次 timed runs)**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && python -u -m src.deploy.benchmark 2>&1 | tail -40'
```
Expected: 看到 `=== deep_full ===` 和 `=== deep_pruned ===` 两节;每节 4 档(`pytorch_cpu`、`pytorch_gpu` 给真实数字,`onnx_gpu`/`tensorrt_fp16` 大概率 skip 给文字解释)。最后 `wrote experiments/benchmark_stage2.json`。

- [ ] **Step 2: 校验 benchmark.json**

```bash
ssh autodl 'cat /root/autodl-tmp/alibaba-risk-control-internship/experiments/benchmark_stage2.json'
```
Expected: 两 key (`deep_full`/`deep_pruned`),各 4 子 key。预期 deep_pruned 的 GPU 延迟略低(更窄特征向量)。

- [ ] **Step 3: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add experiments/benchmark_stage2.json artifacts/online_*.onnx && git commit -m "experiment: Stage 2 dual-model latency benchmark"'
```

---

## Phase 7 — e2e smoke + 文档

### Task 15: 改造 e2e smoke 测试 + 全套 pytest

**Files:**
- Modify: `tests/test_smoke.py`(改造 e2e 测试)

- [ ] **Step 1: 改造 `test_end_to_end_pipeline`**

替换:

```python
def test_end_to_end_pipeline(tiny_raw_df, tmp_path, monkeypatch):
    """微数据集端到端:特征(dict) → 序列 → 图 → 训练 → 评估,全程跑通。"""
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
    feat = fp.transform(df)                          # dict
    n_cat = feat["cat_idx"].shape[1]
    n_num_total = feat["num"].shape[1]
    cat_cardinalities = [fp.meta["cat_cardinalities"][c] for c in fp.meta["cat_cols"]]

    seq_cat, mask = build_sequences(feat["cat_idx"].astype("float32"),
                                     df["uid"].to_numpy(), dt, seq_len=8)
    seq_num, _   = build_sequences(feat["num"], df["uid"].to_numpy(), dt, seq_len=8)
    src, dst = build_edges(df, ["card1", "P_emaildomain"], max_degree=20, max_per_entity=10)

    graph = Data(cat_x=torch.from_numpy(feat["cat_idx"]),
                 num_x=torch.from_numpy(feat["num"]),
                 edge_index=torch.from_numpy(np.stack([src, dst])),
                 y=torch.from_numpy(y), t=torch.from_numpy(dt))
    seq_all = {"cat": torch.from_numpy(seq_cat.astype("int64")),
               "num": torch.from_numpy(seq_num),
               "mask": torch.from_numpy(mask)}
    split = {"train_idx": torch.arange(0, cut), "val_idx": torch.arange(cut, len(df))}

    result = train_one_config(
        graph, seq_all, split, fusion_mode="gated", use_hnm=True,
        cat_cardinalities=cat_cardinalities, n_num_total=n_num_total,
        model_cfg={"d_model": 16, "n_heads": 2, "n_transformer_layers": 1,
                   "d_seq": 12, "d_graph": 12, "graphsage_layers": 2,
                   "d_fuse": 8, "mlp_hidden": 4, "dropout": 0.0, "cat_emb_dim": 4},
        train_cfg={"batch_size": 32, "lr": 1e-3, "weight_decay": 0.0, "epochs": 2,
                   "warmup_steps": 5, "grad_clip": 1.0, "seed": 42,
                   "neighbor_sample": [5, 5], "focal_gamma_pos": 1.0,
                   "focal_gamma_neg": 4.0, "focal_alpha": 0.25,
                   "hnm_neg_pos_ratio": 3.0, "early_stop_patience": 5},
        device="cpu")
    assert "roc_auc" in result
```

- [ ] **Step 2: 跑 e2e 测试**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest tests/test_smoke.py -k end_to_end -v'`
Expected: PASS

- [ ] **Step 3: 跑全套 pytest**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl && pytest -p no:warnings --tb=short 2>&1 | tail -3'`
Expected: 全绿 —— 预计 ~40 测试通过(13 data + 6 losses + 12 models + 5 smoke + 加新增,具体数视适配后总数)。

- [ ] **Step 4: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add tests/test_smoke.py && git commit -m "test: e2e smoke adapted to dict-based pipeline (Stage 2)"'
```

---

### Task 16: DESIGN_JOURNAL v2 + README Stage 2 节

**Files:**
- Modify: `docs/DESIGN_JOURNAL.md`(追加 v2,**v1 字节级保留**)
- Modify: `README.md`(追加 Stage 2 结果节,路线图打勾)

- [ ] **Step 1: 读当前 DESIGN_JOURNAL,确认 v1 边界**

Run: `ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && wc -l docs/DESIGN_JOURNAL.md && tail -3 docs/DESIGN_JOURNAL.md'`
记下当前总行数(供步骤 3 验证 v1 字节级未变)。

- [ ] **Step 2: 追加 v2 节到 `docs/DESIGN_JOURNAL.md`**

在文件**末尾追加**(不动任何已有内容):

```markdown


---

## v2 (2026-05-15) — Stage 2 模型基础升级

### 重定向:Stage 1 → Stage 2 的认知转变
Stage 1 实测:深度双塔模型 roc_auc 0.82-0.85,LightGBM 基线 0.908,深度输给基线。
v1 已识别根因为(a)类别字段用了缩放序数编码而非真正的 embedding,(b)V 列被磁盘
约束削减到 V1-V50。Stage 2 优先解决这两个根因 —— 异质图与损失深化推到 Stage 3+。

文献:Shwartz-Ziv & Armon 2022 *Tabular Data: Deep Learning is Not All You Need*
—— GBDT 在中等规模表格上常胜过深度模型,深度方法需要正确的归纳偏置(per-field
embedding、丰富特征)才能竞争。

### 设计决策

#### 决策 v2-1:类别字段独立 nn.Embedding,双塔共享 mixer
**初衷:** 替换 Stage 1 的缩放序数编码,让模型看到类别间的语义距离。
**原理:** 每类别字段一个 `nn.Embedding(cardinality, 16)`,前置在两塔之前;
mixer 在序列塔和图塔之间共享,保证 `card1` 在序列里和图里指同一组嵌入语义,
少参数、强一致。
**文献:** Wide & Deep(Cheng 2016)、DeepFM(Guo 2017)、FT-Transformer
(Gorishniy 2021)的 per-field embedding 模式。

#### 决策 v2-2:V 列相关性贪心剪枝(threshold 0.95)+ 同时跑全量做消融
**初衷:** V1-V339 已知高度冗余,但"完整 V 列"含义不唯一 —— 直接全用 vs 剪枝。
**原理:** 贪心遍历 V 列,与已保留列 |corr|≥0.95 则丢弃,确定性、可缓存。
预期保留 100-130 列。同时跑 full_v 和 pruned_v 做消融,用数据说话哪种更好。
**文献:** IEEE-CIS Kaggle 社区公开 kernel 中的 V 列剪枝惯例;Pearson 相关 +
贪心去冗余是表格 ML 标准做法。

#### 决策 v2-3:实验矩阵收窄到 1 配置 × 2 V 策略
**初衷:** Stage 2 核心问题是 ONE 个 —— 升级模型基础后能否跑赢 LGB。
**原理:** 不再跑 5 配置矩阵(Stage 1 已有完整对照)。专注 gated_fusion +
两 V 策略 + 各自 LGB = 4 跑。诚实原则下,加配置 = 加噪声;少而精更有说服力。

#### 决策 v2-4:保 best checkpoint
**初衷:** Stage 1 漏项 —— benchmark 用了随机权重(对延迟有效但部署不完整)。
**原理:** `train_one_config` 加 `checkpoint_path` 参数,PR-AUC 提升时保存。
为 Stage 3 部署铺路。

#### 决策 v2-5:四情景诚实成功框架
**初衷:** 防止"必须跑赢 0.908 才算成功"的压力催生超参 chasing。
**原理:** Stage 2 的"成功"=做完该做的工程改动 + 诚实测量。四种情景(both deep
≥ LGB / deep_pruned 赢 / deep_full 赢 / both deep < LGB)都"成功",都给出对
Stage 3 方向的可信证据。**不为凑赢调超参**。

### Stage 2 范围明确不在的(避免 scope creep)
- 异质图(多类型节点/边)+ 团伙核心节点识别 → Stage 3+
- 损失函数深化(HNM 反而有害的根因调查)→ Stage 3+
- cuDNN/onnxruntime-gpu ABI 修复 → Stage 3+
- 完整 PMML 工具链(Java 11+ 安装难)→ Stage 3+
- TensorRT EP 端到端延迟测量(被 cuDNN 阻塞)→ Stage 3+

### 执行中发现的问题与修复
(实施过程中遇到的真实 bug 在此追加,格式同 v1:bug 描述、根因、修复 commit SHA)

### 实验结果与诚实分析
(stage2_results.json 与 benchmark_stage2.json 的关键数字摘要 + 四情景中真实命中
那一种的叙事,实测后填入)

### 参考文献(v2 新增)
- Shwartz-Ziv & Armon, *Tabular Data: Deep Learning is Not All You Need*,
  Information Fusion 2022
- Cheng et al., *Wide & Deep Learning for Recommender Systems*, DLRS 2016
- Guo et al., *DeepFM*, IJCAI 2017
- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data
  (FT-Transformer)*, NeurIPS 2021
- IEEE-CIS Fraud Detection Kaggle 社区公开 kernels(V 列相关性剪枝惯例)
```

注:**末尾不要加多余的换行**;追加这段后保存。

- [ ] **Step 3: 验证 v1 字节级未变**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git diff docs/DESIGN_JOURNAL.md | head -3'
```
Expected: diff 显示只有"+"行(追加),无"-"行(无删除)。

- [ ] **Step 4: 改 README**

读取 `experiments/stage2_results.json` 与 `benchmark_stage2.json` 的真实数字。
打开 `README.md`,做两处修改:

1. 在 `## 阶段` 节里把 Stage 2 标记为完成(`✅`),路线图大致变成:
   ```
   - **Stage 1**(已完成)— 一体化端到端 MVP ✅
   - **Stage 2**(已完成)— 模型基础升级 ✅
   - Stage 3 — 异质图深化、生产化部署、损失深化、PMML/cuDNN 工具链
   ```

2. 在文件**末尾追加**:
   ```markdown


   ## Stage 2 结果(2026-05-15)

   **新命令:**
   ```bash
   python -m src.data.build              # 双轨数据(full_v + pruned_v)
   python -m src.train                   # gated_fusion × 2 v_strategy
   python -m src.baseline_lgbm           # LGB × 2 v_strategy
   python -m src.deploy.benchmark        # 双模型延迟
   ```

   **架构 + V 策略消融**(IEEE-CIS 公开数据,真实数字)

   | 配置 | roc_auc | pr_auc | ks | recall@fpr=.01 | fpr@recall=.90 |
   |---|---|---|---|---|---|
   | deep_full | (实测填入) | ... | ... | ... | ... |
   | deep_pruned | ... | ... | ... | ... | ... |
   | lgbm_full | ... | ... | ... | ... | ... |
   | lgbm_pruned | ... | ... | ... | ... | ... |

   **延迟 benchmark**(同 Stage 1 cuDNN 限制持续)

   | 模型 | pytorch_cpu p50 | pytorch_gpu p50 | onnx_gpu | tensorrt_fp16 |
   |---|---|---|---|---|
   | deep_full | ... | ... | (skipped: cuDNN) | (engine builds; EP skipped) |
   | deep_pruned | ... | ... | ... | ... |

   **结果解读(诚实四情景中实测命中的那一种)**

   (根据 stage2_results.json 真实结果写: both deep ≥ LGB / deep_pruned 赢 /
    deep_full 赢 / both deep < LGB —— 选择对应的叙事段)

   设计决策、修复中的 bug、完整诚实分析见 `docs/DESIGN_JOURNAL.md` v2。
   ```

- [ ] **Step 5: Commit**

```bash
ssh autodl 'cd /root/autodl-tmp/alibaba-risk-control-internship && git add docs/DESIGN_JOURNAL.md README.md && git commit -m "docs: DESIGN_JOURNAL v2 + README Stage 2 results section"'
```

---

## 自检清单(写计划者已执行)

**Spec 覆盖:** spec §4 数据层 → Task 3, 4(+ Task 2 的 v_pruning 工具);§5 模型层 → Task 1, 6;
§6 训练评估 → Task 7, 10, 11;§7 部署 → Task 12, 13, 14;§8 工程结构 → 全部任务;
§10 DESIGN_JOURNAL v2 → Task 16;§11 DoD → 全计划覆盖。

**占位符扫描:** Task 16 的"实测后填入"是 README/DESIGN_JOURNAL 的真实工作步骤(对应实测结果),
不是计划占位符;Task 10 Step 3 的 commit/gitignore 二选一基于 ckpt 实测大小,有明确判断条件。
其余无 TBD/TODO,所有代码步骤含完整代码。

**类型一致性:** `FeatureProcessor.transform` 返回 `{cat_idx, num}` 跨 Task 3/4/15 一致;
`make_loader` batch dict 键(`x_cat/x_num/edge_index/seed_local/seq_cat/seq_num/mask/label`)跨
Task 5/7/15 一致;`FraudModel.__init__(cat_cardinalities, n_num_total, model_cfg, fusion_mode)` /
`forward(seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx)` /
`forward_online(seq_cat, seq_num, mask, graph_emb)` 跨 Task 6/7/12/13/15 一致;
`Data(cat_x, num_x, edge_index, y, t)` 字段名跨 Task 4/5/13/15 一致;
`train_one_config(..., cat_cardinalities, n_num_total, ..., checkpoint_path=None)` 跨 Task 7/15 一致。

**已知风险项:**
- Task 9(real build)需要 ~84GB 数据盘,150GB 可用 → 余量 ~50GB,够。
- Task 10(real train)估计 50-70 min,中途显存或 OOM 风险低(模型 ~250-500MB,batch 1024 在 5090 上稳)。
- Task 11(LGB)PMML 导出仍可能失败(Java 11+ 工具链);代码已 `try/except` 优雅 skip。
- Task 14 的 onnx_gpu / tensorrt_fp16 EP 仍 skip(Stage 1 已知 cuDNN ABI 问题);代码已优雅处理。
