# Ant 风控实习复刻 · Stage 1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 IEEE-CIS 交易数据上构建「Transformer-GRU 序列塔 + GraphSAGE 图塔 → 门控融合」的双塔欺诈检测模型,用 Hybrid Focal Loss + Hard Negative Mining 训练,导出 ONNX/TensorRT 并做延迟 benchmark,产出可映射回简历的实验结果。

**Architecture:** 单一数据集端到端管线。数据层合成 uid → 构造序列与 time-respecting 交易图;模型层双塔 + 门控融合;训练用 NeighborLoader 驱动 batch,序列按 node_idx 侧表查询;部署层把「序列塔+融合头」导出 TensorRT,图 embedding 离线预计算。

**Tech Stack:** Python 3.12, PyTorch 2.8+cu128, PyTorch-Geometric, LightGBM, sklearn2pmml, ONNX, TensorRT, pytest, pandas, pyyaml。

**执行环境:** 所有命令在 AutoDL 主机上执行(用户通过 VSCode-SSH 连接,工作目录即 autodl 主机)。项目根目录:`/root/autodl-tmp/alibaba-risk-control-internship`(软链 `/root/alibaba-risk-control-internship`)。conda base 已装 PyTorch 2.8+cu128。

**关键接口约定(全计划一致):**
- 处理后产物存于 `data/processed/`:`graph.pt`(PyG `Data`,含 `x [Nn,Fdim]`、`edge_index`、`y [Nn]`、`t [Nn]` 时间戳)、`seq_all.pt`(`{"seq": FloatTensor[Nn,L,Fdim], "mask": BoolTensor[Nn,L]}`)、`split.pt`(`{"train_idx": LongTensor, "val_idx": LongTensor}`)、`feature_meta.json`(类别字段基数表 + 数值字段列表)。
- `Nn` = 交易节点数,`L` = `seq_len`,`Fdim` = 单笔交易特征维(类别字段编码值 + 数值字段拼接)。
- 模型 forward 签名:`FraudModel.forward(seq, mask, graph_emb) -> logit [B]`;训练时 `graph_emb` 来自图塔,部署时来自查表。
- 配置读取统一用 `src/config.py::load_config(name)`,返回 dict。

---

## 文件结构

```
alibaba-risk-control-internship/
├── environment.yml / requirements.txt
├── README.md
├── pytest.ini
├── configs/{data,model,train}.yaml
├── docs/DESIGN_JOURNAL.md
├── src/
│   ├── __init__.py
│   ├── config.py              # load_config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py            # Kaggle 下载 + 加载 + join
│   │   ├── uid.py             # uid 合成
│   │   ├── features.py        # 编码器/scaler(仅 train fit)
│   │   ├── sequence.py        # 序列构造
│   │   ├── graph.py           # time-respecting 交易图构造
│   │   └── build.py           # 串起来 + 时间切分 + manifest + 校验
│   ├── models/
│   │   ├── __init__.py
│   │   ├── losses.py          # Focal / HybridFocal / HNM 包装
│   │   ├── sequence_tower.py  # Transformer→GRU
│   │   ├── graph_tower.py     # GraphSAGE
│   │   ├── fusion.py          # 门控融合 + 消融变体
│   │   └── fraud_model.py     # 组装
│   ├── dataset.py             # NeighborLoader 驱动 + 序列侧表
│   ├── evaluate.py            # 指标
│   ├── train.py               # 训练循环 + 实验矩阵 runner
│   ├── baseline_lgbm.py       # LightGBM 基线 + PMML 导出
│   └── deploy/
│       ├── __init__.py
│       ├── export_onnx.py
│       ├── build_trt.py
│       └── benchmark.py
├── experiments/               # 实验结果 JSON
└── tests/
    ├── __init__.py
    ├── conftest.py            # 合成微数据集 fixture
    ├── test_data.py
    ├── test_losses.py
    ├── test_models.py
    └── test_smoke.py
```

---

## Phase 0 — 脚手架与环境

### Task 1: conda 环境与依赖

**Files:**
- Create: `requirements.txt`, `environment.yml`, `pytest.ini`

- [ ] **Step 1: 写 `requirements.txt`**

```
torch-geometric==2.6.1
kaggle==1.6.17
lightgbm==4.5.0
sklearn2pmml==0.110.0
onnx==1.17.0
onnxruntime-gpu==1.20.1
scikit-learn==1.5.2
pandas==2.2.3
pyarrow==18.0.0
pyyaml==6.0.2
pytest==8.3.3
tqdm==4.67.1
```

- [ ] **Step 2: 写 `environment.yml`**

```yaml
name: dfer-riskctrl
channels: [conda-forge]
dependencies:
  - python=3.12
  - pip
  - pip:
      - -r requirements.txt
```

(注:torch/torchvision 复用 base 已装的 2.8+cu128,新环境通过 `--clone base` 创建。)

- [ ] **Step 3: 写 `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: 创建环境并装依赖**

Run:
```bash
cd /root/autodl-tmp/alibaba-risk-control-internship
source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -n dfer-riskctrl --clone base
conda activate dfer-riskctrl
pip install -r requirements.txt
```
Expected: 全部安装成功,无 torch 重装(torch 来自 clone)。

- [ ] **Step 5: 验证关键库**

Run:
```bash
python -c "import torch,torch_geometric,lightgbm,onnx,sklearn2pmml,kaggle; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); import torch_geometric as g; print('pyg',g.__version__)"
```
Expected: 打印 `torch 2.8.0+cu128 cuda True` 和 `pyg 2.6.1`,无 ImportError。
TensorRT 不在此步验证(Task 20 单独处理,属已知风险项)。

- [ ] **Step 6: Commit**

```bash
git add requirements.txt environment.yml pytest.ini
git commit -m "chore: conda env and dependency manifests"
```

---

### Task 2: 项目脚手架与配置

**Files:**
- Create: `src/__init__.py`, `src/config.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/deploy/__init__.py`, `tests/__init__.py`, `configs/data.yaml`, `configs/model.yaml`, `configs/train.yaml`
- Test: `tests/test_data.py`

- [ ] **Step 1: 写失败测试 `tests/test_data.py`(config 部分)**

```python
from src.config import load_config

def test_load_config_returns_dict():
    cfg = load_config("data")
    assert isinstance(cfg, dict)
    assert cfg["seq_len"] > 0
    assert "raw_dir" in cfg and "processed_dir" in cfg

def test_load_config_unknown_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent")
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: 创建空 `__init__.py` 与配置文件**

`src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/deploy/__init__.py`, `tests/__init__.py` 均为空文件。

`configs/data.yaml`:
```yaml
raw_dir: data/raw
processed_dir: data/processed
seq_len: 32
split_ratio: 0.8            # 前 80% 时间作 train
graph_entity_cols: [card1, addr1, P_emaildomain, DeviceInfo]
graph_max_degree: 50        # 每节点入边度数封顶
graph_max_neighbors_per_entity: 20  # 同实体内连边采样上限
expected_fraud_rate: 0.035
fraud_rate_tol: 0.01
```

`configs/model.yaml`:
```yaml
d_model: 128
n_heads: 4
n_transformer_layers: 2
d_seq: 128                  # GRU hidden
d_graph: 128
graphsage_layers: 2
d_fuse: 128
mlp_hidden: 64
dropout: 0.1
cat_emb_dim: 16             # 每个类别字段 embedding 维
```

`configs/train.yaml`:
```yaml
batch_size: 1024
lr: 0.001
weight_decay: 0.00001
epochs: 20
warmup_steps: 500
grad_clip: 1.0
seed: 42
neighbor_sample: [15, 10]   # 两层 GraphSAGE 每层采样邻居数
focal_gamma_pos: 1.0
focal_gamma_neg: 4.0        # 非对称:负样本压更狠
focal_alpha: 0.25
hnm_neg_pos_ratio: 3.0      # 难负:正
early_stop_patience: 4
```

- [ ] **Step 4: 写 `src/config.py`**

```python
from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

def load_config(name: str) -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 5: 运行验证通过**

Run: `pytest tests/test_data.py -k config -v`
Expected: PASS(2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/ tests/__init__.py configs/ tests/test_data.py
git commit -m "feat: project scaffold and config loader"
```

---

## Phase 1 — 数据管线

### Task 3: Kaggle 数据下载与加载

**Files:**
- Create: `src/data/load.py`
- Test: `tests/test_data.py`(追加)

**前置(执行者手动):** 把 Kaggle API token 放到 `~/.kaggle/kaggle.json`,`chmod 600 ~/.kaggle/kaggle.json`。本计划执行时需先向用户索取该 token。

- [ ] **Step 1: 写失败测试(追加到 `tests/test_data.py`)**

```python
import pandas as pd
from src.data.load import join_transaction_identity

def test_join_left_keeps_all_transactions():
    txn = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0],
                        "TransactionDT": [10, 20, 30], "TransactionAmt": [5.0, 6.0, 7.0]})
    idn = pd.DataFrame({"TransactionID": [2], "DeviceType": ["mobile"]})
    merged = join_transaction_identity(txn, idn)
    assert len(merged) == 3
    assert merged.loc[merged.TransactionID == 1, "DeviceType"].isna().all()
    assert merged.loc[merged.TransactionID == 2, "DeviceType"].iloc[0] == "mobile"
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k join -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.load'`

- [ ] **Step 3: 写 `src/data/load.py`**

```python
import subprocess
from pathlib import Path
import pandas as pd

COMPETITION = "ieee-fraud-detection"

def download_raw(raw_dir: str) -> None:
    """用 kaggle CLI 下载并解压竞赛数据到 raw_dir。"""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "train_transaction.csv").exists():
        print(f"raw data already present at {raw}")
        return
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw)],
        check=True,
    )
    subprocess.run(
        ["unzip", "-o", str(raw / f"{COMPETITION}.zip"), "-d", str(raw)],
        check=True,
    )

def join_transaction_identity(txn: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """按 TransactionID 左连接,保留所有交易。"""
    return txn.merge(idn, on="TransactionID", how="left")

def load_raw(raw_dir: str) -> pd.DataFrame:
    """加载 train_transaction + train_identity 并 join。"""
    raw = Path(raw_dir)
    txn = pd.read_csv(raw / "train_transaction.csv")
    idn = pd.read_csv(raw / "train_identity.csv")
    return join_transaction_identity(txn, idn)
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_data.py -k join -v`
Expected: PASS

- [ ] **Step 5: 实际下载数据并核对**

Run:
```bash
python -c "from src.data.load import download_raw, load_raw; download_raw('data/raw'); df = load_raw('data/raw'); print(df.shape, 'fraud rate', round(df.isFraud.mean(), 4))"
```
Expected: 打印 `(590540, 434) fraud rate 0.035` 左右(列数因 join 而 ~434)。

- [ ] **Step 6: Commit**

```bash
git add src/data/load.py tests/test_data.py
git commit -m "feat: Kaggle data download and transaction-identity join"
```

---

### Task 4: uid 合成

**Files:**
- Create: `src/data/uid.py`
- Test: `tests/test_data.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.data.uid import synthesize_uid

def test_uid_groups_same_card_addr():
    txn = pd.DataFrame({
        "card1": [1000, 1000, 2000],
        "addr1": [50.0, 50.0, 80.0],
        "TransactionDT": [86400, 172800, 86400],  # 第1天、第2天、第1天
        "D1": [0.0, 1.0, 0.0],                     # 同卡:首交易至今天数
    })
    uid = synthesize_uid(txn)
    assert uid.iloc[0] == uid.iloc[1]   # 同卡同账户
    assert uid.iloc[0] != uid.iloc[2]

def test_uid_handles_nan():
    txn = pd.DataFrame({"card1": [1000], "addr1": [float("nan")],
                        "TransactionDT": [86400], "D1": [float("nan")]})
    uid = synthesize_uid(txn)
    assert uid.notna().all()
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k uid -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/data/uid.py`**

```python
import pandas as pd

def synthesize_uid(txn: pd.DataFrame) -> pd.Series:
    """社区标准 uid 代理:card1 + addr1 + (TransactionDay - D1)。
    这是启发式代理,非真实 ground truth(见 DESIGN_JOURNAL)。"""
    day = (txn["TransactionDT"] / 86400).astype("int64")
    anchor = (day - txn["D1"].fillna(-1)).astype("int64")
    uid = (
        txn["card1"].fillna(-1).astype("int64").astype(str)
        + "_" + txn["addr1"].fillna(-1).astype("int64").astype(str)
        + "_" + anchor.astype(str)
    )
    return uid.rename("uid")
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_data.py -k uid -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/uid.py tests/test_data.py
git commit -m "feat: uid synthesis (card1+addr1+anchor heuristic)"
```

---

### Task 5: 特征处理(仅 train fit)

**Files:**
- Create: `src/data/features.py`
- Test: `tests/test_data.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.data.features import FeatureProcessor

def test_processor_fits_on_train_only():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    val = pd.DataFrame({"ProductCD": ["A", "C"], "TransactionAmt": [30.0, 40.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    tr = fp.transform(train)
    va = fp.transform(val)
    # 未见类别 "C" 映射到 0(unknown 桶),不报错
    assert va["ProductCD"].iloc[1] == 0
    # 数值标准化用 train 统计量:train 均值处 ~0
    assert abs(tr["TransactionAmt"].mean()) < 1e-6

def test_processor_meta_has_cardinalities():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    # 基数 = 不同类别数 + 1(unknown 桶)
    assert fp.meta["cat_cardinalities"]["ProductCD"] == 3
    assert fp.meta["num_cols"] == ["TransactionAmt"]
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k processor -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/data/features.py`**

```python
import numpy as np
import pandas as pd

# IEEE-CIS 字段分组(已知 schema)
CAT_COLS = (
    ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
     "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["DeviceType", "DeviceInfo"]
    + [f"id_{i:02d}" for i in range(12, 39)]
)
NUM_COLS = (
    ["TransactionAmt", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 340)]
)

class FeatureProcessor:
    """类别字段 → 整数编码(0 = unknown 桶);数值字段 → 标准化 + 缺失指示位。
    所有统计量只在 train 上 fit。"""

    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = list(cat_cols) if cat_cols is not None else list(CAT_COLS)
        self.num_cols = list(num_cols) if num_cols is not None else list(NUM_COLS)
        self._cat_maps = {}      # col -> {value: int>=1}
        self._num_mean = {}
        self._num_std = {}
        self.meta = {}

    def fit(self, df: pd.DataFrame) -> "FeatureProcessor":
        for c in self.cat_cols:
            vals = df[c].astype(str).fillna("nan").unique()
            self._cat_maps[c] = {v: i + 1 for i, v in enumerate(sorted(vals))}
        for c in self.num_cols:
            col = df[c].astype("float64")
            self._num_mean[c] = float(col.mean()) if col.notna().any() else 0.0
            std = float(col.std()) if col.notna().any() else 1.0
            self._num_std[c] = std if std > 1e-8 else 1.0
        self.meta = {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_cardinalities": {c: len(m) + 1 for c, m in self._cat_maps.items()},
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.cat_cols:
            m = self._cat_maps[c]
            out[c] = df[c].astype(str).fillna("nan").map(m).fillna(0).astype("int64")
        for c in self.num_cols:
            col = df[c].astype("float64")
            out[c] = ((col - self._num_mean[c]) / self._num_std[c]).fillna(0.0)
            out[f"{c}__isna"] = col.isna().astype("float32")
        return out
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_data.py -k processor -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/features.py tests/test_data.py
git commit -m "feat: FeatureProcessor with train-only fitting"
```

---

### Task 6: 序列构造

**Files:**
- Create: `src/data/sequence.py`
- Test: `tests/test_data.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from src.data.sequence import build_sequences

def test_sequence_window_and_mask():
    # 2 个 uid,uid "x" 有 3 笔,uid "y" 有 1 笔
    feat = np.array([[1.0], [2.0], [3.0], [9.0]], dtype="float32")
    uid = np.array(["x", "x", "x", "y"])
    dt = np.array([10, 20, 30, 5])
    seq, mask = build_sequences(feat, uid, dt, seq_len=2)
    # 第 3 笔(uid x,dt=30)序列 = 前 2 笔 [feat0, feat1]? 不:含自身,窗口=自身+前1
    # 约定:位置 L-1 是当前交易,L-2..0 是更早的;不足则前端 padding
    assert seq.shape == (4, 2, 1)
    assert mask.shape == (4, 2)
    # uid y 只有 1 笔 → 位置 0 padding(mask False),位置 1 是自身
    yi = 3
    assert mask[yi].tolist() == [False, True]
    assert seq[yi, 1, 0] == 9.0
    # uid x 第 1 笔(dt=10)→ 位置 0 padding,位置 1 自身
    assert mask[0].tolist() == [False, True]
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k sequence -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/data/sequence.py`**

```python
import numpy as np

def build_sequences(feat: np.ndarray, uid: np.ndarray, dt: np.ndarray, seq_len: int):
    """为每笔交易构造「同 uid、含自身、按时间倒推」的滑窗序列。
    返回 seq [N, seq_len, Fdim](位置 seq_len-1 为当前交易),mask [N, seq_len](True=有效)。
    不足 seq_len 的在前端 padding。"""
    n, fdim = feat.shape
    seq = np.zeros((n, seq_len, fdim), dtype="float32")
    mask = np.zeros((n, seq_len), dtype=bool)
    order = np.lexsort((dt, uid))           # 先 uid 再 dt 排序
    # 按 uid 分组遍历
    sorted_uid = uid[order]
    group_start = 0
    for i in range(1, n + 1):
        if i == n or sorted_uid[i] != sorted_uid[group_start]:
            idxs = order[group_start:i]      # 该 uid 的全局索引,已按 dt 升序
            for pos, gi in enumerate(idxs):
                lo = max(0, pos - seq_len + 1)
                window = idxs[lo:pos + 1]    # 含自身,最多 seq_len 笔
                k = len(window)
                seq[gi, seq_len - k:] = feat[window]
                mask[gi, seq_len - k:] = True
            group_start = i
    return seq, mask
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_data.py -k sequence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/sequence.py tests/test_data.py
git commit -m "feat: per-uid sliding-window sequence construction"
```

---

### Task 7: time-respecting 交易图构造

**Files:**
- Create: `src/data/graph.py`
- Test: `tests/test_data.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from src.data.graph import build_edges

def test_edges_are_time_respecting_and_share_entity():
    # 4 笔交易,card1 列:0,1,3 共享 card1=100
    df = pd.DataFrame({
        "card1": [100, 100, 999, 100],
        "addr1": [-1, -1, -1, -1],
        "TransactionDT": [10, 20, 15, 30],
    })
    src, dst = build_edges(df, entity_cols=["card1"], max_degree=10, max_per_entity=10)
    edges = set(zip(src.tolist(), dst.tolist()))
    # 边方向:src 更早 → dst 更晚(time-respecting)
    for s, d in edges:
        assert df["TransactionDT"].iloc[s] < df["TransactionDT"].iloc[d]
    # 交易 0(dt10)→ 交易 1(dt20)应连(同 card1,0 更早)
    assert (0, 1) in edges
    # 交易 2(card1=999)不与任何人连
    assert all(2 not in e for e in edges)

def test_edges_skip_sentinel_entity():
    # addr1 = -1 是哨兵(缺失填充值),不应据此连边
    df = pd.DataFrame({"card1": [-1, -1], "addr1": [-1, -1], "TransactionDT": [10, 20]})
    src, dst = build_edges(df, entity_cols=["card1", "addr1"], max_degree=10, max_per_entity=10)
    assert len(src) == 0
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k edges -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/data/graph.py`**

```python
import numpy as np

SENTINEL = -1  # 缺失填充值,不据此连边

def build_edges(df, entity_cols, max_degree, max_per_entity):
    """构造 time-respecting 同构交易图的边。
    两笔交易若共享某高区分度实体值,则连一条「早 → 晚」有向边。
    每个实体值内连边数封顶 max_per_entity;每个节点入边度数封顶 max_degree。
    返回 (src, dst) 两个 int64 ndarray。"""
    rng = np.random.default_rng(0)
    dt = df["TransactionDT"].to_numpy()
    src_list, dst_list = [], []
    for col in entity_cols:
        vals = df[col].fillna(SENTINEL).to_numpy()
        # 按实体值分组
        order = np.argsort(vals, kind="stable")
        sv = vals[order]
        gs = 0
        for i in range(1, len(sv) + 1):
            if i == len(sv) or sv[i] != sv[gs]:
                if sv[gs] != SENTINEL and i - gs > 1:
                    members = order[gs:i]
                    members = members[np.argsort(dt[members], kind="stable")]
                    if len(members) > max_per_entity:
                        members = np.sort(rng.choice(members, max_per_entity, replace=False))
                    # 同实体内:每个更晚交易连到所有更早交易
                    for a in range(len(members)):
                        for b in range(a):
                            src_list.append(members[b])  # 早
                            dst_list.append(members[a])  # 晚
                gs = i
    if not src_list:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
    src = np.array(src_list, dtype="int64")
    dst = np.array(dst_list, dtype="int64")
    # 入边度数封顶:每个 dst 最多保留 max_degree 条入边
    keep = np.ones(len(dst), dtype=bool)
    order = np.argsort(dst, kind="stable")
    sd = dst[order]
    gs = 0
    for i in range(1, len(sd) + 1):
        if i == len(sd) or sd[i] != sd[gs]:
            if i - gs > max_degree:
                drop = order[gs:i][max_degree:]
                keep[drop] = False
            gs = i
    return src[keep], dst[keep]
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_data.py -k edges -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/graph.py tests/test_data.py
git commit -m "feat: time-respecting transaction graph edge construction"
```

---

### Task 8: 管线串联 + 时间切分 + 校验

**Files:**
- Create: `src/data/build.py`
- Test: `tests/test_data.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.data.build import time_split, validate_split

def test_time_split_is_chronological():
    dt = np.array([5, 1, 9, 3, 7])   # 乱序时间戳
    train_idx, val_idx = time_split(dt, ratio=0.6)
    # train 应是最早的 3 个(dt 1,3,5),val 是最晚 2 个(dt 7,9)
    assert set(dt[train_idx].tolist()) == {1, 3, 5}
    assert set(dt[val_idx].tolist()) == {7, 9}
    # 无重叠
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0

def test_validate_split_rejects_leak():
    import pytest
    dt = np.array([1, 2, 3, 4])
    # 故意构造泄漏:train 含 dt=4,val 含 dt=1
    with pytest.raises(AssertionError):
        validate_split(dt, train_idx=np.array([0, 3]), val_idx=np.array([1, 2]))
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_data.py -k split -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/data/build.py`**

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

    fp = FeatureProcessor()
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
        (dt[src] < dt[dst]).all(), "graph has non-time-respecting edges"
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
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_data.py -k split -v`
Expected: PASS

- [ ] **Step 5: 实际跑全量构建**

Run: `python -c "from src.data.build import build_all; build_all()"`
Expected: 打印 manifest,`fraud_rate` ≈ 0.035,`n_edges` > 0,`data/processed/` 下生成 5 个文件。

- [ ] **Step 6: Commit**

```bash
git add src/data/build.py tests/test_data.py
git commit -m "feat: end-to-end data build with time split and leak validation"
```

---

## Phase 2 — 模型层

### Task 9: 损失函数 Hybrid Focal Loss + HNM

**Files:**
- Create: `src/models/losses.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: 写失败测试 `tests/test_losses.py`**

```python
import torch
import torch.nn.functional as F
from src.models.losses import HybridFocalLoss, hard_negative_mining

def test_focal_reduces_to_bce_when_gamma_zero():
    logits = torch.tensor([0.5, -1.2, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = HybridFocalLoss(gamma_pos=0.0, gamma_neg=0.0, alpha=0.5)
    got = loss(logits, targets)
    expect = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(got, expect, atol=1e-5)

def test_asymmetric_gamma_changes_loss():
    logits = torch.tensor([0.1, 0.1])
    targets = torch.tensor([1.0, 0.0])
    sym = HybridFocalLoss(gamma_pos=2.0, gamma_neg=2.0, alpha=0.5)
    asym = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.5)
    assert not torch.allclose(sym(logits, targets), asym(logits, targets))

def test_loss_is_finite_on_extreme_logits():
    logits = torch.tensor([50.0, -50.0])
    targets = torch.tensor([0.0, 1.0])
    loss = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.25)
    assert torch.isfinite(loss(logits, targets))

def test_hnm_keeps_hardest_negatives():
    # 4 个负样本 loss 值 [0.1, 5.0, 0.2, 4.0],1 个正样本
    per_sample = torch.tensor([0.1, 5.0, 0.2, 4.0, 3.0])
    targets = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    keep = hard_negative_mining(per_sample, targets, neg_pos_ratio=2.0)
    # 1 个正样本 → 保留 2 个最难负样本(idx 1,3)+ 全部正样本(idx 4)
    assert keep.tolist() == [False, True, False, True, True]
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.losses'`

- [ ] **Step 3: 写 `src/models/losses.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFocalLoss(nn.Module):
    """非对称 Focal Loss:正负样本用不同 gamma。
    gamma_pos/gamma_neg 控制对易例的压制强度,alpha 是正类权重。
    gamma_pos=gamma_neg=0 时退化为 BCE(alpha=0.5)。数值稳定:基于 logits。"""

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction

    def per_sample(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # p = sigmoid(logits);p_t = 目标类概率
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return alpha_t * (1 - p_t).clamp(min=1e-6) ** gamma * bce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.per_sample(logits, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def hard_negative_mining(per_sample_loss: torch.Tensor, targets: torch.Tensor,
                         neg_pos_ratio: float) -> torch.Tensor:
    """OHEM:保留全部正样本 + loss 最高的 K 个负样本(K = ratio * 正样本数)。
    返回 bool mask,True = 参与反向。"""
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    keep = pos_mask.clone()
    k = min(int(neg_pos_ratio * max(n_pos, 1)), int(neg_mask.sum().item()))
    if k > 0:
        neg_losses = per_sample_loss.masked_fill(pos_mask, float("-inf"))
        hardest = torch.topk(neg_losses, k).indices
        keep[hardest] = True
    return keep
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_losses.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/losses.py tests/test_losses.py
git commit -m "feat: HybridFocalLoss and hard negative mining"
```

---

### Task 10: 序列塔 Transformer → GRU

**Files:**
- Create: `src/models/sequence_tower.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败测试 `tests/test_models.py`**

```python
import torch
from src.models.sequence_tower import SequenceTower

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
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_models.py -k sequence -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/models/sequence_tower.py`**

```python
import torch
import torch.nn as nn

class SequenceTower(nn.Module):
    """Transformer → GRU 序列塔。
    Transformer(浅,1-2 层)做跨步全局上下文混合;GRU 做带近因偏置的时序压缩。
    输入已是数值化特征向量 [B, L, feat_dim];输出 seq_emb [B, d_seq]。"""

    def __init__(self, feat_dim: int, d_model: int, n_heads: int,
                 n_layers: int, d_seq: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gru = nn.GRU(d_model, d_seq, batch_first=True)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq [B, L, feat_dim];mask [B, L](True = 有效)
        b, l, _ = seq.shape
        h = self.input_proj(seq) + self.pos_emb[:, :l]
        # Transformer 的 padding mask:True = 忽略
        pad_mask = ~mask
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        # padding 位置清零,避免污染 GRU
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)            # h_n [1, B, d_seq]
        return h_n.squeeze(0)
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_models.py -k sequence -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/sequence_tower.py tests/test_models.py
git commit -m "feat: Transformer-to-GRU sequence tower"
```

---

### Task 11: 图塔 GraphSAGE

**Files:**
- Create: `src/models/graph_tower.py`
- Test: `tests/test_models.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.models.graph_tower import GraphTower
from torch_geometric.data import Data

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
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_models.py -k graph -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/models/graph_tower.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphTower(nn.Module):
    """GraphSAGE 图塔。在交易图上聚合邻居信息,输出每节点 graph_emb。
    n_layers 层 SAGEConv,层间 ReLU + dropout。无边时退化为逐节点 MLP 行为。"""

    def __init__(self, feat_dim: int, d_graph: int, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        in_dim = feat_dim
        for _ in range(n_layers):
            self.convs.append(SAGEConv(in_dim, d_graph))
            in_dim = d_graph
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_models.py -k graph -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/graph_tower.py tests/test_models.py
git commit -m "feat: GraphSAGE graph tower"
```

---

### Task 12: 门控融合 + 消融变体

**Files:**
- Create: `src/models/fusion.py`
- Test: `tests/test_models.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.models.fusion import FusionHead

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
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_models.py -k fusion -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/models/fusion.py`**

```python
import torch
import torch.nn as nn

class FusionHead(nn.Module):
    """融合序列与图 embedding 并输出欺诈 logit。
    mode: seq_only / graph_only / concat / gated(消融用,共用代码路径)。
    gated: gate = sigmoid(W[s;g]);fused = gate*s + (1-gate)*g(逐维门控)。"""

    def __init__(self, d_seq: int, d_graph: int, d_fuse: int,
                 mlp_hidden: int, mode: str = "gated", dropout: float = 0.1):
        super().__init__()
        assert mode in {"seq_only", "graph_only", "concat", "gated"}
        self.mode = mode
        self.seq_proj = nn.Linear(d_seq, d_fuse)
        self.graph_proj = nn.Linear(d_graph, d_fuse)
        if mode == "gated":
            self.gate = nn.Linear(2 * d_fuse, d_fuse)
        head_in = 2 * d_fuse if mode == "concat" else d_fuse
        self.mlp = nn.Sequential(
            nn.Linear(head_in, mlp_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1),
        )

    def forward(self, seq_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        s = self.seq_proj(seq_emb)
        g = self.graph_proj(graph_emb)
        if self.mode == "seq_only":
            fused = s
        elif self.mode == "graph_only":
            fused = g
        elif self.mode == "concat":
            fused = torch.cat([s, g], dim=-1)
        else:  # gated
            gate = torch.sigmoid(self.gate(torch.cat([s, g], dim=-1)))
            fused = gate * s + (1 - gate) * g
        return self.mlp(fused).squeeze(-1)
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_models.py -k fusion -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/fusion.py tests/test_models.py
git commit -m "feat: gated fusion head with ablation modes"
```

---

### Task 13: 整模型组装 FraudModel

**Files:**
- Create: `src/models/fraud_model.py`
- Test: `tests/test_models.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.models.fraud_model import FraudModel

def test_fraud_model_train_forward():
    model = FraudModel(feat_dim=16, model_cfg={
        "d_model": 32, "n_heads": 4, "n_transformer_layers": 1, "d_seq": 24,
        "d_graph": 24, "graphsage_layers": 2, "d_fuse": 16, "mlp_hidden": 8,
        "dropout": 0.0}, fusion_mode="gated")
    seq = torch.randn(6, 10, 16)
    mask = torch.ones(6, 10, dtype=torch.bool)
    x = torch.randn(30, 16)
    edge_index = torch.randint(0, 30, (2, 60))
    seed = torch.arange(6)
    logit = model(seq, mask, x, edge_index, seed)
    assert logit.shape == (6,)

def test_fraud_model_online_forward_uses_precomputed_graph_emb():
    model = FraudModel(feat_dim=8, model_cfg={
        "d_model": 16, "n_heads": 2, "n_transformer_layers": 1, "d_seq": 12,
        "d_graph": 12, "graphsage_layers": 2, "d_fuse": 8, "mlp_hidden": 4,
        "dropout": 0.0}, fusion_mode="gated").eval()
    seq = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    graph_emb = torch.randn(3, 12)
    logit = model.forward_online(seq, mask, graph_emb)
    assert logit.shape == (3,)
    # 梯度流检查:train forward 下三塔都有梯度
    model.train()
    x = torch.randn(10, 8); edge_index = torch.randint(0, 10, (2, 20))
    out = model(seq, mask, x, edge_index, torch.arange(3))
    out.sum().backward()
    assert model.seq_tower.input_proj.weight.grad is not None
    assert model.graph_tower.convs[0].lin_l.weight.grad is not None
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_models.py -k fraud_model -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/models/fraud_model.py`**

```python
import torch
import torch.nn as nn
from src.models.sequence_tower import SequenceTower
from src.models.graph_tower import GraphTower
from src.models.fusion import FusionHead

class FraudModel(nn.Module):
    """双塔欺诈检测模型。
    训练:forward(seq, mask, x, edge_index, seed_idx) —— 图塔在子图上算,取 seed 节点 emb。
    部署:forward_online(seq, mask, graph_emb) —— 图 emb 离线预计算后查表传入。"""

    def __init__(self, feat_dim: int, model_cfg: dict, fusion_mode: str = "gated"):
        super().__init__()
        c = model_cfg
        self.seq_tower = SequenceTower(
            feat_dim=feat_dim, d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_transformer_layers"], d_seq=c["d_seq"], dropout=c["dropout"])
        self.graph_tower = GraphTower(
            feat_dim=feat_dim, d_graph=c["d_graph"],
            n_layers=c["graphsage_layers"], dropout=c["dropout"])
        self.fusion = FusionHead(
            d_seq=c["d_seq"], d_graph=c["d_graph"], d_fuse=c["d_fuse"],
            mlp_hidden=c["mlp_hidden"], mode=fusion_mode, dropout=c["dropout"])

    def forward(self, seq, mask, x, edge_index, seed_idx):
        seq_emb = self.seq_tower(seq, mask)
        graph_emb_all = self.graph_tower(x, edge_index)
        graph_emb = graph_emb_all[seed_idx]
        return self.fusion(seq_emb, graph_emb)

    def forward_online(self, seq, mask, graph_emb):
        seq_emb = self.seq_tower(seq, mask)
        return self.fusion(seq_emb, graph_emb)
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_models.py -k fraud_model -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/fraud_model.py tests/test_models.py
git commit -m "feat: FraudModel assembly with train and online forward paths"
```

---

## Phase 3 — 训练与评估

### Task 14: Dataset 与 NeighborLoader 驱动

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_models.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.dataset import make_loader
from torch_geometric.data import Data

def test_loader_yields_aligned_seq_and_seeds():
    n = 40
    graph = Data(x=torch.randn(n, 8),
                 edge_index=torch.randint(0, n, (2, 120)),
                 y=(torch.rand(n) > 0.9).float(),
                 t=torch.arange(n))
    seq_all = {"seq": torch.randn(n, 6, 8), "mask": torch.ones(n, 6, dtype=torch.bool)}
    idx = torch.arange(0, 20)
    loader = make_loader(graph, seq_all, idx, batch_size=8,
                         neighbor_sample=[10, 5], shuffle=False)
    batch = next(iter(loader))
    # batch 应含:子图(x, edge_index)、seed 局部索引、对齐的 seq/mask/label
    assert batch["seq"].shape[0] == batch["label"].shape[0]
    assert batch["seq"].shape[0] <= 8
    assert batch["seed_local"].shape[0] == batch["seq"].shape[0]
    # seed 局部索引指向子图内节点
    assert batch["seed_local"].max() < batch["x"].shape[0]
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_models.py -k loader -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/dataset.py`**

```python
import torch
from torch_geometric.loader import NeighborLoader

def make_loader(graph, seq_all, node_idx, batch_size, neighbor_sample,
                shuffle=True):
    """用 NeighborLoader 驱动 batch:每个 batch 是以一组 seed 交易为中心的采样子图。
    seq/mask/label 按 seed 的原始 node id 从侧表查询,保证对齐。
    yield dict: x, edge_index, seed_local, seq, mask, label。"""
    seq_t = seq_all["seq"]
    mask_t = seq_all["mask"]
    y = graph.y

    base = NeighborLoader(
        graph, num_neighbors=neighbor_sample, input_nodes=node_idx,
        batch_size=batch_size, shuffle=shuffle,
    )

    class _Wrapped:
        def __init__(self, loader):
            self.loader = loader
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            for b in self.loader:
                bs = b.batch_size
                seed_global = b.n_id[:bs]          # seed 的原始 node id
                yield {
                    "x": b.x,
                    "edge_index": b.edge_index,
                    "seed_local": torch.arange(bs),  # NeighborLoader 把 seed 排在前 bs 个
                    "seq": seq_t[seed_global],
                    "mask": mask_t[seed_global],
                    "label": y[seed_global],
                }

    return _Wrapped(base)
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_models.py -k loader -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_models.py
git commit -m "feat: NeighborLoader-driven dataset with aligned sequence side-table"
```

---

### Task 15: 评估指标

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_losses.py`(追加,复用 metric 性质易测)

- [ ] **Step 1: 写失败测试(追加到 `tests/test_losses.py`)**

```python
import numpy as np
from src.evaluate import compute_metrics, recall_at_fixed_fpr

def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_score)
    assert abs(m["roc_auc"] - 1.0) < 1e-6
    assert abs(m["pr_auc"] - 1.0) < 1e-6
    assert 0.0 <= m["ks"] <= 1.0

def test_recall_at_fixed_fpr():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    # FPR <= 0.34 时阈值可把 3 个负样本都判负,2 个正样本都判正 → recall 1.0
    r = recall_at_fixed_fpr(y_true, y_score, fpr=0.34)
    assert abs(r - 1.0) < 1e-6
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_losses.py -k metrics -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/evaluate.py`**

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def recall_at_fixed_fpr(y_true, y_score, fpr: float) -> float:
    """在给定最大 FPR 约束下能达到的最高召回(TPR)。"""
    fprs, tprs, _ = roc_curve(y_true, y_score)
    ok = fprs <= fpr
    return float(tprs[ok].max()) if ok.any() else 0.0

def fpr_at_fixed_recall(y_true, y_score, recall: float) -> float:
    """在保证至少 recall 召回时的最低 FPR(误伤率)。"""
    fprs, tprs, _ = roc_curve(y_true, y_score)
    ok = tprs >= recall
    return float(fprs[ok].min()) if ok.any() else 1.0

def compute_metrics(y_true, y_score) -> dict:
    """主指标集合:ROC-AUC / PR-AUC / KS / 工作点指标。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fprs, tprs, _ = roc_curve(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": float(np.max(tprs - fprs)),
        "recall_at_fpr_0.01": recall_at_fixed_fpr(y_true, y_score, 0.01),
        "fpr_at_recall_0.90": fpr_at_fixed_recall(y_true, y_score, 0.90),
    }
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_losses.py -k metrics -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_losses.py
git commit -m "feat: evaluation metrics (ROC-AUC, PR-AUC, KS, operating points)"
```

---

### Task 16: 训练循环 + 实验矩阵 runner

**Files:**
- Create: `src/train.py`
- Test: `tests/test_smoke.py`(部分,完整 smoke 在 Task 22)

- [ ] **Step 1: 写失败测试 `tests/test_smoke.py`(训练单元部分)**

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_smoke.py -k train_one_config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: 写 `src/train.py`**

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
    torch.manual_seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for b in loader:
        logit = model(b["seq"].to(device), b["mask"].to(device),
                      b["x"].to(device), b["edge_index"].to(device),
                      b["seed_local"].to(device))
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(b["label"].numpy())
    return compute_metrics(np.concatenate(labels), np.concatenate(scores))

def train_one_config(graph, seq_all, split, fusion_mode, use_hnm,
                     model_cfg, train_cfg, device="cuda"):
    """训练单个配置并返回 val 指标。配置 = (fusion_mode, use_hnm, loss 参数)。"""
    _set_seed(train_cfg["seed"])
    feat_dim = graph.x.shape[1]
    model = FraudModel(feat_dim, model_cfg, fusion_mode=fusion_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    loss_fn = HybridFocalLoss(train_cfg["focal_gamma_pos"],
                              train_cfg["focal_gamma_neg"],
                              train_cfg["focal_alpha"], reduction="none")
    train_loader = make_loader(graph, seq_all, split["train_idx"],
                               train_cfg["batch_size"], train_cfg["neighbor_sample"])
    val_loader = make_loader(graph, seq_all, split["val_idx"],
                             train_cfg["batch_size"], train_cfg["neighbor_sample"],
                             shuffle=False)

    best_pr, best_metrics, patience = -1.0, None, 0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for b in train_loader:
            logit = model(b["seq"].to(device), b["mask"].to(device),
                          b["x"].to(device), b["edge_index"].to(device),
                          b["seed_local"].to(device))
            target = b["label"].to(device)
            per_sample = loss_fn.per_sample(logit, target)
            if use_hnm:
                keep = hard_negative_mining(per_sample.detach(), target,
                                            train_cfg["hnm_neg_pos_ratio"])
                loss = per_sample[keep].mean()
            else:
                loss = per_sample.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            opt.step()
        metrics = _evaluate(model, val_loader, device)
        if metrics["pr_auc"] > best_pr:
            best_pr, best_metrics, patience = metrics["pr_auc"], metrics, 0
        else:
            patience += 1
            if patience >= train_cfg["early_stop_patience"]:
                break
    return best_metrics

# 实验矩阵:每行 = 一个对比配置,映射回简历 bullet
EXPERIMENT_MATRIX = [
    {"name": "seq_only",        "fusion_mode": "seq_only",  "use_hnm": False},
    {"name": "graph_only",      "fusion_mode": "graph_only","use_hnm": False},
    {"name": "concat_fusion",   "fusion_mode": "concat",    "use_hnm": False},
    {"name": "gated_fusion",    "fusion_mode": "gated",     "use_hnm": False},
    {"name": "gated_plus_hnm",  "fusion_mode": "gated",     "use_hnm": True},
]

def run_experiment_matrix(device="cuda"):
    """跑全部实验矩阵,结果落 experiments/results.json。"""
    graph = torch.load("data/processed/graph.pt", weights_only=False)
    seq_all = torch.load("data/processed/seq_all.pt", weights_only=False)
    split = torch.load("data/processed/split.pt", weights_only=False)
    model_cfg = load_config("model")
    train_cfg = load_config("train")

    results = {}
    for exp in EXPERIMENT_MATRIX:
        t0 = time.time()
        metrics = train_one_config(graph, seq_all, split, exp["fusion_mode"],
                                   exp["use_hnm"], model_cfg, train_cfg, device)
        metrics["train_seconds"] = round(time.time() - t0, 1)
        results[exp["name"]] = metrics
        print(exp["name"], metrics)

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    run_experiment_matrix()
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_smoke.py -k train_one_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_smoke.py
git commit -m "feat: training loop and experiment matrix runner"
```

---

### Task 17: 跑全量实验矩阵

**Files:** 无新文件,产出 `experiments/results.json`

- [ ] **Step 1: 跑实验矩阵**

Run:
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl
python -m src.train
```
Expected: 5 个配置依次训练完成,打印各自指标,生成 `experiments/results.json`。

- [ ] **Step 2: 核对结果合理性**

Run: `python -c "import json; r=json.load(open('experiments/results.json')); [print(k, 'roc', round(v['roc_auc'],4), 'pr', round(v['pr_auc'],4)) for k,v in r.items()]"`
Expected:
- `gated_fusion` 的 `roc_auc` ≥ `seq_only`(融合带来增益,验证互补性)
- `gated_plus_hnm` 的 `fpr_at_recall_0.90` ≤ `gated_fusion`(HNM 降误伤)
- 各 `roc_auc` 落在 0.85–0.96 区间(若全 ≈ 0.5,说明有 bug,需排查)

> 注:若 `gated_fusion` 未超过 `seq_only`,这是真实结果,不强行调到"超过"。按 spec §0 诚实原则记录,在 DESIGN_JOURNAL 分析原因(IEEE-CIS 构造图信号偏弱属已知前提)。

- [ ] **Step 3: Commit**

```bash
git add experiments/results.json
git commit -m "experiment: stage 1 architecture and loss ablation results"
```

---

### Task 18: LightGBM 基线 + PMML 导出

**Files:**
- Create: `src/baseline_lgbm.py`
- Test: `tests/test_smoke.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
from src.baseline_lgbm import train_lgbm_baseline

def test_lgbm_baseline_runs(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(300, 10)); y_train = (x_train[:, 0] > 0).astype(float)
    x_val = rng.normal(size=(100, 10)); y_val = (x_val[:, 0] > 0).astype(float)
    metrics, model = train_lgbm_baseline(x_train, y_train, x_val, y_val)
    assert metrics["roc_auc"] > 0.9        # x[:,0] 强信号,应学得到
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_smoke.py -k lgbm -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/baseline_lgbm.py`**

```python
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
import torch
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.preprocessing import FunctionTransformer

from src.evaluate import compute_metrics

def train_lgbm_baseline(x_train, y_train, x_val, y_val):
    """在扁平表特征上训 LightGBM。返回 (val 指标, 模型)。"""
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_val)[:, 1]
    return compute_metrics(y_val, scores), clf

def export_pmml(clf, path: str):
    """把 LightGBM 模型导出为 PMML(异构部署的轻量模型那一路)。"""
    pipe = PMMLPipeline([("identity", FunctionTransformer()), ("clf", clf)])
    sklearn2pmml(pipe, path)

def run_baseline():
    """用处理后的全量特征(取每笔交易当前步)训基线 + 导出 PMML。"""
    graph = torch.load("data/processed/graph.pt", weights_only=False)
    split = torch.load("data/processed/split.pt", weights_only=False)
    x = graph.x.numpy(); y = graph.y.numpy()
    tr, va = split["train_idx"].numpy(), split["val_idx"].numpy()
    metrics, clf = train_lgbm_baseline(x[tr], y[tr], x[va], y[va])
    print("lgbm baseline:", metrics)

    Path("experiments").mkdir(exist_ok=True)
    results = json.load(open("experiments/results.json")) \
        if Path("experiments/results.json").exists() else {}
    results["lgbm_baseline"] = metrics
    json.dump(results, open("experiments/results.json", "w"), indent=2)

    Path("artifacts").mkdir(exist_ok=True)
    export_pmml(clf, "artifacts/lgbm_baseline.pmml")
    print("PMML exported to artifacts/lgbm_baseline.pmml")

if __name__ == "__main__":
    run_baseline()
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_smoke.py -k lgbm -v`
Expected: PASS

- [ ] **Step 5: 跑真实基线 + PMML 导出**

Run: `python -m src.baseline_lgbm`
Expected: 打印基线指标,`experiments/results.json` 增加 `lgbm_baseline` 项,生成 `artifacts/lgbm_baseline.pmml`。

- [ ] **Step 6: Commit**

```bash
git add src/baseline_lgbm.py tests/test_smoke.py experiments/results.json
git commit -m "feat: LightGBM baseline with PMML export"
```

---

## Phase 4 — 部署与延迟 Benchmark

### Task 19: ONNX 导出

**Files:**
- Create: `src/deploy/export_onnx.py`
- Test: `tests/test_smoke.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_smoke.py -k onnx -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/deploy/export_onnx.py`**

```python
import numpy as np
import torch
import onnxruntime as ort

class _OnlineWrapper(torch.nn.Module):
    """只暴露在线路径(序列塔 + 融合头),供 ONNX 导出。"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, seq, mask, graph_emb):
        return self.model.forward_online(seq, mask, graph_emb)

def export_online_path(model, feat_dim, seq_len, d_graph, path):
    """把 FraudModel 的在线路径导出为 ONNX(动态 batch 轴)。"""
    wrapper = _OnlineWrapper(model).eval()
    seq = torch.randn(2, seq_len, feat_dim)
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(2, d_graph)
    torch.onnx.export(
        wrapper, (seq, mask, graph_emb), path,
        input_names=["seq", "mask", "graph_emb"], output_names=["logit"],
        dynamic_axes={"seq": {0: "batch"}, "mask": {0: "batch"},
                      "graph_emb": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )

def verify_onnx_parity(model, onnx_path, feat_dim, seq_len, d_graph, atol=1e-4):
    """校验 ONNX 输出与 PyTorch 一致。"""
    model.eval()
    seq = torch.randn(4, seq_len, feat_dim)
    mask = torch.ones(4, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(4, d_graph)
    with torch.no_grad():
        torch_out = model.forward_online(seq, mask, graph_emb).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "seq": seq.numpy(), "mask": mask.numpy(), "graph_emb": graph_emb.numpy(),
    })[0]
    return bool(np.allclose(torch_out, onnx_out, atol=atol))
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_smoke.py -k onnx -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deploy/export_onnx.py tests/test_smoke.py
git commit -m "feat: ONNX export of online inference path with parity check"
```

---

### Task 20: TensorRT 引擎构建(已知风险项)

**Files:**
- Create: `src/deploy/build_trt.py`

> **风险说明:** TensorRT 需匹配 CUDA 12.8。先尝试 `pip install tensorrt`;若失败,执行者需向用户确认是否走 NVIDIA 源或跳过 TRT(benchmark 仍可跑其余 3 档)。本任务在 TRT 不可用时优雅降级。

- [ ] **Step 1: 安装并验证 TensorRT**

Run:
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dfer-riskctrl
pip install tensorrt
python -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
```
Expected: 打印 tensorrt 版本。若失败 → 标记 TRT 不可用,记入 DESIGN_JOURNAL,继续 Task 21(跳过 TRT 档)。

- [ ] **Step 2: 写 `src/deploy/build_trt.py`**

```python
import os

def trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> bool:
    """用 TensorRT Python API 把 ONNX 编译为独立 TensorRT 引擎(FP16),产出 .engine 工件。
    返回是否成功。引擎硬件专属:本机 RTX 5090 / CUDA 12.8。
    注:延迟 benchmark(Task 21)为保证测量口径一致,统一走 ORT 的 TensorRT EP;
    本函数产出的独立引擎是单独的部署工件,也验证 TRT 编译链路可用。"""
    if not trt_available():
        print("TensorRT not available, skipping engine build")
        return False
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolFlag.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # 动态 batch:min 1 / opt 64 / max 256
    name_to_input = {network.get_input(i).name: network.get_input(i)
                     for i in range(network.num_inputs)}
    for name in ["seq", "mask", "graph_emb"]:
        shape = list(name_to_input[name].shape)
        profile.set_shape(name, [1] + shape[1:], [64] + shape[1:], [256] + shape[1:])
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"engine written to {engine_path}")
    return True
```

- [ ] **Step 3: 实际构建引擎(若 TRT 可用)**

先确保有训练好的模型导出的 ONNX(Task 21 Step 1 会生成 `artifacts/online.onnx`)。本步在 Task 21 内联调用,此处仅验证模块可导入:
Run: `python -c "from src.deploy.build_trt import trt_available, build_engine; print('trt available:', trt_available())"`
Expected: 打印 `trt available: True` 或 `False`,无异常。

- [ ] **Step 4: Commit**

```bash
git add src/deploy/build_trt.py
git commit -m "feat: TensorRT engine builder with graceful degradation"
```

---

### Task 21: 延迟 Benchmark

**Files:**
- Create: `src/deploy/benchmark.py`
- Test: `tests/test_smoke.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest tests/test_smoke.py -k benchmark -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写 `src/deploy/benchmark.py`**

```python
import json
import time
from pathlib import Path
import numpy as np
import torch

def _percentiles(times_ms):
    arr = np.array(times_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }

def _make_inputs(batch, seq_len, feat_dim, d_graph, device):
    return (torch.randn(batch, seq_len, feat_dim, device=device),
            torch.ones(batch, seq_len, dtype=torch.bool, device=device),
            torch.randn(batch, d_graph, device=device))

def benchmark_torch(model, feat_dim, seq_len, d_graph, device,
                    n_runs=1000, warmup=50, batch=1):
    """benchmark PyTorch eager 单请求延迟(batch=1 默认)。"""
    model = model.to(device).eval()
    seq, mask, graph_emb = _make_inputs(batch, seq_len, feat_dim, d_graph, device)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_online(seq, mask, graph_emb)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.forward_online(seq, mask, graph_emb)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)

def benchmark_onnx(onnx_path, feat_dim, seq_len, d_graph, providers,
                   n_runs=1000, warmup=50, batch=1):
    """benchmark ONNXRuntime 延迟(providers 控制 CPU/GPU)。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=providers)
    seq = np.random.randn(batch, seq_len, feat_dim).astype("float32")
    mask = np.ones((batch, seq_len), dtype=bool)
    graph_emb = np.random.randn(batch, d_graph).astype("float32")
    feed = {"seq": seq, "mask": mask, "graph_emb": graph_emb}
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)

def run_benchmark():
    """4 档对比:PyTorch-CPU / PyTorch-GPU / ONNX-GPU / TensorRT-FP16。
    结果落 experiments/benchmark.json。"""
    from src.config import load_config
    from src.models.fraud_model import FraudModel
    from src.deploy.export_onnx import export_online_path, verify_onnx_parity
    from src.deploy.build_trt import trt_available, build_engine

    mcfg = load_config("model")
    dcfg = load_config("data")
    seq_len = dcfg["seq_len"]
    feat_dim = json.load(open("data/processed/manifest.json"))["feat_dim"]
    d_graph = mcfg["d_graph"]

    model = FraudModel(feat_dim, mcfg, fusion_mode="gated")
    ckpt = Path("artifacts/best_model.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    Path("artifacts").mkdir(exist_ok=True)
    onnx_path = "artifacts/online.onnx"
    export_online_path(model, feat_dim, seq_len, d_graph, onnx_path)
    assert verify_onnx_parity(model, onnx_path, feat_dim, seq_len, d_graph), \
        "ONNX parity failed — 不信任后续延迟数字"

    results = {}
    results["pytorch_cpu"] = benchmark_torch(model, feat_dim, seq_len, d_graph, "cpu")
    if torch.cuda.is_available():
        results["pytorch_gpu"] = benchmark_torch(model, feat_dim, seq_len, d_graph, "cuda")
        results["onnx_gpu"] = benchmark_onnx(
            onnx_path, feat_dim, seq_len, d_graph, ["CUDAExecutionProvider"])
    if trt_available():
        if build_engine(onnx_path, "artifacts/online.engine", fp16=True):
            results["tensorrt_fp16"] = benchmark_onnx(
                onnx_path, feat_dim, seq_len, d_graph,
                [("TensorrtExecutionProvider", {"trt_fp16_enable": True})])
    else:
        results["tensorrt_fp16"] = {"skipped": "TensorRT not available"}

    with open("experiments/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(k, v)
    return results

if __name__ == "__main__":
    run_benchmark()
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest tests/test_smoke.py -k benchmark -v`
Expected: PASS

- [ ] **Step 5: 跑真实 benchmark**

先确保有 best checkpoint(Task 16 的 `train_one_config` 需补存 checkpoint;若 `artifacts/best_model.pt` 不存在则用随机初始化模型 benchmark,数字仍有效因为只测延迟不测精度)。
Run: `python -m src.deploy.benchmark`
Expected: 打印 4 档(或可用档)延迟,生成 `experiments/benchmark.json`。TensorRT 档应明显快于 PyTorch-CPU 档。

- [ ] **Step 6: Commit**

```bash
git add src/deploy/benchmark.py tests/test_smoke.py experiments/benchmark.json
git commit -m "feat: 4-config latency benchmark (CPU/GPU/ONNX/TensorRT)"
```

---

## Phase 5 — 端到端 smoke 测试与文档

### Task 22: 端到端 smoke 测试

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_smoke.py`(追加端到端用例)

- [ ] **Step 1: 写 `tests/conftest.py`(合成微数据集 fixture)**

```python
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def tiny_raw_df():
    """合成 ~500 行、结构同 IEEE-CIS 关键列的微数据集。"""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": (rng.random(n) < 0.04).astype(int),
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 30, n)),
        "TransactionAmt": rng.exponential(50, n),
        "ProductCD": rng.choice(list("ABCDE"), n),
        "card1": rng.integers(1000, 1050, n),
        "addr1": rng.integers(50, 60, n).astype(float),
        "D1": rng.integers(0, 100, n).astype(float),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "DeviceInfo": rng.choice(["iOS", "Windows", "Android"], n),
    })
    return df
```

- [ ] **Step 2: 写失败测试(追加到 `tests/test_smoke.py`)**

```python
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
```

- [ ] **Step 3: 运行验证(应直接通过 —— 依赖的模块已实现)**

Run: `pytest tests/test_smoke.py -k end_to_end -v`
Expected: PASS

- [ ] **Step 4: 跑全套测试**

Run: `pytest -v`
Expected: 所有测试 PASS(test_data / test_losses / test_models / test_smoke)。

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_smoke.py
git commit -m "test: end-to-end smoke test on synthetic micro-dataset"
```

---

### Task 23: README + DESIGN_JOURNAL v1

**Files:**
- Create: `README.md`, `docs/DESIGN_JOURNAL.md`

- [ ] **Step 1: 写 `README.md`**

````markdown
# 阿里/蚂蚁风控算法组实习 —— 复刻项目(Stage 1)

在 IEEE-CIS 公开交易数据上复刻实习的技术方法:Transformer-GRU 序列塔 + GraphSAGE 图塔
→ 门控融合,Hybrid Focal Loss + Hard Negative Mining 训练,ONNX/TensorRT 部署 benchmark。

## 诚实声明

本项目用公开数据复现**方法论与改进方向**,非蚂蚁生产数据/环境。简历中的业务数字
(AUC 0.98、资损 -8%)绑定于专有数据,无法且不复现。所有数字均为本项目在 IEEE-CIS 上的真实结果。

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -n dfer-riskctrl --clone base
conda activate dfer-riskctrl
pip install -r requirements.txt
```

## 运行

```bash
# 1. 数据(需 ~/.kaggle/kaggle.json)
python -c "from src.data.build import build_all; build_all()"
# 2. 实验矩阵
python -m src.train
# 3. LightGBM 基线 + PMML
python -m src.baseline_lgbm
# 4. 部署 benchmark
python -m src.deploy.benchmark
# 测试
pytest -v
```

## 结果 → 简历 bullet 映射

| 简历 bullet | 对应实验 | 结果文件 |
|-------------|---------|---------|
| Transformer-GRU 行为序列 + GNN 团伙识别 | `gated_fusion` vs `seq_only` 架构消融 | `experiments/results.json` |
| Hybrid Focal Loss + HNM 处理极不平衡 | `gated_plus_hnm` vs `gated_fusion`(看 fpr@recall=0.90) | `experiments/results.json` |
| 离线 AUC | 各配置 `roc_auc` / `pr_auc` | `experiments/results.json` |
| PMML/TensorRT 异构部署 + 延迟优化 | 4 档延迟 benchmark + PMML 导出 | `experiments/benchmark.json` |

## 阶段

- **Stage 1**(本仓库当前)— 一体化端到端 MVP
- Stage 2 — 异质图深化、团伙核心节点识别
- Stage 3 — 生产化可部署系统(Triton 服务、INT8、算子优化)

设计演进见 `docs/DESIGN_JOURNAL.md`,完整设计见 `docs/superpowers/specs/`。
````

- [ ] **Step 2: 写 `docs/DESIGN_JOURNAL.md`**

````markdown
# 设计日志(DESIGN JOURNAL)

版本化、累积式设计记录。每次设计更新**追加新版本小节**,不覆盖旧记录。

---

## v1 (2026-05-14) — Stage 1 初始设计

### 决策 1:路线 A —— 双塔融合一体化模型
**初衷:** 简历描述的是行为序列 + 图 + 损失 + 部署联动的一个系统,而非三个零件。
**原理:** 序列塔捕捉用户自身时序行为(账户盗用、行为突变);图塔捕捉跨实体结构信号
(团伙、资金归集);门控融合逐样本自适应权衡两路。
**文献:** RAGFormer(arXiv:2402.17472)"GNN 学全局特征互补 Transformer 局部特征";
ETH-GBERT(arXiv:2501.02032)全局结构 + 局部语义动态融合。

### 决策 2:Transformer → GRU 顺序
**初衷:** 给当前交易打分,需要"用户当前状态 + 全局上下文"的表示。
**原理:** Transformer 自注意力先把每步全局重表示,GRU 再带近因偏置地压缩成末隐状态。
Transformer 保持浅(1-2 层)避免与 GRU 职责重叠。
**文献:** FTT-GRU(arXiv:2511.00564);Attention-Based Transformer+GRU(MDPI Math 13(9):1484)。

### 决策 3:Hybrid Focal Loss = 非对称 Focal + 类别平衡 α
**初衷:** 万分位级不平衡,需在保召回的同时压误伤。
**原理:** γ_pos ≠ γ_neg 提供召回/误伤调节旋钮;α 处理类别失衡;HNM(OHEM)
进一步专攻会变成误报的难负样本。
**文献:** Focal Loss(Lin 2017);Asymmetric Loss(Ben-Baruch 2020);
Class-Balanced Loss(Cui 2019);OHEM(Shrivastava 2016)。

### 决策 4:图 embedding 离线预计算用于部署
**初衷:** GNN 动态图结构对 TensorRT 不友好。
**原理:** 训练时联合训练(NeighborLoader 驱动);部署时图 emb 离线算好查表,
只有序列塔 + 融合头走 TensorRT。符合工业界做法。

### 决策 5:uid 合成用 card1+addr1+(day−D1) 启发式
**初衷:** IEEE-CIS 无用户 ID。
**局限:** 这是社区标准启发式代理,非真实 ground truth,可能误合并/漏合并账户。

### 诚实前提
- "模型互补"是经验性结论,由架构消融实验验证,不假设。
- IEEE-CIS 非原生图数据集,图按共享实体构造,信号强度中等;强图故事留 Stage 2。
- 简历业务数字绑定蚂蚁专有数据,不复现;本项目报公开数据上的真实数字。

### 实验结果
(Task 17 / 21 完成后填入实际数字)
````

- [ ] **Step 3: 填入实验结果**

把 `experiments/results.json` 和 `experiments/benchmark.json` 的关键数字摘要填入 DESIGN_JOURNAL v1 的「实验结果」小节(架构消融对比、损失消融对比、4 档延迟)。

- [ ] **Step 4: Commit**

```bash
git add README.md docs/DESIGN_JOURNAL.md
git commit -m "docs: README with result-to-resume mapping and DESIGN_JOURNAL v1"
```

---

## 自检清单(写计划者已执行)

**Spec 覆盖:** spec §4 数据管线 → Task 3-8;§5 模型层 → Task 9-13;§6 训练评估 → Task 14-17;
§5.5 LightGBM/PMML → Task 18;§7 部署 benchmark → Task 19-21;§8 工程结构 → Task 1-2;
§9 版本化 README → Task 23;§10 错误处理与测试 → 贯穿各 Task + Task 22;§11 DoD → 全计划覆盖。

**占位符扫描:** 无 TBD/TODO;所有代码步骤含完整代码;所有命令含预期输出。

**类型一致性:** `FraudModel.forward` / `forward_online` 签名跨 Task 13/16/19/21 一致;
`make_loader` 返回 dict 的键(x/edge_index/seed_local/seq/mask/label)跨 Task 14/16 一致;
`HybridFocalLoss.per_sample` 跨 Task 9/16 一致;`compute_metrics` 返回键跨 Task 15/16/18 一致。

**已知风险项:** Task 20 TensorRT 安装(CUDA 12.8 匹配)—— 已设计优雅降级;
Task 21 best checkpoint —— 已说明随机初始化模型 benchmark 延迟仍有效。
````
