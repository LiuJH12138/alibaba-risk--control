# Stage 3a Design: Heterogeneous Graph + Loss Deepening

**Date:** 2026-05-15
**Project:** alibaba-risk-control-internship (IEEE-CIS Fraud Detection reproduction)
**Stage:** 3a — Heterogeneous graph modeling + loss deepening + post-hoc gang core identification
**Predecessor:** Stage 2 (per-field categorical embeddings + full V-column ablation)

---

## 0. 设计初衷与诚实约束

Stage 2 完成模型基础升级后,best deep PR-AUC = **0.4370** (deep_full),best LGB PR-AUC = **0.5556** (lgbm_full),差 ≈ **0.12**(注:Stage 2 deep_pruned PR-AUC = 0.4312,LGBM 在不平衡欺诈检测的 PR-AUC 优势远大于 ROC-AUC 表面看到的差距)。Stage 2 给出诚实结论:**仅特征工程升级不足以反超 LGB**。Stage 3a 假设差距来自 Stage 2 同构图的两个结构性缺陷:

1. 实体(card1/addr1/P_emaildomain/DeviceInfo)的风险先验被埋在 ID 嵌入里,无法显式传播
2. "团伙"信号被均匀稀释到大量 transaction 节点的相邻边

异质图把实体提升为独立节点、赋予 5 维聚合特征(train-only 防泄漏),让团伙信号在 entity 节点处汇聚——这是简历 "异质图建模" 的真实落地形式。

同时,Stage 1 的 `gated_plus_hnm` 配置在 228 秒内被早停误杀(epoch=6 触发 patience=4),Stage 3a 增加 5 重收敛保证,确保每个配置展示真正最优而非半途结果。

**诚实四情景框架(任何结果都"成功")**:

- hetero best PR-AUC > **0.5556** (best LGB lgbm_full) → 深度模型反超传统模型 ✅
- hetero best PR-AUC ∈ (**0.4370**, 0.5556) → 异质图有效但未超 LGB
- hetero best PR-AUC ≈ **0.4370** (best Stage 2 deep deep_full) → 异质图在本数据集帮助有限
- hetero best PR-AUC < **0.4370** → 实现需排查或同构已够

**绝不调超参为了凑赢 LGBM。**

---

## 1. 整体架构与变更总览

**一句话**:Stage 2 同构交易图 → 异质图(transaction + 4 类实体作为独立节点);`GraphTower` → `HeteroGraphTower`(PyG `HeteroConv`);4 配置覆盖图 × 损失;训练加收敛保证;训完跑团伙后处理。

### 1.1 端到端变更图(Δ on top of Stage 2)

```
data/processed/pruned_v/...          (Stage 2 已有,复用)
   │
   ├─[数据层 Δ] build.py 加 build_hetero_graph(df, ...) 产出 HeteroData:
   │           节点:transaction (~590K) + card1 (~12.7K) + addr1 (~500)
   │                  + P_emaildomain (~60) + DeviceInfo (~2K)
   │           节点特征:
   │             - transaction: 沿用 Stage 2 的 cat_x + num_x(共享 mixer 输入)
   │             - 实体:每个实体 5 维聚合统计 (train-only)
   │           边类型(双向 + 1 单向 time-respecting):
   │             ('transaction','paid_with','card1')                + 反向
   │             ('transaction','shipped_to','addr1')               + 反向
   │             ('transaction','sent_to_email','P_emaildomain')    + 反向
   │             ('transaction','on_device','DeviceInfo')           + 反向
   │             ('transaction','next_by_uid','transaction')        (单向时间)
   │           落 data/processed/pruned_v/hetero_graph.pt
   │
   ├─[模型层 Δ] HeteroGraphTower(替换 GraphTower 作主选,GraphTower 保留 ablation):
   │           HeteroConv 包 9 条 SAGEConv,2 层,每层后 ReLU + Dropout
   │           输出 transaction 节点的 d_graph 维 emb [N_seed, d_graph]
   │
   ├─[模型层 Δ] EmbeddingMixer 仍只对 transaction 节点做;实体节点的 entity_feat
   │           是 train 统计预计算的稠密向量,直接当 HeteroData 节点特征
   │
   ├─[损失层 Δ] losses.py 加 3 个新选项/配置:
   │           1. asym_balanced:γ_pos=2, γ_neg=6, α=0.4
   │           2. label_smoothing:在 BCE 基础上加 ε=0.1
   │           3. HNM 根因诊断模式:每 epoch 打"被丢弃负样本"分布
   │
   ├─[训练层 Δ] 收敛保证:epochs 20→40, patience 4→8, min_epochs=10
   │           train.py 加 _record_epoch_metrics() 每 epoch 写 history JSON
   │           加 _convergence_audit() 训练结束打印审计 + warnings
   │           run_stage3a_matrix() 跑 4 配置,落 stage3a_results.json + 4 history JSON
   │
   ├─[分析层 Δ] 新增 src/analysis/centrality.py:
   │           PageRank + degree centrality on fraud subgraph
   │           输出 core_entities.json(top-K 高分 entity per type)
   │
   ├─[可视化 Δ] 新增 src/analysis/plot_curves.py:
   │           读 history JSON → matplotlib 3 子图 → curves_{name}.png
   │
   └─[部署层] 不动(Stage 2 链路保留;hetero 部署留 Stage 3+)
```

### 1.2 不变项(scope creep 防护)

- 序列塔 + 融合头 + EmbeddingMixer:不动
- FeatureProcessor 接口、make_loader(主体)、Stage 2 修复:全部保留
- ONNX/benchmark:不在 Stage 3a 范围
- 同构 GraphTower:**保留**作 ablation 对照基础

### 1.3 收敛保证(贯穿 4 配置,新增硬要求)

1. epochs 40,patience 8,min_epochs 10,warmup 500 步
2. 每 epoch 全量指标 → `experiments/training_history_{name}.json`
3. 训练曲线图自动生成 → `experiments/curves_{name}.png`(3 子图,标 best_epoch 红线)
4. 收敛断言 + warnings:best_epoch == 末尾 / 总 epoch < 15 / 末 5 epoch 震荡 > 0.02
5. 配置间内存隔离 + 子进程 fallback,避免 Stage 2 SIGKILL 复现

---

## 2. 数据层(HeteroData 构造 + 实体节点特征工程)

### 2.1 节点 schema

| 节点类型 | 数量级 | ID 来源 | 节点特征(`x`) |
|---------|-------|---------|---------------|
| `transaction` | ~590K | TransactionID 排序后 0..N-1 | Stage 2 已有 (cat_idx, num),走 `EmbeddingMixer` 投影 |
| `card1` | ~12.7K | 训练集出现的所有 unique 值 | 5 维聚合 |
| `addr1` | ~500 | 同上 | 5 维聚合 |
| `P_emaildomain` | ~60 | 同上 | 5 维聚合 |
| `DeviceInfo` | ~2K | 同上(NaN → "_NA_" 单独类) | 5 维聚合 |

**实体节点的 5 维聚合特征**(只用训练集统计):

```
count            = 该 entity 在 train 中关联的交易数 (log1p 后 z-score)
mean_amt         = 该 entity 关联交易金额均值 (log1p 后 z-score)
std_amt          = 同上的标准差 (log1p 后 z-score)
fraud_rate_train = 该 entity 在 train 中欺诈占比 (clip [0,1])
days_active      = 第一次到最后一次出现的天数跨度 (z-score)
```

**冷启动 entity**(未在 train 出现但在 val/test 出现):用 train 全局均值兜底。**这不是欺诈泄漏来源**——entity 风险先验只用 train 计算,等价于上线时"用历史数据估计的实体风险"。

### 2.2 边 schema

| 关系 (src, rel, dst) | 反向 | 数量级 | 构造规则 |
|---------|------|---------|---------|
| `(transaction, paid_with, card1)` | + `rev_paid_with` | ~590K | 每条 txn → 它的 card1 节点 |
| `(transaction, shipped_to, addr1)` | + 反向 | ~590K | 同上,NaN 跳过 |
| `(transaction, sent_to_email, P_emaildomain)` | + 反向 | ~590K | 同上 |
| `(transaction, on_device, DeviceInfo)` | + 反向 | ~590K | NaN → "_NA_" 节点 |
| `(transaction, next_by_uid, transaction)` | 单向(时间) | ~580K | 同 (card1, addr1) 复合键的相邻 txn,按 TransactionDT 升序 i→i+1 |

**`next_by_uid` time-respecting 性质**:边只从早 → 晚,与 Stage 2 同构图保持一致,防时序泄漏。

**边特征**:本 Stage 不加 edge_attr(YAGNI)。

### 2.3 数据流时间线(防泄漏关键)

```
Step 1: 加载 Stage 2 已有 train.csv(全量)
Step 2: 按 TransactionDT 切分 train/val/test (60/20/20,与 Stage 2 一致)
Step 3: ⚠️ 仅在 train 子集上计算 entity 聚合统计
        → data/processed/pruned_v/entity_stats.json
Step 4: 把 entity_stats 应用到全集:
        - train entity → 用其自身统计
        - val/test 冷启动 entity → 用 train 全局均值兜底
        → data/processed/pruned_v/entity_features_{type}.pt
Step 5: 构造 5 类节点 + 9 条 edge_index → HeteroData
        → data/processed/pruned_v/hetero_graph.pt
```

### 2.4 与 Stage 2 同构图对比

| 维度 | Stage 2 同构 | Stage 3a 异质 |
|------|------------|--------------|
| 节点类型数 | 1 | 5 |
| 边类型数 | 1 | 5 关系 / 9 edge_index |
| 实体角色 | 拼成 uid 用作边构造 | **独立节点,拥有特征** |
| 实体先验信息 | 隐含在 cat_idx 嵌入 | 显式注入(5 维聚合 stat) |
| 消息传递路径 | txn ↔ txn(同 uid) | txn ↔ entity ↔ txn(跨 uid 团伙) |
| 团伙识别可行性 | 仅能定位 transaction 簇 | **能直接定位 entity 团伙核心** |

### 2.5 数据层文件清单

```
src/data/
├── build.py             修改:加 build_hetero_graph(...);build_all() 调用并落盘
├── entity_stats.py      新增:compute_entity_stats(train_df, entity_col) → DataFrame
│                              compute_all_entity_features(train_df, val_df, test_df) → dict
└── graph.py             保留:同构图函数不动,作 ablation

src/dataset.py           修改:make_loader 增加 graph_type 参数 ("homo" | "hetero")

data/processed/pruned_v/
├── graph.pt              已有
├── hetero_graph.pt       新增
├── entity_stats.json     新增
└── entity_features_{card1,addr1,P_emaildomain,DeviceInfo}.pt  新增
```

### 2.6 数据层测试(`tests/test_data.py` 新增 5 个)

- `test_entity_stats_train_only`:断言 entity_stats 计算时未触碰 val/test
- `test_hetero_graph_node_counts`:断言 5 类节点数 == unique 数 + 1(冷启动占位)
- `test_hetero_graph_edge_directions`:断言每对正反向边数量相等
- `test_next_by_uid_time_respecting`:断言 next_by_uid 边的 dst.time > src.time
- `test_cold_start_entity_fallback`:val/test 中新 entity 用 train 均值兜底,非 NaN

### 2.7 风险与降级

| 风险 | 缓解 |
|------|------|
| `hetero_graph.pt` 太大(估 ~500MB) | 边用 int32,节点特征 float32 |
| PyG NeighborLoader hetero 模式较慢 | 预热基准测试,如太慢退回 full-batch |
| 实体节点中心度极不均(card1 头部 1% 占边 30%+) | SAGEConv 用 mean aggr 而非 sum |
| Stage 2 同构图丢了不复用 | hetero_graph.pt 与 graph.pt 共存 |

---

## 3. 模型层(HeteroGraphTower + EmbeddingMixer 协同)

### 3.1 模块边界

```
FraudModel (orchestrator,接口不变)
  ├─ EmbeddingMixer  ← 共享:transaction 节点 cat+num,同时给 sequence_tower 用
  ├─ SequenceTower   ← Stage 2 已有,不动
  ├─ GraphTower      ← Stage 2 同构,保留作 ablation
  ├─ HeteroGraphTower ← 新增,本节核心
  │   └─ EntityProjector (per-type Linear 把 5 维 → d_graph)
  └─ FusionHead      ← Stage 2 已有,不动
```

`HeteroGraphTower` 与同构 `GraphTower` 是兄弟模块,FraudModel 通过 `graph_backbone` 配置项二选一,序列塔/融合头/EmbeddingMixer **一行不改**。

### 3.2 HeteroGraphTower 实现

```python
# src/models/hetero_graph_tower.py(新增)

class EntityProjector(nn.Module):
    """每个实体类型一个 Linear,把 5 维统计 → d_graph 维"""
    def __init__(self, entity_types, in_dim=5, d_graph=64):
        super().__init__()
        self.proj = nn.ModuleDict({
            t: nn.Linear(in_dim, d_graph) for t in entity_types
        })
    def forward(self, x_dict):
        return {t: self.proj[t](x) for t, x in x_dict.items() if t != 'transaction'}


class HeteroGraphTower(nn.Module):
    def __init__(self, mixer_out_dim, d_graph=64, n_layers=2,
                 entity_types=('card1','addr1','P_emaildomain','DeviceInfo'),
                 dropout=0.2):
        super().__init__()
        self.entity_proj = EntityProjector(entity_types, 5, d_graph)
        self.txn_proj = nn.Linear(mixer_out_dim, d_graph)

        # 5 关系 × 双向(next_by_uid 单向)= 9 条 SAGEConv per layer
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv({
                ('transaction','paid_with','card1'):       SAGEConv(d_graph, d_graph),
                ('card1','rev_paid_with','transaction'):   SAGEConv(d_graph, d_graph),
                ('transaction','shipped_to','addr1'):      SAGEConv(d_graph, d_graph),
                ('addr1','rev_shipped_to','transaction'):  SAGEConv(d_graph, d_graph),
                ('transaction','sent_to_email','P_emaildomain'): SAGEConv(d_graph, d_graph),
                ('P_emaildomain','rev_sent_to_email','transaction'): SAGEConv(d_graph, d_graph),
                ('transaction','on_device','DeviceInfo'):  SAGEConv(d_graph, d_graph),
                ('DeviceInfo','rev_on_device','transaction'): SAGEConv(d_graph, d_graph),
                ('transaction','next_by_uid','transaction'): SAGEConv(d_graph, d_graph),
            }, aggr='mean')   # ← mean 应对长尾度数
            self.convs.append(conv)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hetero_data, txn_mixed_emb, seed_local):
        x_dict = self.entity_proj(hetero_data.x_dict)
        x_dict['transaction'] = self.txn_proj(txn_mixed_emb)
        for conv in self.convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {t: self.dropout(self.act(x)) for t, x in x_dict.items()}
        return x_dict['transaction'][seed_local]   # [B, d_graph]
```

### 3.3 设计选择决策表

| 选择 | 原因 | 替代为何放弃 |
|------|------|------------|
| `aggr='mean'` 而非 `'sum'` | card1 头部 1% 占边 30%+;sum 会让头部主导,mean 给冷门节点公平权重 | sum 是 GraphSAGE 原文默认,但本数据集长尾严重 |
| 每边类型独立 SAGEConv | HeteroConv 标配;不同关系语义不同(刷卡 vs 同设备) | 共享权重会丢失关系语义 |
| `n_layers=2` | 2 跳足以覆盖 txn→entity→txn 团伙信号传递 | 3 层过平滑,训练慢 30%+ |
| 实体特征 Linear 投影非 Embedding | 实体特征是连续聚合统计而非 ID | ID 嵌入会丢失"该 entity 历史欺诈率"先验 |
| transaction 节点共享 EmbeddingMixer | 两塔看同一交易,特征空间一致融合才有意义 | 各自独立 mixer 会让两塔学到不一致表征 |

### 3.4 与 EmbeddingMixer 协同(零侵入修改)

```python
# Stage 3a FraudModel 增量(src/models/fraud_model.py)
class FraudModel(nn.Module):
    def __init__(self, cat_cardinalities, n_num_total, model_cfg,
                 fusion_mode, graph_backbone='homo'):  # ← 新增参数
        self.mixer = EmbeddingMixer(cat_cardinalities, ...)
        self.seq_tower = SequenceTower(self.mixer, ...)
        if graph_backbone == 'homo':
            self.graph_tower = GraphTower(self.mixer, ...)
        elif graph_backbone == 'hetero':
            self.graph_tower = HeteroGraphTower(
                mixer_out_dim=self.mixer.out_dim, ...)
        self.fusion = FusionHead(fusion_mode, ...)

    def forward(self, batch):
        txn_mixed = self.mixer(batch['x_cat'], batch['x_num'])
        seq_emb = self.seq_tower(batch['seq_cat'], batch['seq_num'], batch['mask'])
        if self.graph_backbone == 'homo':
            graph_emb = self.graph_tower(txn_mixed, batch['edge_index'], batch['seed_local'])
        else:
            graph_emb = self.graph_tower(batch['hetero_data'], txn_mixed, batch['seed_local'])
        return self.fusion(seq_emb, graph_emb)
```

### 3.5 dataloader 改动

```python
# src/dataset.py 加 hetero 分支
def make_loader(graph_or_hetero, seq_all, indices, batch_size,
                graph_type='homo', neighbor_sample=10):
    if graph_type == 'homo':
        return NeighborLoader(graph_or_hetero, num_neighbors=[neighbor_sample]*2,
                              input_nodes=indices, batch_size=batch_size, ...)
    else:
        return NeighborLoader(
            hetero_data,
            num_neighbors={
                ('transaction','paid_with','card1'): [neighbor_sample]*2,
                ('card1','rev_paid_with','transaction'): [neighbor_sample]*2,
                # ... 9 条边各自指定
            },
            input_nodes=('transaction', indices),
            batch_size=batch_size,
        )
```

batch 字典:`'edge_index'` → `'hetero_data'`(整个 HeteroData 对象塞进 batch),其余 7 个键不变。

### 3.6 参数量 & 显存估算

| 组件 | Stage 2 (homo) | Stage 3a (hetero) | Δ |
|------|---------------|------------------|---|
| EmbeddingMixer | ~430K | ~430K | 0 |
| SequenceTower | ~1.2M | ~1.2M | 0 |
| GraphTower / HeteroGraphTower | ~16K | ~150K | +134K |
| FusionHead (gated) | ~10K | ~10K | 0 |
| **总参数** | **~1.66M** | **~1.79M** | **+8%** |

显存(24GB GPU,batch=256):Stage 2 ~7GB → Stage 3a ~10-12GB peak。余量充足。

### 3.7 模型层测试(`tests/test_models.py` 新增 4 个)

- `test_entity_projector_per_type_independent`:断言两类型 Linear 权重独立
- `test_hetero_graph_tower_forward_shape`:小 HeteroData (10 txn + 5 card1 + 3 addr1) → [B, d_graph]
- `test_hetero_graph_tower_seed_extraction`:断言只取 seed_local 对应 transaction 输出
- `test_fraud_model_backbone_switch`:`graph_backbone='homo' / 'hetero'` 各 forward 一次,断言 logits shape 相同且参数量 hetero > homo

### 3.8 实现风险

| 风险 | 缓解 |
|------|------|
| HeteroConv 在 PyG 2.6.1 API 与文档差异 | Task 实现前先跑 PyG 官方 hetero 教程最小例子 |
| 冷启动节点(均值兜底)在 BatchNorm 引发 NaN | HeteroGraphTower 不用 BatchNorm,只 Dropout + ReLU |
| `aggr='mean'` 在 SAGEConv 是否需显式传入 | 显式写出,避免版本切换坑 |
| txn_proj 把 mixer_out_dim 投到 d_graph 损失信息 | mixer ≈ 96, d_graph=64 损失可控,融合头会拼 seq_emb 补充 |

---

## 4. 训练与评估(4 配置矩阵 + 收敛保证 + 团伙识别)

### 4.1 实验矩阵(Route A · 4 配置)

| 配置名 | 图骨干 | 损失 | HNM | 目的 |
|-------|-------|------|-----|------|
| `hetero_baseline` | hetero | HybridFocal (Stage 2 同) γ_pos=2, γ_neg=4, α=0.5 | 关 | **图升级单变量**:与 Stage 2 deep_pruned 直接对比 |
| `hetero_asym_balanced` | hetero | **更强非对称** γ_pos=2, γ_neg=6, α=0.4 | 关 | **损失加深**:让模型对正样本误判损失放大 |
| `hetero_label_smoothing` | hetero | HybridFocal + **label smoothing ε=0.1** | 关 | **过拟合抑制**:Stage 2 deep_full 过拟合迹象是否能修 |
| `hetero_HNM_root_cause` | hetero | HybridFocal (基础) | **开 + 诊断 hook** | **HNM 根因诊断**:Stage 1 害死 gated_plus_hnm,诊断到底丢了什么 |

**对照基准**(直接复用 Stage 2 真实 PR-AUC):

- Stage 2 `deep_pruned` (homo + HybridFocal, gated): PR-AUC = **0.4312** ← 同 v_strategy + 同模型族,直接对照
- Stage 2 `deep_full` (best Stage 2 deep): PR-AUC = **0.4370** ← Stage 2 最佳深度模型
- Stage 2 `lgbm_pruned`: PR-AUC = **0.5303** ← 传统模型 pruned 对照
- Stage 2 `lgbm_full` (best Stage 2 LGB): PR-AUC = **0.5556** ← 传统模型上限

(参考:这些配置的 ROC-AUC 分别是 0.8639 / 0.8621 / 0.8980 / 0.9016——ROC-AUC 看上去差距小,但 PR-AUC 才是不平衡欺诈检测的真指标。)

### 4.2 收敛保证机制(贯穿 4 配置,硬要求)

#### 4.2.1 训练预算放宽

| 参数 | Stage 2 | Stage 3a | 理由 |
|------|---------|---------|------|
| `epochs` | 20 | **40** | 给足缓冲防 Stage 1 早停误杀 |
| `early_stop_patience` | 4 | **8** | val PR-AUC 在欺诈检测中震荡常见 |
| `min_epochs` | (无) | **10** | 新增:前 10 epoch 不允许早停 |
| `lr_warmup_steps` | 200 | **500** | 异质图初始化更不稳 |

#### 4.2.2 每 epoch 全量指标记录

```python
# src/train.py 新增
training_history.append({
    'epoch': e, 'lr': current_lr,
    'train_loss': avg_train_loss,
    'val_roc_auc': val_roc_auc,
    'val_pr_auc':  val_pr_auc,
    'val_ks':      val_ks,
    'val_recall_at_fpr_0.01': val_recall_at_fpr,
    'epoch_seconds': epoch_time,
})
Path(f'experiments/training_history_{config_name}.json').write_text(
    json.dumps(training_history, indent=2))
```

#### 4.2.3 训练曲线图自动生成

```python
# src/analysis/plot_curves.py(新增)
def plot_curves(history_json_path, out_png):
    """3 子图:loss / pr_auc+roc_auc / lr,best_epoch 红色竖线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
```

每个配置训完自动生成 `experiments/curves_{config_name}.png`,共 4 张。

#### 4.2.4 收敛断言 + 显式审查

```python
def _convergence_audit(history, config_name):
    best_epoch = max(history, key=lambda h: h['val_pr_auc'])['epoch']
    total_epochs = len(history)
    last5 = [h['val_pr_auc'] for h in history[-5:]]

    print(f"[CONVERGENCE AUDIT · {config_name}]")
    print(f"  best_epoch = {best_epoch} / total_epochs_run = {total_epochs}")
    print(f"  last 5 epochs val_pr_auc: {last5}")

    warnings = []
    if best_epoch == total_epochs:
        warnings.append("⚠️  best_epoch == 末尾 epoch:可能仍在提升,需扩大 epochs")
    if total_epochs < 15:
        warnings.append(f"⚠️  仅训练 {total_epochs} epochs (<15),可能早停过早")
    if max(last5) - min(last5) > 0.02:
        warnings.append(f"⚠️  末 5 epoch val_pr_auc 震荡 > 0.02,未收敛")

    return {'best_epoch': best_epoch, 'total_epochs': total_epochs,
            'last5_pr_auc': last5, 'warnings': warnings}
```

**warnings 非空 → `converged=False` 标记落 stage3a_results.json + README 红字披露**。

#### 4.2.5 配置间内存隔离

避免 Stage 2 SIGKILL 复现:

```python
for config in configs:
    metrics = train_one_config(config, ...)
    save_results(metrics)
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
```

外加每个 config 单独 `python -m` 子进程跑 fallback。

### 4.3 评估指标

| 指标 | 计算方式 | 关注点 |
|------|---------|-------|
| ROC-AUC | sklearn | 整体排序能力 |
| **PR-AUC** | sklearn | **主选指标**(不平衡场景金标准) |
| KS | max(TPR-FPR) | 业界风控通用 |
| Recall@FPR=0.01 | 误报 1% 时召回 | 上线决策点 |
| **fraud_subgraph_modularity** | networkx 在欺诈预测子图上算 | 量化"团伙是否被聚成簇" |

### 4.4 团伙识别后处理

```python
# src/analysis/centrality.py(新增)
def identify_fraud_rings(model_ckpt, hetero_graph, val_indices, top_k=20):
    """
    Step 1: 加载训好的 hetero 模型 → 对 val 集打分
    Step 2: 取 prob > 0.9 的高置信欺诈交易
    Step 3: 提取这些交易及其连接的所有 entity → fraud_subgraph
    Step 4: networkx 异质图,每类 entity 节点算:
              - degree centrality
              - PageRank
    Step 5: 输出 top_k entity per type + 该 entity 的 train fraud_rate 对照
    Step 6: 落 experiments/core_entities_{config}.json + 简短 markdown
    """
```

**输出样例(数字到时填实测)**:

```json
{
  "card1": [
    {"id": "card1_1234", "degree": 47, "pagerank": 0.0083,
     "train_fraud_rate": 0.92, "n_train_txns": 156}
  ]
}
```

这是简历"异常团伙核心节点识别"的可交付物——基于异质图自然产出的后分析。

### 4.5 时间预算

| 配置 | 单 epoch | 40 上限 | 8 patience 预期 |
|------|---------|---------|----------------|
| hetero_baseline | ~70s | ~47 min | ~25 min |
| hetero_asym_balanced | ~70s | ~47 min | ~25 min |
| hetero_label_smoothing | ~70s | ~47 min | ~25 min |
| hetero_HNM_root_cause | ~85s | ~57 min | ~30 min |

**4 配置总训练**:1.7 ~ 3 小时 + centrality ~10 min + 曲线 ~2 min。AutoDL 单会话可完成。

### 4.6 训练评估文件清单

```
src/
├── train.py             修改:+ _record_epoch_metrics, _convergence_audit, run_stage3a_matrix
├── losses.py            修改:+ label_smoothing 选项, HNM 诊断 hook
└── analysis/            新增目录
    ├── plot_curves.py        新增
    └── centrality.py         新增

experiments/
├── stage3a_results.json                新增(4 配置主指标 + converged 标记)
├── training_history_{config}.json      新增(4 个)
├── curves_{config}.png                 新增(4 张)
└── core_entities_{config}.json         新增
```

### 4.7 训练评估测试

`tests/test_train.py` 新增 3 个:

- `test_convergence_audit_warns_on_late_best`:best_epoch == 末尾 → warning 非空
- `test_convergence_audit_warns_on_oscillation`:末 5 epoch pr_auc 波动 > 0.02 → warning 非空
- `test_label_smoothing_loss_value`:ε=0.1 与 ε=0 的 loss 差异在预期区间

`tests/test_analysis.py`(新增文件)2 个:

- `test_plot_curves_generates_png`:假 history 跑一次,断言 PNG > 5KB
- `test_centrality_topk_count`:小 hetero 子图跑一次,断言每类 entity 输出 ≤ top_k

### 4.8 失败处理决策树

```
训练结束 → _convergence_audit
  ├─ converged=True → 接受,记录最佳
  └─ converged=False
      ├─ 原因 1: best_epoch == 末尾
      │    → 标记 NEEDS_LONGER,README 红字披露(不自动重训)
      ├─ 原因 2: 仅 < 15 epoch 早停
      │    → 检查 patience 是否被 NaN/异常 loss 触发
      │    → 是 → 排查损失实现,修复后重训
      │    → 否 → 接受(快速收敛后退化,记录原因)
      └─ 原因 3: 末 5 epoch 震荡 > 0.02
           → 标记 UNSTABLE_TRAINING,README 透明披露
```

**核心原则**:任何 converged=False 的配置,README 和 DESIGN_JOURNAL 都必须显式说明,不可静默接受当成"最优结果"。

---

## 5. 项目结构 + DESIGN_JOURNAL v3 + Definition of Done

### 5.1 项目结构 Δ

```
alibaba-risk-control-internship/
├── docs/
│   ├── superpowers/
│   │   ├── specs/
│   │   │   ├── 2026-05-14-ant-riskcontrol-stage1-design.md  (已有)
│   │   │   ├── 2026-05-15-ant-riskcontrol-stage2-design.md  (已有)
│   │   │   └── 2026-05-15-ant-riskcontrol-stage3a-design.md ← 本文档
│   │   └── plans/
│   │       ├── 2026-05-14-ant-riskcontrol-stage1.md         (已有)
│   │       ├── 2026-05-15-ant-riskcontrol-stage2.md         (已有)
│   │       └── 2026-05-15-ant-riskcontrol-stage3a.md        ← writing-plans 产出
│   └── DESIGN_JOURNAL.md                                     ← v1+v2 保留 + v3 追加
│
├── src/
│   ├── data/
│   │   ├── build.py                修改:+ build_hetero_graph()
│   │   ├── entity_stats.py         新增
│   │   ├── features.py             不动
│   │   ├── graph.py                不动(同构保留)
│   │   └── v_pruning.py            不动
│   ├── models/
│   │   ├── embedding_mixer.py      不动
│   │   ├── sequence_tower.py       不动
│   │   ├── graph_tower.py          不动(同构保留)
│   │   ├── hetero_graph_tower.py   新增
│   │   ├── fusion.py               不动
│   │   ├── losses.py               修改:+ label_smoothing, HNM 诊断 hook
│   │   └── fraud_model.py          修改:+ graph_backbone 参数 + 分支
│   ├── analysis/                   新增目录
│   │   ├── __init__.py
│   │   ├── plot_curves.py          新增
│   │   └── centrality.py           新增
│   ├── deploy/                     不动
│   ├── dataset.py                  修改:+ hetero NeighborLoader 分支
│   ├── train.py                    修改:+ run_stage3a_matrix + 收敛审计
│   └── baseline_lgbm.py            不动
│
├── tests/
│   ├── test_data.py                +5 测试
│   ├── test_models.py              +4 测试
│   ├── test_train.py               +3 测试
│   └── test_analysis.py            新增,2 测试
│
├── data/processed/pruned_v/
│   ├── graph.pt                    已有
│   ├── hetero_graph.pt             新增
│   ├── entity_stats.json           新增
│   └── entity_features_*.pt        新增 (4 个文件)
│
├── experiments/
│   ├── stage3a_results.json        新增
│   ├── training_history_*.json     新增 (4 个)
│   ├── curves_*.png                新增 (4 张)
│   └── core_entities_*.json        新增
│
├── configs/
│   ├── data.yaml                   修改:+ hetero_graph 开关, entity_feat_dim
│   ├── model.yaml                  修改:+ graph_backbone, hetero d_graph
│   └── train.yaml                  修改:epochs 40, patience 8, min_epochs 10, warmup 500
│
├── artifacts/
│   ├── best_deep_pruned.pt         Stage 2 已有
│   └── best_hetero_*.pt            新增 (4 个 checkpoint)
│
└── README.md                       修改:+ Stage 3a 结果章节
```

### 5.2 DESIGN_JOURNAL v3 增量内容(append-only,v1+v2 byte-for-byte 保留)

```markdown
---

# v3 — Stage 3a: 异质图建模 + 损失深化 (2026-05-15)

## 设计初衷
[Section 0 完整内容]

## 文献支撑
1. **HeteroGNN for Fraud Detection** — Liu et al., "Heterogeneous Graph Neural
   Networks for Malicious Account Detection" (CIKM 2018, 阿里风控团队)
2. **GraphSAGE** — Hamilton et al., NeurIPS 2017
3. **Time-Respecting Edges** — Paranjape et al., WSDM 2017
4. **Label Smoothing for Imbalanced** — Müller et al., NeurIPS 2019
5. **PageRank for Fraud Ring Detection** — Pandit et al., "NetProbe" (WWW 2007)

## 原理详解
[本设计文档 1-4 节完整复用]

## 实现细节
[由 writing-plans 产出 Stage 3a plan 落地后回填]

## 真实结果(填实测后)
| 配置 | val_pr_auc | val_roc_auc | val_ks | converged | best_epoch / total |
|------|-----------|-------------|--------|-----------|------------------|
| Stage 2 deep_pruned (homo, 对照) | 0.4312 | 0.8639 | 0.5637 | ✅ | (Stage 2 已有) |
| Stage 2 deep_full (homo, best Stage 2 deep) | 0.4370 | 0.8621 | 0.5731 | ✅ | (Stage 2 已有) |
| hetero_baseline | TBD | TBD | TBD | TBD | TBD |
| hetero_asym_balanced | TBD | TBD | TBD | TBD | TBD |
| hetero_label_smoothing | TBD | TBD | TBD | TBD | TBD |
| hetero_HNM_root_cause | TBD | TBD | TBD | TBD | TBD |
| Stage 2 lgbm_pruned (对照) | 0.5303 | 0.8980 | 0.6416 | ✅ | (Stage 2 已有) |
| Stage 2 lgbm_full (best Stage 2 LGB) | 0.5556 | 0.9016 | 0.6475 | ✅ | (Stage 2 已有) |

## 诚实四情景结论(实测后选其一)
- [ ] hetero best PR-AUC > 0.5556 (best LGB lgbm_full):深度模型反超传统模型 ✅
- [ ] hetero best PR-AUC ∈ (0.4370, 0.5556):异质图有效但未超 LGB
- [ ] hetero best PR-AUC ≈ 0.4370 (best Stage 2 deep deep_full):异质图帮助有限
- [ ] hetero best PR-AUC < 0.4370:实现需排查

## 团伙识别
详见 experiments/core_entities_<best_config>.json
top-20 高 PageRank entity + train_fraud_rate 对照,可解释性证据。

## 简历映射
- "行为序列与异质图建模" → SeqTower (S1) + HeteroGraphTower (S3a) + 团伙识别 (S3a)
- "极度不平衡样本处理" → HybridFocal (S1) + 4 损失变体 ablation (S3a)
- "性能优化" → ONNX/TensorRT (S1, S2 验证;hetero 部署留 Stage 3+)
```

### 5.3 README.md Δ(在 Stage 2 章节之后新增)

```markdown
## Stage 3a Results: Heterogeneous Graph + Loss Deepening (2026-05-15)

### Experiment Matrix
[4 配置结果表 + Stage 2 对照行]

### Convergence Audit Summary
[4 配置 best_epoch / total_epochs / converged 状态表]

### Training Curves
[嵌入 4 张 curves_*.png]

### Fraud Ring Identification
[top-10 high-centrality entity 摘要,完整列表见 JSON]

### Resume Bullet Mapping (Stage 3a 增量)
[更新映射表]

### Honest Negative Results / Caveats
[列出所有 converged=False 配置,显式说明原因]
```

### 5.4 Definition of Done

**Hard gates(任一不过 → Stage 3a 不算完成)**:

- [ ] 5 类节点 + 9 条 edge_index 的 HeteroData 成功构造,通过 5 个数据测试
- [ ] HeteroGraphTower forward 在小图和真实图上都能跑通,通过 4 个模型测试
- [ ] 4 配置全部跑完(若有 converged=False,有显式标记记录)
- [ ] 4 张曲线图全部生成,best_epoch 红线清晰可见
- [ ] `experiments/stage3a_results.json` 存在,4 配置 + converged 字段
- [ ] 4 个 `training_history_*.json` 全部存在,每文件含完整 per-epoch 指标
- [ ] 团伙识别后处理跑通,`core_entities_*.json` 存在
- [ ] DESIGN_JOURNAL v3 写完(v1+v2 byte-for-byte 保留),含真实结果填实
- [ ] README.md 新增 Stage 3a 章节,**包含诚实四情景结论(实测选其一)**
- [ ] 全部测试通过:Stage 1+2 已有 37 + Stage 3a 新增 14 = **51 个测试 100% 通过**
- [ ] git 提交链清晰,每个 task 一个 atomic commit,推 GitHub feature 分支

**Soft gates(达到加分,不达不阻塞)**:

- [ ] hetero best PR-AUC ≥ Stage 2 deep_pruned 0.4312(图升级有效的最低门槛)
- [ ] 至少 1 个配置 converged=True 且 PR-AUC ≥ 0.40(达到 Stage 2 deep 同水位)
- [ ] core_entities top-10 中 train_fraud_rate ≥ 0.5 的 entity 占比 ≥ 50%

**绝不放进 DoD 的项**(诚实原则):

- ❌ "PR-AUC 必须 > 0.5556"(不能为赢 LGBM 而调超参)
- ❌ "AUC 必须 ≥ 0.98"(简历的 0.98 与 Ant 私有数据绑定)
- ❌ "loss 必须降低 8%"(同上,不可比)

### 5.5 Stage 3a 显式不做(YAGNI 边界)

- ❌ 异质图 ONNX/TensorRT 部署 → Stage 3b 工具链修复后再做
- ❌ Edge attribute(交易金额作边权)→ Stage 3+
- ❌ Heterogeneous Attention(HAN/HGT)→ Stage 3+
- ❌ 实体节点 dynamic embedding → 工业级才需要,本 stage 静态够用
- ❌ 多任务学习 → scope creep
- ❌ PMML 导出 LGBM → Stage 3b

### 5.6 Stage 3a 与后续 Stage 关系

```
Stage 3a (本 stage)
  ├─ 异质图 + 团伙识别  → 简历"异质图建模"完整落地
  ├─ 4 损失 ablation   → 简历"极度不平衡处理"深化
  └─ 收敛保证机制       → 全局质量保证

Stage 3b (后续,工具链)
  ├─ cuDNN/onnxruntime-gpu ABI 修复
  ├─ Java 11+ + sklearn2pmml 0.130 跑通
  └─ 异质图 ONNX 导出

Stage 3c (后续,推理优化)
  ├─ Triton inference server
  ├─ INT8 量化
  └─ 自定义算子(HeteroConv 融合)
```

---

## 文献参考完整列表

1. Liu, Z., et al. "Heterogeneous Graph Neural Networks for Malicious Account
   Detection." CIKM 2018. (阿里风控团队工作,异质图欺诈检测奠基)
2. Hamilton, W., Ying, Z., Leskovec, J. "Inductive Representation Learning on
   Large Graphs." NeurIPS 2017. (GraphSAGE 原文)
3. Paranjape, A., Benson, A.R., Leskovec, J. "Motifs in Temporal Networks."
   WSDM 2017. (time-respecting edges)
4. Müller, R., Kornblith, S., Hinton, G. "When Does Label Smoothing Help?"
   NeurIPS 2019.
5. Pandit, S., et al. "NetProbe: A Fast and Scalable System for Fraud Detection
   in Online Auction Networks." WWW 2007. (PageRank for fraud ring)
6. Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
   (Stage 1 已引,继续用于 Stage 3a baseline)
7. Shrivastava, A., Gupta, A., Girshick, R. "Training Region-based Object
   Detectors with Online Hard Example Mining." CVPR 2016.
   (Stage 1 已引,Stage 3a 用于 HNM 诊断配置)

---

**End of Stage 3a Design Document.**
