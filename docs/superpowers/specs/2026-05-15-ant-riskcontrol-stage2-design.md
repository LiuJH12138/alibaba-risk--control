# 阿里/蚂蚁风控算法组实习 —— 复刻项目设计文档(Stage 2)

- **日期**:2026-05-15
- **状态**:设计已确认,待写实现计划
- **范围**:专注模型基础升级 —— 类别 embedding + 完整 V 列;异质图、损失深化推到 Stage 3+
- **核心问题**:升级模型基础后,深度双塔模型能否与 LightGBM 0.908 持平或超过?

---

## 0. 背景与定位

本项目第二阶段。Stage 1 已完成并合并到 master、推送至 GitHub(`github.com/LiuJH12138/alibaba-risk--control`),完整记录见 Stage 1 spec 与 `docs/DESIGN_JOURNAL.md` v1。

**Stage 2 在三阶段路线中的位置**(沿用 Stage 1 spec §1 拆分):
- Stage 1 ✅ 一体化端到端 MVP(已交付)
- **Stage 2 ⏳ 深化建模 —— 模型基础升级**(本文档)
- Stage 3 生产化可部署系统(异质图、损失深化、cuDNN/PMML 工具链等也并入此阶段)

### 诚实原则(继承 Stage 1,本 Stage 强化)

简历的 "AUC 0.98 / 资损 -8%" 绑定蚂蚁专有数据,不复现。Stage 2 的所有数字都是 IEEE-CIS 公开数据上的真实结果,**包括不理想的结果**。Stage 2 的"成功"=完成该做的工程改动 + 诚实测量,**不为凑赢 LightGBM 调超参**。详见 §3 的四情景成功框架。

---

## 1. Stage 1 → Stage 2 的认知转变(为什么这是 Stage 2)

Stage 1 实测的关键发现重塑了 Stage 2 的优先级:

- 深度双塔模型(roc_auc 0.82–0.85)**输给了 LightGBM 基线**(roc_auc 0.908)
- DESIGN_JOURNAL v1 已识别根因:
  1. 类别字段用「缩放序数编码」(NaN 修复的副产物),而非真正的 embedding —— 模型看不到类别间的语义距离
  2. V 列削减到 V1-V50(50GB 磁盘约束)—— 丢弃了大量 Vesta 工程特征
- 而**图本身没问题**(graph_only 0.848 ≈ seq_only 0.844)

文献支撑:Shwartz-Ziv & Armon 2022 *Tabular Data: Deep Learning is Not All You Need* —— GBDT 在中等规模表格上常胜过深度模型,深度方法需要正确的归纳偏置(per-field embedding、丰富特征)才能竞争。

**外部条件变更**:用户已将 autodl 数据盘扩容到 150GB(原 50GB),完整 V 列特征集(seq_all.pt ~60GB)放得下。

### Stage 2 的核心问题
> **升级类别 embedding + 完整 V 列后,深度双塔模型能否与 LightGBM 0.908 持平或超过?**
> 无论结果如何,Stage 2 都给出对 Stage 3 方向的可信证据。

---

## 2. 范围(明确边界)

### 2.1 范围内(Stage 2 要做的)
1. **类别 embedding 层** —— 替换 Stage 1 的缩放序数编码
2. **完整 V 列特征** —— 解锁 V1-V339,同时做相关性剪枝消融(全量 vs 剪枝)
3. **数据层接口质变** —— FeatureProcessor 输出从 flat 张量改为 `{cat_idx, num}` 字典
4. **模型层 mixer 前置层** —— `EmbeddingMixer` 把 dict 转回统一张量,两塔本身不动
5. **实验矩阵收窄** —— `gated_fusion × {full_v, pruned_v}` + LightGBM × 两套 = 4 次训练
6. **保 best checkpoint** —— Stage 1 漏项,Stage 3 部署需要
7. **诚实评估 + 四情景成功框架** —— 不为凑赢调参

### 2.2 范围外(明确推 Stage 3+,避免 scope creep)
- 异质图(多类型节点/边)+ 团伙核心节点识别
- 损失函数深化(HNM 反而有害的根因调查、γ/α 调优)
- cuDNN / onnxruntime-gpu ABI 修复
- 完整 PMML 工具链(Java 11+ 安装、numpy 2.x 兼容)
- TensorRT EP 端到端延迟测量(被 cuDNN 阻塞,Stage 3 一并)

### 2.3 成功定义(诚实四情景)

| 实测结果 | Stage 2 叙事 |
|---|---|
| Both deep ≥ LGB | Stage 1 输给 LGB 是数据简化导致;升级后深度反超 |
| deep_pruned ≥ LGB,deep_full < LGB | 完整 V 列引入冗余噪声,剪枝是赢家 |
| deep_full ≥ LGB,deep_pruned < LGB | 完整特征价值显著,剪枝丢了信号 |
| Both deep < LGB | GBDT 表格归纳偏置确实强;深度需异质图/外部特征(Stage 3) |

**四种情景都"成功"** —— 都给出对 Stage 3 方向的可信证据。

---

## 3. 架构总览(相对 Stage 1 的变更)

```
IEEE-CIS 原始 CSV
   │
   ├─[数据层 Δ] FeatureProcessor 改造(§4):
   │           cat → 整数索引 [N, n_cat] (int64)
   │           num → 标准化值 + isna [N, n_num*2] (float32, clip[-10,10] 保留)
   │           输出由「单 flat 张量」→「{cat, num} 字典」(接口质变)
   │
   ├─[数据层 Δ] V 列双轨产物(§4.2):
   │           data/processed/full_v/...    (V1-V339)
   │           data/processed/pruned_v/...  (相关性贪心剪枝 ~100-130 列)
   │
   ├─[模型层 Δ] FraudModel 持有共享 EmbeddingMixer(§5):
   │           cat 索引 → 每字段 nn.Embedding(card, 16) → 拼接
   │           num 直通 → 与 cat embedding 拼接 → 统一 [..., feat_dim_unified]
   │           SequenceTower / GraphTower / FusionHead **完全不变**
   │
   ├─[训练层 Δ] 实验矩阵 4 跑(§6):gated × {full,pruned} + LGB × {full,pruned}
   │           保 best checkpoint(Stage 1 漏项)
   │
   └─[部署层 Δ] ONNX 4 输入版本 + 真实 ckpt + 双模型 benchmark(§7)
               其余链路不变(cuDNN 问题继续诚实记录)
```

### Stage 1 修复继续生效(明确写出来,避免误删)
- SequenceTower 的 plain GRU(无 pack_padded_sequence)—— 保留
- FeatureProcessor 的 unknown 桶 = 0、num 裁剪 [-10,10] —— 保留
- train.py 的 `.cpu().numpy()`、LR warmup —— 保留
- build_trt.py 的 TRT 10.x API + try/except 降级 —— 保留
- run_benchmark 的逐配置 try/except —— 保留

---

## 4. 数据层变更

### 4.1 FeatureProcessor 接口质变

```
Stage 1: transform(df) -> pd.DataFrame   (flat 浮点矩阵,cat 缩放到 [0,1))
Stage 2: transform(df) -> dict
   {
     "cat_idx": int64 [N, n_cat]      # 0..card-1 整数索引(0 = unknown 桶)
     "num":     float32 [N, n_num*2]  # 标准化 [-10,10] + isna 拼接
   }
```
- 这是 Stage 2 工作量主要来源 —— 涟漪到 build.py / dataset.py / fraud_model.py
- `meta` 仍带 `cat_cardinalities`(Stage 1 已有),embedding 层用它 sizing
- Stage 1 的两条 NaN 防护(unknown=0、num clip)**保留**

### 4.2 V 列双轨产物

`build.py` 参数化 + 两次跑:
```python
def build_all(v_strategy: str):  # "full_v" or "pruned_v"
    if v_strategy == "full_v":
        v_cols = [f"V{i}" for i in range(1, 340)]
    else:  # pruned_v
        v_cols = compute_pruned_v_cols(df.iloc[train_idx])
    out = Path(cfg["processed_dir"]) / v_strategy
    # ... 落盘到 out

if __name__ == "__main__":
    for s in ["full_v", "pruned_v"]:
        build_all(s)
```
两套独立 `data/processed/{full_v,pruned_v}/`,各持 5 文件,互不干扰。

### 4.3 V 列相关性剪枝(`compute_pruned_v_cols`)

只在 train 数据上算(防泄漏):
1. `corr = df_train[v_cols].corr().abs()`
2. 贪心剔除:从 V1 到 V339,若与已保留列存在 `|corr| ≥ 0.95`,则丢弃;否则保留
3. 返回保留列名列表(预期 ~100-130 列),缓存到 `data/processed/v_pruned_cols.json`

### 4.4 磁盘与时间预算(已核算)

| | feat_dim 拆解 | seq_all.pt | graph.pt |
|---|---|---|---|
| full_v | 49 cat (int64) + 678 num (339×2 含 isna) | ~56 GB | ~1.7 GB |
| pruned_v | 49 cat + ~284 num (~110×2) | ~28 GB | ~0.9 GB |
| **合计** | | **~84 GB** | **~2.6 GB** |

134 GB 可用 → 装得下两套 + 余量给 ONNX/checkpoint。

### 4.5 测试改动(数据层)
| 测试 | 改动 |
|---|---|
| `test_processor_fits_on_train_only` | 改 dict 取数:`out["cat_idx"][1, 0] == 0`、`out["num"][:, 0].mean() ≈ 0` |
| `test_processor_meta_has_cardinalities` | 不变 |
| `test_processor_output_is_bounded` | 拆两条:cat ∈ [0, card)、num ∈ [-10, 10] |
| `test_join_/uid_/sequence_/edges_/split_*` | 不变(都不依赖 FP 输出格式) |
| 新增 `test_v_column_pruning` | 合成强相关列,断言每组只留 1 个代表 |

---

## 5. 模型层变更

### 5.1 新增 `EmbeddingMixer`(唯一新组件)

```python
class EmbeddingMixer(nn.Module):
    """把 {cat_idx, num} 字典转成统一 [..., feat_dim_unified] 张量。
    每类别字段一个 nn.Embedding;num 直通;最终拼接。形状无关:
    同时支持序列输入 [B,L,n_cat] 和图输入 [N,n_cat]。"""
    def __init__(self, cat_cardinalities: list[int], cat_emb_dim: int, n_num_total: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, cat_emb_dim) for c in cat_cardinalities])
        self.cat_emb_dim = cat_emb_dim
        self.out_dim = len(cat_cardinalities) * cat_emb_dim + n_num_total

    def forward(self, cat_idx, num):
        embs = [emb(cat_idx[..., i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embs, dim=-1)
        return torch.cat([cat_out, num], dim=-1)
```
- ~12 行;形状无关靠 `cat_idx[..., i]`
- 嵌入参数总量 ~250-300K(主 card1=12731×16≈200K,其余字段小)

### 5.2 `FraudModel` 持有 mixer,两塔签名不变

```python
class FraudModel(nn.Module):
    def __init__(self, cat_cardinalities, n_num_total, model_cfg, fusion_mode="gated"):
        c = model_cfg
        self.mixer = EmbeddingMixer(cat_cardinalities, c["cat_emb_dim"], n_num_total)
        feat_dim = self.mixer.out_dim
        self.seq_tower = SequenceTower(feat_dim=feat_dim, ...)   # 塔不动
        self.graph_tower = GraphTower(feat_dim=feat_dim, ...)
        self.fusion = FusionHead(...)

    def forward(self, seq_cat, seq_num, mask, x_cat, x_num, edge_index, seed_idx):
        seq = self.mixer(seq_cat, seq_num)
        x   = self.mixer(x_cat, x_num)
        return self.fusion(self.seq_tower(seq, mask), self.graph_tower(x, edge_index)[seed_idx])

    def forward_online(self, seq_cat, seq_num, mask, graph_emb):
        seq = self.mixer(seq_cat, seq_num)
        return self.fusion(self.seq_tower(seq, mask), graph_emb)
```

**核心设计选择**(写入 DESIGN_JOURNAL v2):
- **mixer 在两塔间共享** —— 序列塔和图塔看同一组 embedding;少参数、强一致
- **塔本身完全不变** —— 接口和实现都不改,Stage 1 的 plain GRU 修复保留
- **`feat_dim` 派生** —— FraudModel 接受 `cat_cardinalities + n_num_total`(从 feature_meta.json 读),自算 mixer.out_dim;configs/model.yaml 仅保留 `cat_emb_dim: 16`

文献支撑:Wide & Deep(Cheng 2016)、DeepFM(Guo 2017)、FT-Transformer(Gorishniy 2021)的 per-field embedding 模式。

### 5.3 测试改动(模型层)
| 测试 | 改动 |
|---|---|
| `test_sequence_tower_*` / `test_graph_tower_*` / `test_fusion_*` | 不变(塔接口未变) |
| `test_fraud_model_*` | 改造为新签名(传 cat/num 拆分输入),并验证 mixer 共享 |
| 新增 `test_embedding_mixer_output_shape` | 验证 [..., out_dim] 形状,2D 和 3D 输入都过 |
| 新增 `test_embedding_mixer_handles_unknown_index` | 索引 0 不崩、产出有限值 |
| `test_loader_yields_aligned_seq_and_seeds` | 改 dict batch:含 `seq_cat/seq_num/x_cat/x_num/mask/seed_local/label` |

---

## 6. 训练与评估

### 6.1 实验矩阵(共 4 次训练)
```
                  full_v        pruned_v
  gated_fusion    deep_full     deep_pruned    ← Stage 2 主结果
  LightGBM        lgbm_full     lgbm_pruned    ← 基线对照
```
- 不再跑 5 配置矩阵(Stage 1 已有完整对照)
- 损失参数(γ_pos=1, γ_neg=4, α=0.25, HNM 比例)沿用 Stage 1,**不调** —— 损失调优推 Stage 3+

### 6.2 实验产出
```
experiments/
  results.json           # Stage 1 已 commit,不动
  stage2_results.json    # 新:4 个 key(deep_full/deep_pruned/lgbm_full/lgbm_pruned)
                         # 每条 metrics + train_seconds + v_strategy 字段
artifacts/
  best_deep_full.pt      # Stage 2 新增:保存 best checkpoint
  best_deep_pruned.pt
  best_lgbm_full.pkl     # LGB 也存
  best_lgbm_pruned.pkl
```

### 6.3 训练循环新增 best checkpoint
`train_one_config` 加 `checkpoint_path` 参数;每次 PR-AUC 提升时保存:
```python
if metrics["pr_auc"] > best_pr:
    best_pr, best_metrics, patience = metrics["pr_auc"], metrics, 0
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
```
新增 `run_stage2_matrix(device="cuda")`:跑 4 个配置 + 各自指定 v_strategy 数据路径 + checkpoint 路径。

### 6.4 LightGBM 对照公平性
- 与深度模型同一份 train/val 切分
- 输入 = `{cat_idx, num}` 拍扁回单矩阵;cat 索引作为整数特征传给 LGB(`categorical_feature=` 参数)
- 这样深度模型和 LGB 看完全一致的特征 —— 公平

### 6.5 时间预算(估算)
- LGB × 2:每次 ~10s,共 < 1 分钟
- deep_full:feat_dim ~1462,seq_all 56GB,每 epoch 比 Stage 1 慢 2-3×,早停后 ~25-40 min
- deep_pruned:feat_dim ~1068,seq_all 28GB,每 epoch 慢 1.5×,早停后 ~15-25 min
- **总训练时间 ~50-70 min**(配置数减少抵消单次更慢)

### 6.6 评估指标
- 沿用 Stage 1 `compute_metrics`(ROC-AUC, PR-AUC, KS, recall@FPR0.01, fpr@recall0.90)
- 自动校验:`result is not None`、`0 ≤ roc_auc ≤ 1`、`isfinite(loss)`

### 6.7 测试改动(训练层)
- `test_train_one_config_runs_and_returns_metrics`:cat+num 拆分;断言 `checkpoint_path` 文件被写入
- `test_lgbm_baseline_runs`:扁平矩阵 + categorical_feature 索引

---

## 7. 部署与 Benchmark 变更

### 7.1 ONNX 导出:4 输入版本
`_OnlineWrapper.forward(seq_cat, seq_num, mask, graph_emb)` —— 比 Stage 1 多一个 `seq_cat`(int64,动态 batch 轴)。
- `nn.Embedding` → ONNX `Gather`(opset 17 原生)
- Parity 检查保持 hard gate(atol=1e-4)

### 7.2 TensorRT 引擎
- `build_engine` 不改(Stage 1 已修好 TRT 10.x API + try/except)
- 优化 profile 的输入名循环改 `["seq_cat","seq_num","mask","graph_emb"]`

### 7.3 真实 best checkpoint
```python
ckpt = Path(f"artifacts/best_deep_{v_strategy}.pt")
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, weights_only=True))
else:
    print(f"WARN: no checkpoint at {ckpt}, using random init for latency only")
```

### 7.4 双模型 benchmark
两个深度模型都测(各 4 档,共 8 个延迟数字):
```
benchmark_stage2.json:
  deep_full:    { pytorch_cpu, pytorch_gpu, onnx_gpu, tensorrt_fp16 }
  deep_pruned:  { pytorch_cpu, pytorch_gpu, onnx_gpu, tensorrt_fp16 }
```
`onnx_gpu` / `tensorrt_fp16` 在本环境仍 skip(沿用 Stage 1 cuDNN 文案)。`pytorch_cpu` vs `pytorch_gpu` 给真实 before/after。预期 deep_pruned 略快。

### 7.5 图 embedding 离线预计算(沿用 Stage 1 + mixer)
```python
@torch.no_grad()
def precompute_graph_emb(model, graph_data):
    x_unified = model.mixer(graph_data["cat"], graph_data["num"])
    return model.graph_tower(x_unified, graph_data["edge_index"])
```
存为 `artifacts/graph_emb_{v_strategy}.pt`(gitignored,可重算)。

### 7.6 测试改动(部署层)
- `test_onnx_export_and_parity`:mock model 走新 4 输入;atol=1e-4 不动
- `test_benchmark_torch_returns_latency_stats`:cat+num 拆分输入

### 7.7 部署产物清单
```
experiments/stage2_results.json       (commit)
experiments/benchmark_stage2.json     (commit)
artifacts/best_deep_{full,pruned}.pt  (commit, ~30-50MB)
artifacts/best_lgbm_{full,pruned}.pkl (commit)
artifacts/online_{full,pruned}.onnx   (commit)
artifacts/online_{full,pruned}.engine (gitignored, 硬件专属)
artifacts/graph_emb_{full,pruned}.pt  (gitignored, 可重算)
```

---

## 8. 工程结构

### 新增(2 文件)
```
src/models/embedding_mixer.py    # ~30 行
src/data/v_pruning.py            # ~25 行
```

### 修改(dict 接口涟漪)
```
src/data/features.py             # transform 返回 dict
src/data/build.py                # 参数化 v_strategy + 跑两次
src/dataset.py                   # make_loader 处理 dict batch
src/models/fraud_model.py        # 持 mixer + 新 forward 签名
src/train.py                     # 保 best checkpoint + run_stage2_matrix
src/baseline_lgbm.py             # 参数化 v_strategy + 扁平化辅助
src/deploy/export_onnx.py        # 4 输入 wrapper
src/deploy/benchmark.py          # 适配新输入 + 双模型 benchmark
```

### 配置不变
`configs/{data,model,train}.yaml` —— `cat_emb_dim: 16` 已在 Stage 1 model.yaml(当时 dead config,Stage 2 终于用上);`feat_dim` 派生,无需 yaml。

### 数据目录新结构
```
data/processed/
  full_v/{graph,seq_all,split}.pt + {feature_meta,manifest}.json
  pruned_v/{...}
  v_pruned_cols.json                # 缓存的剪枝后 V 列名列表
```

### 测试目录(适配 + 新增)
预计测试数 34 → ~40。

---

## 9. 错误处理与测试

- **数据层**:断言 `cat_idx.max() < cardinality`、num 在 [-10,10]、剪枝后 V 列数 > 0 且 ≤ 339
- **训练层**:checkpoint 保存失败不影响结果记录(打 WARN);best_metrics 不为 None 守护
- **部署层**:ONNX 4 输入 parity 仍 hard gate;TRT 引擎构建保持 try/except 降级
- **测试**:沿用 Stage 1 pytest + smoke + 微数据集 e2e,全部适配新接口
- 工作流:同 Stage 1(spec → plan → 子代理驱动实现 → 双重审查 → 终审 → 合并)

---

## 10. DESIGN_JOURNAL v2(版本化要求)

`docs/DESIGN_JOURNAL.md` **追加** v2 小节,**v1 完整保留**:
```markdown
## v1 (2026-05-15) — Stage 1 初始设计与执行
[完整保留,不动一字]

---

## v2 (2026-MM-DD) — Stage 2 模型基础升级

### 重定向:Stage 1 → Stage 2 的认知转变
[Stage 1 实测发现深度输给 LGB;v1 已识别根因为缩放序数 cat + V1-V50;
Stage 2 优先解决根因。文献:Shwartz-Ziv & Armon 2022]

### 设计决策(每条 初衷+原理+文献支撑)
- v2-1:类别字段独立 nn.Embedding,双塔共享一组 mixer
- v2-2:V 列相关性贪心剪枝(threshold 0.95)+ 同时跑全量做消融
- v2-3:实验矩阵收窄到 1 配置 × 2 V 策略(诚实原则)
- v2-4:保 best checkpoint(Stage 1 漏项)
- v2-5:四情景诚实成功框架(不为凑赢调超参)

### Stage 2 范围明确不在的(避免 scope creep)
[异质图/损失深化/cuDNN/PMML 全部推 Stage 3+]

### 执行中发现的问题与修复
[Stage 2 实施时遇到的真实 bug,commit SHA + 根因 + 修复]

### 实验结果与诚实分析
[四情景之一,实测后填入]

### 参考文献(v2 新增)
- Shwartz-Ziv & Armon 2022, Tabular Data: Deep Learning is Not All You Need
- Cheng et al. 2016, Wide & Deep Learning for Recommender Systems
- Guo et al. 2017, DeepFM
- Gorishniy et al. 2021, Revisiting Deep Learning Models for Tabular Data (FT-Transformer)
- IEEE-CIS Kaggle 社区 V 列剪枝公开 kernels
```

### README 更新
- v1 README 主体保留(Stage 1 章节、诚实声明、阶段路线图)
- 末尾追加 `## Stage 2 结果(2026-MM-DD)` 节:新命令、4 行结果表、benchmark 表、四情景中真实命中的那一种叙事
- 头部"## 阶段"路线图把 Stage 2 标 ✅

---

## 11. Stage 2 完成定义(DoD)

- [ ] `EmbeddingMixer` 实现 + 测试通过
- [ ] `FeatureProcessor.transform` 返回 dict 接口,Stage 1 旧测试适配通过
- [ ] V 列剪枝工具 + 缓存 + 测试通过
- [ ] `build.py` 双 v_strategy 跑通,`data/processed/{full_v,pruned_v}/` 各 5 文件齐全
- [ ] `make_loader` 吐 dict batch,loader 测试通过
- [ ] `FraudModel` 持 mixer + 新 forward 签名,所有 model/fraud_model 测试通过
- [ ] `train.py` 保存 best checkpoint,smoke 测试断言文件被写入
- [ ] `run_stage2_matrix` 跑完 4 配置,`stage2_results.json` 落盘
- [ ] LightGBM 在两套 V 上跑通,checkpoint(.pkl)落盘
- [ ] ONNX 4 输入导出 + parity 通过(两个深度模型各一份)
- [ ] `run_benchmark` 双模型,`benchmark_stage2.json` 含 8 条记录
- [ ] `pytest -v` 全绿(预计 ~40 测试)
- [ ] `DESIGN_JOURNAL.md` 追加 v2 小节,**v1 字节级未变**
- [ ] README 追加 Stage 2 结果节,路线图打勾 Stage 2
- [ ] 所有 Stage 2 实施中发现的 bug 在 DESIGN_JOURNAL v2 诚实记录(commit SHA + 根因 + 修复)

---

## 12. 参考文献

- Shwartz-Ziv & Armon, *Tabular Data: Deep Learning is Not All You Need*, Information Fusion 2022
- Cheng et al., *Wide & Deep Learning for Recommender Systems*, DLRS 2016
- Guo et al., *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, IJCAI 2017
- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 (FT-Transformer)
- Hamilton et al., *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS 2017
- IEEE-CIS Fraud Detection competition (Kaggle/Vesta), 2019 —— 公开 kernel 中的 V 列相关性剪枝惯例
- Stage 1 spec(`docs/superpowers/specs/2026-05-14-ant-riskcontrol-stage1-design.md`)
- Stage 1 设计日志(`docs/DESIGN_JOURNAL.md` v1)
