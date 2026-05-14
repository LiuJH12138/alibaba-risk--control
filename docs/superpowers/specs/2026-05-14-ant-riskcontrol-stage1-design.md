# 阿里/蚂蚁风控算法组实习 —— 复刻项目设计文档(Stage 1)

- **日期**:2026-05-14
- **状态**:设计已确认,待写实现计划
- **路线**:路线 A —— 双塔融合一体化模型
- **数据**:单一数据集 IEEE-CIS,一体化端到端系统

---

## 0. 背景与可行性结论

本项目目标是复刻一段实习经历的技术内容:

> 阿里巴巴 蚂蚁集团风控算法组 - 算法实习生(2024.06–2024.09)
> 1. 行为序列与异质图建模:Transformer-GRU 行为序列模型 + GNN 识别资金转移链路核心团伙节点(团伙识别准确率 +15%)
> 2. 极度不平衡样本处理:Hybrid Focal Loss + Hard Negative Mining(保 90%+ 召回,误伤 -20%)
> 3. 高性能推理架构优化:PMML/TensorRT 异构部署,延迟 150ms→45ms,支撑每秒万级 TPS
> 4. 指标:离线 AUC 0.98,上线后核心风险资损率 -8%

### 可行性结论

**不能"完美完整"复现,但能高保真复刻方法论与工程能力。**

无法复现的部分(本质障碍,非能力问题):
- **原始数据**:蚂蚁专有的千万级跨境交易数据,涉密、从未公开、不可带出。
- **生产环境**:双 11 高并发、每秒万级 TPS 在线扫描、生产基础设施。
- **业务指标**:"资损率 -8%、挽回损失逾百万" 是生产 A/B 实验结果。

能够高保真复刻的部分:
- 全部技术方法(Transformer-GRU、GNN、Hybrid Focal Loss + HNM)
- 全部工程能力(ONNX→TensorRT 部署、延迟 benchmark)
- 离线指标与消融实验(在公开数据上)

### 诚实原则(写死,贯穿全项目)

简历里的 "+15% / −20% / AUC 0.98 / 资损 -8%" 是蚂蚁数据上的数字。本项目产出的是**公开数据(IEEE-CIS)上我们自己的真实数字**。我们复现的是**方法论与改进的方向/形态**(消融证明 GNN 有增益、Hybrid Loss 能在保召回下降误伤),用诚实的数字 —— **不编造数字去凑简历**。

现实预期:IEEE-CIS 公开数据上 ROC-AUC 现实区间约 0.93–0.96(Kaggle 顶尖方案量级),不一定到 0.98。

---

## 1. 整体拆分:3 个子项目

整个复刻太大,无法塞进单一 spec。分为 3 个阶段,每个阶段独立走 spec → plan → 实现 → 实验。

| 阶段 | 子项目 | 对应简历项 | 产出 |
|------|--------|-----------|------|
| **Stage 1**(本文档) | 一体化端到端 MVP | 三项全部,最小可信深度 | 能跑通的 pipeline + 真实 AUC/召回-误伤曲线/延迟数字 + 结果映射 README |
| Stage 2 | 深化建模 | 异质图 + 不平衡处理 | 异质 GNN(多类型节点/边)+ 团伙核心节点识别;复现 "+15%"、"90% 召回 / -20% 误伤" 的对比实验与消融 |
| Stage 3 | 生产化可部署系统 | 高性能推理架构 | 真实打分服务(Triton/FastAPI)+ TensorRT INT8 校准 + 算子优化 + 单笔/批量延迟 benchmark + 可接入真实数据接口 |

**本文档只设计 Stage 1。**

---

## 2. 环境

- **云端主机(AutoDL)**:Ubuntu 22.04,RTX 5090 32GB,208 线程,754GB 内存
- **存储**:项目位于数据盘 `/root/autodl-tmp/alibaba-risk-control-internship`(软链 `/root/alibaba-risk-control-internship`)
- **基础**:conda base 已装 PyTorch 2.8.0+cu128,CUDA 12.8 工具链
- **本项目**:新建 conda 环境(不污染 base),复用 PyTorch 2.8+cu128

---

## 3. Stage 1 架构总览

在 IEEE-CIS 交易数据上,构建「序列塔 + 图塔 → 门控融合头」的双塔模型,用 Hybrid Focal Loss + Hard Negative Mining 训练,再把在线推理路径导出 TensorRT 做延迟 benchmark。

```
IEEE-CIS 原始 CSV
   │
   ├─[数据层]─ 合成 uid → 按 uid 聚合交易历史
   │           ├─ 序列构造:每笔交易 → 前 N 笔滑窗序列
   │           └─ 图构造:交易节点,共享实体连边(time-respecting)
   │
   ├─[模型层]─ 序列塔: Transformer 编码器 → GRU → seq_emb
   │           图塔:  GraphSAGE → graph_emb
   │           门控融合头: gate ⊙ s + (1−gate) ⊙ g → MLP → 欺诈 logit
   │
   ├─[训练层]─ 时间切分(早期训练/后期验证,防泄漏)
   │           Hybrid Focal Loss + 批内 Hard Negative Mining
   │
   └─[部署层]─ 序列塔+融合头 → ONNX → TensorRT 引擎
               图 embedding 离线预计算(作为特征查表)
               benchmark:PyTorch eager vs TensorRT 单笔延迟
```

### 关键设计决策

1. **图 embedding 离线预计算**:GNN 动态图结构对 TensorRT 不友好。Stage 1 把图 embedding 离线算好、推理时当特征查表,只有序列塔 + 融合头走 TensorRT。这符合工业界做法(图特征通常离线算)。完整在线图推理留 Stage 3。
2. **PMML 是树模型那一路**:PMML 不支持深度网络。"PMML/TensorRT 异构部署" 的复刻 = 轻量 LightGBM(PMML)+ 深度双塔(TensorRT)两级架构。Stage 1 只演示两条导出链路跑通,真正级联服务留 Stage 3。

---

## 4. 数据管线

### 4.1 获取与加载
- Kaggle CLI 下载 → `data/raw/`:`train_transaction.csv`(590,540×394)+ `train_identity.csv`(144,233×41),按 `TransactionID` 左连接。
- 竞赛 test 集无标签,**只用 train_\* 自己做时间切分**。

### 4.2 特征处理
- 类别特征(ProductCD / card1-6 / addr1-2 / P,R_emaildomain / M1-9 / Device* / id_12-38)→ label-encode → embedding 层
- 数值特征(TransactionAmt 取 log1p / C1-14 / D1-15 / dist / V1-339)→ 标准化;大量 NaN → 填充 + 缺失指示位
- **所有编码器/scaler 只在 train 上 fit**,再 transform 到 val(防泄漏铁律)

### 4.3 uid 合成(用户身份代理)
- IEEE-CIS 无用户 ID。社区标准代理:`uid = card1 + addr1 + (TransactionDay − D1)`。
- **注明**:这是启发式代理,非真实 ground truth,有其局限。会写入设计 README。

### 4.4 序列构造
- 按 uid 分组、按 TransactionDT 排序;第 i 笔交易的序列 = 同 uid 前 N 笔滑窗(N≈16/32,不足 padding),标签 = 第 i 笔的 isFraud
- 输出 `[样本数, N, 特征维]` + mask + label;冷启动(无历史)交易仍打分,走全 padding 序列

### 4.5 图构造
- 节点 = 交易(Stage 1 同构图,异质图留 Stage 2);边 = 两笔交易共享关键实体(同 card1 / addr1 / 邮箱 / DeviceInfo)
- **防爆炸**:热门实体会连出海量边 → 只用高区分度实体 + 度数封顶 + 采样
- **防泄漏(关键)**:边只能指向时间更早的交易(time-respecting graph)
- 用 PyG 构建,节点特征 = 每笔交易特征向量

### 4.6 时间切分
- 按 TransactionDT 排序,前 80% 时间 → train,后 20% → val。随机切分会泄漏,不用。

### 4.7 产物与校验
- 处理后张量缓存到 `data/processed/`
- 数据 manifest:各 split 行数 / 欺诈率 / 序列长度分布 / 图度数分布
- 自动断言:欺诈率 ≈ 3.5%、无 uid 跨切分泄漏、图无「未来→过去」边、编码器仅 train fit

---

## 5. 模型层

### 5.1 序列塔(Transformer → GRU)

```
[B,N,特征] → 嵌入层(每类别字段独立 embedding + 数值线性投影)→ d_model
           → 位置编码(Stage 1 标准位置编码;时间间隔感知编码留 Stage 2)
           → Transformer 编码器 ×1–2 层(多头自注意力 + padding mask)—— 浅,只做跨步上下文混合
           → GRU(吃上下文增强序列,取最后有效隐状态,pack_padded 处理 mask)
           → seq_emb [B, d_seq]
```

**Transformer–GRU 联动原理**(写入设计 README,附文献):
- 选用 **Transformer → GRU** 顺序:Transformer 自注意力先把每个时间步「全局重表示」(知道序列里所有相关步),GRU 再顺序压缩成最终隐状态。GRU 末隐状态 = "用户当前状态,已吸收全局上下文",正好是给当前交易打分要的东西。
- **分工**:Transformer 管「哪些历史交易相关」,GRU 管「按时间收敛成当前状态」。
- **防冗余**:两者都建模时序,naïve 堆叠会职责重叠。Transformer 保持浅(1–2 层),GRU 负责带近因偏置的时序压缩。
- 文献:FTT-GRU(arXiv:2511.00564);Attention-Based Transformer+GRU(MDPI Mathematics 13(9):1484)。

### 5.2 图塔(GraphSAGE)
- 跑 §4.5 的交易图,节点特征 = 单笔交易特征;GraphSAGE ×2 层,邻居采样,只聚合时间更早邻居 → `graph_emb [B, d_graph]`
- **训练方式**:推荐联合训练(PyG `NeighborLoader` 采样子图,与序列 batch 对齐);退路 = 预训练图塔后冻结
- ⚠️ **最棘手工程点**:dataloader 要为 batch 里每笔交易同时吐出(序列张量 + 采样子图 + 标签),需自定义 collate。Stage 1 实现的重点风险项。

### 5.3 门控融合头
```
s = W_s·seq_emb,  g = W_g·graph_emb        (投影到公共维 d_fuse)
gate = σ(W_gate·[s; g])                     (逐维门控)
fused = gate ⊙ s + (1−gate) ⊙ g
logit = MLP(fused)
```
- 逐样本自适应融合优于静态拼接(有些交易序列信号主导、有些图信号主导)。
- 文献:RAGFormer 注意力融合(arXiv:2402.17472);ETH-GBERT 逐样本动态融合(arXiv:2501.02032)。
- 消融变体共用同一代码路径:seq-only / graph-only / concat→MLP / gated。

### 5.4 损失函数:Hybrid Focal Loss + Hard Negative Mining

**Hybrid Focal Loss 明确定义**(写入设计 README):

> Hybrid Focal Loss = 非对称 Focal Loss(正负样本用不同 γ_pos / γ_neg)+ 类别平衡 α 加权(有效样本数)

- 标准 Focal(Lin 2017,RetinaNet):`FL = −α(1−p_t)^γ log(p_t)`,压低易例、聚焦难例
- 非对称 γ(借鉴 ASL, Ben-Baruch 2020):γ_pos ≠ γ_neg —— 召回/误伤的调节旋钮,直接服务"保 90% 召回、降 20% 误伤"
- 类别平衡 α(有效样本数,Cui 2019)
- **Hard Negative Mining**(OHEM, Shrivastava 2016):每 batch 内负样本按 loss 排序,只取最难 top-K 进反向(难负:正 ≈ 3:1)
- ⚠️ Focal 软性压低易例,HNM 硬性丢弃易负,两者互补但可能过度聚焦 → γ 与 HNM 比例需联合调参
- **损失消融**:BCE / Focal / Hybrid Focal / Hybrid Focal + HNM

### 5.5 LightGBM 基线
扁平表特征上训 LightGBM:既是强基线,又是 PMML "异构部署" 那一路。Stage 1 保持最小化。

### 5.6 规模与校验
- d_model / d_seq / d_graph / d_fuse ≈ 128,RTX 5090 32GB 绰绰有余
- 单测:Focal 在 γ=0,α=0.5 时退化为 BCE;ASL 在 γ_pos=γ_neg 时退化为 Focal;用 logits + logsigmoid 保数值稳定;门控输出 ∈[0,1];padding 步不贡献梯度;联合训练下三塔都有梯度

### 5.7 模型互补性 —— 设计依据与诚实前提

| 模型 | 捕捉的信号 | 能抓的欺诈模式 |
|------|-----------|--------------|
| 序列塔(Transformer-GRU) | 用户自身时序行为:支付节奏突变、速度异常 | 账户盗用、行为突变 |
| 图塔(GNN) | 跨实体结构信号:卡与已知欺诈卡共用地址等 | 团伙作案、资金归集、共设备串通 |
| 门控融合头 | 逐样本自适应权衡两路 | —— |

**诚实前提**:"互补" 是经验性结论,不能靠架构假设。若 graph-only AUC ≈ fusion AUC,说明图无贡献。§6.3 的架构消融就是用来验证的。另:IEEE-CIS 非原生图数据集,图是按共享实体构造的,信号强度中等;最强图故事在 Stage 2 异质图。

---

## 6. 训练与评估

### 6.1 训练设置
- AdamW + warmup + cosine 衰减;AMP 混合精度(bf16);梯度裁剪
- batch 开大;固定 seed,可复现
- 早停看 val PR-AUC;best checkpoint 按 PR-AUC 存
- TensorBoard 日志:loss 曲线、各指标、门控值分布

### 6.2 评估指标(极不平衡 → 不只看 AUC)
| 指标 | 用途 |
|------|------|
| ROC-AUC | 主标题指标,映射"离线 AUC" |
| PR-AUC (AP) | 不平衡下的真实度量 |
| Recall@固定FPR / Precision@固定Recall | 工作点指标 |
| 召回–误伤权衡曲线 | 直接映射"保 90% 召回降误伤" |
| KS 统计量 | 风控/信用评分惯用 |
| 混淆矩阵@选定阈值 | —— |

### 6.3 实验矩阵 —— 每个实验映射回一条简历 bullet

| 实验 | 对比项 | 映射简历 |
|------|--------|---------|
| 架构消融 | seq-only / graph-only / concat / gated | "引入 GNN…团伙识别提升 15%" → fusion vs seq-only 的增益 |
| 损失消融 | BCE / Focal / Hybrid Focal / +HNM | "保 90%+ 召回同时误伤降 20%" → 固定召回 90%,看 FP 下降 |
| 基线对比 | LightGBM vs 双塔模型 | 整体方法有效性 |
| 主指标 | 最终 ROC-AUC / PR-AUC | "离线 AUC" |
| 延迟 | PyTorch vs TensorRT(§7) | "150ms→45ms" |

所有实验同切分、同 seed、同评估代码 → 可比;结果自动落 `experiments/*.json`,写进设计 README 对应版本小节。

### 6.4 校验
- 微数据集训练 smoke test;断言 val 指标只在留出时间段上算;NaN/inf 防护;支持断点续训

---

## 7. 部署与延迟 Benchmark

### 7.1 部署路径
```
离线批处理:  GraphSAGE 跑全图 → graph_emb → 存查表(Stage 1 用 parquet/dict)
在线单请求:  收到交易 → 拼装该 uid 近 N 笔序列 → 查表取 graph_emb
            → 序列塔 + 门控融合 + 头 ⟶ TensorRT 引擎 ⟶ 欺诈分
```

### 7.2 导出链路
- PyTorch → 切出在线路径子模块(序列塔+融合+头)→ ONNX(动态 batch 轴)→ TensorRT 引擎(Stage 1 用 FP16;INT8 校准留 Stage 3)
- **强制数值一致性校验**:PyTorch ≈ ONNX ≈ TensorRT(atol 容差内),不通过不信任延迟数字

### 7.3 PMML 异构那一路
- LightGBM 基线 → PMML 导出(sklearn2pmml / nyoka)
- "异构部署" 叙事:轻量树模型 PMML(粗筛/可解释)+ 深度双塔 TensorRT(精排)两级。Stage 1 只演示两条导出链路跑通 + 各自 benchmark,级联服务留 Stage 3。

### 7.4 延迟 Benchmark 方法学
**对比配置(4 档)**:
1. PyTorch eager · CPU(naive 基线,对应"before")
2. PyTorch eager · GPU
3. torch.compile · GPU(中间档)
4. TensorRT FP16 · GPU(优化目标,对应"after")

**测法**:warmup → N≥1000 次计时 → CUDA synchronize → 报 p50/p95/p99 分布;单请求(batch=1)延迟 + 各 batch 吞吐 TPS;分别测「纯模型推理」与「含特征拼装」。

**对应简历的诚实版本**:用 PyTorch-CPU/eager 作 "before"、TensorRT 作 "after",报实测真实加速比。"150ms→45ms" 的形态可复现,确切数字不保证。TPS 实测单卡 5090 可达多少就报多少。

### 7.5 算子优化(简历"优化特征计算算子")
Stage 1 适度范围:torch profiler / nsys 定位瓶颈算子;记录 TensorRT 自动融合了哪些层;特征拼装/类别查表向量化,benchmark 前后对比。自定义 CUDA 算子留 Stage 3。

### 7.6 产物与校验
- `deploy/export_onnx.py` / `build_trt.py` / `benchmark.py`
- Benchmark 报告:4 档延迟表 + TPS 曲线 + 数值一致性报告 → 落 `experiments/` 并写入设计 README
- 引擎构建硬件专属(标注 RTX 5090 / 本机 CUDA);TensorRT 未装时其余 3 档仍可跑

---

## 8. 工程结构

```
alibaba-risk-control-internship/
├── README.md                  # 总览 + 结果→简历bullet映射表
├── docs/
│   ├── DESIGN_JOURNAL.md       # 版本化设计日志(v1/v2…累积,核心交付物)
│   └── superpowers/specs/      # spec 设计文档
├── configs/                    # data/model/train.yaml
├── data/{raw,processed}/       # gitignore
├── src/
│   ├── data/                   # load · features · uid · sequence · graph
│   ├── models/                 # sequence_tower · graph_tower · fusion · losses · fraud_model
│   ├── train.py · evaluate.py · baseline_lgbm.py
│   └── deploy/                 # export_onnx · build_trt · benchmark
├── experiments/                # 实验结果 JSON + 曲线图
├── notebooks/eda.ipynb
├── tests/                      # test_data · test_losses · test_models · test_smoke
└── environment.yml + requirements.txt
```

### 依赖
- 新增:`torch-geometric` `kaggle` `lightgbm` `sklearn2pmml` `onnx` `onnxruntime-gpu` `tensorrt` `scikit-learn` `pandas` `pyarrow` `pytest` `pyyaml`
- ⚠️ TensorRT 需匹配 CUDA 12.8,安装可能要走 NVIDIA 源 —— plan 单列风险项
- `environment.yml` + `requirements.txt` 双留

---

## 9. 版本化设计 README(核心交付物)

`docs/DESIGN_JOURNAL.md` 是贯穿全项目的演进式设计日志:
- 每个设计决策记录「设计初衷 + 真实文献支撑 + 详细原理解释」
- 每次更新**保留旧版本记录、追加新版本**(`## v1 (2026-05-14)` / `## v2 (...)` 累积,不覆盖)
- v1 = Stage 1 初始设计

---

## 10. 错误处理与测试

### 错误处理(贯穿各层)
- 数据层:Kaggle token 无效/文件缺失 → 明确报错;断言行数、欺诈率≈3.5%、无泄漏
- 训练层:NaN/inf 守卫、OOM 友好提示、断点续训
- 部署层:数值一致性断言失败即停;TensorRT 缺失自动降级到其余 3 档
- 配置:yaml schema 校验

### 测试策略
- 单测:`losses`(退化性质 + 数值稳定)、`models`(形状/mask/梯度流)、`data`(序列窗口正确、图无未来边、编码器只 train fit)
- `test_smoke`:微数据集(几千行)端到端跑通 train→eval→export→benchmark
- `pytest` 一键可跑

---

## 11. Stage 1 完成定义(DoD)

- [ ] IEEE-CIS 下载 + 预处理跑通,数据校验通过
- [ ] 双塔模型联合训练跑通,产出 best checkpoint
- [ ] 4 组实验矩阵全部完成,结果落 `experiments/`
- [ ] ONNX→TensorRT 导出 + 数值一致性通过,4 档延迟 benchmark 完成
- [ ] LightGBM→PMML 导出跑通
- [ ] README 映射表 + `DESIGN_JOURNAL.md` v1 完成
- [ ] `pytest` 全绿

---

## 12. 参考文献

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
- Ben-Baruch et al., *Asymmetric Loss for Multi-Label Classification*, 2020
- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019
- Shrivastava et al., *Training Region-based Object Detectors with Online Hard Example Mining*, CVPR 2016
- *FTT-GRU: Hybrid Fast Temporal Transformer with GRU*, arXiv:2511.00564
- *Attention-Based Transformer + GRU*, MDPI Mathematics 13(9):1484
- *RAGFormer / Binding Global and Local Relational Interaction*, arXiv:2402.17472
- *Dynamic Feature Fusion (ETH-GBERT)*, arXiv:2501.02032
- IEEE-CIS Fraud Detection dataset, Kaggle (Vesta Corporation)
- Hamilton et al., *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS 2017
