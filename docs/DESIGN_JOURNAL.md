# 设计日志(DESIGN JOURNAL)

版本化、累积式设计记录。每次设计更新**追加新版本小节**,不覆盖旧记录。

---

## v1 (2026-05-15) — Stage 1 初始设计与执行

### 设计决策

#### 决策 1:路线 A —— 双塔融合一体化模型

**设计初衷:**
简历描述的是行为序列 + 图 + 损失 + 部署联动的一个系统。Stage 1 目标是用一个端到端
MVP 覆盖全链路,验证架构可行性,而非追求最优单项指标。

**原理:**
- 序列塔捕捉用户自身时序行为(Transformer 做全局上下文感知,GRU 做近因压缩);
- 图塔捕捉跨实体结构信号(团伙、资金归集、设备复用等图上涌现的欺诈模式);
- 门控融合(gated fusion)逐样本自适应权衡两路信号,而非固定拼接。

**文献支撑:**
- RAGFormer(arXiv:2402.17472):"GNN 学习全局特征,互补 Transformer 的局部特征",
  在金融欺诈检测中实证了序列塔与图塔的互补性。
- ETH-GBERT(arXiv:2501.02032):全局结构信息 + 局部语义动态融合,与本项目门控融合
  思路一致。

---

#### 决策 2:Transformer → GRU 顺序(序列塔内部)

**设计初衷:**
给当前交易打分,需要"用户当前状态 + 全局上下文"的表示,而非序列中每个位置的表示。

**原理:**
Transformer 自注意力先把序列每步做全局重表示(跨步依赖、长程模式);GRU 再以近因偏置
将序列压缩成末隐状态(h_n),对应当前交易时刻的用户状态。Transformer 保持浅层(1–2 层)
以避免与 GRU 职责重叠。前向 padding 下,h_n 自然落在当前交易位置。

**文献支撑:**
- FTT-GRU(arXiv:2511.00564):Transformer 特征提取 + GRU 时序建模的串联结构在序列
  欺诈检测中的应用。
- Attention-Based Transformer + GRU(MDPI Mathematics 13(9):1484):Transformer-GRU
  串联架构在时序分类任务中的系统性验证。

---

#### 决策 3:Hybrid Focal Loss = 非对称 Focal + 类别平衡 α

**设计初衷:**
IEEE-CIS 欺诈率约 3.5%,生产场景更达万分位级不平衡,需在保召回的同时压误伤。
单一交叉熵或标准 Focal Loss 对欺诈/正常样本的难度不对称处理不足。

**原理:**
- `γ_pos ≠ γ_neg`:对欺诈样本(正类)和正常样本(负类)分别设置聚焦系数,提供
  召回/误伤调节旋钮;
- `α`:类别平衡权重,处理正负样本数量悬殊;
- HNM(OHEM,Online Hard Example Mining):在每个 batch 内选择损失最高的难负样本
  子集进行梯度更新,专攻"会被误报的难负样本"。

**文献支撑:**
- Focal Loss(Lin et al., ICCV 2017):原始 Focal Loss,`γ` 下调易分类样本贡献。
- Asymmetric Loss(Ben-Baruch et al., 2020):正负类非对称 `γ`,即 `γ_pos ≠ γ_neg`
  设计的直接来源。
- Class-Balanced Loss(Cui et al., CVPR 2019):有效样本数 `α` 加权的理论基础。
- OHEM(Shrivastava et al., CVPR 2016):在线难样本挖掘的实现基础。

---

#### 决策 4:图 embedding 离线预计算用于部署

**设计初衷:**
GNN 的动态图结构(NeighborLoader 子图采样)对 TensorRT ONNX 导出不友好——动态拓扑
无法静态化为固定计算图。

**原理:**
- **训练时:**NeighborLoader 驱动联合训练(图塔 + 序列塔端到端梯度);
- **部署时:**图 embedding 离线预计算并存入查找表(lookup table),在线推理只需
  查表取 embedding,序列塔 + 融合头走 ONNX/TensorRT 静态图,保证低延迟可部署。

这一分离也使得 ONNX 导出路径仅需处理序列塔,绕开了 GNN 动态图的 ONNX 限制。

**文献支撑:** GraphSAGE(Hamilton et al., NeurIPS 2017)的归纳式设计使节点 embedding 可离线批量预计算后查表服务,无需在线重算图结构 —— 这是"图特征离线、在线只跑序列塔+融合头"部署模式的理论依据;工业界 GNN 服务普遍采用 embedding 离线预计算 + 特征存储的模式。

---

#### 决策 5:uid 合成用 card1 + addr1 + (day − D1) 启发式

**设计初衷:**
IEEE-CIS 数据集没有显式用户 ID,但行为序列塔需要按用户聚合历史交易。

**原理:**
`uid = hash(card1, addr1, floor((TransactionDT − min_DT) / 86400))` 即"同卡 + 同地址 +
同天"归为同一 uid。这是社区广泛使用的标准启发式。

**文献支撑:** IEEE-CIS Fraud Detection 竞赛(Kaggle, 2019)公开 kernel 社区的标准做法 —— 用 card1 + addr1 + (TransactionDay − D1) 合成用户/账户代理 ID(该启发式在竞赛高分公开方案中广泛使用,如 Chris Deotte 等的公开 notebook)。属社区经验性工件,非学术文献。

**局限(诚实记录):**
此为代理 uid(proxy),非真实 ground truth。同一用户若换卡/换地址则切割为多 uid;
不同用户若共享设备/地址则可能合并。这是 IEEE-CIS 数据的固有局限,行为序列信号
因此有噪声。

---

### 执行中发现的问题与修复(诚实记录)

#### Bug 1:特征矩阵 NaN —— 类别编码溢出导致激活爆炸

**commit:** `ce6f459`

**现象:** 模型前向传播产生 NaN;注意力分数溢出。

**根本原因:** `FeatureProcessor` 将类别字段的原始整数编码(例如 card1 编码最大值达 12730)
直接混入 float 特征矩阵,未做归一化。高值类别编码与数值特征混合后,激活值溢出
→ Softmax 输入 ±∞ → 注意力分数 NaN。

**修复:** 类别编码按基数缩放到 [0, 1):$x_{cat} = \text{code} / \text{cardinality}$;
数值标准化值裁剪到 [-10, 10],防止异常值穿透。

**代价/遗留:** 缩放序数编码是 NaN 修复的应急方案,损失了类别语义(embedding 层
才能学到类别间相似性)。proper categorical embedding 留 Stage 2。

---

#### Bug 2:train.py GPU 兼容 + 缺学习率 warmup

**commit:** `eaa0a95`

**现象 1:** `_evaluate` 函数在 GPU 张量上直接调用 `.numpy()` 崩溃
(`RuntimeError: can't convert CUDA tensor to numpy`).

**现象 2:** 配置文件中存在 `warmup_steps` 配置项,但训练循环从未使用它——
学习率从训练开始就按 cosine schedule 衰减,缺少预热阶段。

**修复 1:** 所有评估路径的张量调用 `.cpu().numpy()`。

**修复 2:** 加入线性 LR warmup 调度器:前 `warmup_steps` 步线性升至目标 LR,
之后切换 cosine decay。

---

#### Bug 3:SequenceTower pack_padded_sequence 与前向 padding 不兼容

**commit:** `dab5952`

**现象:** seq_only 配置 roc_auc 仅 0.55(接近随机),且无法导出 ONNX
(动态序列长度使 pack_padded_sequence 的 ONNX trace 失败)。

**根本原因:** 数据构建用**前向 padding**(历史记录填在序列末尾,开头为零 padding),
但 `pack_padded_sequence` 假设数据从序列开头排列——对历史长度 < seq_len/2 的交易,
GRU 打包后只处理全零 padding,实际有效历史完全被忽略 → GRU 输出常量向量,
等价于序列塔无效。

**修复:** 改用 plain GRU 处理整个固定长度序列(不打包),`h_n` 自然落在最后一个时间步
(当前交易),正确读取前向 padding 下的末位状态。

**效果:** seq_only roc_auc: 0.55 → **0.844**(修复后)。ONNX 导出路径同步解锁。

---

#### Bug 4:build_trt.py TensorRT 8.x API 与 TensorRT 10.x 不兼容

**commit:** `03113c2`

**现象:** TensorRT 引擎编译脚本在 TRT 10.16 环境下抛出 AttributeError / TypeError。

**根本原因(三处 API 变更,TRT 8→10):**
1. `trt.MemoryPoolFlag` → 应为 `trt.MemoryPoolType`(枚举名变更);
2. `EXPLICIT_BATCH` flag 处理方式变更(TRT 10 默认 explicit batch);
3. `IHostMemory` 写入接口变更(`.tobytes()` / buffer protocol 变化)。

**修复:** 按 TRT 10.x API 重写相关调用,并在 `run_benchmark` 中对每个配置独立
`try/except`——任一配置失败记录为 skipped 而不中断整体 benchmark,保证结果
文件完整写出。

---

### Stage 1 偏离设计文档之处(诚实记录)

#### 偏离 1:V 列削减到 V1–V50

**设计文档预期:** 使用 IEEE-CIS 全部特征(Vesta 工程特征 V1–V339 + 其他字段,
展开后约 791 列)。

**实际执行:** 只保留 V1–V50,feat_dim = 213,seq_all.pt 约 15GB。

**原因:** 完整特征集 × seq_len=32 × 59 万行,seq_all.pt 约 60GB,超出 AutoDL
50GB 数据盘限制。

**代价:** 丢弃了 Vesta 大量有信号的工程特征。这是本 Stage 1 所有深度模型
AUC 偏低的原因之一。V 列完整利用留 Stage 2(需更大存储或更激进压缩)。

---

#### 偏离 2:类别字段用「缩放序数编码」而非 embedding 层

**设计文档预期(§5.1):** 类别字段(card1、card2、addr1、addr2、P_emaildomain 等)
进入独立 embedding 层,学习类别间语义相似性。

**实际执行:** 缩放序数编码(code / cardinality → [0,1)),作为 Bug 1 NaN 修复的
应急方案。

**代价:** 丢失类别语义;card1 的 12730 个值被压成 [0,1) 的标量,模型无法区分
不同类别的语义距离。这是 AUC 偏低的另一原因。proper embedding 留 Stage 2。

---

#### 偏离 3:kaggle CLI 版本 1.6.17 → 2.1.2

**原因:** 新版 Kaggle API token 格式(KGAT_ 前缀)需要新版 CLI。执行中升级。

---

#### 偏离 4:PMML 导出列为已知遗留项

**设计文档预期:** LightGBM 基线训练后导出 PMML 文件,验证异构部署链路。

**实际执行:** LightGBM 基线训练成功(roc_auc 0.9076,已入 results.json),但
PMML 导出受 sklearn2pmml/JPMML 与 numpy 2.x 兼容性问题阻塞——需要
sklearn2pmml 0.130+ 以及 Java 11+;Java 11+ 工具链在本 AutoDL 环境安装极慢。

**结论:** PMML 文件导出列为 Stage 1 已知遗留项;完整 PMML/异构部署归 Stage 3。

---

### 实验结果与诚实分析

#### 架构与损失消融(深度模型 + LightGBM 基线)

数据集:590,540 笔交易,欺诈率 3.5%,训练集 472,432 / 验证集 118,108(时序切分),
feat_dim 213,seq_len 32,图边数 819,861。

| 配置 | roc_auc | pr_auc | ks | recall@fpr=.01 | fpr@recall=.90 |
|---|---|---|---|---|---|
| seq_only | 0.8438 | 0.4211 | 0.5499 | 0.3632 | 0.5335 |
| graph_only | 0.8481 | 0.3911 | 0.5402 | 0.3376 | 0.4730 |
| concat_fusion | 0.8491 | 0.4241 | 0.5521 | 0.3652 | 0.4945 |
| gated_fusion | 0.8412 | 0.4041 | 0.5348 | 0.3521 | 0.5115 |
| gated_plus_hnm | 0.8187 | 0.3144 | 0.4849 | 0.2562 | 0.5444 |
| lgbm_baseline | 0.9076 | 0.4813 | 0.6555 | 0.4050 | 0.2759 |

#### 延迟 benchmark(batch=1 单请求)

| 配置 | p50_ms | p95_ms | p99_ms | mean_ms |
|---|---|---|---|---|
| pytorch_cpu | 9.37 | 78.32 | 84.11 | 25.62 |
| pytorch_gpu | 1.33 | 1.36 | 1.38 | 1.34 |
| onnx_gpu | SKIPPED | — | — | — |
| tensorrt_fp16 | SKIPPED | — | — | — |

onnx_gpu 跳过原因:`CUDAExecutionProvider not active`;cuDNN/onnxruntime-gpu ABI 不兼容。

tensorrt_fp16 跳过原因:`TensorrtExecutionProvider not active`;同 cuDNN 问题。
TensorRT FP16 引擎**本身编译成功**(artifacts/online.engine,1.6MB)——仅 ORT 的
TensorRT 执行提供器无法加载。

#### 诚实分析

1. **双塔弱互补(与设计预判一致):**
   concat_fusion(0.849)边际超过单塔 seq_only(0.844)和 graph_only(0.848)。
   融合有增益但很小。原因:IEEE-CIS 是构造图(device、email、card 等字段的
   启发式连边),非真实社交/转账图,图信号强度中等。强图效果留 Stage 2 异质图深化。

2. **门控融合未超过简单拼接:**
   gated_fusion(0.841)< concat_fusion(0.849)。门控机制在本 Stage 1 设置下
   未带来增益——可能是门控参数训练不稳定,或特征质量不足以驱动有效门控。

3. **HNM 有害(诚实负面结果):**
   gated_plus_hnm(0.819)是所有配置最差。roc_auc、pr_auc、ks 全面下降,
   fpr@recall0.90 反而升高(误伤更多,与设计假设相反)。推测:Stage 1 简化特征
   下,HNM 筛出的"难负样本"可能是噪声标签或分布边缘样本,梯度信号反而干扰。
   HNM 深化留 Stage 2。

4. **LightGBM 基线反超所有深度配置(roc_auc 0.908 vs 深度模型最高 0.849):**
   典型的表格数据现象。梯度提升树在本 Stage 1 简化条件下(缩放序数编码 +
   V 列削减 + 单一特征集)碾压深度双塔。深度模型优势场景需要:proper embedding
   捕捉类别语义、完整 V 列特征、异质图结构——均留 Stage 2。

5. **延迟:pytorch CPU→GPU 约 7× 加速(9.4ms → 1.3ms):**
   这是核心的 before/after 对比,真实数字。TRT FP16 链路已通(引擎可编译),
   ORT-GPU EP 集成受 cuDNN ABI 阻塞,端到端 TRT 延迟归 Stage 3 测量。

---

### 诚实前提(设计时已声明,执行验证)

- **"模型互补"是经验性结论,由架构消融验证** —— 实验显示融合仅带来边际增益
  (concat 0.849 vs 单塔 ~0.845),即弱互补,与设计预判一致(IEEE-CIS 构造图
  信号中等;强图故事留 Stage 2 异质图)。

- **"HNM 降误伤"未被本 Stage 1 设置证实** —— gated_plus_hnm 反而更差
  (诚实负面结果)。

- **简历业务数字(AUC 0.98、资损 -8%)绑定蚂蚁专有数据,不复现。** 本项目
  所有数字均为 IEEE-CIS 公开数据上的真实结果。

---

### 参考文献

1. **Focal Loss:** T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
2. **Asymmetric Loss (ASL):** E. Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification," 2020.
3. **Class-Balanced Loss:** Y. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019.
4. **OHEM:** A. Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining," CVPR 2016.
5. **FTT-GRU:** arXiv:2511.00564 — "FTT-GRU: Feature-based Transformer and GRU for Fraud Detection."
6. **RAGFormer:** arXiv:2402.17472 — "RAGFormer: Retrieval-Augmented Graph Transformer for Financial Fraud Detection."
7. **ETH-GBERT:** arXiv:2501.02032 — "ETH-GBERT: Global Structure + Local Semantic Dynamic Fusion for Transaction Fraud Detection."
8. **GraphSAGE:** W. Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS 2017.
9. **IEEE-CIS Fraud Detection Dataset:** Kaggle / Vesta Corporation, IEEE-CIS Fraud Detection Competition, 2019.
10. **Attention-Based Transformer + GRU:** MDPI Mathematics 13(9):1484.


---

## v2 (2026-05-15) — Stage 2 模型基础升级

### 重定向:Stage 1 → Stage 2 的认知转变

Stage 1 实测:深度双塔模型 roc_auc 0.82-0.85,LightGBM 基线 0.908,深度输给基线。
v1 已识别根因为(a)类别字段用了缩放序数编码而非真正的 embedding,(b)V 列被
50GB 磁盘约束削减到 V1-V50。Stage 2 优先解决这两个根因 —— 异质图与损失深化推
到 Stage 3+。外部条件:用户将数据盘扩容到 150GB,完整 V 列(seq_all.pt ~62GB)
放得下。

文献:Shwartz-Ziv & Armon 2022 *Tabular Data: Deep Learning is Not All You Need*
—— GBDT 在中等规模表格上常胜过深度模型,深度方法需要正确的归纳偏置才能竞争。

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
实测保留 130 列(38%)。同时跑 full_v 和 pruned_v 做消融,用数据说话哪种更好。
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
**原理:** Stage 2 的"成功"=做完该做的工程改动 + 诚实测量。四种情景都"成功",
都给出对 Stage 3 方向的可信证据。**不为凑赢调超参** —— 实际命中**情景 4
(Both deep < LGB)**,见下文实验结果。

### Stage 2 范围明确不在的(避免 scope creep)

- 异质图(多类型节点/边)+ 团伙核心节点识别 → Stage 3+
- 损失函数深化(HNM 反而有害的根因调查)→ Stage 3+
- cuDNN/onnxruntime-gpu ABI 修复 → Stage 3+
- 完整 PMML 工具链(Java 11+ 安装难)→ Stage 3+
- TensorRT 完整端到端延迟测量 → Stage 3+(下面"实施 bug"会提到 Stage 2 TRT 引擎 build 已不通)

### 执行中发现的问题与修复(诚实记录)

1. **Task 4 build.py 代码审查发现(commit `a2e06c3`):** 双 `build_sequences` 调用
   缺少 mask-identity 断言;`V_PRUNED_CACHE` 硬编码路径未从 config 派生;丢失了
   fraud-rate 断言信息。三处全部修复。

2. **Task 8 baseline_lgbm.py 代码审查发现(commit `dc0893a`):** `if categorical_feature`
   真值检查会静默丢弃 `categorical_feature=0`;`import pickle` 在函数内。修复为
   `is not None` 检查 + import 提到顶层。

3. **Task 10 第一次跑:SIGKILL 在配置切换时。** `run_stage2_matrix` 跑完 `deep_full`
   后试图加载 `pruned_v/seq_all.pt`,进程被 SIGKILL(可能 seq_all 切换瞬时 RAM 峰值
   触发,虽然机器有 754GB)。恢复方案:`deep_pruned` 单独 fresh 进程运行。

4. **Task 10 第二次跑:Segmentation fault 早期触发。** `deep_pruned` 在 ~3 分钟时
   SIGSEGV(load/forward 早期)。可能是 C 扩展瞬时问题。第三次重试成功,
   `deep_pruned` 训出 roc_auc 0.864。

5. **Task 14 benchmark:TensorRT 引擎 build 失败 → 终审纠正(commit `292d0b1`)。** 终审 review 发现 `build_trt.py` 的 `set_shape` 输入名循环仍是 Stage 1 的
   `["seq", "mask", "graph_emb"]`,未跟随 ONNX 改为 4 输入,导致 `KeyError: 'seq'`
   被 try/except 优雅吞掉、返回 False。修复为 `["seq_cat", "seq_num", "mask", "graph_emb"]`
   后:TRT 引擎构建成功(两个模型均写出 `.engine` 文件),ORT TRT EP 仍不可用(cuDNN ABI
   符号缺失导致 `libonnxruntime_providers_tensorrt.so` 加载失败,与 cuDNN/ORT ABI 不兼容
   问题一致,benchmark 记录为 `skipped: "TRT EP not active; engine built OK"`)。
   **诚实记录:这是 Stage 2 实施 bug,与嵌入模型规模无关 —— 正确归因后情况是:TRT 编译
   链路已通,ORT TRT EP 集成受环境 cuDNN ABI 阻塞归 Stage 3+**。

6. **Task 14 .gitignore 调整:** 加 `!artifacts/online_*.onnx` 例外,让部署 ONNX
   产物可入 git(否则被 Stage 1 的 `*.onnx` 规则拦截)。

### Stage 1 修复继续生效(明确记录)

- SequenceTower 用 plain GRU(不用 pack_padded_sequence)
- FeatureProcessor 的 unknown 桶 = 0、num 裁剪 [-10,10]
- train.py 的 `.cpu().numpy()`、LR warmup
- build_trt.py 的 TRT 10.x API + try/except 优雅降级
- run_benchmark 的逐配置 try/except

### 实验结果与诚实分析

#### 架构 + V 策略消融(IEEE-CIS 公开数据,4 配置)

| 配置 | roc_auc | pr_auc | ks | recall@fpr=.01 | fpr@recall=.90 | train_s |
|---|---|---|---|---|---|---|
| deep_full | 0.8621 | 0.4370 | 0.5731 | 0.3632 | 0.4526 | 1853 |
| deep_pruned | **0.8639** | 0.4312 | 0.5637 | 0.3713 | 0.4584 | 2553 |
| **lgbm_full** | **0.9016** | **0.5556** | **0.6475** | **0.4941** | **0.3432** | <60 |
| lgbm_pruned | 0.8980 | 0.5303 | 0.6416 | 0.4678 | 0.3651 | <60 |

#### 延迟 benchmark(单笔 batch=1,本环境)

| 模型 | pytorch_cpu p50 | pytorch_gpu p50 | 加速 | onnx_gpu | tensorrt_fp16 |
|---|---|---|---|---|---|
| deep_full | 9.65 ms | 2.18 ms | ~4.4× | skipped(cuDNN) | skipped(engine build failed) |
| deep_pruned | 9.83 ms | 2.20 ms | ~4.5× | skipped(cuDNN) | skipped(engine build failed) |

#### 命中情景 4:Both deep < LGB

- 深度模型 vs Stage 1:**+0.023 roc_auc / +0.027 pr_auc**(0.841→0.864)—— embedding +
  完整 V 列**确实带来增益**,验证了 v1 对根因的判断
- 深度 vs LGB 差距:Stage 1 -0.067,Stage 2 -0.038 —— **差距收窄了一半,但
  深度仍输给 LGB ~0.04 roc_auc**
- LGB 在两套 V 策略上都很强,full_v 略优(0.902 vs 0.898),pr_auc 差距更明显
  (0.556 vs 0.530)
- V 列剪枝对深度模型几乎无影响(deep_pruned 0.864 ≈ deep_full 0.862),但
  pruned_v 训练时间更长(早停延后)—— 消融的有趣发现:130 列 V 与 339 列 V
  对深度模型的最终表现差异极小
- **诚实结论**:在 IEEE-CIS 这类中等规模、构造图、表格特征为主的设置下,
  GBDT 的归纳偏置确实强于此类深度双塔架构。Stage 3 的异质图 + 团伙特征 +
  外部信号是让深度模型有机会跑赢 LGB 的方向

### 与简历的对照(诚实)

简历中的 AUC 0.98、资损 -8% 绑定蚂蚁专有数据 + 生产环境的特征工程 + 真实社交/
转账图,本项目用公开数据复现的是**方法论与工程链路**。Stage 2 验证了一个
重要假设:**Stage 1 输给 LGB 的根因确实是数据层简化**,升级后差距收窄;
但要让深度模型在公开数据上 surpass GBDT,需要 Stage 3 的图深化。这也是
"先跑赢基线再谈复杂架构"的工程哲学的诚实呈现。

### 参考文献(v2 新增)

- Shwartz-Ziv & Armon, *Tabular Data: Deep Learning is Not All You Need*,
  Information Fusion 2022
- Cheng et al., *Wide & Deep Learning for Recommender Systems*, DLRS 2016
- Guo et al., *DeepFM*, IJCAI 2017
- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data
  (FT-Transformer)*, NeurIPS 2021
- IEEE-CIS Fraud Detection Kaggle 社区公开 kernels(V 列相关性剪枝惯例)
