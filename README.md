# 阿里/蚂蚁风控算法组实习 —— 复刻项目(Stage 1)

在 IEEE-CIS 公开交易数据上复刻实习的技术方法:Transformer-GRU 序列塔 + GraphSAGE 图塔
→ 门控融合,Hybrid Focal Loss + Hard Negative Mining 训练,ONNX/TensorRT 部署 benchmark。

## 诚实声明

本项目用公开数据复现**方法论与改进方向**,非蚂蚁生产数据/环境。简历中的业务数字
(AUC 0.98、资损 -8%)绑定于专有数据,无法且不复现。本 README 报告的所有数字
均为本项目在 IEEE-CIS 上的真实结果 —— 包括不理想的结果。

---

## 环境

- **平台:** AutoDL 云 GPU,Ubuntu 22.04,NVIDIA RTX 5090(128GB 显存)
- **Python 环境:** conda env `dfer-riskctrl`,从 base 克隆,PyTorch 2.8+cu128
- **额外依赖(requirements.txt 之外,执行中安装):**
  - `torch-sparse`、`torch-scatter`(PyG NeighborSampler 后端)
  - `tensorrt 10.16`(TensorRT FP16 引擎编译)
  - `openjdk`(LightGBM PMML 导出,实际受版本兼容阻塞,见已知遗留项)
- **pip 使用国内镜像:**`-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 环境创建

```bash
conda create -n dfer-riskctrl --clone base
conda activate dfer-riskctrl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# PyG 稀疏后端(按 CUDA 版本选择对应 wheel):
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

---

## 运行

```bash
# 1. 构建数据(需先配置 Kaggle 凭证到 ~/.kaggle/,IEEE-CIS 数据集)
python -m src.data.build

# 2. 实验矩阵(5 配置架构与损失消融)
python -m src.train

# 3. LightGBM 基线 + PMML 导出
python -m src.baseline_lgbm

# 4. 延迟 benchmark(内部含 ONNX 导出 + TensorRT 引擎构建)
python -m src.deploy.benchmark

# 测试
pytest -v
```

结果写入 `experiments/results.json` 和 `experiments/benchmark.json`。

---

## 结果(IEEE-CIS 公开数据,真实数字)

数据集:590,540 笔交易,欺诈率 3.5%,训练集 472,432 / 验证集 118,108(时序切分),
特征维度 213,序列长度 32,图边数 819,861。

### 架构与损失消融(深度模型 + LightGBM 基线)

| 配置 | roc_auc | pr_auc | ks | recall@fpr=.01 | fpr@recall=.90 |
|---|---|---|---|---|---|
| seq_only | 0.8438 | 0.4211 | 0.5499 | 0.3632 | 0.5335 |
| graph_only | 0.8481 | 0.3911 | 0.5402 | 0.3376 | 0.4730 |
| concat_fusion | **0.8491** | **0.4241** | **0.5521** | **0.3652** | 0.4945 |
| gated_fusion | 0.8412 | 0.4041 | 0.5348 | 0.3521 | 0.5115 |
| gated_plus_hnm | 0.8187 | 0.3144 | 0.4849 | 0.2562 | 0.5444 |
| **lgbm_baseline** | **0.9076** | **0.4813** | **0.6555** | **0.4050** | **0.2759** |

### 延迟 benchmark(batch=1 单请求)

| 配置 | p50_ms | p95_ms | p99_ms | mean_ms |
|---|---|---|---|---|
| pytorch_cpu | 9.37 | 78.32 | 84.11 | 25.62 |
| pytorch_gpu | 1.33 | 1.36 | 1.38 | 1.34 |
| onnx_gpu | — | — | — | — |
| tensorrt_fp16 | — | — | — | — |

**onnx_gpu 跳过原因:** `CUDAExecutionProvider not active`;cuDNN/onnxruntime-gpu ABI 不兼容。

**tensorrt_fp16 跳过原因:** `TensorrtExecutionProvider not active`;同 cuDNN 问题。注意:TensorRT FP16 引擎**本身编译成功**(artifacts/online.engine,1.6MB)—— 仅 ORT 的 TensorRT 执行提供器无法加载。

---

## 结果解读(诚实分析)

- 所有深度配置 roc_auc 落在 0.82–0.85,pr_auc 0.31–0.42。**concat_fusion 最好(0.849)**,
  边际超过单塔(seq_only 0.844 / graph_only 0.848)—— 说明双塔**弱互补**:融合有增益但很小。
  这与设计预判一致(IEEE-CIS 是构造图,非原生图,信号中等)。

- **gated_fusion(0.841)没有超过 concat_fusion**,门控机制在本设置下未带来增益。

- **gated_plus_hnm(0.819)是所有配置里最差的** —— HNM 在本 Stage 1 设置下**反而有害**:
  roc_auc、pr_auc 全面下降,fpr@recall0.90 也更高(误伤更多,与"HNM 降误伤"的假设相反)。
  这是一个诚实的负面结果。

- **LightGBM 基线(roc_auc 0.908)反超所有深度配置。** 这是表格数据上的常见现象——梯度提升树
  在中等规模表格特征上常胜过深度模型,尤其在本 Stage 1 的简化条件下(类别用缩放序数编码、
  V 列削减至 V1-V50)。诚实记录:Stage 1 的深度双塔模型尚未跑赢强基线;要让深度模型体现价值,
  需要 Stage 2 的 proper embedding + 异质图 + 完整特征。

- 延迟:**pytorch_cpu p50 9.37ms → pytorch_gpu p50 1.33ms,约 7× 加速** —— 真实的
  before/after 对比。TensorRT FP16 引擎能成功编译(证明 TRT 编译链路通),但 onnxruntime-gpu
  的 CUDA/TensorRT 执行提供器受本环境 cuDNN ABI 不兼容阻塞,故 onnx_gpu / tensorrt_fp16
  的端到端延迟未测得 —— 诚实记录为环境限制,深度部署优化归 Stage 3。

- 与简历对照:简历的 AUC 0.98 / 资损 -8% / 150ms→45ms 是蚂蚁专有数据 + 生产环境的结果。
  本项目复现了**方法论与工程链路**(双塔架构、消融实验、Hybrid Loss、ONNX/TensorRT 部署),
  用的是公开数据上的诚实数字。Stage 1 证明了端到端管线可跑通;模型质量的提升是 Stage 2 的工作。

---

## 结果 → 简历 bullet 映射

| 简历 bullet | 对应实验 | 本项目真实结果 | 说明 |
|---|---|---|---|
| Transformer-GRU 行为序列 + GNN 团伙识别 | 架构消融 seq_only/graph_only/concat/gated | 双塔弱互补:concat 0.849 边际超过单塔 ~0.845;gated 未超 concat | IEEE-CIS 构造图信号中等,强图效果留 Stage 2 异质图 |
| Hybrid Focal Loss + HNM 处理极不平衡 | 损失消融 gated_fusion vs gated_plus_hnm | HNM 在本设置下有害(0.819 < 0.841);诚实负面结果 | Stage 1 简化设置下 HNM 有害,留 Stage 2 深化 |
| 离线 AUC 0.98 | 各配置 roc_auc | 深度模型 0.82–0.85,LightGBM 基线 0.91;非 0.98 | 0.98 来自蚂蚁专有数据;Stage 1 简化(类别缩放序数编码 + V 列削减) |
| PMML/TensorRT 异构部署、150ms→45ms | 4 档延迟 benchmark | pytorch CPU→GPU 7× 加速(9.4→1.3ms);TRT 引擎可编译;ORT-GPU EP 受环境 cuDNN 阻塞;PMML 导出待 Java 11+ | 延迟绝对值与业务场景不同;TRT FP16 链路已通,ORT EP 集成归 Stage 3 |

---

## 阶段

- **Stage 1**(已完成)— 一体化端到端 MVP ✅
- **Stage 2**(已完成)— 模型基础升级:per-field 类别 embedding + 完整 V 列 ✅
- Stage 3 — 异质图深化、团伙核心节点识别、生产化部署、损失深化、PMML/cuDNN 工具链

设计演进与执行中发现的问题见 `docs/DESIGN_JOURNAL.md`,完整设计见 `docs/superpowers/specs/`。

## Stage 2 结果(2026-05-15)

**新命令(在 Stage 1 命令基础上更新):**
```bash
python -m src.data.build              # 双轨数据(full_v + pruned_v)
python -m src.train                   # gated_fusion × 2 v_strategy(写 stage2_results.json)
python -m src.baseline_lgbm           # LGB × 2 v_strategy(append 到 stage2_results.json)
python -m src.deploy.benchmark        # 双模型延迟(写 benchmark_stage2.json)
```

**架构 + V 策略消融**(IEEE-CIS 公开数据)

| 配置 | roc_auc | pr_auc | ks | recall@fpr=.01 | fpr@recall=.90 |
|---|---|---|---|---|---|
| deep_full | 0.8621 | 0.4370 | 0.5731 | 0.3632 | 0.4526 |
| deep_pruned | **0.8639** | 0.4312 | 0.5637 | 0.3713 | 0.4584 |
| **lgbm_full** | **0.9016** | **0.5556** | **0.6475** | **0.4941** | **0.3432** |
| lgbm_pruned | 0.8980 | 0.5303 | 0.6416 | 0.4678 | 0.3651 |

**延迟 benchmark**(单笔 batch=1)

| 模型 | pytorch_cpu p50 | pytorch_gpu p50 | onnx_gpu | tensorrt_fp16 |
|---|---|---|---|---|
| deep_full | 9.65 ms | 2.18 ms (~4.4×) | skipped (cuDNN ABI) | skipped (engine build failed) |
| deep_pruned | 9.83 ms | 2.20 ms (~4.5×) | skipped (cuDNN ABI) | skipped (engine build failed) |

**结果解读(命中情景 4:Both deep < LGB)**

- 深度模型 vs Stage 1:**+0.023 roc_auc**(0.841→0.864)—— embedding + 完整 V 列
  **确实带来增益**,验证了 v1 对根因的判断
- 深度 vs LGB 差距:Stage 1 -0.067,Stage 2 -0.038 —— **差距收窄了一半,但
  深度仍输给 LGB ~0.04 roc_auc**
- V 列剪枝(130 列保留)对深度模型几乎无影响(deep_pruned 0.864 ≈ deep_full 0.862),
  但 LGB 上 full_v 略优 0.004
- **诚实结论**:在 IEEE-CIS 这类中等规模、构造图、表格特征为主的设置下,
  GBDT 的归纳偏置确实强于此类深度双塔架构。Stage 3 的异质图 + 团伙特征 +
  外部信号是让深度模型有机会跑赢 LGB 的方向

设计决策、实施 bug、完整诚实分析见 `docs/DESIGN_JOURNAL.md` v2 节。
