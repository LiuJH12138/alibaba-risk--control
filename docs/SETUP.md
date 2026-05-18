# 30 分钟快速环境搭建 + 跑通(SETUP.md)

> **这份文档定位**:从零开始,**最快路径**让你在自己机器上跑通这个项目至少一次。详细学习见 `LEARNING.md`。

## 0. 前提

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux(Ubuntu 20.04+ 推荐)/ macOS / WSL |
| GPU(强烈推荐) | NVIDIA GPU,显存 ≥ 12GB(否则只能跑 LightGBM 基线) |
| CUDA | 12.8(本项目 PyTorch 2.8 默认),其他 CUDA 版本需相应改 PyG wheel index |
| Python | 3.12(项目用 conda env 锁版本) |
| 内存 | ≥ 32GB(seq_all.pt 在 full_v 模式下 30GB) |
| 磁盘 | ≥ 100GB(数据 + checkpoints + 30GB seq 文件) |
| Kaggle 账号 | 用于下载 IEEE-CIS 数据(免费) |

**没有 GPU 怎么办?** 可以用 [AutoDL](https://www.autodl.com)(本项目就在 AutoDL 跑的),按小时租 RTX 3090 / 4090 / A40,每小时 ¥1-3。

## 1. 克隆仓库

```bash
git clone https://github.com/LiuJH12138/alibaba-risk--control.git
cd alibaba-risk--control
git checkout feature/stage3a-hetero-graph    # 最完整的分支
git log --oneline -5                          # 应该看到 df65587 或更新的 commit
```

## 2. Conda 环境

```bash
# Miniconda 安装(略,见 https://docs.conda.io/projects/miniconda/)

conda env create -f environment.yml
conda activate dfer-riskctrl
```

`environment.yml` 里锁了 Python 3.12 + PyTorch 2.8 + CUDA 12.8。

## 3. PyTorch + PyG(版本敏感)

```bash
# 如果 environment.yml 没自动装,手动装:
pip install torch==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

**验证**:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
```

## 4. 其余 Python 依赖

```bash
pip install -r requirements.txt   # pandas, sklearn, lightgbm, networkx, matplotlib, pytest 等
```

## 5. Kaggle 数据下载

### 5.1 获取 Kaggle API token

1. 登录 [https://www.kaggle.com](https://www.kaggle.com)
2. 右上角头像 → Settings → API → "Create New API Token"
3. 下载 `kaggle.json`
4. 放到 `~/.kaggle/kaggle.json`,权限 `chmod 600 ~/.kaggle/kaggle.json`

### 5.2 接受比赛规则(必需!)

去 [https://www.kaggle.com/competitions/ieee-fraud-detection/rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) 点 "Late Submission" 接受规则,才能下载数据。

### 5.3 下载

```bash
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip ieee-fraud-detection.zip && cd ../..

# 验证文件
ls -lh data/raw/
# 应该有:train_transaction.csv (~700MB), train_identity.csv (~25MB),
#         test_transaction.csv, test_identity.csv, sample_submission.csv
```

## 6. 验证测试套件能跑

```bash
pytest tests/ -q
# 期望:55 passed
```

## 7. 跑最小可行流程(LightGBM,~30 min,不需 GPU)

```bash
# 构造数据(包含 graph.pt + seq_all.pt + hetero_graph.pt)
python -m src.data.build

# 跑 LightGBM 基线
python -m src.baseline_lgbm

# 看结果
cat experiments/stage2_results.json | python -m json.tool
# 应该有 lgbm_full + lgbm_pruned 两组指标
# lgbm_full: roc_auc ~0.90, pr_auc ~0.56
```

## 8. 跑深度学习 + ensemble(~6 小时,需 GPU)

```bash
# Stage 2 深度模型(deep_full + deep_pruned)
python -m src.train

# Stage 3a v1 异质图(4 配置)
python -c "from src.train import run_stage3a_matrix; run_stage3a_matrix()"

# Stage 3a v2 训练策略消融(6 变种)
python -c "from src.train import run_stage3a_matrix, STAGE3A_V2_CONFIGS; \
  run_stage3a_matrix(configs=STAGE3A_V2_CONFIGS)"

# Stage 3a v3 GATv2 + SWA(2 变种)
python -c "from src.train import run_stage3a_matrix, STAGE3A_V3_CONFIGS; \
  run_stage3a_matrix(configs=STAGE3A_V3_CONFIGS)"

# Ensemble 实验(无需 GPU 重训,加载 checkpoints + LGB 融合)
# 参照 docs/LEARNING.md B.8 节的代码
```

## 9. 后续

- 完整学习见 `docs/LEARNING.md`(112KB,Part A-E)
- 演进历史见 `docs/DESIGN_JOURNAL.md`(50KB,v1-v3.3)
- 实验结果见 `experiments/`
- Spec/Plan 见 `docs/superpowers/`

## 常见问题

见 `docs/LEARNING.md` D.2 节(8 个高频问题排坑)。

---

**有问题?** 先看 `LEARNING.md` D.2 排坑节,再开 GitHub issue。
