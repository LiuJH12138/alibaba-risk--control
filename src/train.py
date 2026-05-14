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
        labels.append(b["label"].cpu().numpy())
    return compute_metrics(np.concatenate(labels), np.concatenate(scores))

def train_one_config(graph, seq_all, split, fusion_mode, use_hnm,
                     model_cfg, train_cfg, device="cuda"):
    """训练单个配置并返回 val 指标。配置 = (fusion_mode, use_hnm, loss 参数)。"""
    _set_seed(train_cfg["seed"])
    feat_dim = graph.x.shape[1]
    model = FraudModel(feat_dim, model_cfg, fusion_mode=fusion_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    warmup = train_cfg["warmup_steps"]
    def _lr_lambda(step):
        return min(1.0, (step + 1) / max(1, warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
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
            scheduler.step()
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
