import gc
import json
import time
from pathlib import Path
import numpy as np
import torch

from src.config import load_config
from src.dataset import make_loader, make_hetero_loader
from src.models.fraud_model import FraudModel
from src.models.losses import HybridFocalLoss, hard_negative_mining, hard_negative_mining_with_diagnostics
from src.evaluate import compute_metrics


def _set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)


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


if __name__ == "__main__":
    run_stage2_matrix()
