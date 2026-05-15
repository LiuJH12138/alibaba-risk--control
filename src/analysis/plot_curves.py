"""Stage 3a training-curve generator.

Produces a 3-subplot PNG with a red vertical line marking `best_epoch`
(the epoch with the highest val_pr_auc). Used after every training run.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # headless backend (no DISPLAY on AutoDL)
import matplotlib.pyplot as plt


def plot_curves(history_json_path: str, out_png: str) -> None:
    history = json.loads(Path(history_json_path).read_text())
    if not history:
        raise ValueError(f"empty history at {history_json_path}")
    epochs = [h["epoch"] for h in history]
    best_epoch = max(history, key=lambda h: h["val_pr_auc"])["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Subplot 1: train loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    axes[0].axvline(best_epoch, color="red", linestyle="--",
                    label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Train loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Subplot 2: PR-AUC + ROC-AUC
    axes[1].plot(epochs, [h["val_pr_auc"] for h in history], label="val PR-AUC", color="C0")
    axes[1].plot(epochs, [h["val_roc_auc"] for h in history], label="val ROC-AUC", color="C1")
    axes[1].axvline(best_epoch, color="red", linestyle="--")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Subplot 3: learning-rate schedule
    axes[2].plot(epochs, [h["lr"] for h in history], label="lr")
    axes[2].axvline(best_epoch, color="red", linestyle="--")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("lr")
    axes[2].set_title("Learning rate"); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(Path(history_json_path).stem)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
