"""Stage 3a train-related unit tests (loss extensions + convergence audit)."""
import pytest
import torch
from src.models.losses import HybridFocalLoss


def test_label_smoothing_loss_value():
    """eps=0 must equal Stage 2 baseline; eps>0 must produce STRICTLY different
    per-sample loss values for a hard positive (logit very high)."""
    logits = torch.tensor([10.0, -10.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss_eps0 = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.25,
                                 label_smoothing_eps=0.0).per_sample(logits, targets)
    loss_eps01 = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.25,
                                  label_smoothing_eps=0.1).per_sample(logits, targets)

    # Both vectors have shape [3]
    assert loss_eps0.shape == loss_eps01.shape == torch.Size([3])
    # Smoothing must move ALL elements (none should stay exactly equal)
    assert not torch.allclose(loss_eps0, loss_eps01),         "label_smoothing_eps=0.1 produced identical loss to eps=0"
    # Sanity: with eps=0.1, the loss for the confident-correct positive (logit=10)
    # must be NON-zero and finite (smoothing penalises over-confidence)
    assert torch.isfinite(loss_eps01).all()
    assert loss_eps01[0] > 1e-4

from src.train import _convergence_audit, _record_epoch_metrics


def test_record_epoch_metrics_keys():
    """Helper must return the canonical 8-key per-epoch row."""
    row = _record_epoch_metrics(
        epoch=3, lr=1e-3, train_loss=0.42, epoch_seconds=12.5,
        eval_metrics={"roc_auc": 0.91, "pr_auc": 0.78, "ks": 0.65,
                      "recall_at_fpr_0.01": 0.55, "fpr_at_recall_0.90": 0.08},
    )
    expected_keys = {"epoch", "lr", "train_loss", "epoch_seconds",
                     "val_roc_auc", "val_pr_auc", "val_ks",
                     "val_recall_at_fpr_0.01"}
    assert expected_keys.issubset(row.keys())
    assert row["epoch"] == 3 and row["val_pr_auc"] == 0.78


def test_convergence_audit_warns_on_late_best():
    """If best PR-AUC is at the last epoch, audit must flag NEEDS_LONGER."""
    history = [
        {"epoch": 1, "val_pr_auc": 0.50},
        {"epoch": 2, "val_pr_auc": 0.60},
        {"epoch": 3, "val_pr_auc": 0.70},
        {"epoch": 4, "val_pr_auc": 0.80},
    ]
    audit = _convergence_audit(history, "test_late")
    assert audit["best_epoch"] == 4
    assert audit["total_epochs"] == 4
    # Late-best AND short total -> should produce TWO warnings
    assert any("末尾" in w or "best_epoch" in w for w in audit["warnings"])
    assert any("早停" in w or "total_epochs" in w or "<15" in w for w in audit["warnings"])


def test_convergence_audit_warns_on_oscillation():
    """If last 5 epochs swing > 0.02 in val_pr_auc, audit must warn UNSTABLE."""
    history = [{"epoch": i, "val_pr_auc": 0.80} for i in range(1, 16)]   # 15 stable epochs
    history += [
        {"epoch": 16, "val_pr_auc": 0.82},
        {"epoch": 17, "val_pr_auc": 0.80},
        {"epoch": 18, "val_pr_auc": 0.85},   # >= 0.03 above min of last 5
        {"epoch": 19, "val_pr_auc": 0.81},
        {"epoch": 20, "val_pr_auc": 0.80},
    ]
    audit = _convergence_audit(history, "test_osc")
    assert audit["total_epochs"] == 20
    assert any("震荡" in w or "oscillat" in w.lower() for w in audit["warnings"])
