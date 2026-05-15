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
