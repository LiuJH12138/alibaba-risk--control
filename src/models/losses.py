import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFocalLoss(nn.Module):
    """Asymmetric Focal Loss with optional label smoothing.

    Args:
        gamma_pos / gamma_neg: focal exponents for the positive / negative class
        alpha:                 positive-class weight (Stage 1/2 = 0.25)
        label_smoothing_eps:   Stage 3a addition; when > 0, targets are smoothed
                               t' = t*(1-eps) + 0.5*eps before BCE/focal compute.
                               eps=0 reproduces Stage 2 behavior bit-for-bit.
        reduction:             'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 alpha: float = 0.25, label_smoothing_eps: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.eps = label_smoothing_eps
        self.reduction = reduction

    def per_sample(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing on the binary targets before BCE
        if self.eps > 0.0:
            t = targets * (1.0 - self.eps) + 0.5 * self.eps
        else:
            t = targets
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        # Use the (post-smoothing) target for p_t / weighting so the formula stays consistent
        p_t = p * t + (1 - p) * (1 - t)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        alpha_t = 2 * self.alpha * targets + 2 * (1 - self.alpha) * (1 - targets)
        return alpha_t * (1 - p_t).clamp(min=1e-6) ** gamma * bce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.per_sample(logits, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def hard_negative_mining(per_sample_loss: torch.Tensor, targets: torch.Tensor,
                         neg_pos_ratio: float) -> torch.Tensor:
    """OHEM: keep all positives + K hardest negatives (K = ratio * n_pos).
    Returns a bool mask; True = included in backward pass."""
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    keep = pos_mask.clone()
    k = min(int(neg_pos_ratio * max(n_pos, 1)), int(neg_mask.sum().item()))
    if k > 0:
        neg_losses = per_sample_loss.masked_fill(pos_mask, float("-inf"))
        hardest = torch.topk(neg_losses, k).indices
        keep[hardest] = True
    return keep

def hard_negative_mining_with_diagnostics(per_sample_loss: torch.Tensor,
                                          targets: torch.Tensor,
                                          neg_pos_ratio: float,
                                          probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Same selection logic as hard_negative_mining, plus a diagnostics dict.

    Args:
        per_sample_loss: [B] per-sample loss values (already detached)
        targets:         [B] binary labels (float)
        neg_pos_ratio:   how many hard negatives to keep per positive
        probs:           [B] predicted P(fraud), used for diagnostics ONLY

    Returns:
        keep:        [B] bool mask, True = participate in backward
        diagnostics: dict with keys
            n_pos, n_neg, n_kept_neg,
            mean_prob_kept_neg, mean_prob_dropped_neg,
            max_prob_dropped_neg
    """
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    keep = pos_mask.clone()
    k = min(int(neg_pos_ratio * max(n_pos, 1)), n_neg)
    if k > 0:
        neg_losses = per_sample_loss.masked_fill(pos_mask, float("-inf"))
        hardest = torch.topk(neg_losses, k).indices
        keep[hardest] = True
    n_kept_neg = int((keep & neg_mask).sum().item())
    dropped_neg_mask = neg_mask & ~keep
    diagnostics = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_kept_neg": n_kept_neg,
        "mean_prob_kept_neg": float(probs[keep & neg_mask].mean().item()) if n_kept_neg > 0 else 0.0,
        "mean_prob_dropped_neg": float(probs[dropped_neg_mask].mean().item()) if dropped_neg_mask.any() else 0.0,
        "max_prob_dropped_neg": float(probs[dropped_neg_mask].max().item()) if dropped_neg_mask.any() else 0.0,
    }
    return keep, diagnostics
