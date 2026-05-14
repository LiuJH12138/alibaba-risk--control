import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFocalLoss(nn.Module):
    """非对称 Focal Loss:正负样本用不同 gamma。
    gamma_pos/gamma_neg 控制对易例的压制强度,alpha 是正类权重。
    gamma_pos=gamma_neg=0 时退化为 BCE(alpha=0.5)。数值稳定:基于 logits。"""

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction

    def per_sample(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # p = sigmoid(logits);p_t = 目标类概率
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
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
    """OHEM:保留全部正样本 + loss 最高的 K 个负样本(K = ratio * 正样本数)。
    返回 bool mask,True = 参与反向。"""
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
