import numpy as np
import torch
import torch.nn.functional as F
from src.models.losses import HybridFocalLoss, hard_negative_mining
from src.evaluate import compute_metrics, recall_at_fixed_fpr

def test_focal_reduces_to_bce_when_gamma_zero():
    logits = torch.tensor([0.5, -1.2, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = HybridFocalLoss(gamma_pos=0.0, gamma_neg=0.0, alpha=0.5)
    got = loss(logits, targets)
    expect = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(got, expect, atol=1e-5)

def test_asymmetric_gamma_changes_loss():
    logits = torch.tensor([0.1, 0.1])
    targets = torch.tensor([1.0, 0.0])
    sym = HybridFocalLoss(gamma_pos=2.0, gamma_neg=2.0, alpha=0.5)
    asym = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.5)
    assert not torch.allclose(sym(logits, targets), asym(logits, targets))

def test_loss_is_finite_on_extreme_logits():
    logits = torch.tensor([50.0, -50.0])
    targets = torch.tensor([0.0, 1.0])
    loss = HybridFocalLoss(gamma_pos=1.0, gamma_neg=4.0, alpha=0.25)
    assert torch.isfinite(loss(logits, targets))

def test_hnm_keeps_hardest_negatives():
    # 4 个负样本 loss 值 [0.1, 5.0, 0.2, 4.0],1 个正样本
    per_sample = torch.tensor([0.1, 5.0, 0.2, 4.0, 3.0])
    targets = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    keep = hard_negative_mining(per_sample, targets, neg_pos_ratio=2.0)
    # 1 个正样本 → 保留 2 个最难负样本(idx 1,3)+ 全部正样本(idx 4)
    assert keep.tolist() == [False, True, False, True, True]

def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_score)
    assert abs(m["roc_auc"] - 1.0) < 1e-6
    assert abs(m["pr_auc"] - 1.0) < 1e-6
    assert 0.0 <= m["ks"] <= 1.0

def test_recall_at_fixed_fpr():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    # FPR <= 0.34 时阈值可把 3 个负样本都判负,2 个正样本都判正 → recall 1.0
    r = recall_at_fixed_fpr(y_true, y_score, fpr=0.34)
    assert abs(r - 1.0) < 1e-6
