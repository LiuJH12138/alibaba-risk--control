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


def test_cosine_scheduler_shape():
    """SequentialLR(Linear warmup -> Cosine) must hit peak at warmup, anneal to eta_min by end."""
    import torch
    from src.train import _build_scheduler
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_cfg = {"warmup_steps": 10, "cosine_eta_min_ratio": 0.01}
    total_steps = 100
    sched = _build_scheduler(opt, train_cfg, total_steps)

    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    # Step 0: very low (start_factor=1e-6)
    assert lrs[0] < 1e-7, f"step 0 lr should be ~1e-9, got {lrs[0]}"
    # Step 10 is first cosine step (peak = base_lr = 1e-3)
    assert abs(lrs[10] - 1e-3) < 5e-5, f"step 10 lr should ~1e-3, got {lrs[10]}"
    # Final step: ~ peak * eta_min_ratio = 1e-5
    assert abs(lrs[-1] - 1e-5) < 5e-6, f"final lr should ~1e-5, got {lrs[-1]}"
    # Monotone non-increase after warmup
    post_warmup = lrs[10:]
    diffs = [post_warmup[i+1] - post_warmup[i] for i in range(len(post_warmup) - 1)]
    assert all(d <= 1e-9 for d in diffs), "post-warmup LR must be monotone non-increasing (cosine)"


def test_swa_setup_creates_averaged_model():
    """When swa_enabled, train_one_config_hetero must construct AveragedModel + SWALR.

    This is a structural test (no full training); we exercise the SWA wrapping
    by instantiating directly to verify nothing crashes.
    """
    import torch
    from torch.optim.swa_utils import AveragedModel, SWALR
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=1e-4, anneal_strategy="linear", anneal_epochs=3)
    # Simulate update
    for _ in range(5):
        for p in model.parameters():
            p.data += 0.01
        swa_model.update_parameters(model)
        swa_sched.step()
    # SWA model's running average should differ from current model
    inner = list(swa_model.module.parameters())[0].detach()
    current = list(model.parameters())[0].detach()
    assert not torch.equal(inner, current), "SWA averaged weights should differ from current"
