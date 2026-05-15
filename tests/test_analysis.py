"""Stage 3a analysis-layer unit tests (curve plotting + centrality)."""
import json
from pathlib import Path
import pytest


def test_plot_curves_generates_png(tmp_path):
    from src.analysis.plot_curves import plot_curves
    history = [
        {"epoch": e, "lr": 1e-3, "train_loss": 1.0 / e, "epoch_seconds": 5.0,
         "val_roc_auc": 0.8 + 0.01 * e, "val_pr_auc": 0.5 + 0.02 * e,
         "val_ks": 0.6 + 0.01 * e, "val_recall_at_fpr_0.01": 0.4 + 0.01 * e}
        for e in range(1, 11)
    ]
    h_path = tmp_path / "training_history_dummy.json"
    h_path.write_text(json.dumps(history))
    out_png = tmp_path / "curves_dummy.png"
    plot_curves(str(h_path), str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 5000, \
        f"expected PNG > 5KB, got {out_png.stat().st_size if out_png.exists() else 'missing'}"
