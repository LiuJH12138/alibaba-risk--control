import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def recall_at_fixed_fpr(y_true, y_score, fpr: float) -> float:
    """在给定最大 FPR 约束下能达到的最高召回(TPR)。"""
    fprs, tprs, _ = roc_curve(y_true, y_score)
    ok = fprs <= fpr
    return float(tprs[ok].max()) if ok.any() else 0.0

def fpr_at_fixed_recall(y_true, y_score, recall: float) -> float:
    """在保证至少 recall 召回时的最低 FPR(误伤率)。"""
    fprs, tprs, _ = roc_curve(y_true, y_score)
    ok = tprs >= recall
    return float(fprs[ok].min()) if ok.any() else 1.0

def compute_metrics(y_true, y_score) -> dict:
    """主指标集合:ROC-AUC / PR-AUC / KS / 工作点指标。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fprs, tprs, _ = roc_curve(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": float(np.max(tprs - fprs)),
        "recall_at_fpr_0.01": recall_at_fixed_fpr(y_true, y_score, 0.01),
        "fpr_at_recall_0.90": fpr_at_fixed_recall(y_true, y_score, 0.90),
    }
