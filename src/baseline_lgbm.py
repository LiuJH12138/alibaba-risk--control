import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
import torch
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.preprocessing import FunctionTransformer

from src.evaluate import compute_metrics

def train_lgbm_baseline(x_train, y_train, x_val, y_val):
    """在扁平表特征上训 LightGBM。返回 (val 指标, 模型)。"""
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_val)[:, 1]
    return compute_metrics(y_val, scores), clf

def export_pmml(clf, path: str):
    """把 LightGBM 模型导出为 PMML(异构部署的轻量模型那一路)。"""
    pipe = PMMLPipeline([("identity", FunctionTransformer()), ("clf", clf)])
    sklearn2pmml(pipe, path)

def run_baseline():
    """用处理后的全量特征(取每笔交易当前步)训基线 + 导出 PMML。"""
    graph = torch.load("data/processed/graph.pt", weights_only=False)
    split = torch.load("data/processed/split.pt", weights_only=False)
    x = graph.x.numpy(); y = graph.y.numpy()
    tr, va = split["train_idx"].numpy(), split["val_idx"].numpy()
    metrics, clf = train_lgbm_baseline(x[tr], y[tr], x[va], y[va])
    print("lgbm baseline:", metrics)

    Path("experiments").mkdir(exist_ok=True)
    results = json.load(open("experiments/results.json")) \
        if Path("experiments/results.json").exists() else {}
    results["lgbm_baseline"] = metrics
    json.dump(results, open("experiments/results.json", "w"), indent=2)

    Path("artifacts").mkdir(exist_ok=True)
    export_pmml(clf, "artifacts/lgbm_baseline.pmml")
    print("PMML exported to artifacts/lgbm_baseline.pmml")

if __name__ == "__main__":
    run_baseline()
