import json
import pickle
from pathlib import Path
import numpy as np
import lightgbm as lgb
import torch
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.preprocessing import FunctionTransformer

from src.evaluate import compute_metrics


def flatten_for_lgbm(graph_data) -> tuple[np.ndarray, list[int]]:
    """把 graph 的 cat_x + num_x 拍扁回单矩阵给 LGB。
    返回 (X [N, n_cat + n_num_total], categorical_feature_indices)。"""
    cat = graph_data.cat_x.numpy()
    num = graph_data.num_x.numpy()
    X = np.concatenate([cat, num], axis=1).astype("float32")
    cat_idx = list(range(cat.shape[1]))    # 前 n_cat 列是类别
    return X, cat_idx


def train_lgbm_baseline(x_train, y_train, x_val, y_val, categorical_feature=None):
    """在扁平表特征上训 LightGBM。"""
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        class_weight="balanced",
    )
    fit_kwargs = {"categorical_feature": categorical_feature} if categorical_feature is not None else {}
    clf.fit(x_train, y_train, **fit_kwargs)
    scores = clf.predict_proba(x_val)[:, 1]
    return compute_metrics(y_val, scores), clf


def export_pmml(clf, path: str):
    pipe = PMMLPipeline([("identity", FunctionTransformer()), ("clf", clf)])
    sklearn2pmml(pipe, path)


def run_baseline(v_strategy: str):
    """v_strategy ∈ {full_v, pruned_v}。结果合并入 stage2_results.json,模型存 .pkl。"""
    proc_dir = Path("data/processed") / v_strategy
    graph = torch.load(proc_dir / "graph.pt", weights_only=False)
    split = torch.load(proc_dir / "split.pt", weights_only=False)
    X, cat_idx = flatten_for_lgbm(graph)
    y = graph.y.numpy()
    tr, va = split["train_idx"].numpy(), split["val_idx"].numpy()
    metrics, clf = train_lgbm_baseline(X[tr], y[tr], X[va], y[va],
                                       categorical_feature=cat_idx)
    metrics["v_strategy"] = v_strategy
    name = f"lgbm_{v_strategy.replace('_v', '')}"   # lgbm_full, lgbm_pruned
    print(f"{name}: {metrics}")

    # 增量更新 stage2_results.json
    Path("experiments").mkdir(exist_ok=True)
    out_path = Path("experiments/stage2_results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results[name] = metrics
    out_path.write_text(json.dumps(results, indent=2))

    # 存模型 + 尝试 PMML(失败不致命)
    Path("artifacts").mkdir(exist_ok=True)
    with open(f"artifacts/best_lgbm_{v_strategy.replace('_v', '')}.pkl", "wb") as f:
        pickle.dump(clf, f)
    try:
        export_pmml(clf, f"artifacts/lgbm_baseline_{v_strategy.replace('_v', '')}.pmml")
        print("PMML exported")
    except Exception as e:
        print(f"PMML export skipped (Java toolchain): {e}")


if __name__ == "__main__":
    for s in ["full_v", "pruned_v"]:
        run_baseline(s)
