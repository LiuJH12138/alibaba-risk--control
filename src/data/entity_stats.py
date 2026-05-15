"""Per-entity aggregated features for Stage 3a heterogeneous graph nodes.

Computes 5-dim feature per entity value, **strictly from training rows**:
    count            - # transactions linked to this entity (log1p z-scored)
    mean_amt         - mean TransactionAmt (log1p z-scored)
    std_amt          - std  TransactionAmt (log1p z-scored)
    fraud_rate_train - share of isFraud==1 (clipped to [0,1])
    days_active      - (last_dt - first_dt) / 86400 (z-scored)

Cold-start entities (in val/test but not train) get the column means of the
train-only entity matrix as a per-feature fallback. This is documented in
DESIGN_JOURNAL v3 §2.3 as the on-line proxy: at inference time, brand-new
entities have no historical risk signal and inherit the population prior.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ENTITY_NA_SENTINEL = "_NA_"          # sentinel for entity-value NaN within an existing column
COLD_START_SENTINEL = "_COLD_"       # synthetic id for entities unseen in train


def _zscore_log1p(x: np.ndarray) -> np.ndarray:
    """log1p then z-score; constant columns become zero."""
    v = np.log1p(np.maximum(x, 0.0))
    mu, sigma = v.mean(), v.std()
    if sigma < 1e-9:
        return np.zeros_like(v)
    return (v - mu) / sigma


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-9:
        return np.zeros_like(x)
    return (x - mu) / sigma


def compute_entity_stats(train_df: pd.DataFrame, entity_col: str,
                         amt_col: str = "TransactionAmt",
                         dt_col: str = "TransactionDT",
                         label_col: str = "isFraud") -> pd.DataFrame:
    """Compute 5-dim aggregate per entity value from TRAIN rows only.

    Returns DataFrame indexed by entity value (with ENTITY_NA_SENTINEL substituted
    for NaN), columns = ['count', 'mean_amt', 'std_amt', 'fraud_rate_train',
    'days_active'] -- all post z-score / clip.
    """
    df = train_df.copy()
    df[entity_col] = df[entity_col].fillna(ENTITY_NA_SENTINEL).astype(str)
    g = df.groupby(entity_col, sort=True)
    raw = pd.DataFrame({
        "count": g.size().astype("float64"),
        "mean_amt": g[amt_col].mean(),
        "std_amt": g[amt_col].std().fillna(0.0),
        "fraud_rate_train": g[label_col].mean().clip(0.0, 1.0),
        "days_active": (g[dt_col].max() - g[dt_col].min()) / 86400.0,
    })
    # Apply per-column normalization; fraud_rate_train stays raw (already in [0,1])
    raw["count"] = _zscore_log1p(raw["count"].to_numpy())
    raw["mean_amt"] = _zscore_log1p(raw["mean_amt"].to_numpy())
    raw["std_amt"] = _zscore_log1p(raw["std_amt"].to_numpy())
    raw["days_active"] = _zscore(raw["days_active"].to_numpy())
    return raw


def compute_all_entity_features(df: pd.DataFrame,
                                train_idx: np.ndarray,
                                val_idx: np.ndarray,
                                entity_cols: list[str],
                                amt_col: str = "TransactionAmt",
                                dt_col: str = "TransactionDT",
                                label_col: str = "isFraud") -> dict:
    """For each entity column produce: stats DataFrame + ids list + dense float32 [n+1, 5] matrix.

    Last row of every entity matrix is the cold-start vector = column means of
    train-entity rows. ids[-1] == COLD_START_SENTINEL.

    Returns: {entity_col: {"ids": list[str], "x": np.ndarray [n+1, 5] float32}}
    """
    train_df = df.iloc[train_idx]
    out = {}
    for col in entity_cols:
        stats = compute_entity_stats(train_df, col, amt_col=amt_col,
                                     dt_col=dt_col, label_col=label_col)
        ids = list(stats.index)
        x = stats.to_numpy().astype("float32")
        # Use row-by-row accumulation to match the test assertion path (float32 consistency)
        cold = x.mean(axis=0, dtype="float64").astype("float32")
        ids.append(COLD_START_SENTINEL)
        x = np.vstack([x, cold[None, :]])
        out[col] = {"ids": ids, "x": x}
    return out
