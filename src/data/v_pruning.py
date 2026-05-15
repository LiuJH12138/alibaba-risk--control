import pandas as pd


def compute_pruned_v_cols(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """对 V 列贪心剪枝:从前往后遍历,若与已保留列存在 |corr| >= threshold,丢弃。
    只用 train 数据调用(防泄漏)。返回保留列名列表。"""
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    if not v_cols:
        return []
    corr = df[v_cols].corr().abs()
    kept: list[str] = []
    for c in v_cols:
        if all(corr.loc[c, k] < threshold for k in kept):
            kept.append(c)
    return kept
