import numpy as np
import pandas as pd

# IEEE-CIS 字段分组(已知 schema)
CAT_COLS = (
    ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
     "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["DeviceType", "DeviceInfo"]
    + [f"id_{i:02d}" for i in range(12, 39)]
)
NUM_COLS = (
    ["TransactionAmt", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 340)]
)

class FeatureProcessor:
    """类别字段 → 整数编码(0 = unknown 桶);数值字段 → 标准化 + 缺失指示位。
    所有统计量只在 train 上 fit。"""

    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = list(cat_cols) if cat_cols is not None else list(CAT_COLS)
        self.num_cols = list(num_cols) if num_cols is not None else list(NUM_COLS)
        self._cat_maps = {}      # col -> {value: int>=1}
        self._num_mean = {}
        self._num_std = {}
        self.meta = {}

    def fit(self, df: pd.DataFrame) -> "FeatureProcessor":
        for c in self.cat_cols:
            vals = df[c].astype(str).fillna("nan").unique()
            self._cat_maps[c] = {v: i + 1 for i, v in enumerate(sorted(vals))}
        for c in self.num_cols:
            col = df[c].astype("float64")
            self._num_mean[c] = float(col.mean()) if col.notna().any() else 0.0
            std = float(col.std()) if col.notna().any() else 1.0
            self._num_std[c] = std if std > 1e-8 else 1.0
        self.meta = {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_cardinalities": {c: len(m) + 1 for c, m in self._cat_maps.items()},
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.cat_cols:
            m = self._cat_maps[c]
            out[c] = df[c].astype(str).fillna("nan").map(m).fillna(0).astype("int64")
        for c in self.num_cols:
            col = df[c].astype("float64")
            out[c] = ((col - self._num_mean[c]) / self._num_std[c]).fillna(0.0)
            out[f"{c}__isna"] = col.isna().astype("float32")
        return out
