import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.data.load import join_transaction_identity
from src.data.uid import synthesize_uid
from src.data.features import FeatureProcessor
from src.data.sequence import build_sequences
from src.data.graph import build_edges
from src.data.v_pruning import compute_pruned_v_cols

from src.data.build import time_split, validate_split

def test_load_config_returns_dict():
    cfg = load_config("data")
    assert isinstance(cfg, dict)
    assert cfg["seq_len"] > 0
    assert "raw_dir" in cfg and "processed_dir" in cfg

def test_load_config_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent")


def test_join_left_keeps_all_transactions():
    txn = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0],
                        "TransactionDT": [10, 20, 30], "TransactionAmt": [5.0, 6.0, 7.0]})
    idn = pd.DataFrame({"TransactionID": [2], "DeviceType": ["mobile"]})
    merged = join_transaction_identity(txn, idn)
    assert len(merged) == 3
    assert merged.loc[merged.TransactionID == 1, "DeviceType"].isna().all()
    assert merged.loc[merged.TransactionID == 2, "DeviceType"].iloc[0] == "mobile"


def test_uid_groups_same_card_addr():
    txn = pd.DataFrame({
        "card1": [1000, 1000, 2000],
        "addr1": [50.0, 50.0, 80.0],
        "TransactionDT": [86400, 172800, 86400],  # day1, day2, day1
        "D1": [0.0, 1.0, 0.0],                     # days since first txn
    })
    uid = synthesize_uid(txn)
    assert uid.iloc[0] == uid.iloc[1]   # same card same account
    assert uid.iloc[0] != uid.iloc[2]

def test_uid_handles_nan():
    txn = pd.DataFrame({"card1": [1000], "addr1": [float("nan")],
                        "TransactionDT": [86400], "D1": [float("nan")]})
    uid = synthesize_uid(txn)
    assert uid.notna().all()


def test_processor_fits_on_train_only():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    val = pd.DataFrame({"ProductCD": ["A", "C"], "TransactionAmt": [30.0, 40.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    tr = fp.transform(train); va = fp.transform(val)
    # dict 接口
    assert set(tr.keys()) == {"cat_idx", "num"}
    # 未见类别 "C" 映射到 0(unknown 桶)—— cat 现在是整数索引
    assert va["cat_idx"][1, 0] == 0
    # 数值标准化:train 列 0(TransactionAmt)均值 ≈ 0
    assert abs(tr["num"][:, 0].mean()) < 1e-6

def test_processor_meta_has_cardinalities():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    assert fp.meta["cat_cardinalities"]["ProductCD"] == 3
    assert fp.meta["num_cols"] == ["TransactionAmt"]

def test_processor_output_is_bounded():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "card1": np.arange(500),                      # 高基数 cat
        "TransactionAmt": rng.normal(0, 1, size=500),
    })
    fp = FeatureProcessor(cat_cols=["card1"], num_cols=["TransactionAmt"])
    fp.fit(train)
    out = fp.transform(train)
    # cat: 整数索引,在 [0, cardinality);本例 cardinality = 501
    assert out["cat_idx"].min() >= 0 and out["cat_idx"].max() < 501
    # num: 标准化 + 裁剪 [-10, 10]
    assert out["num"].min() >= -10.0 and out["num"].max() <= 10.0


def test_sequence_window_and_mask():
    # 2 个 uid,uid "x" 有 3 笔,uid "y" 有 1 笔
    feat = np.array([[1.0], [2.0], [3.0], [9.0]], dtype="float32")
    uid = np.array(["x", "x", "x", "y"])
    dt = np.array([10, 20, 30, 5])
    seq, mask = build_sequences(feat, uid, dt, seq_len=2)
    # 第 3 笔(uid x,dt=30)序列 = 前 2 笔 [feat0, feat1]? 不:含自身,窗口=自身+前1
    # 约定:位置 L-1 是当前交易,L-2..0 是更早的;不足则前端 padding
    assert seq.shape == (4, 2, 1)
    assert mask.shape == (4, 2)
    # uid y 只有 1 笔 → 位置 0 padding(mask False),位置 1 是自身
    yi = 3
    assert mask[yi].tolist() == [False, True]
    assert seq[yi, 1, 0] == 9.0
    # uid x 第 1 笔(dt=10)→ 位置 0 padding,位置 1 自身
    assert mask[0].tolist() == [False, True]


def test_edges_are_time_respecting_and_share_entity():
    # 4 笔交易,card1 列:0,1,3 共享 card1=100
    df = pd.DataFrame({
        "card1": [100, 100, 999, 100],
        "addr1": [-1, -1, -1, -1],
        "TransactionDT": [10, 20, 15, 30],
    })
    src, dst = build_edges(df, entity_cols=["card1"], max_degree=10, max_per_entity=10)
    edges = set(zip(src.tolist(), dst.tolist()))
    # 边方向:src 更早 → dst 更晚(time-respecting)
    for s, d in edges:
        assert df["TransactionDT"].iloc[s] < df["TransactionDT"].iloc[d]
    # 交易 0(dt10)→ 交易 1(dt20)应连(同 card1,0 更早)
    assert (0, 1) in edges
    # 交易 2(card1=999)不与任何人连
    assert all(2 not in e for e in edges)

def test_edges_skip_sentinel_entity():
    # addr1 = -1 是哨兵(缺失填充值),不应据此连边
    df = pd.DataFrame({"card1": [-1, -1], "addr1": [-1, -1], "TransactionDT": [10, 20]})
    src, dst = build_edges(df, entity_cols=["card1", "addr1"], max_degree=10, max_per_entity=10)
    assert len(src) == 0


def test_time_split_is_chronological():
    dt = np.array([5, 1, 9, 3, 7])   # 乱序时间戳
    train_idx, val_idx = time_split(dt, ratio=0.6)
    # train 应是最早的 3 个(dt 1,3,5),val 是最晚 2 个(dt 7,9)
    assert set(dt[train_idx].tolist()) == {1, 3, 5}
    assert set(dt[val_idx].tolist()) == {7, 9}
    # 无重叠
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_validate_split_rejects_leak():
    dt = np.array([1, 2, 3, 4])
    # 故意构造泄漏:train 含 dt=4,val 含 dt=1
    with pytest.raises(AssertionError):
        validate_split(dt, train_idx=np.array([0, 3]), val_idx=np.array([1, 2]))

def test_v_column_pruning_keeps_one_per_correlated_group():
    rng = np.random.default_rng(0)
    n = 200
    base1 = rng.normal(size=n)
    base2 = rng.normal(size=n)
    df = pd.DataFrame({
        "V1": base1,
        "V2": base1 + 0.005 * rng.normal(size=n),  # |corr|≈1 with V1
        "V3": base2,
        "V4": base2 + 0.005 * rng.normal(size=n),  # |corr|≈1 with V3
        "V5": rng.normal(size=n),                   # 独立
    })
    kept = compute_pruned_v_cols(df, threshold=0.95)
    assert kept == ["V1", "V3", "V5"]   # 贪心顺序保留首个代表


# ===== Stage 3a: entity stats (train-only computation, cold-start fallback) =====
import numpy as np
import pandas as pd
from src.data.entity_stats import compute_entity_stats, compute_all_entity_features


def test_entity_stats_train_only(tiny_raw_df):
    """entity_stats must depend only on train rows; mutating val rows must not change output."""
    df = tiny_raw_df.copy()
    n = len(df)
    train_idx = np.arange(0, int(n * 0.8))
    val_idx = np.arange(int(n * 0.8), n)

    stats_a = compute_entity_stats(df.iloc[train_idx], entity_col="card1",
                                   amt_col="TransactionAmt",
                                   dt_col="TransactionDT",
                                   label_col="isFraud")
    # Now scramble val labels AND val amounts: train-only stats must be unchanged
    df.loc[val_idx, "isFraud"] = 1 - df.loc[val_idx, "isFraud"]
    df.loc[val_idx, "TransactionAmt"] = df.loc[val_idx, "TransactionAmt"] * 1000.0
    stats_b = compute_entity_stats(df.iloc[train_idx], entity_col="card1",
                                   amt_col="TransactionAmt",
                                   dt_col="TransactionDT",
                                   label_col="isFraud")
    pd.testing.assert_frame_equal(stats_a, stats_b)


def test_cold_start_entity_fallback(tiny_raw_df):
    """val/test entities not in train must be filled with train-population mean (no NaN)."""
    df = tiny_raw_df.copy()
    n = len(df)
    train_idx = np.arange(0, int(n * 0.8))
    val_idx = np.arange(int(n * 0.8), n)
    # Inject a brand-new card1 value into val rows
    df.loc[val_idx[0], "card1"] = 999999

    feats = compute_all_entity_features(
        df=df, train_idx=train_idx, val_idx=val_idx,
        entity_cols=["card1", "addr1", "P_emaildomain", "DeviceInfo"],
        amt_col="TransactionAmt", dt_col="TransactionDT", label_col="isFraud",
    )
    # feats is dict[entity_col] -> dict {ids: list, x: np.ndarray [n_unique+1, 5]}
    card1_block = feats["card1"]
    assert "_COLD_" in card1_block["ids"], "cold-start sentinel must exist"
    assert not np.any(np.isnan(card1_block["x"])), "no NaN allowed in entity features"
    # The 999999 entity should map to the cold-start row
    cold_idx = card1_block["ids"].index("_COLD_")
    cold_vec = card1_block["x"][cold_idx]
    # cold-start vector must equal column means of train-entity vectors
    train_only_rows = np.array([card1_block["x"][i] for i, eid in enumerate(card1_block["ids"])
                                if eid != "_COLD_"])
    np.testing.assert_allclose(cold_vec, train_only_rows.mean(axis=0, dtype="float64").astype("float32"), rtol=1e-5)
