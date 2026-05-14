import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.data.load import join_transaction_identity
from src.data.uid import synthesize_uid
from src.data.features import FeatureProcessor
from src.data.sequence import build_sequences
from src.data.graph import build_edges


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
    tr = fp.transform(train)
    va = fp.transform(val)
    # 未见类别 "C" 映射到 0(unknown 桶),不报错
    assert va["ProductCD"].iloc[1] == 0
    # 数值标准化用 train 统计量:train 均值处 ~0
    assert abs(tr["TransactionAmt"].mean()) < 1e-6

def test_processor_meta_has_cardinalities():
    train = pd.DataFrame({"ProductCD": ["A", "B"], "TransactionAmt": [10.0, 20.0]})
    fp = FeatureProcessor(cat_cols=["ProductCD"], num_cols=["TransactionAmt"])
    fp.fit(train)
    # 基数 = 不同类别数 + 1(unknown 桶)
    assert fp.meta["cat_cardinalities"]["ProductCD"] == 3
    assert fp.meta["num_cols"] == ["TransactionAmt"]


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

def test_processor_output_is_bounded():
    # 高基数类别列(如 IEEE-CIS card1)的原始整数编码不能泄漏进特征矩阵
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "card1": np.arange(500),                      # 500 个不同值 → 基数 501
        "TransactionAmt": rng.normal(0, 1, size=500),
    })
    fp = FeatureProcessor(cat_cols=["card1"], num_cols=["TransactionAmt"])
    fp.fit(train)
    out = fp.transform(train)
    # 所有输出列必须有界 —— 不能有原始大整数编码漏出
    assert out.abs().to_numpy().max() <= 11.0, f"unbounded feature: max abs = {out.abs().to_numpy().max()}"
