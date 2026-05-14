import pandas as pd
import pytest
from src.config import load_config

def test_load_config_returns_dict():
    cfg = load_config("data")
    assert isinstance(cfg, dict)
    assert cfg["seq_len"] > 0
    assert "raw_dir" in cfg and "processed_dir" in cfg

def test_load_config_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent")

from src.data.load import join_transaction_identity

def test_join_left_keeps_all_transactions():
    txn = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0],
                        "TransactionDT": [10, 20, 30], "TransactionAmt": [5.0, 6.0, 7.0]})
    idn = pd.DataFrame({"TransactionID": [2], "DeviceType": ["mobile"]})
    merged = join_transaction_identity(txn, idn)
    assert len(merged) == 3
    assert merged.loc[merged.TransactionID == 1, "DeviceType"].isna().all()
    assert merged.loc[merged.TransactionID == 2, "DeviceType"].iloc[0] == "mobile"

from src.data.uid import synthesize_uid

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

from src.data.features import FeatureProcessor

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

import numpy as np
from src.data.sequence import build_sequences

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
