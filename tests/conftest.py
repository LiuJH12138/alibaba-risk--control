import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def tiny_raw_df():
    """合成 ~500 行、结构同 IEEE-CIS 关键列的微数据集。"""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": (rng.random(n) < 0.04).astype(int),
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 30, n)),
        "TransactionAmt": rng.exponential(50, n),
        "ProductCD": rng.choice(list("ABCDE"), n),
        "card1": rng.integers(1000, 1050, n),
        "addr1": rng.integers(50, 60, n).astype(float),
        "D1": rng.integers(0, 100, n).astype(float),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "DeviceInfo": rng.choice(["iOS", "Windows", "Android"], n),
    })
    return df
