import pandas as pd


def synthesize_uid(txn: pd.DataFrame) -> pd.Series:
    """社区标准 uid 代理:card1 + addr1 + (TransactionDay - D1)。
    这是启发式代理,非真实 ground truth(见 DESIGN_JOURNAL)。"""
    day = (txn["TransactionDT"] / 86400).astype("int64")
    anchor = (day - txn["D1"].fillna(-1)).astype("int64")
    uid = (
        txn["card1"].fillna(-1).astype("int64").astype(str)
        + "_" + txn["addr1"].fillna(-1).astype("int64").astype(str)
        + "_" + anchor.astype(str)
    )
    return uid.rename("uid")
