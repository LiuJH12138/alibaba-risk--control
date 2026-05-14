import subprocess
from pathlib import Path
import pandas as pd

COMPETITION = "ieee-fraud-detection"


def download_raw(raw_dir: str) -> None:
    """用 kaggle CLI 下载并解压竞赛数据到 raw_dir。

    Note: kaggle CLI 2.x dropped the -c flag; competition is now a positional arg.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "train_transaction.csv").exists():
        print(f"raw data already present at {raw}")
        return
    subprocess.run(
        ["kaggle", "competitions", "download", COMPETITION, "-p", str(raw)],
        check=True,
    )
    subprocess.run(
        ["unzip", "-o", str(raw / f"{COMPETITION}.zip"), "-d", str(raw)],
        check=True,
    )


def join_transaction_identity(txn: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """按 TransactionID 左连接,保留所有交易。"""
    return txn.merge(idn, on="TransactionID", how="left")


def load_raw(raw_dir: str) -> pd.DataFrame:
    """加载 train_transaction + train_identity 并 join。"""
    raw = Path(raw_dir)
    txn = pd.read_csv(raw / "train_transaction.csv")
    idn = pd.read_csv(raw / "train_identity.csv")
    return join_transaction_identity(txn, idn)
