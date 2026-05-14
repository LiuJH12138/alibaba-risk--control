import numpy as np

def build_sequences(feat: np.ndarray, uid: np.ndarray, dt: np.ndarray, seq_len: int):
    """为每笔交易构造「同 uid、含自身、按时间倒推」的滑窗序列。
    返回 seq [N, seq_len, Fdim](位置 seq_len-1 为当前交易),mask [N, seq_len](True=有效)。
    不足 seq_len 的在前端 padding。"""
    n, fdim = feat.shape
    seq = np.zeros((n, seq_len, fdim), dtype="float32")
    mask = np.zeros((n, seq_len), dtype=bool)
    order = np.lexsort((dt, uid))           # 先 uid 再 dt 排序
    # 按 uid 分组遍历
    sorted_uid = uid[order]
    group_start = 0
    for i in range(1, n + 1):
        if i == n or sorted_uid[i] != sorted_uid[group_start]:
            idxs = order[group_start:i]      # 该 uid 的全局索引,已按 dt 升序
            for pos, gi in enumerate(idxs):
                lo = max(0, pos - seq_len + 1)
                window = idxs[lo:pos + 1]    # 含自身,最多 seq_len 笔
                k = len(window)
                seq[gi, seq_len - k:] = feat[window]
                mask[gi, seq_len - k:] = True
            group_start = i
    return seq, mask
