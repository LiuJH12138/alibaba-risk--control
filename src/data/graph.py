import numpy as np

SENTINEL = -1        # 整数列的缺失填充哨兵
STR_SENTINEL = "__MISSING__"  # 字符串列的缺失填充哨兵

def build_edges(df, entity_cols, max_degree, max_per_entity):
    """构造 time-respecting 同构交易图的边。
    两笔交易若共享某高区分度实体值,则连一条「早 → 晚」有向边。
    每个实体值内连边数封顶 max_per_entity;每个节点入边度数封顶 max_degree。
    返回 (src, dst) 两个 int64 ndarray。"""
    rng = np.random.default_rng(0)
    dt = df["TransactionDT"].to_numpy()
    src_list, dst_list = [], []
    for col in entity_cols:
        raw = df[col]
        # 根据列的实际值类型选择合适的哨兵,避免混合类型导致 argsort 失败
        if raw.dtype == object or str(raw.dtype) == "string":
            # 字符串列:fillna 用字符串哨兵,保证 argsort 可比较
            sentinel = STR_SENTINEL
            vals = raw.fillna(sentinel).astype(str).to_numpy()
        else:
            sentinel = SENTINEL
            vals = raw.fillna(sentinel).to_numpy()
        # 按实体值分组
        order = np.argsort(vals, kind="stable")
        sv = vals[order]
        gs = 0
        for i in range(1, len(sv) + 1):
            if i == len(sv) or sv[i] != sv[gs]:
                if sv[gs] != sentinel and i - gs > 1:
                    members = order[gs:i]
                    members = members[np.argsort(dt[members], kind="stable")]
                    if len(members) > max_per_entity:
                        members = np.sort(rng.choice(members, max_per_entity, replace=False))
                    # 同实体内:每个更晚交易连到所有更早交易
                    for a in range(len(members)):
                        for b in range(a):
                            src_list.append(members[b])  # 早
                            dst_list.append(members[a])  # 晚
                gs = i
    if not src_list:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
    src = np.array(src_list, dtype="int64")
    dst = np.array(dst_list, dtype="int64")
    # 入边度数封顶:每个 dst 最多保留 max_degree 条入边
    keep = np.ones(len(dst), dtype=bool)
    order = np.argsort(dst, kind="stable")
    sd = dst[order]
    gs = 0
    for i in range(1, len(sd) + 1):
        if i == len(sd) or sd[i] != sd[gs]:
            if i - gs > max_degree:
                drop = order[gs:i][max_degree:]
                keep[drop] = False
            gs = i
    return src[keep], dst[keep]
