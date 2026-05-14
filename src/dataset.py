import torch
from torch_geometric.loader import NeighborLoader

def make_loader(graph, seq_all, node_idx, batch_size, neighbor_sample,
                shuffle=True):
    """用 NeighborLoader 驱动 batch:每个 batch 是以一组 seed 交易为中心的采样子图。
    seq/mask/label 按 seed 的原始 node id 从侧表查询,保证对齐。
    yield dict: x, edge_index, seed_local, seq, mask, label。"""
    seq_t = seq_all["seq"]
    mask_t = seq_all["mask"]
    y = graph.y

    base = NeighborLoader(
        graph, num_neighbors=neighbor_sample, input_nodes=node_idx,
        batch_size=batch_size, shuffle=shuffle,
    )

    class _Wrapped:
        def __init__(self, loader):
            self.loader = loader
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            for b in self.loader:
                bs = b.batch_size
                seed_global = b.n_id[:bs]          # seed 的原始 node id
                yield {
                    "x": b.x,
                    "edge_index": b.edge_index,
                    "seed_local": torch.arange(bs),  # NeighborLoader 把 seed 排在前 bs 个
                    "seq": seq_t[seed_global],
                    "mask": mask_t[seed_global],
                    "label": y[seed_global],
                }

    return _Wrapped(base)
