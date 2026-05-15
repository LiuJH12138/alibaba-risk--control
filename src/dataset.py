import torch
from torch_geometric.loader import NeighborLoader

def make_loader(graph, seq_all, node_idx, batch_size, neighbor_sample, shuffle=True):
    """Stage 1/2 homogeneous NeighborLoader. Yields 8-key dict:
       x_cat / x_num / edge_index / seed_local / seq_cat / seq_num / mask / label."""
    seq_cat_t = seq_all["cat"]
    seq_num_t = seq_all["num"]
    mask_t = seq_all["mask"]
    y = graph.y

    base = NeighborLoader(graph, num_neighbors=neighbor_sample, input_nodes=node_idx,
                          batch_size=batch_size, shuffle=shuffle)

    class _Wrapped:
        def __init__(self, loader): self.loader = loader
        def __len__(self): return len(self.loader)
        def __iter__(self):
            for b in self.loader:
                bs = b.batch_size
                seed_global = b.n_id[:bs]
                yield {
                    "x_cat": b.cat_x,
                    "x_num": b.num_x,
                    "edge_index": b.edge_index,
                    "seed_local": torch.arange(bs),
                    "seq_cat": seq_cat_t[seed_global],
                    "seq_num": seq_num_t[seed_global],
                    "mask": mask_t[seed_global],
                    "label": y[seed_global],
                }

    return _Wrapped(base)


def make_hetero_loader(hetero_graph, seq_all, node_idx, batch_size, neighbor_sample,
                      shuffle=True):
    """Stage 3a heterogeneous NeighborLoader. Seeds are transaction nodes only.

    `neighbor_sample` is a list (e.g. [15, 10]); the same fan-out is applied to
    every relation type. Yields a dict with the 7 per-transaction keys plus
    `hetero_data` (the sampled HeteroData subgraph) and `seed_local` (positions
    of the seed transactions inside the subgraph's transaction node block).
    """
    seq_cat_t = seq_all["cat"]
    seq_num_t = seq_all["num"]
    mask_t = seq_all["mask"]
    y = hetero_graph["transaction"].y

    # Apply uniform fan-out across all relations of the hetero graph
    fanout = {et: list(neighbor_sample) for et in hetero_graph.edge_types}
    base = NeighborLoader(
        hetero_graph,
        num_neighbors=fanout,
        input_nodes=("transaction", node_idx),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    class _Wrapped:
        def __init__(self, loader): self.loader = loader
        def __len__(self): return len(self.loader)
        def __iter__(self):
            for b in self.loader:
                bs = b["transaction"].batch_size
                seed_global = b["transaction"].n_id[:bs]
                yield {
                    "hetero_data": b,
                    "seed_local": torch.arange(bs),
                    "seq_cat": seq_cat_t[seed_global],
                    "seq_num": seq_num_t[seed_global],
                    "mask": mask_t[seed_global],
                    "label": y[seed_global],
                }

    return _Wrapped(base)
