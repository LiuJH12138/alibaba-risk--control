import torch
import torch.nn as nn

class EmbeddingMixer(nn.Module):
    """把 {cat_idx, num} 字典转成统一 [..., feat_dim_unified] 张量。
    每类别字段一个独立 nn.Embedding;num 直通;最终拼接。
    形状无关:同时支持序列输入 [B, L, n_cat] 和图输入 [N, n_cat]。"""

    def __init__(self, cat_cardinalities, cat_emb_dim: int, n_num_total: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(c), cat_emb_dim) for c in cat_cardinalities]
        )
        self.cat_emb_dim = cat_emb_dim
        self.n_num_total = n_num_total
        self.out_dim = len(cat_cardinalities) * cat_emb_dim + n_num_total

    def forward(self, cat_idx: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        # cat_idx: [..., n_cat] long;  num: [..., n_num_total] float
        embs = [emb(cat_idx[..., i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embs, dim=-1)         # [..., n_cat * cat_emb_dim]
        return torch.cat([cat_out, num], dim=-1)  # [..., out_dim]
