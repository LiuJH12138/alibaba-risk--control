import torch
import torch.nn as nn

class SequenceTower(nn.Module):
    """Transformer → GRU 序列塔。
    Transformer(浅,1-2 层)做跨步全局上下文混合;GRU 做带近因偏置的时序压缩。
    输入已是数值化特征向量 [B, L, feat_dim];输出 seq_emb [B, d_seq]。"""

    def __init__(self, feat_dim: int, d_model: int, n_heads: int,
                 n_layers: int, d_seq: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gru = nn.GRU(d_model, d_seq, batch_first=True)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq [B, L, feat_dim];mask [B, L](True = 有效)
        b, l, _ = seq.shape
        h = self.input_proj(seq) + self.pos_emb[:, :l]
        # Transformer 的 padding mask:True = 忽略
        pad_mask = ~mask
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        # padding 位置清零;数据为前向 padding(真实数据在末尾),
        # plain GRU 顺序处理整个序列,h_n 即末位置(当前交易)的隐状态。
        # 不用 pack_padded_sequence:它假设 padding 在末尾,与本项目的前向 padding 不符,
        # 且 pack 不可 ONNX 导出。
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        _, h_n = self.gru(h)                 # h_n [1, B, d_seq]
        return h_n.squeeze(0)
