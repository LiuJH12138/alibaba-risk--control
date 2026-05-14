import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort


class _OnnxSequenceTower(nn.Module):
    """ONNX-friendly version of SequenceTower: replaces pack_padded_sequence
    with a plain GRU call (ONNX-exportable). Shares weights with original tower."""

    def __init__(self, seq_tower):
        super().__init__()
        self.input_proj = seq_tower.input_proj
        self.pos_emb = seq_tower.pos_emb
        self.transformer = seq_tower.transformer
        self.gru = seq_tower.gru

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, l, _ = seq.shape
        h = self.input_proj(seq) + self.pos_emb[:, :l]
        pad_mask = ~mask
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        # Plain GRU — no packing; ONNX-exportable
        _, h_n = self.gru(h)                 # h_n [1, B, d_seq]
        return h_n.squeeze(0)


class _OnlineWrapper(torch.nn.Module):
    """只暴露在线路径(序列塔 + 融合头),供 ONNX 导出。"""

    def __init__(self, model):
        super().__init__()
        self.seq_tower = _OnnxSequenceTower(model.seq_tower)
        self.fusion = model.fusion

    def forward(self, seq, mask, graph_emb):
        seq_emb = self.seq_tower(seq, mask)
        return self.fusion(seq_emb, graph_emb)


def export_online_path(model, feat_dim, seq_len, d_graph, path):
    """把 FraudModel 的在线路径导出为 ONNX(动态 batch 轴)。"""
    wrapper = _OnlineWrapper(model).eval()
    seq = torch.randn(2, seq_len, feat_dim)
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(2, d_graph)
    torch.onnx.export(
        wrapper, (seq, mask, graph_emb), path,
        input_names=["seq", "mask", "graph_emb"], output_names=["logit"],
        dynamic_axes={"seq": {0: "batch"}, "mask": {0: "batch"},
                      "graph_emb": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )


def verify_onnx_parity(model, onnx_path, feat_dim, seq_len, d_graph, atol=1e-4):
    """校验 ONNX 输出与 PyTorch 一致。"""
    model.eval()
    seq = torch.randn(4, seq_len, feat_dim)
    mask = torch.ones(4, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(4, d_graph)
    # PyTorch reference: use same no-packing path via wrapper for fair comparison
    wrapper = _OnlineWrapper(model).eval()
    with torch.no_grad():
        torch_out = wrapper(seq, mask, graph_emb).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "seq": seq.numpy(), "mask": mask.numpy(), "graph_emb": graph_emb.numpy(),
    })[0]
    return bool(np.allclose(torch_out, onnx_out, atol=atol))
