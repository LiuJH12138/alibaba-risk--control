import numpy as np
import torch
import onnxruntime as ort

class _OnlineWrapper(torch.nn.Module):
    """只暴露在线路径(序列塔 + 融合头),供 ONNX 导出。"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, seq, mask, graph_emb):
        return self.model.forward_online(seq, mask, graph_emb)

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
    with torch.no_grad():
        torch_out = model.forward_online(seq, mask, graph_emb).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "seq": seq.numpy(), "mask": mask.numpy(), "graph_emb": graph_emb.numpy(),
    })[0]
    return bool(np.allclose(torch_out, onnx_out, atol=atol))
