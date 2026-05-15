import numpy as np
import torch
import onnxruntime as ort


class _OnlineWrapper(torch.nn.Module):
    """只暴露在线路径(mixer + 序列塔 + 融合头),供 ONNX 导出。"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, seq_cat, seq_num, mask, graph_emb):
        return self.model.forward_online(seq_cat, seq_num, mask, graph_emb)


def export_online_path(model, n_cat, n_num_total, seq_len, d_graph, path):
    """把 FraudModel 在线路径导出为 ONNX(动态 batch 轴)。"""
    wrapper = _OnlineWrapper(model).eval()
    seq_cat = torch.zeros(2, seq_len, n_cat, dtype=torch.long)
    seq_num = torch.randn(2, seq_len, n_num_total)
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(2, d_graph)
    torch.onnx.export(
        wrapper, (seq_cat, seq_num, mask, graph_emb), path,
        input_names=["seq_cat", "seq_num", "mask", "graph_emb"],
        output_names=["logit"],
        dynamic_axes={k: {0: "batch"}
                      for k in ["seq_cat", "seq_num", "mask", "graph_emb", "logit"]},
        opset_version=17,
    )


def verify_onnx_parity(model, onnx_path, n_cat, n_num_total, cat_cardinalities,
                       seq_len, d_graph, atol=1e-4):
    """校验 ONNX 输出与 PyTorch 一致(随机有效索引 + 随机数值)。"""
    model.eval()
    cards = torch.tensor(cat_cardinalities)
    seq_cat = torch.stack([torch.randint(0, int(cards[i]), (4, seq_len))
                           for i in range(n_cat)], dim=-1)
    seq_num = torch.randn(4, seq_len, n_num_total)
    mask = torch.ones(4, seq_len, dtype=torch.bool)
    graph_emb = torch.randn(4, d_graph)
    with torch.no_grad():
        torch_out = model.forward_online(seq_cat, seq_num, mask, graph_emb).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "seq_cat": seq_cat.numpy(), "seq_num": seq_num.numpy(),
        "mask": mask.numpy(), "graph_emb": graph_emb.numpy(),
    })[0]
    return bool(np.allclose(torch_out, onnx_out, atol=atol))
