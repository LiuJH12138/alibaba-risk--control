import json
import time
from pathlib import Path
import numpy as np
import torch


def _percentiles(times_ms):
    arr = np.array(times_ms)
    return {"p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean())}


def _make_inputs(batch, seq_len, cat_cardinalities, n_num_total, d_graph, device):
    cards = torch.tensor(cat_cardinalities)
    seq_cat = torch.stack([torch.randint(0, int(cards[i]), (batch, seq_len), device=device)
                           for i in range(len(cat_cardinalities))], dim=-1)
    seq_num = torch.randn(batch, seq_len, n_num_total, device=device)
    mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    graph_emb = torch.randn(batch, d_graph, device=device)
    return seq_cat, seq_num, mask, graph_emb


def benchmark_torch(model, cat_cardinalities, n_num_total, seq_len, d_graph,
                    device, n_runs=1000, warmup=50, batch=1):
    model = model.to(device).eval()
    seq_cat, seq_num, mask, graph_emb = _make_inputs(
        batch, seq_len, cat_cardinalities, n_num_total, d_graph, device)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_online(seq_cat, seq_num, mask, graph_emb)
        if device == "cuda": torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.forward_online(seq_cat, seq_num, mask, graph_emb)
            if device == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)


def benchmark_onnx(onnx_path, cat_cardinalities, n_num_total, seq_len, d_graph,
                   providers, n_runs=1000, warmup=50, batch=1):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=providers)
    cards = np.array(cat_cardinalities)
    seq_cat = np.stack([np.random.randint(0, int(cards[i]), (batch, seq_len))
                        for i in range(len(cat_cardinalities))], axis=-1).astype("int64")
    seq_num = np.random.randn(batch, seq_len, n_num_total).astype("float32")
    mask = np.ones((batch, seq_len), dtype=bool)
    graph_emb = np.random.randn(batch, d_graph).astype("float32")
    feed = {"seq_cat": seq_cat, "seq_num": seq_num, "mask": mask, "graph_emb": graph_emb}
    for _ in range(warmup): sess.run(None, feed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times), sess.get_providers()


def _benchmark_one_model(name, v_strategy, ckpt_path):
    """benchmark 一个深度模型(4 档)。返回 dict。"""
    from src.config import load_config
    from src.models.fraud_model import FraudModel
    from src.deploy.export_onnx import export_online_path, verify_onnx_parity
    from src.deploy.build_trt import trt_available, build_engine

    mcfg = load_config("model")
    proc_dir = Path("data/processed") / v_strategy
    manifest = json.loads((proc_dir / "manifest.json").read_text())
    meta = json.loads((proc_dir / "feature_meta.json").read_text())
    cat_cardinalities = [meta["cat_cardinalities"][c] for c in meta["cat_cols"]]
    n_num_total = manifest["n_num_total"]
    n_cat = manifest["n_cat"]
    seq_len = manifest["seq_len"]
    d_graph = mcfg["d_graph"]

    model = FraudModel(cat_cardinalities, n_num_total, mcfg, fusion_mode="gated")
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    else:
        print(f"WARN: no checkpoint at {ckpt_path}, using random init for latency only")
    model.eval()

    Path("artifacts").mkdir(exist_ok=True)
    onnx_path = f"artifacts/online_{v_strategy.replace('_v','')}.onnx"
    export_online_path(model, n_cat, n_num_total, seq_len, d_graph, onnx_path)
    assert verify_onnx_parity(model, onnx_path, n_cat, n_num_total,
                              cat_cardinalities, seq_len, d_graph), "ONNX parity failed"

    res = {}
    res["pytorch_cpu"] = benchmark_torch(model, cat_cardinalities, n_num_total,
                                         seq_len, d_graph, "cpu")
    if torch.cuda.is_available():
        res["pytorch_gpu"] = benchmark_torch(model, cat_cardinalities, n_num_total,
                                             seq_len, d_graph, "cuda")
        try:
            stats, providers = benchmark_onnx(onnx_path, cat_cardinalities, n_num_total,
                                              seq_len, d_graph, ["CUDAExecutionProvider"])
            if "CUDAExecutionProvider" in providers:
                res["onnx_gpu"] = stats
            else:
                res["onnx_gpu"] = {"skipped": f"CUDAExecutionProvider not active (got {providers})"}
        except Exception as e:
            res["onnx_gpu"] = {"skipped": f"ORT CUDA load error: {e}"}

    if trt_available():
        engine = f"artifacts/online_{v_strategy.replace('_v','')}.engine"
        if build_engine(onnx_path, engine, fp16=True):
            try:
                stats, providers = benchmark_onnx(
                    onnx_path, cat_cardinalities, n_num_total, seq_len, d_graph,
                    [("TensorrtExecutionProvider", {"trt_fp16_enable": True})])
                if "TensorrtExecutionProvider" in providers:
                    res["tensorrt_fp16"] = stats
                else:
                    res["tensorrt_fp16"] = {"skipped": f"TRT EP not active (got {providers}); engine built OK"}
            except Exception as e:
                res["tensorrt_fp16"] = {"skipped": f"ORT TRT EP error: {e}; engine built OK"}
        else:
            res["tensorrt_fp16"] = {"skipped": "engine build failed"}
    else:
        res["tensorrt_fp16"] = {"skipped": "TensorRT not available"}
    return res


def run_benchmark():
    """对两个深度模型各跑 4 档 benchmark,落 experiments/benchmark_stage2.json。"""
    Path("experiments").mkdir(exist_ok=True)
    out_path = Path("experiments/benchmark_stage2.json")
    results = {}
    for v_strategy in ["full_v", "pruned_v"]:
        name = f"deep_{v_strategy.replace('_v','')}"
        ckpt = f"artifacts/best_{name}.pt"
        print(f"\n=== {name} ===")
        results[name] = _benchmark_one_model(name, v_strategy, ckpt)
        for k, v in results[name].items():
            print(f"  {k}: {v}")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
