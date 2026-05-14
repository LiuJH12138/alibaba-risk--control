import json
import time
from pathlib import Path
import numpy as np
import torch

def _percentiles(times_ms):
    arr = np.array(times_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }

def _make_inputs(batch, seq_len, feat_dim, d_graph, device):
    return (torch.randn(batch, seq_len, feat_dim, device=device),
            torch.ones(batch, seq_len, dtype=torch.bool, device=device),
            torch.randn(batch, d_graph, device=device))

def benchmark_torch(model, feat_dim, seq_len, d_graph, device,
                    n_runs=1000, warmup=50, batch=1):
    """benchmark PyTorch eager 单请求延迟(batch=1 默认)。"""
    model = model.to(device).eval()
    seq, mask, graph_emb = _make_inputs(batch, seq_len, feat_dim, d_graph, device)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_online(seq, mask, graph_emb)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.forward_online(seq, mask, graph_emb)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)

def benchmark_onnx(onnx_path, feat_dim, seq_len, d_graph, providers,
                   n_runs=1000, warmup=50, batch=1):
    """benchmark ONNXRuntime 延迟(providers 控制 CPU/GPU)。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=providers)
    seq = np.random.randn(batch, seq_len, feat_dim).astype("float32")
    mask = np.ones((batch, seq_len), dtype=bool)
    graph_emb = np.random.randn(batch, d_graph).astype("float32")
    feed = {"seq": seq, "mask": mask, "graph_emb": graph_emb}
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)

def run_benchmark():
    """4 档对比:PyTorch-CPU / PyTorch-GPU / ONNX-GPU / TensorRT-FP16。
    结果落 experiments/benchmark.json。
    每个 config 独立 try/except:单个 config 失败不中断其余 config。"""
    from src.config import load_config
    from src.models.fraud_model import FraudModel
    from src.deploy.export_onnx import export_online_path, verify_onnx_parity
    from src.deploy.build_trt import trt_available, build_engine

    mcfg = load_config("model")
    dcfg = load_config("data")
    seq_len = dcfg["seq_len"]
    with open("data/processed/manifest.json") as f:
        feat_dim = json.load(f)["feat_dim"]
    d_graph = mcfg["d_graph"]

    model = FraudModel(feat_dim, mcfg, fusion_mode="gated")
    ckpt = Path("artifacts/best_model.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    Path("artifacts").mkdir(exist_ok=True)
    onnx_path = "artifacts/online.onnx"
    export_online_path(model, feat_dim, seq_len, d_graph, onnx_path)
    # Parity check is a hard gate — a failure means the export is broken.
    assert verify_onnx_parity(model, onnx_path, feat_dim, seq_len, d_graph), \
        "ONNX parity failed — 不信任后续延迟数字"

    results = {}

    # --- pytorch_cpu ---
    try:
        print("Benchmarking pytorch_cpu ...")
        results["pytorch_cpu"] = benchmark_torch(
            model, feat_dim, seq_len, d_graph, "cpu")
        print("pytorch_cpu:", results["pytorch_cpu"])
    except Exception as exc:
        print(f"[pytorch_cpu] FAILED: {exc}")
        results["pytorch_cpu"] = {"skipped": str(exc)}

    # --- pytorch_gpu ---
    if torch.cuda.is_available():
        try:
            print("Benchmarking pytorch_gpu ...")
            results["pytorch_gpu"] = benchmark_torch(
                model, feat_dim, seq_len, d_graph, "cuda")
            print("pytorch_gpu:", results["pytorch_gpu"])
        except Exception as exc:
            print(f"[pytorch_gpu] FAILED: {exc}")
            results["pytorch_gpu"] = {"skipped": str(exc)}
    else:
        results["pytorch_gpu"] = {"skipped": "CUDA not available"}

    # --- onnx_gpu ---
    if torch.cuda.is_available():
        try:
            print("Benchmarking onnx_gpu ...")
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            active = sess.get_providers()
            if "CUDAExecutionProvider" not in active:
                raise RuntimeError(
                    f"CUDAExecutionProvider not active (got {active}); "
                    "likely cuDNN/onnxruntime-gpu incompatibility")
            seq_np = np.random.randn(1, seq_len, feat_dim).astype("float32")
            mask_np = np.ones((1, seq_len), dtype=bool)
            graph_np = np.random.randn(1, d_graph).astype("float32")
            feed = {"seq": seq_np, "mask": mask_np, "graph_emb": graph_np}
            for _ in range(50):
                sess.run(None, feed)
            times = []
            for _ in range(1000):
                t0 = time.perf_counter()
                sess.run(None, feed)
                times.append((time.perf_counter() - t0) * 1000)
            results["onnx_gpu"] = _percentiles(times)
            print("onnx_gpu:", results["onnx_gpu"])
        except Exception as exc:
            print(f"[onnx_gpu] SKIPPED: {exc}")
            results["onnx_gpu"] = {"skipped": str(exc)}
    else:
        results["onnx_gpu"] = {"skipped": "CUDA not available"}

    # --- tensorrt_fp16 ---
    if trt_available():
        try:
            print("Building TensorRT engine ...")
            engine_ok = build_engine(onnx_path, "artifacts/online.engine", fp16=True)
            if not engine_ok:
                raise RuntimeError("build_engine returned False")
            print("Benchmarking tensorrt_fp16 via ORT TensorRT EP ...")
            import onnxruntime as ort
            trt_providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": True})]
            trt_sess = ort.InferenceSession(onnx_path, providers=trt_providers)
            trt_active = trt_sess.get_providers()
            if "TensorrtExecutionProvider" not in trt_active:
                raise RuntimeError(
                    f"TensorrtExecutionProvider not active (got {trt_active}); "
                    "likely cuDNN/onnxruntime-gpu incompatibility — engine built but EP unavailable")
            seq_np = np.random.randn(1, seq_len, feat_dim).astype("float32")
            mask_np = np.ones((1, seq_len), dtype=bool)
            graph_np = np.random.randn(1, d_graph).astype("float32")
            feed = {"seq": seq_np, "mask": mask_np, "graph_emb": graph_np}
            for _ in range(50):
                trt_sess.run(None, feed)
            times = []
            for _ in range(1000):
                t0 = time.perf_counter()
                trt_sess.run(None, feed)
                times.append((time.perf_counter() - t0) * 1000)
            results["tensorrt_fp16"] = _percentiles(times)
            print("tensorrt_fp16:", results["tensorrt_fp16"])
        except Exception as exc:
            print(f"[tensorrt_fp16] SKIPPED: {exc}")
            results["tensorrt_fp16"] = {"skipped": str(exc)}
    else:
        results["tensorrt_fp16"] = {"skipped": "TensorRT not available"}

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== benchmark.json ===")
    for k, v in results.items():
        print(k, v)
    return results

if __name__ == "__main__":
    run_benchmark()
