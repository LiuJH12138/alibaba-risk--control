import os


def trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> bool:
    """用 TensorRT Python API 把 ONNX 编译为独立 TensorRT 引擎(FP16),产出 .engine 工件。
    返回是否成功。引擎硬件专属:本机 RTX 5090 / CUDA 12.8。
    注:延迟 benchmark(Task 21)为保证测量口径一致,统一走 ORT 的 TensorRT EP;
    本函数产出的独立引擎是单独的部署工件,也验证 TRT 编译链路可用。"""
    if not trt_available():
        print("TensorRT not available, skipping engine build")
        return False
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolFlag.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # 动态 batch:min 1 / opt 64 / max 256
    name_to_input = {network.get_input(i).name: network.get_input(i)
                     for i in range(network.num_inputs)}
    for name in ["seq", "mask", "graph_emb"]:
        shape = list(name_to_input[name].shape)
        profile.set_shape(name, [1] + shape[1:], [64] + shape[1:], [256] + shape[1:])
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"engine written to {engine_path}")
    return True
