"""ONNX -> TensorRT FP16 engine, built with the CALLING venv's TensorRT.

Why not rfdetr's own `trtexec` helper: it shells out to whatever `trtexec` is on PATH, which here
is the system TRT 10.11 - engines it produces cannot be deserialized by the 10.13.3.9 runtime the
venvs use. So we build through the Python API at the correct version, replicating rfdetr's own
flags (`--fp16 --memPoolSize=workspace:4096`). Build settings only; no model logic.

All engines in the comparison (D-FINE, YOLO26, RF-DETR) are therefore TensorRT 10.13.3.9 + FP16.
"""

from pathlib import Path


def build(onnx_path, engine_path, half: bool = True, workspace_gb: int = 4) -> Path:
    import tensorrt as trt

    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    if engine_path.exists():
        return engine_path

    lg = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(lg)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, lg)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if half:
        cfg.set_flag(trt.BuilderFlag.FP16)

    print(f"  building {engine_path.name} (TRT {trt.__version__}, fp16={half}) ...", flush=True)
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(ser)
    return engine_path
