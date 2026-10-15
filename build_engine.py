"""ONNX -> TensorRT FP16 engine, built with the CALLING venv's TensorRT.

Why not rfdetr's own `trtexec` helper: it shells out to whatever `trtexec` is on PATH, which here
is the system TRT 10.11 - engines it produces cannot be deserialized by the 10.13.3.9 runtime the
venvs use. So we build through the Python API at the correct version, replicating rfdetr's own
flags (`--fp16 --memPoolSize=workspace:4096`, passed by infer_rfdetr.py). Build settings only; no
model logic. The default `workspace_gb=None` sets no limit (TensorRT's default: all device memory),
the same as ultralytics' engine export; a 4 GB cap leaves SegFormer-B4/B5 at 1024x2048 with no
tactic for the decode-head fuse conv.

All engines in the comparison (D-FINE, YOLO26, RF-DETR) are therefore TensorRT 10.13.3.9 + FP16.
"""

from pathlib import Path


def build(
    onnx_path,
    engine_path,
    half: bool = True,
    workspace_gb: int | None = None,
    fp32_layers: str | None = None,
) -> Path:
    """fp32_layers: regex on layer names to keep in fp32 inside an fp16 engine, for ops the model
    itself runs in fp32 under autocast (the fp32 ONNX graph does not record that)."""
    import re

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
    if workspace_gb:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if half:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if fp32_layers:
        n = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if not re.search(fp32_layers, layer.name):
                continue
            outs = [layer.get_output(j) for j in range(layer.num_outputs)]
            if not all(o.dtype == trt.float32 for o in outs):  # skip shape/int tensors
                continue
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            n += 1
        if not n:
            raise RuntimeError(f"fp32_layers={fp32_layers!r} matched no layer in {onnx_path}")
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print(f"  pinned {n} layers to fp32 ({fp32_layers})", flush=True)

    print(f"  building {engine_path.name} (TRT {trt.__version__}, fp16={half}) ...", flush=True)
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(ser)
    return engine_path
