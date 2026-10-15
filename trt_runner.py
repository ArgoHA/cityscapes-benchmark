"""Minimal TensorRT executor for frameworks that ship no TensorRT runtime (SegFormer, EfficientViT).

YOLO26 (AutoBackend) and D-FINE (TRTModel) bring their own TRT inference; HF transformers and the
EfficientViT repo do not. This is the thinnest possible bridge: static-shape bs1 engine,
persistent GPU buffers, `execute_async_v3`. No pre/postprocessing lives here: the caller feeds the
framework's own preprocessed tensor in and hands the raw output to the framework's own postprocess.
"""

from pathlib import Path

import tensorrt as trt
import torch

_TRT2TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.uint8: torch.uint8,
    trt.bool: torch.bool,
}


class TRTRunner:
    def __init__(self, engine_path: str | Path):
        lg = trt.Logger(trt.Logger.ERROR)
        with trt.Runtime(lg) as rt:
            self.engine = rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError(f"{engine_path}: dynamic shape {shape}; build a static bs1 engine")
            buf = torch.empty(
                shape, dtype=_TRT2TORCH[self.engine.get_tensor_dtype(name)], device="cuda"
            )
            self.ctx.set_tensor_address(name, buf.data_ptr())
            mode = self.engine.get_tensor_mode(name)
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs)[name] = buf
        if len(self.inputs) != 1:
            raise ValueError(f"expected one input, got {list(self.inputs)}")
        self.input = next(iter(self.inputs.values()))

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: preprocessed tensor of the engine's input shape. Returns the output buffers (GPU)."""
        if tuple(x.shape) != tuple(self.input.shape):
            raise ValueError(f"input {tuple(x.shape)} != engine {tuple(self.input.shape)}")
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.input.copy_(x, non_blocking=True)  # also casts if the engine input is fp16
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.stream)
        return list(self.outputs.values())
