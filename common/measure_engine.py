"""Pure engine latency for any .engine — uniform across D-FINE / YOLO26 / RF-DETR.

The end-to-end numbers (dump.py) measure each framework's SHIPPED inference path, which is what a
user actually experiences. But both rivals do heavy CPU preprocessing on the 2048x1024 frame
(ultralytics: letterbox ~1.7 ms; rfdetr: PIL Resize, several ms), which dwarfs the network and
would turn the latency table into a preprocessing-quality contest.

So we report both:
  e2e ms     - framework's own path, end to end            (dump.py)
  engine ms  - TRT execute only, dummy input, no H2D/D2H   (this file)

`engine ms` isolates the architecture and is the number comparable to the vendors' published
TensorRT figures. Same loop, same warmup, same GPU for every engine.

  uv run python -m common.measure_engine [--csv out.csv] <a.engine> <b.engine> ...
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorrt as trt
import torch
from tabulate import tabulate

_TRT2TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def _read_plan(path: Path) -> bytes:
    """Ultralytics prefixes its .engine with `int32 len + metadata JSON` (their AutoBackend skips
    it before deserializing). Everyone else writes a bare plan. Detect and strip."""
    blob = path.read_bytes()
    n = int.from_bytes(blob[:4], "little", signed=True)
    if 0 < n < 100_000 and blob[4 : 4 + n].lstrip()[:1] == b"{":
        return blob[4 + n :]
    return blob


def engine_ms(path: Path, iters: int = 300, warmup: int = 50) -> tuple[str, float]:
    lg = trt.Logger(trt.Logger.ERROR)
    with trt.Runtime(lg) as rt:
        eng = rt.deserialize_cuda_engine(_read_plan(path))
    ctx = eng.create_execution_context()

    in_shape = None
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = tuple(1 if d < 0 else d for d in eng.get_tensor_shape(n))
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(n, shape)
            in_shape = shape
        buf = torch.empty(shape, dtype=_TRT2TORCH[eng.get_tensor_dtype(n)], device="cuda")
        ctx.set_tensor_address(n, buf.data_ptr())
        globals().setdefault("_keep", []).append(buf)  # keep buffers alive

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    lat = []
    with torch.cuda.stream(stream):
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    return f"{in_shape[2]}x{in_shape[3]}", float(np.median(lat))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("engines", nargs="+")
    args = ap.parse_args()

    rows = []
    for p in args.engines:
        p = Path(p)
        hw, ms = engine_ms(p)  # a failure here must not silently become a missing row
        rows.append({"engine": str(p), "name": p.stem, "input": hw, "engine_ms": round(ms, 2)})
    df = pd.DataFrame(rows).set_index("engine")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=True))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv)
