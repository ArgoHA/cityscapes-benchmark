"""Params + GMACs with ONE counter for every framework.

Published MACs come from different tools (thop, fvcore, mmcv, torchprofile) that disagree on what
counts, so we recount everything with torch's built-in `FlopCounterMode` at the benchmark input.
It counts matmul/conv/attention FLOPs (2 per MAC); we report GMACs = FLOPs / 2, the unit the
papers use. Elementwise ops are not counted by any of these tools, so the numbers stay comparable.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode


@torch.no_grad()
def model_stats(model: torch.nn.Module, x: torch.Tensor) -> dict:
    params = sum(p.numel() for p in model.parameters())
    with FlopCounterMode(display=False) as fc:
        model(x)
    return {"params_m": round(params / 1e6, 2), "gmacs": round(fc.get_total_flops() / 2e9, 1)}
