from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.autograd import Function

try:  # torch >= 2.4
    from torch.amp import custom_bwd as _custom_bwd
    from torch.amp import custom_fwd as _custom_fwd

    _fwd: Any = _custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    _bwd: Any = _custom_bwd(device_type="cuda")
except (ImportError, TypeError):  # pragma: no cover - older torch
    from torch.cuda.amp import custom_bwd as _bwd
    from torch.cuda.amp import custom_fwd as _legacy_custom_fwd

    _fwd = _legacy_custom_fwd(cast_inputs=torch.float32)


class TruncatedExponential(Function):
    """``exp`` whose gradient is clamped for numerical stability (from torch-ngp)."""

    @staticmethod
    @_fwd
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @_bwd
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return g * torch.exp(torch.clamp(x, min=-15, max=15))


def trunc_exp(x: torch.Tensor) -> torch.Tensor:
    return TruncatedExponential.apply(x)


def _shifted_trunc_exp(x: torch.Tensor) -> torch.Tensor:
    return trunc_exp(x - 1)


def density_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "trunc_exp":
        return _shifted_trunc_exp
    if name == "relu":
        return F.relu
    raise ValueError(f"unknown density activation {name!r}")
