"""tiny-cuda-nn modules with a pure-PyTorch fallback.

The published models use `tiny-cuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_. When it
is not installed (or ``DAC_DISABLE_TCNN=1`` is set) equivalent PyTorch modules are used
instead, so the code also runs on CPU for development and testing. The fallback is much
slower and its checkpoints are not interchangeable with tiny-cuda-nn ones.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import nn

log = logging.getLogger(__name__)


def _import_tcnn() -> Any:
    if os.environ.get("DAC_DISABLE_TCNN"):
        return None
    try:
        import tinycudann
    except (ImportError, OSError):  # missing, or built for an unavailable CUDA runtime
        return None
    return tinycudann


tcnn: Any = _import_tcnn()

_warned = False


def has_tcnn() -> bool:
    return tcnn is not None


def _warn_fallback() -> None:
    global _warned
    if not _warned:
        log.warning(
            "tiny-cuda-nn is not available; using the pure-PyTorch fallback. This is slow and "
            "its checkpoints are not compatible with tiny-cuda-nn ones."
        )
        _warned = True


_ACTIVATIONS = {"None": nn.Identity, "ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid}


class TorchMLP(nn.Module):
    """Bias-free MLP matching tiny-cuda-nn's ``FullyFusedMLP``/``CutlassMLP`` architecture."""

    def __init__(self, n_input_dims: int, n_output_dims: int, network_config: dict[str, Any]) -> None:
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        width = network_config["n_neurons"]
        n_hidden = network_config["n_hidden_layers"]
        hidden_act = _ACTIVATIONS[network_config.get("activation", "ReLU")]
        dims = [n_input_dims] + [width] * n_hidden + [n_output_dims]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(d_in, d_out, bias=False)
            nn.init.xavier_uniform_(linear.weight)
            layers.append(linear)
            is_last = i == len(dims) - 2
            layers.append(_ACTIVATIONS[network_config.get("output_activation", "None")]() if is_last else hidden_act())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float())


class TorchSphericalHarmonics(nn.Module):
    """Degree-4 real spherical harmonics of directions given in ``[0, 1]^3``, as in tiny-cuda-nn."""

    def __init__(self, n_input_dims: int, encoding_config: dict[str, Any]) -> None:
        super().__init__()
        if n_input_dims != 3 or encoding_config.get("otype") != "SphericalHarmonics":
            raise NotImplementedError("the PyTorch fallback only implements SphericalHarmonics on 3D inputs")
        if encoding_config.get("degree") != 4:
            raise NotImplementedError("the PyTorch fallback only implements degree-4 spherical harmonics")
        self.n_input_dims = 3
        self.n_output_dims = 16

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        x, y, z = (directions.float() * 2 - 1).unbind(-1)
        xy, xz, yz = x * y, x * z, y * z
        x2, y2, z2 = x * x, y * y, z * z
        return torch.stack(
            [
                torch.full_like(x, 0.28209479177387814),
                -0.48860251190291987 * y,
                0.48860251190291987 * z,
                -0.48860251190291987 * x,
                1.0925484305920792 * xy,
                -1.0925484305920792 * yz,
                0.94617469575755997 * z2 - 0.31539156525251999,
                -1.0925484305920792 * xz,
                0.54627421529603959 * x2 - 0.54627421529603959 * y2,
                0.59004358992664352 * y * (-3.0 * x2 + y2),
                2.8906114426405538 * xy * z,
                0.45704579946446572 * y * (1.0 - 5.0 * z2),
                0.3731763325901154 * z * (5.0 * z2 - 3.0),
                0.45704579946446572 * x * (1.0 - 5.0 * z2),
                1.4453057213202769 * z * (x2 - y2),
                0.59004358992664352 * x * (-x2 + 3.0 * y2),
            ],
            dim=-1,
        )


def network(n_input_dims: int, n_output_dims: int, network_config: dict[str, Any]) -> nn.Module:
    if tcnn is not None:
        return tcnn.Network(n_input_dims=n_input_dims, n_output_dims=n_output_dims, network_config=network_config)
    _warn_fallback()
    return TorchMLP(n_input_dims, n_output_dims, network_config)


def encoding(n_input_dims: int, encoding_config: dict[str, Any]) -> nn.Module:
    if tcnn is not None:
        return tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=encoding_config)
    _warn_fallback()
    return TorchSphericalHarmonics(n_input_dims, encoding_config)
