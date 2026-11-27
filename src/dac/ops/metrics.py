"""Image quality metrics. Images are ``[H, W, 3]`` tensors with values in ``[0, 1]``."""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import torch

# Smallest side for which torchmetrics' default 5-scale MS-SSIM is defined.
MS_SSIM_MIN_SIZE = 176


def _check_image(img: torch.Tensor) -> None:
    if img.dim() != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected an [H, W, 3] image, got shape {tuple(img.shape)}")
    if img.max() > 1.05 or img.min() < -0.05:
        raise ValueError("expected image values in [0, 1]")


def psnr(rgb: torch.Tensor, gt: torch.Tensor) -> float:
    _check_image(rgb)
    _check_image(gt)
    mse = torch.mean((rgb - gt) ** 2).item()
    return 10 * math.log10(1.0 / mse)


def ssim(rgb: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM as implemented in MipNeRF (NaN for images smaller than the 11x11 filter).

    https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    """
    import scipy.signal

    filter_size, filter_sigma, k1, k2, max_val = 11, 1.5, 0.01, 0.03, 1.0
    a = rgb.cpu().numpy()
    b = gt.cpu().numpy()
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"expected two [H, W, 3] images, got {a.shape} and {b.shape}")
    if min(a.shape[:2]) < filter_size:
        return float("nan")  # image smaller than the filter

    # Separable 1D Gaussian blur.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    def blur(z: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                scipy.signal.convolve2d(
                    scipy.signal.convolve2d(z[..., i], filt[:, None], mode="valid"), filt[None, :], mode="valid"
                )
                for i in range(z.shape[-1])
            ],
            -1,
        )

    mu0, mu1 = blur(a), blur(b)
    mu00, mu11, mu01 = mu0 * mu0, mu1 * mu1, mu0 * mu1
    sigma00 = np.maximum(0.0, blur(a**2) - mu00)
    sigma11 = np.maximum(0.0, blur(b**2) - mu11)
    sigma01 = blur(a * b) - mu01
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    return float(np.mean(numer / denom))


@functools.cache
def _ms_ssim_metric() -> Any:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

    return MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)


def msssim(rgb: torch.Tensor, gt: torch.Tensor) -> float:
    """Multi-scale SSIM (NaN for images smaller than ``MS_SSIM_MIN_SIZE``)."""
    _check_image(rgb)
    _check_image(gt)
    if min(rgb.shape[:2]) < MS_SSIM_MIN_SIZE:
        return float("nan")  # too small for five 11x11-filtered scales
    metric = _ms_ssim_metric()
    value = metric(rgb.permute(2, 0, 1)[None].cpu(), gt.permute(2, 0, 1)[None].cpu())
    metric.reset()  # do not accumulate state across calls
    return float(value.item())


@functools.cache
def _lpips_model(net_name: str, device: str) -> Any:
    try:
        import lpips
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("LPIPS requires the optional dependency: pip install 'dac[metrics]'") from e
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(rgb: torch.Tensor, gt: torch.Tensor, net_name: str = "alex", device: str = "cpu") -> float:
    model = _lpips_model(net_name, device)
    gt = gt.permute(2, 0, 1).contiguous().to(device)
    rgb = rgb.permute(2, 0, 1).contiguous().to(device)
    return float(model(gt, rgb, normalize=True).item())


def image_metrics(rgb: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """The metrics reported during validation."""
    return {
        "mse": torch.mean((gt - rgb) ** 2).item(),
        "psnr": psnr(rgb, gt),
        "ssim": ssim(rgb, gt),
        "ms_ssim": msssim(rgb, gt),
    }
