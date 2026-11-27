from __future__ import annotations

import pytest
import torch

from dac.ops.losses import alpha_interlevel_loss, distortion_loss, interlevel_loss, plane_tv
from dac.rendering.ray_samplers import PDFSampler, RayBundle, UniformSampler
from dac.rendering.spatial_distortions import SceneContraction


def bundle(n: int = 4) -> RayBundle:
    return RayBundle(
        origins=torch.zeros(n, 3),
        directions=torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1),
        nears=torch.full((n, 1), 2.0),
        fars=torch.full((n, 1), 6.0),
    )


def test_uniform_sampler_is_deterministic_in_eval_mode() -> None:
    sampler = UniformSampler().eval()
    samples = sampler(bundle(), num_samples=4)
    torch.testing.assert_close(samples.starts[0, :, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    torch.testing.assert_close(samples.ends[0, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(samples.get_positions()[0, :, 2], torch.tensor([2.5, 3.5, 4.5, 5.5]))
    # Training mode jitters samples within their bins.
    jittered = UniformSampler().train()(bundle(), num_samples=4)
    assert ((jittered.starts >= 2) & (jittered.ends <= 6)).all()
    assert not torch.equal(jittered.starts, samples.starts)


def test_weights_follow_the_volume_rendering_equation() -> None:
    samples = UniformSampler().eval()(bundle(1), num_samples=3)
    density = torch.tensor([[[0.0], [1.0], [2.0]]])
    weights, alphas = samples.get_weights(density)
    expected_alpha = 1 - torch.exp(-density * 4 / 3)  # bins are 4/3 long
    transmittance = torch.tensor([1.0, 1.0, torch.exp(torch.tensor(-4 / 3)).item()])
    torch.testing.assert_close(alphas[0, :, 0], expected_alpha[0, :, 0])
    torch.testing.assert_close(weights[0, :, 0], expected_alpha[0, :, 0] * transmittance)
    assert weights.sum() <= 1


def test_pdf_sampler_concentrates_samples() -> None:
    coarse = UniformSampler().eval()(bundle(2), num_samples=8)
    weights = torch.zeros(2, 8, 1)
    weights[:, 5] = 1.0  # all mass in [4.5, 5.0]
    fine = PDFSampler(include_original=False, histogram_padding=0.0).eval()(bundle(2), coarse, weights, num_samples=16)
    centres = fine.get_positions()[..., 2]
    assert centres.shape == (2, 16)
    assert ((centres > 4.4) & (centres < 5.1)).float().mean() > 0.8
    assert (fine.spacing_starts[..., 1:, 0] >= fine.spacing_starts[..., :-1, 0]).all()


def test_scene_contraction() -> None:
    contraction = SceneContraction(order=float("inf"))
    inside = torch.tensor([[[0.5, -0.2, 0.9]]])
    torch.testing.assert_close(contraction(inside), inside)
    far = torch.tensor([[[100.0, -3.0, 0.0]]])
    out = contraction(far)
    assert out.abs().max() < 2
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(1.99))


def test_interlevel_losses_vanish_for_matching_histograms() -> None:
    samples = UniformSampler().eval()(bundle(3), num_samples=6)
    weights = torch.rand(3, 6, 1)
    assert interlevel_loss([weights, weights], [samples, samples]).item() == pytest.approx(0.0, abs=1e-12)
    assert alpha_interlevel_loss(weights, samples, [weights, weights], [samples, samples]).item() == pytest.approx(
        0.0, abs=1e-12
    )
    # A proposal that misses the mass is penalised.
    assert interlevel_loss([torch.zeros_like(weights), weights], [samples, samples]).item() > 0


def test_plane_tv_and_distortion() -> None:
    assert plane_tv(torch.ones(1, 2, 5, 5)).item() == 0.0
    ramp = torch.arange(4.0).repeat(4, 1)[None, None]
    assert plane_tv(ramp).item() > 0
    samples = UniformSampler().eval()(bundle(2), num_samples=4)
    weights = torch.full((2, 4, 1), 0.25)
    # Only the uniform term contributes (see the distortion_loss docstring).
    expected = (1 / 3) * (samples.deltas * weights**2).mean()
    torch.testing.assert_close(distortion_loss(weights, samples), expected)
