"""Adaptive Projected Guidance (APG) helpers for Bernini sampling."""

from __future__ import annotations

import torch


class MomentumBuffer:
    """EMA buffer for smoothing guidance differences across timesteps."""

    def __init__(self, momentum: float = -0.5):
        self.momentum = momentum
        self.running_average = 0

    def update(self, value: torch.Tensor):
        self.running_average = value + self.momentum * self.running_average


def normalize_diff(
    diff: torch.Tensor,
    base_pred: torch.Tensor,
    momentum_buffer: MomentumBuffer | None = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> torch.Tensor:
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        diff_n = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale = torch.minimum(torch.ones_like(diff_n), norm_threshold / diff_n)
        diff = diff * scale

    v0 = diff.double()
    v1 = base_pred.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return (v0_orthogonal + eta * v0_parallel).to(diff.dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer | None = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> torch.Tensor:
    nd = normalize_diff(
        pred_cond - pred_uncond, pred_cond, momentum_buffer, eta, norm_threshold
    )
    return pred_uncond + guidance_scale * nd
