"""Bernini-only diffusion sampling loop (bypasses legacy WanVideo predict_with_cfg)."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from tqdm import tqdm

from .context_windows.context import create_window_mask, get_context_scheduler, WindowTracker
from .utils import log, offload_transformer
from ..bernini.guidance_modes import uses_apg_guidance
from comfy import model_management as mm
from comfy.utils import ProgressBar


@dataclass
class BerniniDenoiseConfig:
    transformer: Any
    patcher: Any
    model_meta: dict
    sample_scheduler: Any
    timesteps: torch.Tensor
    cfg: list[float]
    text_embeds: dict
    seq_len: int
    freqs: Any
    context_latents: list | None
    latent_video_length: int
    steps: int
    ttm_start_step: int
    device: torch.device
    dtype: torch.dtype
    guidance_mode: str
    apg_eta: float
    apg_momentum: float
    apg_norm_threshold: float
    context_options: dict | None
    cache_state: list
    force_offload: bool
    seed_g: torch.Generator
    scheduler_step_args: dict
    transformer_options: dict
    vae_upscale_factor: int
    is_looped: bool = False
    context_fn: Callable | None = None
    batched_cfg: bool = False


def _build_momentum_buffer(cfg: BerniniDenoiseConfig):
    if not uses_apg_guidance(cfg.guidance_mode):
        return None
    from .bernini_guidance import MomentumBuffer
    return MomentumBuffer(cfg.apg_momentum)


def _embed_list(embeds) -> list:
    if embeds is None:
        return []
    return embeds if isinstance(embeds, list) else [embeds]


def _combine_noise_preds(
    cfg: BerniniDenoiseConfig,
    z: torch.Tensor,
    idx: int,
    cfg_scale: float,
    noise_cond: torch.Tensor,
    noise_uncond: torch.Tensor,
    momentum_buf,
):
    from .bernini_guidance import normalized_guidance as _normalized_guidance

    if uses_apg_guidance(cfg.guidance_mode):
        sigma = cfg.sample_scheduler.sigmas[idx]
        x_cond = z - sigma * noise_cond
        x_uncond = z - sigma * noise_uncond
        x_guided = _normalized_guidance(
            x_cond, x_uncond, cfg_scale, momentum_buf, cfg.apg_eta, cfg.apg_norm_threshold
        )
        return (z - x_guided) / sigma

    return noise_uncond + cfg_scale * (noise_cond - noise_uncond)


def _predict_single(
    cfg: BerniniDenoiseConfig,
    z: torch.Tensor,
    timestep: torch.Tensor,
    idx: int,
    positive,
    negative,
    cfg_scale: float,
    cache_state: list | None,
    context_latents,
    context_window_start: int,
    momentum_buf,
):
    transformer = cfg.transformer
    total = max(len(cfg.timesteps) - cfg.ttm_start_step, 1)
    step_pct = idx / total
    last_step = idx >= len(cfg.timesteps) - cfg.ttm_start_step - 1

    core_opts = {**cfg.transformer_options, "bernini_core": True}
    base = {
        "y": None,
        "clip_fea": None,
        "seq_len": cfg.seq_len,
        "freqs": cfg.freqs,
        "t": timestep,
        "device": cfg.device,
        "is_uncond": False,
        "current_step": idx,
        "current_step_percentage": step_pct,
        "last_step": last_step,
        "total_steps": cfg.steps,
        "context_latents": context_latents,
        "context_window_start": context_window_start,
        "transformer_options": core_opts,
        "nag_params": cfg.text_embeds.get("nag_params", {}),
        "nag_context": cfg.text_embeds.get("nag_prompt_embeds"),
    }

    pos = _embed_list(positive)
    neg = _embed_list(negative)
    use_batched = (
        cfg.batched_cfg
        and not math.isclose(cfg_scale, 1.0)
        and len(pos) == 1
        and len(neg) == 1
        and not cfg.text_embeds.get("nag_prompt_embeds")
    )

    if use_batched:
        noise_preds, _, cache_cond = transformer(
            context=pos + neg,
            x=[z, z],
            pred_id=cache_state[0] if cache_state else None,
            **base,
        )
        noise_cond, noise_uncond = noise_preds[0], noise_preds[1]
        combined = _combine_noise_preds(cfg, z, idx, cfg_scale, noise_cond, noise_uncond, momentum_buf)
        return combined, [cache_cond, cache_cond]

    base["x"] = [z]
    base["nag_context"] = cfg.text_embeds.get("nag_prompt_embeds")
    noise_cond, _, cache_cond = transformer(
        context=positive,
        pred_id=cache_state[0] if cache_state else None,
        **base,
    )
    noise_cond = noise_cond[0]

    if math.isclose(cfg_scale, 1.0):
        return noise_cond, [cache_cond]

    base["is_uncond"] = True
    base["nag_context"] = None
    noise_uncond, _, cache_uncond = transformer(
        context=negative,
        pred_id=cache_state[1] if cache_state else None,
        **base,
    )
    noise_uncond = noise_uncond[0]

    combined = _combine_noise_preds(cfg, z, idx, cfg_scale, noise_cond, noise_uncond, momentum_buf)
    return combined, [cache_cond, cache_uncond]


def run_bernini_denoise(
    cfg: BerniniDenoiseConfig,
    latent: torch.Tensor,
    callback=None,
) -> torch.Tensor:
    """Run the slim Bernini denoise loop; returns latent tensor [C, T, H, W]."""
    log.info("Bernini runtime active")
    momentum_buf = _build_momentum_buffer(cfg)
    positive_all = cfg.text_embeds["prompt_embeds"]
    negative_all = cfg.text_embeds["negative_prompt_embeds"]
    if not negative_all and not all(math.isclose(c, 1.0) for c in cfg.cfg):
        raise ValueError("Negative embeddings required for CFG scale > 1.0")

    window_tracker = WindowTracker(verbose=bool(cfg.context_options and cfg.context_options.get("verbose")))
    ctx_opts = cfg.context_options
    context_frames = context_stride = context_overlap = None
    fuse_method = "linear"
    if ctx_opts is not None:
        context_frames = (ctx_opts["context_frames"] - 1) // 4 + 1
        context_stride = ctx_opts["context_stride"] // 4
        context_overlap = ctx_opts["context_overlap"] // 4
        fuse_method = ctx_opts.get("fuse_method", "linear")

    pbar = ProgressBar(len(cfg.timesteps) - cfg.ttm_start_step)
    cache_state = cfg.cache_state

    for idx, t in enumerate(tqdm(cfg.timesteps[cfg.ttm_start_step:], disable=ctx_opts is not None)):
        z = latent.to(cfg.device)
        timestep = torch.tensor([t]).to(cfg.device)
        cfg_scale = cfg.cfg[idx]

        if ctx_opts is not None and cfg.context_fn is not None and latent.shape[1] > context_frames:
            counter = torch.zeros_like(z, device=cfg.device)
            noise_pred = torch.zeros_like(z, device=cfg.device)
            context_queue = list(
                cfg.context_fn(
                    idx, cfg.steps, cfg.latent_video_length,
                    context_frames, context_stride, context_overlap,
                )
            )
            for c in context_queue:
                window_id = window_tracker.get_window_id(c)
                if cfg.cache_state is not None:
                    current_cache = window_tracker.get_teacache(window_id, cache_state)
                else:
                    current_cache = None

                sliced_ctx = None
                if cfg.context_latents:
                    sliced_ctx = []
                    for lat in cfg.context_latents:
                        if lat.ndim >= 2 and lat.shape[1] > 1 and max(c) < lat.shape[1]:
                            sliced_ctx.append(lat[:, c].to(cfg.device))
                        else:
                            sliced_ctx.append(lat.to(cfg.device))

                partial_z = z[:, c]
                pred, new_cache = _predict_single(
                    cfg, partial_z, timestep, idx,
                    positive_all, negative_all, cfg_scale,
                    current_cache, sliced_ctx, c[0], momentum_buf,
                )
                if new_cache is not None:
                    window_tracker.cache_states[window_id] = new_cache
                window_mask = create_window_mask(
                    pred, c, z.shape[1], context_overlap,
                    looped=cfg.is_looped, window_type=fuse_method,
                )
                noise_pred[:, c] += pred * window_mask
                counter[:, c] += window_mask
            noise_pred /= counter.clamp_min(1e-6)
        else:
            noise_pred, cache_state = _predict_single(
                cfg, z, timestep, idx,
                positive_all, negative_all, cfg_scale,
                cache_state, cfg.context_latents, 0, momentum_buf,
            )

        if callback is not None:
            callback_latent = (z - noise_pred * t.to(cfg.device) / 1000).detach()
            callback(idx, callback_latent.permute(1, 0, 2, 3), None, len(cfg.timesteps))
        else:
            pbar.update(1)

        latent = cfg.sample_scheduler.step(
            noise_pred.unsqueeze(0), timestep, z.unsqueeze(0), **cfg.scheduler_step_args
        )[0].squeeze(0).detach().cpu()
        del noise_pred, z, timestep

    if cfg.force_offload and not cfg.model_meta.get("auto_cpu_offload"):
        offload_transformer(cfg.transformer)

    mm.soft_empty_cache()
    gc.collect()
    return latent
