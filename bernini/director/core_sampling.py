"""ComfyUI core dual-stage sampling helpers for the official Bernini backend."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("ComfyUI-Bernini.director.core_sampling")


def apply_model_sampling_shift(model, shift: float):
    if shift <= 0.0:
        return model

    from comfy_extras.nodes_model_advanced import ModelSamplingSD3

    patcher = ModelSamplingSD3()
    patched, = patcher.patch(model, shift)
    return patched


def apply_apg(model, *, eta: float, momentum: float, norm_threshold: float):
    if eta == 1.0 and momentum == 0.0 and norm_threshold <= 0.0:
        return model

    from comfy_extras.nodes_apg import project

    running_avg = 0
    prev_sigma = None

    def pre_cfg_function(args):
        nonlocal running_avg, prev_sigma

        if len(args["conds_out"]) == 1:
            return args["conds_out"]

        cond = args["conds_out"][0]
        uncond = args["conds_out"][1]
        sigma = args["sigma"][0]
        cond_scale = args["cond_scale"]

        if prev_sigma is not None and sigma > prev_sigma:
            running_avg = 0
        prev_sigma = sigma

        guidance = cond - uncond

        if momentum != 0:
            if not torch.is_tensor(running_avg):
                running_avg = guidance
            else:
                running_avg = momentum * running_avg + guidance
            guidance = running_avg

        if norm_threshold > 0:
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale = torch.minimum(
                torch.ones_like(guidance_norm),
                norm_threshold / guidance_norm,
            )
            guidance = guidance * scale

        guidance_parallel, guidance_orthogonal = project(guidance, cond)
        modified_guidance = guidance_orthogonal + eta * guidance_parallel
        modified_cond = (uncond + modified_guidance) + (cond - uncond) / cond_scale
        return [modified_cond, uncond] + args["conds_out"][2:]

    patched = model.clone()
    patched.set_model_sampler_pre_cfg_function(pre_cfg_function)
    return patched


def sample_dual_stage(
    *,
    model_high,
    model_low,
    positive,
    negative,
    latent,
    high_seed: int,
    low_seed: int,
    high_cfg: float,
    low_cfg: float,
    steps: int,
    split_step: int,
    sampler_name: str,
    scheduler: str,
):
    from nodes import KSamplerAdvanced

    sampler = KSamplerAdvanced()
    split_step = max(1, min(int(split_step), int(steps) - 1))

    latent_high, = sampler.sample(
        model_high,
        "enable",
        int(high_seed),
        int(steps),
        float(high_cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        0,
        split_step,
        "enable",
    )

    latent_low, = sampler.sample(
        model_low,
        "disable",
        int(low_seed),
        int(steps),
        float(low_cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_high,
        split_step,
        int(steps),
        "disable",
    )
    return latent_low
