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
    high_noise_only: bool = False,
    return_high: bool = False,
):
    from nodes import KSamplerAdvanced

    sampler = KSamplerAdvanced()
    split_step = max(1, min(int(split_step), int(steps) - 1))

    def sample_advanced(
        model,
        add_noise,
        seed,
        cfg,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        *,
        capture_denoised=False,
    ):
        captured = {"samples": None}
        latent_preview = None
        original_prepare_callback = None

        if capture_denoised:
            try:
                import latent_preview as latent_preview_module

                latent_preview = latent_preview_module
                original_prepare_callback = latent_preview.prepare_callback

                def prepare_callback_with_capture(*args, **kwargs):
                    callback = original_prepare_callback(*args, **kwargs)

                    def wrapped_callback(step, x0, x, total_steps):
                        if torch.is_tensor(x0):
                            captured["samples"] = x0.detach().cpu()
                        if callback is not None:
                            return callback(step, x0, x, total_steps)
                        return None

                    return wrapped_callback

                latent_preview.prepare_callback = prepare_callback_with_capture
            except Exception as exc:
                log.debug("Could not capture core KSampler denoised samples: %s", exc)
                latent_preview = None

        try:
            sampled, = sampler.sample(
                model,
                add_noise,
                int(seed),
                int(steps),
                float(cfg),
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                int(start_at_step),
                int(end_at_step),
                return_with_leftover_noise,
            )
        finally:
            if latent_preview is not None and original_prepare_callback is not None:
                latent_preview.prepare_callback = original_prepare_callback

        denoised = None
        if captured["samples"] is not None:
            denoised = dict(sampled)
            denoised["samples"] = captured["samples"]
        return sampled, denoised

    latent_high, denoised_high = sample_advanced(
        model_high,
        "enable",
        high_seed,
        high_cfg,
        latent,
        0,
        split_step,
        "enable",
        capture_denoised=high_noise_only and return_high,
    )

    if high_noise_only:
        if return_high:
            return latent_high, denoised_high or latent_high
        return latent_high

    latent_low, _ = sample_advanced(
        model_low,
        "disable",
        low_seed,
        low_cfg,
        latent_high,
        split_step,
        steps,
        "disable",
    )
    if return_high:
        return latent_low, latent_high
    return latent_low
