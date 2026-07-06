"""Map ComfyUI KSampler names to Bernini/Wan schedulers and Comfy sigma curves."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

SAMPLER_NAME_TO_BERNINI: dict[str, str] = {
    "dpmpp_2m_sde": "dpm++_sde",
    "dpmpp_2m_sde_gpu": "dpm++_sde",
    "dpmpp_2m_sde_heun": "dpm++_sde",
    "dpmpp_2m_sde_heun_gpu": "dpm++_sde",
    "dpmpp_sde": "dpm++_sde",
    "dpmpp_sde_gpu": "dpm++_sde",
    "dpmpp_2m": "dpm++",
    "dpmpp_2m_cfg_pp": "dpm++",
    "dpmpp_2s_ancestral": "dpm++",
    "dpmpp_2s_ancestral_cfg_pp": "dpm++",
    "dpmpp_3m_sde": "dpm++_sde",
    "dpmpp_3m_sde_gpu": "dpm++_sde",
    "euler": "euler",
    "euler_cfg_pp": "euler",
    "euler_ancestral": "euler",
    "euler_ancestral_cfg_pp": "euler",
    "deis": "deis",
    "lcm": "lcm",
    "res_multistep": "res_multistep",
    "res_multistep_cfg_pp": "res_multistep",
    "res_multistep_ancestral": "res_multistep",
    "res_multistep_ancestral_cfg_pp": "res_multistep",
    "er_sde": "er_sde",
    "uni_pc": "unipc",
    "uni_pc_bh2": "unipc",
}


# WanVideoScheduler (v1) bernini scheduler id -> Comfy KSampler sampler_name
LEGACY_BERNINI_TO_KSAMPLER: dict[str, str] = {
    "euler": "euler",
    "euler/beta": "euler",
    "longcat_distill_euler": "euler",
    "dpm++": "dpmpp_2m",
    "dpm++/beta": "dpmpp_2m",
    "dpm++_sde": "dpmpp_2m_sde",
    "dpm++_sde/beta": "dpmpp_2m_sde",
    "unipc": "uni_pc",
    "unipc/beta": "uni_pc",
    "deis": "deis",
    "lcm": "lcm",
    "lcm/beta": "lcm",
    "res_multistep": "res_multistep",
    "er_sde": "er_sde",
}


def _comfy_schedulers() -> set[str]:
    import comfy.samplers

    return set(comfy.samplers.KSampler.SCHEDULERS)


def _is_comfy_scheduler(value) -> bool:
    if value is None:
        return False
    return str(value).strip() in _comfy_schedulers()


def legacy_bernini_scheduler_to_sampler_name(bernini_scheduler: str) -> str:
    """Map old WanVideoScheduler combo id to Comfy KSampler sampler_name."""
    key = (bernini_scheduler or "").strip()
    if key in LEGACY_BERNINI_TO_KSAMPLER:
        return LEGACY_BERNINI_TO_KSAMPLER[key]
    if key in SAMPLER_NAME_TO_BERNINI:
        return key
    return "euler"


DEFAULT_FLOW_SHIFT = 5.0


def _coerce_flow_shift(shift) -> float:
    value = float(shift)
    if value <= 0.0:
        log.warning(
            "Bernini Scheduler: shift=%s is invalid for flow-match schedulers; using default %s",
            shift,
            DEFAULT_FLOW_SHIFT,
        )
        return DEFAULT_FLOW_SHIFT
    return value


def _try_remap_partial_legacy_v2(
    sampler_name,
    scheduler,
    steps,
    shift,
    start_step,
    end_step,
    enhance_hf=False,
):
    """Remap 6-value workflows after manually inserting the scheduler combo.

    Example saved as: [euler, simple, 5, 0, 4, false]
    Intended meaning: steps=8, shift=5, start=0, end=4
    """
    if not _is_comfy_scheduler(scheduler):
        return None

    legacy_id = str(sampler_name or "").strip()
    if legacy_id not in LEGACY_BERNINI_TO_KSAMPLER and legacy_id not in SAMPLER_NAME_TO_BERNINI:
        return None

    shift_f = float(shift)
    steps_i = int(steps)
    start_i = int(start_step)
    end_i = int(end_step) if isinstance(end_step, int) else (0 if end_step in (False, 0) else int(end_step))

    # steps=old shift, shift=old start, start=old end, end=0/false
    if shift_f == 0.0 and start_i > 0 and end_i == 0 and 1 <= steps_i <= 32:
        remapped_shift = float(steps_i if steps_i >= 4 else DEFAULT_FLOW_SHIFT)
        remapped_steps = 8
        remapped_start = 0
        remapped_end = start_i
        remapped_enhance = bool(end_step) if isinstance(end_step, bool) else bool(enhance_hf)
        log.warning(
            "Bernini Scheduler: detected shifted legacy widgets; remapped "
            "%r -> steps=%s shift=%s start=%s end=%s",
            [sampler_name, scheduler, steps, shift, start_step, end_step],
            remapped_steps,
            remapped_shift,
            remapped_start,
            remapped_end,
        )
        return (
            legacy_bernini_scheduler_to_sampler_name(legacy_id),
            scheduler,
            remapped_steps,
            remapped_shift,
            remapped_start,
            remapped_end,
            remapped_enhance,
        )
    return None


def _is_flow_match_scheduler(sample_scheduler) -> bool:
    name = sample_scheduler.__class__.__name__.lower()
    return "flowmatch" in name or "flow_match" in name


def _slice_sigmas_to_timesteps(sample_scheduler, timesteps: torch.Tensor) -> torch.Tensor | None:
    """Recover sliced sigmas from full_sigmas + all_timesteps when lengths disagree."""
    full = getattr(sample_scheduler, "full_sigmas", None)
    all_ts = getattr(sample_scheduler, "all_timesteps", None)
    if full is None or all_ts is None or len(timesteps) <= 0:
        return None
    if not isinstance(full, torch.Tensor):
        full = torch.as_tensor(full, dtype=torch.float32)
    if not isinstance(all_ts, torch.Tensor):
        all_ts = torch.as_tensor(all_ts, dtype=torch.float32)
    t0 = timesteps.reshape(-1)[0].to(dtype=all_ts.dtype, device=all_ts.device)
    matches = (all_ts.reshape(-1) == t0).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None
    start = int(matches[0].item())
    end = start + len(timesteps) + 1
    if end > len(full):
        return None
    return full[start:end]


def attach_scheduler_timesteps(sample_scheduler, timesteps, device=None, stored_sigmas=None):
    """Keep diffusers scheduler state aligned with the sliced timestep list."""
    if timesteps is None:
        raise ValueError("Bernini Scheduler timesteps are missing")
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.as_tensor(timesteps, dtype=torch.float32)
    else:
        timesteps = timesteps.to(dtype=torch.float32)
    if device is not None:
        timesteps = timesteps.to(device)
    if len(timesteps) == 0:
        raise ValueError("Bernini Scheduler produced an empty timestep slice (check steps/start_step/end_step)")
    if torch.isnan(timesteps).any():
        raise ValueError(
            "Bernini Scheduler produced invalid timesteps (NaN). "
            f"Check shift/start_step/end_step; flow-match schedulers require shift > 0 (default {DEFAULT_FLOW_SHIFT})."
        )

    sample_scheduler.timesteps = timesteps
    n = len(timesteps) + 1

    if stored_sigmas is not None:
        sigmas = stored_sigmas
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        else:
            sigmas = sigmas.to(dtype=torch.float32)
        if device is not None:
            sigmas = sigmas.to(device)
        if len(sigmas) != n:
            raise ValueError(
                f"Bernini Scheduler stored sigmas length {len(sigmas)} != timesteps+1 ({n})"
            )
        sample_scheduler.sigmas = sigmas
    elif hasattr(sample_scheduler, "sigmas") and sample_scheduler.sigmas is not None:
        sigmas = sample_scheduler.sigmas
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        else:
            sigmas = sigmas.to(dtype=torch.float32)
        if device is not None:
            sigmas = sigmas.to(device)
        # Custom sgm_uniform / flow-match slices carry len(timesteps)+1 sigmas from get_scheduler.
        if len(sigmas) == n:
            sample_scheduler.sigmas = sigmas
        else:
            sliced = _slice_sigmas_to_timesteps(sample_scheduler, timesteps)
            if sliced is not None and len(sliced) == n:
                sample_scheduler.sigmas = sliced.to(device=timesteps.device, dtype=torch.float32)
            elif _is_flow_match_scheduler(sample_scheduler):
                raise ValueError(
                    f"Flow-match scheduler sigmas length {len(sigmas)} != timesteps+1 ({n}). "
                    "Re-run Bernini Scheduler node; ensure scheduler dict includes matching sigmas."
                )
            else:
                terminal = torch.zeros(1, device=timesteps.device, dtype=torch.float32)
                sample_scheduler.sigmas = torch.cat([timesteps / 1000.0, terminal])
    if hasattr(sample_scheduler, "num_inference_steps"):
        sample_scheduler.num_inference_steps = len(timesteps)
    if hasattr(sample_scheduler, "_step_index"):
        sample_scheduler._step_index = None
    if hasattr(sample_scheduler, "_begin_index"):
        sample_scheduler._begin_index = None
    return sample_scheduler, timesteps


def normalize_wanvideo_schedulerv2_inputs(
    sampler_name,
    scheduler,
    steps,
    shift,
    start_step,
    end_step,
    enhance_hf=False,
):
    """Remap legacy 6-widget WanVideoScheduler values saved into v2 node slots.

    Old layout: [scheduler, steps, shift, start_step, end_step, enhance_hf]
    New layout: [sampler_name, scheduler, steps, shift, start_step, end_step, enhance_hf]

    When an old workflow loads, widget[1] (steps) lands in the scheduler slot as an int,
    shifting all following values and often producing empty timestep slices.
    """
    partial = _try_remap_partial_legacy_v2(
        sampler_name, scheduler, steps, shift, start_step, end_step, enhance_hf
    )
    if partial is not None:
        sampler_name, scheduler, steps, shift, start_step, end_step, enhance_hf = partial
    elif not _is_comfy_scheduler(scheduler):
        try:
            legacy_steps = int(scheduler)
        except (TypeError, ValueError):
            legacy_steps = None
        if legacy_steps is not None:
            legacy_bernini = str(sampler_name or "").strip()
            if legacy_bernini in LEGACY_BERNINI_TO_KSAMPLER or legacy_bernini in SAMPLER_NAME_TO_BERNINI:
                remapped_sampler = legacy_bernini_scheduler_to_sampler_name(legacy_bernini)
                remapped_scheduler = "simple"
                remapped_steps = legacy_steps
                remapped_shift = float(steps)
                remapped_start = int(shift)
                remapped_end = int(start_step)
                remapped_enhance = bool(end_step) if isinstance(end_step, bool) else bool(enhance_hf)
                log.warning(
                    "Bernini Scheduler: detected legacy widget layout; remapped "
                    "%r -> sampler_name=%r scheduler=%r steps=%s shift=%s start=%s end=%s",
                    [sampler_name, scheduler, steps, shift, start_step, end_step],
                    remapped_sampler,
                    remapped_scheduler,
                    remapped_steps,
                    remapped_shift,
                    remapped_start,
                    remapped_end,
                )
                sampler_name = remapped_sampler
                scheduler = remapped_scheduler
                steps = remapped_steps
                shift = remapped_shift
                start_step = remapped_start
                end_step = remapped_end
                enhance_hf = remapped_enhance

    shift = _coerce_flow_shift(shift)
    return sampler_name, scheduler, steps, shift, start_step, end_step, enhance_hf


def resolve_bernini_scheduler(sampler_name: str) -> str:
    """Map Comfy KSampler sampler_name to Bernini/Wan scheduler id."""
    name = (sampler_name or "").strip()
    mapped = SAMPLER_NAME_TO_BERNINI.get(name)
    if mapped is None:
        supported = ", ".join(sorted(k for k in SAMPLER_NAME_TO_BERNINI if not k.endswith("_gpu")))
        raise ValueError(
            f"sampler_name {name!r} has no Bernini/Wan equivalent. "
            f"Supported sampler_name values include: {supported}."
        )
    return mapped


def flowmatch_sgm_uniform_sigmas(steps: int, device: torch.device) -> torch.Tensor:
    timesteps = torch.linspace(1000.0, 0.0, steps + 1, device=device)[:-1]
    return torch.cat((timesteps / 1000.0, torch.zeros(1, device=device)))


def resolve_sigma_schedule(
    *,
    sigmas: torch.Tensor | None,
    scheduler: str,
    steps: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build Comfy KSampler-style sigma curve for Bernini sampling."""
    if sigmas is not None:
        return sigmas

    sched = (scheduler or "").strip() if isinstance(scheduler, str) else str(scheduler or "").strip()
    if not sched:
        return None
    if sched == "sgm_uniform":
        return flowmatch_sgm_uniform_sigmas(steps, device)
    return None
