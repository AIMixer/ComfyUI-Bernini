"""Bernini Director Official — timeline UI + ComfyUI core Bernini execution."""

from __future__ import annotations

import comfy.samplers

from ..director.executor_core import execute_director_plan_core
from .director_common import (
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
    director_perf_inputs,
)

_CATEGORY = "Bernini"

# Defaults aligned with ComfyUI PR #14216 attachment
# `Bernini_testing_video_edit_02.json` (rv2v + source video + reference image).
_OFFICIAL_GLOBAL_PROMPT = "An old man is digging with a shovel"
_OFFICIAL_NEGATIVE_PROMPT = "bad video"


def official_timeline_required_inputs() -> dict:
    """Timeline widgets for Official node — defaults match the core Bernini rv2v example."""
    inputs = timeline_required_inputs()
    combo_options, combo_meta = inputs["task_type"]

    gp_meta = dict(inputs["global_prompt"][1])
    gp_meta["default"] = _OFFICIAL_GLOBAL_PROMPT
    gp_meta["tooltip"] = (
        "User prompt only — rv2v system prefix is prepended automatically "
        "(same as official Bernini_testing_video_edit_02.json)."
    )

    neg_meta = dict(inputs["negative_prompt"][1])
    neg_meta["default"] = _OFFICIAL_NEGATIVE_PROMPT

    frames_meta = dict(inputs["total_frames"][1])
    frames_meta["tooltip"] = (
        "Frame count from uploaded source video / timeline UI. "
        "Official rv2v example uses 145 frames; empty t2v canvas default is 81."
    )

    return {
        **inputs,
        "task_type": (combo_options, combo_meta),
        "global_prompt": ("STRING", gp_meta),
        "negative_prompt": ("STRING", neg_meta),
        "total_frames": ("INT", frames_meta),
    }


class BerniniDirectorOfficial:
    """Same in-node timeline as Bernini Director; uses ComfyUI native Bernini + KSampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Standard VAE (VAELoader). Official rv2v example: wan_2.1_vae / "
                            "Wan2_1_VAE_bf16; connect externally."
                        ),
                    },
                ),
                "model_high": (
                    "MODEL",
                    {"tooltip": "High-noise Bernini-R UNET (e.g. Wan22_Bernini_HIGH_fp8_e4m3fn_scaled)."},
                ),
                "model_low": (
                    "MODEL",
                    {"tooltip": "Low-noise Bernini-R UNET (e.g. Wan22_Bernini_LOW_fp8_e4m3fn_scaled)."},
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "CLIPLoader type wan — official example: umt5_xxl_fp8_e4m3fn_scaled."
                        ),
                    },
                ),
                **official_timeline_required_inputs(),
            },
            "optional": {
                "bd_grp_sample": ("BDGROUP", {"default": "官方采样"}),
                "steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Total steps — official rv2v example: BasicScheduler 6.",
                    },
                ),
                "split_step": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 199,
                        "tooltip": "High/low split — official rv2v example: SplitSigmas at 3.",
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "res_multistep",
                        "tooltip": "Official rv2v example: KSamplerSelect res_multistep.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": "Official rv2v example: BasicScheduler simple.",
                    },
                ),
                "model_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "ModelSamplingSD3 shift. Official Bernini rv2v chain does not patch shift "
                            "(leave 0). Wan 2.2 generic t2v blueprints often use 5.0."
                        ),
                    },
                ),
                "apg_eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "APG parallel/orthogonal mix — official rv2v example has no APG node.",
                    },
                ),
                "apg_momentum": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -5.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "APG EMA momentum — 0 disables smoothing (official default).",
                    },
                ),
                "apg_norm_threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "tooltip": "APG L2 clip threshold — 0 disables APG (official rv2v has no APG).",
                    },
                ),
                **director_perf_inputs(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, steps=6, split_step=3, **_kwargs):
        # Linked inputs (vae, model_*, clip) are None during prompt validation — ComfyUI
        # already checks required links and return-type matching before this runs.
        if int(split_step) >= int(steps):
            return "split_step must be less than steps."
        if input_types is not None:
            expected = {
                "vae": "VAE",
                "model_high": "MODEL",
                "model_low": "MODEL",
                "clip": "CLIP",
            }
            for name, want in expected.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
        return True

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "audio", "frame_count", "report", "source_images", "fps")
    OUTPUT_IS_LIST = (True, True, False, False, True, False)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Bernini Director with ComfyUI official Bernini path: CLIP text encode + BerniniConditioning "
        "+ dual-stage KSampler + VAEDecode. Defaults follow Bernini_testing_video_edit_02.json (rv2v). "
        "source_images and fps outputs match the KJ Director node."
    )

    def execute(
        self,
        vae,
        model_high,
        model_low,
        clip,
        task_type,
        global_prompt,
        negative_prompt,
        high_noise_cfg,
        high_noise_seed,
        low_noise_cfg,
        low_noise_seed,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        steps=6,
        split_step=3,
        sampler="res_multistep",
        scheduler="simple",
        model_shift=0.0,
        apg_eta=1.0,
        apg_momentum=0.0,
        apg_norm_threshold=0.0,
        clear_vram_between_segments=True,
        export_source_images=False,
        **kwargs,
    ):
        del kwargs  # bd_grp_* headers

        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
        )

        combined, segment_outputs, report = execute_director_plan_core(
            plan,
            node_id=unique_id,
            vae=vae,
            model_high=model_high,
            model_low=model_low,
            clip=clip,
            negative_prompt=negative_prompt,
            high_noise_cfg=high_noise_cfg,
            high_noise_seed=high_noise_seed,
            low_noise_cfg=low_noise_cfg,
            low_noise_seed=low_noise_seed,
            steps=steps,
            split_step=split_step,
            sampler=sampler,
            scheduler=scheduler,
            model_shift=model_shift,
            apg_eta=apg_eta,
            apg_momentum=apg_momentum,
            apg_norm_threshold=apg_norm_threshold,
            clear_vram_between_segments=clear_vram_between_segments,
        )

        return finalize_director_outputs(
            plan,
            combined,
            segment_outputs,
            report,
            export_source_images=export_source_images,
        )
