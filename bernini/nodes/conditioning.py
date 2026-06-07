"""ComfyUI native conditioning nodes for Bernini."""

from __future__ import annotations

import torch

import node_helpers
from comfy import model_management as mm

from ..context_pipeline import (
    SemanticContextPipeline,
    collect_reference_batches,
    flatten_reference_kwargs,
)
from ..encoders import ComfyCoreEncoder
from ..ref_images import reference_image_input_types
from ..task_modes import TASK_DESCRIPTIONS


def _shared_optional_inputs() -> dict:
    return {
        "source_video": (
            "IMAGE",
            {
                "tooltip": (
                    "Edit base / canvas video (v2v, rv2v). "
                    "Resized to width × height and trimmed to length."
                ),
            },
        ),
        "reference_video": (
            "IMAGE",
            {
                "tooltip": (
                    "Moving content to insert into the source (ads2v). "
                    "Long edge capped at ref_max_edge, native aspect kept."
                ),
            },
        ),
        **reference_image_input_types(),
        "ref_max_edge": (
            "INT",
            {
                "default": 848,
                "min": 16,
                "max": 8192,
                "step": 16,
                "tooltip": "Max long-edge for reference video and images.",
            },
        ),
        "ref_max_size": (
            "INT",
            {
                "default": 848,
                "min": 16,
                "max": 8192,
                "step": 16,
                "tooltip": "Alias of ref_max_edge.",
            },
        ),
    }


def _run_conditioning(
    positive,
    negative,
    vae,
    width,
    height,
    length,
    batch_size,
    source_video=None,
    reference_video=None,
    ref_max_edge=848,
    ref_max_size=None,
    **kwargs,
):
    if ref_max_size is not None:
        ref_max_edge = ref_max_size

    extra_refs = flatten_reference_kwargs(kwargs)
    ref_batches = collect_reference_batches(None, extra_refs)

    pipeline = SemanticContextPipeline(ComfyCoreEncoder(vae))
    result = pipeline.run(
        width=width,
        height=height,
        num_frames=length,
        ref_max_edge=ref_max_edge,
        source_video=source_video,
        reference_video=reference_video,
        reference_batches=ref_batches,
    )

    latent = torch.zeros(
        [batch_size, *result.target_shape],
        device=mm.intermediate_device(),
    )

    if result.context_latents:
        payload = {"context_latents": result.context_latents}
        positive = node_helpers.conditioning_set_values(positive, payload)
        negative = node_helpers.conditioning_set_values(negative, payload)

    task_hint = f"{result.task.label} — {TASK_DESCRIPTIONS[result.task.mode]}"
    if result.task.stream_count:
        task_hint += f" ({result.task.stream_count} stream(s))"

    return positive, negative, {"samples": latent}, task_hint


class BerniniConditioning:
    """Drop-in alias for kijai PR #14216 BerniniConditioning (3 outputs)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": _shared_optional_inputs(),
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "conditioning/video_models"

    def apply(self, *args, **kwargs):
        positive, negative, latent, _ = _run_conditioning(*args, **kwargs)
        return positive, negative, latent


class BerniniPlannerConditioning:
    """Attach Bernini in-context latents to ComfyUI conditioning.

    Compatible with native Wan 2.2 Bernini support (see Comfy-Org/ComfyUI#14216).
    Streams are ordered: source video → reference video → reference images,
    each receiving a sequential source_id for segment-aware 3D RoPE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": _shared_optional_inputs(),
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "task_mode")
    FUNCTION = "apply"
    CATEGORY = "Bernini"

    def apply(self, *args, **kwargs):
        return _run_conditioning(*args, **kwargs)
