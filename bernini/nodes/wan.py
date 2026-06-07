"""Bernini context embed nodes (Wan pipeline)."""

from __future__ import annotations

import logging

from ..context_pipeline import SemanticContextPipeline
from ..ref_images import collect_reference_batches, flatten_reference_kwargs, reference_image_input_types
from ..encoders import WanVaeEncoder
from ..task_modes import TASK_DESCRIPTIONS

log = logging.getLogger("ComfyUI-Bernini")


class BerniniWanContextEmbeds:
    """Build WANVIDIMAGE_EMBEDS with Bernini context_latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("WANVAE",),
            },
            "optional": {
                "source_video": (
                    "IMAGE",
                    {"tooltip": "Source video canvas for v2v / rv2v editing."},
                ),
                "reference_video": (
                    "IMAGE",
                    {"tooltip": "Reference motion clip for content insertion (ads2v)."},
                ),
                **reference_image_input_types(),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "ref_max_size": (
                    "INT",
                    {
                        "default": 848,
                        "min": 16,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Max long-edge for reference video and images.",
                    },
                ),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("image_embeds", "task_mode")
    FUNCTION = "build"
    CATEGORY = "Bernini"

    def build(
        self,
        vae,
        width,
        height,
        num_frames,
        source_video=None,
        reference_video=None,
        ref_max_size=848,
        tiled_vae=False,
        force_offload=True,
        **kwargs,
    ):
        ref_batches = collect_reference_batches(None, flatten_reference_kwargs(kwargs))

        pipeline = SemanticContextPipeline(
            WanVaeEncoder(vae, tiled=tiled_vae, force_offload=force_offload)
        )
        result = pipeline.run(
            width=width,
            height=height,
            num_frames=num_frames,
            ref_max_edge=ref_max_size,
            source_video=source_video,
            reference_video=reference_video,
            reference_batches=ref_batches,
        )

        if result.context_latents:
            log.info(
                "Bernini context: task=%s, streams=%d",
                result.task.label,
                result.task.stream_count,
            )

        embeds = {
            "target_shape": result.target_shape,
            "num_frames": num_frames,
            "context_latents": result.context_latents or None,
        }

        task_hint = f"{result.task.label} — {TASK_DESCRIPTIONS[result.task.mode]}"
        return (embeds, task_hint)


class BerniniWanContextMerge:
    """Merge Bernini context_latents into an existing WANVIDIMAGE_EMBEDS dict."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "vae": ("WANVAE",),
            },
            "optional": {
                "source_video": (
                    "IMAGE",
                    {"tooltip": "Source video canvas for v2v / rv2v editing."},
                ),
                "reference_video": (
                    "IMAGE",
                    {"tooltip": "Reference motion clip for content insertion (ads2v)."},
                ),
                **reference_image_input_types(),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "ref_max_size": (
                    "INT",
                    {
                        "default": 848,
                        "min": 16,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Max long-edge for reference video and images.",
                    },
                ),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "replace_existing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Replace existing context_latents when True; append when False.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("image_embeds", "task_mode")
    FUNCTION = "merge"
    CATEGORY = "Bernini"

    def merge(
        self,
        image_embeds,
        vae,
        width,
        height,
        num_frames,
        source_video=None,
        reference_video=None,
        ref_max_size=848,
        tiled_vae=False,
        force_offload=True,
        replace_existing=True,
        **kwargs,
    ):
        ref_batches = collect_reference_batches(None, flatten_reference_kwargs(kwargs))

        pipeline = SemanticContextPipeline(
            WanVaeEncoder(vae, tiled=tiled_vae, force_offload=force_offload)
        )
        result = pipeline.run(
            width=width,
            height=height,
            num_frames=num_frames,
            ref_max_edge=ref_max_size,
            source_video=source_video,
            reference_video=reference_video,
            reference_batches=ref_batches,
        )

        merged = dict(image_embeds)
        if result.context_latents:
            if replace_existing or not merged.get("context_latents"):
                merged["context_latents"] = result.context_latents
            else:
                merged["context_latents"] = list(merged["context_latents"]) + result.context_latents

        merged.setdefault("target_shape", result.target_shape)
        merged.setdefault("num_frames", num_frames)

        task_hint = f"{result.task.label} — {TASK_DESCRIPTIONS[result.task.mode]}"
        return (merged, task_hint)
