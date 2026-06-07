"""High-level Bernini semantic context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .encoders import ContextStreamEncoder, build_context_latents
from .ref_images import MAX_REFERENCE_IMAGES, collect_reference_batches, flatten_reference_kwargs
from .task_modes import TaskSummary, infer_task

__all__ = [
    "SemanticContextPipeline",
    "SemanticContextResult",
    "collect_reference_batches",
    "flatten_reference_kwargs",
    "wan_target_shape",
]


@dataclass
class SemanticContextResult:
    context_latents: list[torch.Tensor]
    task: TaskSummary
    target_shape: tuple[int, int, int, int]


def wan_target_shape(num_frames: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Latent shape (C, F, H, W) for Wan 2.2 Bernini models."""
    return (16, (num_frames - 1) // 4 + 1, height // 8, width // 8)


class SemanticContextPipeline:
    """Prepare Bernini in-context latent streams from multimodal inputs."""

    def __init__(self, encoder: ContextStreamEncoder):
        self._encoder = encoder

    def run(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        ref_max_edge: int,
        source_video: torch.Tensor | None = None,
        reference_video: torch.Tensor | None = None,
        reference_batches: Sequence[torch.Tensor] | None = None,
    ) -> SemanticContextResult:
        ref_batches = list(reference_batches or [])
        ref_image_count = sum(batch.shape[0] for batch in ref_batches)
        if ref_image_count > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Bernini supports at most {MAX_REFERENCE_IMAGES} reference images "
                f"(image0–image{MAX_REFERENCE_IMAGES - 1}); got {ref_image_count}."
            )

        task = TaskSummary(
            mode=infer_task(
                source_video is not None,
                reference_video is not None,
                ref_image_count,
            ),
            stream_count=0,
            has_source=source_video is not None,
            has_ref_video=reference_video is not None,
            ref_image_count=ref_image_count,
        )

        latents = build_context_latents(
            self._encoder,
            source_video=source_video,
            reference_video=reference_video,
            reference_images=ref_batches,
            width=width,
            height=height,
            frame_limit=num_frames,
            ref_max_edge=ref_max_edge,
        )

        task = TaskSummary(
            mode=task.mode,
            stream_count=len(latents),
            has_source=task.has_source,
            has_ref_video=task.has_ref_video,
            ref_image_count=task.ref_image_count,
        )

        return SemanticContextResult(
            context_latents=latents,
            task=task,
            target_shape=wan_target_shape(num_frames, height, width),
        )
