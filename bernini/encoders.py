"""VAE encoding adapters for ComfyUI core and Wan WANVAE backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from comfy import model_management as mm

from .image_prep import fit_canvas, fit_long_edge, normalize_to_vae_range


class ContextStreamEncoder(ABC):
    """Encodes visual inputs into Bernini context latent streams."""

    @abstractmethod
    def encode_source_video(
        self, frames: torch.Tensor, width: int, height: int, frame_limit: int
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_reference_video(
        self, frames: torch.Tensor, max_edge: int, frame_limit: int
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_reference_image(self, image: torch.Tensor, max_edge: int) -> torch.Tensor:
        raise NotImplementedError

    def encode_reference_batch(
        self, images: torch.Tensor, max_edge: int
    ) -> list[torch.Tensor]:
        return [self.encode_reference_image(images[i : i + 1], max_edge) for i in range(images.shape[0])]

    def offload(self) -> None:
        """Release GPU memory held by the underlying VAE, if applicable."""


class ComfyCoreEncoder(ContextStreamEncoder):
    """Encoder using ComfyUI built-in VAE (for native Wan / Bernini conditioning)."""

    def __init__(self, vae):
        self._vae = vae

    def encode_source_video(
        self, frames: torch.Tensor, width: int, height: int, frame_limit: int
    ) -> torch.Tensor:
        clipped = fit_canvas(frames[:frame_limit], width, height)
        return self._vae.encode(clipped)

    def encode_reference_video(
        self, frames: torch.Tensor, max_edge: int, frame_limit: int
    ) -> torch.Tensor:
        clipped = fit_long_edge(frames[:frame_limit], max_edge)
        return self._vae.encode(clipped)

    def encode_reference_image(self, image: torch.Tensor, max_edge: int) -> torch.Tensor:
        resized = fit_long_edge(image, max_edge)
        return self._vae.encode(resized)

    def offload(self) -> None:
        pass


class WanVaeEncoder(ContextStreamEncoder):
    """Encoder using the plugin WANVAE (Wan 2.1 VAE)."""

    def __init__(self, vae, tiled: bool = False, force_offload: bool = True):
        self._vae = vae
        self._tiled = tiled
        self._force_offload = force_offload
        self._device = mm.get_torch_device()
        self._offload_device = mm.unet_offload_device()

    def _run_encode(self, tensor_c_f_h_w: torch.Tensor) -> torch.Tensor:
        self._vae.to(self._device)
        latent = self._vae.encode(
            [tensor_c_f_h_w.to(device=self._device, dtype=self._vae.dtype)],
            self._device,
            tiled=self._tiled,
        )[0]
        if self._force_offload:
            self._vae.to(self._offload_device)
        return latent.to(self._offload_device)

    def _video_to_vae_input(self, frames: torch.Tensor) -> torch.Tensor:
        return normalize_to_vae_range(frames).permute(3, 0, 1, 2)

    def encode_source_video(
        self, frames: torch.Tensor, width: int, height: int, frame_limit: int
    ) -> torch.Tensor:
        canvas = fit_canvas(frames[:frame_limit], width, height)
        return self._run_encode(self._video_to_vae_input(canvas))

    def encode_reference_video(
        self, frames: torch.Tensor, max_edge: int, frame_limit: int
    ) -> torch.Tensor:
        resized = fit_long_edge(frames[:frame_limit], max_edge)
        return self._run_encode(self._video_to_vae_input(resized))

    def encode_reference_image(self, image: torch.Tensor, max_edge: int) -> torch.Tensor:
        resized = fit_long_edge(image, max_edge)
        return self._run_encode(self._video_to_vae_input(resized))

    def offload(self) -> None:
        if self._force_offload:
            self._vae.to(self._offload_device)


def build_context_latents(
    encoder: ContextStreamEncoder,
    *,
    source_video: torch.Tensor | None,
    reference_video: torch.Tensor | None,
    reference_images: Sequence[torch.Tensor],
    width: int,
    height: int,
    frame_limit: int,
    ref_max_edge: int,
) -> list[torch.Tensor]:
    """Build ordered Bernini context streams: source → ref video → ref images."""
    streams: list[torch.Tensor] = []

    if source_video is not None:
        streams.append(encoder.encode_source_video(source_video, width, height, frame_limit))

    if reference_video is not None:
        streams.append(encoder.encode_reference_video(reference_video, ref_max_edge, frame_limit))

    for batch in reference_images:
        if batch is None or batch.shape[0] == 0:
            continue
        streams.extend(encoder.encode_reference_batch(batch, ref_max_edge))

    encoder.offload()
    return streams
