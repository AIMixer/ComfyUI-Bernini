"""Bernini T5 encoder configuration bundle (no prompts / task_type)."""

from __future__ import annotations

import folder_paths

_CATEGORY = "Bernini"


class BerniniT5Config:
    """T5 model / precision / cache settings — connect to Bernini Director or other Bernini nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "Loaded from ComfyUI/models/text_encoders"},
                ),
                "precision": (["fp32", "bf16"], {"default": "bf16"}),
                "quantization": (
                    ["disabled", "fp8_e4m3fn"],
                    {"default": "disabled", "tooltip": "Optional T5 quantization."},
                ),
                "use_disk_cache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Cache embeddings under engine/text_embed_cache."},
                ),
                "device": (
                    ["gpu", "cpu"],
                    {"default": "gpu", "tooltip": "Device for T5 text encoding."},
                ),
            },
        }

    RETURN_TYPES = ("BERNINIT5CONFIG",)
    RETURN_NAMES = ("t5_config",)
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Bundle T5 encoder settings (model, precision, quantization, cache, device). "
        "Connect t5_config to Bernini Director — prompts and task_type stay on the Director UI."
    )

    def build(
        self,
        model_name: str,
        precision: str,
        quantization: str = "disabled",
        use_disk_cache: bool = True,
        device: str = "gpu",
    ):
        return (
            {
                "model_name": model_name,
                "precision": precision,
                "quantization": quantization,
                "use_disk_cache": use_disk_cache,
                "device": device,
            },
        )


def resolve_t5_config(t5_config: dict | None) -> dict:
    if not t5_config or not isinstance(t5_config, dict):
        raise ValueError("t5_config is required — connect a Bernini T5 Config node.")
    required = ("model_name", "precision", "quantization", "use_disk_cache", "device")
    missing = [k for k in required if k not in t5_config]
    if missing:
        raise ValueError(f"Invalid t5_config: missing {missing}")
    return t5_config
