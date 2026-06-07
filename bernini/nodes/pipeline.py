"""Bernini-branded ComfyUI nodes wrapping the local inference engine."""

from __future__ import annotations

from ...engine.bernini_core_nodes import (
    WanVideoContextOptions,
    WanVideoDecode,
    WanVideoSetBlockSwap,
)
from .text_encode import BerniniTextEncodeCached
from .t5_config import BerniniT5Config
from ...engine.nodes_model_loading import (
    WanVideoBlockSwap,
    WanVideoLoraSelect,
    WanVideoLoraSelectMulti,
    WanVideoModelLoader,
    WanVideoSetLoRAs,
    WanVideoTorchCompileSettings,
    WanVideoVAELoader,
)
from ...engine.nodes_sampler import (
    WanVideoSamplerExtraArgs,
    WanVideoSchedulerv2,
    WanVideoSamplerv2,
)

from .wan import BerniniWanContextEmbeds, BerniniWanContextMerge

_CATEGORY = "Bernini"


def _rebrand(cls, display_name: str):
    return type(
        display_name.replace(" ", ""),
        (cls,),
        {"__doc__": cls.__doc__, "CATEGORY": _CATEGORY},
    )


BerniniModelLoader = _rebrand(WanVideoModelLoader, "Bernini Model Loader")
BerniniVAELoader = _rebrand(WanVideoVAELoader, "Bernini VAE Loader")
BerniniBlockSwap = _rebrand(WanVideoBlockSwap, "Bernini Block Swap")
BerniniSetBlockSwap = _rebrand(WanVideoSetBlockSwap, "Bernini Set Block Swap")
BerniniLoraSelect = _rebrand(WanVideoLoraSelect, "Bernini LoRA Select")
BerniniLoraSelectMulti = _rebrand(WanVideoLoraSelectMulti, "Bernini LoRA Select Multi")
BerniniSetLoRAs = _rebrand(WanVideoSetLoRAs, "Bernini Set LoRAs")
BerniniTorchCompile = _rebrand(WanVideoTorchCompileSettings, "Bernini Torch Compile")
BerniniContextOptions = _rebrand(WanVideoContextOptions, "Bernini Context Options")
BerniniSamplerExtraArgs = _rebrand(WanVideoSamplerExtraArgs, "Bernini Sampler Extra Args")
BerniniScheduler = _rebrand(WanVideoSchedulerv2, "Bernini Scheduler")
BerniniSampler = _rebrand(WanVideoSamplerv2, "Bernini Sampler")
BerniniDecode = _rebrand(WanVideoDecode, "Bernini Decode")

BerniniContextEmbeds = BerniniWanContextEmbeds
BerniniContextEmbeds.CATEGORY = _CATEGORY
BerniniContextMerge = BerniniWanContextMerge
BerniniContextMerge.CATEGORY = _CATEGORY

PIPELINE_NODE_MAPPINGS = {
    "BerniniModelLoader": BerniniModelLoader,
    "BerniniVAELoader": BerniniVAELoader,
    "BerniniBlockSwap": BerniniBlockSwap,
    "BerniniSetBlockSwap": BerniniSetBlockSwap,
    "BerniniLoraSelect": BerniniLoraSelect,
    "BerniniLoraSelectMulti": BerniniLoraSelectMulti,
    "BerniniSetLoRAs": BerniniSetLoRAs,
    "BerniniTorchCompile": BerniniTorchCompile,
    "BerniniTextEncodeCached": BerniniTextEncodeCached,
    "BerniniT5Config": BerniniT5Config,
    "BerniniContextEmbeds": BerniniContextEmbeds,
    "BerniniContextMerge": BerniniContextMerge,
    "BerniniContextOptions": BerniniContextOptions,
    "BerniniSamplerExtraArgs": BerniniSamplerExtraArgs,
    "BerniniScheduler": BerniniScheduler,
    "BerniniSampler": BerniniSampler,
    "BerniniDecode": BerniniDecode,
}

PIPELINE_DISPLAY_NAMES = {
    "BerniniModelLoader": "Bernini Model Loader",
    "BerniniVAELoader": "Bernini VAE Loader",
    "BerniniBlockSwap": "Bernini Block Swap",
    "BerniniSetBlockSwap": "Bernini Set Block Swap",
    "BerniniLoraSelect": "Bernini LoRA Select",
    "BerniniLoraSelectMulti": "Bernini LoRA Select Multi",
    "BerniniSetLoRAs": "Bernini Set LoRAs",
    "BerniniTorchCompile": "Bernini Torch Compile",
    "BerniniTextEncodeCached": "Bernini Text Encode Cached",
    "BerniniT5Config": "Bernini T5 Config",
    "BerniniContextEmbeds": "Bernini Context Embeds",
    "BerniniContextMerge": "Bernini Context Merge",
    "BerniniContextOptions": "Bernini Context Options",
    "BerniniSamplerExtraArgs": "Bernini Sampler Extra Args",
    "BerniniScheduler": "Bernini Scheduler",
    "BerniniSampler": "Bernini Sampler",
    "BerniniDecode": "Bernini Decode",
}
