"""Explicit errors for WanVideo features not bundled in ComfyUI-Bernini."""


def reject_echoshot():
    raise ValueError(
        "EchoShot multi-shot prompting is not supported in ComfyUI-Bernini."
    )


def reject_multitalk():
    raise ValueError(
        "MultiTalk / InfiniteTalk / LongCat-Avatar audio is not supported in ComfyUI-Bernini."
    )


def reject_mtv_motion():
    raise ValueError(
        "MTV Crafter motion control is not supported in ComfyUI-Bernini."
    )


def check_state_dict_for_unsupported(sd: dict, multitalk_model=None) -> None:
    keys = sd.keys()
    if multitalk_model is not None:
        reject_multitalk()
    if any("multitalk_audio_proj" in k for k in keys) and any(
        "audio_cross_attn" in k for k in keys
    ):
        reject_multitalk()
    if "blocks.1.audio_cross_attn.kv_linear.weight" in sd and "audio_proj.proj1.weight" in sd:
        reject_multitalk()
    if any("blocks.0.motion_attn." in k for k in keys):
        reject_mtv_motion()
    if "LQ_proj_in.norm1.gamma" in sd:
        raise ValueError(
            "FlashVSR models are not supported in ComfyUI-Bernini."
        )
