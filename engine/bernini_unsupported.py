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


def reject_pose_face_model():
    raise ValueError(
        "Pose/face animation models are not supported in ComfyUI-Bernini."
    )


def reject_pose_face_inputs(image_embeds):
    if not image_embeds:
        return
    blocked = {
        "looping": "seamless looping",
        "pose_latents": "pose latents",
        "face_pixels": "face pixels",
        "ref_masks": "reference masks",
        "pose_images": "pose images",
        "bg_images": "background images",
        "start_ref_image": "loop start reference",
    }
    for key, label in blocked.items():
        value = image_embeds.get(key)
        if value is not None and value is not False:
            raise ValueError(
                f"{label.capitalize()} inputs are not supported in ComfyUI-Bernini."
            )
    if image_embeds.get("is_masked"):
        raise ValueError("Masked pose/face inputs are not supported in ComfyUI-Bernini.")


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
    if "pose_patch_embedding.weight" in sd:
        reject_pose_face_model()
    if any("face_adapter.fuser_blocks" in k for k in keys):
        reject_pose_face_model()
    if any(k.startswith("face_encoder.") for k in keys):
        reject_pose_face_model()
