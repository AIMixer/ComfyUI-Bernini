"""Bernini guidance_mode values (aligned with official bytedance/Bernini bernini/cli.py)."""

# Official Bernini renderer CLI / case-file modes.
OFFICIAL_GUIDANCE_MODES = ["rv2v", "v2v", "v2v_chain", "t2v", "r2v_apg", "v2v_apg", "t2v_apg"]

# ComfyUI dropdown: disable guidance, generic APG alias, then official task modes.
BERNINI_GUIDANCE_MODES = ["none", "apg", *OFFICIAL_GUIDANCE_MODES]

APG_GUIDANCE_MODES = frozenset({"apg", "r2v_apg", "v2v_apg", "t2v_apg"})


def uses_apg_guidance(guidance_mode: str) -> bool:
    """True when sampler should apply Adaptive Projected Guidance."""
    return guidance_mode in APG_GUIDANCE_MODES
