"""ComfyUI-Bernini custom node entry point."""

from .bernini.nodes.conditioning import BerniniConditioning, BerniniPlannerConditioning
from .bernini.nodes.pipeline import PIPELINE_DISPLAY_NAMES, PIPELINE_NODE_MAPPINGS
from .bernini.nodes.director import BerniniDirector, BerniniDirectorExecute
from .bernini.nodes.wan import BerniniWanContextEmbeds, BerniniWanContextMerge

NODE_CLASS_MAPPINGS = {
    **PIPELINE_NODE_MAPPINGS,
    "BerniniConditioning": BerniniConditioning,
    "BerniniPlannerConditioning": BerniniPlannerConditioning,
    # Legacy aliases (same implementations as pipeline nodes)
    "BerniniWanContextEmbeds": BerniniWanContextEmbeds,
    "BerniniWanContextMerge": BerniniWanContextMerge,
    "BerniniDirector": BerniniDirector,
    "BerniniDirectorExecute": BerniniDirectorExecute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PIPELINE_DISPLAY_NAMES,
    "BerniniConditioning": "Bernini Conditioning",
    "BerniniPlannerConditioning": "Bernini Planner Conditioning",
    "BerniniWanContextEmbeds": "Bernini Wan Context Embeds",
    "BerniniWanContextMerge": "Bernini Wan Context Merge",
    "BerniniDirector": "Bernini Director",
    "BerniniDirectorExecute": "Bernini Director Execute",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-Bernini")

try:
    from .bernini.director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "Bernini Director HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /bernini/director/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("Bernini Director HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
