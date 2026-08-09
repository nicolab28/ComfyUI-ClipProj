"""ClipProj -- swap a large text encoder for a small one plus a learned projection.

EXPERIMENTAL. See README.md for the method, the measured results and the limits.
"""

from .clipproj_nodes import (ClipProjApply, ClipProjDeviceLoader, ClipProjFree,
                             ClipProjGenerate, ClipProjLoader)
from .clipproj_projection import register_folder

register_folder()

NODE_CLASS_MAPPINGS = {
    "ClipProjDeviceLoader": ClipProjDeviceLoader,
    "ClipProjApply": ClipProjApply,
    "ClipProjLoader": ClipProjLoader,
    "ClipProjGenerate": ClipProjGenerate,
    "ClipProjFree": ClipProjFree,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipProjDeviceLoader": "ClipProj Device Loader",
    "ClipProjApply": "ClipProj Apply",
    "ClipProjLoader": "ClipProj Loader (all-in-one)",
    "ClipProjGenerate": "ClipProj Generate / Caption",
    "ClipProjFree": "ClipProj Free VRAM",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
