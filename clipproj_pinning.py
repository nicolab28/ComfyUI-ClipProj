"""Pin a ModelPatcher so ComfyUI cannot move its weights.

Required whenever a model is loaded with offload_device equal to load_device.
In that configuration an unload frees nothing -- the weights are already "on" the
offload device -- yet ComfyUI removes the model from its memory accounting. It
then believes it has more VRAM than it really does and eventually OOMs.
Neutralising the unload entry points stops it from trying.

Deliberately minimal: the patcher's class is swapped for a subclass whose unload
methods do nothing. No change to ComfyUI core, and the object stays a
ModelPatcher in every other respect.
"""

import gc
import logging
import weakref

import torch

import comfy.model_management as mm

_PINNED_CLASSES = {}
_ACTIVE = {}


def _pinned_class(cls):
    """Build (and memoise) the inert subclass matching `cls`."""
    if cls in _PINNED_CLASSES:
        return _PINNED_CLASSES[cls]

    def model_unload(self, *args, **kwargs):
        return False  # "freed nothing": ComfyUI moves on to the next model

    def partially_unload(self, *args, **kwargs):
        return 0

    def partially_unload_ram(self, *args, **kwargs):
        return 0  # would push weights back to RAM: exactly what we prevent

    def detach(self, *args, **kwargs):
        return self.model

    pinned = type("Pinned" + cls.__name__, (cls,), {
        "model_unload": model_unload,
        "partially_unload": partially_unload,
        "partially_unload_ram": partially_unload_ram,
        "detach": detach,
        "_clipproj_pinned": True,
        "_clipproj_origin": cls,
    })
    _PINNED_CLASSES[cls] = pinned
    return pinned


def unload_patcher(patcher, label=""):
    """Unpin a patcher, then actually free the VRAM it holds.

    Unpinning alone is not enough: when offload_device equals load_device,
    ComfyUI's unload moves nothing and the card stays full. So the offload target
    is switched to RAM before unloading, and the model is removed from the loaded
    model list.

    The weights are deliberately moved rather than destroyed. Releasing their
    storage in place would free the card without touching the bus, but it also
    leaves a model ComfyUI can never restore: it keeps its own staged copy and
    expects to write it back into those very tensors, which then no longer have a
    shape. A Free node placed downstream of a loader would turn a silent reload
    into a hard failure.

    Args:
        patcher: the ModelPatcher to release.
        label (str): human-readable name for the log.

    Returns:
        float: GB freed, as reported by the patcher.
    """
    if patcher is None:
        return 0.0
    if getattr(patcher, "_clipproj_pinned", False):
        patcher.__class__ = patcher._clipproj_origin

    try:
        size = patcher.model_size() / 1024 ** 3
    except Exception:
        size = 0.0

    try:
        if patcher.offload_device == patcher.load_device:
            patcher.offload_device = torch.device("cpu")
    except Exception:
        pass

    freed = False
    for lm in list(mm.current_loaded_models):
        if getattr(lm, "model", None) is patcher:
            try:
                lm.model_unload()
                freed = True
            except Exception as e:
                logging.warning("[ClipProj] partial unload of %s: %s", label, e)
            try:
                mm.current_loaded_models.remove(lm)
            except ValueError:
                pass

    gc.collect()
    mm.soft_empty_cache(force=True)
    if freed and label:
        logging.info("[ClipProj] %s unloaded (%.2f GB freed)", label, size)
    return size if freed else 0.0


def release_role(role, key=None):
    """Free the model holding `role`, unless it already matches `key`."""
    entry = _ACTIVE.get(role)
    if entry is None:
        return
    prev_key, ref, label = entry
    if key is not None and prev_key == key:
        return
    unload_patcher(ref(), label)
    _ACTIVE.pop(role, None)


def release_all():
    """Free every model pinned by ClipProj.

    Returns:
        tuple[int, float]: (models freed, GB freed).
    """
    count, total = 0, 0.0
    for role in list(_ACTIVE.keys()):
        _, ref, label = _ACTIVE[role]
        got = unload_patcher(ref(), label)
        if got > 0:
            count += 1
            total += got
        _ACTIVE.pop(role, None)
    return count, total


def pin_patcher(patcher, label="", role=None, key=None):
    """Make `patcher` untouchable by ComfyUI's memory manager.

    Args:
        patcher: the ModelPatcher to pin.
        label (str): human-readable name for the log.
        role (str|None): exclusive category. Whatever was pinned under this role
            is released first, unless `key` is identical.
        key (str|None): model identifier, typically its filename.

    Returns:
        The patcher, pinned.
    """
    if patcher is None:
        return patcher
    if role is not None:
        # Compare the live object, not its name. Matching keys used to mean
        # "already loaded, nothing to release", but ComfyUI re-runs a loader node
        # whenever its cache is dropped -- producing a second copy of the very
        # same checkpoint while the first stays pinned, hence unfreeable, and
        # losing the only reference to it. Successive runs then filled the card.
        prev = _ACTIVE.get(role)
        if prev is not None and prev[1]() is not patcher:
            unload_patcher(prev[1](), prev[2])
            _ACTIVE.pop(role, None)
    if not getattr(patcher, "_clipproj_pinned", False):
        patcher.__class__ = _pinned_class(patcher.__class__)
        if label:
            logging.info("[ClipProj] %s pinned on %s (ComfyUI will not move it)",
                         label, patcher.load_device)
    if role is not None:
        _ACTIVE[role] = (key, weakref.ref(patcher), label or role)
    return patcher


def is_pinned(patcher):
    """Whether `patcher` is currently pinned."""
    return getattr(patcher, "_clipproj_pinned", False)
