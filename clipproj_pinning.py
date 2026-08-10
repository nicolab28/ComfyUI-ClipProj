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

    if not freed:
        # ComfyUI ne connait plus ce modele -- il l'a retire de sa liste alors
        # que nos methodes inertes l'empechaient de liberer quoi que ce soit.
        # Personne ne le deplacera donc jamais, et ses poids resteraient sur la
        # carte indefiniment. C'est le cas quand le loader se reexecute avec la
        # meme configuration : une seconde copie apparait et la premiere devient
        # orpheline. On la ramene en RAM soi-meme, sans detruire son stockage,
        # pour que le modele reste restaurable.
        try:
            modele = getattr(patcher, "model", None)
            if modele is not None and hasattr(modele, "to"):
                modele.to(device=torch.device("cpu"))
                freed = True
                if label:
                    logging.info("[ClipProj] %s moved back to RAM (%.2f GB), "
                                 "ComfyUI had lost track of it", label, size)
        except Exception as e:
            logging.warning("[ClipProj] could not move %s back to RAM: %s", label, e)

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
    prev_key, ref, label = entry[0], entry[1], entry[2]
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
        ref, label = _ACTIVE[role][1], _ACTIVE[role][2]
        got = unload_patcher(ref(), label)
        if got > 0:
            count += 1
            total += got
        _ACTIVE.pop(role, None)
    return count, total


def release_absent(node_ids):
    """Free every pinned model whose loader node has left the graph.

    A role is named after the node that created it, so a role whose node no
    longer feeds any output has no owner left and nothing will ever release it:
    the loader only frees the previous occupant when it runs again, and a node
    that was disconnected, muted or deleted never runs.

    Args:
        node_ids (set[str]): ids of the nodes that actually feed an output,
            as computed by _atteignables -- not merely the ids present in the
            submitted graph, which still include disconnected nodes.

    Returns:
        int: number of models released.
    """
    n = 0
    actifs = list(_ACTIVE)
    for role in actifs:
        _, sep, node = role.partition(":")
        if sep and node not in node_ids:
            release_role(role)
            n += 1
    if actifs:
        logging.info("[%s] graph watch: %d pinned (%s), %d reachable nodes, "
                     "%d released", 'ClipProj', len(actifs),
                     ", ".join(r.partition(":")[2] for r in actifs),
                     len(node_ids), n)
    return n


def _atteignables(prompt, sorties):
    """Noeuds dont la sortie sert reellement, en remontant depuis les sorties.

    Presence et utilite sont deux choses differentes : ComfyUI transmet tous les
    noeuds qui ne sont pas en sourdine, y compris ceux dont plus rien ne consomme
    le resultat. Se contenter de verifier qu'un identifiant figure dans le graphe
    ne libere donc jamais un noeud simplement debranche, qui est pourtant le cas
    courant.

    Args:
        prompt (dict): graphe au format API, indexe par identifiant de noeud.
        sorties (list): identifiants des noeuds terminaux a executer.

    Returns:
        set[str]: identifiants atteignables. Tout le graphe si les sorties sont
            inconnues, pour ne rien liberer sur une supposition.
    """
    if not sorties:
        return {str(k) for k in prompt}
    vus, pile = set(), [str(s) for s in sorties]
    while pile:
        n = pile.pop()
        if n in vus or n not in prompt:
            continue
        vus.add(n)
        entrees = prompt[n].get("inputs") or {}
        for v in entrees.values():
            # Un lien est [identifiant_source, index_de_sortie] ; une valeur
            # litterale ne l'est pas.
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)):
                pile.append(str(v[0]))
    return vus


def install_graph_watch(flag="_clipproj_graph_watch"):
    """Release orphaned models when a graph starts running.

    Hooked on the executor rather than on the prompt submission handler: the
    latter fires when a job is queued, which may well be while another job is
    still using the very model we would be freeing. The executor runs the jobs
    one after the other, so by the time it is called nothing else holds the card.

    Args:
        flag (str): attribute marking the executor as already hooked, so that
            two packages sharing this module do not wrap it twice.
    """
    try:
        import execution
    except ImportError:  # not running inside ComfyUI
        return
    cls = getattr(execution, "PromptExecutor", None)
    if cls is None or getattr(cls, flag, False):
        return
    original = getattr(cls, "execute_async", None)
    if original is None:  # older ComfyUI, or the method was renamed
        return

    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        try:
            release_absent(_atteignables(prompt, execute_outputs))
        except Exception as e:
            logging.warning("[ClipProj] releasing orphaned models: %s", e)
        return await original(self, prompt, prompt_id, extra_data, execute_outputs)

    cls.execute_async = execute_async
    setattr(cls, flag, True)
    logging.info("[%s] graph watch installed", 'ClipProj')


def install_unload_hook(flag="_clipproj_unload_hook"):
    """Make ComfyUI's global unload reach the pinned models too.

    "Free model and node cache", and the same request sent between two jobs,
    both end up in unload_all_models(). A pinned model ignores it by design --
    that is the whole point of pinning -- so the button silently does nothing for
    the encoder, which is not what anyone pressing it expects. Unpinning first
    restores the expected behaviour without weakening the protection during a
    run: this path is only ever taken when the user asks for it.
    """
    original = getattr(mm, "unload_all_models", None)
    if original is None or getattr(original, flag, False):
        return

    def unload_all_models():
        release_all()
        return original()

    setattr(unload_all_models, flag, True)
    mm.unload_all_models = unload_all_models
    logging.info("[%s] unload hook installed", 'ClipProj')


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
    carte = getattr(patcher, "load_device", None)
    if carte is not None:
        for autre in list(_ACTIVE):
            if autre == role:
                continue
            e = _ACTIVE[autre]
            if len(e) > 3 and e[3] == carte and e[1]() is not patcher:
                logging.info("[ClipProj] releasing %s: another encoder is taking %s",
                             e[2], carte)
                unload_patcher(e[1](), e[2])
                _ACTIVE.pop(autre, None)

    if not getattr(patcher, "_clipproj_pinned", False):
        patcher.__class__ = _pinned_class(patcher.__class__)
        if label:
            logging.info("[ClipProj] %s pinned on %s (ComfyUI will not move it)",
                         label, patcher.load_device)
    if role is not None:
        _ACTIVE[role] = (key, weakref.ref(patcher), label or role,
                         getattr(patcher, "load_device", None))
    return patcher


def is_pinned(patcher):
    """Whether `patcher` is currently pinned."""
    return getattr(patcher, "_clipproj_pinned", False)
