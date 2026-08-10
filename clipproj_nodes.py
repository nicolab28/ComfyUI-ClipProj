"""ClipProj nodes: a small text encoder projected into a large one's space.

The wrapper exposes an object that behaves like the CLIP the diffusion model
expects -- same tokenisation, same conditioning shape, same extra keys. It
therefore drops into the existing "clip" input without rewiring the graph.

The tokenisation implemented here is MiniMax H3's: raw text, no chat template and
no special tokens, with vision blocks spliced in as
"<Picture i>: " + <|vision_start|> + embeddings + <|vision_end|>. That is the only
model pair validated so far.
"""

import json
import logging
import weakref
import os
import struct
import time

import torch

import comfy.model_management as mm
import comfy.ops
import comfy.sd
import folder_paths

from .clipproj_pinning import (pin_patcher, release_all, release_device,
                              release_role)
from .clipproj_projection import (build_control, build_residual, guess_cond_dim,
                                  list_projections, load_projection)

PAD_TOKEN = 151643
IM_START = 151644
IM_END = 151645
VISION_START = 151652
VISION_END = 151653


def gpu_devices():
    """Available devices as 'cuda:N', falling back to cpu."""
    if not torch.cuda.is_available():
        return ["cpu"]
    return ["cuda:%d" % i for i in range(torch.cuda.device_count())] + ["cpu"]


# Hidden size of the vision merger output -> the CLIPType that instantiates the
# matching Qwen3-VL architecture. Read straight from the safetensors header, so
# quantised variants (fp8, nvfp4, int8_convrot) are recognised just the same.
_ARCH_BY_DIM = {2560: ("krea2", "4B"), 4096: ("boogu", "8B"),
                5120: ("minimax", "32B")}
_MERGER_KEYS = ("model.visual.merger.linear_fc2.weight",
                "visual.merger.linear_fc2.weight")


def clip_types():
    """Encoder types known to ComfyUI, 'auto' first."""
    names = sorted(t.name.lower() for t in comfy.sd.CLIPType)
    for first in ("minimax", "boogu", "krea2"):
        if first in names:
            names.remove(first)
            names.insert(0, first)
    return ["auto"] + names


def detect_arch(path):
    """Read the safetensors header to identify the Qwen3-VL variant.

    Args:
        path (str): path to the encoder checkpoint.

    Returns:
        tuple[str, str]|None: (clip type, human label) or None if unrecognised.
    """
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
    except Exception:
        return None
    for key in _MERGER_KEYS:
        entry = header.get(key)
        if entry and entry.get("shape"):
            return _ARCH_BY_DIM.get(entry["shape"][0])
    return None


def _submodel(clip):
    """Return the inner SDClipModel, bypassing the TEModel overrides.

    Specialised TEModels (Krea2, Mage...) override encode_token_weights to strip
    a template or stack several layers. We address the transformer directly to
    get the raw hidden state.
    """
    csm = clip.cond_stage_model
    name = getattr(csm, "clip", None)
    if name is not None and hasattr(csm, name):
        return getattr(csm, name)
    for attr in dir(csm):
        if attr.startswith("_"):
            continue
        sub = getattr(csm, attr, None)
        if hasattr(sub, "encode_token_weights") and hasattr(sub, "transformer"):
            return sub
    raise RuntimeError("No sub-model found in the provided CLIP")


def _raw_tokenizer(clip):
    """Return the CLIP's underlying HuggingFace tokenizer."""
    tk = clip.tokenizer
    name = getattr(tk, "clip_name", None)
    if name is not None and hasattr(tk, name):
        sub = getattr(tk, name)
        if hasattr(sub, "tokenizer"):
            return sub.tokenizer
    for attr in dir(tk):
        if attr.startswith("_"):
            continue
        sub = getattr(tk, attr, None)
        if hasattr(sub, "tokenizer"):
            return sub.tokenizer
    raise RuntimeError("No tokenizer found in the provided CLIP")


def tags_from_embeds_info(seq_len, embeds_info):
    """Tag vision positions 0 and text 1, the way MiniMax H3 does.

    The whole vision block carries tag 0, including the flanking
    <|vision_start|> and <|vision_end|> tokens, hence the one-position widening
    on each side. These tags drive the DiT's adaLN.
    """
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags



def install_video_blocks(sm):
    """Teach a student encoder to read MiniMax H3's two-frame video blocks.

    ComfyUI implements that path on MiniMaxQwen3VL, a subclass reserved for the
    32B. A 4B or an 8B is a plain Qwen3VL and would take the pair for a single
    image, with the wrong grid and the wrong token count -- silently. The method
    is therefore replaced on the instance, and delegates to the original for
    everything that is not a video block.
    """
    tr = getattr(sm, "transformer", None)
    if tr is None or getattr(tr, "_clipproj_video", False):
        return
    try:
        from comfy.text_encoders.minimax import process_video_block
    except Exception as e:
        logging.warning("[ClipProj] video blocks unavailable: %s", e)
        return
    original = tr.preprocess_embed

    def preprocess_embed(embed, device):
        if embed.get("type") == "image" and embed.get("minimax_video_block", False):
            flatten, grid = process_video_block(embed["data"])
            merged, deepstack = tr.visual(flatten.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return original(embed, device)

    tr.preprocess_embed = preprocess_embed
    tr._clipproj_video = True



# Chaque rechargement du loader construit un nouveau ProjectedCLIP, donc un
# nouveau cache GPU (W, moyennes, ecarts-types, et le reseau residuel qui pese a
# lui seul 576 Mo en fp32). L'ancien garde le sien tant que ComfyUI n'a pas
# remplace la sortie du noeud, ce qui n'arrive qu'APRES le chargement du nouvel
# encodeur : au moment precis ou la carte est le plus sollicitee, elle porte deux
# jeux de projections. Un registre faible permet de les vider avant. Faible pour
# qu'il ne retienne rien lui-meme.
_PROJETES = []


def _enregistrer_projete(instance):
    """Suit une instance sans la maintenir en vie."""
    _PROJETES.append(weakref.ref(instance))


def purge_projections(carte):
    """Vide le cache GPU de toutes les projections posees sur `carte`.

    Args:
        carte: le peripherique a degager.

    Returns:
        float: Mo liberes, tels que comptes avant la purge.
    """
    total = 0.0
    vivants = []
    for ref in _PROJETES:
        obj = ref()
        if obj is None:
            continue
        vivants.append(ref)
        cache = obj.__dict__.get("_gpu")
        if not cache or cache.get("device") != carte:
            continue
        for cle in ("p", "mlp"):
            valeur = cache.get(cle)
            if valeur is None:
                continue
            tenseurs = (valeur.values() if isinstance(valeur, dict)
                        else (p for p in valeur.parameters()))
            total += sum(t.numel() * t.element_size() for t in tenseurs) / 2**20
        cache.clear()
    _PROJETES[:] = vivants
    if total:
        logging.info("[ClipProj] %.0f MB of projection caches cleared on %s", total, carte)
    return total


class ProjectedCLIP:
    """A small encoder disguised as a large one, by linear projection."""

    def __init__(self, base, projection_name):
        self.__dict__["_base"] = base
        self.__dict__["_proj_name"] = projection_name
        self.__dict__["_proj"] = load_projection(projection_name)
        self.__dict__["_key"] = getattr(base.cond_stage_model, "clip", "qwen3vl_4b")
        # Device-side copy of the projection, made once.
        self.__dict__["_gpu"] = {}
        _enregistrer_projete(self)

    def __getattr__(self, name):
        """Delegate anything not redefined here to the underlying CLIP."""
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_base"], name, value)

    def clone(self):
        """Clone the wrapper by cloning the underlying CLIP."""
        return ProjectedCLIP(self._base.clone(), self._proj_name)

    def tokenize(self, text, return_word_ids=False, images=[],
                 minimax_ref_items=None, **kwargs):
        """Tokenise the MiniMax H3 way: raw text with vision blocks spliced in.

        Raises:
            ValueError: if ref2va references (video / audio) are requested.
        """
        tok = _raw_tokenizer(self._base)
        entries = []

        def add_text(s):
            entries.extend((t, 1.0) for t in tok(s, add_special_tokens=False)["input_ids"])

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            embed = {"type": "image", "data": data, "original_type": "image"}
            if video_block:
                # Read back by preprocess_embed, which then routes the pair
                # through process_video_block instead of the image path.
                embed["minimax_video_block"] = True
            entries.append((embed, 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            # ref2va. Reference tokens are re-read at every sampling step, so a
            # projection error compounds instead of acting once — this path is
            # experimental. Ordinals are 1-based per type, matching
            # MiniMaxH3Tokenizer, so the prompt's <Picture i> tags line up.
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] = counters.get(kind, 0) + 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    # Audio never enters Qwen: only its label does.
                    add_text("<Audio %d>: " % counters["audio"])
                else:
                    # Video. MiniMax H3 does not treat a clip as a series of
                    # images: it pairs the frames two by two into the vision
                    # tower's temporal patch, and prefixes each pair with the
                    # timestamp of its midpoint. Frames are expected at 2 fps,
                    # and an odd count is padded by repeating the last one so
                    # the final pair is complete.
                    frames = item["data"]
                    stamps = item.get("timestamps")
                    if stamps is None:
                        stamps = [i / 2.0 for i in range(frames.shape[0])]
                    stamps = list(stamps)
                    if frames.shape[0] % 2 == 1:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        stamps.append(stamps[-1])
                    add_text("<Video %d>: " % counters["video"])
                    for k in range(0, frames.shape[0], 2):
                        add_text("<%.1f seconds>" % ((stamps[k] + stamps[k + 1]) / 2.0))
                        add_vision(frames[k:k + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text("<Picture %d>: " % (i + 1))
                add_vision(img)
        add_text(text)

        if len(entries) == 0:
            entries.append((PAD_TOKEN, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {self._key: [entries]}

    def _encode(self, tokens):
        """Read the chosen tap, then project.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (cond [B, seq, d_out], tags [seq]).
        """
        proj = self._proj
        base = self._base
        sm = _submodel(base)
        install_video_blocks(sm)

        mm.load_models_gpu([base.patcher])
        device = base.patcher.load_device

        pairs = tokens[self._key] if isinstance(tokens, dict) else tokens
        pairs = [[(t[0], t[1]) for t in seq] for seq in pairs]

        tap = int(proj["tap"])

        # Tags depend on where the vision blocks sit, which only process_tokens
        # knows. We intercept the forward call to capture it.
        captured = {}
        orig_forward = sm.transformer.forward

        def capturing_forward(*args, **kwargs):
            captured["embeds_info"] = kwargs.get("embeds_info", [])
            return orig_forward(*args, **kwargs)

        saved = (sm.layer, sm.layer_idx, sm.layer_norm_hidden_state, sm.execution_device)
        try:
            sm.transformer.forward = capturing_forward
            sm.layer = [tap] if tap >= 0 else "last"
            sm.layer_idx = None
            sm.layer_norm_hidden_state = False
            sm.execution_device = device
            with mm.cuda_device_context(device):
                with torch.no_grad():
                    out = sm.encode_token_weights(pairs)
        finally:
            sm.transformer.forward = orig_forward
            (sm.layer, sm.layer_idx, sm.layer_norm_hidden_state, sm.execution_device) = saved

        h = out[0]
        if h.dim() == 4:  # [B, n_taps, seq, d_in]: a single tap was requested
            h = h[:, 0]
        h = h.float()

        dev = h.device
        d_in = h.shape[-1]
        cache = self.__dict__["_gpu"]
        if cache.get("device") != dev:
            cache.clear()
            cache["device"] = dev
        if "p" not in cache:
            if "control" in proj:
                cache["p"] = build_control(proj["control"], d_in, guess_cond_dim(), dev)
            else:
                if d_in != proj["W"].shape[0]:
                    raise ValueError(
                        "Dimension mismatch: the encoder outputs %d, the "
                        "projection expects %d." % (d_in, proj["W"].shape[0]))
                cache["p"] = {k: proj[k].to(dev) for k in
                              ("W", "mean_in", "std_in", "mean_out", "std_out")}
                cache["mlp"] = build_residual(proj, dev)
        p = cache["p"]

        # Standardised space, where the ridge was fitted. The residual network,
        # when the file carries one, corrects inside that same space so that it
        # works at the scale of what it is correcting. Its last layer was
        # trained from a zero initialisation, so a freshly trained network that
        # learned nothing reproduces the matrix exactly.
        xn = (h - p["mean_in"]) / p["std_in"]
        yn = xn @ p["W"]
        reseau = cache.get("mlp")
        if reseau is not None:
            # Le reseau reste dans le type du fichier ; on convertit
            # autour de lui. Un residu fp16 tient alors en fp16 en VRAM,
            # pas seulement sur le disque.
            td = reseau[0].weight.dtype
            yn = yn + reseau(xn.to(td)).float()
        cond = yn * p["std_out"] + p["mean_out"]

        # Token 0 is an attention sink: its direction is constant across prompts
        # (cosine 1.0000 measured over 1966 of them) and carries no information
        # from the text, yet its norm reaches 16 500 against 291 for a text
        # token. Calibration excluded it — rightly, its extreme values would
        # wreck the statistics — so W has never seen one and projects it to an
        # arbitrary direction with a huge norm. That is invisible on a 200-token
        # prompt where it is 0.5 % of the positions, and ruinous on a 7-token one
        # where it is 14 %. Since the vector is constant, substituting its
        # measured value is exact rather than approximate.
        sink = proj.get("sink_out")
        if sink is not None and cond.shape[1] > 0:
            cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)

        cond = cond.to(mm.intermediate_device())
        tags = tags_from_embeds_info(cond.shape[1], captured.get("embeds_info", []))
        return cond, tags

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        """Mirror comfy.sd.CLIP.encode_from_tokens on the projected model."""
        cond, tags = self._encode(tokens)
        if return_dict:
            out = {"cond": cond, "pooled_output": None, "minimax_token_tags": tags}
            self._base.add_hooks_to_dict(out)
            return out
        if return_pooled:
            return cond, None
        return cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={},
                                     show_pbar=True):
        """Return conditioning in ComfyUI's format: [[tensor, dict]].

        Step scheduling is meaningless here: the projection is static, so a
        single conditioning is produced.
        """
        cond, tags = self._encode(tokens)
        extra = {"pooled_output": None, "minimax_token_tags": tags}
        extra.update(add_dict)
        self._base.add_hooks_to_dict(extra)
        return [[cond, extra]]

    def encode(self, text):
        """Encode a string directly."""
        return self.encode_from_tokens(self.tokenize(text))


def _wrap(clip, projection):
    """Wrap a CLIP and log which projection was selected.

    No check on the loaded model: any variant with a matching output dimension
    will do, quantised or fine-tuned included. If the dimensions disagree,
    encoding raises an explicit error.
    """
    wrapped = ProjectedCLIP(clip, projection)
    p = wrapped._proj
    if "control" in p:
        logging.info("[ClipProj] control %s: a reference point, not a learned "
                     "projection", projection)
    else:
        logging.info("[ClipProj] %s | tap %d | %d -> %d%s", projection,
                     int(p["tap"]), p["W"].shape[0], p["W"].shape[1],
                     " | cos_test %.4f" % float(p["cos_test"]) if "cos_test" in p else "")
    return wrapped


def _load_encoder(clip_name, clip_type, device, mode, unique_id):
    """Load an encoder onto a specific device.

    In resident mode offload_device equals load_device: an unload would free
    nothing while removing the model from ComfyUI's memory accounting, which
    then oversells the VRAM until it OOMs. Pinning stops it from trying. In
    streaming and dynamic modes the offload target is RAM, so ComfyUI can
    genuinely free the card.
    """
    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    dev = torch.device(device)
    embeddings = folder_paths.get_folder_paths("embeddings")

    # Identity of a loaded instance. The filename alone is not enough: moving the
    # same checkpoint to another card, or switching residency mode, produces a
    # different resident object that must replace the previous one. Everything
    # that changes what sits in VRAM belongs in this key.
    key = "%s@%s:%s" % (clip_name, device, mode)

    # Release whatever this node held BEFORE loading the replacement. Doing it
    # afterwards means both encoders sit in VRAM at once, and on a tight card the
    # load fails before the release ever runs. No-op when the key is unchanged.
    role = "ClipProj:%s" % unique_id
    # Liberer AVANT de charger, et liberer toute la carte, pas seulement ce que
    # ce noeud y avait mis. Deux raisons. La cle n'est plus consultee : quand ce
    # code s'execute, ComfyUI a deja decide de recharger, donc s'abstenir ne
    # sauvait aucun chargement et laissait seulement une copie orpheline. Et le
    # balayage porte sur la carte parce qu'un autre noeud a pu y epingler un
    # modele : sur une carte qui n'en tient qu'un, le pic des deux ensemble
    # n'est pas un accident de courbe, c'est un OOM.
    release_role(role)
    release_device(dev, garde=role)
    purge_projections(dev)
    if dev.type == "cuda":
        # Sans ceci les blocs restent reserves par l'allocateur torch et
        # ComfyUI lit une carte encore pleine au moment de decider.
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

    # La detection tourne meme quand le type est impose. Elle cherche la tour
    # visuelle, donc elle distingue un Qwen3-VL d'un Qwen3 ordinaire -- ce que
    # la largeur ne fait pas : les deux familles partagent 2560 et 4096, la
    # projection se charge sans rien signaler et sort quelque chose qui ignore
    # le prompt. Choisir le type a la main court-circuitait ce garde-fou.
    label = ""
    found = detect_arch(path)
    if clip_type == "auto":
        if found is None:
            raise ValueError(
                "Could not identify the architecture of %s. It may not be a "
                "Qwen3-VL checkpoint: no vision tower was found in it, and a "
                "text-only Qwen3 has the same hidden width, so nothing else "
                "would have caught it. Pick the type by hand to load it "
                "anyway: krea2 for a 4B, boogu for an 8B, minimax for the "
                "32B." % clip_name)
        clip_type, label = found
        label = " [%s detected]" % label
    elif found is not None and found[0] != clip_type:
        raise ValueError(
            "%s is a %s, but the type is set to %s. Set it to auto, or to %s."
            % (clip_name, found[1], clip_type, found[0]))
    elif found is None:
        logging.warning(
            "[ClipProj] no vision tower found in %s. If it is a text-only Qwen3 "
            "rather than a Qwen3-VL, the projection will load without "
            "complaint and produce conditioning that ignores your prompt: the "
            "two families share the same hidden width, so nothing checks it.",
            clip_name)
    ctype = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.KREA2)

    if mode == "resident":
        clip = comfy.sd.load_clip(
            ckpt_paths=[path], embedding_directory=embeddings, clip_type=ctype,
            model_options={"load_device": dev, "offload_device": dev},
            disable_dynamic=True)
        mm.load_models_gpu([clip.patcher], force_full_load=True)
        pin_patcher(clip.patcher, "encoder %s on %s" % (clip_name, device),
                    role=role, key=key)
    else:
        clip = comfy.sd.load_clip(
            ckpt_paths=[path], embedding_directory=embeddings, clip_type=ctype,
            model_options={"load_device": dev,
                           "offload_device": mm.text_encoder_offload_device()})
    logging.info("[ClipProj] %s (%s%s) loaded in %s mode on %s: %.2f GB", clip_name,
                 clip_type, label, mode, dev, clip.patcher.model_size() / 1024 ** 3)
    return clip


MODE_TOOLTIP = ("resident: loaded in one go and pinned, fastest.\n"
                "streaming / dynamic: pageable weights, ComfyUI can free the card.")


class ClipProjDeviceLoader:
    """Load a text encoder on the GPU of your choice, without projecting.

    The stock CLIPLoader only offers 'default' and 'cpu', so there is no way to
    target a specific card on a multi-GPU machine. This node fills that gap and
    is useful with or without a projection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (clip_types(), {
                    "default": "auto",
                    "tooltip": "auto reads the checkpoint header and picks the "
                               "matching architecture."}),
                "device": (gpu_devices(), {"tooltip": "GPU that receives the encoder"}),
                "mode": (["resident", "streaming", "dynamic"],
                         {"default": "resident", "tooltip": MODE_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "ClipProj"
    DESCRIPTION = "Load a text encoder on a specific GPU. No projection applied."

    def load(self, clip_name, type, device, mode="resident", unique_id=None):
        """Load the encoder and return it unchanged."""
        return (_load_encoder(clip_name, type, device, mode, unique_id),)


class ClipProjApply:
    """Project an already loaded CLIP, without reloading it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A small encoder, already loaded"}),
                "projection": (list_projections(), {
                    "tooltip": "Learned matrix, or a <control:...> reference"}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "apply"
    CATEGORY = "ClipProj"
    DESCRIPTION = ("Insert the projection between an encoder and the diffusion "
                   "model's clip input.")

    def apply(self, clip, projection):
        """Return the projected version of the given CLIP."""
        return (_wrap(clip, projection),)


class ClipProjLoader:
    """Load a small text encoder on a specific GPU and project it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Small encoder, for example a Qwen3-VL-4B"}),
                "type": (clip_types(), {
                    "default": "auto",
                    "tooltip": "auto reads the checkpoint header and picks the "
                               "matching architecture. Override only if that "
                               "fails: krea2 = 4B, boogu = 8B, minimax = 32B."}),
                "projection": (list_projections(), {
                    "tooltip": "Learned matrix, or a <control:...> reference"}),
                "device": (gpu_devices(), {"tooltip": "GPU that receives the encoder"}),
                "mode": (["resident", "streaming", "dynamic"],
                         {"default": "resident", "tooltip": MODE_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "ClipProj"
    DESCRIPTION = "Load a small text encoder and project it into the large one's space."

    def load(self, clip_name, type, projection, device, mode="resident", unique_id=None):
        """Load the encoder, then return its projected version."""
        clip = _load_encoder(clip_name, type, device, mode, unique_id)
        return (_wrap(clip, projection),)


class AnyType(str):
    """Wildcard type: accepts any link, used purely for execution ordering."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class ClipProjFree:
    """Free the encoders loaded by ClipProj.

    Resident mode pins the weights so ComfyUI does not move them: without that
    it would believe it had freed VRAM that is in fact still occupied, and end
    up OOMing. The trade-off is that they never leave on their own. This node
    forces the release when you need it -- typically before loading another
    large model in the same graph.

    The input and output exist only to place the node at the right point in the
    execution order: connect whatever must finish before the purge.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (["ClipProj encoders", "all models"], {
                    "default": "ClipProj encoders",
                    "tooltip": "'all models' also calls ComfyUI's global unload, "
                               "including the diffusion model and the VAEs."}),
            },
            "optional": {
                "after": (ANY, {"tooltip": "What must finish before the purge"}),
            },
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("after", "info")
    FUNCTION = "free"
    CATEGORY = "ClipProj"
    OUTPUT_NODE = True
    DESCRIPTION = "Free the VRAM held by pinned encoders."

    def free(self, scope, after=None):
        """Unload models according to the requested scope.

        Returns:
            tuple: (the input unchanged, a readable summary)
        """
        count, total = release_all()
        # Les caches de projection ne sont pas des modeles ComfyUI : personne
        # d'autre ne les libere, et un bouton Free qui laisse 600 Mo sur la carte
        # ment sur ce qu'il fait.
        mo = sum(purge_projections(torch.device("cuda", i))
                 for i in range(torch.cuda.device_count()))
        info = "%d encoder(s) freed, %.2f GB" % (count, total + mo / 1024.0)
        if scope == "all models":
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
            info += " + ComfyUI global unload"
        logging.info("[ClipProj] %s", info)
        return (after, info)


class ClipProjGenerate:
    """Generate text, images included, on the already resident encoder.

    ComfyUI's SDClipModel.generate does not pass the visual inputs down to the
    transformer: it drops embeds_info and never calls build_image_inputs. Image
    tokens then sit at linear positions instead of Qwen3-VL's 3D mRoPE, with no
    DeepStack injection, which makes any image description worthless. This node
    restores the full path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The same encoder used for conditioning"}),
                "system": ("STRING", {"multiline": True,
                                      "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "Describe this image."}),
                "max_length": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                                          "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0,
                                    "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to describe (optional)"}),
                "precision": (["weights", "float16", "bfloat16", "float32"], {
                    "default": "weights",
                    "tooltip": "Compute dtype. 'weights' follows the loaded model: "
                               "otherwise ComfyUI runs float16 weights in bfloat16 "
                               "and re-casts everything on every token."}),
                "preload_head": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prepare the output matrix once instead of "
                               "re-casting it on every token. Turn off if VRAM "
                               "is tight."}),
                # Appended last on purpose: ComfyUI restores widget values by
                # position, so inserting an input above an existing one shifts
                # every saved workflow and lands a string in a float field.
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Discourages tokens already produced, which is what "
                               "breaks an answer that locks into a repeating "
                               "phrase. Only applies while sampling: at "
                               "temperature 0 ComfyUI takes the most likely token "
                               "outright and ignores this. To cure a loop, raise "
                               "temperature to about 0.3 first, then this towards "
                               "1.2."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "ClipProj"
    DESCRIPTION = ("Text generation and image captioning on the already resident "
                   "weights. Noticeably slower than a dedicated engine.")

    def generate(self, clip, system, prompt, max_length, temperature, top_p, top_k,
                 seed, image=None, precision="weights", preload_head=True,
                 repetition_penalty=1.05):
        """Generate text from a prompt and, if given, an image.

        Returns:
            tuple[str]: the generated text, special tokens stripped.
        """
        base = clip._base if isinstance(clip, ProjectedCLIP) else clip
        sm = _submodel(base)
        tok = _raw_tokenizer(base)

        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        if precision == "weights":
            exec_dtype = getattr(sm.transformer, "dtype", None)
            if exec_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                exec_dtype = None  # exotic dtype: let ComfyUI decide
        else:
            exec_dtype = dtypes[precision]

        mm.load_models_gpu([base.patcher])
        device = base.patcher.load_device

        entries = []

        def add_text(s):
            entries.extend(tok(s, add_special_tokens=False)["input_ids"])

        # Chat template: without it the model continues the text instead of
        # answering. 151644 = <|im_start|>, 151645 = <|im_end|>.
        entries.append(IM_START)
        add_text("system\n%s" % system)
        entries.append(IM_END)
        add_text("\n")
        entries.append(IM_START)
        add_text("user\n")
        if image is not None:
            entries.append(VISION_START)
            entries.append({"type": "image", "data": image, "original_type": "image"})
            entries.append(VISION_END)
        add_text(prompt)
        entries.append(IM_END)
        add_text("\n")
        entries.append(IM_START)
        add_text("assistant\n")

        t0 = time.time()
        saved = sm.execution_device
        saved_logits = None
        try:
            sm.execution_device = device

            # logits() sends the output matrix through cast_bias_weight on every
            # token. With a 151936-entry vocabulary that is hundreds of MB
            # re-cast per token, which dominates everything else in
            # autoregressive decoding. Prepare it once instead.
            if preload_head:
                tr = sm.transformer
                head = getattr(tr.model, "lm_head", None) or tr.model.embed_tokens
                with mm.cuda_device_context(device):
                    # cast_bias_weight returns (weight, bias) in legacy mode and
                    # (weight, bias, offload_stream) with offloadable=True: take
                    # the first element either way.
                    res = comfy.ops.cast_bias_weight(
                        head, dtype=exec_dtype, device=device, offloadable=False)
                hw = res[0] if isinstance(res, (tuple, list)) else res
                saved_logits = tr.logits

                def fast_logits(x, _w=hw):
                    return torch.nn.functional.linear(x[:, -1:], _w, None)

                tr.logits = fast_logits

            with mm.cuda_device_context(device):
                with torch.no_grad():
                    embeds, _, _, embeds_info = sm.process_tokens([entries], device)
                    # The missing link: mRoPE positions, mask and DeepStack.
                    pos_ids, vis_masks, deepstack = sm.transformer.build_image_inputs(
                        embeds, embeds_info)
                    ids = sm.transformer.generate(
                        embeds=embeds, do_sample=temperature > 0.0,
                        max_length=max_length, temperature=temperature,
                        top_k=top_k, top_p=top_p, min_p=0.0,
                        repetition_penalty=repetition_penalty, seed=seed,
                        position_ids=pos_ids, visual_pos_masks=vis_masks,
                        deepstack_embeds=deepstack, embeds_info=embeds_info,
                        execution_dtype=exec_dtype)
        finally:
            sm.execution_device = saved
            if saved_logits is not None:
                sm.transformer.logits = saved_logits

        dt = time.time() - t0
        text = tok.decode(ids, skip_special_tokens=True).strip()
        logging.info("[ClipProj] %d tokens in %.1f s (%.1f tok/s), computed in %s%s",
                     len(ids), dt, len(ids) / max(dt, 1e-6), exec_dtype,
                     " (with image)" if image is not None else "")
        return (text,)
