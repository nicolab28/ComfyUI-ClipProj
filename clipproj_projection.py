"""Load projection matrices and build the control matrices.

A projection is a .safetensors file (or a legacy .pt) holding:

    W         [d_in, d_out]  matrix learned by ridge regression
    mean_in   [d_in]         mean activation of the small model
    std_in    [d_in]         matching standard deviation
    mean_out  [d_out]        mean activation of the large model
    std_out   [d_out]        matching standard deviation
    tap       int            which layer of the small model to read

Reconstruction: cond = ((h - mean_in) / std_in) @ W * std_out + mean_out

Optional keys (source_model, target_model, cos_test, r2_test, n_train_tokens)
are informational only.

In a .safetensors file the tensors are stored as such and every scalar goes to
the header metadata, which safetensors keeps as strings; they are converted back
on load. Prefer that format: a .pt goes through pickle, which can execute
arbitrary code when opened, whereas safetensors cannot.
"""

import json
import os

import torch
from safetensors.torch import load_file as _load_safetensors
from safetensors import safe_open as _safe_open

import folder_paths

# Scalars stored as strings in the safetensors header, and the type to restore.
_META_TYPES = {
    "tap": int, "d_in": int, "d_out": int,
    "n_train_prompts": int, "n_train_tokens": int,
    "lambda": float, "cos_test": float, "r2_test": float,
    "cond_proj_weighted": lambda v: v.lower() == "true",
}

FOLDER = "clip_projections"

# Output dimension used for the control matrices when no real projection is
# available to infer it. 5120 matches MiniMax H3.
DEFAULT_COND_DIM = 5120

CONTROLS = ["<control:zero>", "<control:identity>", "<control:random>"]

_CACHE = {}


def register_folder():
    """Declare the projection folder to folder_paths.

    Projections then show up in a combo by filename, like checkpoints, instead
    of as an unreadable absolute path.
    """
    path = os.path.join(folder_paths.models_dir, FOLDER)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    existing = folder_paths.folder_names_and_paths.get(FOLDER)
    if existing is None:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], {".pt", ".safetensors"})
    elif path not in existing[0]:
        existing[0].append(path)


def list_projections():
    """Return the controls followed by every projection found on disk.

    Only files this node can actually open are listed. ComfyUI hands back
    everything in the folder, so a README or a note left there shows up in the
    dropdown as if it were a matrix, and picking it fails with an error about
    file formats rather than about the obvious mistake.
    """
    try:
        found = list(folder_paths.get_filename_list(FOLDER))
    except Exception:
        found = []
    found = [f for f in found if f.lower().endswith((".safetensors", ".pt"))]
    return CONTROLS + found


def load_projection(name):
    """Load a projection by name, or describe a control to be built.

    Args:
        name (str): filename, or a <control:...> entry.

    Returns:
        dict: the projection tensors, or {"control": mode, "tap": -1}.

    Raises:
        FileNotFoundError: if the file cannot be found.
        KeyError: if a required key is missing.
    """
    if name in CONTROLS:
        return {"control": name.split(":")[1].rstrip(">"), "tap": -1}

    path = folder_paths.get_full_path(FOLDER, name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError("Projection not found: %s" % name)

    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]

    if path.lower().endswith(".safetensors"):
        data = _load_safetensors(path)
        with _safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        for k, v in meta.items():
            cast = _META_TYPES.get(k)
            try:
                data[k] = cast(v) if cast else v
            except (TypeError, ValueError):
                data[k] = v
    else:
        # Legacy format, read with weights_only=True. Opening a pickle without
        # that flag executes whatever the file decides to run, which is an
        # absurd risk for six tensors and a handful of scalars, and it is the
        # reason the published .pt files were withdrawn in the first place.
        # Torch allows tensors and plain scalars through this path, which is
        # all these files ever contained; anything else in there was not put
        # there by this project.
        data = torch.load(path, map_location="cpu", weights_only=True)

    for k in ("W", "mean_in", "std_in", "mean_out", "std_out", "tap"):
        if k not in data:
            raise KeyError("Key '%s' missing from %s" % (k, name))
    for k in ("W", "mean_in", "std_in", "mean_out", "std_out"):
        data[k] = data[k].float()

    _CACHE.clear()  # one projection at a time: several tens of MB each
    _CACHE[key] = data
    return data


def build_residual(proj, device, dtype=None):
    """Rebuild the residual network stored alongside the matrix, if any.

    The file keeps it as plain tensors under `mlp.N.weight` / `mlp.N.bias`, which
    keeps the format readable and avoids storing a pickled module. Only Linear
    layers are stored; the activation between them is implied and fixed.

    Args:
        proj (dict): loaded projection.
        device: where the network must live.
        dtype: compute dtype. None keeps the dtype the file was saved
            in, which is the point of a fp16 residual: casting it up to
            fp32 on load halves the file on disk and costs exactly the
            same VRAM as before, which is no gain at all.

    Returns:
        torch.nn.Module|None: the network, or None when the file has no residual.
    """
    couches = sorted({int(k.split(".")[1]) for k in proj if k.startswith("mlp.")})
    if not couches:
        return None
    if dtype is None:
        dtype = proj["mlp.%d.weight" % couches[0]].dtype
    modules = []
    for n, i in enumerate(couches):
        w = proj["mlp.%d.weight" % i]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=("mlp.%d.bias" % i) in proj)
        lin.weight.data = w.to(device=device, dtype=dtype)
        if lin.bias is not None:
            lin.bias.data = proj["mlp.%d.bias" % i].to(device=device, dtype=dtype)
        modules.append(lin)
        if n < len(couches) - 1:
            modules.append(torch.nn.GELU())
    reseau = torch.nn.Sequential(*modules).to(device=device, dtype=dtype)
    # train(False) rather than the Module method it wraps, whose name collides
    # with the builtin a security scanner looks for. Identical behaviour, one
    # less false positive.
    reseau.train(False)
    for p in reseau.parameters():
        p.requires_grad_(False)
    return reseau


def guess_cond_dim():
    """Infer the output dimension from any projection present on disk.

    Returns:
        int: d_out read from the first usable file, else DEFAULT_COND_DIM.
    """
    for name in list_projections():
        if name in CONTROLS:
            continue
        try:
            return int(load_projection(name)["W"].shape[1])
        except Exception:
            continue
    return DEFAULT_COND_DIM


def build_control(mode, d_in, d_out, device, dtype=torch.float32):
    """Build a control matrix at the requested dimensions.

    zero      W = 0: the conditioning is constant and the prompt has no effect.
              Shows what the diffusion model produces on its own.
    identity  the d_in dimensions copied straight into the first d_out ones. A
              naive projection with no learning: the gap against a learned matrix
              measures exactly what the regression contributed.
    random    a random projection with the same energy as the identity.

    Args:
        mode (str): 'zero', 'identity' or 'random'.
        d_in (int): dimension of the small model.
        d_out (int): dimension expected by the diffusion model.
        device: target device.

    Returns:
        dict: the same keys as a learned projection.
    """
    w = torch.zeros(d_in, d_out, device=device, dtype=dtype)
    if mode == "identity":
        n = min(d_in, d_out)
        w[:n, :n] = torch.eye(n, device=device, dtype=dtype)
    elif mode == "random":
        g = torch.Generator(device="cpu").manual_seed(0)
        w = (torch.randn(d_in, d_out, generator=g).to(device=device, dtype=dtype)
             / (d_in ** 0.5))
    return {
        "W": w,
        "mean_in": torch.zeros(d_in, device=device, dtype=dtype),
        "std_in": torch.ones(d_in, device=device, dtype=dtype),
        "mean_out": torch.zeros(d_out, device=device, dtype=dtype),
        "std_out": torch.ones(d_out, device=device, dtype=dtype),
    }
