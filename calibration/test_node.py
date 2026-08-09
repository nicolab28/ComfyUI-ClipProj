#!/usr/bin/env python
"""Test de bout en bout du custom node ben-h3proj, sans lancer ComfyUI.

Charge reellement le Qwen3-VL-4B, l'enveloppe dans ProjectedH3CLIP, puis appelle
exactement la sequence qu'utilise MiniMaxH3ImageToVideo :

    tokens = clip.tokenize(prompt, images=[])
    cond   = clip.encode_from_tokens_scheduled(tokens)

et verifie que le conditionnement produit a la forme attendue par le DiT H3.
"""

import importlib.util
import os

# Defaults derived from this file's location: calibration/ lives inside the
# custom node, itself inside ComfyUI/custom_nodes/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMFY = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", _COMFY)
TE_4B = os.environ.get(
    "H3_TE_4B", os.path.join(_COMFY, "models", "text_encoders", "qwen3vl_4b_bf16.safetensors"))
NODE_DIR = os.path.join(COMFY_DIR, "custom_nodes", "ben-h3proj")

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402

PROMPT = ("In summer, a woman smiling, wears a mini red skirt and a white top, "
          "is walking on a street, then she sits down on a public bench")


def load_module():
    """Importe le node par chemin (le dossier contient un tiret)."""
    spec = importlib.util.spec_from_file_location(
        "ben_h3proj", os.path.join(NODE_DIR, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("Import du node...", flush=True)
    mod = load_module()
    print("  nodes exposes : %s" % list(mod.NODE_CLASS_MAPPINGS.keys()))

    found = [p for p in mod.list_projections() if not p.startswith("<")]
    if not found:
        print("  AUCUNE projection trouvee dans %s" % mod.PROJECTION_DIRS)
        return 1
    proj_path = found[0]
    print("  projection    : %s" % proj_path)

    proj = mod.load_projection(proj_path)
    print("  tap %d | W %s | cos_test %s"
          % (int(proj["tap"]), tuple(proj["W"].shape), proj.get("cos_test", "?")))

    print("")
    print("Chargement du 4B : %s" % os.path.basename(TE_4B), flush=True)
    import comfy.sd  # noqa: E402
    import comfy.model_management as mm  # noqa: E402

    if not os.path.isfile(TE_4B):
        print("  ENCODEUR INTROUVABLE : %s" % TE_4B)
        return 1

    dev = mm.get_torch_device()
    base = comfy.sd.load_clip(
        ckpt_paths=[TE_4B], embedding_directory=None,
        clip_type=comfy.sd.CLIPType.KREA2,
        model_options={"load_device": dev, "offload_device": dev},
        disable_dynamic=True)
    mm.load_models_gpu([base.patcher], force_full_load=True)
    print("  charge sur %s : %.2f Go" % (dev, base.patcher.model_size() / 1024 ** 3))

    print("")
    print("Enveloppe et encodage (sequence de MiniMaxH3ImageToVideo)...", flush=True)
    clip = mod.ProjectedH3CLIP(base, proj_path)

    tokens = clip.tokenize(PROMPT, images=[])
    key = list(tokens.keys())[0]
    n_tok = len(tokens[key][0])
    print("  tokenize   -> cle '%s', %d tokens" % (key, n_tok))

    cond = clip.encode_from_tokens_scheduled(tokens)
    print("  encode     -> %d conditionnement(s)" % len(cond))

    ok = True
    t, extra = cond[0][0], cond[0][1]
    print("  tenseur    : %s  dtype %s  device %s" % (tuple(t.shape), t.dtype, t.device))
    print("  extras     : %s" % sorted(extra.keys()))

    if t.dim() != 3:
        print("  PROBLEME : le conditionnement doit etre [B, seq, dim]")
        ok = False
    if t.shape[-1] != proj["W"].shape[1]:
        print("  PROBLEME : dim %d au lieu de %d" % (t.shape[-1], proj["W"].shape[1]))
        ok = False
    if t.shape[1] != n_tok:
        print("  PROBLEME : %d positions pour %d tokens" % (t.shape[1], n_tok))
        ok = False
    if "minimax_token_tags" not in extra:
        print("  PROBLEME : minimax_token_tags absent")
        ok = False
    else:
        tags = extra["minimax_token_tags"]
        print("  tags       : %s  somme %d (attendu %d)"
              % (tuple(tags.shape), int(tags.sum()), n_tok))
        if tags.shape[0] != t.shape[1]:
            print("  PROBLEME : tags de longueur %d pour %d positions"
                  % (tags.shape[0], t.shape[1]))
            ok = False
    if not torch.isfinite(t).all():
        print("  PROBLEME : valeurs non finies dans le conditionnement")
        ok = False

    print("  stats      : moy %+.4f  ecart-type %.4f  min %+.2f  max %+.2f"
          % (t.mean(), t.std(), t.min(), t.max()))
    print("  cible 32B  : moy %+.4f  ecart-type %.4f (moyennes de mean_out / std_out)"
          % (proj["mean_out"].mean(), proj["std_out"].mean()))

    print("")
    print("Refus attendu des images de reference...", flush=True)
    try:
        clip.tokenize(PROMPT, images=[torch.zeros(1, 64, 64, 3)])
        print("  PROBLEME : aucune erreur levee alors que des images sont fournies")
        ok = False
    except ValueError as e:
        print("  OK -- %s" % str(e).split(".")[0])

    print("")
    if ok:
        print("  TOUT EST BON. Redemarrer ComfyUI puis, dans le workflow :")
        print("    remplacer le node 156 (MiniMax H3 CLIP Loader) par")
        print("    'MiniMax H3 Proj CLIP Loader (4B)' -- meme sortie CLIP,")
        print("    le node 133 et le reste du graphe restent inchanges.")
        return 0
    print("  DES PROBLEMES SUBSISTENT (voir ci-dessus)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
