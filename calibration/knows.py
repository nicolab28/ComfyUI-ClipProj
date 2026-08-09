#!/usr/bin/env python
"""Le petit modele connait-il la personne, ou est-ce la projection qui la perd ?

Question a trancher : sur un nom propre nu, la generation donne quelqu'un d'autre,
alors qu'avec un role ajoute (« l'actrice X dans tel film ») la personne revient.
Deux causes possibles, et elles n'appellent pas le meme remede :

  - le 4B ne sait pas qui c'est. Rien a reparer, aucune projection ne restitue
    une information jamais encodee.
  - le 4B le sait, mais W abime la direction concernee. Reparable : plus de
    donnees, ou un MLP a la place de la matrice.

Le test contourne entierement W : on demande a l'encodeur de decrire la personne
en texte clair, par son propre chemin de generation. S'il la decrit correctement,
la connaissance est la et le probleme est dans la projection.
"""

import importlib.util
import os
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
TE_DIR = os.environ.get("H3_TE_DIR", r"D:\ComfyUI-Launcher\_models\text_encoders")
TE = os.path.join(TE_DIR, os.environ.get(
    "H3_TE", r"Qwen3vl_4b\qwen3vl_4b_int8_convrot.safetensors"))
CLIP_TYPE = os.environ.get("H3_CLIP_TYPE", "krea2")
NODE_DIR = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-ClipProj")

# Personnes a tester. Une par ligne, telles qu'elles seraient ecrites dans un
# prompt. Surchargeable par H3_NAMES, separees par des points-virgules.
NAMES = [n.strip() for n in os.environ.get(
    "H3_NAMES", "Scarlett Johansson;Will Smith;Mickey Mouse").split(";") if n.strip()]

SYSTEM = ("You answer factual questions about well-known people and characters. "
          "If you do not know who someone is, say exactly \"I do not know who that is.\" "
          "Never invent an appearance.")

QUESTION = ("Describe the physical appearance of %s in two sentences: hair colour "
            "and length, eye colour, face shape, build. Then name one work they "
            "are famous for.")

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402


def load_node():
    """Importe le package du node par chemin (son dossier contient un tiret)."""
    spec = importlib.util.spec_from_file_location(
        "clipproj_pkg", os.path.join(NODE_DIR, "__init__.py"),
        submodule_search_locations=[NODE_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clipproj_pkg"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    if not os.path.isfile(TE):
        print("Encodeur introuvable : %s" % TE)
        return 1

    mod = load_node()
    generate = mod.NODE_CLASS_MAPPINGS["ClipProjGenerate"]()

    print("Encodeur : %s" % os.path.basename(TE))
    print("Chemin   : generation directe, la matrice de projection n'intervient pas.")
    print("", flush=True)

    clip = comfy.sd.load_clip(ckpt_paths=[TE], embedding_directory=None,
                              clip_type=CLIP_TYPE)
    mm.load_models_gpu([clip.patcher])
    print("Charge sur %s" % clip.patcher.load_device, flush=True)

    for name in NAMES:
        print("")
        print("=" * 78)
        print("### %s" % name)
        print("=" * 78, flush=True)
        (text,) = generate.generate(
            clip=clip, system=SYSTEM, prompt=QUESTION % name,
            max_length=160, temperature=0.0, top_p=1.0, top_k=0, seed=0)
        print(text, flush=True)

    print("")
    print("Lecture : une description juste signifie que la connaissance est dans")
    print("l'encodeur et que c'est la projection qui la perd -- donc reparable.")
    print("Un refus ou une description fausse signifie que le modele ne la connait")
    print("pas, et aucune matrice n'y changera quoi que ce soit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
