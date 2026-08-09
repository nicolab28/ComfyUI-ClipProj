#!/usr/bin/env python
"""Convertit les projections .pt en .safetensors.

Un .pt passe par pickle : l'ouvrir peut executer du code arbitraire. Pour un
fichier telecharge, c'est un risque reel et gratuit -- ces projections ne
contiennent que des tenseurs et quelques scalaires, safetensors suffit.

Les tenseurs sont ecrits tels quels. Les scalaires (tap, lambda, cosinus, noms
de modeles) vont dans l'en-tete, que safetensors ne stocke qu'en chaines de
caracteres ; le node les reconvertit au chargement.

Le .pt d'origine n'est pas touche : la conversion est verifiee tenseur par
tenseur avant de rendre la main.
"""

import glob
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = os.environ.get(
    "H3_PROJ_DIR",
    r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI\models\clip_projections")

TENSEURS = ("W", "mean_in", "std_in", "mean_out", "std_out", "sink_out")


def convertir(path):
    """Ecrit le .safetensors correspondant et verifie qu'il relit identique.

    Returns:
        tuple[str, str]: (nom du fichier, message d'etat).
    """
    nom = os.path.basename(path)
    data = torch.load(path, map_location="cpu", weights_only=False)

    tensors, meta = {}, {}
    for k, v in data.items():
        if torch.is_tensor(v):
            tensors[k] = v.contiguous().float()
        elif isinstance(v, bool):
            meta[k] = "true" if v else "false"
        else:
            meta[k] = str(v)

    manquants = [k for k in ("W", "mean_in", "std_in", "mean_out", "std_out")
                 if k not in tensors]
    if manquants:
        return nom, "IGNORE (tenseurs absents : %s)" % ", ".join(manquants)
    if "tap" not in meta and "tap" not in tensors:
        return nom, "IGNORE (tap absent)"

    dest = os.path.splitext(path)[0] + ".safetensors"
    save_file(tensors, dest, metadata=meta)

    relu = load_file(dest)
    for k, v in tensors.items():
        if k not in relu or not torch.equal(relu[k], v):
            os.remove(dest)
            return nom, "ECHEC de verification sur %s, fichier retire" % k

    gain = os.path.getsize(path) - os.path.getsize(dest)
    return nom, "ok  %d tenseurs, %d scalaires, %+d octets" % (
        len(tensors), len(meta), -gain)


def main():
    if not os.path.isdir(SRC):
        print("Dossier introuvable : %s" % SRC)
        return 1

    fichiers = sorted(glob.glob(os.path.join(SRC, "*.pt")))
    if not fichiers:
        print("Aucun .pt dans %s" % SRC)
        return 0

    print("Source : %s" % SRC)
    print("")
    for p in fichiers:
        nom, etat = convertir(p)
        print("  %-40s %s" % (nom, etat))
    print("")
    print("Les .pt d'origine sont conserves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
