#!/usr/bin/env python
"""Ajoute le vecteur du token sink aux projections existantes.

Le premier token d'une sequence est un "attention sink" : sa direction est
constante d'un prompt a l'autre (cosinus 1.0000 mesure sur 1966 prompts) et il ne
porte aucune information du texte. Sa norme atteint pourtant 16 500 contre 291
pour un token de texte, soit 57 fois plus.

La calibration l'excluait, a juste titre : ses valeurs extremes ecraseraient les
statistiques. Mais le node le projetait quand meme, avec une matrice qui ne
l'avait jamais vu -- produisant un vecteur arbitraire de norme enorme. Invisible
sur un prompt de 200 tokens, ou il pese 0,5 % des positions ; devastateur sur un
prompt de 7 tokens, ou il en represente 14 %.

Comme le vecteur est constant, le remplacer par sa valeur mesuree n'est pas une
approximation : c'est la valeur exacte que le 32B aurait produite.
"""

import glob
import os
import sys

import torch

CACHE = os.environ.get("H3_TARGETS", r"D:\tmp\h3_data\h3_probe_out\targets.pt")
PROJ_DIRS = [
    r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI\models\clip_projections",
    r"D:\tmp\h3_data\h3_probe_out",
]


def measure_sink(path):
    """Mesure le vecteur sink moyen dans les cibles encodees par le gros modele.

    Returns:
        tuple[torch.Tensor, float]: (vecteur moyen, cosinus minimal observe).
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sink = torch.stack([t[0].float() for t in blob["targets"]])
    mean = sink.mean(0)
    cos = torch.nn.functional.cosine_similarity(sink, mean.unsqueeze(0), dim=1)
    return mean, float(cos.min())


def main():
    if not os.path.isfile(CACHE):
        print("Cible introuvable : %s" % CACHE)
        return 1

    sink, cos_min = measure_sink(CACHE)
    print("sink mesure : %d dims, norme %.1f, cosinus minimal %.4f"
          % (sink.numel(), sink.norm(), cos_min))
    if cos_min < 0.99:
        print("ATTENTION : le sink n'est pas constant (cosinus min %.4f). "
              "Le remplacer par sa moyenne serait une approximation grossiere."
              % cos_min)
        return 1

    seen = set()
    for d in PROJ_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
            name = os.path.basename(p)
            if name in ("targets.pt", "sink_mean.pt", "probe_tensors.pt"):
                continue
            data = torch.load(p, map_location="cpu", weights_only=False)
            if "W" not in data:
                continue
            if data["W"].shape[1] != sink.numel():
                print("  %-44s ignore (sortie %d != %d)"
                      % (name, data["W"].shape[1], sink.numel()))
                continue
            data["sink_out"] = sink.clone()
            torch.save(data, p)
            tag = "" if name in seen else "  <-"
            seen.add(name)
            print("  %-44s sink ajoute%s" % (name, tag))

    print("\n%d fichiers mis a jour." % len(seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
