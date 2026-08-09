#!/usr/bin/env python
"""Fabrique des matrices temoins a partir d'une projection apprise.

Les temoins servent de points de comparaison pour repondre a une question simple :
la matrice apprise apporte-t-elle vraiment quelque chose, ou le DiT se debrouille-t-il
seul ? Ils reprennent les statistiques de sortie (mean_out / std_out) de la projection
reelle, donc le conditionnement produit reste dans la bonne region de l'espace du 32B :
seule l'information venant du prompt change.

  ZERO      W = 0. Le conditionnement vaut mean_out en toute position, quel que soit
            le prompt. Controle negatif absolu : ce que la video montre alors provient
            entierement du DiT et du bruit initial.

  IDENTITY  Les 2560 dimensions du 4B, standardisees, recopiees telles quelles dans
            les 2560 premieres des 5120, puis remises a l'echelle du 32B. Les 2560
            dimensions restantes valent mean_out. Projection naive, sans le moindre
            apprentissage : l'ecart avec la projection reelle mesure exactement ce que
            la regression a appris.
"""

import os

# Defaults derived from this file's location: calibration/ lives inside the
# custom node, itself inside ComfyUI/custom_nodes/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMFY = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
import sys

import torch

OUT_DIR = os.environ.get("H3_PROJ_DIR", os.path.join(_COMFY, "models", "clip_projections"))
SOURCE = os.environ.get("H3_PROJ_SOURCE", "")


def log(msg):
    """Affiche un message immediatement."""
    print(msg, flush=True)


def pick_source():
    """Retourne le chemin de la projection servant de gabarit.

    Returns:
        str: chemin du .pt, en ecartant les temoins deja produits.
    """
    if SOURCE and os.path.isfile(SOURCE):
        return SOURCE
    cands = [os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))
             if f.lower().endswith(".pt") and "temoin" not in f.lower()]
    if not cands:
        raise FileNotFoundError("Aucune projection apprise dans %s" % OUT_DIR)
    return cands[0]


def save(data, path, label):
    """Ecrit un temoin et journalise ses caracteristiques."""
    torch.save(data, path)
    w = data["W"]
    log("  %-34s W %s  |W| %.3f  -> %s"
        % (label, tuple(w.shape), w.norm(), os.path.basename(path)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src_path = pick_source()
    log("Gabarit : %s" % os.path.basename(src_path))

    src = torch.load(src_path, map_location="cpu", weights_only=False)
    d_in, d_out = src["W"].shape
    mean_out, std_out = src["mean_out"].float(), src["std_out"].float()
    mean_in, std_in = src["mean_in"].float(), src["std_in"].float()
    tap = int(src["tap"])
    log("  d_in %d -> d_out %d, tap %d" % (d_in, d_out, tap))
    log("")

    common = {
        "mean_in": mean_in, "std_in": std_in,
        "mean_out": mean_out, "std_out": std_out,
        "tap": tap, "d_in": d_in, "d_out": d_out,
        "source_model": src.get("source_model", "temoin - tout encodeur"),
        "target_model": src.get("target_model", ""),
    }

    # ZERO : aucune information issue du prompt.
    save({**common, "W": torch.zeros(d_in, d_out),
          "lambda": 0.0, "cos_test": 0.0, "r2_test": 0.0,
          "note": "temoin ZERO : conditionnement constant = mean_out"},
         os.path.join(OUT_DIR, "h3_temoin_ZERO_aucune_info.pt"), "ZERO (aucune info)")

    # IDENTITY : recopie brute des dimensions du 4B, sans apprentissage.
    n = min(d_in, d_out)
    w_id = torch.zeros(d_in, d_out)
    w_id[:n, :n] = torch.eye(n)
    save({**common, "W": w_id,
          "lambda": 0.0, "cos_test": 0.0, "r2_test": 0.0,
          "note": "temoin IDENTITY : 2560 dims recopiees, reste = mean_out"},
         os.path.join(OUT_DIR, "h3_temoin_IDENTITY_brute.pt"), "IDENTITY (recopie brute)")

    log("")
    log("Contenu de %s :" % OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        if f.lower().endswith(".pt"):
            size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024 ** 2
            log("  %8.1f Mo  %s" % (size, f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
