#!/usr/bin/env python
"""Assemble les fragments produits par plusieurs processus d'encodage.

Chaque processus traite un prompt sur N et ecrit son propre fichier. Les
fragments sont donc entrelaces : le fragment 0 porte les indices 0, 2, 4... et
le fragment 1 les indices 1, 3, 5... L'assemblage remet tout dans l'ordre des
indices, ce qui n'est pas cosmetique -- h3_fit et h3_mlp decoupent leur jeu de
test par `k % 6`, donc un ordre different donnerait un autre decoupage et des
chiffres non comparables d'une calibration a l'autre.

Les fragments sont verifies avant fusion : meme modele, memes taps, aucun indice
en double. Un melange silencieux de deux encodeurs differents produirait une
matrice absurde sans lever la moindre erreur.
"""

import os
import sys
import time

import torch

ENTREES = [p for p in os.environ.get("H3_MERGE_IN", "").split(",") if p.strip()]
SORTIE = os.environ.get("H3_MERGE_OUT", "")
EFFACE = os.environ.get("H3_MERGE_DELETE", "") not in ("", "0")


def log(m):
    print(m, flush=True)


def main():
    if len(ENTREES) < 2 or not SORTIE:
        log("Renseigner H3_MERGE_IN (fichiers separes par des virgules) "
            "et H3_MERGE_OUT.")
        return 1
    for p in ENTREES:
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1

    t0 = time.time()
    blobs = []
    for p in ENTREES:
        log("lecture de %s..." % os.path.basename(p))
        b = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        log("  %d prompts, modele %s, taps %s"
            % (len(b["indices"]), b["model"], b["taps"]))
        blobs.append(b)

    ref = blobs[0]
    for b in blobs[1:]:
        if b["model"] != ref["model"]:
            log("Modeles differents : %s contre %s" % (b["model"], ref["model"]))
            return 1
        if b["taps"] != ref["taps"]:
            log("Taps differents : %s contre %s" % (b["taps"], ref["taps"]))
            return 1

    paires = []
    for b in blobs:
        for k, i in enumerate(b["indices"]):
            paires.append((i, b["ids"][k], b["data"][k]))
    paires.sort(key=lambda x: x[0])

    indices = [p[0] for p in paires]
    if len(set(indices)) != len(indices):
        log("Indices en double entre fragments : fusion refusee.")
        return 1

    blob = dict(ref)
    blob["indices"] = indices
    blob["ids"] = [p[1] for p in paires]
    blob["data"] = [p[2] for p in paires]
    blob["shard"] = 0
    blob["shards"] = 1

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    torch.save(blob, SORTIE)
    toks = sum(len(x) for x in blob["ids"])
    log("")
    log("ecrit %s" % SORTIE)
    log("  %d prompts, %d tokens, %.1f Go, %.0f min"
        % (len(indices), toks, os.path.getsize(SORTIE) / 1024 ** 3,
           (time.time() - t0) / 60))

    if EFFACE:
        for p in ENTREES:
            os.remove(p)
            log("  fragment supprime : %s" % os.path.basename(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
