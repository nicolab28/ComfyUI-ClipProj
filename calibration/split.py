#!/usr/bin/env python
"""Scinde un encodage en deux, selon que le prompt nomme une personne tenue a l'ecart.

Les prompts au format H3 ont ete produits pour les 500 personnes d'un coup, sans
distinguer celles qui devaient rester inconnues de la calibration. Les encoder a
nouveau serait absurde ; il suffit de decouper le blob, puisque chaque entree
porte l'indice de son prompt dans le corpus d'origine.

Le decoupage se fait sur le texte, pas sur les identifiants de tokens : un nom
peut se retrouver decoupe differemment selon ce qui le precede, alors que sa
presence dans la phrase, elle, ne se discute pas.
"""

import json
import os
import sys

import torch

ENTREE = os.environ.get("H3_SPLIT_IN", "")
CORPUS = os.environ.get("H3_SPLIT_CORPUS", r"D:\tmp\h3_data\h3_cel_long.jsonl")
NOMS = os.environ.get("H3_SPLIT_NOMS", r"D:\tmp\h3_data\h3_cel_noms.json")
MIN_WORDS = int(os.environ.get("H3_MIN_WORDS", "3"))
SUFFIXE_TRAIN = os.environ.get("H3_SPLIT_TRAIN", "_train")
SUFFIXE_TEST = os.environ.get("H3_SPLIT_TEST", "_testnom")


def log(m):
    print(m, flush=True)


def charge_corpus():
    """Reproduit exactement la selection faite a l'encodage."""
    vus, gardes = set(), []
    jsonl = CORPUS.lower().endswith(".jsonl")
    with open(CORPUS, "r", encoding="utf-8") as f:
        for ligne in f:
            if jsonl:
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    p = json.loads(ligne)["prompt"].strip()
                except Exception:
                    continue
            else:
                p = ligne.strip()
            if len(p.split()) < MIN_WORDS:
                continue
            k = p[:120]
            if k in vus:
                continue
            vus.add(k)
            gardes.append(p)
    return gardes


def main():
    if not ENTREE or not os.path.isfile(ENTREE):
        log("Renseigner H3_SPLIT_IN.")
        return 1
    with open(NOMS, "r", encoding="utf-8") as f:
        d = json.load(f)
    tenus = sorted(d["test"], key=len, reverse=True)
    log("%d noms tenus a l'ecart" % len(tenus))

    corpus = charge_corpus()
    log("%d prompts dans le corpus" % len(corpus))

    blob = torch.load(ENTREE, map_location="cpu", weights_only=False, mmap=True)
    log("%d prompts encodes, modele %s" % (len(blob["indices"]), blob["model"]))

    tr, te = [], []
    for k, i in enumerate(blob["indices"]):
        texte = corpus[i] if i < len(corpus) else ""
        (te if any(n in texte for n in tenus) else tr).append(k)
    log("  %d a l'entrainement, %d de controle" % (len(tr), len(te)))

    base, ext = os.path.splitext(ENTREE)
    for suffixe, garde in ((SUFFIXE_TRAIN, tr), (SUFFIXE_TEST, te)):
        if not garde:
            continue
        b = dict(blob)
        b["indices"] = [blob["indices"][k] for k in garde]
        b["ids"] = [blob["ids"][k] for k in garde]
        b["data"] = [blob["data"][k] for k in garde]
        chemin = base + suffixe + ext
        torch.save(b, chemin)
        log("  ecrit %s  (%d prompts, %.2f Go)"
            % (os.path.basename(chemin), len(garde),
               os.path.getsize(chemin) / 1024 ** 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
