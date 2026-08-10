#!/usr/bin/env python
"""Combien de contextes faut-il par nom propre pour que la projection le tienne ?

La matrice CELEB a vu quarante phrases par personne et reconstruit ses noms a
0,999. C'est certainement bien au-dela du necessaire, et la question n'est pas
academique : a budget constant, diviser par huit le nombre de contextes multiplie
par huit le nombre de gens couverts.

Le protocole tient sur les encodages deja produits, sans rien recalculer. Pour
chaque valeur de N, on garde les N premiers contextes de chaque nom pour ajuster,
on tient le reste a l'ecart, et on mesure la reconstruction sur les tokens du nom
dans ces contextes non vus. Les noms sont donc les memes des deux cotes : on
mesure la couverture d'une personne, pas la generalisation a une autre.
"""

import os
import re
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
CIBLE = os.environ.get("H3_CTX_TARGET", r"D:\tmp\h3_data\encode\gens_train_target32b.pt")
ENTREE = os.environ.get("H3_CTX_INPUT", r"D:\tmp\h3_data\encode\gens_train_input4b.pt")
PROMPTS = os.environ.get("H3_CTX_PROMPTS", r"D:\tmp\h3_data\h3_gens_train.txt")
VALEURS = [int(x) for x in os.environ.get("H3_CTX_N", "2,5,10,20,40").split(",")]
LAMBDAS = [float(x) for x in os.environ.get(
    "H3_CTX_LAMBDAS", "1e2,1e3,1e4,1e5").split(",")]
MIN_WORDS = int(os.environ.get("H3_MIN_WORDS", "5"))
DEVICE = os.environ.get("H3_ACC_DEVICE", "cuda")
DROP_FIRST = 1

sys.path.insert(0, COMFY_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import comfy.text_encoders.minimax as minimax  # noqa: E402
from gens import NOMS_TRAIN, NOMS_TEST  # noqa: E402
from fit import Accumulateur  # noqa: E402

NOMS = sorted(NOMS_TRAIN + NOMS_TEST, key=len, reverse=True)


def log(m):
    print(m, flush=True)


def charge_prompts():
    vus, gardes = set(), []
    with open(PROMPTS, "r", encoding="utf-8") as f:
        for ligne in f:
            p = ligne.strip()
            if len(p.split()) < MIN_WORDS:
                continue
            k = p[:120]
            if k in vus:
                continue
            vus.add(k)
            gardes.append(p)
    return gardes


def trouve(suite, motif):
    n, m = len(suite), len(motif)
    for i in range(n - m + 1):
        if suite[i:i + m] == motif:
            return i
    return -1


def main():
    for p in (CIBLE, ENTREE, PROMPTS):
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1

    tok = minimax.MiniMaxH3Tokenizer()
    ids_nom = {n: tok._text_ids(" " + n) for n in NOMS}
    ids_debut = {n: tok._text_ids(n) for n in NOMS}

    corpus = charge_prompts()
    tg = torch.load(CIBLE, map_location="cpu", weights_only=False, mmap=True)
    en = torch.load(ENTREE, map_location="cpu", weights_only=False, mmap=True)
    ic = {i: k for k, i in enumerate(tg["indices"])}
    ie = {i: k for k, i in enumerate(en["indices"])}
    communs = [i for i in en["indices"] if i in ic]
    rang = 0 if len(en["taps"]) == 1 else en["taps"].index(24)

    # Regroupement par personne, avec la position du nom dans la sequence.
    par_nom, spans = {}, {}
    for i in communs:
        texte = corpus[i]
        ids = tg["ids"][ic[i]]
        for n in NOMS:
            if n not in texte:
                continue
            pos = -1
            for motif in (ids_nom[n], ids_debut[n]):
                pos = trouve(ids, motif)
                if pos >= 0:
                    spans[i] = (pos, pos + len(motif))
                    break
            if pos >= 0:
                par_nom.setdefault(n, []).append(i)
            break

    tailles = [len(v) for v in par_nom.values()]
    log("%d noms, %.1f contextes chacun en moyenne (min %d, max %d)"
        % (len(par_nom), sum(tailles) / len(tailles), min(tailles), max(tailles)))
    log("")

    d_in = en["data"][0].shape[-1]
    d_out = tg["data"][0].shape[-1]
    dev = torch.device(DEVICE)

    log("  %5s %8s %9s %9s %9s" % ("N", "prompts", "tokens", "cos nom", "cos reste"))
    log("  %s" % ("-" * 46))

    for N in VALEURS:
        train, test = [], []
        for n, liste in par_nom.items():
            train.extend(liste[:N])
            test.extend(liste[N:])
        if not test:
            log("  %5d : plus rien a tenir a l'ecart" % N)
            continue

        acc = Accumulateur(d_in, d_out, DEVICE)
        for i in train:
            x = en["data"][ie[i]][rang].float()
            y = tg["data"][ic[i]].float()
            m = min(x.shape[0], y.shape[0])
            if m > DROP_FIRST:
                acc.ajoute(x[DROP_FIRST:m], y[DROP_FIRST:m])

        meilleur = None
        for lam in LAMBDAS:
            w, mx, sdx, my, sdy = acc.resous(lam)
            W = w.float().to(dev)
            mi, si = mx.float().to(dev), sdx.float().to(dev)
            mo, so = my.float().to(dev), sdy.float().to(dev)
            s_nom = n_nom = 0.0
            s_aut = n_aut = 0.0
            with torch.no_grad():
                for i in test:
                    x = en["data"][ie[i]][rang].to(dev, torch.float32)
                    y = tg["data"][ic[i]].to(dev, torch.float32)
                    m = min(x.shape[0], y.shape[0])
                    x, y = x[:m], y[:m]
                    p = ((x - mi) / si) @ W * so + mo
                    c = torch.nn.functional.cosine_similarity(p, y, dim=1)
                    a, b = spans[i]
                    masque = torch.zeros(m, dtype=torch.bool, device=dev)
                    masque[max(a, DROP_FIRST):min(b, m)] = True
                    autre = ~masque
                    autre[:DROP_FIRST] = False
                    s_nom += float(c[masque].sum())
                    n_nom += int(masque.sum())
                    s_aut += float(c[autre].sum())
                    n_aut += int(autre.sum())
            cn = s_nom / max(1, n_nom)
            ca = s_aut / max(1, n_aut)
            if meilleur is None or cn > meilleur[1]:
                meilleur = (lam, cn, ca)
            del W, mi, si, mo, so

        log("  %5d %8d %9d %9.4f %9.4f"
            % (N, len(train), acc.n, meilleur[1], meilleur[2]))

    log("")
    log("  Lecture : le N a partir duquel « cos nom » cesse de monter est le")
    log("  nombre de contextes utile. Au-dela, le budget est mieux depense a")
    log("  couvrir des personnes supplementaires.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
