#!/usr/bin/env python
"""Calibre la projection a partir des encodages deja sur disque.

Separe de h3_run.py, qui encodait et calibrait d'un bloc : maintenant que les
encodages sont conserves, on peut refaire une calibration en quelques minutes
sans remobiliser les modeles, et le meme cache sert a l'entrainement du MLP.

Les matrices normales sont accumulees en streaming, prompt par prompt, donc la
memoire ne depend pas de la taille du corpus.

Le token sink est exclu de l'ajustement -- sa norme de 16 500 contre 291 pour un
token ordinaire ecraserait les statistiques -- puis sa valeur mesuree est ecrite
dans le fichier, le node la substituant a l'inference.
"""

import json
import os
import sys
import time

import torch
from safetensors.torch import save_file

CIBLE = os.environ.get("H3_FIT_TARGET", r"D:\tmp\h3_data\encode\target32b_full.pt")
ENTREE = os.environ.get("H3_FIT_INPUT", r"D:\tmp\h3_data\encode\input4b_full.pt")
SORTIE = os.environ.get("H3_FIT_OUT", r"D:\tmp\h3_data\encode\projection.safetensors")
TAP = int(os.environ.get("H3_FIT_TAP", "24"))

# Corpus supplementaires, sous la forme cible.pt|entree.pt, separes par des
# virgules. La barre verticale plutot que le deux-points : un chemin Windows
# commence par une lettre de lecteur suivie de ce meme caractere. Sert a combler un angle mort du corpus principal sans le refaire.
#
# Aucune ponderation n'est prevue et ce n'est pas un oubli : les directions de
# l'espace qui portent un nom propre ne sont excitees que par les tokens de ce
# nom. Un corpus qui ne pese que quelques pour cent du melange y apporte
# neanmoins la totalite de l'information disponible, la ou le corpus principal
# n'en apportait aucune.
EXTRA = [p for p in os.environ.get("H3_FIT_EXTRA", "").split(",") if p.strip()]
# Repetition des corpus supplementaires dans l'accumulation. La ridge n'a pas
# d'epoques : le seul moyen de donner plus de poids a un corpus est de compter
# ses tokens plusieurs fois dans le systeme normal. Repeter revient exactement a
# multiplier sa contribution a X'X et X'Y.
EXTRA_REPEAT = int(os.environ.get("H3_FIT_EXTRA_REPEAT", "1"))
TEST_UN_SUR = int(os.environ.get("H3_FIT_TEST_EVERY", "6"))
LAMBDAS = [float(x) for x in os.environ.get(
    "H3_LAMBDAS", "1e1,1e2,1e3,1e4,1e5,1e6").split(",")]
# Ecrire une matrice par valeur de lambda au lieu de la seule meilleure. Le
# cosinus ne predit pas le rendu -- une matrice moins bien notee sur tous les
# agregats peut reussir un prompt que la mieux notee rate -- donc la seule
# facon de choisir est de les essayer en generation.
TOUTES = os.environ.get("H3_FIT_SAVE_ALL", "") not in ("", "0")
DEVICE = os.environ.get("H3_ACC_DEVICE", "cuda")
DROP_FIRST = 1


def log(m):
    print(m, flush=True)


class Accumulateur:
    """Accumule X'X et X'Y ainsi que les moments d'ordre 1 et 2."""

    def __init__(self, d_in, d_out, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)
        self.xx, self.xy = z(d_in, d_in), z(d_in, d_out)
        self.sx, self.sy = z(d_in), z(d_out)
        self.sx2, self.sy2 = z(d_in), z(d_out)
        self.n = 0
        self.device = device

    def ajoute(self, x, y):
        x = x.to(self.device, torch.float32)
        y = y.to(self.device, torch.float32)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.sx2 += (x * x).sum(0)
        self.sy2 += (y * y).sum(0)
        self.n += x.shape[0]

    def resous(self, lam):
        """Ridge sur donnees centrees reduites. Resolution en float64."""
        n = self.n
        xx, xy = self.xx.double().cpu(), self.xy.double().cpu()
        sx, sy = self.sx.double().cpu(), self.sy.double().cpu()
        mx, my = sx / n, sy / n
        sdx = (self.sx2.double().cpu() / n - mx * mx).clamp_min(1e-12).sqrt()
        sdy = (self.sy2.double().cpu() / n - my * my).clamp_min(1e-12).sqrt()
        # Division par n : cxx devient la matrice de correlation, de diagonale 1,
        # et le terme lam/n ajoute plus bas pese alors reellement. Sans elle la
        # diagonale vaut n et la regularisation ne fait plus rien, quelle que
        # soit la valeur de lambda. Meme convention que h3_run.py, pour que les
        # lambdas retenus restent comparables d'un script a l'autre.
        cxx = (xx - torch.outer(sx, sx) / n) / (sdx.unsqueeze(1) * sdx.unsqueeze(0) * n)
        cxy = (xy - torch.outer(sx, sy) / n) / (sdx.unsqueeze(1) * sdy.unsqueeze(0) * n)
        d = cxx.shape[0]
        w = torch.linalg.solve(cxx + lam / n * torch.eye(d, dtype=torch.float64), cxy)
        return w, mx, sdx, my, sdy


def main():
    for p in (CIBLE, ENTREE):
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1

    t0 = time.time()
    log("Chargement de la cible...")
    tg = torch.load(CIBLE, map_location="cpu", weights_only=False, mmap=True)
    log("  %d prompts, modele %s" % (len(tg["indices"]), tg["model"]))
    log("Chargement des entrees...")
    en = torch.load(ENTREE, map_location="cpu", weights_only=False, mmap=True)
    log("  %d prompts, modele %s, taps %s"
        % (len(en["indices"]), en["model"], en["taps"]))
    log("  charge en %.0f s" % (time.time() - t0))

    if TAP not in en["taps"]:
        log("Tap %d absent du fichier d'entree (%s)" % (TAP, en["taps"]))
        return 1
    rang = en["taps"].index(TAP)

    par_index_c = {i: k for k, i in enumerate(tg["indices"])}
    communs = [i for i in en["indices"] if i in par_index_c]
    log("  %d prompts communs aux deux encodages" % len(communs))

    train = [i for k, i in enumerate(communs) if k % TEST_UN_SUR != 0]
    test = [i for k, i in enumerate(communs) if k % TEST_UN_SUR == 0]
    log("  %d entrainement, %d test" % (len(train), len(test)))
    log("")

    par_index_e = {i: k for k, i in enumerate(en["indices"])}
    d_in = en["data"][0].shape[-1]
    d_out = tg["data"][0].shape[-1]
    acc = Accumulateur(d_in, d_out, DEVICE)

    # Le sink est le premier token de la cible : direction constante, norme
    # enorme. Mesure ici, exclu de l'ajustement, substitue a l'inference.
    sinks = []

    # Les deux modeles ne produisent pas le meme nombre de tokens pour un meme
    # prompt : tokeniseurs differents. L'alignement se fait donc par le debut et
    # la queue la plus longue est jetee. C'est correct tant que l'ecart reste
    # marginal ; s'il est gros, l'appariement token a token est faux et le
    # cosinus plafonne pour une raison qui n'a rien a voir avec la couche.
    ecarts = []

    t0 = time.time()
    for n, i in enumerate(train):
        y = tg["data"][par_index_c[i]].float()
        x = en["data"][par_index_e[i]][rang].float()
        m = min(x.shape[0], y.shape[0])
        ecarts.append((x.shape[0], y.shape[0]))
        sinks.append(y[0])
        if m > DROP_FIRST:
            acc.ajoute(x[DROP_FIRST:m], y[DROP_FIRST:m])
        if (n + 1) % 1000 == 0:
            log("  %d/%d accumules, %d tokens" % (n + 1, len(train), acc.n))
    log("  %d tokens d'entrainement pour %d inconnues, %.0f s"
        % (acc.n, d_in, time.time() - t0))

    lx = torch.tensor([a for a, _ in ecarts], dtype=torch.float32)
    ly = torch.tensor([b for _, b in ecarts], dtype=torch.float32)
    d = lx - ly
    log("  longueurs : eleve %.1f, cible %.1f, ecart moyen %.2f token "
        "(min %d, max %d, identiques %.1f %%)"
        % (lx.mean(), ly.mean(), d.mean(), d.min(), d.max(),
           100.0 * float((d == 0).float().mean())))

    # Corpus supplementaires. Ils entrent dans les memes accumulateurs, donc dans
    # la meme resolution : une seule matrice en sort, pas un melange de deux.
    # Leurs prompts de test ne sont pas evalues ici -- le jeu de test reste celui
    # du corpus principal, pour que le cosinus affiche reste comparable aux
    # calibrations precedentes.
    for paire in EXTRA:
        c, e = paire.split("|", 1)
        if not (os.path.isfile(c) and os.path.isfile(e)):
            log("  corpus supplementaire introuvable, ignore : %s" % paire)
            continue
        tg2 = torch.load(c, map_location="cpu", weights_only=False, mmap=True)
        en2 = torch.load(e, map_location="cpu", weights_only=False, mmap=True)
        ic2 = {i: k for k, i in enumerate(tg2["indices"])}
        ie2 = {i: k for k, i in enumerate(en2["indices"])}
        r2 = 0 if len(en2["taps"]) == 1 else en2["taps"].index(TAP)
        avant = acc.n
        for _ in range(EXTRA_REPEAT):
          for i in en2["indices"]:
            if i not in ic2:
                continue
            y = tg2["data"][ic2[i]].float()
            x = en2["data"][ie2[i]][r2].float()
            m = min(x.shape[0], y.shape[0])
            if m > DROP_FIRST:
                acc.ajoute(x[DROP_FIRST:m], y[DROP_FIRST:m])
        log("  + %s x%d : %d tokens, soit %.1f %% du total"
            % (os.path.basename(c), EXTRA_REPEAT, acc.n - avant,
               100.0 * (acc.n - avant) / acc.n))
        del tg2, en2
        import gc
        gc.collect()

    sink = torch.stack(sinks).mean(0)
    cos_sink = torch.nn.functional.cosine_similarity(
        torch.stack(sinks), sink.unsqueeze(0), dim=1)
    log("  sink : norme %.1f, cosinus minimal %.4f" % (sink.norm(), cos_sink.min()))
    log("")

    # Jeu de test garde en float32 : le passer en float64 couterait 15 Go pour
    # une precision dont un cosinus n'a aucun besoin.
    # Troncature prompt par prompt AVANT la concatenation. L'inverse -- tout
    # concatener puis couper au total le plus court -- decale x par rapport a y
    # d'autant de tokens que le prompt precedent en avait en trop, et le decalage
    # s'accumule : au bout de quelques prompts les paires ne correspondent plus a
    # rien et le cosinus mesure du bruit.
    xs, ys = [], []
    for i in test:
        x = en["data"][par_index_e[i]][rang].float()
        y = tg["data"][par_index_c[i]].float()
        m = min(x.shape[0], y.shape[0])
        if m > DROP_FIRST:
            xs.append(x[DROP_FIRST:m])
            ys.append(y[DROP_FIRST:m])
    xte, yte = torch.cat(xs), torch.cat(ys)
    del xs, ys
    m = xte.shape[0]
    log("  %d tokens de test, %.1f Go" % (m, (xte.numel() + yte.numel()) * 4 / 1024 ** 3))
    log("")

    log("  lambda |   R2 test | cosinus test")
    log("  -------+-----------+-------------")
    best, resultats = None, []
    for lam in LAMBDAS:
        w, mx, sdx, my, sdy = acc.resous(lam)
        dev = torch.device(DEVICE)
        wf = w.float().to(dev)
        mxf, sdxf = mx.float().to(dev), sdx.float().to(dev)
        myf, sdyf = my.float().to(dev), sdy.float().to(dev)
        # Evaluation par tranches sur la carte. Le meme calcul sur processeur
        # coute 2,9 TFLOP par valeur de lambda, soit plusieurs minutes chacune ;
        # les tranches evitent d'avoir a y loger les 6,7 Go du jeu de test.
        num = den = 0.0
        scos = 0.0
        n_lignes = 0
        for k in range(0, xte.shape[0], 32768):
            xb = xte[k:k + 32768].to(dev, torch.float32)
            yb = yte[k:k + 32768].to(dev, torch.float32)
            pred = ((xb - mxf) / sdxf) @ wf
            gold = (yb - myf) / sdyf
            num += float(((gold - pred) ** 2).sum())
            den += float((gold ** 2).sum())
            scos += float(torch.nn.functional.cosine_similarity(
                pred, gold, dim=1).sum())
            n_lignes += xb.shape[0]
            del xb, yb, pred, gold
        r2 = 1.0 - num / den
        cos = scos / n_lignes
        del wf, mxf, sdxf, myf, sdyf
        log("  %6.0e | %9.4f | %12.4f" % (lam, r2, cos))
        resultats.append((lam, r2, cos))
        if best is None or cos > best[2]:
            best = (lam, r2, cos)

    log("")
    log("  retenu : lambda %.0e, R2 %.4f, cosinus %.4f" % best)

    if TOUTES:
        base, ext = os.path.splitext(SORTIE)
        for lam, r2, cos in resultats:
            w, mx, sdx, my, sdy = acc.resous(lam)
            t = {"W": w.float().contiguous(),
                 "mean_in": mx.float().contiguous(), "std_in": sdx.float().contiguous(),
                 "mean_out": my.float().contiguous(), "std_out": sdy.float().contiguous(),
                 "sink_out": sink.float().contiguous()}
            m = {"tap": str(TAP), "lambda": str(lam), "d_in": str(d_in),
                 "d_out": str(d_out), "cos_test": str(cos), "r2_test": str(r2),
                 "source_model": en["model"], "target_model": tg["model"],
                 "n_train_prompts": str(len(train)), "n_train_tokens": str(acc.n)}
            chemin = "%s_lam%s%s" % (base, ("%.0e" % lam).replace("+0", ""), ext)
            save_file(t, chemin, metadata=m)
            log("  ecrit %s" % os.path.basename(chemin))
    if best[0] in (LAMBDAS[0], LAMBDAS[-1]):
        log("  ATTENTION : lambda sur le bord de la grille, elargir H3_LAMBDAS")

    w, mx, sdx, my, sdy = acc.resous(best[0])
    tensors = {"W": w.float().contiguous(),
               "mean_in": mx.float().contiguous(), "std_in": sdx.float().contiguous(),
               "mean_out": my.float().contiguous(), "std_out": sdy.float().contiguous(),
               "sink_out": sink.float().contiguous()}
    meta = {"tap": str(TAP), "lambda": str(best[0]), "d_in": str(d_in), "d_out": str(d_out),
            "cos_test": str(best[2]), "r2_test": str(best[1]),
            "source_model": en["model"], "target_model": tg["model"],
            "n_train_prompts": str(len(train)), "n_train_tokens": str(acc.n)}
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    save_file(tensors, SORTIE, metadata=meta)
    log("")
    log("  ecrit : %s  (%.1f Mo)" % (SORTIE, os.path.getsize(SORTIE) / 1024 ** 2))
    with open(os.path.splitext(SORTIE)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
