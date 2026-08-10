#!/usr/bin/env python
"""Entraine un reseau en residu par-dessus la projection lineaire.

La matrice est au plafond de ce qu'un modele lineaire peut faire : huit fois plus
de donnees n'ont rapporte que 1,8 % de cosinus. Un reseau non lineaire devrait
aller plus loin, a condition de ne pas defaire ce qui marche deja.

D'ou le residu : la sortie vaut la projection lineaire plus la correction du
reseau, et la derniere couche est initialisee a zero. Au premier pas, le modele
reproduit donc exactement la matrice actuelle. Il ne peut qu'ameliorer, et si
l'entrainement echoue on retombe sur le point de depart.

Tout se passe dans l'espace centre reduit de la ridge, pour que le residu
travaille a la meme echelle que ce qu'il corrige.

L'arret anticipe se fait sur une separation PAR PROMPT et non par token : les
tokens d'un meme prompt sont correles, une separation par token laisserait fuiter
l'entrainement dans la validation et flatterait le resultat.
"""

import json
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

CIBLE = os.environ.get("H3_FIT_TARGET", r"D:\tmp\h3_data\encode\target32b_full.pt")
ENTREE = os.environ.get("H3_FIT_INPUT", r"D:\tmp\h3_data\encode\input4b_full.pt")
RIDGE = os.environ.get("H3_MLP_RIDGE", r"D:\tmp\h3_data\encode\projection.safetensors")
SORTIE = os.environ.get("H3_MLP_OUT", r"D:\tmp\h3_data\encode\projection_mlp.safetensors")

CACHE = int(os.environ.get("H3_MLP_HIDDEN", "8192"))
PROFONDEUR = int(os.environ.get("H3_MLP_DEPTH", "1"))

# Corpus supplementaire, sous la forme cible.pt|entree.pt, plusieurs paires
# separees par des virgules. La barre verticale plutot que le deux-points : un
# chemin Windows commence par une lettre de lecteur suivie de ce meme caractere. Sert a corriger un angle mort du corpus principal
# sans le refaire : les tokens ajoutes s'ajoutent a l'entrainement du residu,
# la ridge n'est pas retouchee.
#
# H3_MLP_EXTRA_REPEAT repete ces tokens. Le corpus de celebrites pese 48 000
# tokens contre 1,36 million pour le corpus general, soit 3,4 % : sans
# repetition il ne pese pas assez pour inflechir la descente. Repeter plutot que
# ponderer la perte garde la meme echelle de gradient par lot.
EXTRA = [p for p in os.environ.get("H3_MLP_EXTRA", "").split(",") if p.strip()]
EXTRA_REPEAT = int(os.environ.get("H3_MLP_EXTRA_REPEAT", "1"))
EPOQUES = int(os.environ.get("H3_MLP_EPOCHS", "60"))
LOT = int(os.environ.get("H3_MLP_BATCH", "8192"))
LR = float(os.environ.get("H3_MLP_LR", "1e-3"))
WD = float(os.environ.get("H3_MLP_WD", "0.01"))
PATIENCE = int(os.environ.get("H3_MLP_PATIENCE", "8"))
# Ecriture intermediaire du meilleur etat, toutes les N epoques. Un
# entrainement de deux cents epoques dure plus d'une heure : sans cela, une
# interruption ne laisse rien du tout, et on ne peut pas essayer le resultat
# avant la fin.
TOUS_LES = int(os.environ.get("H3_MLP_CHECKPOINT", "20"))
TEST_UN_SUR = int(os.environ.get("H3_FIT_TEST_EVERY", "6"))
DEVICE = os.environ.get("H3_ACC_DEVICE", "cuda")
DROP_FIRST = 1


# Le produit par la ridge et les etapes de l'optimiseur restent en float32 ;
# TF32 les accelere sans changer visiblement le resultat sur des matrices de
# cette taille.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def log(m):
    print(m, flush=True)


def charge_ridge(path):
    d = load_file(path)
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    return d, meta


def main():
    for p in (CIBLE, ENTREE, RIDGE):
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1

    ridge, meta = charge_ridge(RIDGE)
    tap = int(meta["tap"])
    log("Ridge de depart : tap %s, cosinus %s" % (meta["tap"], meta["cos_test"][:6]))

    t0 = time.time()
    log("Chargement des encodages...")
    tg = torch.load(CIBLE, map_location="cpu", weights_only=False, mmap=True)
    en = torch.load(ENTREE, map_location="cpu", weights_only=False, mmap=True)
    rang = en["taps"].index(tap)
    log("  %.0f s" % (time.time() - t0))

    ic = {i: k for k, i in enumerate(tg["indices"])}
    ie = {i: k for k, i in enumerate(en["indices"])}
    communs = [i for i in en["indices"] if i in ic]
    train = [i for k, i in enumerate(communs) if k % TEST_UN_SUR != 0]
    test = [i for k, i in enumerate(communs) if k % TEST_UN_SUR == 0]
    log("  %d prompts : %d entrainement, %d validation" % (len(communs), len(train), len(test)))

    mi, si = ridge["mean_in"].float(), ridge["std_in"].float()
    mo, so = ridge["mean_out"].float(), ridge["std_out"].float()
    W = ridge["W"].float()

    def empile(liste):
        xs, ys = [], []
        for i in liste:
            x = en["data"][ie[i]][rang].float()
            y = tg["data"][ic[i]].float()
            m = min(x.shape[0], y.shape[0])
            if m <= DROP_FIRST:
                continue
            xs.append(((x[DROP_FIRST:m] - mi) / si).half())
            ys.append(((y[DROP_FIRST:m] - mo) / so).half())
        return torch.cat(xs), torch.cat(ys)

    t0 = time.time()
    xtr, ytr = empile(train)
    xte, yte = empile(test)
    log("  %d tokens d'entrainement, %d de validation, %.0f s"
        % (xtr.shape[0], xte.shape[0], time.time() - t0))

    # Les deux blobs pesent une cinquantaine de gigaoctets et tout ce dont on a
    # besoin est desormais dans xtr/ytr. Les liberer avant d'entrainer evite de
    # les garder en memoire pendant une heure pour rien.
    del tg, en
    import gc
    gc.collect()

    # Corpus supplementaires, standardises avec les memes moments que la ridge :
    # le residu travaille dans SON espace, pas dans celui du corpus ajoute.
    for paire in EXTRA:
        c, e = paire.split("|", 1)
        if not (os.path.isfile(c) and os.path.isfile(e)):
            log("  corpus supplementaire introuvable, ignore : %s" % paire)
            continue
        tg2 = torch.load(c, map_location="cpu", weights_only=False, mmap=True)
        en2 = torch.load(e, map_location="cpu", weights_only=False, mmap=True)
        ic2 = {i: k for k, i in enumerate(tg2["indices"])}
        ie2 = {i: k for k, i in enumerate(en2["indices"])}
        r2 = 0 if len(en2["taps"]) == 1 else en2["taps"].index(tap)
        xs, ys = [], []
        for i in en2["indices"]:
            if i not in ic2:
                continue
            x = en2["data"][ie2[i]][r2].float()
            y = tg2["data"][ic2[i]].float()
            m = min(x.shape[0], y.shape[0])
            if m <= DROP_FIRST:
                continue
            xs.append(((x[DROP_FIRST:m] - mi) / si).half())
            ys.append(((y[DROP_FIRST:m] - mo) / so).half())
        xa, ya = torch.cat(xs), torch.cat(ys)
        del tg2, en2, xs, ys
        gc.collect()
        part = 100.0 * xa.shape[0] * EXTRA_REPEAT / (xtr.shape[0] + xa.shape[0] * EXTRA_REPEAT)
        log("  + %s : %d tokens x%d, soit %.1f %% du melange"
            % (os.path.basename(c), xa.shape[0], EXTRA_REPEAT, part))
        xtr = torch.cat([xtr] + [xa] * EXTRA_REPEAT)
        ytr = torch.cat([ytr] + [ya] * EXTRA_REPEAT)
        del xa, ya
        gc.collect()
    log("  encodages liberes, %.1f Go conserves"
        % ((xtr.numel() + ytr.numel() + xte.numel() + yte.numel()) * 2 / 1024 ** 3))
    log("")

    d_in, d_out = xtr.shape[1], ytr.shape[1]
    dev = torch.device(DEVICE)
    Wd = W.to(dev)

    # Une couche cachee par defaut. Le node reconstruit le reseau a partir des
    # seules cles mlp.N du fichier et intercale un GELU entre chaque paire, donc
    # la profondeur passe sans rien changer de son cote.
    couches = []
    for k in range(PROFONDEUR):
        couches.append(torch.nn.Linear(d_in if k == 0 else CACHE, CACHE))
        couches.append(torch.nn.GELU())
    couches.append(torch.nn.Linear(CACHE if PROFONDEUR else d_in, d_out))
    reseau = torch.nn.Sequential(*couches).to(dev)
    # Derniere couche a zero : au depart la sortie vaut exactement la ridge.
    torch.nn.init.zeros_(reseau[-1].weight)
    torch.nn.init.zeros_(reseau[-1].bias)
    n_par = sum(p.numel() for p in reseau.parameters())
    log("  reseau %d -> %s -> %d, %.1f M parametres, %.0f Mo"
        % (d_in, " -> ".join([str(CACHE)] * PROFONDEUR) or "rien",
           d_out, n_par / 1e6, n_par * 4 / 1024 ** 2))

    opt = torch.optim.AdamW(reseau.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOQUES)

    def evalue():
        reseau.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for k in range(0, xte.shape[0], 32768):
                x = xte[k:k + 32768].to(dev).float()
                y = yte[k:k + 32768].to(dev).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    correction = reseau(x)
                p = x @ Wd + correction.float()
                c = torch.nn.functional.cosine_similarity(p, y, dim=1)
                tot += float(c.sum())
                n += c.numel()
        reseau.train()
        return tot / n

    base = evalue()
    log("  cosinus de depart (ridge seule) : %.4f" % base)
    log("")
    log("  epoque |   perte | cosinus validation")
    log("  -------+---------+-------------------")

    def ecris(etat, cos, provisoire=False):
        """Ecrit la projection augmentee du residu."""
        tensors = dict(ridge)
        for k, v in etat.items():
            tensors["mlp." + k] = v.float().contiguous()
        m = dict(meta)
        m.update({"mlp": "true", "mlp_hidden": str(CACHE),
                  "mlp_depth": str(PROFONDEUR),
                  "cos_test": str(cos), "cos_test_ridge": str(base)})
        save_file(tensors, SORTIE, metadata=m)
        if not provisoire:
            with open(os.path.splitext(SORTIE)[0] + ".json", "w",
                      encoding="utf-8") as f:
                json.dump(m, f, indent=2)
        return m

    meilleur, meilleur_etat, patience = base, None, 0
    for ep in range(EPOQUES):
        perm = torch.randperm(xtr.shape[0])
        somme, nlots = 0.0, 0
        for k in range(0, xtr.shape[0], LOT):
            idx = perm[k:k + LOT]
            x = xtr[idx].pin_memory().to(dev, non_blocking=True).float()
            y = ytr[idx].pin_memory().to(dev, non_blocking=True).float()
            # Seul le reseau passe en bf16 : c'est lui qui coute, deux matmuls
            # contre un pour la ridge, et mesure a 242 ms en float32 contre 91.
            # La base reste en pleine precision, sinon le residu corrigerait du
            # bruit d'arrondi au lieu de corriger la projection.
            lineaire = x @ Wd
            with torch.autocast("cuda", dtype=torch.bfloat16):
                correction = reseau(x)
            p = lineaire + correction.float()
            perte = torch.nn.functional.mse_loss(p, y)
            opt.zero_grad(set_to_none=True)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(reseau.parameters(), 1.0)
            opt.step()
            somme += float(perte)
            nlots += 1
        sched.step()
        cos = evalue()
        marque = ""
        if cos > meilleur:
            meilleur = cos
            meilleur_etat = {k: v.detach().cpu().clone() for k, v in reseau.state_dict().items()}
            patience = 0
            marque = "  <-"
        else:
            patience += 1
        log("  %6d | %7.4f | %17.4f%s" % (ep, somme / nlots, cos, marque))
        if (TOUS_LES and meilleur_etat is not None
                and (ep + 1) % TOUS_LES == 0 and meilleur > base):
            ecris(meilleur_etat, meilleur, provisoire=True)
            log("         ecrit en cours de route (%.4f)" % meilleur)
        if patience >= PATIENCE:
            log("  arret anticipe : %d epoques sans progres" % PATIENCE)
            break

    log("")
    log("  ridge seule      %.4f" % base)
    log("  ridge + reseau   %.4f   (%+.4f)" % (meilleur, meilleur - base))

    if meilleur_etat is None or meilleur <= base:
        log("  le reseau n'apporte rien, rien n'est ecrit")
        return 0

    ecris(meilleur_etat, meilleur)
    log("  ecrit : %s  (%.0f Mo)" % (SORTIE, os.path.getsize(SORTIE) / 1024 ** 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
