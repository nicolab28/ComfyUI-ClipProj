#!/usr/bin/env python
"""Compare l'erreur de reconstruction sur les tokens d'un nom propre et ailleurs.

C'est la mesure qui separe les deux causes possibles de l'echec sur les gens
connus, qui appellent des remedes opposes.

Si l'erreur est bien plus forte sur les tokens du nom que sur le reste du
prompt, la projection extrapole faute d'avoir vu ces regions de l'espace pendant
la calibration : le corpus est en cause et l'enrichir suffit.

Si l'erreur est la meme, la projection fait son travail et c'est l'eleve qui
n'encode pas l'information. Aucun corpus n'y changera quoi que ce soit, il
faudra un autre eleve.

Le decoupage se fait par recherche de la sous-suite d'identifiants du nom dans
ceux du prompt : le tokeniseur est charge seul, sans poids, donc la mesure ne
mobilise aucune carte au-dela de la multiplication par la matrice.
"""

import json
import os
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
CIBLE = os.environ.get("H3_NAMES_TARGET", r"D:\tmp\h3_data\encode\target32b_gens.pt")
ENTREE = os.environ.get("H3_NAMES_INPUT", r"D:\tmp\h3_data\encode\input4b_gens.pt")
PROMPTS = os.environ.get("H3_NAMES_PROMPTS", r"D:\tmp\h3_data\h3_prompts_gens.txt")
PROJ_DIR = os.environ.get(
    "H3_NAMES_PROJ_DIR",
    r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI\models\clip_projections")
PROJS = [p.strip() for p in os.environ.get("H3_NAMES_PROJS", "").split(";") if p.strip()]
MIN_WORDS = int(os.environ.get("H3_MIN_WORDS", "15"))
# "1" pour les extremes, "2" pour la liste complete.
MAXW = int(os.environ.get("H3_NAMES_MAXWORDS", "0"))
MINW = int(os.environ.get("H3_NAMES_MINWORDS", "0"))
DETAIL = os.environ.get("H3_NAMES_DETAIL", "")
if DETAIL == "0":
    DETAIL = ""
DEVICE = os.environ.get("H3_ACC_DEVICE", "cuda")
DROP_FIRST = 1

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.text_encoders.minimax as minimax  # noqa: E402

# Les memes noms que le generateur de corpus. Les deux listes sont reunies : un
# corpus de test ne contient que des noms tenus a l'ecart, mais la recherche doit
# pouvoir les localiser aussi.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_JSON = os.environ.get("H3_PG_NOMS", "")
if _JSON and os.path.isfile(_JSON):
    import json as _json
    with open(_JSON, "r", encoding="utf-8") as _f:
        _d = _json.load(_f)
    NOMS_TRAIN, NOMS_TEST = _d["train"], _d["test"]
else:
    from gens import NOMS_TRAIN, NOMS_TEST  # noqa: E402

# Les plus longs d'abord : « Will Smith » ne doit pas etre trouve avant un nom
# dont il serait un prefixe.
NOMS = sorted(NOMS_TRAIN + NOMS_TEST, key=len, reverse=True)


def log(m):
    print(m, flush=True)


def charge_prompts():
    """Reproduit exactement la selection faite a l'encodage."""
    vus, gardes = set(), []
    jsonl = PROMPTS.lower().endswith(".jsonl")
    with open(PROMPTS, "r", encoding="utf-8") as f:
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


def construit_reseau(proj, device):
    """Reconstruit le residu s'il y en a un, comme le fait le node."""
    couches = sorted({int(k.split(".")[1]) for k in proj if k.startswith("mlp.")})
    if not couches:
        return None
    modules = []
    for n, i in enumerate(couches):
        w = proj["mlp.%d.weight" % i]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=("mlp.%d.bias" % i) in proj)
        lin.weight.data = w.to(device)
        if lin.bias is not None:
            lin.bias.data = proj["mlp.%d.bias" % i].to(device)
        modules.append(lin)
        if n < len(couches) - 1:
            modules.append(torch.nn.GELU())
    return torch.nn.Sequential(*modules).to(device).eval()


def charge_proj(nom):
    chemin = os.path.join(PROJ_DIR, nom)
    d = load_file(chemin)
    with safe_open(chemin, framework="pt") as f:
        meta = f.metadata() or {}
    return d, meta


def trouve(suite, motif, depuis=0):
    """Premiere position de `motif` dans `suite`, ou -1."""
    n, m = len(suite), len(motif)
    for i in range(depuis, n - m + 1):
        if suite[i:i + m] == motif:
            return i
    return -1


def main():
    for p in (CIBLE, ENTREE, PROMPTS):
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1
    if not PROJS:
        log("Renseigner H3_NAMES_PROJS, projections separees par des points-virgules.")
        return 1

    log("Chargement du tokeniseur...")
    tok = minimax.MiniMaxH3Tokenizer()
    ids_nom = {n: tok._text_ids(" " + n) for n in NOMS}
    ids_nom_debut = {n: tok._text_ids(n) for n in NOMS}

    corpus = charge_prompts()
    log("  %d prompts dans le corpus" % len(corpus))

    log("Chargement des encodages...")
    tg = torch.load(CIBLE, map_location="cpu", weights_only=False, mmap=True)
    en = torch.load(ENTREE, map_location="cpu", weights_only=False, mmap=True)
    log("  cible %s, entree %s, taps %s"
        % (tg["model"], en["model"], en["taps"]))

    ic = {i: k for k, i in enumerate(tg["indices"])}
    ie = {i: k for k, i in enumerate(en["indices"])}
    communs = [i for i in en["indices"] if i in ic]
    log("  %d prompts communs" % len(communs))

    # Filtre de longueur. Le conditionnement se degrade sur les prompts courts,
    # ou le nom pese un quart des tokens au lieu d'un dixieme : il faut pouvoir
    # regarder les deux regimes separement, une moyenne sur les deux ne dirait
    # rien de l'un ni de l'autre.
    if MAXW or MINW:
        avant = len(communs)
        communs = [i for i in communs
                   if (not MINW or len(corpus[i].split()) >= MINW)
                   and (not MAXW or len(corpus[i].split()) <= MAXW)]
        log("  %d retenus par le filtre de longueur (%s a %s mots)"
            % (len(communs), MINW or "-", MAXW or "-"))
        if not communs:
            log("  aucun prompt ne passe le filtre")
            return 1
        del avant

    # Reperage du nom dans chaque prompt, une fois pour toutes.
    spans, qui, sans = {}, {}, 0
    for i in communs:
        texte = corpus[i]
        ids = tg["ids"][ic[i]]
        trouve_ici = None
        for n in NOMS:
            if n not in texte:
                continue
            for motif in (ids_nom[n], ids_nom_debut[n]):
                p = trouve(ids, motif)
                if p >= 0:
                    trouve_ici = (p, p + len(motif))
                    break
            if trouve_ici:
                qui[i] = n
                break
        if trouve_ici is None:
            sans += 1
        else:
            spans[i] = trouve_ici
    log("  nom localise dans %d prompts, introuvable dans %d"
        % (len(spans), sans))
    if not spans:
        return 1
    longueurs = [b - a for a, b in spans.values()]
    log("  le nom occupe %.1f tokens en moyenne" % (sum(longueurs) / len(longueurs)))
    log("")

    dev = torch.device(DEVICE)
    rang = 0 if len(en["taps"]) == 1 else en["taps"].index(24)

    log("  %-44s %8s %8s %8s" % ("projection", "nom", "reste", "ecart"))
    log("  %s" % ("-" * 72))
    for nom_proj in PROJS:
        try:
            d, meta = charge_proj(nom_proj)
        except Exception as e:
            log("  %-44s  illisible : %s" % (nom_proj[:44], e))
            continue

        W = d["W"].to(dev)
        mi, si = d["mean_in"].to(dev), d["std_in"].to(dev)
        mo, so = d["mean_out"].to(dev), d["std_out"].to(dev)
        reseau = construit_reseau(d, dev)

        s_nom = n_nom = 0.0
        s_autre = n_autre = 0.0
        par_nom = {}
        for i in spans:
            x = en["data"][ie[i]][rang].to(dev, torch.float32)
            y = tg["data"][ic[i]].to(dev, torch.float32)
            m = min(x.shape[0], y.shape[0])
            x, y = x[:m], y[:m]
            xn = (x - mi) / si
            p = xn @ W
            if reseau is not None:
                p = p + reseau(xn)
            p = p * so + mo
            c = torch.nn.functional.cosine_similarity(p, y, dim=1)

            a, b = spans[i]
            masque = torch.zeros(m, dtype=torch.bool, device=dev)
            masque[max(a, DROP_FIRST):min(b, m)] = True
            masque[:DROP_FIRST] = False
            autre = ~masque
            autre[:DROP_FIRST] = False

            sn = float(c[masque].sum().detach())
            nn = int(masque.sum())
            s_nom += sn
            n_nom += nn
            s_autre += float(c[autre].sum().detach())
            n_autre += int(autre.sum())

            # Une moyenne sur cinquante noms masque completement les
            # disparites entre eux : un nom peut etre tres mal reconstruit
            # pendant qu'un autre l'est parfaitement, et c'est justement ce qui
            # decide du rendu. On garde donc le detail.
            q = qui[i]
            a_, b_ = par_nom.get(q, (0.0, 0))
            par_nom[q] = (a_ + sn, b_ + nn)

        cn = s_nom / max(1, n_nom)
        ca = s_autre / max(1, n_autre)
        log("  %-44s %8.4f %8.4f %+8.4f" % (nom_proj[:44], cn, ca, cn - ca))

        if DETAIL:
            classe = sorted(((s / max(1, n), q) for q, (s, n) in par_nom.items()))
            if DETAIL == "2":
                for v, q in classe:
                    log("        %6.4f  %s" % (v, q))
            else:
                log("      les dix moins bien reconstruits :")
                for v, q in classe[:10]:
                    log("        %6.4f  %s" % (v, q))
                log("      les cinq mieux reconstruits :")
                for v, q in classe[-5:]:
                    log("        %6.4f  %s" % (v, q))
        del W, mi, si, mo, so, reseau

    log("")
    log("  Lecture : un ecart nettement negatif signifie que la projection est")
    log("  moins bonne sur le nom que sur le reste, donc que le corpus manque de")
    log("  noms. Un ecart proche de zero disculpe la projection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
