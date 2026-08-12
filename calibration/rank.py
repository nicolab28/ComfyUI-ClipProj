#!/usr/bin/env python
"""Compare la quantite d'information reellement portee par les deux
conditionnements : les douze taps du 4B que consomme Krea2, contre l'etat final
du 32B que consomme H3.

Compter les dimensions ne suffit pas. Douze taps du meme flux residuel sont
fortement correles -- le tap 32 et le tap 35 sont presque le meme vecteur --
donc 30720 nombres ne font pas 30720 directions utiles. On mesure donc le rang
effectif, par le ratio de participation des valeurs propres de la covariance :

    PR = (somme des valeurs propres)^2 / somme des carres

qui vaut d le long de d directions d'egale variance, et 1 si tout tient sur une
seule. Le calcul passe par la matrice de Gram sur un echantillon de tokens, dont
les valeurs propres sont celles de la covariance a des zeros pres, ce qui evite
de former une covariance 30720 x 30720.

Chaque tap est standardise avant concatenation : le `txtfusion` de Krea2 applique
un RMSNorm par tap, donc laisser un tap de grande norme ecraser les autres
mesurerait un artefact d'echelle plutot que de l'information.

Variables : H3_RANG_TAPS4B (fragment du 4B), H3_RANG_TARGET (sortie du 32B),
H3_RANG_TOKENS (taille de l'echantillon), H3_RANG_DEVICE.
"""

import os
import torch

TAPS4B = os.environ["H3_RANG_TAPS4B"]
TARGET = os.environ["H3_RANG_TARGET"]
N_TOKENS = int(os.environ.get("H3_RANG_TOKENS", "8000"))
DEVICE = torch.device(os.environ.get("H3_RANG_DEVICE", "cuda:2"))

torch.manual_seed(0)


def echantillon(sequences, longueurs, n):
    """Tire n positions de token au hasard, en ignorant le remplissage.

    Les fragments sont ecrits a longueur fixe, 320 positions, et les prompts
    plus courts sont completes. Echantillonner sans tenir compte de la longueur
    reelle ferait entrer du remplissage dans la covariance, dont le rang est
    exactement un.

    La position 0 est ecartee : c'est un puits d'attention, de norme 16500
    contre 291 pour un token de texte, et sa variance ecrase tout le reste. Le
    mesurer donnerait un rang effectif de 1 pour n'importe quel modele.

    Args:
        sequences (list[torch.Tensor]): un tenseur [seq, d] par prompt.
        longueurs (list[int]): nombre de tokens reels de chaque prompt.
        n (int): nombre de positions voulues.

    Returns:
        torch.Tensor: [n, d] sur le peripherique de calcul, en float32.
    """
    lots = []
    part = max(1, n // len(sequences))
    for s, L in zip(sequences, longueurs):
        L = min(L, s.shape[0])
        k = min(part, max(1, L - 1))
        idx = torch.randperm(L - 1)[:k] + 1   # on saute le puits
        lots.append(s[idx].to(DEVICE, torch.float32))
    return torch.cat(lots, dim=0)[:n]


def rang_effectif(x):
    """Ratio de participation des valeurs propres de la covariance de x.

    Args:
        x (torch.Tensor): [n, d], centre a l'interieur.

    Returns:
        tuple[float, float]: (rang effectif, part de variance des 10 premieres).
    """
    x = x - x.mean(dim=0, keepdim=True)
    g = (x @ x.T) / x.shape[0]
    vp = torch.linalg.eigvalsh(g.double()).clamp(min=0)
    pr = (vp.sum() ** 2 / (vp ** 2).sum()).item()
    top = (vp.sort(descending=True).values[:10].sum() / vp.sum()).item()
    return pr, top


print("chargement...")
a = torch.load(TAPS4B, map_location="cpu", mmap=True, weights_only=False)
b = torch.load(TARGET, map_location="cpu", mmap=True, weights_only=False)
taps = list(a["taps"])
print("  4B  : %d prompts, taps %s" % (len(a["data"]), taps))
print("  32B : %d prompts" % len(b["data"]))

# Les deux encodages doivent porter sur les memes prompts : on apparie par
# l'index de corpus, pas par la position dans le fichier.
pos32 = {ix: k for k, ix in enumerate(b["indices"])}
paires = [(k, pos32[ix]) for k, ix in enumerate(a["indices"]) if ix in pos32]
print("  %d prompts communs" % len(paires))
long4 = [len(a["ids"][k]) for k, _ in paires]

# --- 32B : un seul etat, 5120 dimensions ---------------------------------
def normaliser(e):
    """Met a l'echelle par la RMS globale, comme le RMSNorm du DiT.

    On ne centre pas : soustraire la moyenne par dimension retirerait la
    composante commune aux taps, c'est-a-dire exactement la redondance qu'on
    cherche a mesurer.
    """
    return e / e.pow(2).mean().sqrt().clamp(min=1e-6)


cible = normaliser(echantillon([b["data"][j] for _, j in paires], long4, N_TOKENS))
pr32, top32 = rang_effectif(cible)
print("\n32B, etat final       : %5d dims, rang effectif %7.1f, "
      "10 premieres directions %5.1f %% de la variance"
      % (cible.shape[1], pr32, 100 * top32))

# --- 4B : les douze taps de Krea2, standardises puis concatenes -----------
voulus = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
dispo = [t for t in voulus if t in taps]
print("\n4B, taps demandes %s" % voulus)
print("   taps presents dans le fragment : %s" % dispo)

morceaux = []
for t in dispo:
    j = taps.index(t)
    morceaux.append(normaliser(
        echantillon([a["data"][k][j] for k, _ in paires], long4, N_TOKENS)))

pile = torch.cat(morceaux, dim=1)
pr4, top4 = rang_effectif(pile)
print("\n4B, %2d taps empiles   : %5d dims, rang effectif %7.1f, "
      "10 premieres directions %5.1f %% de la variance"
      % (len(dispo), pile.shape[1], pr4, 100 * top4))

# Redondance entre taps : cosinus moyen entre deux taps sur le meme token.
print("\ncosinus moyen entre taps, meme token :")
print("        " + "".join("%7d" % t for t in dispo))
for i, ti in enumerate(dispo):
    ligne = "  %3d  " % ti
    for j, _ in enumerate(dispo):
        c = torch.nn.functional.cosine_similarity(
            morceaux[i], morceaux[j], dim=1).mean().item()
        ligne += "%7.3f" % c
    print(ligne)

print("\nrang effectif par dimension declaree :")
print("  32B  %.4f   (%.1f / %d)" % (pr32 / cible.shape[1], pr32, cible.shape[1]))
print("  4B   %.4f   (%.1f / %d)" % (pr4 / pile.shape[1], pr4, pile.shape[1]))
