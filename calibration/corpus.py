#!/usr/bin/env python
"""Fabrique le corpus de noms propres a partir de la liste TMDB filtree.

Remplace la liste de soixante noms ecrite a la main : celle-ci n'etait qu'une
demonstration, et la mesure a montre qu'une personne absente du corpus reste
cassee -- la projection n'apprend rien des noms propres en general, seulement de
ceux qu'elle a vus.

Cinq contextes par personne sont ecrits ici alors que deux suffisent (0,9875
contre 0,9945 sur des contextes non vus). Le texte ne coute rien, c'est
l'encodage qui coute : en produire cinq laisse le choix du ratio au moment de la
calibration, par simple sous-echantillonnage, sans avoir a tout refaire.

Les phrases restent courtes et nues. Le format H3 long est couvert separement
par h3_promptgen.py, qui fait reecrire ces memes descriptions par un LLM.
"""

import json
import os
import random
import sys

LISTE = os.environ.get("H3_CORPUS_LISTE", r"D:\tmp\h3_data\celebrites.json")
DOSSIER = os.environ.get("H3_CORPUS_DIR", r"D:\tmp\h3_data")
PREFIXE = os.environ.get("H3_CORPUS_PREFIXE", "h3_cel")
GRAINE = int(os.environ.get("H3_CORPUS_SEED", "20260810"))
DEBUT = int(os.environ.get("H3_CORPUS_DEBUT", "0"))        # rang du premier nom
N_NOMS = int(os.environ.get("H3_CORPUS_NOMS", "0"))        # 0 = toute la liste
PAR_NOM = int(os.environ.get("H3_CORPUS_PAR_NOM", "5"))
# Noms tenus entierement a l'ecart, pour mesurer ce que la calibration ne
# generalise pas. Preleves dans le haut du classement, pas dans la queue, sinon
# on mesurerait surtout des gens que le modele ne connait pas.
N_TENUS = int(os.environ.get("H3_CORPUS_TENUS", "150"))

ACTIVITES = [
    "riding a bicycle along a canal", "eating a plate of pasta at a small table",
    "skiing down a snowy slope", "walking a dog through a park",
    "playing an acoustic guitar", "drinking coffee at a counter",
    "reading a newspaper on a bench", "cooking in a home kitchen",
    "boarding a train with a suitcase", "painting at an easel",
    "swimming in an outdoor pool", "repairing a motorcycle in a garage",
    "shopping for vegetables at a market", "typing on a laptop in a library",
    "throwing a ball to a child", "climbing a rock face with ropes",
    "waiting alone at a bus stop", "carrying groceries up a staircase",
    "playing chess in a public square", "fishing from a small wooden boat",
    "hanging washing on a line", "sweeping the floor of an empty shop",
    "queuing outside a cinema", "feeding pigeons on a plaza",
    "sharpening a knife in a kitchen", "tuning a radio in a car",
    "folding a paper map on a bonnet", "lighting a candle on a windowsill",
    "running up a flight of stone steps", "sitting on the edge of a fountain",
]

LIEUX = [
    "in a narrow city street", "on a windswept beach", "in a hotel lobby",
    "in a wheat field", "on a railway platform", "in a hospital corridor",
    "in a crowded night market", "on a rooftop terrace",
    "in a wood-panelled study", "in an underground car park",
    "at the edge of a pine forest", "in a laundrette",
    "on a suspension bridge", "in a greenhouse", "in a car repair shop",
    "on a ferry deck", "in a school gymnasium", "in a stone courtyard",
]

METEO = [
    "in heavy rain", "under a clear blue sky", "in thick fog",
    "during a snowfall", "in strong wind", "under an overcast sky",
    "in bright midday sun", "at golden hour", "just after a storm",
    "in light drizzle", "under a starry sky", "in oppressive heat",
]

MOMENTS = ["at dawn", "in the early morning", "at midday", "in the afternoon",
           "at dusk", "at night", "late at night"]

CAMERA = [
    "medium close-up", "wide shot", "over-the-shoulder shot",
    "low angle shot", "handheld tracking shot", "static locked-off shot",
    "slow dolly in", "close-up on the face",
]

ATTRIBUTS_M = [
    "in his early twenties", "in his thirties", "in his forties",
    "in his fifties", "as an old man, white hair", "with a shaved head",
    "with long hair", "wearing round glasses", "with a full beard",
    "clean-shaven, short dark hair", "wearing a heavy winter coat",
    "in a plain white t-shirt", "in a formal dark suit",
]

ATTRIBUTS_F = [
    "in her early twenties", "in her thirties", "in her forties",
    "in her fifties", "as an old woman, white hair", "with a short pixie cut",
    "with long hair", "wearing round glasses", "with hair tied back",
    "with straight dark hair", "wearing a heavy winter coat",
    "in a plain white t-shirt", "in a formal dark dress",
]

# Trois longueurs. Le corpus general n'a rien sous quinze mots -- son minimum est
# un filtre que j'ai pose moi-meme -- et c'est justement sous ce seuil que la
# generation se degrade. Les formes nues sont donc majoritaires ici.
def tres_court(nom, att, act, lieu, meteo, moment, cam):
    return "%s %s." % (nom, act)


def court(nom, att, act, lieu, meteo, moment, cam):
    return "%s %s, %s." % (nom, att, act)


def moyen(nom, att, act, lieu, meteo, moment, cam):
    return "A short video of %s %s, %s, %s." % (nom, att, act, meteo)


def long(nom, att, act, lieu, meteo, moment, cam):
    # Le lieu n'est plus tire separement : chaque activite porte deja le sien
    # -- « at a market », « in a garage » -- et en ajouter un second produisait
    # des scenes qui se contredisent, du texte que le modele ne voit jamais.
    return ("A short video of %s %s, %s %s, %s. %s, shallow depth of field."
            % (nom, att, act, moment, meteo, cam))


FORMES = [tres_court, court, court, moyen, moyen, long]


def fabrique(noms, rnd, par_nom):
    lignes = []
    for p in noms:
        attrs = ATTRIBUTS_F if p["genre"] == "f" else ATTRIBUTS_M
        vus = set()
        essais = 0
        while len(vus) < par_nom and essais < par_nom * 30:
            essais += 1
            ligne = rnd.choice(FORMES)(
                p["nom"], rnd.choice(attrs), rnd.choice(ACTIVITES),
                rnd.choice(LIEUX), rnd.choice(METEO), rnd.choice(MOMENTS),
                rnd.choice(CAMERA))
            if ligne in vus:
                continue
            vus.add(ligne)
            lignes.append(ligne)
    return lignes


def ecris(chemin, lignes):
    with open(chemin, "w", encoding="utf-8") as f:
        for l in lignes:
            f.write(l + "\n")
    mots = sum(len(l.split()) for l in lignes) / max(1, len(lignes))
    print("  %-28s %6d prompts, %.1f mots" % (os.path.basename(chemin),
                                              len(lignes), mots))


def main():
    if not os.path.isfile(LISTE):
        print("Liste introuvable : %s" % LISTE)
        return 1
    with open(LISTE, "r", encoding="utf-8") as f:
        tous = json.load(f)
    tous = tous[DEBUT:]
    if N_NOMS > 0:
        tous = tous[:N_NOMS]

    rnd = random.Random(GRAINE)
    # Les noms tenus a l'ecart sont pris un sur k dans le haut du classement,
    # pour que le jeu de controle ait la meme notoriete moyenne que le reste.
    if N_TENUS <= 0:
        tenus = []
    else:
        pas = max(2, len(tous) // N_TENUS)
        # Decale d'un demi-pas : commencer a l'indice 0 emporterait le nom le
        # plus celebre de la liste, qui est justement celui qu'il faut couvrir.
        tenus = [p for i, p in enumerate(tous)
                 if i % pas == pas // 2][:N_TENUS]
    vus_tenus = {p["nom"] for p in tenus}
    entraines = [p for p in tous if p["nom"] not in vus_tenus]

    print("%d noms : %d a l'entrainement, %d tenus a l'ecart"
          % (len(tous), len(entraines), len(tenus)))

    train = fabrique(entraines, rnd, PAR_NOM)
    rnd.shuffle(train)
    test = fabrique(tenus, rnd, PAR_NOM)

    os.makedirs(DOSSIER, exist_ok=True)
    ecris(os.path.join(DOSSIER, PREFIXE + "_train.txt"), train)
    ecris(os.path.join(DOSSIER, PREFIXE + "_test_nom.txt"), test)

    with open(os.path.join(DOSSIER, PREFIXE + "_noms.json"), "w",
              encoding="utf-8") as f:
        json.dump({"train": [p["nom"] for p in entraines],
                   "test": [p["nom"] for p in tenus]}, f, ensure_ascii=False)
    print("  %-28s %d + %d noms" % (PREFIXE + "_noms.json",
                                    len(entraines), len(tenus)))
    print("")
    for l in train[:4]:
        print("  %s" % l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
