#!/usr/bin/env python
"""Fabrique un corpus de prompts centres sur des personnes connues.

Le corpus de calibration actuel n'en contient pratiquement pas : environ 70
lignes sur 8632, soit de l'ordre de 0,02 % des tokens d'entrainement. Les
directions qui encodent un visage connu ne sont donc contraintes par rien
pendant l'ajustement, et la matrice y fait n'importe quoi sans que le cosinus
moyen s'en apercoive.

Le corpus est volontairement redondant : chaque nom revient avec de nombreuses
activites et de nombreux attributs. On cherche ici a savoir jusqu'ou une
projection peut aller si on la sur-entraine sur des gens, pas a produire un jeu
equilibre.

D'ou la separation en trois fichiers, qui est tout l'interet du dispositif :

    train          ce que la projection voit
    test_activite  memes noms, activites jamais associees a eux
    test_nom       noms entierement absents de l'entrainement

Une projection qui reussit sur `train` et echoue sur `test_activite` a memorise
des couples, elle n'a rien appris de transferable. Une qui tient sur
`test_activite` mais tombe sur `test_nom` a appris a traiter ces gens-la.
Une qui tient sur les trois a appris quelque chose sur les noms propres en
general, et c'est la seule situation qui justifierait d'elargir le corpus.

L'attribut est la partie qui echoue en generation -- « young Will Smith » donne
un grisonnant -- donc il est varie systematiquement. La longueur du prompt varie
aussi : le conditionnement se degrade sur les prompts courts, il faut les deux.
"""

import os
import random
import sys

DOSSIER = os.environ.get("H3_GENS_DIR", r"D:\tmp\h3_data")
GRAINE = int(os.environ.get("H3_GENS_SEED", "12345"))
PAR_NOM = int(os.environ.get("H3_GENS_PER_NAME", "40"))

NOMS_TRAIN = [
    "Will Smith", "Denzel Washington", "Morgan Freeman", "Samuel L. Jackson",
    "Tom Cruise", "Brad Pitt", "Leonardo DiCaprio", "Keanu Reeves",
    "Robert Downey Jr.", "Matt Damon", "Ryan Gosling", "Christian Bale",
    "Hugh Jackman", "Idris Elba", "Jason Statham", "Daniel Craig",
    "Scarlett Johansson", "Natalie Portman", "Emma Stone", "Charlize Theron",
    "Angelina Jolie", "Nicole Kidman", "Julia Roberts", "Cate Blanchett",
    "Meryl Streep", "Anne Hathaway", "Jennifer Lawrence", "Margot Robbie",
    "Zendaya", "Halle Berry", "Viola Davis", "Lupita Nyong'o",
    "Jackie Chan", "Bruce Lee", "Jet Li", "Donnie Yen",
    "Tony Leung", "Andy Lau", "Gong Li", "Zhang Ziyi",
    "Lionel Messi", "Cristiano Ronaldo", "Michael Jordan", "LeBron James",
    "Barack Obama", "Elon Musk", "Oprah Winfrey", "Beyonce",
]

# Jamais vus a l'entrainement. Meme nature que les autres -- acteurs, sportifs,
# figures publiques, occidentaux et asiatiques -- pour que l'echec eventuel ne
# s'explique pas par un changement de famille.
NOMS_TEST = [
    "Ken Watanabe", "Hiroyuki Sanada", "Takeshi Kitano", "Rinko Kikuchi",
    "Kylian Mbappe", "Serena Williams", "Freddie Mercury", "Steve Jobs",
]

ACTIVITES_TRAIN = [
    "riding a bicycle along a canal",
    "eating a plate of pasta at a small table",
    "skiing down a snowy slope",
    "walking a dog through a park",
    "playing an acoustic guitar",
    "drinking coffee at a counter",
    "reading a newspaper on a bench",
    "cooking in a home kitchen",
    "boarding a train with a suitcase",
    "painting at an easel",
    "swimming in an outdoor pool",
    "repairing a motorcycle in a garage",
    "shopping for vegetables at a market",
    "typing on a laptop in a library",
    "throwing a ball to a child",
    "climbing a rock face with ropes",
]

# Jamais associees aux noms d'entrainement.
ACTIVITES_TEST = [
    "playing chess in a public square",
    "fishing from a small wooden boat",
    "carrying groceries up a staircase",
    "waiting alone at a bus stop in the rain",
]

ATTRIBUTS_M = [
    "in his early twenties, smooth unlined face",
    "in his thirties, jet black hair, clean-shaven",
    "in his forties, short beard",
    "in his fifties, silver hair",
    "as an old man, deeply lined face, white hair",
    "with a shaved head",
    "with long shoulder-length hair",
    "wearing round glasses",
]

ATTRIBUTS_F = [
    "in her early twenties, smooth unlined face",
    "in her thirties, long dark hair",
    "in her forties, hair tied back",
    "in her fifties, shoulder-length grey hair",
    "as an old woman, deeply lined face, white hair",
    "with a short pixie cut",
    "with long shoulder-length hair",
    "wearing round glasses",
]

DECORS = [
    "sunny afternoon, shallow depth of field",
    "overcast grey daylight, muted colours",
    "warm tungsten light from the side, dim room",
    "blue hour, city lights coming on",
    "harsh midday sun, hard shadows",
    "soft window light, quiet interior",
]

VOIX_M = ["He speaks a single calm sentence to the camera.",
          "He laughs briefly, then says a few words.",
          "He says nothing, only ambient sound."]

VOIX_F = ["She speaks a single calm sentence to the camera.",
          "She laughs briefly, then says a few words.",
          "She says nothing, only ambient sound."]

FEMININS = {
    "Scarlett Johansson", "Natalie Portman", "Emma Stone", "Charlize Theron",
    "Angelina Jolie", "Nicole Kidman", "Julia Roberts", "Cate Blanchett",
    "Meryl Streep", "Anne Hathaway", "Jennifer Lawrence", "Margot Robbie",
    "Zendaya", "Halle Berry", "Viola Davis", "Lupita Nyong'o", "Gong Li",
    "Zhang Ziyi", "Rinko Kikuchi", "Serena Williams", "Oprah Winfrey",
    "Beyonce",
}

# Les trois longueurs, du minimum au prompt complet. Un corpus qui ne
# contiendrait que des prompts longs laisserait les courts hors distribution,
# et c'est justement sur les courts que la generation se degrade.
def court(nom, att, act, dec, voix):
    return "%s %s." % (nom, act)


def moyen(nom, att, act, dec, voix):
    return "A short video of %s %s, %s." % (nom, att, act)


def long(nom, att, act, dec, voix):
    return ("A short video of %s %s, %s. %s. %s"
            % (nom, att, act, dec, voix))


FORMES = [court, moyen, long]


def fabrique(noms, activites, rnd, par_nom):
    lignes = []
    for nom in noms:
        fem = nom in FEMININS
        attrs = ATTRIBUTS_F if fem else ATTRIBUTS_M
        voix = VOIX_F if fem else VOIX_M
        vus = set()
        essais = 0
        while len(vus) < par_nom and essais < par_nom * 20:
            essais += 1
            a, ac = rnd.choice(attrs), rnd.choice(activites)
            d, v = rnd.choice(DECORS), rnd.choice(voix)
            forme = rnd.choice(FORMES)
            ligne = forme(nom, a, ac, d, v)
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
    print("  %-34s %5d prompts, %.1f mots" % (os.path.basename(chemin),
                                              len(lignes), mots))


def main():
    rnd = random.Random(GRAINE)
    os.makedirs(DOSSIER, exist_ok=True)

    train = fabrique(NOMS_TRAIN, ACTIVITES_TRAIN, rnd, PAR_NOM)
    rnd.shuffle(train)

    # Memes noms, activites jamais vues avec eux.
    t_act = fabrique(NOMS_TRAIN, ACTIVITES_TEST, rnd, 3)
    # Noms jamais vus, activites d'entrainement pour ne changer qu'une chose.
    t_nom = fabrique(NOMS_TEST, ACTIVITES_TRAIN, rnd, 12)

    print("ecrit dans %s" % DOSSIER)
    ecris(os.path.join(DOSSIER, "h3_gens_train.txt"), train)
    ecris(os.path.join(DOSSIER, "h3_gens_test_activite.txt"), t_act)
    ecris(os.path.join(DOSSIER, "h3_gens_test_nom.txt"), t_nom)

    print("")
    print("  %d noms a l'entrainement, %d tenus a l'ecart"
          % (len(NOMS_TRAIN), len(NOMS_TEST)))
    print("  %d activites a l'entrainement, %d tenues a l'ecart"
          % (len(ACTIVITES_TRAIN), len(ACTIVITES_TEST)))
    print("")
    print("  A comparer en generation, meme seed :")
    for titre, lot in (("vu a l'entrainement", train),
                       ("activite jamais vue", t_act),
                       ("nom jamais vu", t_nom)):
        exemple = next(l for l in lot if l.startswith("A short video"))
        print("    [%s]" % titre)
        print("    %s" % exemple)
    return 0


if __name__ == "__main__":
    sys.exit(main())
