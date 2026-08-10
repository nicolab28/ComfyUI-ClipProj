#!/usr/bin/env python
"""Transforme des descriptions courtes en prompts au format H3, sur N cartes.

Le corpus de celebrites actuel est fait de phrases nues d'une vingtaine de mots.
Utile pour couvrir les noms propres, inutile pour couvrir le format reel des
prompts : ceux du corpus general font 157 mots de mediane et suivent une
structure en sections. Une matrice calibree sur les deux regimes a besoin des
deux, et c'est ce que ce script fabrique.

Un serveur ollama ne sait pas router une requete vers une carte donnee, donc on
lance une instance par carte, chacune sur son port, et on repartit le travail
entre elles. Les serveurs sont demarres et arretes ici plutot que dans le script
shell : ca evite toute redirection et ca garantit qu'ils ne survivent pas au
script.

L'ecriture se fait en JSONL au fil de l'eau et le fichier est relu au demarrage,
donc une interruption ne coute que les generations en vol.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ollama : une instance par carte, gratuit, mobilise les GPU.
# anthropic ou mistral : rien a installer, laisse les cartes libres pour un
# entrainement en cours, et ecrit dans un style plus proche des vrais prompts
# qu'un 8B quantifie en Q4.
# Un ou plusieurs moteurs, separes par des virgules. Avec deux moteurs, les
# ouvriers sont repartis entre eux : le corpus melange alors deux styles, ce qui
# evite de calibrer sur les tics d'un seul modele.
MOTEURS = [m.strip().lower() for m in
           os.environ.get("H3_PG_BACKEND", "ollama").split(",") if m.strip()]
MOTEUR = MOTEURS[0]

OLLAMA = os.environ.get(
    "H3_OLLAMA",
    r"C:\Users\Nicolas\AppData\Local\Programs\Ollama\ollama.exe")
DEFAUTS = {"ollama": "huihui_ai/qwen3-vl-abliterated:8b-instruct",
           "anthropic": "claude-haiku-4-5-20251001",
           "mistral": "mistral-small-latest",
           "gemini": "gemini-3.6-flash"}
MODELES = {m: os.environ.get("H3_PG_MODEL_" + m.upper(), DEFAUTS[m])
           for m in MOTEURS if m in DEFAUTS}
MODELE = MODELES.get(MOTEUR, DEFAUTS["ollama"])
GPUS = [g.strip() for g in os.environ.get("H3_PG_GPUS", "1,3,4").split(",") if g.strip()]
PORT0 = int(os.environ.get("H3_PG_PORT0", "11500"))
PAR_CARTE = int(os.environ.get("H3_PG_PARALLEL", "2"))
# Nombre d'ouvriers simultanes quand le moteur est une API distante.
CONCURRENCE = int(os.environ.get("H3_PG_CONCURRENCY", "8"))
# Une cle par moteur distant. Elle peut venir de l'environnement ou d'un
# fichier, ce qui evite de la voir passer dans une ligne de commande.
_VARS = {"anthropic": "ANTHROPIC_API_KEY", "mistral": "MISTRAL_API_KEY",
         "gemini": "GEMINI_API_KEY"}


def _cle(moteur):
    v = os.environ.get(_VARS.get(moteur, ""), "")
    if v:
        return v.strip()
    f = os.environ.get("H3_PG_KEYFILE_" + moteur.upper(), "")
    if f and os.path.isfile(f):
        with open(f, "r", encoding="utf-8") as g:
            brut = g.read()
        # Le fichier peut etre un copier-coller de la console, avec des
        # libelles autour de la cle. On repere donc la cle par sa forme
        # plutot que de prendre le fichier entier.
        for motif in (r"AQ\.[A-Za-z0-9_\-]{20,80}", r"AIza[A-Za-z0-9_\-]{30,40}"):
            trouve = re.findall(motif, brut)
            if trouve:
                return trouve[0]
        return brut.strip()
    return ""


CLES = {m: _cle(m) for m in MOTEURS}

GRAINES = os.environ.get("H3_PG_SEEDS", r"D:\tmp\h3_data\h3_gens_train.txt")
SORTIE = os.environ.get("H3_PG_OUT", r"D:\tmp\h3_data\h3_gens_h3fmt.jsonl")
LIMITE = int(os.environ.get("H3_PG_LIMIT", "0"))          # 0 = tout
TEMPERATURE = float(os.environ.get("H3_PG_TEMP", "0.8"))
# Plafond genereux : les modeles Gemini 3.x consomment d'abord ce budget
# en raisonnement -- 864 tokens sur 900 mesures -- et tronquent la reponse
# si on le serre. Le brider n'economise rien, il fait juste echouer.
MAX_TOKENS = int(os.environ.get("H3_PG_MAXTOK", "4000"))
ESSAIS = int(os.environ.get("H3_PG_RETRIES", "2"))
# Chaque graine passe par TOUS les moteurs plutot qu'un seul. Le corpus
# contient alors chaque personne vue par deux redacteurs differents, ce qui
# evite de calibrer sur les tics d'un seul modele.
TOUS_MOTEURS = os.environ.get("H3_PG_TOUS_MOTEURS", "1") not in ("", "0")
# Budget de reflexion des modeles Gemini 3.x.
REFLEXION = int(os.environ.get("H3_PG_REFLEXION", "128"))
LOG = os.environ.get("H3_PG_LOG", r"D:\tmp\h3_data\ollama")

# Liste de noms produite par h3_corpus.py a partir du classement TMDB. On
# retombe sur les soixante noms ecrits a la main si elle n'existe pas, mais elle
# est la reference : une personne absente du corpus reste cassee, donc la liste
# doit etre celle qui a servi a fabriquer les graines.
NOMS_JSON = os.environ.get("H3_PG_NOMS", r"D:	mp\h3_data\h3_cel_noms.json")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(NOMS_JSON):
    with open(NOMS_JSON, "r", encoding="utf-8") as _f:
        _d = json.load(_f)
    NOMS = sorted(_d["train"] + _d["test"], key=len, reverse=True)
else:
    from corpus import NOMS_TRAIN, NOMS_TEST  # noqa: E402
    NOMS = sorted(NOMS_TRAIN + NOMS_TEST, key=len, reverse=True)

# Les sections attendues, dans l'ordre ou MiniMax H3 les lit. Reprises telles
# quelles d'un prompt qui fonctionne, pas inventees.
SECTIONS = ["subject_definitions", "summary", "detailed_description",
            "overall_soundscape", "non_diegetic_music"]

SYSTEME = """You expand a one-line video description into a MiniMax H3 prompt.

Output EXACTLY these sections, in this order, each on its own line followed by \
its content:

subject_definitions:
summary:
detailed_description:
overall_soundscape:
non_diegetic_music:

Rules, all mandatory:
- Write the person's name EXACTLY as given, spelled identically, at least twice. \
Never replace it with "the actor", "the man", "she" alone, or any paraphrase.
- Keep every physical attribute given in the input, word for word where possible.
- subject_definitions describes the person and any other subject: age, hair, \
build, clothing.
- summary is two sentences: what happens, and how long the shot lasts.
- detailed_description covers lens, aperture, lighting, colour grade, grain and \
camera movement, then one or two shots written as [Shot 1] 00:00.000-00:04.000 \
with framing, action and effect.
- If anyone speaks, give the line in quotes and keep it short: speech must fill \
no more than two thirds of the shot duration, roughly 2.5 syllables per second \
of window. A crowded line comes out slurred.
- overall_soundscape lists diegetic sound only.
- non_diegetic_music is "N/A - no score." unless music is genuinely part of the \
scene.
- Be concrete: name materials, colours, fabrics, surfaces, and what the light does. "A worn leather jacket over a grey shirt" beats "casual clothing".
- Between 250 and 450 words total, aiming for 320. A thin prompt is as wrong as an overlong one.
- No markdown, no bullet points, no headings other than the five section names, 
no preamble, no commentary.
"""

GABARIT = """One-line description:
%s

The person is: %s
Write the MiniMax H3 prompt now."""


def log(m):
    print(m, flush=True)


def charge_graines():
    """Descriptions courtes de depart, dedoublonnees, avec le nom repere."""
    vus, sorties = set(), []
    with open(GRAINES, "r", encoding="utf-8") as f:
        for ligne in f:
            p = ligne.strip()
            if not p or p in vus:
                continue
            vus.add(p)
            nom = next((n for n in NOMS if n in p), None)
            if nom is None:
                continue
            sorties.append({"seed": p, "name": nom})
    return sorties if LIMITE <= 0 else sorties[:LIMITE]


def deja_faits():
    """Couples (graine, moteur) deja produits, pour reprendre sans doublon.

    La graine seule ne suffit plus : la meme description passe par chaque
    moteur, donc une reprise doit savoir lequel a deja repondu.
    """
    paires = set()
    if not os.path.isfile(SORTIE):
        return paires
    with open(SORTIE, "r", encoding="utf-8") as f:
        for ligne in f:
            try:
                d = json.loads(ligne)
                paires.add((d["seed"], d.get("engine", "")))
            except Exception:
                continue
    return paires


def demarre_serveurs():
    """Un ollama par carte, chacun sur son port. Retourne (procs, urls)."""
    os.makedirs(LOG, exist_ok=True)
    procs, urls = [], []
    for k, g in enumerate(GPUS):
        port = PORT0 + k
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = g
        env["OLLAMA_HOST"] = "127.0.0.1:%d" % port
        env["OLLAMA_NUM_PARALLEL"] = str(PAR_CARTE)
        # Sans cela le modele est decharge entre deux requetes et rechargé a
        # chaque fois, ce qui coute plus que la generation elle-meme.
        env["OLLAMA_KEEP_ALIVE"] = "30m"
        env["OLLAMA_MAX_LOADED_MODELS"] = "1"
        sortie = open(os.path.join(LOG, "ollama_gpu%s.log" % g), "w",
                      encoding="utf-8", errors="replace")
        p = subprocess.Popen([OLLAMA, "serve"], env=env,
                             stdout=sortie, stderr=sortie)
        procs.append((p, sortie))
        urls.append("http://127.0.0.1:%d" % port)
        log("  carte %s -> port %d" % (g, port))
    return procs, urls


def attend(url, secondes=120):
    """Vrai des que le serveur repond."""
    t0 = time.time()
    while time.time() - t0 < secondes:
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def _poste(url, corps, entetes, chemin_reponse):
    """Requete JSON, puis extraction du texte par une suite de cles."""
    req = urllib.request.Request(url, data=json.dumps(corps).encode("utf-8"),
                                 headers=entetes)
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    for k in chemin_reponse:
        d = d[k]
    return d


def genere(moteur, url, seed, nom, strict=False):
    """Une generation, quel que soit le moteur. Retourne le texte, ou leve."""
    consigne = SYSTEME
    if strict:
        consigne += ("\nThe previous attempt was rejected. Respect every rule, "
                     "especially the exact spelling of the name and the five "
                     "section names.\n")
    demande = GABARIT % (seed, nom)

    if moteur == "anthropic":
        return _poste(
            "https://api.anthropic.com/v1/messages",
            {"model": MODELES[moteur], "max_tokens": MAX_TOKENS,
             "temperature": TEMPERATURE, "system": consigne,
             "messages": [{"role": "user", "content": demande}]},
            {"Content-Type": "application/json",
             "x-api-key": CLES[moteur],
             "anthropic-version": "2023-06-01"},
            ["content", 0, "text"])

    if moteur == "gemini":
        return _poste(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "%s:generateContent?key=%s" % (MODELES[moteur], CLES[moteur]),
            {"system_instruction": {"parts": [{"text": consigne}]},
             "contents": [{"parts": [{"text": demande}]}],
             # Un budget de reflexion explicite et bas. A zero le modele
             # refuse la requete ; laisse libre il en consomme 1600, et la
             # consigne textuelle « ne delibere pas » ne le calme quasiment pas
             # -- 1294 contre 1616 mesures.
             "generationConfig": {"temperature": TEMPERATURE,
                                  "maxOutputTokens": MAX_TOKENS,
                                  "thinkingConfig": {
                                      "thinkingBudget": REFLEXION}}},
            {"Content-Type": "application/json"},
            ["candidates", 0, "content", "parts", 0, "text"])

    if moteur == "mistral":
        return _poste(
            "https://api.mistral.ai/v1/chat/completions",
            {"model": MODELES[moteur], "max_tokens": MAX_TOKENS,
             "temperature": TEMPERATURE,
             "messages": [{"role": "system", "content": consigne},
                          {"role": "user", "content": demande}]},
            {"Content-Type": "application/json",
             "Authorization": "Bearer %s" % CLES[moteur]},
            ["choices", 0, "message", "content"])

    return _poste(
        url + "/api/generate",
        {"model": MODELES.get(moteur, MODELE), "system": consigne,
         "prompt": demande,
         "stream": False,
         "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}},
        {"Content-Type": "application/json"},
        ["response"])


def valide(texte, nom):
    """Retourne None si le texte convient, sinon la raison du rejet.

    Le nom doit apparaitre a l'identique : c'est toute la raison d'etre de ce
    corpus, un prompt qui dit « the actress » n'apporte rien.
    """
    if nom not in texte:
        return "nom absent"
    manquantes = [s for s in SECTIONS if not re.search(r"(?mi)^\s*%s\s*:" % s, texte)]
    if manquantes:
        return "sections manquantes : %s" % ", ".join(manquantes)
    mots = len(texte.split())
    if mots < 110:
        return "trop court (%d mots)" % mots
    if mots > 600:
        return "trop long (%d mots)" % mots
    if "```" in texte or re.search(r"(?m)^\s*[-*]\s", texte):
        return "markdown ou liste a puces"
    return None


def main():
    if not os.path.isfile(OLLAMA):
        log("ollama introuvable : %s" % OLLAMA)
        return 1
    if not os.path.isfile(GRAINES):
        log("graines introuvables : %s" % GRAINES)
        return 1

    graines = charge_graines()
    faits_paires = deja_faits()
    restant = graines
    log("%d graines, %d couples (graine, moteur) deja produits"
        % (len(graines), len(faits_paires)))

    procs = []
    try:
        # Une place d'ouvrier est un couple (moteur, url). Les moteurs distants
        # n'utilisent pas l'url ; ollama ouvre une instance par carte. Avec deux
        # moteurs les places alternent, donc le corpus est ecrit moitie par l'un,
        # moitie par l'autre.
        places = []
        distants = [m for m in MOTEURS if m in ("anthropic", "mistral", "gemini")]
        locaux = [m for m in MOTEURS if m == "ollama"]

        for m in distants:
            if not CLES.get(m):
                log("cle absente pour %s, moteur ignore (%s ou H3_PG_KEYFILE_%s)"
                    % (m, _VARS.get(m, "?"), m.upper()))
                continue
            part = max(1, CONCURRENCE // max(1, len(distants)))
            places.extend((m, "") for _ in range(part))
            log("moteur %s, modele %s, %d requetes simultanees"
                % (m, MODELES[m], part))

        if locaux:
            log("demarrage des serveurs...")
            procs, urls = demarre_serveurs()
            for u in urls:
                if attend(u):
                    log("  %s pret" % u)
                    places.extend(("ollama", u) for _ in range(PAR_CARTE))
                else:
                    log("  %s ne repond pas, ignore" % u)

        if not places:
            log("aucun moteur disponible")
            return 1

        moteurs_actifs = sorted({m for m, _ in places})
        travail = {m: queue.Queue() for m in moteurs_actifs}
        n_taches = 0
        for k, g in enumerate(restant):
            cibles = (moteurs_actifs if TOUS_MOTEURS
                      else [moteurs_actifs[k % len(moteurs_actifs)]])
            for m in cibles:
                if (g["seed"], m) in faits_paires:
                    continue
                travail[m].put(g)
                n_taches += 1
        log("  %d taches pour %d graines" % (n_taches, len(restant)))

        verrou = threading.Lock()
        fichier = open(SORTIE, "a", encoding="utf-8")
        compteurs = {"ok": 0, "rejet": 0, "erreur": 0}
        t0 = time.time()

        def ouvrier(moteur, url):
            while True:
                try:
                    g = travail[moteur].get_nowait()
                except queue.Empty:
                    return
                raison = "jamais tente"
                essai = 0
                attente = 4.0
                patiences = 0
                while essai < ESSAIS:
                    try:
                        texte = genere(moteur, url, g["seed"], g["name"],
                                       strict=essai > 0)
                    except urllib.error.HTTPError as e:
                        if e.code in (429, 502, 503, 529) and patiences < 6:
                            # Debit limite ou service sature : patienter sans
                            # consommer un essai, sinon on abandonne des graines
                            # pour une raison qui n'a rien a voir avec elles.
                            # Le compteur est indispensable : sans lui, un
                            # service durablement sature fait tourner l'ouvrier
                            # sans fin et le script ne se termine jamais.
                            patiences += 1
                            time.sleep(attente)
                            attente = min(attente * 2, 60.0)
                            continue
                        raison = "http %d" % e.code
                        essai += 1
                        continue
                    except Exception as e:
                        raison = "erreur reseau : %s" % e
                        essai += 1
                        time.sleep(2.0)
                        continue
                    essai += 1
                    texte = texte.strip()
                    raison = valide(texte, g["name"])
                    if raison is None:
                        with verrou:
                            fichier.write(json.dumps(
                                {"seed": g["seed"], "name": g["name"],
                                 "engine": moteur, "prompt": texte},
                                ensure_ascii=False) + "\n")
                            fichier.flush()
                            compteurs["ok"] += 1
                            n = compteurs["ok"] + compteurs["rejet"] + compteurs["erreur"]
                            if n % max(1, min(20, n_taches // 4)) == 0:
                                dt = time.time() - t0
                                reste = (n_taches - n) * dt / max(1, n)
                                log("  %d/%d  %.1f s/prompt  %d rejets  reste %.0f min"
                                    % (n, n_taches, dt / n, compteurs["rejet"],
                                       reste / 60))
                        break
                if raison is not None:
                    with verrou:
                        compteurs["rejet" if raison else "erreur"] += 1
                        if compteurs["rejet"] <= 5:
                            log("  rejete (%s) : %s" % (raison, g["seed"][:60]))

        fils = []
        for moteur, u in places:
            t = threading.Thread(target=ouvrier, args=(moteur, u), daemon=True)
            t.start()
            fils.append(t)
        for t in fils:
            t.join()

        fichier.close()
        dt = time.time() - t0
        log("")
        log("%d produits, %d rejetes, %d erreurs, en %.0f min"
            % (compteurs["ok"], compteurs["rejet"], compteurs["erreur"], dt / 60))
        log("ecrit : %s" % SORTIE)
    finally:
        log("arret des serveurs")
        for p, f in procs:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                f.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
