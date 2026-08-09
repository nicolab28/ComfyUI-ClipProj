#!/usr/bin/env python
"""La projection se degrade-t-elle sur les prompts courts ?

Le corpus de calibration ecarte tout prompt de moins de 15 mots (MIN_WORDS dans
h3_run.py). La matrice et les statistiques de normalisation n'ont donc jamais vu
de sequence courte, alors que c'est le cas d'usage le plus frequent a la main.

Ce script mesure la fidelite reelle en fonction de la longueur : pour chaque
prompt, il encode avec le 32B (la verite), avec le petit modele, applique la
projection exactement comme le custom node -- substitution du sink comprise --
et compare token par token.

Le token 0 est exclu de la moyenne : il est remplace par sa valeur mesuree, donc
exact par construction, et l'inclure gonflerait artificiellement le resultat sur
les prompts courts, ou il pese le plus.
"""

import gc
import os
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
TE_DIR = os.environ.get("H3_TE_DIR", r"D:\ComfyUI-Launcher\_models\text_encoders")
TE_32B = os.path.join(TE_DIR, os.environ.get(
    "H3_TE_32B", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"))
TE_S = os.path.join(TE_DIR, os.environ.get(
    "H3_TE", r"Qwen3vl_4b\qwen3vl_4b_int8_convrot.safetensors"))
STUDENT_TYPE = os.environ.get("H3_STUDENT_TYPE", "krea2")
PROJ = os.environ.get("H3_PROJ", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI\models"
                                 r"\clip_projections\h3_qwen3vl_4b_int8convrot_tap24.pt")
PROMPT_FILE = os.environ.get("H3_PROMPTS", r"D:\tmp\h3_data\h3_prompts.txt")
N_BASE = int(os.environ.get("H3_LEN_BASE", "24"))
LONGUEURS = [int(x) for x in os.environ.get(
    "H3_LEN_WORDS", "2,3,5,8,12,20,40,80").split(",")]

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402

# Les prompts reels qui ont motive la mesure : une meme demande de plus en plus
# depouillee. Ils sont mesures a part, tels quels.
REELS = [
    "portrait looking the camera, Lionel Messi, Argentine professional footballer who plays as a forward",
    "portrait looking the camera, Lionel Messi, Argentine professional footballer",
    "portrait looking the camera, Lionel Messi, professional footballer",
    "portrait looking the camera, Lionel Messi",
]


def log(m):
    print(m, flush=True)


def submodel(clip, name):
    return getattr(clip.cond_stage_model, name)


def corpus_tronque():
    """Construit les prompts de test : troncatures a longueur croissante.

    Returns:
        list[tuple[str, str]]: (etiquette, texte).
    """
    bases = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if len(p.split()) >= max(LONGUEURS):
                bases.append(p)
            if len(bases) >= N_BASE:
                break
    out = []
    for n in LONGUEURS:
        for p in bases:
            out.append(("%d mots" % n, " ".join(p.split()[:n])))
    for p in REELS:
        out.append(("reel %d mots" % len(p.split()), p))
    return out


def main():
    for p in (TE_32B, TE_S, PROJ):
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1

    proj = torch.load(PROJ, map_location="cpu", weights_only=False)
    log("Projection : %s" % os.path.basename(PROJ))
    log("  tap %d | W %s | cosinus annonce %.4f | sink %s"
        % (proj["tap"], tuple(proj["W"].shape), proj.get("cos_test", float("nan")),
           "oui" if "sink_out" in proj else "non"))

    tests = corpus_tronque()
    log("  %d prompts a mesurer" % len(tests))
    log("")

    log("Chargement du 32B (cible)...")
    clip32 = comfy.sd.load_clip(ckpt_paths=[TE_32B], embedding_directory=None,
                               clip_type=comfy.sd.CLIPType.MINIMAX)
    mm.load_models_gpu([clip32.patcher])

    ids_all, cibles = [], []
    for i, (_, txt) in enumerate(tests):
        tok = clip32.tokenize(txt)
        ids = [t[0] for t in tok["qwen3vl_32b"][0]]
        with torch.no_grad():
            out = submodel(clip32, "qwen3vl_32b").encode_token_weights(
                [[(t, 1.0) for t in ids]])
        ids_all.append(ids)
        cibles.append(out[0][0].float().cpu())
        if (i + 1) % 40 == 0:
            log("  %d/%d encodes" % (i + 1, len(tests)))
    del clip32
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    log("  32B decharge")
    log("")

    log("Chargement de %s (eleve)..." % os.path.basename(TE_S))
    ctype = getattr(comfy.sd.CLIPType, STUDENT_TYPE.upper())
    clips = comfy.sd.load_clip(ckpt_paths=[TE_S], embedding_directory=None,
                               clip_type=ctype)
    mm.load_models_gpu([clips.patcher])
    name = getattr(clips.cond_stage_model, "clip", None)
    sm = submodel(clips, name)
    sm.layer, sm.layer_idx, sm.layer_norm_hidden_state = "all", None, False

    tap = int(proj["tap"])
    W = proj["W"].double()
    mi, si = proj["mean_in"].double(), proj["std_in"].double()
    mo, so = proj["mean_out"].double(), proj["std_out"].double()
    sink = proj.get("sink_out")

    par_etiquette = {}
    for i, ((etq, _), ids) in enumerate(zip(tests, ids_all)):
        with torch.no_grad():
            out = sm.encode_token_weights([[(t, 1.0) for t in ids]])
        z = out[0]
        if z.dim() == 4:
            z = z[0]
        h = z[tap].float().cpu().double()

        cond = ((h - mi) / si) @ W * so + mo
        if sink is not None and cond.shape[0] > 0:
            cond[0] = sink.double()

        cible = cibles[i].double()
        n = min(cond.shape[0], cible.shape[0])
        if n <= 1:
            continue
        cos = torch.nn.functional.cosine_similarity(cond[1:n], cible[1:n], dim=1)
        par_etiquette.setdefault(etq, []).append(float(cos.mean()))

    log("")
    log("=" * 62)
    log("Cosinus moyen par token, token sink exclu")
    log("=" * 62)
    log("  longueur      | prompts | cosinus")
    log("  --------------+---------+---------")
    for n in LONGUEURS:
        etq = "%d mots" % n
        v = par_etiquette.get(etq)
        if v:
            log("  %-13s | %7d | %7.4f" % (etq, len(v), sum(v) / len(v)))
    log("")
    for (etq, txt) in tests:
        if etq.startswith("reel"):
            v = par_etiquette.get(etq)
            if v:
                log("  %7.4f  %s" % (sum(v) / len(v), txt))
    log("")
    log("Lecture : si le cosinus s'effondre sous 15 mots, la cause est le corpus")
    log("de calibration, qui ecarte ces longueurs -- et une recalibration incluant")
    log("des prompts courts corrige le probleme.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
