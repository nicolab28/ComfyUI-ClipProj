#!/usr/bin/env python
"""La projection reconstruit-elle moins bien le francais que l'anglais ?

Le corpus de calibration est integralement anglais : le script d'extraction ne
retient que `.i18n.en.p` du dataset Seedance. La matrice n'a donc jamais vu un
token francais, alors que la prononciation exige une information bien plus fine
que le sens visuel -- une erreur qui laisse passer une description de scene sans
dommage peut suffire a brouiller des phonemes.

Methode : des paires de prompts identiques au mot pres, sauf la replique citee,
francaise dans un cas et anglaise dans l'autre. Comparer les deux cosinus evite
tout reperage de position dans la sequence, donc toute erreur de decoupage.

Une troisieme serie, en francais integral, borne le cas extreme.
"""

import gc
import os
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
TE_DIR = os.environ.get("H3_TE_DIR", r"D:\ComfyUI-Launcher\_models\text_encoders")
TE_32B = os.path.join(TE_DIR, os.environ.get(
    "H3_TE_32B", r"Qwen3vl_32b\qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"))
TE_S = os.path.join(TE_DIR, os.environ.get(
    "H3_TE", r"Qwen3vl_4b\qwen3vl_4b_int8_convrot.safetensors"))
STUDENT_TYPE = os.environ.get("H3_STUDENT_TYPE", "krea2")
PROJ = os.environ.get("H3_PROJ", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI\models"
                                 r"\clip_projections\h3_qwen3vl_4b_int8convrot_tap24.pt")

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402

SOCLE = ("A woman sits on a wooden public bench in a park at midday, medium close-up, "
         "the camera is locked off at eye level, soft overcast daylight, she looks "
         "directly into the lens and says %s, her lips moving naturally in sync with "
         "the words, a light breeze moves her hair, nothing else in the frame changes.")

REPLIQUES = [
    ('in French, "Nicolas, comment vas-tu ?"',
     'in English, "Nicolas, how are you?"'),
    ('in French, "Je suis vraiment heureuse de te revoir aujourd\'hui."',
     'in English, "I am really happy to see you again today."'),
    ('in French, "Il pleuvait beaucoup hier soir dans la rue."',
     'in English, "It was raining a lot last night in the street."'),
    ('in French, "Peux-tu me passer le journal, s\'il te plait ?"',
     'in English, "Can you hand me the newspaper, please?"'),
]

# Cas extreme : le prompt entier en francais, pas seulement la replique.
INTEGRAL = [
    "Une femme est assise sur un banc public en bois dans un parc a midi, plan "
    "rapproche, la camera est fixe a hauteur des yeux, lumiere douce de ciel couvert, "
    "elle regarde droit dans l'objectif et parle, une brise legere agite ses cheveux.",
    "Un homme age repare son filet de peche sur un quai au lever du soleil, plan "
    "large, la lumiere rasante traverse la brume, des mouettes passent au second plan.",
    "Une voiture rouge traverse lentement une rue pavee mouillee la nuit, les "
    "enseignes au neon se refletent dans les flaques, la camera suit en travelling.",
    "Un chat noir saute depuis une etagere en bois vers le sol carrele d'une cuisine "
    "ensoleillee, gros plan, mouvement fluide, poussiere visible dans la lumiere.",
]


def log(m):
    print(m, flush=True)


def submodel(clip, name):
    return getattr(clip.cond_stage_model, name)


def main():
    for p in (TE_32B, TE_S, PROJ):
        if not os.path.isfile(p):
            log("Introuvable : %s" % p)
            return 1

    proj = torch.load(PROJ, map_location="cpu", weights_only=False)
    log("Projection : %s  (cosinus annonce %.4f)"
        % (os.path.basename(PROJ), proj.get("cos_test", float("nan"))))
    log("Eleve      : %s" % os.path.basename(TE_S))
    log("")

    textes, etiquettes = [], []
    for fr, en in REPLIQUES:
        textes.append(SOCLE % fr)
        etiquettes.append("replique FR")
        textes.append(SOCLE % en)
        etiquettes.append("replique EN")
    for t in INTEGRAL:
        textes.append(t)
        etiquettes.append("prompt FR integral")

    log("Chargement du 32B (cible)...")
    clip32 = comfy.sd.load_clip(ckpt_paths=[TE_32B], embedding_directory=None,
                                clip_type=comfy.sd.CLIPType.MINIMAX)
    mm.load_models_gpu([clip32.patcher], force_full_load=True)

    ids_all, cibles = [], []
    for t in textes:
        tok = clip32.tokenize(t)
        ids = [x[0] for x in tok["qwen3vl_32b"][0]]
        with torch.no_grad():
            out = submodel(clip32, "qwen3vl_32b").encode_token_weights(
                [[(i, 1.0) for i in ids]])
        ids_all.append(ids)
        cibles.append(out[0][0].float().cpu())
    del clip32
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    log("  32B decharge")
    log("")

    log("Chargement de l'eleve...")
    clips = comfy.sd.load_clip(ckpt_paths=[TE_S], embedding_directory=None,
                               clip_type=getattr(comfy.sd.CLIPType, STUDENT_TYPE.upper()))
    mm.load_models_gpu([clips.patcher], force_full_load=True)
    name = getattr(clips.cond_stage_model, "clip", None)
    sm = submodel(clips, name)
    sm.layer, sm.layer_idx, sm.layer_norm_hidden_state = "all", None, False

    tap = int(proj["tap"])
    W = proj["W"].double()
    mi, si = proj["mean_in"].double(), proj["std_in"].double()
    mo, so = proj["mean_out"].double(), proj["std_out"].double()
    sink = proj.get("sink_out")

    par_etiquette = {}
    detail = []
    for i, (etq, ids) in enumerate(zip(etiquettes, ids_all)):
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
        cos = torch.nn.functional.cosine_similarity(cond[1:n], cible[1:n], dim=1)
        v = float(cos.mean())
        par_etiquette.setdefault(etq, []).append(v)
        detail.append((etq, v, n, textes[i]))

    log("")
    log("=" * 66)
    log("Cosinus moyen par token, sink exclu")
    log("=" * 66)
    for etq in ("replique EN", "replique FR", "prompt FR integral"):
        v = par_etiquette.get(etq)
        if v:
            log("  %-20s | %d prompts | %.4f" % (etq, len(v), sum(v) / len(v)))

    en = par_etiquette.get("replique EN")
    fr = par_etiquette.get("replique FR")
    if en and fr:
        d = sum(fr) / len(fr) - sum(en) / len(en)
        log("")
        log("  ecart FR - EN sur socle identique : %+.4f" % d)
        log("")
        if d < -0.02:
            log("  Le francais est nettement moins bien reconstruit. Le corpus")
            log("  monolingue est bien en cause, et un recalibrage multilingue")
            log("  a un sens.")
        else:
            log("  Pas d'ecart significatif : le corpus n'explique pas la degradation")
            log("  du francais entendue en generation. Chercher ailleurs -- la branche")
            log("  audio du DiT est le suspect suivant.")

    log("")
    log("Detail :")
    for etq, v, n, t in detail:
        log("  %.4f  %-20s %3d tokens  %s" % (v, etq, n, t[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
