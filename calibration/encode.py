#!/usr/bin/env python
"""Encode un corpus de prompts et ecrit un fragment sur disque.

Sert a preparer l'entrainement du MLP : il faut cette fois conserver les etats
caches, alors que la calibration par ridge les accumulait a la volee sans jamais
les stocker.

Le travail se decoupe entre plusieurs processus par H3_SHARD et H3_SHARDS, un
par carte. Un decoupage statique suffit ici : les deux 3090 sont identiques et
les prompts sont melanges, donc les deux moities durent le meme temps. Chaque
processus ecrit son propre fichier, l'assemblage se fait ensuite par index.

Les tenseurs sont stockes en bfloat16 : la calibration passe de toute facon en
float32 pour l'accumulation, et le double de precision doublerait le disque pour
rien.
"""

import gc
import json
import os
import sys
import time

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
TE_DIR = os.environ.get("H3_TE_DIR", r"D:\ComfyUI-Launcher\_models\text_encoders")
MODEL = os.path.join(TE_DIR, os.environ["H3_MODEL"])
CLIP_TYPE = os.environ.get("H3_CLIP_TYPE", "minimax")
PROMPT_FILE = os.environ.get("H3_PROMPTS", r"D:\tmp\h3_data\h3_prompts.txt")
OUT = os.environ["H3_OUT"]

N_TOTAL = int(os.environ.get("H3_N_TOTAL", "0"))   # 0 = tout le corpus exploitable
MAX_TOKENS = int(os.environ.get("H3_MAX_TOKENS", "320"))
MIN_WORDS = int(os.environ.get("H3_MIN_WORDS", "15"))

# Liste de taps a conserver, vide pour la sortie finale du modele cible.
TAPS = [int(x) for x in os.environ.get("H3_TAPS", "").split(",") if x.strip()]

SHARD = int(os.environ.get("H3_SHARD", "0"))
SHARDS = int(os.environ.get("H3_SHARDS", "1"))

# Encodage deja fait dont il faut reprendre les identifiants de tokens au lieu
# de retokeniser. Indispensable des que les deux modeles serviront a former des
# paires : leurs tokeniseurs ne decoupent pas le texte de la meme facon -- 225
# tokens contre 202 en moyenne ici -- et laisser chacun tokeniser de son cote
# fait glisser l'appariement au fil de la phrase. On impose donc a l'eleve la
# segmentation de la cible, comme le faisait h3_run.py.
IDS_FROM = os.environ.get("H3_IDS", "")

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402


def log(m):
    print("[shard %d/%d] %s" % (SHARD, SHARDS, m), flush=True)


def load_prompts():
    """Corpus filtre et dedoublonne, dans un ordre stable.

    Meme selection que la calibration par ridge, pour que les deux puissent
    partager le meme cache.

    Deux formats acceptes. Un prompt par ligne pour les corpus courts, et JSONL
    pour les prompts au format H3 : ceux-la contiennent des retours a la ligne
    entre leurs sections, qu'un fichier ligne a ligne ne saurait pas porter et
    qu'il serait faux d'aplatir -- les vrais prompts en ont.
    """
    seen, kept = set(), []
    jsonl = PROMPT_FILE.lower().endswith(".jsonl")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if jsonl:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)["prompt"].strip()
                except Exception:
                    continue
            else:
                p = line.strip()
            if len(p.split()) < MIN_WORDS:
                continue
            key = p[:120]
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
    return kept if N_TOTAL <= 0 else kept[:N_TOTAL]


def submodel(clip):
    """Sous-modele interne, dont le nom depend de l'architecture."""
    name = getattr(clip.cond_stage_model, "clip", None)
    if name is None:
        for cand in ("qwen3vl_32b", "qwen3vl_8b", "qwen3vl_4b"):
            if hasattr(clip.cond_stage_model, cand):
                name = cand
                break
    return getattr(clip.cond_stage_model, name), name


def main():
    if not os.path.isfile(MODEL):
        log("Modele introuvable : %s" % MODEL)
        return 1

    tous = load_prompts()
    indices = list(range(SHARD, len(tous), SHARDS))
    log("%d prompts au total, %d pour ce fragment" % (len(tous), len(indices)))

    ids_imposes = None
    if IDS_FROM:
        if not os.path.isfile(IDS_FROM):
            log("Source d'identifiants introuvable : %s" % IDS_FROM)
            return 1
        log("lecture des identifiants de %s" % os.path.basename(IDS_FROM))
        src = torch.load(IDS_FROM, map_location="cpu", weights_only=False, mmap=True)
        ids_imposes = dict(zip(src["indices"], src["ids"]))
        # Le blob porte aussi ses etats caches, une dizaine de gigaoctets dont on
        # n'a que faire ici : les liberer avant de charger le modele.
        del src
        gc.collect()
        indices = [i for i in indices if i in ids_imposes]
        log("%d prompts conserves, longueur moyenne %.1f tokens"
            % (len(indices),
               sum(len(ids_imposes[i]) for i in indices) / max(1, len(indices))))

    ctype = getattr(comfy.sd.CLIPType, CLIP_TYPE.upper())
    log("chargement de %s" % os.path.basename(MODEL))
    t0 = time.time()
    clip = comfy.sd.load_clip(ckpt_paths=[MODEL], embedding_directory=None,
                              clip_type=ctype)
    mm.load_models_gpu([clip.patcher])
    sm, name = submodel(clip)
    log("pret sur %s en %.0f s (sous-modele %s)"
        % (clip.patcher.load_device, time.time() - t0, name))

    if TAPS:
        sm.layer, sm.layer_idx, sm.layer_norm_hidden_state = "all", None, False

    ids_out, data_out = [], []
    t0 = time.time()
    for n, i in enumerate(indices):
        if ids_imposes is not None:
            ids = ids_imposes[i][:MAX_TOKENS]
        else:
            tok = clip.tokenize(tous[i])
            ids = [t[0] for t in tok[name][0]][:MAX_TOKENS]
        with torch.no_grad():
            out = sm.encode_token_weights([[(t, 1.0) for t in ids]])
        z = out[0]
        if TAPS:
            if z.dim() == 4:
                z = z[0]
            z = z[TAPS] if len(TAPS) < z.shape[0] else z
        else:
            z = z[0]
        ids_out.append(ids)
        data_out.append(z.to(torch.bfloat16).cpu())

        if (n + 1) % 200 == 0:
            dt = time.time() - t0
            reste = dt / (n + 1) * (len(indices) - n - 1)
            log("%d/%d  %.2f s/prompt  reste %.0f min"
                % (n + 1, len(indices), dt / (n + 1), reste / 60))

    del clip
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)

    blob = {"indices": indices, "ids": ids_out, "data": data_out,
            "model": os.path.basename(MODEL), "taps": TAPS,
            "max_tokens": MAX_TOKENS, "min_words": MIN_WORDS,
            "n_corpus": len(tous), "shard": SHARD, "shards": SHARDS}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(blob, OUT)
    toks = sum(len(x) for x in ids_out)
    log("ecrit %s  %d prompts, %d tokens, %.1f Go, %.0f min"
        % (OUT, len(indices), toks,
           os.path.getsize(OUT) / 1024 ** 3, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
