#!/usr/bin/env python
"""Sonde comparative des representations Qwen3-VL-32B (conditioning MiniMax H3) vs Qwen3-VL-4B.

Objectif : mesurer, SANS rien entrainer, a quel point les etats caches du 4B portent
la meme information que la couche 50 du 32B utilisee par le DiT H3.

Le 32B fournit la cible Y de shape [seq, 5120] (couche 50, brute, sans final norm --
cf. comfy/text_encoders/minimax.py). Le 4B fournit X de shape [n_layers+1, seq, 2560]
(toutes les couches en un seul forward via layer="all").

Trois mesures par couche du 4B :
  - CKA lineaire      : similarite de structure relationnelle, invariante a toute
                        transformation lineaire inversible. Ne necessite aucun
                        apprentissage. C'est la mesure de reference ici.
  - R2 ridge (holdout): part de variance de Y expliquee par une projection lineaire
                        de X, apprise sur une moitie des tokens et evaluee sur l'autre.
                        Indicatif seulement : sur un seul prompt le nombre de tokens
                        est trop faible devant les 2560 inconnues.
  - cosine moyen      : cosinus moyen entre Y predit et Y reel, sur le holdout.

Les tokens sont produits UNE SEULE FOIS par le tokenizer MiniMax puis injectes tels
quels dans les deux modeles : meme vocabulaire Qwen3-VL (151936), donc alignement
position par position garanti, et aucun template de chat parasite.

Les modeles sont charges sequentiellement (jamais les deux en VRAM simultanement).
"""

import gc
import json
import os

# Defaults derived from this file's location: calibration/ lives inside the
# custom node, itself inside ComfyUI/custom_nodes/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMFY = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
import sys

# comfy parse sys.argv a l'import : on le neutralise avant.
sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", _COMFY)
TE_DIR = os.environ.get("H3_TE_DIR", os.path.join(_COMFY, "models", "text_encoders"))
OUT_DIR = os.environ.get("H3_OUT_DIR", os.path.join(_HERE, "out"))

TE_32B = os.path.join(TE_DIR, "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors")
TE_4B = os.path.join(TE_DIR, "qwen3vl_4b_bf16.safetensors")

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402

# Prompts de sonde : longueurs et registres volontairement contrastes.
PROMPTS = [
    "a red ball falls into the ground",
    "wide shot, an ostrich roaming the black volcanic plains of Iceland at dawn, "
    "low fog drifting over the ground, handheld camera slowly pushing in, "
    "muted colors, distant wind and gravel crunching underfoot",
    "A 15-second cinematic sequence in 8K photorealistic quality. The scene is set on an "
    "endless salt flat mirror reflecting the sky, with oppressive dark clouds above and a "
    "minimalist cool-toned palette. [00:00-00:05] Shot 1: extremely low-angle upward shot, "
    "telephoto lens zooming in. A lone figure walks across the water, footsteps echoing. "
    "[00:05-00:10] Shot 2: extreme close-up on the face, then rapid zoom out. The figure "
    "stops, stares coldly at the camera, and snaps their fingers. [00:10-00:15] Shot 3: "
    "high-angle aerial shot rotating and descending rapidly as the mirror surface dissolves "
    "into a vast black vortex, swallowing the frame. Ambient score swells then cuts to silence.",
    "the man does NOT turn around; he stays perfectly still while the door behind him opens "
    "slowly, and only after the dog has already left the room does the light finally change "
    "from warm amber to cold blue",
]


def log(msg):
    """Affiche un message d'avancement immediatement."""
    print(msg, flush=True)


def describe(t, label):
    """Affiche les statistiques descriptives brutes d'un tenseur [seq, hidden]."""
    f = t.flatten().float()
    norms = t.norm(dim=-1)
    log("    %-22s shape=%-16s mean=%+.4f std=%.4f min=%+.3f max=%+.3f "
        "|v|moy=%.3f |v|std=%.3f"
        % (label, str(tuple(t.shape)), f.mean(), f.std(), f.min(), f.max(),
           norms.mean(), norms.std()))


def load_te(path, clip_type, label):
    """Charge un encodeur texte ComfyUI et le place en VRAM.

    Args:
        path (str): chemin du safetensors.
        clip_type (comfy.sd.CLIPType): type d'encodeur.
        label (str): nom lisible pour les logs.

    Returns:
        comfy.sd.CLIP: l'encodeur charge.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Encodeur introuvable : %s" % path)
    log("[%s] chargement de %s" % (label, os.path.basename(path)))
    clip = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None, clip_type=clip_type)
    mm.load_models_gpu([clip.patcher])
    log("[%s] charge sur %s" % (label, mm.get_torch_device()))
    return clip


def unload(clip, label):
    """Libere completement un encodeur de la VRAM."""
    del clip
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    log("[%s] decharge" % label)


def submodel(clip, name):
    """Retourne le SDClipModel interne, en contournant les surcharges du TEModel.

    Krea2TEModel.encode_token_weights applique un strip de template et un reshape
    12-couches dont on ne veut pas : on adresse directement le sous-modele.
    """
    return getattr(clip.cond_stage_model, name)


def token_ids(clip32, text):
    """Tokenise avec le tokenizer MiniMax et retourne la liste d'ids entiers.

    Le tokenizer H3 n'applique aucun chat template et pose add_special_tokens=False,
    donc les ids correspondent exactement au texte.
    """
    tok = clip32.tokenize(text)
    return [t[0] for t in tok["qwen3vl_32b"][0]]


def encode_target(clip32, ids):
    """Encode avec le 32B et retourne la cible [seq, 5120] en float32 CPU."""
    pairs = [[(i, 1.0) for i in ids]]
    with torch.no_grad():
        out = submodel(clip32, "qwen3vl_32b").encode_token_weights(pairs)
    return out[0][0].float().cpu()


def encode_all_layers(clip4, ids):
    """Encode avec le 4B et retourne toutes les couches [n+1, seq, 2560] float32 CPU.

    layer="all" collecte l'etat cache a l'entree de chaque bloc puis la sortie finale :
    l'index k vaut donc l'entree du bloc k (= sortie du bloc k-1), et le dernier index
    est la sortie apres norm finale. layer_norm_hidden_state=False garde les etats
    intermediaires bruts, comparables a la sortie non normalisee du 32B.
    """
    sm = submodel(clip4, "qwen3vl_4b")
    sm.layer = "all"
    sm.layer_idx = None
    sm.layer_norm_hidden_state = False
    pairs = [[(i, 1.0) for i in ids]]
    with torch.no_grad():
        out = sm.encode_token_weights(pairs)
    z = out[0]
    if z.dim() == 4:  # [B, n, seq, h]
        z = z[0]
    return z.float().cpu()


def outlier_report(t, label):
    """Quantifie les activations massives d'un tenseur [seq, hidden].

    Les LLM concentrent des valeurs enormes sur quelques dimensions et sur le
    premier token (attention sink). Non traitees, elles dominent toute mesure de
    similarite lineaire et la rendent ininterpretable.
    """
    per_dim = t.std(0)
    med = per_dim.median()
    n_dim_out = int((per_dim > 10 * med).sum())
    norms = t.norm(dim=-1)
    med_n = norms.median()
    tok_out = [int(i) for i in torch.nonzero(norms > 10 * med_n).flatten()[:8]]
    log("    %-22s dims>10x mediane: %4d/%d   tokens>10x: %s"
        % (label, n_dim_out, t.shape[1], tok_out if tok_out else "aucun"))


def nanmean(vals):
    """Moyenne en ignorant les nan. Retourne nan si tout est nan."""
    good = [v for v in vals if v == v]
    return sum(good) / len(good) if good else float("nan")


def standardize(t, eps=1e-6):
    """Centre-reduit chaque dimension pour neutraliser les activations massives."""
    return (t - t.mean(0, keepdim=True)) / (t.std(0, keepdim=True) + eps)


def linear_cka(x, y, drop_first=1, robust=True):
    """CKA lineaire entre deux representations de dimensions differentes.

    Mesure a quel point les deux espaces induisent la meme geometrie relationnelle
    entre tokens. Invariante a toute transformation lineaire inversible et a
    l'echelle, donc elle ne penalise pas le simple fait que les dimensions different.

    Args:
        x (torch.Tensor): [n, d1]
        y (torch.Tensor): [n, d2]
        drop_first (int): nombre de tokens de tete a exclure (attention sink).
        robust (bool): centre-reduit chaque dimension avant calcul. Sans cela, les
            activations massives saturent la mesure a ~1.0 quel que soit le contenu.

    Returns:
        float: score dans [0, 1]. 1 = structures identiques.
    """
    if drop_first and x.shape[0] > drop_first + 2:
        x, y = x[drop_first:], y[drop_first:]
    if robust:
        x, y = standardize(x), standardize(y)
    x = (x - x.mean(0, keepdim=True)).double()
    y = (y - y.mean(0, keepdim=True)).double()
    xty = x.T @ y
    num = (xty * xty).sum()
    den = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    if den <= 0:
        return float("nan")
    return float(num / den)


def ridge_holdout(x, y, lam=1e2, drop_first=1):
    """Regression ridge de y sur x, apprise sur les tokens pairs, evaluee sur les impairs.

    Source et cible sont centrees-reduites par dimension : sans cela le R2 ne
    refleterait que la reconstruction des quelques dimensions a activation massive.

    Args:
        x (torch.Tensor): [n, d1] source.
        y (torch.Tensor): [n, d2] cible.
        lam (float): coefficient de regularisation.
        drop_first (int): nombre de tokens de tete a exclure (attention sink).

    Returns:
        tuple[float, float]: (R2 holdout, cosinus moyen holdout).
    """
    if drop_first and x.shape[0] > drop_first + 2:
        x, y = x[drop_first:], y[drop_first:]
    x, y = standardize(x), standardize(y)
    n = x.shape[0]
    if n < 8:
        return float("nan"), float("nan")
    tr = torch.arange(0, n, 2)
    te = torch.arange(1, n, 2)
    xtr, ytr = x[tr].double(), y[tr].double()
    xte, yte = x[te].double(), y[te].double()

    mx, my = xtr.mean(0, keepdim=True), ytr.mean(0, keepdim=True)
    xc, yc = xtr - mx, ytr - my

    d = xc.shape[1]
    a = xc.T @ xc + lam * torch.eye(d, dtype=torch.float64)
    w = torch.linalg.solve(a, xc.T @ yc)

    pred = (xte - mx) @ w + my
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot)

    cos = torch.nn.functional.cosine_similarity(pred, yte, dim=1).mean()
    return r2, float(cos)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log("=" * 78)
    log("PHASE 1 : encodage avec le 32B (cible)")
    log("=" * 78)
    clip32 = load_te(TE_32B, comfy.sd.CLIPType.MINIMAX, "32B")

    ids_all = []
    targets = []
    for i, p in enumerate(PROMPTS):
        ids = token_ids(clip32, p)
        y = encode_target(clip32, ids)
        ids_all.append(ids)
        targets.append(y)
        log("  prompt %d : %d tokens" % (i, len(ids)))
        describe(y, "32B couche 50")
        outlier_report(y, "32B couche 50")
        if y.shape[0] != len(ids):
            log("  ATTENTION : seq (%d) != nb tokens (%d) -- alignement rompu"
                % (y.shape[0], len(ids)))

    unload(clip32, "32B")

    log("")
    log("=" * 78)
    log("PHASE 2 : encodage avec le 4B (toutes les couches)")
    log("=" * 78)
    clip4 = load_te(TE_4B, comfy.sd.CLIPType.KREA2, "4B")

    sources = []
    for i, ids in enumerate(ids_all):
        x = encode_all_layers(clip4, ids)
        sources.append(x)
        log("  prompt %d : %d tokens -> %d taps de %s"
            % (i, len(ids), x.shape[0], str(tuple(x.shape[1:]))))
        for k in (0, x.shape[0] // 4, x.shape[0] // 2, 3 * x.shape[0] // 4, x.shape[0] - 1):
            describe(x[k], "4B tap %d" % k)
            outlier_report(x[k], "4B tap %d" % k)
        if x.shape[1] != targets[i].shape[0]:
            log("  ATTENTION : seq 4B (%d) != seq 32B (%d) -- alignement rompu"
                % (x.shape[1], targets[i].shape[0]))

    unload(clip4, "4B")

    log("")
    log("=" * 78)
    log("PHASE 3 : comparaison par couche")
    log("=" * 78)

    n_layers = sources[0].shape[0]
    results = []

    log("")
    log("  CKA brut = sature par les activations massives (temoin, a ignorer).")
    log("  CKA rob. = token sink retire + dimensions centrees-reduites. C'est celui qui compte.")
    log("")
    log("  couche | CKA brut | CKA rob. |  R2 ho. | cos ho. |  (moyenne sur %d prompts)"
        % len(PROMPTS))
    log("  -------+----------+----------+---------+---------")
    for k in range(n_layers):
        raws, ckas, r2s, coss = [], [], [], []
        for i in range(len(PROMPTS)):
            x = sources[i][k]
            y = targets[i]
            n = min(x.shape[0], y.shape[0])
            x, y = x[:n], y[:n]
            raws.append(linear_cka(x, y, drop_first=0, robust=False))
            ckas.append(linear_cka(x, y))
            r2, cos = ridge_holdout(x, y)
            r2s.append(r2)
            coss.append(cos)
        m_raw = nanmean(raws)
        m_cka = nanmean(ckas)
        m_r2 = nanmean(r2s)
        m_cos = nanmean(coss)
        results.append({"layer": k, "cka_raw": m_raw, "cka": m_cka,
                        "r2_holdout": m_r2, "cos_holdout": m_cos})
        log("  %6d | %8.4f | %8.4f | %7.4f | %7.4f" % (k, m_raw, m_cka, m_r2, m_cos))

    best = max(results, key=lambda r: r["cka"] if r["cka"] == r["cka"] else -1)
    log("")
    log("  Meilleure couche par CKA robuste : %d  (CKA %.4f, cos holdout %.4f)"
        % (best["layer"], best["cka"], best["cos_holdout"]))

    log("")
    log("  Reperes de lecture du CKA :")
    log("    > 0.70  structure tres proche, une projection lineaire a de bonnes chances")
    log("    0.50-0.70  parent mais incomplet, viser un MLP plutot qu'une matrice")
    log("    < 0.50  le 4B n'encode pas la meme chose, la piste lineaire est morte")
    log("")
    log("  Le R2 et le cosinus holdout sont INDICATIFS : avec %d tokens par prompt"
        % min(len(i) for i in ids_all))
    log("  pour 2560 inconnues, la ridge est fortement sous-determinee. Seul le run")
    log("  complet sur quelques milliers de prompts les rendra concluants.")

    ref = {
        "prompts": PROMPTS,
        "n_tokens": [len(i) for i in ids_all],
        "n_layer_taps": n_layers,
        "results": results,
        "best_layer_by_cka": best,
    }
    out_json = os.path.join(OUT_DIR, "probe_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)
    torch.save({"targets": targets, "sources": sources, "ids": ids_all},
               os.path.join(OUT_DIR, "probe_tensors.pt"))
    log("")
    log("  Resultats  : %s" % out_json)
    log("  Tenseurs   : %s" % os.path.join(OUT_DIR, "probe_tensors.pt"))


if __name__ == "__main__":
    main()
