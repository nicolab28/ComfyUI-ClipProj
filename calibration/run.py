#!/usr/bin/env python
"""Run complet : le 4B peut-il remplacer le 32B comme encodeur de conditionnement H3 ?

Deux mesures, sur des prompts DIFFERENTS (contrairement a la sonde, qui comparait
les tokens d'un meme prompt et etait donc gonflee par l'identite lexicale) :

  1. CKA inter-prompts : un vecteur par prompt (moyenne des tokens, sink exclu),
     puis CKA sur l'ensemble des prompts. Mesure si les deux modeles rangent les
     prompts de la meme facon dans leur espace -- c'est la question semantique.

  2. Regression ridge token-level : on apprend W (2560 -> 5120) sur les tokens du
     jeu d'entrainement et on evalue sur un jeu de test disjoint. Avec plusieurs
     dizaines de milliers de tokens pour 2560 inconnues, le systeme est enfin
     surdetermine et le resultat est concluant.

Les matrices normales X'X et X'Y sont accumulees en streaming : la memoire ne
depend pas du nombre de prompts. Les deux modeles ne sont jamais charges ensemble.
"""

import gc
import hashlib
import json
import os

# Defaults derived from this file's location: calibration/ lives inside the
# custom node, itself inside ComfyUI/custom_nodes/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMFY = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
import sys
import time

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", _COMFY)
TE_DIR = os.environ.get("H3_TE_DIR", os.path.join(_COMFY, "models", "text_encoders"))
OUT_DIR = os.environ.get("H3_OUT_DIR", os.path.join(_HERE, "out"))
PROMPT_FILE = os.environ.get("H3_PROMPTS", os.path.join(_HERE, "out", "prompts.txt"))

N_TRAIN = int(os.environ.get("H3_N_TRAIN", "300"))
N_TEST = int(os.environ.get("H3_N_TEST", "60"))
MAX_TOKENS = int(os.environ.get("H3_MAX_TOKENS", "320"))
MIN_WORDS = 15
LAMBDAS = [1e1, 1e2, 1e3, 1e4]

TE_32B = os.path.join(TE_DIR, os.environ.get(
    "H3_TE_32B", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"))
TE_4B = os.path.join(TE_DIR, os.environ.get("H3_TE_4B", "qwen3vl_4b_bf16.safetensors"))

sys.path.insert(0, COMFY_DIR)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402

DROP_FIRST = 1  # token d'attention sink


def log(msg):
    """Affiche un message d'avancement immediatement."""
    print(msg, flush=True)


def load_te(path, clip_type, label):
    """Charge un encodeur texte ComfyUI et le place en VRAM."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Encodeur introuvable : %s" % path)
    log("[%s] chargement de %s" % (label, os.path.basename(path)))
    clip = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None, clip_type=clip_type)
    mm.load_models_gpu([clip.patcher])
    log("[%s] pret sur %s" % (label, mm.get_torch_device()))
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
    """Retourne le SDClipModel interne, en contournant les surcharges du TEModel."""
    return getattr(clip.cond_stage_model, name)


def load_prompts():
    """Charge, filtre et separe les prompts en train / test.

    Returns:
        tuple[list[str], list[str]]: (train, test), disjoints.
    """
    if not os.path.isfile(PROMPT_FILE):
        raise FileNotFoundError(
            "Fichier de prompts absent : %s (lancer la preparation d'abord)" % PROMPT_FILE)
    seen = set()
    kept = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if len(p.split()) < MIN_WORDS:
                continue
            key = p[:120]
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
    need = N_TRAIN + N_TEST
    if len(kept) < need:
        raise RuntimeError("Seulement %d prompts exploitables, %d requis" % (len(kept), need))
    # Prise en quinconce : evite qu'un bloc thematique contigu ne finisse entier
    # d'un seul cote de la separation train/test.
    train, test = [], []
    for i, p in enumerate(kept[:need]):
        (test if i % 6 == 0 else train).append(p)
    return train[:N_TRAIN], test[:N_TEST]


def token_ids(clip32, text):
    """Tokenise avec le tokenizer MiniMax et retourne la liste d'ids, tronquee."""
    tok = clip32.tokenize(text)
    ids = [t[0] for t in tok["qwen3vl_32b"][0]]
    return ids[:MAX_TOKENS]


def encode_target(clip32, ids):
    """Encode avec le 32B : cible [seq, 5120] float32 CPU."""
    with torch.no_grad():
        out = submodel(clip32, "qwen3vl_32b").encode_token_weights([[(i, 1.0) for i in ids]])
    return out[0][0].float().cpu()


def setup_4b(clip4):
    """Configure le 4B pour restituer toutes les couches brutes."""
    sm = submodel(clip4, "qwen3vl_4b")
    sm.layer = "all"
    sm.layer_idx = None
    sm.layer_norm_hidden_state = False
    return sm


def encode_source(sm, ids):
    """Encode avec le 4B : [n_taps, seq, 2560] float32 CPU."""
    with torch.no_grad():
        out = sm.encode_token_weights([[(i, 1.0) for i in ids]])
    z = out[0]
    if z.dim() == 4:
        z = z[0]
    return z.float().cpu()


def linear_cka(x, y):
    """CKA lineaire sur representations centrees-reduites par dimension."""
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    y = (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True) + 1e-6)
    x = (x - x.mean(0, keepdim=True)).double()
    y = (y - y.mean(0, keepdim=True)).double()
    xty = x.T @ y
    num = (xty * xty).sum()
    den = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return float(num / den) if den > 0 else float("nan")


class Accumulator:
    """Accumule les statistiques suffisantes d'une regression, en memoire constante.

    Stocke X'X, X'Y ainsi que les sommes et sommes de carres necessaires au
    centrage-reduction, sans jamais conserver les activations elles-memes.
    """

    def __init__(self, d_in, d_out):
        self.xtx = torch.zeros(d_in, d_in, dtype=torch.float64)
        self.xty = torch.zeros(d_in, d_out, dtype=torch.float64)
        self.sx = torch.zeros(d_in, dtype=torch.float64)
        self.sx2 = torch.zeros(d_in, dtype=torch.float64)
        self.sy = torch.zeros(d_out, dtype=torch.float64)
        self.sy2 = torch.zeros(d_out, dtype=torch.float64)
        self.n = 0

    def add(self, x, y):
        """Ajoute un lot de tokens alignes.

        Args:
            x (torch.Tensor): [n, d_in]
            y (torch.Tensor): [n, d_out]
        """
        xd, yd = x.double(), y.double()
        self.xtx += xd.T @ xd
        self.xty += xd.T @ yd
        self.sx += xd.sum(0)
        self.sx2 += (xd * xd).sum(0)
        self.sy += yd.sum(0)
        self.sy2 += (yd * yd).sum(0)
        self.n += x.shape[0]

    def moments(self):
        """Retourne (mean_x, std_x, mean_y, std_y) du corpus accumule."""
        mx = self.sx / self.n
        sdx = (self.sx2 / self.n - mx * mx).clamp_min(1e-12).sqrt()
        my = self.sy / self.n
        sdy = (self.sy2 / self.n - my * my).clamp_min(1e-12).sqrt()
        return mx, sdx, my, sdy

    def solve(self, lam):
        """Resout la ridge standardisee et retourne (W, mx, sdx, my, sdy).

        La standardisation est appliquee analytiquement aux moments accumules,
        ce qui evite de repasser sur les donnees.
        """
        mx, sdx, my, sdy = self.moments()
        n = self.n
        # Covariances centrees, puis mise a l'echelle par les ecarts-types.
        cxx = (self.xtx - n * torch.outer(mx, mx)) / (torch.outer(sdx, sdx) * n)
        cxy = (self.xty - n * torch.outer(mx, my)) / (torch.outer(sdx, sdy) * n)
        d = cxx.shape[0]
        w = torch.linalg.solve(cxx + lam / n * torch.eye(d, dtype=torch.float64), cxy)
        return w, mx, sdx, my, sdy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test = load_prompts()
    log("Prompts : %d entrainement / %d test" % (len(train), len(test)))
    allp = train + test
    n_train = len(train)

    cache = os.path.join(OUT_DIR, "targets.pt")
    # Signature du corpus : permet de reutiliser l'encodage 32B (le plus couteux)
    # quand seul le modele eleve change. hashlib et non hash() : le hachage des
    # chaines Python est randomise a chaque processus, donc jamais reproductible
    # d'une execution a l'autre.
    digest = hashlib.sha1("\n".join(allp).encode("utf-8")).hexdigest()[:12]
    sig = "%s|%d|%d|%d|%s" % (os.path.basename(TE_32B), n_train, len(test),
                              MAX_TOKENS, digest)

    # Un cache produit avant ce correctif porte une signature irreproductible.
    # H3_REBLESS_CACHE=1 le revalide sans refaire les 25 minutes d'encodage 32B,
    # sous reserve qu'il contienne bien le meme nombre de prompts.
    if os.environ.get("H3_REBLESS_CACHE") == "1" and os.path.isfile(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob.get("sig") != sig and len(blob.get("targets", [])) == len(allp):
            blob["sig"] = sig
            torch.save(blob, cache)
            log("  cache revalide : %d prompts, signature %s" % (len(allp), digest))
        elif len(blob.get("targets", [])) != len(allp):
            log("  cache ignore : %d prompts stockes contre %d attendus"
                % (len(blob.get("targets", [])), len(allp)))
        del blob
        gc.collect()

    log("")
    log("=" * 78)
    log("PHASE 1 : encodage 32B (cible)")
    log("=" * 78)
    t_load32 = t_enc32 = 0.0
    cached = None
    if os.path.isfile(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob.get("sig") == sig:
            cached = blob
    if cached is not None:
        ids_all, targets = cached["ids"], cached["targets"]
        ntok = sum(t.shape[0] for t in targets)
        log("  cible reutilisee depuis %s (%d prompts, %d tokens)"
            % (cache, len(targets), ntok))
    else:
        t_load32 = time.time()
        clip32 = load_te(TE_32B, comfy.sd.CLIPType.MINIMAX, "32B")
        t_load32 = time.time() - t_load32
        t_enc32 = time.time()
        ids_all, targets, ntok = [], [], 0
        for i, p in enumerate(allp):
            ids = token_ids(clip32, p)
            y = encode_target(clip32, ids)
            ids_all.append(ids)
            targets.append(y.half())
            ntok += y.shape[0]
            if (i + 1) % 100 == 0 or i == 0:
                log("  %d/%d prompts, %d tokens cumules" % (i + 1, len(allp), ntok))
        t_enc32 = time.time() - t_enc32
        log("  chargement %.1f s | encodage %.1f s (%.0f tokens/s, %.2f s/prompt)"
            % (t_load32, t_enc32, ntok / max(t_enc32, 1e-6), t_enc32 / len(allp)))
        unload(clip32, "32B")
        torch.save({"ids": ids_all, "targets": targets, "sig": sig}, cache)
        log("  cible sauvegardee : %s (%d tokens)" % (cache, ntok))

    log("")
    log("=" * 78)
    log("PHASE 2 : encodage 4B + accumulation")
    log("=" * 78)
    t_load4 = time.time()
    clip4 = load_te(TE_4B, comfy.sd.CLIPType.KREA2, "4B")
    t_load4 = time.time() - t_load4
    t_enc4 = time.time()
    sm = setup_4b(clip4)

    probe = encode_source(sm, ids_all[0])
    n_taps, d_in = probe.shape[0], probe.shape[2]
    d_out = targets[0].shape[1]
    taps = list(range(0, n_taps, 3))
    if n_taps - 1 not in taps:
        taps.append(n_taps - 1)
    log("  %d taps disponibles, %d retenus : %s" % (n_taps, len(taps), taps))

    accs = {k: Accumulator(d_in, d_out) for k in taps}
    test_x = {k: [] for k in taps}
    test_y = []
    pool_src = {k: [] for k in taps}
    pool_tgt = []

    for i, ids in enumerate(ids_all):
        x = encode_source(sm, ids)
        y = targets[i].float()
        n = min(x.shape[1], y.shape[0])
        sl = slice(DROP_FIRST, n)
        if n - DROP_FIRST < 4:
            continue
        yv = y[sl]
        if i < n_train:
            for k in taps:
                accs[k].add(x[k][sl], yv)
        else:
            test_y.append(yv.half())
            for k in taps:
                test_x[k].append(x[k][sl].half())
        # Vecteur-prompt pour le CKA inter-prompts.
        pool_tgt.append(yv.mean(0))
        for k in taps:
            pool_src[k].append(x[k][sl].mean(0))
        if (i + 1) % 25 == 0 or i == 0:
            log("  %d/%d prompts encodes" % (i + 1, len(ids_all)))

    t_enc4 = time.time() - t_enc4
    log("  chargement %.1f s | encodage+accumulation %.1f s (%.2f s/prompt)"
        % (t_load4, t_enc4, t_enc4 / len(ids_all)))
    unload(clip4, "4B")

    log("")
    log("=" * 78)
    log("PHASE 3 : resultats")
    log("=" * 78)
    log("  tokens d'entrainement accumules : %d  (pour %d inconnues par sortie)"
        % (accs[taps[0]].n, d_in))

    pool_tgt_m = torch.stack(pool_tgt)
    yte = torch.cat([t.float() for t in test_y])

    results = []
    log("")
    log("   tap | CKA inter-p |  meilleur lambda |   R2 test | cosinus test")
    log("  -----+-------------+------------------+-----------+-------------")
    for k in taps:
        cka = linear_cka(torch.stack(pool_src[k]), pool_tgt_m)

        xte = torch.cat([t.float() for t in test_x[k]])
        best = None
        for lam in LAMBDAS:
            w, mx, sdx, my, sdy = accs[k].solve(lam)
            pred = ((xte.double() - mx) / sdx) @ w
            gold = (yte.double() - my) / sdy
            ss_res = ((gold - pred) ** 2).sum()
            ss_tot = (gold ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot)
            cos = float(torch.nn.functional.cosine_similarity(pred, gold, dim=1).mean())
            if best is None or cos > best[2]:
                best = (lam, r2, cos)
        results.append({"tap": k, "cka_inter_prompt": cka,
                        "lambda": best[0], "r2_test": best[1], "cos_test": best[2]})
        log("  %4d | %11.4f | %16.0f | %9.4f | %12.4f"
            % (k, cka, best[0], best[1], best[2]))

    top = max(results, key=lambda r: r["cos_test"])
    log("")
    log("  Meilleur tap : %d  (cosinus test %.4f, R2 %.4f, CKA inter-prompts %.4f)"
        % (top["tap"], top["cos_test"], top["r2_test"], top["cka_inter_prompt"]))
    log("")
    log("  Lecture du cosinus test (jeu disjoint, biais lexical elimine) :")
    log("    > 0.95  une simple matrice suffit -- passer au custom node")
    log("    0.85-0.95  tenter un MLP 2 couches avant de conclure")
    log("    < 0.80  le 4B ne porte pas l'information, changer de modele eleve")

    # Sauvegarde de la projection du meilleur tap, prete a etre rechargee par un
    # custom node : cond_32B ~= ((h_4B - mx) / sdx) @ W * sdy + my
    w, mx, sdx, my, sdy = accs[top["tap"]].solve(top["lambda"])
    src = os.path.splitext(os.path.basename(TE_4B))[0]
    proj_path = os.path.join(OUT_DIR, "h3_proj_%s_tap%d.pt" % (src, top["tap"]))
    torch.save({"W": w.float(), "mean_in": mx.float(), "std_in": sdx.float(),
                "mean_out": my.float(), "std_out": sdy.float(),
                "tap": top["tap"], "lambda": top["lambda"],
                "d_in": d_in, "d_out": d_out,
                "cos_test": top["cos_test"], "r2_test": top["r2_test"],
                # Modele sur lequel W est calibree : le custom node avertit si
                # un autre encodeur est charge.
                "source_model": os.path.basename(TE_4B),
                "target_model": os.path.basename(TE_32B),
                "n_train_prompts": n_train,
                "n_train_tokens": accs[taps[0]].n}, proj_path)

    with open(os.path.join(OUT_DIR, "run_results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_train": n_train, "n_test": len(test),
                   "n_tokens_train": accs[taps[0]].n, "results": results,
                   "best": top,
                   "timings": {"load_32b_s": t_load32, "encode_32b_s": t_enc32,
                               "load_4b_s": t_load4, "encode_4b_s": t_enc4}}, f, indent=2)
    log("")
    log("  Temps  : 32B %.0f s chargement + %.0f s encodage | 4B %.0f s + %.0f s"
        % (t_load32, t_enc32, t_load4, t_enc4))
    log("  Projection : %s" % proj_path)
    log("  Resultats  : %s" % os.path.join(OUT_DIR, "run_results.json"))


if __name__ == "__main__":
    main()
