#!/usr/bin/env python
"""Les trois modeles tokenisent-ils exactement pareil ?

Toute la methode repose sur cette hypothese : un meme prompt doit donner les
memes identifiants aux memes positions dans le 32B et dans le petit modele, sans
quoi la correspondance position par position n'a aucun sens.

Le code de ComfyUI la rend probable -- MiniMaxH3Tokenizer instancie la meme
classe Qwen3VLSDTokenizer que le 4B et le 8B, seules l'embedding_size et la cle
changent -- mais une lecture de code n'est pas une mesure. On compare donc les
sorties sur un corpus volontairement hostile : accents, langues non latines,
ponctuation, emoji, balises, chiffres, espaces multiples.
"""

import os
import sys

sys.argv = [sys.argv[0]]

COMFY_DIR = os.environ.get("H3_COMFY_DIR", r"D:\ComfyUI-Launcher\ComfyUI_270\ComfyUI")
PROMPT_FILE = os.environ.get("H3_PROMPTS", r"D:\tmp\h3_data\h3_prompts.txt")
N_CORPUS = int(os.environ.get("H3_TOK_N", "2000"))

sys.path.insert(0, COMFY_DIR)

import comfy.text_encoders.minimax as minimax  # noqa: E402
import comfy.text_encoders.qwen3vl as qwen3vl  # noqa: E402

# Cas choisis pour casser un tokenizer s'il differe quelque part.
PIEGES = [
    "a red ball falls onto the ground",
    "Nicolas, comment vas-tu ? Ça va très bien, merci beaucoup.",
    "Où est passé l'œuf de Noël ? Straße, mañana, ångström.",
    "日本語のテキスト、中文文本，한국어 텍스트",
    "Здравствуйте, как дела? Γειά σου κόσμε. مرحبا بالعالم",
    "emoji: 🎬 🎥 📽️ 🍿 👩‍🚀 👨‍👩‍👧‍👦 🏳️‍🌈",
    "<d> <video> <image> <|im_start|> <|im_end|> <|vision_start|>",
    "[Shot 1] 00:00.000-00:03.000. The video cuts directly to a close-up.",
    "1234567890 3.14159 1e-9 0x1F 100% $50 €30 £20",
    "   multiple    spaces\tand\ttabs   ",
    "line one\nline two\n\nline four",
    "MiXeD CaSe AnD Punctuation!!! ??? ... --- ___ ***",
    "a" * 300,
    "",
]


def charger_corpus():
    """Prompts reels du corpus de calibration, si disponibles."""
    if not os.path.isfile(PROMPT_FILE):
        return []
    out = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                out.append(p)
            if len(out) >= N_CORPUS:
                break
    return out


def main():
    tok32 = minimax.MiniMaxH3Tokenizer()
    tok4 = qwen3vl.tokenizer(model_type="qwen3vl_4b")()
    tok8 = qwen3vl.tokenizer(model_type="qwen3vl_8b")()

    brut32 = tok32.qwen3vl_32b.tokenizer
    brut4 = tok4.qwen3vl_4b.tokenizer
    brut8 = tok8.qwen3vl_8b.tokenizer

    print("Vocabulaires : 32B %d | 4B %d | 8B %d"
          % (len(brut32), len(brut4), len(brut8)))
    print("Objets identiques en memoire : 32B/4B %s | 32B/8B %s"
          % (brut32 is brut4, brut32 is brut8))
    print("")

    textes = PIEGES + charger_corpus()
    print("%d textes a comparer (%d pieges + %d prompts reels)"
          % (len(textes), len(PIEGES), len(textes) - len(PIEGES)))

    ecarts4, ecarts8, total = [], [], 0
    for t in textes:
        a = brut32(t, add_special_tokens=False)["input_ids"]
        b = brut4(t, add_special_tokens=False)["input_ids"]
        c = brut8(t, add_special_tokens=False)["input_ids"]
        total += len(a)
        if a != b:
            ecarts4.append(t)
        if a != c:
            ecarts8.append(t)

    print("%d tokens compares" % total)
    print("")
    print("  32B vs 4B : %s" % ("identique" if not ecarts4
                                else "%d texte(s) differents" % len(ecarts4)))
    print("  32B vs 8B : %s" % ("identique" if not ecarts8
                                else "%d texte(s) differents" % len(ecarts8)))

    for nom, ecarts, brut in (("4B", ecarts4, brut4), ("8B", ecarts8, brut8)):
        for t in ecarts[:5]:
            a = brut32(t, add_special_tokens=False)["input_ids"]
            b = brut(t, add_special_tokens=False)["input_ids"]
            print("")
            print("  %s -- texte : %r" % (nom, t[:70]))
            print("     32B : %s" % a[:20])
            print("     %-3s : %s" % (nom, b[:20]))

    # Comparaison exhaustive du vocabulaire, identifiant par identifiant.
    v32, v4, v8 = brut32.get_vocab(), brut4.get_vocab(), brut8.get_vocab()
    print("")
    print("Vocabulaire complet, entree par entree :")
    print("  32B vs 4B : %s" % ("identique" if v32 == v4 else "DIFFERENT"))
    print("  32B vs 8B : %s" % ("identique" if v32 == v8 else "DIFFERENT"))
    if v32 != v4:
        seuls32 = set(v32) - set(v4)
        seuls4 = set(v4) - set(v32)
        print("    presents seulement dans le 32B : %d" % len(seuls32))
        print("    presents seulement dans le 4B  : %d" % len(seuls4))
        for t in list(seuls32)[:10]:
            print("      32B seul : %r -> %d" % (t, v32[t]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
