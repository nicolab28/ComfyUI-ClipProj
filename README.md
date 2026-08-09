# ComfyUI-ClipProj

**Swap a large text encoder for a small one, with a learned linear projection.**

**Version 0.1.2** — matrices are now `.safetensors`; `.pt` is still read but no longer distributed, since pickle can execute code on load.

> ## ⚠️ Proof of concept — working, but a proof of concept
> **It runs and it produces good video**, and every number below was measured on real hardware rather than estimated. It is still a proof of concept, not a finished product: built and tested on a single setup — **Windows 11, NVIDIA, ComfyUI 0.31.0** — with deliberately limited exploration. Expect rough edges and breaking changes. **Use at your own risk.**

Diffusion models spend a lot of VRAM on their text encoder. MiniMax H3 uses a **Qwen3-VL-32B** truncated to 50 layers — **15.7 GB in NVFP4** — solely to turn a prompt into a `[seq, 5120]` conditioning tensor.

ClipProj replaces it with a **Qwen3-VL-4B** (2560 dims) plus a learned linear map into the 5120-dim space the DiT expects:

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out
```

**15.7 GB → 4.5 GB** with the int8_convrot encoder (8.3 GB in bf16, 5.2 GB in fp8). The DiT, the VAEs and the sampler are untouched: the node returns an object that behaves like the official CLIP, so it drops into the existing `clip` input with no rewiring.

![Full MiniMax H3 pipeline conditioned by a projected Qwen3-VL-4B](examples/minimax_h3_clipproj.png)

*The complete pipeline — encoder, projection, conditioning, sampling, decode and video output. `examples/minimax_h3_clipproj.json`*

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nicolab28/ComfyUI-ClipProj
```

Restart ComfyUI. **There is no `requirements.txt` and nothing to `pip install`** — the nodes import only `torch` and ComfyUI's own modules, all already present in any working installation.

On first launch the folder `ComfyUI/models/clip_projections/` is created. Put the projection matrices there (see *Models and matrices* below).

Example workflows are in `examples/`: drag any `.json` onto the ComfyUI canvas.

### Tested on

| | |
|---|---|
| OS | **Windows 11** — Linux and macOS should work but were not tested |
| GPU | NVIDIA (RTX 3090 / 4070 / 3060). Not tested on AMD or Apple Silicon |
| ComfyUI | 0.31.0 |
| PyTorch | 2.13.0 + CUDA 13.0, Python 3.13 |

The package is declared `OS Independent` because nothing in the code is platform-specific: no absolute paths, no `win32` calls, no hard-coded separators. Only the *testing* happened on Windows. If you run it elsewhere and hit a problem, that is a bug worth reporting rather than an expected limitation.

## Why it works

The 4B and the 32B share the **same tokenizer** (151936 tokens): a prompt yields the same tokens at the same positions in both. That makes a position-by-position mapping between their hidden states learnable — there is no alignment problem.

Calibration is **ridge regression**, not training. No gradients, no epochs, no learning rate. Encode N prompts with both models, accumulate `XᵀX` and `XᵀY` in streaming (constant memory), solve.

## Measured results

MiniMax H3, tap 24 of 36, Seedance prompt corpus:

| Corpus | Tokens | Cross-prompt CKA | Test cosine | Test R² |
|---|---|---|---|---|
| 200 prompts | 37 361 | 0.95 | 0.699 | 0.490 |
| 2 000 prompts | 288 608 | 0.92 | **0.712** | 0.507 |

Eight times the data buys 1.8 % of cosine: **the linear projection is at its ceiling**, not starved of data. Going further needs an MLP, not more prompts.

A cosine of 0.71 sounds poor and **is not** — the DiT tolerates far more than the metric suggests. What holds up in actual generation:

- simple prompts ✅
- structured multi-shot prompts (`subject_definitions`, timecoded shots, `overall_soundscape`) — four distinct cuts, no bleed between them ✅
- **fl2va with first and last frame** ✅, although `W` only ever saw text positions
- robust to swapping encoder weights: a `W` calibrated on bf16 works on an abliterated fp8 variant, and on `int8_convrot` — whose rotation turns out to be compensated, so the activations stay in the expected frame

## Changes in 0.1.2

**Matrices are read from `.safetensors`.** A `.pt` goes through pickle, which can execute arbitrary code the moment it is opened — an absurd risk for a file holding nothing but tensors and a handful of scalars. The scalars now live in the safetensors header, which stores strings, and are converted back on load. Legacy `.pt` files are still accepted so nothing breaks, but the published matrices are safetensors only. `calibration/to_safetensors.py` converts your own, verifying each tensor before it returns.

## Changes in 0.1.1

**The attention-sink vector is now substituted at inference, and this one matters.** The first token of a sequence is an attention sink: its direction is constant from one prompt to the next — cosine 1.0000 measured over 1966 prompts — and it carries nothing from the text, yet its norm reaches 16 500 against 291 for a text token. Calibration excluded it, rightly, since its extreme values would wreck the statistics. But the node projected it anyway, through a matrix that had never seen one, producing an arbitrary vector of enormous norm. That is invisible on a 200-token prompt where it is 0.5 % of the positions, and ruinous on a 7-token one where it is 14 % — which is exactly the short-prompt breakage people hit. Because the vector is constant, substituting its measured value is exact rather than approximate. **Re-download the matrices**: the fix lives in them.

**ref2va now works** and is no longer refused. Load the encoder in `resident` mode for it: the `dynamic` path crashes inside ComfyUI's vision tower with int8 encoders, and that only surfaces when an image is present.

**Encoder architecture is detected automatically** from the checkpoint header, so `type` can stay on `auto` instead of guessing between `krea2`, `boogu` and `minimax`.

**Encoder swapping frees properly.** The previous model is released *before* the replacement loads, rather than after, so a tight card no longer fails mid-swap. A loaded instance is now identified by file, device *and* mode, so changing card or residency releases the old one. And a loader re-run by ComfyUI's cache no longer stacks a second copy of the same checkpoint in VRAM.

**`repetition_penalty` is exposed** on the Generate node. Note it only applies while sampling: at temperature 0 ComfyUI takes the most likely token outright and ignores it.

## Known limitations

The 4B holds up **well enough** on everything tried so far — but the trials were not pushed far, and the honest summary is that **you lose knowledge the 32B has**.

**Named references depend on the encoder, not on the projection.** This is measured rather than guessed: asking the encoder to describe a person in plain text bypasses the matrix entirely and shows where the knowledge actually stops. The 4B places Scarlett Johansson correctly as Black Widow but believes she has dark brown hair; the 8B describes her correctly as blonde with blue eyes. When a proper noun renders as the wrong person, question the encoder first — `calibration/knows.py` runs that test for any name, via the `H3_NAMES` environment variable.

The workaround is to describe rather than name: *"the actress X as [role], blonde, ..."* recovers an identity the bare name loses, on both models. A name is a fragile signal carried by two or three tokens; a description spreads it over a dozen redundant ones and the reconstruction error averages out instead of accumulating. None of this applies to **ref2va**, where identity comes from the reference image.

**Speech in languages other than English degrades.** Reproduced with French: the 32B pronounces it cleanly, a projected 4B or 8B does not. It is not a corpus problem — on identical English prompts differing only in the quoted line, French tokens reconstruct at 0.8974 against 0.8996 for English, which is noise. A cosine of 0.90 is ample for visual semantics and insufficient for phonetics: the DiT's audio branch is far more demanding than its image branch, and a language the model handles less confidently has less margin to absorb the error.

**Other known gaps:**

- image references work in **fl2va** and **ref2va**, but `W` was calibrated on text positions only, so vision positions are projected out of their training distribution
- the linear projection is **at its ceiling**: eight times more calibration data bought 1.8 % of cosine. Going further needs an MLP, not more prompts
- quantisation costs facts: `int8_convrot` against `bf16` shows measurable factual errors appearing, fine for general use, worth knowing for prompts that lean on proper nouns

## Control matrices — run these first

Three synthetic matrices are selectable in the node. They exist to prove that `W` is doing the work rather than the diffusion model.

| Control | What it does | Output for *"a red ball on a wood table"* |
|---|---|---|
| `<control:zero>` | `W = 0`, constant conditioning | a countryside landscape — prompt entirely ignored |
| `<control:identity>` | raw copy of the 2560 dims, no learning | a golden object in flames — unusable |
| learned `W` | the calibrated projection | the red ball on a wood table |

`‖W_identity‖ = 50.6` vs `‖W_learned‖ = 52.4` — near-identical energy, so the difference is purely structural. **Always run these controls before trusting a new projection.**

## Nodes

| Node | Purpose |
|---|---|
| **ClipProj Device Loader** | loads any text encoder on a chosen GPU, without projecting |
| **ClipProj Apply** | projects an already-loaded CLIP, like a LoRA |
| **ClipProj Loader (all-in-one)** | loads a small encoder on a chosen GPU and projects it |
| **ClipProj Generate / Caption** | text generation and image captioning on the **same resident weights** |
| **ClipProj Free VRAM** | releases pinned encoders on demand |

The stock `CLIPLoader` only offers `default` and `cpu` for its device, so there is no way to target a specific card on a multi-GPU machine. That is why the loaders exist.

`Generate / Caption` exists because ComfyUI's `SDClipModel.generate` drops `embeds_info` and never calls `build_image_inputs`: image tokens end up at linear positions instead of Qwen3-VL's 3D mRoPE and without DeepStack injection, which makes captioning fail. This node restores the full path.

![Turning a one-line description into a structured H3 prompt](examples/rewrite_h3_prompt.png)

*"an old fisherman mends his net on a quay at sunrise, 10 seconds" in, a full three-shot H3 prompt out. `examples/rewrite_h3_prompt.json`*

![Captioning an image on the same resident weights](examples/caption_image.png)

*Image captioning, no second model loaded. `examples/caption_image.json`*

Generation is **far slower than a dedicated engine** (llama.cpp, vLLM). ComfyUI's ops re-cast weights on every forward — sound for diffusion, costly when decoding one token at a time. Two mitigations are exposed (`precision`, `preload_head`); expect around 20 tok/s on a 3090. For heavy use, prefer a real inference server.

## Models and matrices

**This repository ships no model weights.** They stay with their authors — download them from the official sources below and point the node at them.

| Role | Where | Licence |
|---|---|---|
| **Projection matrices** | **[NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)** | MIT |
| Diffusion model + VAEs | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | custom |
| Text encoder (small) | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) — `text_encoders/qwen3vl_4b_*` | Apache 2.0 |

The matrices go in `models/clip_projections/`. Each `.pt` records the model it was calibrated on, but any variant of matching output dimension will load — including quantised or fine-tuned ones.

The 32B text encoder is **no longer needed**, which is the entire point.

## Calibrating your own

The method is not specific to MiniMax H3. It applies wherever a large text encoder has a smaller sibling **in the same family**. Qwen3-VL ships in 2B, 4B, 8B, 30B-A3B, 32B and 235B-A22B, all sharing one tokenizer — so any pair among them is a candidate:

- Flux 2 / Klein — Mistral3-24B pruned
- Ideogram4, Boogu, JoyImage — Qwen3-VL-8B, replaceable by the 4B or 2B

Two requirements: an **identical tokenizer** between the two models, and handling multi-layer taps when the target consumes a stack rather than a single layer.

See `calibration/`:

| Script | Purpose |
|---|---|
| `probe.py` | CKA probe — is the information even present? Run before anything else |
| `run.py` | full calibration, produces the `.pt` |
| `add_sink.py` | measures the attention-sink vector and writes it into existing matrices |
| `make_controls.py` | builds the zero / identity control matrices |
| `test_node.py` | end-to-end check without launching ComfyUI |
| `knows.py` | asks the encoder to describe a name in plain text, bypassing the matrix |
| `length.py` | reconstruction fidelity as a function of prompt length |
| `language.py` | reconstruction fidelity of a non-English line against its English twin |

Every path is an environment variable with a sensible default; read the docstring at the top of each script.

The last three exist because a projection can be blamed for things it does not do. Before assuming the matrix is at fault, check whether the encoder knows the fact at all (`knows.py`), whether short prompts are genuinely worse (`length.py` — they are barely), and whether another language reconstructs worse (`language.py` — it does not). All three hypotheses looked obvious and all three were wrong.

Three traps worth knowing, each of which cost real time to find:

1. **Massive activations.** Token 0 (the attention sink) and 5–14 dimensions out of 5120 carry values hundreds of times the standard deviation. Left alone they saturate any linear similarity measure — CKA reads 0.9999 across twenty consecutive layers and means nothing. Drop the sink and standardise per dimension — then substitute its measured value at inference, or short prompts break.
2. **Lexical identity.** CKA measured *within* a prompt is inflated by both models sharing a tokenizer. Measure **across prompts** (one mean vector per prompt).
3. **A regularisation optimum on the edge of the grid is not an optimum.** If the retained lambda is the largest value tried, the real one is beyond it and the matrix is under-fitted. `run.py` now warns.

## Credits

Vibe-coded with **Anthropic Claude Code (Opus 5)** — design, implementation, calibration scripts and this README. Every number quoted above was measured on real hardware, not estimated: where a prediction turned out wrong, the measurement won and the text was corrected.

## Licence and responsibility

The **code** in this repository is MIT (see `LICENSE`).

Everything else is not ours, and using it is your responsibility:

- **Qwen3-VL** is published by Alibaba under **Apache 2.0**. Read and comply with its terms and acceptable-use policy.
- **MiniMax H3** ships under a **custom licence** (`license: other` on Hugging Face). Read it before any use, particularly commercial.
- **Projection matrices** are derived from the activations of both models. Their legal status is unclear. They are provided as-is, for research, with no claim of ownership over anything derived from the underlying models.

This project is **not affiliated with, endorsed by, or connected to** Alibaba / Qwen, MiniMax, or Comfy Org. All trademarks belong to their respective owners.

You remain responsible for what you generate and for complying with the licences of every model you load. The authors accept no liability for any use of this software.
