# Why not the other way round

Asked three times now, and it is a good question: if a 4B can stand in for the 32B on MiniMax H3, why not put the 32B in front of Krea 2, which normally conditions on a 4B? More knowledge for the same wiring.

The short answer is that it would cost 11 GB to get something slightly worse than what a 4B already gives you. The reasons are worth writing down, because two of them are not obvious and one only became clear after reading Krea 2's source.

## A projection converts, it never adds

A fitted map has a target, and the target is its ceiling. Fitting a 32B into Krea 2's space means fitting it toward what the 4B produces, because that is what Krea 2's DiT was trained to read. So the best possible outcome is **exactly the 4B**, and no better.

Where the 32B knows a person or a building the 4B does not, the target does not know them either. The regression sees no reward for carrying that knowledge and learns to reproduce the gap rather than fill it. The extra knowledge has nowhere to go.

And a conversion is never free. Measured on held-out prompts, our own map reconstructs the 32B at a cosine of 0.79 in the direction that does work. A map in the other direction would lose a comparable amount. So the realistic outcome is the 4B's conditioning, minus the conversion loss, in exchange for 11 GB of VRAM.

Our direction trades fidelity for memory. That direction trades fidelity for **more** memory. It is dominated on both axes.

## Truncating the dimensions does not work either

5120 and 2560 happen to be a factor of two apart, which invites the idea of dropping half. The two models share no basis: coordinate 7 of the 32B means nothing in particular to the 4B. That is precisely why a learned matrix is needed rather than a slice. And the learned matrix still caps at the 4B.

## Krea 2 is built differently, and that raises the price

This part only became clear on reading [`comfy/text_encoders/krea2.py`](https://github.com/comfyanonymous/ComfyUI) and `comfy/ldm/krea2/model.py`.

Krea 2 does not condition on one hidden state. It takes **twelve**, at taps 2, 5, 8 … 35 of the Qwen3-VL-4B, carried as `12 × 2560 = 30720` values per token. Its DiT has a dedicated small transformer for them, `txtfusion`: two attention blocks applied to each tap separately with shared weights, then a learned `Linear(12, 1)` that mixes the twelve, then two more blocks on the merged result.

So a map from the 32B would be `5120 → 30720`, which is 157M parameters for the linear part alone against 13M for the H3 one. And parameter count is the easy part: you cannot manufacture twelve depths out of one. Tap 2 of a 4B is nearly lexical and tap 35 is semantic; they are different functions of the same text. You would need several taps of the 32B as input, which turns one regression into twelve coupled ones and requires re-encoding the corpus keeping taps on both sides.

## What those twelve taps actually carry

Since the design was surprising, it seemed worth measuring rather than assuming. Effective rank, via the participation ratio of the eigenvalues of the covariance, which equals *d* when variance is spread evenly over *d* directions and 1 when it all sits on one. Measured on 8000 real token positions, attention sink excluded, each side scaled by its own RMS as the DiT's RMSNorm would:

| conditioning | width | effective rank | top 10 directions |
|---|---|---|---|
| H3, the 32B's final state | 5120 | **113.4** | 23.6 % of the variance |
| Krea 2, twelve 4B taps stacked | 30720 | **13.0** | 86.2 % of the variance |

Six times the numbers, a ninth of the spread. The taps are strongly correlated because a residual stream is additive and each tap largely contains the previous one. Mean cosine between taps, same token, no centering:

```
        2      5      8     11     14     17     20     23     26     29     32     35
  2  1.000  0.378  0.295  0.260  0.229  0.226  0.197  0.179  0.173  0.173  0.189  0.187
 17  0.226  0.327  0.381  0.406  0.463  1.000  0.503  0.481  0.486  0.484  0.508  0.514
 32  0.189  0.298  0.382  0.400  0.455  0.508  0.553  0.566  0.622  0.644  1.000  0.706
 35  0.187  0.299  0.391  0.408  0.461  0.514  0.552  0.564  0.621  0.648  0.706  1.000
```

Every pair is positively correlated, over the whole depth: 0.706 between the last two taps, and still 0.187 between the first and the last.

Two honest caveats. Effective rank measures how variance spreads, not what a model can use — a low-variance direction can carry decisive information, and the depth trajectory is real information even when it weighs little. And the 32B measured here is NVFP4 while the 4B is bf16; quantisation noise is roughly isotropic and inflates effective rank, so the real gap is somewhat smaller than 113 against 13.

What this does **not** say is that Krea 2 receives less than H3. It says the twelve taps are far more redundant than their width suggests, which is what one would expect from twelve samples of one residual stream.

## The one case where it would make sense

If the 32B is already resident because you are running H3, and you want Krea 2 in the same graph without loading a second encoder, then a 32B to Krea 2 projection saves you the 4B's 5 GB. You pay for it in image fidelity. That is narrow, and it is the only version of this idea that is not dominated.

## Why H3 was the easy case

H3 takes a single final hidden state, 5120 wide. One target, one regression, one tap. That is the entire reason a closed-form ridge worked at all. If H3 were built like Krea 2, this project would probably not exist.

## Reproducing

`calibration/rank.py` measures the effective rank and the tap correlations from two encodings. The Krea 2 architecture is read directly from ComfyUI's own source, no reverse engineering involved: `comfy/text_encoders/krea2.py` for the taps and the flattening, `comfy/ldm/krea2/model.py` for `txtfusion`.
