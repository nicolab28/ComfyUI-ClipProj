# Measurements

Every number quoted in the README or in the r/StableDiffusion thread, with the method that produced it and the script that reproduces it. Where a measurement contradicted what I had claimed, the measurement is kept and the claim is marked wrong.

Machine: i7-14700KF, 128 GB RAM, five GPUs behind a PLX switch at PCIe 4.0 x8. Windows 11, ComfyUI 0.31.0, PyTorch 2.13 + CUDA 13.0, Python 3.13.

Reconstruction figures come from `calibration/run.py`, generation figures from MiniMax H3 at 832x480 or 864x480, 6 steps with the Turbo LoRA unless stated.

---

## Does the small model carry the information at all

`calibration/probe.py`, cross-prompt CKA between the 4B and the 32B, one mean vector per prompt, attention sink dropped and each dimension standardised.

```
CKA, 200 prompts    0.95
CKA, 2000 prompts   0.92
```

Two traps here cost real time. Token 0 and a handful of dimensions out of 5120 carry values hundreds of times the standard deviation; left alone they saturate the measure, and CKA reads 0.9999 across twenty consecutive layers, which means nothing. And CKA measured *within* a prompt is inflated by the shared tokenizer, so it has to be measured across prompts.

## How well the projection reconstructs

`calibration/run.py`, ridge regression, tap 24 of 36, held-out test set.

| Encoder | Corpus | Tokens | Test cosine | Test R² |
|---|---|---|---|---|
| `qwen3vl_4b_bf16` | 200 prompts | 37 361 | 0.699 | 0.490 |
| `qwen3vl_4b_bf16` | 2 000 prompts | 288 608 | 0.712 | 0.507 |
| `qwen3vl_4b_int8_convrot` | 240 prompts | 37 851 | 0.697 | 0.489 |
| `qwen3vl_8b_nvfp4` | 203 prompts | 37 851 | 0.731 | 0.538 |

Eight times the data buys 1.8 % of cosine. The linear map is at its ceiling, not starved.

A cosine of 0.71 sounds unusable and is not. What that actually means is documented below, in what cosine fails to predict.

## Does one matrix work across encoder variants

A matrix calibrated on `qwen3vl_4b_bf16`, applied to an abliterated fp8 variant:

```
cosine gap    0.0023
```

So yes. One matrix per size covers every variant of that size, including `int8_convrot`, whose rotation turns out to be compensated so the activations stay in the expected frame.

## Are the tokenizers really identical

The whole method rests on this: the same prompt must give the same ids at the same positions in both models. `calibration/tokens.py`.

```
2014 texts, 472 582 tokens compared
32B against 4B    identical
32B against 8B    identical
full vocabulary, entry by entry    identical
```

The corpus was chosen to break a tokenizer if it could: French accents, Japanese, Chinese, Korean, Russian, Greek, Arabic, composed emoji, `<d>`, `<|im_start|>`, timecodes, multiple spaces, tabs, a 300 character run of one letter, and the empty string.

The embedding matrix has 151 936 rows while the tokenizer declares 151 669, so 267 slots are reserved and unreachable. If tags had been added to the MiniMax checkpoint that is where they would be, and they are not.

## The attention sink

Token 0 of any sequence:

```
direction, cosine against its own mean over 1966 prompts    1.0000
norm                                                        16 527
norm of an ordinary text token                              291
```

It carries nothing from the text and its direction never changes. Calibration excludes it, correctly, because its magnitude would wreck the statistics. But v0.1.0 projected it anyway through a matrix that had never seen one, producing an arbitrary vector of enormous norm: 0.5 % of the positions on a 200 token prompt and nobody notices, 14 % on a 7 token prompt and the result is ruined.

Since the vector is constant, substituting its measured value is exact rather than approximate. `calibration/add_sink.py` writes it into a matrix.

## Do short prompts break

I assumed they did, because the calibration corpus filters out anything under 15 words. `calibration/length.py`, per-token cosine, sink excluded, sink substitution applied.

```
 2 words    0.9081
 3 words    0.9142
 5 words    0.9171
 8 words    0.9227
12 words    0.9280
20 words    0.9326
40 words    0.9368
80 words    0.9366
```

Three points of decline between 80 words and 2. **My hypothesis was wrong.** The short-prompt failures people reported were the attention sink, not the corpus.

## Does a non-English line reconstruct worse

Spoken French degrades badly in generation, so I assumed the English-only corpus was to blame. `calibration/language.py` compares prompts identical word for word except the quoted line.

```
English line in an English prompt    0.8996
French line in the same prompt       0.8974
gap                                  -0.0021
whole prompt in French               0.8699
```

**Wrong again.** Two thousandths is noise. A French line embedded in an English prompt reconstructs exactly as well as its English twin, which is the actual use case. Only a fully French prompt drops, and not to a level that would explain what you hear.

## What cosine does not predict

Two independent results say the reconstruction cosine is a poor guide to what matters.

The 8B has the better cosine of the two projected models, 0.731 against 0.697 on its own corpus, and is by far the worst at speech: it drops the language entirely and speaks the French and Spanish lines in English.

And the `condition_proj` weighting below raised the reported cosine from 0.697 to 0.845 while changing the output not at all.

## The condition_proj weighting, which does nothing

The idea, from u/stddealer: the DiT does not read the conditioning directly, it passes it through `condition_proj`, a `Linear(5120 → 5376)`. That layer is very uneven.

```
singular values    max 37.40    median 4.20    min 0.10
conditioning                    362
top decile over bottom decile   45x
energy in the first 10 % of directions    52 %
```

So plain ridge spends as much effort on a direction the DiT will multiply by 0.10 as on one it will multiply by 37. Calibrating against the output of that layer, then mapping back through the pseudo-inverse, should minimise the error the DiT actually sees. Reported cosine rose from 0.697 to 0.845 on the 4B and 0.731 to 0.860 on the 8B.

Then I compared what the matrices output, on random input vectors:

```
4B   weighted against unweighted, same corpus    0.999998
8B   weighted against unweighted, same corpus    0.999999
4B   weighted against the 2000-prompt matrix     0.744034
```

**They are the same function.** The entire gain was an artefact of the measurement space: applying an invertible transform to both vectors before taking a cosine flatters agreement on the dominant directions.

Unregularised least squares is invariant to an invertible transform of the targets, so fitting in one space and mapping back recovers the same map. Only the ridge penalty breaks that invariance, and with 37 851 tokens against λ = 1000 it barely binds. The idea is sound and would matter with far less data or far stronger regularisation. Not here.

## Is the pipeline deterministic

Everything below compares encoders at a fixed seed, which is worthless if the pipeline is not deterministic. Three renders of the same prompt, same seed 42, same 8B encoder:

```
run 1    video d87b6fbfcb0b8f1a    audio 0ab97337b143ba44
run 2    video d87b6fbfcb0b8f1a    audio 0ab97337b143ba44
run 3    video d87b6fbfcb0b8f1a    audio 0ab97337b143ba44
```

md5 of the decoded streams, not of the container. Byte identical. So every difference between encoders is caused by the encoder.

## Does a smaller encoder save time

Same prompt, same seed, same settings, only the `clip` input changing.

```
4B     175 s
8B     174 s
32B    178 s

8B again    174 s
8B again    171 s
```

Two identical runs differ by 3 seconds, the same spread as the difference between the three encoders. **The encoder size is not measurable in total time.**

That is expected: the encoder runs once per generation, a single prefill with no autoregressive decoding, 0.385 s against 238 s of sampling in my workflow. Streaming 14.6 GB over PCIe 4.0 x8 once costs about a second.

Two caveats. With 128 GB of RAM the checkpoints are served from the OS file cache and never truly read from disk; with less RAM the 32B costs more. And a machine that cannot fit the 32B at all falls back to CPU, which is where the 30 minutes against 30 seconds reported by u/hum_ma comes from.

Independently confirmed by u/Mammoth_Reindeer_941 on a 2x3080 box: 349 s against 360 s, also inside the noise.

## Audio level

Measured with `ffmpeg -af volumedetect` on the three comparison clips, same prompt and seed.

```
             mean       peak
4B         -28.3 dB   -10.9 dB
8B         -21.8 dB    -5.5 dB
32B        -20.7 dB    -2.5 dB
```

The projected 4B is 7.6 dB quieter than the reference in mean and 8.4 dB in peak. u/Mammoth_Reindeer_941 reports a much larger gap on an ambience-only prompt, -18 dB against -50 and -39, near the null controls. Speech appears to survive the projection better than ambient sound.

Note that the 8B sits almost at the reference level while being the worst at pronouncing anything, so loudness and content fail separately.

I also had audio level problems with H3 before this node existed, on the native 32B path, and never diagnosed them. Some of the larger gaps reported may be two problems stacked.

## Speech, by language

Same prompt, same seed, one line per shot in a different language. Transcribed per shot with whisper.cpp, since transcribing the whole clip in one pass lets the detected language of one segment contaminate the next.

```
expected   Three lemons, and one always falls.
4B         3 Lemons and one Always Falls.               exact
8B         3 Lemmens, et 1 Olsies Fals.
32B        (truncated by the fixed window)

expected   La fete foraine ouvre a la tombee du jour.
4B         La fete forere ouvre a la tombee du Your.
8B         La fete foret in opera to the temple.
32B        La fete foraine ouvre a la tombee du jour.   exact

expected   El perro corre sobre la hierba mojada.
4B         El Perocor sobre la hier bramorada.
8B         El Perro cause over the Hereup Moyen.
32B        El Perocore sobre la Yerba Mojada.
```

The 32B keeps all three languages. The 4B keeps the accent and slurs the words. The 8B abandons the language and speaks English throughout. Whisper is itself a model and can mishear, so treat this as an indication rather than a verdict.

## Where named references actually break

`calibration/knows.py` asks the encoder to describe a person in plain text, which bypasses the matrix entirely.

```
Scarlett Johansson, hair and eyes

4B int8_convrot    dark brown hair, deep brown eyes
4B bf16            dark brown hair, blue eyes
8B nvfp4           wavy blonde hair, striking blue eyes
```

All three know she plays Black Widow. Only the 8B describes her correctly. So when a proper noun renders as the wrong person, the projection is transmitting a wrong memory faithfully rather than losing a correct one. **My original claim that the projection loses named people was wrong.**

Note also that int8 quantisation costs facts: the eye colour, and Will Smith's filmography, are both correct in bf16 and wrong in int8_convrot.

None of this applies to ref2va, where identity comes from the reference image.

## Feeding the residual several taps, which does not work

The residual sees the same layer the ridge was calibrated on, tap 24. Neighbouring layers are not redundant with it — cosine 0.7 between adjacent taps — so there is signal next to the one being used, and the encodings for taps 22 and 26 were already on disk. It costs nothing to try.

It loses. Same corpus, same ridge, same 16384-wide hidden layer, input widened from 2560 to 7680:

| residual input | parameters | best test cosine |
|---|---|---|
| tap 24 alone | 126M | **0.7944** |
| taps 22, 24, 26 | 210M | 0.7878 |

Run twice, once with the usual patience of 8 and once with patience 30 so that the cosine learning-rate schedule had the same room. Both peak early and then decline: the second run peaks at 0.7878 around epoch 16 and falls to 0.783 by epoch 46 while the training loss keeps dropping from 0.5575 to 0.3091. That is overfitting, not a plateau.

The extra capacity is spent memorising. It also converges faster at the very start, 0.7546 at epoch 1 against 0.7489, which is exactly what a model with more parameters and correlated inputs does before it turns.

What this does not settle: the taps tried are the two immediately either side of 24, because those were the ones already encoded, and they are therefore the most redundant choice available. Widely spaced taps might behave differently. But the failure mode here is capacity against 1.1M training tokens rather than a shortage of signal, and that does not change with the spacing.

## A methodology note

`calibration/run.py` selects the ridge λ from a grid. An optimum landing on the edge of that grid is not an optimum: the real value lies beyond it and the matrix is under-fitted. The 8B originally retained 1e4, which was the largest value tried. Widening the grid to 1e7 showed 1e4 was genuinely interior, so the result stood, but it stood by luck. The script now warns.
