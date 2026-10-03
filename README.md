# Temporally Sensitive Neural and Physiological Decoding of Fine-Grained Affective States Estimated by Continuous Facial Expressions

Core code for the paper: a reference implementation of its network and objective.
Including the cross-modal regression network, the CCC-DILATE objectives, and a runnable
inference demo on a real held-out example from the MAHNOB-HCI dataset.

## Contents

```
cccdilate/   encoders.py, attention.py, model.py (CCCDilateNet), losses.py
demo.py      reference inference demo (writes a figure)
assets/      MAHNOB_LOSO_1.pth (md5 2c93477a…), MAHNOB_norm.csv (md5 5b0b4198…),
             MAHNOB_LOSO_1_sample.npz (md5 ba4e4344…); requirements.txt, LICENSE
```

## Install

```bash
pip install -r requirements.txt
```

`torch`, `numpy` and `matplotlib` are the only packages the release imports; the research
pipeline also used pandas, scipy, scikit-learn, mne-features, torcheeg and numba.

## Run

```bash
python demo.py                # bundled held-out MAHNOB subject P01 windows
python demo.py --synthetic    # random tensors of the same shape, no data assets
```

CPU only, no dataset download, nothing read outside the repository. The real run prints
the held-out scores:

```
valence       0.5618    0.6444
arousal       0.3760    0.4711
```

It also writes a figure (default `demo_output.png`, `--out-fig` to change): valence left,
arousal right, ground truth and prediction across all 48,600 scored frames, smoothed with
a Hann-window moving average of 301 frames (`--smooth N` to change, `--smooth 1` to
disable). `--synthetic` writes no figure; `--norm`, `--sample` and `--checkpoint` point
the demo at your own statistics, tensors or weights.

## Model

`CCCDilateNet` maps a `[B, S, C*4]` window to `[B, S, 2]`, one (valence, arousal) pair per
stored frame, in three stages: **per-modality encoders** (each step's channel vector
projected to 32 features, then two temporal convolutions halving the time axis twice,
32 → 64 → 128 channels, plus sinusoidal positional encoding); **gated cross-modal
attention** (six independent pre-gated cross-attentions, one per ordered modality pair,
each followed by a contextual-aware gate fusing the attended context with the query
modality's own representation); and a **temporal model** (the six contexts concatenated to
768 channels, layer normalised, passed through single-head channel attention, projected to
`d_model = 256`, then a 2-layer `TransformerEncoder` and a linear head). The trailing
factor 4 is preprocessing: 128 Hz streams (eye tracking 60 Hz) are resampled to 120 Hz and
four consecutive samples per channel are packed into one stored frame, channel-major;
`forward` unfolds it to `[B, S*4, C]`. For the released MAHNOB configuration `C = 100`:
EEG 64 (32 electrodes plus 32 gamma envelopes), GSR 18 and eye 18 (both expanded to
position, velocity and acceleration). The bundled example covers the MAHNOB-HCI data,
whose synchronised streams are EEG, GSR and eye tracking, and the network takes those
three modalities as its input.

## Objective

`CCCLoss` optimises `1 − CCC` with `CCC = 2·cov(y, ŷ) / (var(y) + var(ŷ) + (ȳ − ŷ̄)²)`.
`DILATE` adds a temporal-shape term, `α·L_shape + (1 − α)·L_temporal`, where `L_shape` is
the soft-DTW alignment cost and `L_temporal = ⟨A*_γ, Ω⟩ / T²` the expected temporal
displacement of the soft alignment (`Ω[i,j] = (i−j)²`). `CCCDILATELoss` combines the two as
`L_CCC + λ·L_DILATE`. The paper's objective configuration is **α = 0.8** and
**λ = 0.001**, the defaults of these classes.

## Checkpoint

`assets/MAHNOB_LOSO_1.pth` — the paper's network for held-out subject **P01** (LOSO fold 1,
trained on the remaining subjects), re-exported at its best-validation epoch. On the
complete held-out split it scores valence CCC **0.5618** / PCC 0.6444 and arousal CCC
**0.3760** / PCC 0.4711 for this held-out subject. The paper reports the cohort-level mean
over the 27 subjects, so this per-subject value is not printed there.

Export note: the research script snapshotted its best epoch with
`best_model_wts = model.state_dict()`, whose detached *views* share storage with the live
parameters, so later optimiser steps mutated it and the file first written held the
final-epoch state. Re-running with `{k: v.detach().cpu().clone() for k, v in
model.state_dict().items()}` reproduced the original trajectory (best validation loss
−0.102405 against −0.102404) and exported the selected epoch.

## Normalisation

Every input column and every target uses the shipped statistics; anyone reproducing the
paper's numbers must apply the same ones. `assets/MAHNOB_norm.csv` has the fixed header
`group,index,mean,scale` and 402 rows: 400 `eeg` rows (`index` 0–399, one per input feature
column) and 2 `label` rows (`index` 0 = valence, 1 = arousal). Values use `%.17g` and are
parsed by `index`, so file order does not matter; a column is normalised as
`x_norm = (x_raw - mean) / scale`. Pass your own file with `--norm` to run without the
original cohort. The statistics are the `mean_`/`scale_` of a `StandardScaler` fit on the
full 30-subject cohort; targets were built by per-subject z-scoring, then clipping to ±3
standard deviations, then a global `StandardScaler`, and the optional zero-phase band-pass
on labels is commented out in the research script, so it was not applied.

## Verification

The released network is numerically identical to the original class with the same checkpoint
(`max|diff| = 0.000e+00` over the full held-out split and on random inputs), the checkpoint
loads with `strict=True` (184/184 keys), and the released losses match the original
implementations.

## Data availability

Developed and evaluated on public affective-computing corpora (MAHNOB-HCI and others listed
in the paper), available from their providers under their own terms. The bundled example
covers the MAHNOB-HCI data, whose synchronised streams are EEG, GSR and eye tracking; the
clinical cohort cannot be shared (IRB restrictions), so it is not part of this release.
The complete data-processing and training pipeline will be released with the final version
of the paper.

## Licence

MIT — see [LICENSE](LICENSE).
