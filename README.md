# CCC-DILATE — inference release

Reference implementation of the **CCC-DILATE** model for continuous emotion
(valence / arousal) regression from three synchronised physiological and
behavioural modalities: **EEG**, **GSR** and **eye tracking**.

This repository is deliberately *inference-only*. It contains the network
structure and the objective function of the paper and nothing else: no training
loop, no dataset loader, no experiment scripts, no debug scaffolding. It is
extracted from the research code that produced the published results, and the
released network was verified to be numerically identical to the original model
class (see [Verification](#verification)).

---

## Contents

```
cccdilate/
    __init__.py     public API
    encoders.py     SpatialFeatureExtractor, TemporalFeatureExtractor, PositionalEncoding
    attention.py    GatedMultiHeadCrossAttention, CrossModalFusionBlock (CAG)
    model.py        CCCDilateNet — the full tri-modal network
    losses.py       CCCLoss, MAECCCLoss, DILATE, CCCDILATELoss
demo.py             end-to-end inference demo (real sample or --synthetic)
assets/
    MAHNOB_LOSO_1.pth            released checkpoint (18,526,225 B, md5 2c93477a…)
    MAHNOB_norm.csv              normalisation statistics (20,526 B, md5 5b0b4198…)
    MAHNOB_LOSO_1_sample.npz     complete held-out P01 split (96,246,593 B, md5 ba4e4344…)
requirements.txt
LICENSE
```

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` lists only `torch` and `numpy` — the two packages that
`cccdilate/` and `demo.py` actually import. The original research pipeline
additionally used `pandas`, `scipy`, `scikit-learn`, `mne-features`, `torcheeg`,
`matplotlib`, `tqdm` and `numba` for data preparation and training; none of them
is needed for inference or for the DILATE objective as released here.

## Run the demo

```bash
python demo.py                # bundled held-out MAHNOB subject P01 windows
python demo.py --synthetic    # random tensors of the same shape (no data assets)
```

Both modes run in a few seconds on CPU and read only files inside this
repository. `--synthetic` performs exactly the same forward pass on random
tensors of the same shape, so it is useful as a dependency-free smoke test.

The normalisation statistics are an explicit *input* of the demo rather than
something buried in the weights:

```bash
python demo.py --norm assets/MAHNOB_norm.csv \
               --sample assets/MAHNOB_LOSO_1_sample.npz \
               --checkpoint assets/MAHNOB_LOSO_1.pth
```

`demo.py` reconstructs the model input from the raw features with
`(features_raw - eeg_mean) / eeg_scale`, prints the maximum absolute deviation
from the bundled `scaled_checksum` window — `3.05e-05` with the shipped files —
and aborts if the deviation exceeds `1e-4`. The residual comes from float32
rounding of the stored raw features and of the scaled result, amplified where
`eeg_scale` is small (the scales span `2.3e-04` to `592`), so the threshold leaves
roughly a 30× margin while a wrong statistics file is off by orders of magnitude.
Substituting a different statistics file therefore fails loudly instead of
silently changing the input distribution.

## Normalisation contract

**Every input column and every target is normalised with the shipped statistics.
Anyone reproducing the paper's numbers must apply exactly these.**

`assets/MAHNOB_norm.csv` is a plain CSV with the fixed header

```
group,index,mean,scale
```

and 402 data rows: 400 rows with `group=eeg` (one per input feature column,
`index` 0-399) followed by 2 rows with `group=label` (`index` 0 = valence,
1 = arousal). The file begins and ends like this

```
group,index,mean,scale
eeg,0,-0.00039019342736318417,0.3149272068549363
eeg,1,-0.00024487833322921676,0.30908702381091951
eeg,2,0.00015873465427530744,0.30757184586665431
...
eeg,399,5.7762039450981992e-06,0.67002783083305273
label,0,-0.011225347744048301,0.92710046039981953
label,1,-0.013844164284793859,0.94278025112448616
```

Values are written with `%.17g` so that they round-trip float64 exactly. Rows are
parsed by `index` within each `group`, so a user can rebuild the two vectors
without relying on file order. A column is normalised as

```
x_normalised = (x_raw - mean) / scale
```

The 400 input columns are the flattened 100 channels × 4 packed sub-samples of a
frame, in the order produced by the preprocessing: EEG 64 (32 electrodes + 32
gamma envelopes), then GSR 18, then eye 18 — each channel contributing its four
sub-samples consecutively. The statistics are the `mean_` / `scale_` of a
`sklearn.preprocessing.StandardScaler` fit on the full 30-subject cohort feature
matrix (1286130 rows × 400 columns), so they are cohort-level, not per-subject.

Targets are built in three stages, exactly as in the research pipeline:

1. **per-subject z-scoring** — each subject's valence/arousal series is
   standardised with that subject's own mean and standard deviation;
2. **clipping** to ±3 standard deviations of the pooled, per-subject-standardised
   labels;
3. **global `StandardScaler`** over the pooled labels — its `mean_` / `scale_` are
   the two `label` rows of the CSV.

The research script also contains an optional zero-phase band-pass stage for the
labels, but that block is commented out in the version that produced this
checkpoint, so **no band-pass was applied**. `label_mean` / `label_scale` are
shipped so that predictions can be mapped back to the original rating scale:
`y_raw = y_pred * label_scale + label_mean`.

## Model

`CCCDilateNet` maps a window of tri-modal features `[B, S, C*4]` to one
(valence, arousal) pair per stored frame, `[B, S, 2]`.

```
x → per-modality CNN encoder → positional encoding → gated cross-modal attention
  (6 directions) → contextual-aware gate → channel attention → linear fusion
  → TransformerEncoder → regression head
```

### Input tensor layout

`4` is not a modality: the preprocessing resamples the 128 Hz physiological
streams (60 Hz for eye tracking) to 120 Hz and then packs four consecutive
samples of every channel into a single stored frame, channel-major
(`ch0.f0, ch0.f1, ch0.f2, ch0.f3, ch1.f0, …`). A stored frame therefore carries
`C*4 = 400` numbers. The first operation of `forward` unfolds that packing,

```
[B, S, C, 4] → permute → [B, S, 4, C] → reshape → [B, S*4, C]
```

recovering the native 120 Hz series with `C` channels per step. Two temporal
blocks then halve the time axis twice, so the output has exactly `S` steps — one
prediction per stored frame.

For the released MAHNOB configuration `C = 64 + 18 + 18 = 100`, split
contiguously as:

| slice | channels | content |
|---|---|---|
| `x[:, :, :64]` | 64 | 32 EEG electrodes + their 32 gamma-band (30–49 Hz) Hilbert envelopes |
| `x[:, :, 64:82]` | 18 | GSR: 1 channel + 5 band envelopes, expanded to position / velocity / acceleration |
| `x[:, :, 82:]` | 18 | eye tracking: 6 gaze and pupil channels expanded to position / velocity / acceleration |

### Gated cross-modal attention (pre-gating)

For a query modality *m* attending to a context modality *n*, with
`d_head = d_model / nhead`:

```
Q = W_q X_m      K = W_k X_n      V = W_v X_n
logits = Q Kᵀ / √d_head
G_raw  = tanh(Q) tanh(K)ᵀ
G      = (G_raw / d_head + 1) / 2          ∈ [0, 1]
A      = softmax(logits ⊙ G)
out    = W_o (A V)
```

Every entry of `G_raw` is a sum of `d_head` products of `tanh` values, so
`G_raw / d_head ∈ [-1, 1]` and the gate is **exactly** bounded in `[0, 1]`: it can
attenuate a logit but never invert its sign. The gate is computed from the same
projected `Q` and `K` as the logits, which is why the mechanism is called
*pre*-gating — no separate gating network is needed, and the gate is conditioned
on the same representation the attention already uses.

### Contextual-aware gate (CAG)

All six ordered modality pairs (`EEG←GSR`, `EEG←EYE`, `GSR←EEG`, `GSR←EYE`,
`EYE←EEG`, `EYE←GSR`) are computed by **independent** attention modules — no
weight sharing between directions. Each attended context `Q̂_{m←n}` is then
fused with the query modality's own representation `Q_m`:

```
G       = ReLU(W_h^G Q̂_{m←n} + W_q^G Q_m)     # how much context to admit
E       = ReLU(W_E Q̂_{m←n})                   # context evidence
C       = LayerNorm(E) ⊙ LayerNorm(G)         # gated context
out     = ReLU(fc_C(C)) + ReLU(fc_Q(Q_m))     # gated context + query residual
```

The CAG parameters are shared by the two directions that share a query modality
(the gate is a property of the query modality), giving six outputs
`(eeg←gsr, eeg←eye, gsr←eeg, gsr←eye, eye←eeg, eye←gsr)` that are concatenated to
`6 · d_modal = 768` channels.

### Fusion and temporal model

The 768-channel concatenation is layer-normalised, passed through single-head
channel attention (`nn.MultiheadAttention(embed_dim=768, num_heads=1)`), projected
to `d_model = 256`, scaled by `√d_model` and fed to a 2-layer
`TransformerEncoder` (`nhead=8`, `dim_feedforward=512`, dropout 0.2, no
additional positional encoding — position information was already injected
per modality). A linear head produces the two regression outputs.

### Released configuration

| setting | value |
|---|---|
| `d_model` | 256 |
| `nhead` | 8 |
| `num_layers` | 2 |
| `dim_feedforward` | 512 |
| `d_modal` (per-modality width) | 128 |
| channels | EEG 64, GSR 18, eye 18 |
| dropout | 0.2 |
| parameters | 4,357,858 |

## Objective: CCC-DILATE

`cccdilate/losses.py` implements the paper's hybrid objective

```
L_CCC-DILATE = L_CCC + λ · L_DILATE
```

* `CCCLoss` — `1 − CCC`, with the batch-global concordance correlation
  coefficient `2·cov(y, ŷ) / (var(y) + var(ŷ) + (ȳ − ŷ̄)²)`, so decorrelation,
  mis-scaling and bias are penalised by a single term.
* `MAECCCLoss` — the same term plus a smooth-L1 loss.
* `DILATE` — the batch-wise DILATE loss
  (Le Guen & Thome, 2019): `α·L_shape + (1−α)·L_temporal`, where `L_shape` is the
  soft-DTW alignment cost and `L_temporal = ⟨A*_γ, Ω⟩ / T²` measures the expected
  temporal displacement of the soft alignment. `Ω[i,j] = (i−j)²`.
* `CCCDILATELoss` — the combination above; `dilate_weight` is `λ` and
  `dilate_alpha` is `α` (0.5 in the paper).

The soft-DTW dynamic programs are executed in NumPy float32 on the CPU, exactly
as in the original implementation, and wrapped in custom autograd `Function`s.
The released port was checked against the original `batchdilate` package (see
[Verification](#verification)).

## Released checkpoint

`assets/MAHNOB_LOSO_1.pth` — the three-modality LOSO model for held-out subject
P01 (fold 1), trained on the remaining subjects, re-exported at its
best-validation epoch.
(MD5 `2c93477a08e68d27137ac308b3b95a40`, 18,526,225 bytes.)

**Provenance — LOSO fold 1, subject P01 held out.** The weights come from the
`crossattn_fold_pcag_CCCMAE_LOSO_1.pth` experiment of the research repository,
produced by the leave-one-subject-out driver
`predict_tf_att_crossattn_pcag_loso_dsh.py`. The relevant lines of that script
are:

```python
for fold in range(fold_start, fold_end):                       # fold 1 -> P01
    patient_name = f'P{fold:02}'
    save_path = 'models/' + patient_name + '/'
    train_index = data_merged.index[data_merged['subject_id'] != fold]   # P02..P30
    val_index   = data_merged.index[data_merged['subject_id'] == fold]   # P01
    ...
    model = EEGTransformer()
    criterion = MAE_CCC_Loss()
    model = train_model(..., fold=fold, use_transformer=True, append_labels=True)
    torch.save(model.state_dict(), f'{save_path}crossattn_fold_pcag_CCCMAE_LOSO_{fold}.pth')
```

Subjects are read in `for itr in range(1, 31)` with `subject_id = itr`, so
`fold = 1` selects **subject P01 as the held-out validation subject** and trains
on subjects P02–P30. The name of the enclosing directory (`models/P01/`) refers to
*the fold*, not to the training data: **the shipped model never saw P01 during
training**, and every window of the bundled sample is a genuine held-out sample.

Training configuration recorded by the script: `SEQ_LEN = 200` stored frames,
`OVERLAP = 0.3` (60 frames of leading context are discarded from the output),
`batch_size = 64`, Adam, `lr = 1e-4`, criterion `MAE_CCC_Loss()`, and the
transformer branch (`use_transformer=True`).

### Held-out performance

**Expected held-out performance of the bundled checkpoint** — measured in float32
over the entire held-out P01 validation split (243 windows, 48,600 scored frames)
and reproduced end to end by `demo.py` through the released code path:

| dimension | CCC | PCC |
|---|---|---|
| valence | 0.5618 | 0.6444 |
| arousal | 0.3760 | 0.4711 |

These match the figures reported for this subject in the paper (valence CCC
0.5618, arousal CCC 0.3759).

### Provenance of the exported weights

The bundled checkpoint is the **best-validation-epoch** model for LOSO fold 1
(subject P01 held out). It was re-exported from the same training run with a
corrected snapshot routine; the training configuration was left byte-identical
(seed 42, `MAE_CCC_Loss`, Adam `lr = 1e-4`, `SEQ_LEN = 200`, overlap 0.3, batch
size 64, static phase then transformer phase, patience 50), and the re-run's best
validation loss of `-0.102405` reproduces the `-0.102404` recorded by the original
run. The trajectory is therefore the same and only the exported artefact differs.

**What the corrected routine fixes.** The research script captured the best epoch
with

```python
best_model_wts = model.state_dict()
```

`nn.Module.state_dict()` returns detached *views* that share storage with the live
parameters, so every optimiser step after the snapshot mutated the "snapshot" in
place. The file written at the end of training was consequently the final-epoch
state — which scores valence CCC 0.3407 and arousal CCC 0.1601 on this split —
rather than the epoch the validation loss had selected. Storing the selected epoch
requires cloning:

```python
best_model_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```

This is a documented reproducibility pitfall that affects the legacy `predict_tf*`
driver family, including the whole `predict_tf_att_crossattn_pcag_loso*` family
that produced this checkpoint and its modality-ablation siblings. The newer
`grid_*` / `lam_*` sweep drivers and `clin_sweep.py` already clone the snapshot
correctly and reproduce the numbers they report.

The re-run wrote to a separate output directory; the original research checkpoint
was left untouched.

The same script writes one checkpoint per held-out subject
(`models/P{01..30}/crossattn_fold_pcag_CCCMAE_LOSO_{1..30}.pth`) as well as
modality-ablation variants (`…_LOSO_EEG_{fold}.pth`, `…_GSR_…`, `…_EYE_…`,
`…_EEGGSR_…`, `…_EEGEEYE_…`, `…_EYEGSR_…`). Only the tri-modal fold-1 checkpoint is
bundled here; the others are not part of this release.

### Bundled demo sample

`assets/MAHNOB_LOSO_1_sample.npz` was exported by instrumenting the original LOSO
pipeline and dumping the **complete held-out P01 validation split** of fold 1,
immediately before the model call and before normalisation — so the bundle carries
raw features plus the statistics check described below.

| key | shape | meaning |
|---|---|---|
| `features_raw` | `[243, 260, 400]` | model input windows **before** standardisation |
| `scaled_checksum` | `[260, 400]` | one window after `(x - eeg_mean) / eeg_scale` |
| `checksum_window` | scalar | which bundled window `scaled_checksum` belongs to (0) |
| `valence` | `[243, 200]` | aligned valence target (normalised) |
| `arousal` | `[243, 200]` | aligned arousal target (normalised) |
| `window_indices` | `[243]` | `0 … 242` (all of them) |
| `context_length`, `sequence_length`, `total_length`, `stride` | scalar | `60`, `200`, `260`, `1` |
| `subject`, `fold` | scalar | `'P01'`, `1` |

The 60 leading frames of each window are context only: consecutive windows overlap
by 30 %, so the first 60 output frames of a window are produced from too little
left context and are discarded, leaving 200 scored frames per window. `demo.py`
applies exactly that convention.

All 243 windows are bundled — 48,600 scored frames, the complete inter-subject
evaluation for this fold — so the demo's printed scores **are** the checkpoint's
held-out scores, with no subsetting.

**Expected demo output** (identical to the checkpoint's held-out performance, since
the bundled set is the whole split):

| dimension | CCC | PCC |
|---|---|---|
| valence | 0.5618 | 0.6444 |
| arousal | 0.3760 | 0.4711 |

## Verification

The following checks were run against the original research code; the first two
are reproducible from this repository alone.

1. the shipped checkpoint loads into `CCCDilateNet` with
   `load_state_dict(..., strict=True)` — 184/184 keys, no missing or unexpected
   keys, zero parameter deviation after loading (this is asserted by `demo.py`);
2. `python demo.py` on the bundled held-out sample, and
   `python demo.py --synthetic`;
3. forward-pass equivalence: the released network and the original
   `EEGTransformer` class, loaded with the same checkpoint file, produce outputs
   agreeing to `< 1e-5` (in practice bit-identical) — both on random inputs of
   several shapes and on the real held-out windows;
4. whole-split equivalence: on the entire held-out P01 validation split
   (243 windows, 48,600 scored frames) the two implementations agree with
   `max|diff| = 0.000e+00`, and the labels reconstructed from the instrumented
   pipeline match the original run's CSV to `1.2e-07`;
5. the shipped checkpoint reproduces the original run's own in-training
   best-epoch predictions: `max|diff| = 3.1e-03` for valence and `4.0e-03` for
   arousal, with CCC and PCC of `1.0000` between the two prediction sets. The
   residual is float16: the original predictions were recorded under CUDA
   autocast, the released path runs float32. The run's own validation criterion
   (window-weighted mean of the batch-level `MAE_CCC_Loss`) scores the two sets
   as `-0.102404` and `-0.102438`, a gap of `3.3e-05`;
6. numerical equivalence of `CCCLoss`, `MAECCCLoss`, `DILATE` and
   `CCCDILATELoss` against the original `CCC_Loss`, `MAE_CCC_Loss`,
   `CombinedLoss` and `batchdilate.DTWShpTime`.

## Data availability

The model was developed and evaluated on public affective-computing corpora
(MAHNOB-HCI and the other datasets listed in the paper). These datasets are
available from their original providers under their own terms; the preprocessing
and training code needed to reproduce the experiments will be released with the
final version of the paper.

The clinical cohort used for the paper's clinical validation **cannot be shared**:
the data are covered by the institutional review board approval under which they
were collected and are not redistributable. The clinical configuration is
therefore not part of this release.

## Licence

MIT — see [LICENSE](LICENSE).
