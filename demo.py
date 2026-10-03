#!/usr/bin/env python
"""Reference inference demo for the paper's network.

Temporally Sensitive Neural and Physiological Decoding of Fine-Grained Affective
States Estimated by Continuous Facial Expressions.

The demo loads the bundled checkpoint, rebuilds the model input from **raw**
(un-normalised) held-out features using the shipped normalisation statistics,
checks the reconstruction against a normalised checksum window, runs a forward
pass, reports CCC and Pearson correlation for valence and arousal against the
bundled targets, and writes a prediction-versus-target figure. The bundle holds
the complete held-out P01 split, so those numbers are the checkpoint's held-out
scores.

Usage
-----
    python demo.py                # bundled P01 held-out sample
    python demo.py --synthetic    # random tensors of the same shape (no data assets)

Both modes run in a few seconds on CPU. ``--synthetic`` writes no figure, since
random labels carry no curve worth plotting.

Normalisation
-------------
The released statistics were fit once on the full training cohort and are shipped
in ``assets/MAHNOB_norm.csv``. Pass ``--norm`` with your own file (same format) to
run on your own recordings without access to the original cohort; see the README
for the exact format.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cccdilate import CCCDilateNet, ccc, pcc

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "assets" / "MAHNOB_LOSO_1.pth"
DEFAULT_SAMPLE = REPO_ROOT / "assets" / "MAHNOB_LOSO_1_sample.npz"
DEFAULT_NORM = REPO_ROOT / "assets" / "MAHNOB_norm.csv"

PAPER_TITLE = (
    "Temporally Sensitive Neural and Physiological Decoding of Fine-Grained\n"
    "Affective States Estimated by Continuous Facial Expressions"
)

# Window geometry of the released model. A window holds 260 stored frames; the
# first 60 output frames are discarded because they were produced from too little
# left context (consecutive windows overlap by 30%), leaving 200 scored frames.
# Each stored frame carries 4 sub-samples of 100 channels (see CCCDilateNet).
WINDOW_LEN = 260
CONTEXT_LEN = 60
FRAMES_PER_WINDOW = 200
FEATURES_PER_FRAME = 400
OUTPUT_DIM = 2

# Number of windows in --synthetic mode; the per-window geometry is identical to
# the bundled sample, the count is smaller only to keep the smoke test instant.
SYNTHETIC_WINDOWS = 4

# Max |(raw - mean) / scale - scaled_checksum| accepted for the reconstruction
# check. The bundle stores float32 tensors, so the round trip is exact only up to
# float32 rounding; the measured value is printed. A wrong statistics file is off
# by orders of magnitude more than this.
NORMALISATION_TOLERANCE = 1e-4

# Default Hann smoothing applied to the figure curves, in scored frames.
DEFAULT_SMOOTHING = 301

NORM_GROUPS = ("eeg", "label")
DIMENSIONS = ("valence", "arousal")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="model state_dict (default: %(default)s)",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="bundled feature/target sample (default: %(default)s)",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=DEFAULT_NORM,
        help="normalisation statistics, CSV with columns group,index,mean,scale "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=Path("demo_output.png"),
        help="where to write the prediction-versus-target figure (default: %(default)s)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=DEFAULT_SMOOTHING,
        help="Hann-window length in frames for the figure curves; 1 disables "
        "smoothing (default: %(default)s)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="ignore the bundled assets and run on random tensors of the same shape",
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="seed for --synthetic (default: %(default)s)")
    return parser.parse_args(argv)


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Hann-window moving average that does not droop at the ends.

    The signal is edge-padded by ``window // 2`` samples on both sides and the
    normalised kernel is applied with ``mode="valid"``, so every output sample is
    a convex combination of real neighbours and the first and last values are
    preserved rather than pulled toward zero.
    """
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1  # keep the kernel symmetric about each sample
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_norm(path: Path):
    """Read the normalisation CSV.

    Expected header: ``group,index,mean,scale`` with 400 ``eeg`` rows (one per
    input feature column) and 2 ``label`` rows (valence, arousal).

    Returns:
        ``(eeg_mean, eeg_scale, label_mean, label_scale)``, each ordered by
        ``index``.
    """
    if not path.is_file():
        sys.exit(f"error: normalisation file not found: {path}")

    collected = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"group", "index", "mean", "scale"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            sys.exit(
                f"error: {path} must have columns {sorted(required)}, "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            collected.setdefault(row["group"], []).append(
                (int(row["index"]), float(row["mean"]), float(row["scale"]))
            )

    norms = {}
    for group, rows in collected.items():
        rows.sort()
        indices = [row[0] for row in rows]
        if indices != list(range(len(rows))):
            sys.exit(f"error: group '{group}' indices must be 0..{len(rows) - 1}, got {indices[:5]}...")
        norms[group] = (
            np.array([row[1] for row in rows], dtype=np.float64),
            np.array([row[2] for row in rows], dtype=np.float64),
        )

    for group in NORM_GROUPS:
        if group not in norms:
            sys.exit(f"error: {path} has no rows for group '{group}'")
    if norms["eeg"][0].size != FEATURES_PER_FRAME:
        sys.exit(f"error: expected {FEATURES_PER_FRAME} 'eeg' rows, found {norms['eeg'][0].size}")
    if norms["label"][0].size != OUTPUT_DIM:
        sys.exit(f"error: expected {OUTPUT_DIM} 'label' rows, found {norms['label'][0].size}")

    return norms["eeg"][0], norms["eeg"][1], norms["label"][0], norms["label"][1]


def load_model(checkpoint_path: Path, device: torch.device) -> CCCDilateNet:
    """Instantiate the network and load the released weights (strict)."""
    if not checkpoint_path.is_file():
        sys.exit(f"error: checkpoint not found: {checkpoint_path}")
    model = CCCDilateNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model.to(device).eval()
    return model


def load_sample(sample_path: Path):
    """Return ``(raw, checksum, targets, meta)`` from the bundled .npz."""
    if not sample_path.is_file():
        sys.exit(f"error: sample not found: {sample_path}")
    data = np.load(sample_path, allow_pickle=False)
    raw = torch.from_numpy(data["features_raw"]).float()
    checksum = torch.from_numpy(data["scaled_checksum"]).float()
    targets = torch.stack(
        [torch.from_numpy(data[name]).float() for name in DIMENSIONS],
        dim=-1,
    )
    meta = {
        "subject": str(data["subject"]),
        "fold": int(data["fold"]),
        "windows": data["window_indices"].tolist(),
        "context_length": int(data["context_length"]),
        "checksum_window": int(data["checksum_window"]),
    }
    return raw, checksum, targets, meta


def make_synthetic_sample(k: int = SYNTHETIC_WINDOWS, seed: int = 0):
    """Random tensors with the same per-window geometry as the bundled sample."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(k, WINDOW_LEN, FEATURES_PER_FRAME, generator=generator)
    targets = torch.randn(k, FRAMES_PER_WINDOW, OUTPUT_DIM, generator=generator)
    return raw, targets


def save_figure(true: np.ndarray, pred: np.ndarray, path: Path, window: int) -> bool:
    """Write the target-versus-prediction figure for both dimensions.

    ``true`` and ``pred`` are ``[windows, scored frames, 2]``; each dimension is
    flattened over the concatenated windows, so the x axis runs over the whole
    held-out recording.

    ``matplotlib`` is imported here rather than at module scope so that a missing
    or unusable installation degrades to a warning: the scores are still reported
    and the demo still exits successfully. Returns ``True`` if the figure was
    written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # environment-dependent, e.g. a broken libstdc++
        print(f"figure      : skipped, matplotlib unavailable ({exc})")
        return False

    figure, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    steps = np.arange(true.shape[0] * true.shape[1])

    for index, (name, axis) in enumerate(zip(DIMENSIONS, axes)):
        y_true = true[..., index].ravel()
        y_pred = pred[..., index].ravel()
        axis.plot(steps, smooth_curve(y_true, window), color="#1f77b4",
                  linewidth=1.4, label="target")
        axis.plot(steps, smooth_curve(y_pred, window), color="#d62728",
                  linewidth=1.4, label="prediction")
        axis.set_title(
            f"{name}  (CCC {ccc(y_true, y_pred):.4f}, PCC {pcc(y_true, y_pred):.4f})"
        )
        axis.set_xlabel("scored frame index (concatenated held-out windows)")
        axis.set_ylabel("standardised value")
        axis.legend(loc="upper right", frameon=False)
        axis.grid(alpha=0.25)

    figure.suptitle(
        "Reference inference demo — held-out subject P01 "
        f"(Hann smoothing, window = {window} frames)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    started = time.perf_counter()

    model = load_model(args.checkpoint, device)

    print("=" * 74)
    print(PAPER_TITLE)
    print("Reference inference demo")
    print("=" * 74)
    print(f"checkpoint  : {args.checkpoint}")
    print(f"parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"device      : {device}")
    print()

    if args.synthetic:
        raw, targets = make_synthetic_sample(seed=args.seed)
        model_input = raw
        context_length = CONTEXT_LEN
        print(f"input       : {SYNTHETIC_WINDOWS} synthetic random windows "
              f"(no normalisation applied)")
    else:
        eeg_mean, eeg_scale, label_mean, label_scale = load_norm(args.norm)
        raw, checksum, targets, meta = load_sample(args.sample)
        context_length = meta["context_length"]
        window = meta["checksum_window"]

        print(f"sample      : {args.sample}")
        print(f"  subject   : {meta['subject']} (LOSO fold {meta['fold']}, held out from training)")
        print(f"  windows   : {len(meta['windows'])} held-out windows "
              f"(indices {meta['windows'][0]}..{meta['windows'][-1]})")
        print(f"norm file   : {args.norm}")
        print(f"  eeg stats : {eeg_mean.size} columns, scale in "
              f"[{eeg_scale.min():.4g}, {eeg_scale.max():.4g}]")
        print(f"  labelstats: mean={np.array2string(label_mean, precision=4)} "
              f"scale={np.array2string(label_scale, precision=4)}")
        print()

        # Rebuild the model input from the raw features with the loaded statistics,
        # and prove the statistics are the ones the bundle was normalised with.
        reconstructed = (raw.numpy().astype(np.float64) - eeg_mean) / eeg_scale
        model_input = torch.from_numpy(reconstructed.astype(np.float32))
        deviation = (model_input[window] - checksum).abs().max().item()
        print("normalisation: (features_raw - eeg_mean) / eeg_scale")
        print(f"  check window {window}: max|reconstructed - scaled_checksum| = "
              f"{deviation:.3e}  (tolerance {NORMALISATION_TOLERANCE:.0e})")
        if deviation > NORMALISATION_TOLERANCE:
            sys.exit("error: normalisation round trip failed; wrong statistics file?")
        print()

    model_input = model_input.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        predictions = model(model_input)[:, context_length:, :]

    if predictions.shape != targets.shape:
        sys.exit(
            f"error: prediction shape {tuple(predictions.shape)} != target shape {tuple(targets.shape)}"
        )

    pred = predictions.cpu().numpy()
    true = targets.cpu().numpy()

    print(f"input window: {tuple(model_input.shape)}  "
          f"({WINDOW_LEN - context_length} scored frames per window after discarding "
          f"{context_length} context frames)")
    print(f"predictions : {tuple(predictions.shape)}")
    print(f"targets     : {tuple(targets.shape)}")
    print()
    print(f"{'dimension':<10}{'CCC':>10}{'PCC':>10}")
    print("-" * 30)
    for index, name in enumerate(DIMENSIONS):
        y_true = true[..., index].ravel()
        y_pred = pred[..., index].ravel()
        print(f"{name:<10}{ccc(y_true, y_pred):>10.4f}{pcc(y_true, y_pred):>10.4f}")
    print()

    if args.synthetic:
        print("(labels are random: the correlations above are a smoke test only;")
        print(" no figure is written in --synthetic mode)")
    else:
        print(f"({true.shape[0]} windows = the complete held-out P01 split for this fold,")
        print(f" {true.shape[0] * true.shape[1]} scored frames; no subsetting)")
        if save_figure(true, pred, args.out_fig, args.smooth):
            print(f"figure      : {args.out_fig.resolve()}  "
                  f"(Hann smoothing, window = {args.smooth} frames)")

    print(f"elapsed     : {time.perf_counter() - started:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
