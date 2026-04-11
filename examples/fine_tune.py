"""
Fine-Tune bmshj2018-factorized on Photographer Images (Path C — Domain Adaptation)

The pretrained bmshj2018-factorized model was trained on general-purpose images
(ImageNet, CLIC). Photographer images have a specific statistical distribution —
colour profiles, bokeh blur, skin tones, HDR — that the model has never specialised
for. Fine-tuning continues training on photographer-specific images, nudging the
weights slightly toward this narrower distribution without forgetting what was
already learned.

What changes:   model weights (nudged toward photographer distribution)
What stays:     architecture, inference speed, quality levels

The research question:
    "Does specialising a neural compression model for photographer content
     close the quality gap it has versus AVIF on real photographic images?"

Expected training dataset structure (numbered subdirectories):
    my_images/
        00000/image1.jpg
        01000/image2.jpg
        ...

Usage:
    # Fine-tune at quality levels 1, 3, 6 then evaluate vs pretrained + AVIF:
    python fine_tune.py --train-dir /path/to/my_images --test-dir ./portrait_test_images

    # Quick sanity check (few epochs, few images):
    python fine_tune.py --train-dir /path/to/my_images --test-dir ./portrait_test_images \\
        --epochs 2 --n-source-images 200 --crops-per-image 3

    # GPU run with more training data:
    python fine_tune.py --train-dir /path/to/my_images --test-dir ./portrait_test_images \\
        --device cuda --epochs 15 --n-source-images 5000 --batch-size 16

    # Skip training, just evaluate existing checkpoints:
    python fine_tune.py --test-dir ./portrait_test_images --eval-only

Requirements:
    pip install pytorch-msssim
    pip install pillow-heif   # optional, for AVIF baseline
"""

import argparse
import io
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from compressai.losses import RateDistortionLoss
from compressai.ops import compute_padding
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import bmshj2018_factorized

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# OPTIONAL DEPS
# ─────────────────────────────────────────────────────────────

try:
    from pytorch_msssim import ms_ssim as compute_msssim
    MSSSIM_AVAILABLE = True
except ImportError:
    compute_msssim = None
    MSSSIM_AVAILABLE = False
    print("Note: pytorch-msssim not installed — MS-SSIM metrics will be skipped")

AVIF_AVAILABLE = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    AVIF_AVAILABLE = True
except ImportError:
    try:
        import pillow_avif  # noqa: F401
        AVIF_AVAILABLE = True
    except ImportError:
        print("Note: pillow-heif not installed — AVIF baseline will be skipped")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path(__file__).resolve().parent

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Lambda values that match each pretrained bmshj2018-factorized quality level.
# Using the same lambda during fine-tuning preserves the rate-distortion operating point.
QUALITY_LAMBDA_MAP = {
    1: 0.0018,
    2: 0.0035,
    3: 0.0075,
    4: 0.0150,
    5: 0.0300,
    6: 0.0450,
    7: 0.0900,
    8: 0.1800,
}

# AVIF quality sweep for the traditional baseline curve
AVIF_QUALITIES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

COLORS = {
    "Pretrained":  "#e74c3c",
    "Fine-tuned":  "#2ecc71",
    "AVIF":        "#f39c12",
}
MARKERS = {
    "Pretrained":  "o",
    "Fine-tuned":  "^",
    "AVIF":        "P",
}
LINESTYLES = {
    "Pretrained":  "--",
    "Fine-tuned":  "-",
    "AVIF":        ":",
}


def sync_device(device: str, enabled: bool = True) -> None:
    """Synchronize CUDA for more accurate timing when requested."""
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def print_timing_breakdown(
    title: str,
    timings: Dict[str, float],
    *,
    count: Optional[int] = None,
) -> None:
    total = sum(timings.values())
    print(f"  {title}")
    if total <= 0:
        print("    No timing data collected")
        return

    for label, seconds in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        pct = (seconds / total) * 100 if total > 0 else 0.0
        extra = f" | avg {seconds / count:.4f}s" if count else ""
        print(f"    {label:<20} {format_seconds(seconds):>8} | {pct:5.1f}%{extra}")

    print(f"    {'Total':<20} {format_seconds(total):>8}")

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class PhotographerCropDataset(Dataset):
    """
    Random 256×256 crops from a large nested-directory photographer dataset.

    Scans all subdirectories recursively, randomly selects n_source_images from
    the full collection, then repeats each path crops_per_image times so each
    epoch sees different random crops of the same image.
    """

    def __init__(
        self,
        root: Path,
        n_source_images: int = 3000,
        crops_per_image: int = 5,
        crop_size: int = 256,
        seed: int = 42,
    ):
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

        rng = random.Random(seed)
        scan_start = time.perf_counter()

        print(f"  Scanning {root} for images...", flush=True)
        all_images = [
            p for p in root.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not all_images:
            raise RuntimeError(
                f"No images found in {root}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        print(f"  Found {len(all_images):,} total images")
        print(f"  Scan time: {format_seconds(time.perf_counter() - scan_start)}")

        if len(all_images) > n_source_images:
            sampled = rng.sample(all_images, n_source_images)
        else:
            sampled = list(all_images)
            if len(sampled) < n_source_images:
                print(
                    f"  Warning: only {len(sampled):,} images available "
                    f"(fewer than requested {n_source_images:,})"
                )

        # Each image appears crops_per_image times; each call to __getitem__
        # takes a new random crop, so the epoch sees diverse patches.
        self.samples = sampled * crops_per_image
        rng.shuffle(self.samples)

        print(
            f"  Dataset: {len(sampled):,} images × {crops_per_image} crops/image "
            f"= {len(self.samples):,} patches per epoch"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            # Upscale tiny images rather than skipping them
            if w < self.crop_size or h < self.crop_size:
                scale = math.ceil(self.crop_size / min(w, h))
                img = img.resize((w * scale, h * scale), Image.BILINEAR)
            crop = transforms.RandomCrop(self.crop_size)(img)
            return self.to_tensor(crop)
        except Exception:
            # Return a black patch for corrupt images so the worker doesn't crash
            return torch.zeros(3, self.crop_size, self.crop_size)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def make_optimizers(
    model: nn.Module, lr: float, aux_lr: float
) -> Tuple[optim.Optimizer, optim.Optimizer]:
    conf = {
        "net": {"type": "Adam", "lr": lr},
        "aux": {"type": "Adam", "lr": aux_lr},
    }
    opts = net_aux_optimizer(model, conf)
    return opts["net"], opts["aux"]


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    epoch: int,
    clip_max_norm: float = 1.0,
    timing_breakdown: bool = False,
) -> Dict:
    model.train()
    device = next(model.parameters()).device
    device_type = getattr(device, "type", str(device))

    sum_loss = sum_bpp = sum_mse = 0.0
    n_batches = 0
    timing_totals = {
        "data_wait": 0.0,
        "to_device": 0.0,
        "forward_loss": 0.0,
        "main_backward_step": 0.0,
        "aux_backward_step": 0.0,
    }
    step_start = time.perf_counter()

    for i, x in enumerate(dataloader):
        batch_start = time.perf_counter()
        timing_totals["data_wait"] += batch_start - step_start

        transfer_start = time.perf_counter()
        x = x.to(device, non_blocking=(device_type == "cuda"))
        sync_device(device_type, timing_breakdown)
        timing_totals["to_device"] += time.perf_counter() - transfer_start

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        forward_start = time.perf_counter()
        out_net = model(x)
        out_loss = criterion(out_net, x)
        sync_device(device_type, timing_breakdown)
        timing_totals["forward_loss"] += time.perf_counter() - forward_start

        main_step_start = time.perf_counter()
        out_loss["loss"].backward()

        if clip_max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()
        sync_device(device_type, timing_breakdown)
        timing_totals["main_backward_step"] += time.perf_counter() - main_step_start

        aux_step_start = time.perf_counter()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        sync_device(device_type, timing_breakdown)
        timing_totals["aux_backward_step"] += time.perf_counter() - aux_step_start

        sum_loss += out_loss["loss"].item()
        sum_bpp  += out_loss["bpp_loss"].item()
        sum_mse  += out_loss.get("mse_loss", torch.zeros(1)).item()
        n_batches += 1

        if i % 100 == 0:
            pct = 100.0 * i / len(dataloader)
            print(
                f"    Epoch {epoch} "
                f"[{i * len(x):6d}/{len(dataloader.dataset):6d} ({pct:3.0f}%)] "
                f"Loss: {out_loss['loss'].item():.4f}  "
                f"BPP: {out_loss['bpp_loss'].item():.4f}  "
                f"Aux: {aux_loss.item():.4f}",
                flush=True,
            )

        step_start = time.perf_counter()

    if timing_breakdown and n_batches:
        print_timing_breakdown(
            f"Epoch {epoch} timing breakdown ({n_batches} batches)",
            timing_totals,
            count=n_batches,
        )

    return {
        "loss": sum_loss / n_batches,
        "bpp":  sum_bpp  / n_batches,
        "mse":  sum_mse  / n_batches,
        "timings": timing_totals,
    }


def fine_tune_quality(
    quality: int,
    train_dir: Path,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    n_source_images: int,
    crops_per_image: int,
    lr: float,
    aux_lr: float,
    device: str,
    num_workers: int,
    seed: int,
    timing_breakdown: bool = False,
) -> Path:
    """Fine-tune bmshj2018-factorized at one quality level. Returns checkpoint path."""

    lmbda = QUALITY_LAMBDA_MAP[quality]
    out_path = checkpoint_dir / f"q{quality}_finetuned.pth"

    print(f"\n{'='*60}")
    print(f"  Fine-tuning  quality={quality}  λ={lmbda}  LR={lr:.0e}")
    print(f"{'='*60}")

    model = bmshj2018_factorized(quality=quality, pretrained=True).to(device)

    dataset = PhotographerCropDataset(
        root=train_dir,
        n_source_images=n_source_images,
        crops_per_image=crops_per_image,
        crop_size=256,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    criterion = RateDistortionLoss(lmbda=lmbda)
    optimizer, aux_optimizer = make_optimizers(model, lr, aux_lr)
    # ReduceLROnPlateau with patience=2 lets the LR drop if training stalls
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            model,
            criterion,
            loader,
            optimizer,
            aux_optimizer,
            epoch,
            timing_breakdown=timing_breakdown,
        )
        elapsed = time.time() - t0
        lr_scheduler.step(metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch}/{epochs}  {elapsed:.0f}s  "
            f"Loss: {metrics['loss']:.4f}  BPP: {metrics['bpp']:.4f}  "
            f"LR: {current_lr:.1e}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "quality":    quality,
                    "lmbda":      lmbda,
                    "epoch":      epoch,
                    "best_loss":  best_loss,
                    "state_dict": model.state_dict(),
                },
                out_path,
            )
            print(f"  -> Best checkpoint saved: {out_path}")

    return out_path


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse == 0 else 20 * math.log10(1.0) - 10 * math.log10(mse)


def evaluate_neural(
    model: nn.Module,
    images: List[Path],
    device: str,
    timing_breakdown: bool = False,
) -> Dict:
    """Compress each test image with a neural model; return averaged metrics."""
    model.eval()
    model.update()

    bpps: List[float] = []
    psnrs: List[float] = []
    ms_ssims: List[float] = []
    timing_totals = {
        "load_to_tensor": 0.0,
        "pad": 0.0,
        "compress_decompress": 0.0,
        "metrics": 0.0,
    }

    for i, img_path in enumerate(images, 1):
        try:
            load_start = time.perf_counter()
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            h, w = x.shape[2], x.shape[3]
            sync_device(device, timing_breakdown)
            timing_totals["load_to_tensor"] += time.perf_counter() - load_start

            pad_start = time.perf_counter()
            pad, unpad = compute_padding(h, w, min_div=64)
            x_padded = F.pad(x, pad)
            sync_device(device, timing_breakdown)
            timing_totals["pad"] += time.perf_counter() - pad_start

            codec_start = time.perf_counter()
            with torch.inference_mode():
                out_enc = model.compress(x_padded)
                compressed_bytes = sum(
                    len(s) for sublist in out_enc["strings"] for s in sublist
                )
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)
            sync_device(device, timing_breakdown)
            timing_totals["compress_decompress"] += time.perf_counter() - codec_start

            metrics_start = time.perf_counter()
            bpp  = (compressed_bytes * 8) / (h * w)
            mse  = F.mse_loss(x.cpu(), x_hat.cpu()).item()
            psnr = psnr_from_mse(mse)
            bpps.append(bpp)
            psnrs.append(psnr)

            if MSSSIM_AVAILABLE:
                ms_val = compute_msssim(x.cpu(), x_hat.cpu(), data_range=1.0).item()
                ms_ssims.append(float(ms_val))
            timing_totals["metrics"] += time.perf_counter() - metrics_start

            print(
                f"    [{i:2d}/{len(images)}] {img_path.name} | "
                f"BPP: {bpp:.4f} | PSNR: {psnr:.2f} dB"
            )

        except Exception as e:
            print(f"    [{i:2d}/{len(images)}] {img_path.name} FAILED: {e}")

    if not bpps:
        return {"avg_bpp": None, "avg_psnr": None, "avg_ms_ssim": None}

    if timing_breakdown:
        print_timing_breakdown(
            f"Neural evaluation timing ({len(bpps)} images)",
            timing_totals,
            count=len(bpps),
        )

    return {
        "avg_bpp":     round(float(np.mean(bpps)),     4),
        "avg_psnr":    round(float(np.mean(psnrs)),    2),
        "avg_ms_ssim": round(float(np.mean(ms_ssims)), 4) if ms_ssims else None,
        "timings": timing_totals,
    }


def run_avif_curve(images: List[Path], timing_breakdown: bool = False) -> List[Dict]:
    """Evaluate AVIF across all quality levels; return one dict per quality."""
    if not AVIF_AVAILABLE:
        print("  AVIF not available — skipping traditional baseline")
        return []

    print(f"\n  AVIF baseline ({len(AVIF_QUALITIES)} quality levels)...")
    entries = []
    total_timing = {
        "load_to_tensor": 0.0,
        "encode_decode": 0.0,
        "metrics": 0.0,
    }
    total_images = 0

    for q in AVIF_QUALITIES:
        bpps: List[float] = []
        psnrs: List[float] = []
        ms_ssims: List[float] = []
        quality_timing = {
            "load_to_tensor": 0.0,
            "encode_decode": 0.0,
            "metrics": 0.0,
        }

        for img_path in images:
            try:
                load_start = time.perf_counter()
                img = Image.open(img_path).convert("RGB")
                h, w = img.size[1], img.size[0]
                original = transforms.ToTensor()(img).unsqueeze(0)
                quality_timing["load_to_tensor"] += time.perf_counter() - load_start

                codec_start = time.perf_counter()
                buf = io.BytesIO()
                img.save(buf, format="AVIF", quality=q)
                compressed_bytes = buf.tell()
                buf.seek(0)

                reconstructed_img = Image.open(buf).convert("RGB")
                x_hat = transforms.ToTensor()(reconstructed_img).unsqueeze(0)
                quality_timing["encode_decode"] += time.perf_counter() - codec_start

                metrics_start = time.perf_counter()
                bpp  = (compressed_bytes * 8) / (h * w)
                mse  = F.mse_loss(original, x_hat).item()
                bpps.append(bpp)
                psnrs.append(psnr_from_mse(mse))

                if MSSSIM_AVAILABLE:
                    ms_ssims.append(
                        float(compute_msssim(original, x_hat, data_range=1.0).item())
                    )
                quality_timing["metrics"] += time.perf_counter() - metrics_start
            except Exception as e:
                print(f"    {img_path.name} AVIF q={q} failed: {e}")

        if not bpps:
            continue

        entry = {
            "quality":     q,
            "avg_bpp":     round(float(np.mean(bpps)),     4),
            "avg_psnr":    round(float(np.mean(psnrs)),    2),
            "avg_ms_ssim": round(float(np.mean(ms_ssims)), 4) if ms_ssims else None,
            "timings":     quality_timing,
        }
        entries.append(entry)
        total_images += len(bpps)
        for key, value in quality_timing.items():
            total_timing[key] += value
        print(
            f"    q={q:3d} | BPP: {entry['avg_bpp']:.4f} | "
            f"PSNR: {entry['avg_psnr']:.2f} dB"
        )
        if timing_breakdown:
            print_timing_breakdown(
                f"AVIF timing for q={q}",
                quality_timing,
                count=len(bpps),
            )

    if timing_breakdown and total_images:
        print_timing_breakdown(
            "AVIF timing breakdown (all qualities)",
            total_timing,
            count=total_images,
        )

    return entries


# ─────────────────────────────────────────────────────────────
# BD-RATE
# ─────────────────────────────────────────────────────────────

def bd_rate(
    rate1: List[float],
    metric1: List[float],
    rate2: List[float],
    metric2: List[float],
) -> Optional[float]:
    """Bjontegaard Delta Rate — negative means codec2 needs fewer bits."""
    if len(rate1) < 2 or len(rate2) < 2:
        return None

    pts1 = sorted(zip(metric1, rate1))
    pts2 = sorted(zip(metric2, rate2))
    m1, r1 = zip(*pts1)
    m2, r2 = zip(*pts2)

    min_m = max(min(m1), min(m2))
    max_m = min(max(m1), max(m2))
    if max_m <= min_m:
        return None

    deg = min(3, len(r1) - 1, len(r2) - 1)
    if deg < 1:
        return None

    log_r1 = np.log(np.asarray(r1))
    log_r2 = np.log(np.asarray(r2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = np.polyfit(m1, log_r1, deg)
        p2 = np.polyfit(m2, log_r2, deg)

    i1 = np.polyint(p1)
    i2 = np.polyint(p2)

    v1 = np.polyval(i1, max_m) - np.polyval(i1, min_m)
    v2 = np.polyval(i2, max_m) - np.polyval(i2, min_m)

    avg_diff = (v2 - v1) / (max_m - min_m)
    return float((np.exp(avg_diff) - 1) * 100)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_rd_curves(
    neural_results: Dict[str, List[Dict]],
    avif_entries: List[Dict],
    output_path: str,
) -> None:
    """
    Plot BPP vs PSNR (and optionally MS-SSIM) for pretrained, fine-tuned, and AVIF.

    neural_results: {"Pretrained": [{"quality":q,"avg_bpp":b,"avg_psnr":p,...}, ...],
                     "Fine-tuned": [...]}
    avif_entries:   [{"quality":q,"avg_bpp":b,"avg_psnr":p,...}, ...]
    """
    n_subplots = 2 if MSSSIM_AVAILABLE else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 6))
    if n_subplots == 1:
        axes = [axes]

    def _plot_series(ax, entries, label, key):
        valid = [e for e in entries if e.get(key) is not None and e.get("avg_bpp") is not None]
        if not valid:
            return
        valid = sorted(valid, key=lambda e: e["avg_bpp"])
        bpps   = [e["avg_bpp"] for e in valid]
        values = [e[key] for e in valid]
        ax.plot(
            bpps, values,
            color=COLORS.get(label, "black"),
            marker=MARKERS.get(label, "o"),
            linestyle=LINESTYLES.get(label, "-"),
            linewidth=2.5,
            markersize=8,
            label=label,
        )

    # PSNR
    ax_psnr = axes[0]
    for label, entries in neural_results.items():
        _plot_series(ax_psnr, entries, label, "avg_psnr")
    _plot_series(ax_psnr, avif_entries, "AVIF", "avg_psnr")

    ax_psnr.set_xlabel("Bit-rate [bpp]", fontsize=12)
    ax_psnr.set_ylabel("PSNR [dB]", fontsize=12)
    ax_psnr.set_title(
        "Rate-Distortion: BPP vs PSNR\n"
        "bmshj2018-factorized — Pretrained vs Fine-tuned vs AVIF",
        fontsize=11,
    )
    ax_psnr.legend(fontsize=11)
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.set_xlim(left=0)

    # MS-SSIM
    if MSSSIM_AVAILABLE:
        ax_ms = axes[1]
        for label, entries in neural_results.items():
            _plot_series(ax_ms, entries, label, "avg_ms_ssim")
        _plot_series(ax_ms, avif_entries, "AVIF", "avg_ms_ssim")

        ax_ms.set_xlabel("Bit-rate [bpp]", fontsize=12)
        ax_ms.set_ylabel("MS-SSIM [higher is better]", fontsize=12)
        ax_ms.set_title(
            "Rate-Distortion: BPP vs MS-SSIM\n"
            "bmshj2018-factorized — Pretrained vs Fine-tuned vs AVIF",
            fontsize=11,
        )
        ax_ms.legend(fontsize=11)
        ax_ms.grid(True, alpha=0.3)
        ax_ms.set_xlim(left=0)
        ax_ms.set_ylim(bottom=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  R-D curves saved -> {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def collect_test_images(test_dir: Path, max_images: Optional[int] = None) -> List[Path]:
    images = sorted(
        p for p in test_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No test images found in {test_dir}")
    if max_images:
        images = images[:max_images]
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune bmshj2018-factorized for photographer images (Path C)"
    )
    parser.add_argument(
        "--train-dir", type=Path, default=None,
        help="Root of photographer training dataset (subdirectory structure)",
    )
    parser.add_argument(
        "--test-dir", type=Path,
        default=EXAMPLES_DIR / "portrait_test_images",
        help="Folder of test images for evaluation (default: portrait_test_images/)",
    )
    parser.add_argument(
        "--qualities", type=int, nargs="+", default=[1, 3, 6],
        choices=list(QUALITY_LAMBDA_MAP.keys()),
        help="Quality levels to fine-tune (default: 1 3 6)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Fine-tuning epochs per quality level (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--n-source-images", type=int, default=3000,
        help="Number of training images to sample from train-dir (default: 3000)",
    )
    parser.add_argument(
        "--crops-per-image", type=int, default=5,
        help="Random crops per source image per epoch (default: 5)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Main optimizer learning rate (default: 1e-5). Keep low to avoid forgetting.",
    )
    parser.add_argument(
        "--aux-lr", type=float, default=1e-4,
        help="Auxiliary (entropy model quantile) learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu or cuda (auto-detected by default)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=EXAMPLES_DIR / "fine_tune_checkpoints",
        help="Directory to save/load model checkpoints",
    )
    parser.add_argument(
        "--output-json", type=str, default="fine_tune_results.json",
        help="Path for raw evaluation results JSON",
    )
    parser.add_argument(
        "--output-rd", type=str, default="fine_tune_rd.png",
        help="Path for the R-D curve comparison plot",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; evaluate existing checkpoints in --checkpoint-dir",
    )
    parser.add_argument(
        "--max-test-images", type=int, default=None,
        help="Limit evaluation to N test images (useful for quick sanity checks)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset sampling (default: 42)",
    )
    parser.add_argument(
        "--timing-breakdown", action="store_true",
        help="Print detailed timing summaries to help identify bottlenecks",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ── Validate arguments ───────────────────────────────────
    if not args.eval_only and args.train_dir is None:
        parser.error("--train-dir is required unless --eval-only is set")

    if not args.test_dir.exists():
        parser.error(f"--test-dir does not exist: {args.test_dir}")

    if not args.eval_only and not args.train_dir.exists():
        parser.error(f"--train-dir does not exist: {args.train_dir}")

    # ── Header ───────────────────────────────────────────────
    print("=" * 60)
    print("PATH C — DOMAIN ADAPTATION VIA FINE-TUNING")
    print("=" * 60)
    print(f"  Model:          bmshj2018-factorized")
    print(f"  Qualities:      {args.qualities}")
    print(f"  Device:         {args.device.upper()}")
    if not args.eval_only:
        print(f"  Train dir:      {args.train_dir}")
        print(f"  Source images:  {args.n_source_images:,}")
        print(f"  Crops/image:    {args.crops_per_image}")
        print(f"  Epochs:         {args.epochs}")
        print(f"  LR:             {args.lr:.0e}  Aux LR: {args.aux_lr:.0e}")
        total_patches = args.n_source_images * args.crops_per_image
        print(f"  Total patches:  {total_patches:,}/epoch")
    print(f"  Test dir:       {args.test_dir}")
    print(f"  Checkpoints:    {args.checkpoint_dir}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")

    # ── Collect test images ──────────────────────────────────
    print(f"\n  Collecting test images from {args.test_dir}...")
    test_images = collect_test_images(args.test_dir, args.max_test_images)
    print(f"  {len(test_images)} test images found")
    if any(p.suffix.lower() in {".jpg", ".jpeg"} for p in test_images):
        print(
            "  Note: test images are JPEG (lossy source). PSNR is measured against\n"
            "  the JPEG original, which is valid for the photographer experiment\n"
            "  but cannot be compared to Kodak/paper benchmarks."
        )

    # ── Fine-tuning loop ─────────────────────────────────────
    finetuned_checkpoints: Dict[int, Path] = {}

    if args.eval_only:
        print("\n  --eval-only: loading existing checkpoints...")
        for q in args.qualities:
            ckpt = args.checkpoint_dir / f"q{q}_finetuned.pth"
            if not ckpt.exists():
                print(f"  Warning: no checkpoint found for quality {q} at {ckpt}")
            else:
                finetuned_checkpoints[q] = ckpt
                print(f"  Found q={q} checkpoint: {ckpt}")
    else:
        for q in args.qualities:
            ckpt_path = fine_tune_quality(
                quality=q,
                train_dir=args.train_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                n_source_images=args.n_source_images,
                crops_per_image=args.crops_per_image,
                lr=args.lr,
                aux_lr=args.aux_lr,
                device=args.device,
                num_workers=args.num_workers,
                seed=args.seed,
                timing_breakdown=args.timing_breakdown,
            )
            finetuned_checkpoints[q] = ckpt_path
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # ── Evaluate both pretrained and fine-tuned ──────────────
    pretrained_results: List[Dict] = []
    finetuned_results:  List[Dict] = []

    for q in args.qualities:
        lmbda = QUALITY_LAMBDA_MAP[q]

        # Pretrained baseline
        print(f"\n{'─'*50}")
        print(f"  Evaluating PRETRAINED quality={q}...")
        print(f"{'─'*50}")
        pretrained_model = bmshj2018_factorized(
            quality=q, pretrained=True
        ).eval().to(args.device)
        pretrained_metrics = evaluate_neural(
            pretrained_model,
            test_images,
            args.device,
            timing_breakdown=args.timing_breakdown,
        )
        pretrained_metrics["quality"] = q
        pretrained_results.append(pretrained_metrics)
        print(
            f"  Pretrained q={q}: BPP={pretrained_metrics['avg_bpp']:.4f}  "
            f"PSNR={pretrained_metrics['avg_psnr']:.2f} dB"
        )
        del pretrained_model
        if args.device == "cuda":
            torch.cuda.empty_cache()

        # Fine-tuned model
        if q not in finetuned_checkpoints:
            print(f"  No fine-tuned checkpoint for quality={q} — skipping")
            continue

        print(f"\n{'─'*50}")
        print(f"  Evaluating FINE-TUNED quality={q}...")
        print(f"{'─'*50}")
        finetuned_model = bmshj2018_factorized(quality=q, pretrained=False).to(args.device)
        ckpt = torch.load(finetuned_checkpoints[q], map_location=args.device)
        finetuned_model.load_state_dict(ckpt["state_dict"])
        finetuned_model.eval()

        finetuned_metrics = evaluate_neural(
            finetuned_model,
            test_images,
            args.device,
            timing_breakdown=args.timing_breakdown,
        )
        finetuned_metrics["quality"] = q
        finetuned_results.append(finetuned_metrics)

        delta_psnr = (finetuned_metrics["avg_psnr"] or 0) - (pretrained_metrics["avg_psnr"] or 0)
        delta_bpp  = (finetuned_metrics["avg_bpp"]  or 0) - (pretrained_metrics["avg_bpp"]  or 0)
        print(
            f"  Fine-tuned q={q}: BPP={finetuned_metrics['avg_bpp']:.4f}  "
            f"PSNR={finetuned_metrics['avg_psnr']:.2f} dB  "
            f"(ΔPSNR={delta_psnr:+.2f} dB  ΔBPP={delta_bpp:+.4f})"
        )
        del finetuned_model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── AVIF baseline ────────────────────────────────────────
    avif_entries = run_avif_curve(
        test_images,
        timing_breakdown=args.timing_breakdown,
    )

    # ── Save results JSON ────────────────────────────────────
    all_results = {
        "pretrained": pretrained_results,
        "finetuned":  finetuned_results,
        "avif":       avif_entries,
        "config": {
            "qualities":       args.qualities,
            "epochs":          args.epochs,
            "lr":              args.lr,
            "aux_lr":          args.aux_lr,
            "n_source_images": args.n_source_images,
            "crops_per_image": args.crops_per_image,
            "n_test_images":   len(test_images),
            "device":          args.device,
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved -> {args.output_json}")

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Pretrained vs Fine-tuned vs AVIF")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Quality':<9} {'BPP':<8} {'PSNR (dB)':<12} {'MS-SSIM'}")
    print(f"  {'─'*58}")

    for entry in pretrained_results:
        q    = entry["quality"]
        bpp  = entry["avg_bpp"]  or float("nan")
        psnr = entry["avg_psnr"] or float("nan")
        ms   = f"{entry['avg_ms_ssim']:.4f}" if entry.get("avg_ms_ssim") else "n/a"
        print(f"  {'Pretrained':<22} q={q:<7} {bpp:<8.4f} {psnr:<12.2f} {ms}")

    print(f"  {'─'*58}")
    for entry in finetuned_results:
        q    = entry["quality"]
        bpp  = entry["avg_bpp"]  or float("nan")
        psnr = entry["avg_psnr"] or float("nan")
        ms   = f"{entry['avg_ms_ssim']:.4f}" if entry.get("avg_ms_ssim") else "n/a"
        print(f"  {'Fine-tuned':<22} q={q:<7} {bpp:<8.4f} {psnr:<12.2f} {ms}")

    if avif_entries:
        print(f"  {'─'*58}")
        for entry in avif_entries:
            bpp  = entry["avg_bpp"]
            psnr = entry["avg_psnr"]
            ms   = f"{entry['avg_ms_ssim']:.4f}" if entry.get("avg_ms_ssim") else "n/a"
            print(
                f"  {'AVIF':<22} q={entry['quality']:<7} "
                f"{bpp:<8.4f} {psnr:<12.2f} {ms}"
            )

    # ── BD-Rate summary ──────────────────────────────────────
    pre_bpp   = [e["avg_bpp"]  for e in pretrained_results if e.get("avg_bpp")]
    pre_psnr  = [e["avg_psnr"] for e in pretrained_results if e.get("avg_psnr")]
    ft_bpp    = [e["avg_bpp"]  for e in finetuned_results  if e.get("avg_bpp")]
    ft_psnr   = [e["avg_psnr"] for e in finetuned_results  if e.get("avg_psnr")]
    avif_bpp  = [e["avg_bpp"]  for e in avif_entries        if e.get("avg_bpp")]
    avif_psnr = [e["avg_psnr"] for e in avif_entries        if e.get("avg_psnr")]

    if len(pre_bpp) >= 2 and len(ft_bpp) >= 2:
        delta = bd_rate(pre_bpp, pre_psnr, ft_bpp, ft_psnr)
        if delta is not None:
            direction = "better" if delta < 0 else "worse"
            print(f"\n  BD-Rate (Fine-tuned vs Pretrained): {delta:+.1f}%")
            print(
                f"  Fine-tuned uses {abs(delta):.1f}% {'fewer' if delta < 0 else 'more'} "
                f"bits than pretrained at matched PSNR — {direction}"
            )

    if len(pre_bpp) >= 2 and len(avif_bpp) >= 2:
        delta = bd_rate(avif_bpp, avif_psnr, pre_bpp, pre_psnr)
        if delta is not None:
            print(f"  BD-Rate (Pretrained vs AVIF):       {delta:+.1f}%")

    if len(ft_bpp) >= 2 and len(avif_bpp) >= 2:
        delta = bd_rate(avif_bpp, avif_psnr, ft_bpp, ft_psnr)
        if delta is not None:
            print(f"  BD-Rate (Fine-tuned vs AVIF):       {delta:+.1f}%")

    # ── Plot ─────────────────────────────────────────────────
    print()
    plot_rd_curves(
        neural_results={
            "Pretrained": pretrained_results,
            "Fine-tuned": finetuned_results,
        },
        avif_entries=avif_entries,
        output_path=args.output_rd,
    )

    print(f"""
{'='*60}
WHAT THE RESULTS MEAN
{'='*60}

fine_tune_rd.png
  BPP vs PSNR curves for pretrained, fine-tuned, and AVIF.
  Fine-tuned curve above/left of pretrained = domain adaptation worked.
  Gap to AVIF shows how much neural compression still needs to improve.

BD-Rate interpretation:
  Negative = fine-tuned needs fewer bits than pretrained at same quality.
  -5% means you save 5% of bitrate for the same PSNR.

If fine-tuned ≈ pretrained:
  The dataset may be too similar to the original training set,
  or more epochs / lower lambda may be needed.

Thesis takeaway:
  This experiment quantifies whether specialist training is worth
  the engineering cost relative to simply using AVIF on photographer images.
{'='*60}
""")


if __name__ == "__main__":
    main()
