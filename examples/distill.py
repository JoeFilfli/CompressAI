"""
Knowledge Distillation for Learned Image Compression
(bmshj2018-factorized and mbt2018 students, stronger teachers)

Distillation flow:
    1. A stronger pretrained teacher (frozen, eval mode) runs g_a(x) on each
       batch to produce y_teacher.
    2. The student runs its full forward pass, producing its own y_student,
       reconstruction, and rate estimates.
    3. A 1x1 conv adapter projects y_student from student's channel count to
       the teacher's channel count (identity when they already match). The
       adapter is trained jointly with the student.
    4. Distillation loss = MSE(adapter(y_student), y_teacher)
       Total       loss = RD_loss + alpha * distill_loss

This pushes the student's latent representation toward the teacher's richer
latent while keeping the student's compression objective (rate + distortion)
intact.

Before/after analysis:
    * RD curve: BPP vs PSNR for each student, pretrained vs distilled.
    * Runtime: average encode/decode ms per codec (averaged over qualities),
      drawn as a grouped bar chart (log y-axis when spread is >10x).

Output:
    distill_checkpoints/<student>_q{q}_distilled_from_<teacher>.pth
    distillation_results.json
    distillation_rd_curves.png
    distillation_runtime.png

Supported students:
    bmshj2018-factorized  (pretrained zoo model)
    mbt2018               (pretrained zoo model)
    tiny-hyperprior       (custom small ScaleHyperprior, N=64 M=96, no pretrained)

Supported teachers:
    bmshj2018-factorized, mbt2018, mbt2018-mean, cheng2020-anchor, cheng2020-attn

Usage examples:
    # --- Smoke test: tiny-hyperprior with bmshj teacher ---
    python examples/distill.py \\
        --students tiny-hyperprior \\
        --teacher-for-tiny-hyperprior bmshj2018-factorized \\
        --train-dir examples/high_quality_images \\
        --test-dir examples/kodak \\
        --qualities 3 --epochs 1 \\
        --n-source-images 50 --crops-per-image 2 \\
        --batch-size 4 --num-workers 2

    # --- Smoke test: tiny-hyperprior with cheng teacher ---
    python examples/distill.py \\
        --students tiny-hyperprior \\
        --teacher-for-tiny-hyperprior cheng2020-anchor \\
        --train-dir examples/high_quality_images \\
        --test-dir examples/kodak \\
        --qualities 3 --epochs 1 \\
        --n-source-images 50 --crops-per-image 2 \\
        --batch-size 4 --num-workers 2

    # --- Main teacher comparison: bmshj2018-factorized as teacher ---
    python examples/distill.py \\
        --students tiny-hyperprior \\
        --teacher-for-tiny-hyperprior bmshj2018-factorized \\
        --train-dir examples/high_quality_images \\
        --test-dir examples/kodak \\
        --qualities 3 6 --epochs 10 \\
        --n-source-images 5000 --crops-per-image 4 \\
        --batch-size 8

    # --- Main teacher comparison: cheng2020-anchor as teacher ---
    python examples/distill.py \\
        --students tiny-hyperprior \\
        --teacher-for-tiny-hyperprior cheng2020-anchor \\
        --train-dir examples/high_quality_images \\
        --test-dir examples/kodak \\
        --qualities 3 6 --epochs 10 \\
        --n-source-images 5000 --crops-per-image 4 \\
        --batch-size 8

    # --- Combined plot after both runs (reads all checkpoints) ---
    python examples/distill.py \\
        --students tiny-hyperprior \\
        --test-dir examples/kodak \\
        --qualities 3 6 --eval-only

    # --- Only distill one built-in student ---
    python examples/distill.py \\
        --students bmshj2018-factorized \\
        --teacher-for-bmshj2018-factorized mbt2018 \\
        --train-dir /path/to/my_images --test-dir examples/kodak
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required on headless HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.losses import RateDistortionLoss
from compressai.ops import compute_padding
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

# Reuse helpers from fine_tune.py (same directory) rather than duplicating them.
from fine_tune import (
    PhotographerCropDataset,
    QUALITY_LAMBDA_MAP,
    bd_rate,
    collect_test_images,
    collect_training_images,
    psnr_from_mse,
    split_train_validation_images,
    sync_device,
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from pytorch_msssim import ms_ssim as compute_msssim
    MSSSIM_AVAILABLE = True
except ImportError:
    compute_msssim = None
    MSSSIM_AVAILABLE = False
    print("Note: pytorch-msssim not installed — MS-SSIM will be skipped")


EXAMPLES_DIR = Path(__file__).resolve().parent

# Default teacher per student. User can override via CLI.
DEFAULT_TEACHERS: Dict[str, str] = {
    "bmshj2018-factorized": "mbt2018",
    "mbt2018":              "cheng2020-anchor",
    "tiny-hyperprior":      "bmshj2018-factorized",
}

SUPPORTED_STUDENTS = list(DEFAULT_TEACHERS.keys())
SUPPORTED_TEACHERS = [
    "bmshj2018-factorized",
    "mbt2018",
    "mbt2018-mean",
    "cheng2020-anchor",
    "cheng2020-attn",
]

# Students that have no official pretrained checkpoint. Built-in students
# (bmshj2018-factorized, mbt2018) stay pretrained=True by default; these
# are always built from scratch.
STUDENTS_WITHOUT_PRETRAINED = {"tiny-hyperprior"}

# Highest quality level each teacher actually ships a checkpoint for.
TEACHER_MAX_QUALITY: Dict[str, int] = {
    "cheng2020-anchor": 6,
    "cheng2020-attn":   6,
}

# Stable palette for (student, teacher) distilled curves. Unknown teachers
# get assigned from a fallback palette.
TEACHER_COLOR_CYCLE = [
    "#2ecc71", "#9b59b6", "#1abc9c", "#f39c12", "#16a085", "#c0392b",
]
TEACHER_MARKER_CYCLE = ["^", "D", "v", "P", "X", "*"]
PRETRAINED_COLOR = "#e74c3c"
PRETRAINED_MARKER = "o"
# Reference codecs (pretrained baselines) draw with a muted, dotted style
# so they don't fight with distilled curves visually.
REFERENCE_COLOR_CYCLE = ["#34495e", "#7f8c8d", "#2c3e50", "#95a5a6"]
REFERENCE_MARKER_CYCLE = ["x", "+", "*", "."]
LINESTYLES = {"pretrained": "--", "distilled": "-", "reference": ":"}


# ─────────────────────────────────────────────────────────────
# MODEL / CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def build_model(name: str, quality: int, pretrained: bool) -> nn.Module:
    if name not in image_models:
        raise ValueError(f"Unknown model: {name}")
    # Models with no published checkpoint must always be built from scratch.
    if pretrained and name in STUDENTS_WITHOUT_PRETRAINED:
        pretrained = False
    return image_models[name](quality=quality, pretrained=pretrained)


def distilled_checkpoint_name(student: str, quality: int, teacher: str) -> str:
    return f"{student}_q{quality}_distilled_from_{teacher}.pth"


_CKPT_SEP = "_distilled_from_"


def parse_checkpoint_name(path: Path) -> Optional[Tuple[str, int, str]]:
    """Inverse of distilled_checkpoint_name. Returns (student, quality, teacher) or None."""
    stem = path.name
    if not stem.endswith(".pth") or _CKPT_SEP not in stem:
        return None
    left, teacher_pth = stem.rsplit(_CKPT_SEP, 1)
    teacher = teacher_pth[:-4]  # strip .pth
    if "_q" not in left:
        return None
    student, q_part = left.rsplit("_q", 1)
    try:
        quality = int(q_part)
    except ValueError:
        return None
    return student, quality, teacher


def discover_distilled_checkpoints(
    checkpoint_dir: Path,
    student: str,
    qualities: List[int],
) -> Dict[int, Dict[str, Path]]:
    """Scan checkpoint_dir for every teacher variant of (student, q). Returns
    { quality: { teacher_name: checkpoint_path } }."""
    out: Dict[int, Dict[str, Path]] = {q: {} for q in qualities}
    if not checkpoint_dir.exists():
        return out
    for p in checkpoint_dir.glob(f"{student}_q*{_CKPT_SEP}*.pth"):
        parsed = parse_checkpoint_name(p)
        if parsed is None:
            continue
        s, q, t = parsed
        if s != student or q not in out:
            continue
        out[q][t] = p
    return out


def resolve_teacher_quality(teacher: str, requested: int) -> int:
    max_q = TEACHER_MAX_QUALITY.get(teacher, 8)
    return min(requested, max_q)


def infer_latent_channels(model: nn.Module, device: str) -> int:
    """Push a tiny dummy through model.g_a to discover the y channel count."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, device=device)
            y = model.g_a(dummy)
    finally:
        if was_training:
            model.train()
    return int(y.shape[1])


# ─────────────────────────────────────────────────────────────
# ADAPTER
# ─────────────────────────────────────────────────────────────

class ChannelAdapter(nn.Module):
    """
    1x1 conv that maps student latent channels -> teacher latent channels.
    Identity when the two already match (no parameters added).

    Trained jointly with the student; only used during distillation training.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def configure_distill_optimizers(
    student: nn.Module,
    adapter: nn.Module,
    lr: float,
    aux_lr: float,
) -> Tuple[optim.Optimizer, optim.Optimizer]:
    """Main optimizer covers student "net" params + adapter; aux covers entropy quantiles."""
    conf = {
        "net": {"type": "Adam", "lr": lr},
        "aux": {"type": "Adam", "lr": aux_lr},
    }
    opts = net_aux_optimizer(student, conf)
    net_opt, aux_opt = opts["net"], opts["aux"]

    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
    if adapter_params:
        net_opt.add_param_group({"params": adapter_params, "lr": lr})
    return net_opt, aux_opt


def _capture_student_y(student: nn.Module):
    """
    Context manager-like helper: registers a forward hook on student.g_a that
    appends its output to a list. Use: 'with this:' or as (handle, storage).
    """
    storage: List[torch.Tensor] = []

    def _hook(_module, _inputs, outputs):
        storage.append(outputs)

    handle = student.g_a.register_forward_hook(_hook)
    return handle, storage


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    adapter: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    net_opt: optim.Optimizer,
    aux_opt: optim.Optimizer,
    epoch: int,
    alpha: float,
    clip_max_norm: float,
    device: str,
    log_interval: int = 10,
) -> Dict[str, float]:
    student.train()
    adapter.train()
    teacher.eval()

    is_cuda = str(device).startswith("cuda")
    sum_total = sum_rd = sum_distill = sum_bpp = 0.0
    n_batches = 0
    n_total = len(loader)

    t_epoch_start = time.time()
    t_batch_start = time.time()

    for i, x in enumerate(loader):
        t_data = time.time() - t_batch_start  # time spent waiting for data

        t_compute_start = time.time()
        x = x.to(device, non_blocking=is_cuda)

        net_opt.zero_grad()
        aux_opt.zero_grad()

        handle, captured = _capture_student_y(student)
        try:
            out_student = student(x)
        finally:
            handle.remove()
        y_student = captured[0]

        rd_out = criterion(out_student, x)
        rd_loss = rd_out["loss"]

        with torch.no_grad():
            y_teacher = teacher.g_a(x)

        y_student_proj = adapter(y_student)
        distill_loss = F.mse_loss(y_student_proj, y_teacher)

        total_loss = rd_loss + alpha * distill_loss
        total_loss.backward()

        if clip_max_norm > 0:
            params = list(student.parameters()) + list(adapter.parameters())
            nn.utils.clip_grad_norm_(params, clip_max_norm)
        net_opt.step()

        aux_loss = student.aux_loss()
        aux_loss.backward()
        aux_opt.step()

        if is_cuda:
            torch.cuda.synchronize()
        t_compute = time.time() - t_compute_start

        sum_total   += total_loss.item()
        sum_rd      += rd_loss.item()
        sum_distill += distill_loss.item()
        sum_bpp     += rd_out["bpp_loss"].item()
        n_batches   += 1

        t_batch_total = t_data + t_compute
        elapsed = time.time() - t_epoch_start
        batches_left = n_total - (i + 1)
        eta = batches_left * (elapsed / (i + 1)) if i >= 0 else 0.0

        if i % log_interval == 0:
            pct = 100.0 * (i + 1) / max(n_total, 1)
            print(
                f"    Epoch {epoch} "
                f"[{(i + 1) * len(x):5d}/{len(loader.dataset):5d} ({pct:4.1f}%)] "
                f"Total: {total_loss.item():.4f}  "
                f"RD: {rd_loss.item():.4f}  "
                f"Distill: {distill_loss.item():.4f}  "
                f"BPP: {rd_out['bpp_loss'].item():.4f}  "
                f"Aux: {aux_loss.item():.4f}  "
                f"| data: {t_data*1000:.0f}ms  compute: {t_compute*1000:.0f}ms  "
                f"ETA: {eta:.0f}s",
                flush=True,
            )

        t_batch_start = time.time()

    n = max(n_batches, 1)
    return {
        "loss":         sum_total / n,
        "rd_loss":      sum_rd / n,
        "distill_loss": sum_distill / n,
        "bpp_loss":     sum_bpp / n,
    }


def validate_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    adapter: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    alpha: float,
    device: str,
) -> Dict[str, float]:
    student.eval()
    adapter.eval()
    teacher.eval()

    sum_total = sum_rd = sum_distill = sum_bpp = 0.0
    n_batches = 0

    with torch.inference_mode():
        for x in loader:
            x = x.to(device, non_blocking=str(device).startswith("cuda"))

            handle, captured = _capture_student_y(student)
            try:
                out_student = student(x)
            finally:
                handle.remove()
            y_student = captured[0]

            rd_out = criterion(out_student, x)
            rd_loss = rd_out["loss"]

            y_teacher = teacher.g_a(x)
            y_student_proj = adapter(y_student)
            distill_loss = F.mse_loss(y_student_proj, y_teacher)

            total_loss = rd_loss + alpha * distill_loss

            sum_total   += total_loss.item()
            sum_rd      += rd_loss.item()
            sum_distill += distill_loss.item()
            sum_bpp     += rd_out["bpp_loss"].item()
            n_batches   += 1

    if n_batches == 0:
        raise RuntimeError("Validation loader produced zero batches")

    n = n_batches
    return {
        "loss":         sum_total / n,
        "rd_loss":      sum_rd / n,
        "distill_loss": sum_distill / n,
        "bpp_loss":     sum_bpp / n,
    }


def distill_one_quality(
    student_name: str,
    teacher_name: str,
    quality: int,
    train_images: List[Path],
    val_images: List[Path],
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    n_source_images: int,
    crops_per_image: int,
    val_crops_per_image: int,
    lr: float,
    aux_lr: float,
    alpha: float,
    clip_max_norm: float,
    device: str,
    num_workers: int,
    prefetch_factor: int,
    loader_backend: str,
    seed: int,
) -> Path:
    """Distill one student at one quality; save best-validation checkpoint. Returns its path."""

    teacher_quality = resolve_teacher_quality(teacher_name, quality)
    lmbda = QUALITY_LAMBDA_MAP[quality]
    out_path = checkpoint_dir / distilled_checkpoint_name(student_name, quality, teacher_name)

    print(f"\n{'='*72}")
    print(
        f"  Distilling {student_name} q={quality} "
        f"from {teacher_name} q={teacher_quality} | λ={lmbda} α={alpha}"
    )
    print(f"{'='*72}")

    teacher = build_model(teacher_name, teacher_quality, pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_model(student_name, quality, pretrained=True).to(device)

    t_ch = infer_latent_channels(teacher, device)
    s_ch = infer_latent_channels(student, device)
    adapter = ChannelAdapter(s_ch, t_ch).to(device)
    print(
        f"  Student latent channels: {s_ch} | Teacher latent channels: {t_ch} | "
        f"Adapter: {'Identity' if s_ch == t_ch else f'Conv2d({s_ch}->{t_ch}, 1x1)'}"
    )

    is_cuda = str(device).startswith("cuda")

    train_dataset = PhotographerCropDataset(
        image_paths=train_images,
        n_source_images=n_source_images,
        crops_per_image=crops_per_image,
        crop_size=256,
        seed=seed,
        loader_backend=loader_backend,
        crop_mode="random",
        dataset_label="Train set",
    )
    train_kwargs = {
        "dataset":            train_dataset,
        "batch_size":         batch_size,
        "shuffle":            True,
        "num_workers":        num_workers,
        "pin_memory":         is_cuda,
        "drop_last":          True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(**train_kwargs)

    val_n_sources = min(len(val_images), n_source_images)
    val_dataset = PhotographerCropDataset(
        image_paths=val_images,
        n_source_images=val_n_sources,
        crops_per_image=val_crops_per_image,
        crop_size=256,
        seed=seed,
        loader_backend=loader_backend,
        crop_mode="center",
        dataset_label="Validation set",
    )
    val_kwargs = {
        "dataset":            val_dataset,
        "batch_size":         batch_size,
        "shuffle":            False,
        "num_workers":        num_workers,
        "pin_memory":         is_cuda,
        "drop_last":          False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        val_kwargs["prefetch_factor"] = prefetch_factor
    val_loader = DataLoader(**val_kwargs)

    criterion = RateDistortionLoss(lmbda=lmbda)
    net_opt, aux_opt = configure_distill_optimizers(student, adapter, lr, aux_lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(net_opt, mode="min", factor=0.5, patience=2)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            student, teacher, adapter, criterion,
            train_loader, net_opt, aux_opt,
            epoch, alpha, clip_max_norm, device,
        )
        elapsed = time.time() - t0
        val_metrics = validate_one_epoch(
            student, teacher, adapter, criterion, val_loader, alpha, device,
        )
        lr_scheduler.step(val_metrics["loss"])
        current_lr = net_opt.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch}/{epochs}  {elapsed:.0f}s  "
            f"TrainLoss: {metrics['loss']:.4f}  ValLoss: {val_metrics['loss']:.4f}  "
            f"TrainDistill: {metrics['distill_loss']:.4f}  "
            f"ValDistill: {val_metrics['distill_loss']:.4f}  "
            f"TrainBPP: {metrics['bpp_loss']:.4f}  ValBPP: {val_metrics['bpp_loss']:.4f}  "
            f"LR: {current_lr:.1e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "student":          student_name,
                    "teacher":          teacher_name,
                    "quality":          quality,
                    "teacher_quality":  teacher_quality,
                    "lmbda":            lmbda,
                    "alpha":            alpha,
                    "epoch":            epoch,
                    "best_val_loss":    best_val_loss,
                    "state_dict":       student.state_dict(),
                    "adapter_state":    adapter.state_dict(),
                    "adapter_channels": {"in": s_ch, "out": t_ch},
                },
                out_path,
            )
            print(f"  -> Best checkpoint saved: {out_path}")

    if is_cuda:
        torch.cuda.empty_cache()

    return out_path


# ─────────────────────────────────────────────────────────────
# EVALUATION (RD + runtime)
# ─────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    images: List[Path],
    device: str,
    warmup_iters: int = 1,
    label: str = "",
) -> Optional[Dict]:
    """Compress/decompress each test image; return averaged BPP, PSNR, MS-SSIM, enc/dec ms."""
    model.eval()
    model.update()

    if images and warmup_iters > 0:
        try:
            img = Image.open(images[0]).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            pad, _ = compute_padding(x.shape[2], x.shape[3], min_div=64)
            x_padded = F.pad(x, pad)
            with torch.inference_mode():
                for _ in range(warmup_iters):
                    out_enc = model.compress(x_padded)
                    model.decompress(out_enc["strings"], out_enc["shape"])
            sync_device(device)
        except Exception as exc:
            print(f"  Warmup skipped ({exc})")

    bpps:     List[float] = []
    psnrs:    List[float] = []
    ms_ssims: List[float] = []
    enc_ms:   List[float] = []
    dec_ms:   List[float] = []

    for i, img_path in enumerate(images, 1):
        try:
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            h, w = x.shape[2], x.shape[3]
            pad, unpad = compute_padding(h, w, min_div=64)
            x_padded = F.pad(x, pad)

            with torch.inference_mode():
                sync_device(device)
                t0 = time.perf_counter()
                out_enc = model.compress(x_padded)
                sync_device(device)
                encode_ms = (time.perf_counter() - t0) * 1000

                compressed_bytes = sum(
                    len(s) for sublist in out_enc["strings"] for s in sublist
                )

                sync_device(device)
                t0 = time.perf_counter()
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                sync_device(device)
                decode_ms = (time.perf_counter() - t0) * 1000

                x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)

            bpp = (compressed_bytes * 8) / (h * w)
            mse = F.mse_loss(x.cpu(), x_hat.cpu()).item()

            bpps.append(bpp)
            psnrs.append(psnr_from_mse(mse))
            enc_ms.append(encode_ms)
            dec_ms.append(decode_ms)

            if MSSSIM_AVAILABLE:
                ms_ssims.append(
                    float(compute_msssim(x.cpu(), x_hat.cpu(), data_range=1.0).item())
                )

            print(
                f"    [{i:3d}/{len(images)}] {img_path.name} | "
                f"BPP: {bpp:.4f} | PSNR: {psnrs[-1]:.2f} dB | "
                f"Enc: {encode_ms:.1f}ms | Dec: {decode_ms:.1f}ms"
            )
        except Exception as exc:
            print(f"    [{i:3d}/{len(images)}] {img_path.name} FAILED: {exc}")

    if not bpps:
        return None

    return {
        "label":         label,
        "avg_bpp":       round(float(np.mean(bpps)),     4),
        "avg_psnr":      round(float(np.mean(psnrs)),    2),
        "avg_ms_ssim":   round(float(np.mean(ms_ssims)), 4) if ms_ssims else None,
        "avg_encode_ms": round(float(np.mean(enc_ms)),   1),
        "avg_decode_ms": round(float(np.mean(dec_ms)),   1),
        "n_images":      len(bpps),
    }


def load_distilled_student(
    student_name: str,
    quality: int,
    checkpoint_path: Path,
    device: str,
) -> nn.Module:
    model = build_model(student_name, quality, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def _valid_rd_entries(entries: List[Dict]) -> List[Dict]:
    return sorted(
        [e for e in entries
         if e and e.get("avg_psnr") is not None and e.get("avg_bpp") is not None],
        key=lambda e: e["avg_bpp"],
    )


def _teacher_style(teacher: str, index: int) -> Tuple[str, str]:
    return (
        TEACHER_COLOR_CYCLE[index % len(TEACHER_COLOR_CYCLE)],
        TEACHER_MARKER_CYCLE[index % len(TEACHER_MARKER_CYCLE)],
    )


def plot_rd_curves(
    results: Dict,
    output_path: str,
    reference_results: Optional[Dict[str, List[Dict]]] = None,
) -> None:
    """
    results shape:
        { student: { "pretrained": [entry, ...],
                     "distilled":  { teacher: [entry, ...] } } }
    reference_results shape:
        { codec_name: [entry, ...] }  — drawn on every student subplot with a
        muted dotted line, so a pretrained baseline (e.g. bmshj2018-factorized)
        can be compared side-by-side with each distilled student curve.
    """
    students = list(results.keys())
    n_sub = len(students)
    if n_sub == 0:
        print("  No RD data to plot")
        return
    reference_results = reference_results or {}

    fig, axes = plt.subplots(1, n_sub, figsize=(7 * n_sub, 6))
    if n_sub == 1:
        axes = [axes]

    for ax, student in zip(axes, students):
        states = results[student]
        any_curve = False

        pre_valid = _valid_rd_entries(states.get("pretrained", []))
        if pre_valid:
            ax.plot(
                [e["avg_bpp"]  for e in pre_valid],
                [e["avg_psnr"] for e in pre_valid],
                color=PRETRAINED_COLOR,
                marker=PRETRAINED_MARKER,
                linestyle=LINESTYLES["pretrained"],
                linewidth=2.5,
                markersize=8,
                label=f"pretrained {student}",
            )
            any_curve = True

        distilled = states.get("distilled", {}) or {}
        for idx, teacher in enumerate(sorted(distilled.keys())):
            d_valid = _valid_rd_entries(distilled[teacher])
            if not d_valid:
                continue
            color, marker = _teacher_style(teacher, idx)
            ax.plot(
                [e["avg_bpp"]  for e in d_valid],
                [e["avg_psnr"] for e in d_valid],
                color=color,
                marker=marker,
                linestyle=LINESTYLES["distilled"],
                linewidth=2.5,
                markersize=8,
                label=f"{student} distilled ← {teacher}",
            )
            any_curve = True

        for idx, codec in enumerate(sorted(reference_results.keys())):
            r_valid = _valid_rd_entries(reference_results[codec])
            if not r_valid:
                continue
            ref_color = REFERENCE_COLOR_CYCLE[idx % len(REFERENCE_COLOR_CYCLE)]
            ref_marker = REFERENCE_MARKER_CYCLE[idx % len(REFERENCE_MARKER_CYCLE)]
            ax.plot(
                [e["avg_bpp"]  for e in r_valid],
                [e["avg_psnr"] for e in r_valid],
                color=ref_color,
                marker=ref_marker,
                linestyle=LINESTYLES["reference"],
                linewidth=2.0,
                markersize=9,
                label=f"reference (pretrained {codec})",
            )
            any_curve = True

        ax.set_title(f"{student}\npretrained vs distilled (per teacher)")
        ax.set_xlabel("Bit-rate [bpp]", fontsize=12)
        ax.set_ylabel("PSNR [dB]", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        if any_curve:
            ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  RD curves saved -> {output_path}")


def plot_runtime(
    results: Dict,
    output_path: str,
    reference_results: Optional[Dict[str, List[Dict]]] = None,
) -> None:
    """
    Grouped bar chart of average encode/decode ms per codec, averaged across
    quality levels. Codec = "<student> pretrained", "<student> distilled ←
    <teacher>", or "<codec> (reference)". One pair of bars per codec.
    """
    codec_labels: List[str] = []
    enc_means:    List[float] = []
    dec_means:    List[float] = []
    reference_results = reference_results or {}

    for student, states in results.items():
        pre_entries = states.get("pretrained", [])
        enc = [e["avg_encode_ms"] for e in pre_entries
               if e and e.get("avg_encode_ms") is not None]
        dec = [e["avg_decode_ms"] for e in pre_entries
               if e and e.get("avg_decode_ms") is not None]
        if enc and dec:
            codec_labels.append(f"{student}\npretrained")
            enc_means.append(float(np.mean(enc)))
            dec_means.append(float(np.mean(dec)))

        distilled = states.get("distilled", {}) or {}
        for teacher in sorted(distilled.keys()):
            entries = distilled[teacher]
            enc = [e["avg_encode_ms"] for e in entries
                   if e and e.get("avg_encode_ms") is not None]
            dec = [e["avg_decode_ms"] for e in entries
                   if e and e.get("avg_decode_ms") is not None]
            if not enc or not dec:
                continue
            codec_labels.append(f"{student}\ndistilled ← {teacher}")
            enc_means.append(float(np.mean(enc)))
            dec_means.append(float(np.mean(dec)))

    for codec in sorted(reference_results.keys()):
        entries = reference_results[codec]
        enc = [e["avg_encode_ms"] for e in entries
               if e and e.get("avg_encode_ms") is not None]
        dec = [e["avg_decode_ms"] for e in entries
               if e and e.get("avg_decode_ms") is not None]
        if not enc or not dec:
            continue
        codec_labels.append(f"{codec}\n(reference)")
        enc_means.append(float(np.mean(enc)))
        dec_means.append(float(np.mean(dec)))

    if not codec_labels:
        print("  No runtime data to plot")
        return

    x = np.arange(len(codec_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(codec_labels)), 6))

    bars_enc = ax.bar(x - width / 2, enc_means, width, label="Encode", color="#3498db")
    bars_dec = ax.bar(x + width / 2, dec_means, width, label="Decode", color="#e67e22")

    all_times = enc_means + dec_means
    if min(all_times) > 0 and max(all_times) / min(all_times) > 10:
        ax.set_yscale("log")
        ax.set_ylabel("Average time per image [ms, log scale]", fontsize=12)
    else:
        ax.set_ylabel("Average time per image [ms]", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(codec_labels, fontsize=10)
    ax.set_title(
        "Encode/Decode Runtime per Codec\n(averaged across quality levels)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (bars_enc, bars_dec):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Runtime chart saved -> {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation for learned image compression (RD + runtime analysis)"
    )
    parser.add_argument(
        "--students", nargs="+",
        default=SUPPORTED_STUDENTS,
        choices=SUPPORTED_STUDENTS,
        help="Student models to distill (default: both)",
    )
    parser.add_argument(
        "--teacher-for-bmshj2018-factorized",
        nargs="+",
        default=[DEFAULT_TEACHERS["bmshj2018-factorized"]],
        choices=SUPPORTED_TEACHERS,
        help="One or more teachers for bmshj2018-factorized student "
             "(trains one checkpoint per teacher)",
    )
    parser.add_argument(
        "--teacher-for-mbt2018",
        nargs="+",
        default=[DEFAULT_TEACHERS["mbt2018"]],
        choices=SUPPORTED_TEACHERS,
        help="One or more teachers for mbt2018 student "
             "(trains one checkpoint per teacher)",
    )
    parser.add_argument(
        "--teacher-for-tiny-hyperprior",
        nargs="+",
        default=[DEFAULT_TEACHERS["tiny-hyperprior"]],
        choices=SUPPORTED_TEACHERS,
        help="One or more teachers for tiny-hyperprior student "
             "(e.g. --teacher-for-tiny-hyperprior bmshj2018-factorized cheng2020-anchor)",
    )
    parser.add_argument(
        "--reference-codecs",
        nargs="*",
        default=[],
        choices=list(image_models.keys()),
        help="Extra pretrained codec(s) to evaluate as reference curves on the "
             "RD and runtime plots (e.g. 'bmshj2018-factorized' to show the "
             "pretrained baseline alongside distilled students).",
    )
    parser.add_argument(
        "--train-dir", type=Path, default=None,
        help="Root directory of training images (recursively scanned). Required unless --eval-only.",
    )
    parser.add_argument(
        "--test-dir", type=Path,
        default=EXAMPLES_DIR / "kodak",
        help="Folder of test images for RD/runtime evaluation",
    )
    parser.add_argument(
        "--qualities", type=int, nargs="+", default=[1, 3, 6],
        choices=list(QUALITY_LAMBDA_MAP.keys()),
        help="Quality levels to distill + evaluate (default: 1 3 6)",
    )
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--n-source-images", type=int,   default=5000,
                        help="How many training images to sample per epoch (default: 5000)")
    parser.add_argument("--crops-per-image", type=int,   default=4)
    parser.add_argument("--val-crops-per-image", type=int, default=1)
    parser.add_argument("--val-ratio",       type=float, default=0.2)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--aux-lr",          type=float, default=1e-4)
    parser.add_argument("--alpha",           type=float, default=0.1,
                        help="Distillation loss weight (default: 0.1)")
    parser.add_argument("--clip-max-norm",   type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers",     type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--loader-backend",
        choices=["auto", "torchvision", "pil"],
        default="auto",
    )
    parser.add_argument(
        "--image-manifest", type=Path, default=None,
        help="Optional JSON cache of training image paths (recommended for 70k images)",
    )
    parser.add_argument("--refresh-image-manifest", action="store_true")
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=EXAMPLES_DIR / "distill_checkpoints",
    )
    parser.add_argument(
        "--output-json", type=str,
        default="distillation_results.json",
    )
    parser.add_argument(
        "--output-rd", type=str,
        default="distillation_rd_curves.png",
    )
    parser.add_argument(
        "--output-runtime", type=str,
        default="distillation_runtime.png",
    )
    parser.add_argument("--eval-only",       action="store_true",
                        help="Skip training; evaluate existing distilled checkpoints")
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--seed",            type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    teachers_for: Dict[str, List[str]] = {
        "bmshj2018-factorized": list(args.teacher_for_bmshj2018_factorized),
        "mbt2018":              list(args.teacher_for_mbt2018),
        "tiny-hyperprior":      list(args.teacher_for_tiny_hyperprior),
    }
    # Deduplicate while preserving order.
    for s, ts in teachers_for.items():
        seen = set()
        teachers_for[s] = [t for t in ts if not (t in seen or seen.add(t))]

    if not args.eval_only and args.train_dir is None:
        raise SystemExit("--train-dir is required unless --eval-only is set")
    if not args.test_dir.exists():
        raise SystemExit(f"--test-dir does not exist: {args.test_dir}")
    if not args.eval_only and not args.train_dir.exists():
        raise SystemExit(f"--train-dir does not exist: {args.train_dir}")

    print("=" * 72)
    print("KNOWLEDGE DISTILLATION FOR LEARNED IMAGE COMPRESSION")
    print("=" * 72)
    print(f"  Students:  {args.students}")
    for s in args.students:
        ts = teachers_for[s]
        print(f"    {s:<22} <- teacher(s): {', '.join(ts)}")
    if args.reference_codecs:
        print(f"  Reference:  {', '.join(args.reference_codecs)} (pretrained)")
    print(f"  Qualities:  {args.qualities}")
    print(f"  Device:     {args.device.upper()}")
    print(f"  Test dir:   {args.test_dir}")
    if not args.eval_only:
        print(f"  Train dir:  {args.train_dir}")
        print(f"  Epochs:     {args.epochs}   Batch: {args.batch_size}  "
              f"α: {args.alpha}  LR: {args.lr:.0e}  AuxLR: {args.aux_lr:.0e}")
        print(f"  Workers:    {args.num_workers}  Prefetch: {args.prefetch_factor}")
        print(f"  Source imgs/epoch: {args.n_source_images:,}  "
              f"Crops/img: {args.crops_per_image}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print("=" * 72)

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Test images ────────────────────────────────────────────
    print(f"\n  Collecting test images from {args.test_dir}...")
    test_images = collect_test_images(args.test_dir, args.max_test_images)
    print(f"  {len(test_images)} test images found")

    # ── Distill (unless --eval-only) ───────────────────────────
    # After training, we rescan checkpoint_dir for every teacher variant
    # so the final plots/JSON can show an accumulated multi-teacher view
    # built up across separate invocations.
    if not args.eval_only:
        print()
        all_images = collect_training_images(
            args.train_dir,
            manifest_path=args.image_manifest,
            refresh_manifest=args.refresh_image_manifest,
        )
        train_images, val_images = split_train_validation_images(
            all_images, val_ratio=args.val_ratio, seed=args.seed,
        )
        print(
            f"  Train/val split: {len(train_images):,} train, "
            f"{len(val_images):,} validation"
        )

        for s in args.students:
            for t in teachers_for[s]:
                for q in args.qualities:
                    distill_one_quality(
                        student_name=s,
                        teacher_name=t,
                        quality=q,
                        train_images=train_images,
                        val_images=val_images,
                        checkpoint_dir=args.checkpoint_dir,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        n_source_images=args.n_source_images,
                        crops_per_image=args.crops_per_image,
                        val_crops_per_image=args.val_crops_per_image,
                        lr=args.lr,
                        aux_lr=args.aux_lr,
                        alpha=args.alpha,
                        clip_max_norm=args.clip_max_norm,
                        device=args.device,
                        num_workers=args.num_workers,
                        prefetch_factor=args.prefetch_factor,
                        loader_backend=args.loader_backend,
                        seed=args.seed,
                    )

    # { student: { quality: { teacher: path } } }
    print("\n  Scanning for distilled checkpoints (all teachers)...")
    distilled_ckpts: Dict[str, Dict[int, Dict[str, Path]]] = {}
    for s in args.students:
        distilled_ckpts[s] = discover_distilled_checkpoints(
            args.checkpoint_dir, s, args.qualities,
        )
        for q in args.qualities:
            teachers_found = sorted(distilled_ckpts[s][q].keys())
            if teachers_found:
                print(f"  {s} q={q}: {teachers_found}")
            else:
                print(f"  {s} q={q}: (none)")

    # ── Evaluate pretrained + distilled (every teacher) per (student, q) ─
    results: Dict[str, Dict] = {
        s: {"pretrained": [], "distilled": {}} for s in args.students
    }

    for s in args.students:
        can_pretrain = s not in STUDENTS_WITHOUT_PRETRAINED
        for q in args.qualities:
            pre_metrics: Optional[Dict] = None
            if can_pretrain:
                print(f"\n{'─'*64}")
                print(f"  Evaluating PRETRAINED {s} q={q}")
                print(f"{'─'*64}")
                pre = build_model(s, q, pretrained=True).eval().to(args.device)
                pre_metrics = evaluate_model(
                    pre, test_images, args.device,
                    label=f"{s} pretrained q={q}",
                )
                if pre_metrics is not None:
                    pre_metrics["quality"] = q
                    pre_metrics["student"] = s
                    pre_metrics["variant"] = "pretrained"
                    results[s]["pretrained"].append(pre_metrics)
                    print(
                        f"  Pretrained q={q}: BPP={pre_metrics['avg_bpp']:.4f}  "
                        f"PSNR={pre_metrics['avg_psnr']:.2f} dB  "
                        f"Enc={pre_metrics['avg_encode_ms']:.1f}ms  "
                        f"Dec={pre_metrics['avg_decode_ms']:.1f}ms"
                    )
                del pre
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                print(f"\n  Skipping PRETRAINED eval for {s} (no published checkpoint)")

            # Distilled: every teacher we found a checkpoint for
            teacher_paths = distilled_ckpts[s][q]
            if not teacher_paths:
                print(f"  No distilled checkpoint for {s} q={q} — skipping")
                continue

            for teacher, ckpt_path in sorted(teacher_paths.items()):
                print(f"\n{'─'*64}")
                print(f"  Evaluating DISTILLED {s} q={q} (teacher={teacher})")
                print(f"{'─'*64}")
                distilled = load_distilled_student(s, q, ckpt_path, args.device)
                d_metrics = evaluate_model(
                    distilled, test_images, args.device,
                    label=f"{s} distilled←{teacher} q={q}",
                )
                if d_metrics is not None:
                    d_metrics["quality"] = q
                    d_metrics["student"] = s
                    d_metrics["teacher"] = teacher
                    d_metrics["variant"] = f"distilled←{teacher}"
                    results[s]["distilled"].setdefault(teacher, []).append(d_metrics)

                    if pre_metrics is not None:
                        delta_psnr = (d_metrics["avg_psnr"] or 0) - (pre_metrics["avg_psnr"] or 0)
                        delta_bpp  = (d_metrics["avg_bpp"]  or 0) - (pre_metrics["avg_bpp"]  or 0)
                        extra = f"  (ΔPSNR={delta_psnr:+.2f} dB  ΔBPP={delta_bpp:+.4f})"
                    else:
                        extra = ""
                    print(
                        f"  Distilled←{teacher} q={q}: "
                        f"BPP={d_metrics['avg_bpp']:.4f}  "
                        f"PSNR={d_metrics['avg_psnr']:.2f} dB{extra}  "
                        f"Enc={d_metrics['avg_encode_ms']:.1f}ms  "
                        f"Dec={d_metrics['avg_decode_ms']:.1f}ms"
                    )
                del distilled
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

    # ── Evaluate reference codecs (pretrained baselines for the plots) ─
    reference_results: Dict[str, List[Dict]] = {}
    for codec in args.reference_codecs:
        print(f"\n{'─'*64}")
        print(f"  Evaluating REFERENCE (pretrained) {codec}")
        print(f"{'─'*64}")
        entries: List[Dict] = []
        seen_q: set = set()
        for q in args.qualities:
            ref_q = resolve_teacher_quality(codec, q)
            if ref_q in seen_q:
                continue  # don't re-eval when clamping collapses qualities
            seen_q.add(ref_q)
            try:
                ref_model = build_model(codec, ref_q, pretrained=True).eval().to(args.device)
            except Exception as exc:
                print(f"  Skipping {codec} q={ref_q}: {exc}")
                continue
            ref_metrics = evaluate_model(
                ref_model, test_images, args.device,
                label=f"{codec} reference q={ref_q}",
            )
            if ref_metrics is not None:
                ref_metrics["quality"] = ref_q
                ref_metrics["requested_quality"] = q
                ref_metrics["codec"] = codec
                ref_metrics["variant"] = "reference"
                entries.append(ref_metrics)
                print(
                    f"  Reference {codec} q={ref_q}: "
                    f"BPP={ref_metrics['avg_bpp']:.4f}  "
                    f"PSNR={ref_metrics['avg_psnr']:.2f} dB  "
                    f"Enc={ref_metrics['avg_encode_ms']:.1f}ms  "
                    f"Dec={ref_metrics['avg_decode_ms']:.1f}ms"
                )
            del ref_model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        if entries:
            reference_results[codec] = entries

    # ── Save JSON ──────────────────────────────────────────────
    payload = {
        "config": {
            "students":         args.students,
            "teachers":         teachers_for,
            "reference_codecs": args.reference_codecs,
            "qualities":        args.qualities,
            "epochs":           args.epochs,
            "alpha":            args.alpha,
            "lr":               args.lr,
            "aux_lr":           args.aux_lr,
            "batch_size":       args.batch_size,
            "n_source_images":  args.n_source_images,
            "crops_per_image":  args.crops_per_image,
            "val_ratio":        args.val_ratio,
            "val_crops_per_image": args.val_crops_per_image,
            "train_dir":        str(args.train_dir) if args.train_dir else None,
            "test_dir":         str(args.test_dir),
            "n_test_images":    len(test_images),
            "device":           args.device,
            "eval_only":        args.eval_only,
        },
        "results":        results,
        "reference_results": reference_results,
        "distilled_checkpoints": {
            s: {
                str(q): {t: str(p) for t, p in tm.items()}
                for q, tm in qmap.items()
            }
            for s, qmap in distilled_ckpts.items()
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results JSON saved -> {args.output_json}")

    # ── BD-Rate summary (per (student, teacher) vs pretrained baseline) ─
    print(f"\n{'='*72}")
    print("BD-RATE (distilled vs pretrained, per student × teacher)")
    print(f"{'='*72}")
    for s in args.students:
        pre = results[s]["pretrained"]
        pre_bpp  = [e["avg_bpp"]  for e in pre if e.get("avg_bpp")  is not None]
        pre_psnr = [e["avg_psnr"] for e in pre if e.get("avg_psnr") is not None]

        distilled_by_teacher = results[s]["distilled"]
        if not distilled_by_teacher:
            print(f"  {s:<22}: no distilled runs available")
            continue

        if s in STUDENTS_WITHOUT_PRETRAINED or len(pre_bpp) < 2:
            if s in STUDENTS_WITHOUT_PRETRAINED:
                reason = "no pretrained baseline"
            else:
                reason = "need >= 2 pretrained quality points"
            print(f"  {s:<22}: BD-Rate skipped ({reason})")
            # Still report per-teacher raw quality points for clarity.
            for teacher in sorted(distilled_by_teacher.keys()):
                entries = distilled_by_teacher[teacher]
                qs = sorted([e.get("quality") for e in entries if e.get("quality") is not None])
                print(f"    └─ {teacher}: distilled qualities = {qs}")
            continue

        for teacher in sorted(distilled_by_teacher.keys()):
            entries = distilled_by_teacher[teacher]
            dis_bpp  = [e["avg_bpp"]  for e in entries if e.get("avg_bpp")  is not None]
            dis_psnr = [e["avg_psnr"] for e in entries if e.get("avg_psnr") is not None]
            label = f"{s} ← {teacher}"
            if len(dis_bpp) >= 2:
                delta = bd_rate(pre_bpp, pre_psnr, dis_bpp, dis_psnr)
                if delta is not None:
                    direction = "better" if delta < 0 else "worse"
                    print(
                        f"  {label:<44}: {delta:+.1f}%  "
                        f"(distilled uses {abs(delta):.1f}% "
                        f"{'fewer' if delta < 0 else 'more'} bits "
                        f"at matched PSNR — {direction})"
                    )
                else:
                    print(f"  {label:<44}: BD-Rate unavailable (non-overlapping PSNR)")
            else:
                print(f"  {label:<44}: need >= 2 distilled quality points")

    # ── Plots ──────────────────────────────────────────────────
    print()
    plot_rd_curves(results, args.output_rd, reference_results=reference_results)
    plot_runtime( results, args.output_runtime, reference_results=reference_results)

    print(f"""
{'='*72}
INTERPRETATION
{'='*72}
RD curve ({args.output_rd}):
  Distilled curve above/left of pretrained at matched quality level
  means distillation lowered bpp or raised PSNR. A negative BD-Rate
  value above confirms that quantitatively.

Runtime bar chart ({args.output_runtime}):
  Distillation does not change architecture, so encode/decode time
  for pretrained vs distilled should be nearly identical for the same
  student. Differences between students (bmshj2018 vs mbt2018) reflect
  the autoregressive context model in mbt2018 being much slower,
  especially at decode.
{'='*72}
""")


if __name__ == "__main__":
    main()
