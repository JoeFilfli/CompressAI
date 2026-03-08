"""Profile the neural image compression pipeline.

Loads a single image and measures execution time for four stages:
  1. NN Encoder (Inference)   — g_a (+ h_a for hyperprior models)
  2. Entropy Encoding         — arithmetic / range-ANS encoding
  3. Entropy Decoding         — arithmetic / range-ANS decoding
  4. NN Decoder (Inference)   — g_s (+ h_s for hyperprior models)

Usage:
    python profile_pipeline.py <image_path> [options]

Example:
    python profile_pipeline.py kodim01.png -m bmshj2018-factorized -q 3
    python profile_pipeline.py kodim01.png -m mbt2018-mean -q 4 --device cuda
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

import compressai
from compressai.zoo import models


def pad_to_multiple(x, div=64):
    """Pad tensor so spatial dims are divisible by *div*."""
    _, _, h, w = x.shape
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, h, w


def load_image(path, device):
    from PIL import Image
    from torchvision.transforms import ToTensor

    img = Image.open(path).convert("RGB")
    x = ToTensor()(img).unsqueeze(0).to(device)
    return x


# ── Factorized-prior models (no hyper-analysis / hyper-synthesis) ────────────


def profile_factorized(net, x):
    """Profile FactorizedPrior and FactorizedPriorReLU."""

    # Stage 1 — NN Encoder
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    y = net.g_a(x)
    torch.cuda.synchronize() if x.is_cuda else None
    enc_nn_time = time.perf_counter() - t0

    # Stage 2 — Entropy Encoding
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    y_strings = net.entropy_bottleneck.compress(y)
    torch.cuda.synchronize() if x.is_cuda else None
    enc_entropy_time = time.perf_counter() - t0

    shape = y.size()[-2:]

    # Stage 3 — Entropy Decoding
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    y_hat = net.entropy_bottleneck.decompress(y_strings, shape)
    torch.cuda.synchronize() if x.is_cuda else None
    dec_entropy_time = time.perf_counter() - t0

    # Stage 4 — NN Decoder
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    x_hat = net.g_s(y_hat).clamp_(0, 1)
    torch.cuda.synchronize() if x.is_cuda else None
    dec_nn_time = time.perf_counter() - t0

    num_bytes = sum(len(s) for s in y_strings)

    return enc_nn_time, enc_entropy_time, dec_entropy_time, dec_nn_time, x_hat, num_bytes


# ── Scale-hyperprior models ─────────────────────────────────────────────────


def profile_scale_hyperprior(net, x):
    """Profile ScaleHyperprior (bmshj2018-hyperprior)."""

    # Stage 1 — NN Encoder (g_a + h_a)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    y = net.g_a(x)
    z = net.h_a(torch.abs(y))
    torch.cuda.synchronize() if x.is_cuda else None
    enc_nn_time = time.perf_counter() - t0

    # Stage 2 — Entropy Encoding (z + y)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    z_strings = net.entropy_bottleneck.compress(z)
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    scales_hat = net.h_s(z_hat)
    indexes = net.gaussian_conditional.build_indexes(scales_hat)
    y_strings = net.gaussian_conditional.compress(y, indexes)
    torch.cuda.synchronize() if x.is_cuda else None
    enc_entropy_time = time.perf_counter() - t0

    shape = z.size()[-2:]

    # Stage 3 — Entropy Decoding (z + y)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    z_hat = net.entropy_bottleneck.decompress(z_strings, shape)
    scales_hat = net.h_s(z_hat)
    indexes = net.gaussian_conditional.build_indexes(scales_hat)
    y_hat = net.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)
    torch.cuda.synchronize() if x.is_cuda else None
    dec_entropy_time = time.perf_counter() - t0

    # Stage 4 — NN Decoder (g_s)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    x_hat = net.g_s(y_hat).clamp_(0, 1)
    torch.cuda.synchronize() if x.is_cuda else None
    dec_nn_time = time.perf_counter() - t0

    num_bytes = sum(len(s) for s in y_strings) + sum(len(s) for s in z_strings)

    return enc_nn_time, enc_entropy_time, dec_entropy_time, dec_nn_time, x_hat, num_bytes


# ── Mean-scale-hyperprior models ────────────────────────────────────────────


def profile_mean_scale_hyperprior(net, x):
    """Profile MeanScaleHyperprior (mbt2018-mean)."""

    # Stage 1 — NN Encoder (g_a + h_a)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    y = net.g_a(x)
    z = net.h_a(y)
    torch.cuda.synchronize() if x.is_cuda else None
    enc_nn_time = time.perf_counter() - t0

    # Stage 2 — Entropy Encoding (z + y)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    z_strings = net.entropy_bottleneck.compress(z)
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    gaussian_params = net.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = net.gaussian_conditional.build_indexes(scales_hat)
    y_strings = net.gaussian_conditional.compress(y, indexes, means=means_hat)
    torch.cuda.synchronize() if x.is_cuda else None
    enc_entropy_time = time.perf_counter() - t0

    shape = z.size()[-2:]

    # Stage 3 — Entropy Decoding (z + y)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    z_hat = net.entropy_bottleneck.decompress(z_strings, shape)
    gaussian_params = net.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = net.gaussian_conditional.build_indexes(scales_hat)
    y_hat = net.gaussian_conditional.decompress(
        y_strings, indexes, means=means_hat
    )
    torch.cuda.synchronize() if x.is_cuda else None
    dec_entropy_time = time.perf_counter() - t0

    # Stage 4 — NN Decoder (g_s)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    x_hat = net.g_s(y_hat).clamp_(0, 1)
    torch.cuda.synchronize() if x.is_cuda else None
    dec_nn_time = time.perf_counter() - t0

    num_bytes = sum(len(s) for s in y_strings) + sum(len(s) for s in z_strings)

    return enc_nn_time, enc_entropy_time, dec_entropy_time, dec_nn_time, x_hat, num_bytes


# ── Dispatcher ───────────────────────────────────────────────────────────────

MODEL_PROFILERS = {
    "bmshj2018-factorized": profile_factorized,
    "bmshj2018-factorized-relu": profile_factorized,
    "bmshj2018-factorized-wavelet": profile_factorized,
    "bmshj2018-hyperprior": profile_scale_hyperprior,
    "mbt2018-mean": profile_mean_scale_hyperprior,
    "cheng2020-anchor": profile_mean_scale_hyperprior,
    "cheng2020-attn": profile_mean_scale_hyperprior,
}


def get_profiler(model_name, net):
    """Return the appropriate profiling function for the given model."""
    if model_name in MODEL_PROFILERS:
        return MODEL_PROFILERS[model_name]
    # Fallback: detect by architecture attributes
    if hasattr(net, "h_a"):
        if hasattr(net, "gaussian_conditional"):
            return profile_mean_scale_hyperprior
        return profile_scale_hyperprior
    return profile_factorized


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Profile a CompressAI neural image compression pipeline."
    )
    p.add_argument("image", type=str, help="Path to the input image.")
    p.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=list(models.keys()),
        help="Pretrained model (default: %(default)s).",
    )
    p.add_argument(
        "-q",
        "--quality",
        type=int,
        default=1,
        help="Quality level (default: %(default)s).",
    )
    p.add_argument(
        "--metric",
        default="mse",
        choices=["mse", "ms-ssim"],
        help="Optimized metric (default: %(default)s).",
    )
    p.add_argument(
        "-d",
        "--device",
        default="cpu",
        help="Device: cpu or cuda (default: %(default)s).",
    )
    p.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of runs; report the average (default: %(default)s).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    coder = compressai.available_entropy_coders()[0]
    compressai.set_entropy_coder(coder)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model {args.model} (quality={args.quality}) ...")
    t_load = time.perf_counter()
    net = (
        models[args.model](quality=args.quality, metric=args.metric, pretrained=True)
        .to(device)
        .eval()
    )
    net.update()
    t_load = time.perf_counter() - t_load
    print(f"Model loaded in {t_load:.3f}s\n")

    profiler = get_profiler(args.model, net)

    # ── Load image ───────────────────────────────────────────────────────
    x = load_image(args.image, device)
    _, _, orig_h, orig_w = x.shape
    x, _, _ = pad_to_multiple(x)

    # ── Profile ──────────────────────────────────────────────────────────
    totals = [0.0, 0.0, 0.0, 0.0]
    total_bytes = 0

    for r in range(args.runs):
        with torch.inference_mode():
            enc_nn, enc_ent, dec_ent, dec_nn, x_hat, num_bytes = profiler(net, x)
        totals[0] += enc_nn
        totals[1] += enc_ent
        totals[2] += dec_ent
        totals[3] += dec_nn
        total_bytes = num_bytes

    n = args.runs
    avg = [t / n for t in totals]
    total = sum(avg)

    # ── Compute PSNR ─────────────────────────────────────────────────────
    x_orig = x[:, :, :orig_h, :orig_w]
    x_rec = x_hat[:, :, :orig_h, :orig_w]
    mse_val = ((x_orig - x_rec) ** 2).mean().item()
    psnr = -10 * torch.tensor(mse_val).log10().item() if mse_val > 0 else float("inf")

    original_pixels = orig_h * orig_w
    bpp = total_bytes * 8 / original_pixels

    # ── Report ───────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Image        : {args.image}")
    print(f"  Resolution   : {orig_w} x {orig_h}")
    print(f"  Model        : {args.model}  (quality={args.quality})")
    print(f"  Device       : {device}")
    print(f"  Entropy coder: {coder}")
    print(f"  Runs         : {n}")
    print("=" * 60)
    print(f"  {'Stage':<25} {'Time (s)':>10} {'Share':>8}")
    print("-" * 60)
    print(f"  {'NN Encoder (Inference)':<25} {avg[0]:>10.4f} {avg[0]/total*100:>7.1f}%")
    print(f"  {'Entropy Encoding':<25} {avg[1]:>10.4f} {avg[1]/total*100:>7.1f}%")
    print(f"  {'Entropy Decoding':<25} {avg[2]:>10.4f} {avg[2]/total*100:>7.1f}%")
    print(f"  {'NN Decoder (Inference)':<25} {avg[3]:>10.4f} {avg[3]/total*100:>7.1f}%")
    print("-" * 60)
    print(f"  {'Total':<25} {total:>10.4f} {'100.0%':>8}")
    print("=" * 60)
    print(f"  Bitstream    : {total_bytes} bytes  ({bpp:.4f} bpp)")
    print(f"  PSNR         : {psnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
