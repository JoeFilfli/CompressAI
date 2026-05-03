"""
benchmark_optimized.py

Baseline vs. CPU-optimized inference on every .pth checkpoint in compress_app/.

Baseline   : torch.no_grad(), default entropy coder
Optimized  : inference_mode, ANS entropy coder, model.update(), channels-last
             memory format, tuned thread count, torch.compile (PyTorch 2+)

Usage (from compress_app/):
    python benchmark_optimized.py
"""

import copy
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compressai
import torch
from torchvision.transforms.functional import to_tensor

from utils import (
    compute_msssim,
    compute_psnr,
    crop_to_original,
    infer_checkpoint_base,
    list_custom_checkpoints,
    load_image,
    load_model,
    pad_to_multiple,
    scan_images,
)

APP_DIR   = Path(__file__).parent
IMAGE_DIR = APP_DIR / "test_images"
DEVICE    = "cpu"

# Limit images for a faster run (set to None to use all)
MAX_IMAGES = None

# ── CPU thread tuning (process-wide, applied once) ────────────────────────────
_N_THREADS = os.cpu_count() or 4
torch.set_num_threads(_N_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already initialised elsewhere

# ── Feature detection ─────────────────────────────────────────────────────────
_AVAILABLE_CODERS = compressai.available_entropy_coders()
_DEFAULT_CODER    = _AVAILABLE_CODERS[0]
_HAS_ANS          = "ans" in _AVAILABLE_CODERS
_OPT_CODER        = "ans" if _HAS_ANS else _DEFAULT_CODER
_HAS_COMPILE      = hasattr(torch, "compile")


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_model(model, images, *, inference_mode: bool, channels_last: bool):
    """Compress + decompress every image; return per-image record list."""
    records = []
    n = len(images)
    ctx = torch.inference_mode if inference_mode else torch.no_grad

    for i, img_path in enumerate(images, 1):
        img        = load_image(img_path)
        orig_bytes = img_path.stat().st_size
        x          = to_tensor(img).unsqueeze(0).to(DEVICE)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        x_padded, orig_h, orig_w = pad_to_multiple(x)

        t0 = time.perf_counter()
        with ctx():
            compressed = model.compress(x_padded)
        comp_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        with ctx():
            decompressed = model.decompress(compressed["strings"], compressed["shape"])
        decomp_time = time.perf_counter() - t0

        x_hat  = crop_to_original(decompressed["x_hat"], orig_h, orig_w)
        x_orig = crop_to_original(x_padded, orig_h, orig_w)

        comp_bytes = sum(len(s[0]) for s in compressed["strings"])
        psnr   = compute_psnr(x_orig, x_hat)
        msssim = compute_msssim(x_orig, x_hat)

        print(
            f"  [{i:>3}/{n}] {img_path.name}  "
            f"c={comp_time:.3f}s d={decomp_time:.3f}s  "
            f"PSNR={psnr:.2f} dB",
            flush=True,
        )

        records.append({
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "psnr":       psnr,
            "msssim":     msssim,
            "comp_time":  comp_time,
            "decomp_time":decomp_time,
        })

    return records


def _summarise(records):
    n            = len(records)
    total_in     = sum(r["orig_bytes"]  for r in records)
    total_out    = sum(r["comp_bytes"]  for r in records)
    total_comp   = sum(r["comp_time"]   for r in records)
    total_decomp = sum(r["decomp_time"] for r in records)
    total_t      = total_comp + total_decomp
    return {
        "n":            n,
        "in_mb":        total_in  / 1024**2,
        "comp_mb":      total_out / 1024**2,
        "ratio":        total_in  / total_out if total_out else 0,
        "avg_psnr":     sum(r["psnr"]   for r in records) / n,
        "avg_msssim":   sum(r["msssim"] for r in records) / n,
        "total_t":      total_t,
        "avg_comp_t":   total_comp   / n,
        "avg_decomp_t": total_decomp / n,
        "avg_t":        total_t / n,
        "throughput":   n / total_t,
    }


def _load_model_for_ckpt(ckpt):
    path    = ckpt["path"]
    quality = ckpt["quality"] or 4
    detected = infer_checkpoint_base(path)
    base    = detected.get("base") or "tiny-hyperprior"
    model   = load_model(base, quality, DEVICE, checkpoint_path=path)
    return model, base, quality


def _apply_cpu_optimizations(model):
    """
    Return an optimized copy. Does NOT mutate the input model.

    Applied:
      1. model.update(force=True)  — rebuild CDF tables
      2. channels-last layout      — better cache use for conv layers
      3. compile g_a / g_s / h_a / h_s  — fused kernels (PyTorch 2+)
    """
    m = copy.deepcopy(model)

    # 1. Rebuild CDF tables (entropy coder)
    if hasattr(m, "update"):
        m.update(force=True)
        print("  [opt] model.update(force=True): OK")

    # 2. Channels-last memory format
    try:
        m = m.to(memory_format=torch.channels_last)
        print("  [opt] channels-last memory format: OK")
    except Exception as e:
        print(f"  [opt] channels-last: skipped — {e}")

    # 3. Compile neural submodules (skip entropy coder C-extensions)
    if _HAS_COMPILE:
        compiled_any = False
        for sub_name in ("g_a", "g_s", "h_a", "h_s"):
            sub = getattr(m, sub_name, None)
            if sub is not None:
                try:
                    setattr(m, sub_name, torch.compile(sub, fullgraph=False))
                    compiled_any = True
                except Exception as e:
                    print(f"  [opt] torch.compile({sub_name}): skipped — {e}")
        if compiled_any:
            print(f"  [opt] torch.compile (g_a/g_s/h_a/h_s): OK")
    else:
        print("  [opt] torch.compile: not available (requires PyTorch >= 2.0)")

    return m


def _warmup(model, img_path, *, channels_last: bool):
    """One untimed pass so torch.compile finishes JIT compilation."""
    img    = load_image(img_path)
    x      = to_tensor(img).unsqueeze(0).to(DEVICE)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    x_padded, _, _ = pad_to_multiple(x)
    with torch.inference_mode():
        compressed = model.compress(x_padded)
        model.decompress(compressed["strings"], compressed["shape"])
    print("  [opt] warmup pass: done")


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(results):
    W   = 52
    SEP = W + 88
    print(f"\n\n{'═' * SEP}")
    print(
        f"{'Checkpoint':<{W}} {'PSNR(dB)':>9} {'MS-SSIM':>9}"
        f" {'Base avg/img':>13} {'Opt avg/img':>12} {'Speedup':>8}"
        f" {'Base img/s':>11} {'Opt img/s':>10}"
    )
    print(f"{'─' * SEP}")
    for label, b, o in results:
        if o is None:
            print(f"  {label:<{W}}  [optimized run failed]")
            continue
        speedup = b["avg_t"] / o["avg_t"] if o["avg_t"] > 0 else float("inf")
        print(
            f"{label:<{W}} {o['avg_psnr']:>9.2f} {o['avg_msssim']:>9.4f}"
            f" {b['avg_t']:>12.3f}s {o['avg_t']:>11.3f}s {speedup:>7.2f}x"
            f" {b['throughput']:>10.3f} {o['throughput']:>10.3f}"
        )
    print(f"{'═' * SEP}\n")


def _plot_table(results, out_path: Path):
    col_labels = [
        "Checkpoint",
        "PSNR (dB)", "MS-SSIM",
        "Baseline\navg/img (s)", "Optimized\navg/img (s)", "Speedup",
        "Baseline\n(img/s)", "Optimized\n(img/s)",
    ]

    rows = []
    for label, b, o in results:
        if o is None:
            rows.append([label] + ["—"] * (len(col_labels) - 1))
            continue
        speedup = b["avg_t"] / o["avg_t"] if o["avg_t"] > 0 else float("inf")
        rows.append([
            label,
            f"{o['avg_psnr']:.2f}",
            f"{o['avg_msssim']:.4f}",
            f"{b['avg_t']:.3f}",
            f"{o['avg_t']:.3f}",
            f"{speedup:.2f}×",
            f"{b['throughput']:.3f}",
            f"{o['throughput']:.3f}",
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_w  = max(16, n_cols * 2.0)
    fig_h  = max(3,  n_rows * 0.9 + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_cols)))

    HEADER_BG = "#1a252f"
    EVEN_BG   = "#ecf0f1"
    ODD_BG    = "#ffffff"
    GREEN_BG  = "#a9dfbf"
    YELLOW_BG = "#fdebd0"
    RED_BG    = "#f5b7b1"

    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold")

    for row_idx, (_, b, o) in enumerate(results, 1):
        bg = EVEN_BG if row_idx % 2 == 0 else ODD_BG
        for col in range(n_cols):
            tbl[row_idx, col].set_facecolor(bg)

        if o is not None:
            speedup = b["avg_t"] / o["avg_t"] if o["avg_t"] > 0 else 0
            spd_col = 5
            if speedup >= 1.5:
                tbl[row_idx, spd_col].set_facecolor(GREEN_BG)
                tbl[row_idx, spd_col].set_text_props(fontweight="bold")
            elif speedup >= 1.0:
                tbl[row_idx, spd_col].set_facecolor(YELLOW_BG)
            else:
                tbl[row_idx, spd_col].set_facecolor(RED_BG)

    opt_labels = (
        f"Optimizations: inference_mode  |  ANS coder ({_OPT_CODER})  |  "
        f"channels-last  |  model.update()  |  "
        f"torch.compile={'on' if _HAS_COMPILE else 'off'}  |  "
        f"threads={_N_THREADS}"
    )
    plt.suptitle(
        "Baseline vs. CPU-Optimized Inference — Distilled Models",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.title(opt_labels, fontsize=8, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison table saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    images = scan_images(IMAGE_DIR, recursive=False)
    if MAX_IMAGES:
        images = images[:MAX_IMAGES]
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    checkpoints = list_custom_checkpoints(APP_DIR)
    if not checkpoints:
        print(f"No .pth checkpoints found in {APP_DIR}")
        sys.exit(1)

    print("=" * 64)
    print("  CompressAI — Baseline vs. Optimized Benchmark")
    print("=" * 64)
    print(f"  Device          : {DEVICE}")
    print(f"  CPU threads     : {_N_THREADS}")
    print(f"  Images          : {len(images)}")
    print(f"  Checkpoints     : {len(checkpoints)}")
    print(f"  Baseline coder  : {_DEFAULT_CODER}")
    print(f"  Optimized coder : {_OPT_CODER}")
    print(f"  torch.compile   : {'available' if _HAS_COMPILE else 'not available (PyTorch < 2.0)'}")
    print("=" * 64)

    results = []

    for ckpt in checkpoints:
        label = ckpt["label"]

        # ── BASELINE ──────────────────────────────────────────────────────────
        print(f"\n{'━'*64}")
        print(f"  BASELINE  {label}")
        print(f"{'━'*64}")
        compressai.set_entropy_coder(_DEFAULT_CODER)
        try:
            model, base, quality = _load_model_for_ckpt(ckpt)
            print(f"  base={base}  quality={quality}\n")
            baseline_records = _run_model(
                model, images,
                inference_mode=False,
                channels_last=False,
            )
            baseline_s = _summarise(baseline_records)
            print(
                f"\n  Baseline summary: avg={baseline_s['avg_t']:.3f}s/img  "
                f"throughput={baseline_s['throughput']:.3f} img/s  "
                f"PSNR={baseline_s['avg_psnr']:.2f} dB"
            )
        except Exception as e:
            print(f"  ERROR in baseline: {e}")
            continue

        # ── OPTIMIZED ─────────────────────────────────────────────────────────
        print(f"\n{'━'*64}")
        print(f"  OPTIMIZED  {label}")
        print(f"{'━'*64}")
        compressai.set_entropy_coder(_OPT_CODER)
        print(f"  [opt] entropy coder: {_OPT_CODER}")
        print(f"  [opt] inference_mode: enabled")
        print(f"  [opt] num_threads: {_N_THREADS}")

        try:
            model_opt = _apply_cpu_optimizations(model)
            _warmup(model_opt, images[0], channels_last=True)
            print()
            opt_records = _run_model(
                model_opt, images,
                inference_mode=True,
                channels_last=True,
            )
            opt_s = _summarise(opt_records)
            speedup = baseline_s["avg_t"] / opt_s["avg_t"] if opt_s["avg_t"] > 0 else float("inf")
            print(
                f"\n  Optimized summary: avg={opt_s['avg_t']:.3f}s/img  "
                f"throughput={opt_s['throughput']:.3f} img/s  "
                f"Speedup={speedup:.2f}x"
            )
        except Exception as e:
            print(f"  ERROR in optimized: {e}")
            opt_s = None

        results.append((label, baseline_s, opt_s))

    if not results:
        print("\nNo results to display.")
        sys.exit(1)

    _print_comparison(results)
    _plot_table(results, APP_DIR / "benchmark_comparison.png")


if __name__ == "__main__":
    main()
