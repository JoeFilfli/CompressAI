"""
Benchmark: Neural Compression vs Traditional Codecs

Models tested:
    Neural:      bmshj2018-factorized
    Traditional: JPEG, WebP, AVIF

Measures per image per quality level:
    BPP         — bits per pixel (lower = smaller file)
    PSNR        — reconstruction quality in dB (higher = better)
    MS-SSIM     — structural perceptual quality (higher = better)
    LPIPS       — deep perceptual distance (lower = better)
    Encode time — compression time in ms
    Decode time — decompression time in ms

Output:
    benchmark_results.json  — raw numbers for every image/quality
    rd_curve.png            — Figure 1: BPP vs PSNR
    rd_msssim.png           — Figure 2: BPP vs MS-SSIM
    rd_lpips.png            — Figure 3: BPP vs LPIPS
    bd_rate_results.json    — Figure 4 data: pairwise BD-Rate table
    bd_rate_table.png       — Figure 4: pairwise BD-Rate table image
    runtime_chart.png       — Encode/Decode time comparison

Usage:
    # Standard Kodak benchmark (PNG input — valid for paper comparison):
    python benchmark_neural_vs_traditional.py --input-dir ./kodak

    # Photographer experiment (JPEG/AVIF input — different research question):
    python benchmark_neural_vs_traditional.py --input-dir ./my_photos --source-type lossy

    # Quick sanity check (5 images only):
    python benchmark_neural_vs_traditional.py --input-dir ./kodak --max-images 5

    # With GPU:
    python benchmark_neural_vs_traditional.py --input-dir ./kodak --device cuda

AVIF support requires:
    pip install pillow-heif

Optional perceptual metrics:
    pip install pytorch-msssim lpips
"""

import argparse
import io
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from compressai.ops import compute_padding
from compressai.zoo import bmshj2018_factorized

try:
    from pytorch_msssim import ms_ssim as compute_msssim
    MSSSIM_AVAILABLE = True
except ImportError:
    compute_msssim = None
    MSSSIM_AVAILABLE = False
    print("⚠️  MS-SSIM unavailable — install with: pip install pytorch-msssim")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS unavailable — install with: pip install lpips")

# ─────────────────────────────────────────────────────────────
# AVIF SUPPORT — optional, gracefully disabled if not installed
# ─────────────────────────────────────────────────────────────

AVIF_AVAILABLE = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    AVIF_AVAILABLE = True
    print("✅ AVIF support enabled (pillow-heif)")
except ImportError:
    try:
        import pillow_avif  # alternative package
        AVIF_AVAILABLE = True
        print("✅ AVIF support enabled (pillow-avif)")
    except ImportError:
        print("⚠️  AVIF not available — install with: pip install pillow-heif")
        print("   Benchmark will run without AVIF.\n")

# ─────────────────────────────────────────────────────────────
# SUPPORTED INPUT EXTENSIONS
# ─────────────────────────────────────────────────────────────

# Clean lossless sources (valid for standard benchmark)
LOSSLESS_EXTENSIONS = {".png", ".bmp", ".tiff", ".tif"}

# Lossy sources (valid only for photographer experiment)
LOSSY_EXTENSIONS    = {".jpg", ".jpeg", ".avif", ".heic", ".webp"}

ALL_EXTENSIONS = LOSSLESS_EXTENSIONS | LOSSY_EXTENSIONS

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

NEURAL_MODELS = {
    "bmshj2018-factorized": bmshj2018_factorized,
}

NEURAL_QUALITIES_BY_MODEL = {
    "bmshj2018-factorized": [1, 2, 3, 4, 5, 6, 7, 8],
}

JPEG_QUALITIES = [5, 10, 15, 20, 30, 40, 55, 70, 80, 90, 95]
WEBP_QUALITIES = [5, 10, 15, 20, 30, 40, 55, 70, 80, 90, 95]
AVIF_QUALITIES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# Note: AVIF quality scale is different from JPEG — 
# AVIF 60 ≈ JPEG 85 in terms of visual output

COLORS = {
    "bmshj2018-factorized": "#2ecc71",
    "JPEG":                 "#e74c3c",
    "WebP":                 "#3498db",
    "AVIF":                 "#f39c12",
}

MARKERS = {
    "bmshj2018-factorized": "^",
    "JPEG":                 "o",
    "WebP":                 "s",
    "AVIF":                 "P",
}

EXAMPLES_DIR = Path(__file__).resolve().parent
LPIPS_MODELS: Dict[str, torch.nn.Module] = {}
LPIPS_INIT_FAILED = False

# ─────────────────────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────────────────────

def check_input_images(images: List[Path], source_type: str) -> bool:
    """
    Warn the user if input images are lossy when running standard benchmark.

    source_type:
        "lossless" — PNG/TIFF, valid for standard benchmark
        "lossy"    — JPEG/AVIF, valid only for photographer experiment
        "auto"     — detect from file extensions, warn if mixed
    """
    extensions = {p.suffix.lower() for p in images}
    lossy_found   = bool(extensions & LOSSY_EXTENSIONS)
    lossless_found = bool(extensions & LOSSLESS_EXTENSIONS)

    print(f"\n  Input file types detected: {sorted(extensions)}")

    if source_type == "lossless" and lossy_found:
        print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  ⚠️  WARNING: LOSSY INPUT IMAGES DETECTED                ║
  ║                                                          ║
  ║  Your folder contains JPEG/AVIF files (already lossy).   ║
  ║  Using them as source will produce INVALID results for   ║
  ║  the standard benchmark because:                         ║
  ║                                                          ║
  ║  • PSNR is measured against already-damaged pixels       ║
  ║  • Neural model encodes JPEG artifacts, not clean photo  ║
  ║  • Results cannot be compared to published papers        ║
  ║                                                          ║
  ║  For standard benchmark → use Kodak PNG images           ║
  ║  For photographer test  → add --source-type lossy        ║
  ╚══════════════════════════════════════════════════════════╝
        """)
        return False

    if source_type == "lossy":
        print("""
  ℹ️  PHOTOGRAPHER EXPERIMENT MODE
  Input images are lossy (JPEG/AVIF). This is valid for testing
  "can neural compression further compress real photo archives?"
  Results CANNOT be compared to Kodak/paper benchmarks.
  This is a separate, practical experiment.
        """)

    if source_type == "auto":
        if lossy_found and lossless_found:
            print("""
  ⚠️  MIXED INPUT: both lossless (PNG) and lossy (JPEG/AVIF) found.
  All images will be treated as lossless. If this is intentional,
  use --source-type lossy to suppress this warning.
            """)
        elif lossy_found:
            print("""
  ⚠️  All inputs are lossy (JPEG/AVIF). Results will be valid only
  for the photographer experiment, NOT the standard benchmark.
  Use --source-type lossy to suppress this warning.
            """)

    return True


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0) - 10 * math.log10(mse)


def calculate_bpp(compressed_bytes: int, height: int, width: int) -> float:
    return (compressed_bytes * 8) / (height * width)


def calculate_ms_ssim(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Optional[float]:
    if not MSSSIM_AVAILABLE:
        return None
    return float(compute_msssim(original, reconstructed, data_range=1.0).item())


def get_lpips_model(device: str):
    global LPIPS_INIT_FAILED

    if not LPIPS_AVAILABLE:
        return None
    if LPIPS_INIT_FAILED:
        return None

    model = LPIPS_MODELS.get(device)
    if model is None:
        try:
            model = lpips.LPIPS(net="alex").eval().to(device)
        except Exception as exc:
            LPIPS_INIT_FAILED = True
            print(f"⚠️  LPIPS disabled — could not initialize model: {exc}")
            return None
        LPIPS_MODELS[device] = model
    return model


def calculate_lpips(
    original: torch.Tensor, reconstructed: torch.Tensor, device: str
) -> Optional[float]:
    model = get_lpips_model(device)
    if model is None:
        return None

    with torch.inference_mode():
        original_scaled = original.to(device) * 2 - 1
        reconstructed_scaled = reconstructed.to(device) * 2 - 1
        return float(model(original_scaled, reconstructed_scaled).item())


def average_or_none(values: List[Optional[float]], digits: Optional[int] = None) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None

    avg = float(np.mean(filtered))
    if digits is None:
        return avg
    return round(avg, digits)


def format_metric(value: Optional[float], fmt: str, missing: str = "n/a") -> str:
    if value is None:
        return missing
    return format(value, fmt)


def metric_suffix(result: Dict) -> str:
    parts = []
    if result.get("ms_ssim") is not None:
        parts.append(f"MS-SSIM: {result['ms_ssim']:.4f}")
    if result.get("lpips") is not None:
        parts.append(f"LPIPS: {result['lpips']:.4f}")
    return " | ".join(parts)


def bd_rate(rate1, metric1, rate2, metric2) -> Optional[float]:
    """Bjontegaard Delta Rate: negative means codec2 needs fewer bits."""
    if len(rate1) < 2 or len(rate2) < 2:
        return None

    points1 = sorted(zip(metric1, rate1))
    points2 = sorted(zip(metric2, rate2))
    metric1_sorted, rate1_sorted = zip(*points1)
    metric2_sorted, rate2_sorted = zip(*points2)

    min_metric = max(min(metric1_sorted), min(metric2_sorted))
    max_metric = min(max(metric1_sorted), max(metric2_sorted))
    if max_metric <= min_metric:
        return None

    degree = min(3, len(rate1_sorted) - 1, len(rate2_sorted) - 1)
    if degree < 1:
        return None

    log_rate1 = np.log(np.asarray(rate1_sorted))
    log_rate2 = np.log(np.asarray(rate2_sorted))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = np.polyfit(metric1_sorted, log_rate1, degree)
        p2 = np.polyfit(metric2_sorted, log_rate2, degree)

    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)

    int1 = np.polyval(p_int1, max_metric) - np.polyval(p_int1, min_metric)
    int2 = np.polyval(p_int2, max_metric) - np.polyval(p_int2, min_metric)

    avg_diff = (int2 - int1) / (max_metric - min_metric)
    return float((np.exp(avg_diff) - 1) * 100)


def compute_bd_rate_table(results: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    codecs = [
        codec_name
        for codec_name, quality_points in results.items()
        if len([p for p in quality_points if p.get("avg_psnr") is not None]) >= 2
    ]

    table: Dict[str, Dict[str, Optional[float]]] = {}
    for reference_codec in codecs:
        table[reference_codec] = {}
        ref_points = sorted(results[reference_codec], key=lambda x: x["avg_bpp"])
        ref_rates = [p["avg_bpp"] for p in ref_points]
        ref_psnr = [p["avg_psnr"] for p in ref_points]

        for test_codec in codecs:
            if reference_codec == test_codec:
                table[reference_codec][test_codec] = 0.0
                continue

            test_points = sorted(results[test_codec], key=lambda x: x["avg_bpp"])
            test_rates = [p["avg_bpp"] for p in test_points]
            test_psnr = [p["avg_psnr"] for p in test_points]
            table[reference_codec][test_codec] = bd_rate(
                ref_rates, ref_psnr, test_rates, test_psnr
            )

    return table


def save_bd_rate_table_image(bd_table: Dict[str, Dict[str, Optional[float]]], output_path: str):
    if not bd_table:
        print("  ⚠️  Skipping BD-Rate table image — no comparable curves available")
        return

    codecs = list(bd_table.keys())
    cell_text = []
    for reference_codec in codecs:
        row = []
        for test_codec in codecs:
            value = bd_table[reference_codec][test_codec]
            if value is None:
                row.append("n/a")
            else:
                row.append(f"{value:+.1f}%")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(codecs) * 2.1), max(3.5, len(codecs) * 0.8 + 1.8)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=codecs,
        colLabels=codecs,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")

    ax.set_title(
        "BD-Rate Table (matched PSNR)\n"
        "Cell[row, col] = bitrate delta of col vs row; negative is better",
        fontsize=12,
        pad=18,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ BD-Rate table saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# NEURAL COMPRESSION
# ─────────────────────────────────────────────────────────────

def compress_neural(img: Image.Image, model, device: str) -> Dict:
    """Compress one image with a CompressAI model."""
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    h, w = x.shape[2], x.shape[3]

    pad, unpad = compute_padding(h, w, min_div=64)
    x_padded = F.pad(x, pad)

    with torch.inference_mode():
        t0 = time.perf_counter()
        out_enc = model.compress(x_padded)
        encode_ms = (time.perf_counter() - t0) * 1000

        compressed_bytes = sum(
            len(s) for sublist in out_enc["strings"] for s in sublist
        )

        t0 = time.perf_counter()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        decode_ms = (time.perf_counter() - t0) * 1000

        x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)

    ms_ssim_val = calculate_ms_ssim(x, x_hat)
    lpips_val = calculate_lpips(x, x_hat, device)

    return {
        "bpp":       calculate_bpp(compressed_bytes, h, w),
        "psnr":      calculate_psnr(x, x_hat),
        "ms_ssim":   ms_ssim_val,
        "lpips":     lpips_val,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
    }


# ─────────────────────────────────────────────────────────────
# TRADITIONAL CODECS  (JPEG, WebP, AVIF)
# ─────────────────────────────────────────────────────────────

def compress_traditional(img: Image.Image, codec: str, quality: int, device: str) -> Dict:
    """
    Compress one image with JPEG, WebP, or AVIF.

    AVIF quality notes:
        Pillow/pillow-heif uses 0-100 where higher = better quality.
        Internally maps to quantizer parameter (lower Q = better).
        AVIF 60 ≈ JPEG 85 visually.
    """
    original_tensor = transforms.ToTensor()(img).unsqueeze(0)
    h, w = img.size[1], img.size[0]
    buffer = io.BytesIO()

    t0 = time.perf_counter()

    if codec == "JPEG":
        img.save(buffer, format="JPEG", quality=quality,
                 subsampling=0, optimize=True)

    elif codec == "WebP":
        img.save(buffer, format="WEBP", quality=quality, method=6)

    elif codec == "AVIF":
        if not AVIF_AVAILABLE:
            raise RuntimeError(
                "AVIF not available. Install with: pip install pillow-heif"
            )
        # pillow-heif uses quality=0-100, higher = better
        img.save(buffer, format="AVIF", quality=quality)

    else:
        raise ValueError(f"Unknown codec: {codec}")

    encode_ms = (time.perf_counter() - t0) * 1000
    compressed_bytes = buffer.tell()

    buffer.seek(0)
    t0 = time.perf_counter()
    reconstructed = Image.open(buffer).convert("RGB")
    decode_ms = (time.perf_counter() - t0) * 1000

    reconstructed_tensor = transforms.ToTensor()(reconstructed).unsqueeze(0)
    ms_ssim_val = calculate_ms_ssim(original_tensor, reconstructed_tensor)
    lpips_val = calculate_lpips(original_tensor, reconstructed_tensor, device)

    return {
        "bpp":       calculate_bpp(compressed_bytes, h, w),
        "psnr":      calculate_psnr(original_tensor, reconstructed_tensor),
        "ms_ssim":   ms_ssim_val,
        "lpips":     lpips_val,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
    }


# ─────────────────────────────────────────────────────────────
# HELPER — run one codec over all images at all quality levels
# ─────────────────────────────────────────────────────────────

def run_traditional_codec(
    codec: str,
    qualities: List[int],
    images: List[Path],
    n: int,
    device: str,
) -> List[Dict]:
    """Returns list of per-quality-level averaged results."""
    entries = []
    print(f"\n{'─'*60}\n  {codec}\n{'─'*60}")

    for q in qualities:
        bpps, psnrs, ms_ssims, lpips_scores, encs, decs = [], [], [], [], [], []
        failed = 0

        for img_path in images:
            try:
                r = compress_traditional(
                    Image.open(img_path).convert("RGB"), codec, q, device
                )
                bpps.append(r["bpp"])
                psnrs.append(r["psnr"])
                ms_ssims.append(r["ms_ssim"])
                lpips_scores.append(r["lpips"])
                encs.append(r["encode_ms"])
                decs.append(r["decode_ms"])
            except Exception as e:
                failed += 1
                print(f"    ⚠️  {img_path.name} failed: {e}")

        if not bpps:
            print(f"  q={q:3d} | ALL IMAGES FAILED — skipping")
            continue

        entry = {
            "quality":       q,
            "avg_bpp":       round(float(np.mean(bpps)),  4),
            "avg_psnr":      round(float(np.mean(psnrs)), 2),
            "avg_ms_ssim":   average_or_none(ms_ssims, 4),
            "avg_lpips":     average_or_none(lpips_scores, 4),
            "avg_encode_ms": round(float(np.mean(encs)),  1),
            "avg_decode_ms": round(float(np.mean(decs)),  1),
            "n_images":      len(bpps),
        }
        entries.append(entry)

        fail_note = f" ({failed} failed)" if failed else ""
        print(f"  q={q:3d} | BPP: {entry['avg_bpp']:.4f} | "
              f"PSNR: {entry['avg_psnr']:.2f} dB | "
              f"MS-SSIM: {format_metric(entry['avg_ms_ssim'], '.4f')} | "
              f"LPIPS: {format_metric(entry['avg_lpips'], '.4f')} | "
              f"Enc: {entry['avg_encode_ms']:.1f}ms | "
              f"Dec: {entry['avg_decode_ms']:.1f}ms{fail_note}")

    return entries


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_rd_curves(
    results: Dict,
    metric_key: str,
    output_path: str,
    device: str,
    source_type: str,
):
    metric_specs = {
        "psnr": {
            "entry_key": "avg_psnr",
            "ylabel": "PSNR [dB]",
            "title_metric": "PSNR",
            "lower_is_better": False,
            "legend_loc": "lower right",
        },
        "ms_ssim": {
            "entry_key": "avg_ms_ssim",
            "ylabel": "MS-SSIM [higher is better]",
            "title_metric": "MS-SSIM",
            "lower_is_better": False,
            "legend_loc": "lower right",
        },
        "lpips": {
            "entry_key": "avg_lpips",
            "ylabel": "LPIPS [lower is better]",
            "title_metric": "LPIPS",
            "lower_is_better": True,
            "legend_loc": "upper right",
        },
    }
    spec = metric_specs[metric_key]

    plt.figure(figsize=(13, 8))
    plotted_any = False

    for codec_name, quality_points in results.items():
        if not quality_points:
            continue
        points = sorted(
            [p for p in quality_points if p.get(spec["entry_key"]) is not None],
            key=lambda x: x["avg_bpp"],
        )
        if not points:
            continue

        bpps = [p["avg_bpp"] for p in points]
        metric_values = [p[spec["entry_key"]] for p in points]
        is_neural = codec_name in NEURAL_MODELS
        plotted_any = True

        plt.plot(
            bpps, metric_values,
            color=COLORS.get(codec_name, "black"),
            marker=MARKERS.get(codec_name, "o"),
            linestyle="-" if is_neural else "--",
            linewidth=2.5 if is_neural else 1.8,
            markersize=7,
            label=codec_name,
        )

    if not plotted_any:
        plt.close()
        print(f"  ⚠️  Skipping {metric_key.upper()} curve — no data available")
        return

    source_label = (
        "Kodak Dataset — 24 uncompressed PNGs"
        if source_type == "lossless"
        else "Photographer Images — lossy source (JPEG/AVIF input)"
    )

    plt.xlabel("Bit-rate [bpp]", fontsize=13)
    plt.ylabel(spec["ylabel"], fontsize=13)
    plt.title(
        f"Rate-Distortion Curves ({spec['title_metric']}) — {source_label}\n"
        f"Solid = Neural   |   Dashed = Traditional   |   Device: {device.upper()}",
        fontsize=12
    )
    plt.legend(fontsize=11, loc=spec["legend_loc"])
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    if metric_key == "psnr":
        plt.ylim(bottom=26)
    elif metric_key == "ms_ssim":
        plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {spec['title_metric']} R-D curve saved → {output_path}")


def plot_runtime(results: Dict, output_path: str, device: str):
    codec_names, avg_enc, avg_dec = [], [], []
    for codec_name, quality_points in results.items():
        if not quality_points:
            continue
        codec_names.append(codec_name)
        avg_enc.append(np.mean([p["avg_encode_ms"] for p in quality_points]))
        avg_dec.append(np.mean([p["avg_decode_ms"] for p in quality_points]))

    x = np.arange(len(codec_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, avg_enc, width,
                   label="Encode (ms)", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, avg_dec, width,
                   label="Decode (ms)", color="#e74c3c", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15,
                f"{h:.0f}ms", ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylabel("Time per image (ms) — log scale", fontsize=12)
    ax.set_title(
        f"Average Encode / Decode Time per Image\n"
        f"Averaged across all quality levels   |   Device: {device.upper()}",
        fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels(codec_names, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Runtime chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN BENCHMARK
# ─────────────────────────────────────────────────────────────

def run_benchmark(
    input_dir:   Path,
    device:      str  = "cpu",
    max_images:  Optional[int] = None,
    source_type: str  = "auto",
    output_json: str  = "benchmark_results.json",
    output_rd:   str  = "rd_curve.png",
    output_msssim: str = "rd_msssim.png",
    output_lpips: str = "rd_lpips.png",
    output_bd_json: str = "bd_rate_results.json",
    output_bd_table: str = "bd_rate_table.png",
    output_rt:   str  = "runtime_chart.png",
):
    # ── Discover images ──────────────────────────────────────
    images = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in ALL_EXTENSIONS
    ])
    if max_images:
        images = images[:max_images]

    if not images:
        print(f"❌  No supported images found in {input_dir}")
        print(f"    Supported: {sorted(ALL_EXTENSIONS)}")
        return

    n = len(images)

    # ── Header ───────────────────────────────────────────────
    print("=" * 70)
    print("BENCHMARK — Neural vs Traditional Image Compression")
    print("=" * 70)
    print(f"  Images     : {n} from {input_dir}")
    print(f"  Device     : {device.upper()}")
    print(f"  Neural     : {', '.join(NEURAL_MODELS.keys())}")
    trad = "JPEG, WebP, AVIF" if AVIF_AVAILABLE else "JPEG, WebP  (AVIF unavailable)"
    print(f"  Traditional: {trad}")
    print(f"  Source type: {source_type}")
    print("=" * 70)

    # ── Input validation ─────────────────────────────────────
    valid = check_input_images(images, source_type)
    if not valid and source_type == "lossless":
        print("  Aborting. Fix your input or use --source-type lossy.\n")
        return

    results: Dict[str, List[Dict]] = {}

    # ── Traditional codecs ───────────────────────────────────
    results["JPEG"] = run_traditional_codec("JPEG", JPEG_QUALITIES, images, n, device)
    results["WebP"] = run_traditional_codec("WebP", WEBP_QUALITIES, images, n, device)

    if AVIF_AVAILABLE:
        results["AVIF"] = run_traditional_codec("AVIF", AVIF_QUALITIES, images, n, device)
    else:
        print("\n  Skipping AVIF (not installed)")

    # ── Neural models ─────────────────────────────────────────
    for model_name, model_fn in NEURAL_MODELS.items():
        print(f"\n{'─'*60}\n  {model_name}\n{'─'*60}")
        results[model_name] = []

        for q in NEURAL_QUALITIES_BY_MODEL[model_name]:
            print(f"  Loading quality {q}...", flush=True)
            model = model_fn(quality=q, pretrained=True).eval().to(device)
            model.update()

            bpps, psnrs, ms_ssims, lpips_scores, encs, decs = [], [], [], [], [], []
            for i, img_path in enumerate(images, 1):
                try:
                    r = compress_neural(
                        Image.open(img_path).convert("RGB"), model, device
                    )
                    bpps.append(r["bpp"])
                    psnrs.append(r["psnr"])
                    ms_ssims.append(r["ms_ssim"])
                    lpips_scores.append(r["lpips"])
                    encs.append(r["encode_ms"])
                    decs.append(r["decode_ms"])
                    extra_metrics = metric_suffix(r)
                    print(f"    [{i:2d}/{n}] {img_path.name} | "
                          f"BPP: {r['bpp']:.4f} | PSNR: {r['psnr']:.2f} dB | "
                          f"{extra_metrics + ' | ' if extra_metrics else ''}"
                          f"Enc: {r['encode_ms']:.0f}ms | Dec: {r['decode_ms']:.0f}ms")
                except Exception as e:
                    print(f"    [{i:2d}/{n}] {img_path.name} FAILED: {e}")

            if not bpps:
                continue

            entry = {
                "quality":       q,
                "avg_bpp":       round(float(np.mean(bpps)),  4),
                "avg_psnr":      round(float(np.mean(psnrs)), 2),
                "avg_ms_ssim":   average_or_none(ms_ssims, 4),
                "avg_lpips":     average_or_none(lpips_scores, 4),
                "avg_encode_ms": round(float(np.mean(encs)),  1),
                "avg_decode_ms": round(float(np.mean(decs)),  1),
                "n_images":      len(bpps),
            }
            results[model_name].append(entry)
            print(f"  ✓ q={q} avg → BPP: {entry['avg_bpp']:.4f} | "
                  f"PSNR: {entry['avg_psnr']:.2f} dB | "
                  f"MS-SSIM: {format_metric(entry['avg_ms_ssim'], '.4f')} | "
                  f"LPIPS: {format_metric(entry['avg_lpips'], '.4f')} | "
                  f"Enc: {entry['avg_encode_ms']:.0f}ms | "
                  f"Dec: {entry['avg_decode_ms']:.0f}ms\n")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ── Save JSON ─────────────────────────────────────────────
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Raw results saved → {output_json}")

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  (averaged over all quality levels and all images)")
    print(f"{'='*70}")
    print(f"  {'Codec':<22} {'Avg BPP':<10} {'Avg PSNR':<10} "
          f"{'Avg MS-SSIM':<13} {'Avg LPIPS':<11} "
          f"{'Avg Enc (ms)':<13} {'Avg Dec (ms)'}")
    print(f"  {'─'*92}")
    for codec_name, pts in results.items():
        if not pts:
            continue
        print(f"  {codec_name:<22} "
              f"{np.mean([p['avg_bpp']       for p in pts]):<10.4f} "
              f"{np.mean([p['avg_psnr']      for p in pts]):<10.2f} "
              f"{format_metric(average_or_none([p.get('avg_ms_ssim') for p in pts]), '.4f'):<13} "
              f"{format_metric(average_or_none([p.get('avg_lpips') for p in pts]), '.4f'):<11} "
              f"{np.mean([p['avg_encode_ms'] for p in pts]):<13.1f} "
              f"{np.mean([p['avg_decode_ms'] for p in pts]):.1f}")

    # ── Plots ─────────────────────────────────────────────────
    print()
    plot_rd_curves(results, "psnr", output_rd, device, source_type)
    plot_rd_curves(results, "ms_ssim", output_msssim, device, source_type)
    plot_rd_curves(results, "lpips", output_lpips, device, source_type)
    plot_runtime(results, output_rt, device)

    bd_table = compute_bd_rate_table(results)
    with open(output_bd_json, "w") as f:
        json.dump(bd_table, f, indent=2)
    print(f"  ✅ BD-Rate results saved → {output_bd_json}")
    save_bd_rate_table_image(bd_table, output_bd_table)

    if "bmshj2018-factorized" in bd_table:
        print(f"\n{'='*70}")
        print("BD-RATE VS BMSHJ2018-FACTORIZED  (matched PSNR)")
        print(f"{'='*70}")
        for codec_name in bd_table:
            if codec_name == "bmshj2018-factorized":
                continue
            value = bd_table[codec_name].get("bmshj2018-factorized")
            if value is None:
                print(f"  {codec_name:<18} n/a")
                continue
            print(f"  {codec_name:<18} {value:+.1f}%")

    # ── AVIF-specific note ────────────────────────────────────
    if AVIF_AVAILABLE and "AVIF" in results:
        print("""
  ℹ️  AVIF NOTE FOR THESIS:
  AVIF uses AV1 video compression technology — the same codec
  that appears as "AV1" in the CompressAI paper benchmark graph.
  On that graph, AV1 sits INSIDE the neural cluster, nearly
  matching cheng2020. Your AVIF results should show the same.
  This is the strongest argument against neural compression
  for practical use — acknowledge it honestly in your thesis.
        """)

    print(f"""
{'='*70}
HOW TO READ YOUR RESULTS
{'='*70}

rd_curve.png
  Figure 1: BPP vs PSNR
  Higher + further left = better codec

rd_msssim.png
  Figure 2: BPP vs MS-SSIM
  Better captures structural perceptual quality than PSNR

rd_lpips.png
  Figure 3: BPP vs LPIPS
  Lower is better

bd_rate_table.png / bd_rate_results.json
  Figure 4: pairwise BD-Rate at matched PSNR
  Negative means the column codec needs fewer bits than the row codec

runtime_chart.png (log scale)
  JPEG/WebP:  5–150ms
  AVIF:       200–2000ms  ← slower than JPEG/WebP, faster than neural
  Neural CPU: can be much slower than traditional codecs
  Neural GPU: 50–500ms

THESIS TAKEAWAY:
  PSNR alone may hide perceptual differences between codecs
  MS-SSIM and LPIPS help show whether neural compression looks better
  BD-Rate gives you one paper-style number for the full curve
{'='*70}
""")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark neural vs traditional image compression"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=str(EXAMPLES_DIR / "kodak"),
        help="Folder containing input images"
    )
    parser.add_argument(
        "--source-type", type=str,
        choices=["lossless", "lossy", "auto"],
        default="auto",
        help=(
            "lossless = PNG/TIFF, valid for standard benchmark; "
            "lossy    = JPEG/AVIF input, photographer experiment only; "
            "auto     = detect and warn (default)"
        )
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda (auto-detected by default)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit to N images — use 5 for a quick sanity check"
    )
    parser.add_argument("--output-json", type=str, default="benchmark_results.json")
    parser.add_argument("--output-rd",   type=str, default="rd_curve.png")
    parser.add_argument("--output-msssim", type=str, default="rd_msssim.png")
    parser.add_argument("--output-lpips", type=str, default="rd_lpips.png")
    parser.add_argument("--output-bd-json", type=str, default="bd_rate_results.json")
    parser.add_argument("--output-bd-table", type=str, default="bd_rate_table.png")
    parser.add_argument("--output-rt",   type=str, default="runtime_chart.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print("   Neural models will be ~100x faster than CPU.\n")
    else:
        print("\n⚠️  No GPU — running on CPU.")
        print("   LPIPS and neural compression can be slow on CPU.")
        print("   Tip: use --max-images 5 for a quick test first.\n")

    run_benchmark(
        input_dir=Path(args.input_dir),
        device=args.device,
        max_images=args.max_images,
        source_type=args.source_type,
        output_json=args.output_json,
        output_rd=args.output_rd,
        output_msssim=args.output_msssim,
        output_lpips=args.output_lpips,
        output_bd_json=args.output_bd_json,
        output_bd_table=args.output_bd_table,
        output_rt=args.output_rt,
    )


if __name__ == "__main__":
    main()
