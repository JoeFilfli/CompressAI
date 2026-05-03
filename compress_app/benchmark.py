"""
Benchmark all models (built-in + custom .pth checkpoints) on test_images/.
Reports avg PSNR, avg MS-SSIM, timing, and size metrics.
"""

import io
import time
from pathlib import Path

import torch
from PIL import Image
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

APP_DIR = Path(__file__).parent
IMAGE_DIR = APP_DIR / "test_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUILTIN_CONFIGS = [
    ("bmshj2018-factorized", 4),
    ("bmshj2018-factorized", 6),
    ("bmshj2018-hyperprior", 4),
    ("bmshj2018-hyperprior", 6),
    ("cheng2020-attn", 3),
    ("cheng2020-attn", 6),
]

CODEC_CONFIGS = [
    ("AVIF", 50),
    ("AVIF", 80),
    ("WEBP", 50),
    ("WEBP", 85),
]


def run_model(model, images):
    records = []
    n = len(images)
    for i, img_path in enumerate(images, 1):
        img = load_image(img_path)
        orig_bytes = img_path.stat().st_size
        x = to_tensor(img).unsqueeze(0).to(DEVICE)
        x_padded, orig_h, orig_w = pad_to_multiple(x)

        t0 = time.time()
        with torch.no_grad():
            compressed = model.compress(x_padded)
        comp_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            decompressed = model.decompress(compressed["strings"], compressed["shape"])
        decomp_time = time.time() - t0

        x_hat = crop_to_original(decompressed["x_hat"], orig_h, orig_w)
        x_orig = crop_to_original(x_padded, orig_h, orig_w)

        comp_bytes = sum(len(s[0]) for s in compressed["strings"])
        psnr   = compute_psnr(x_orig, x_hat)
        msssim = compute_msssim(x_orig, x_hat)
        total_t = comp_time + decomp_time

        print(f"  [{i:>3}/{n}] {img_path.name:<16}  {total_t:.2f}s "
              f"(c={comp_time:.2f}s d={decomp_time:.2f}s)  "
              f"PSNR={psnr:.2f}dB  MS-SSIM={msssim:.4f}", flush=True)

        records.append({
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "psnr":       psnr,
            "msssim":     msssim,
            "comp_time":  comp_time,
            "decomp_time":decomp_time,
        })
    return records


def summarise(records):
    n = len(records)
    total_in     = sum(r["orig_bytes"]  for r in records)
    total_out    = sum(r["comp_bytes"]  for r in records)
    total_comp   = sum(r["comp_time"]   for r in records)
    total_decomp = sum(r["decomp_time"] for r in records)
    return {
        "n":            n,
        "in_mb":        total_in  / 1024**2,
        "comp_mb":      total_out / 1024**2,
        "ratio":        total_in / total_out if total_out else 0,
        "avg_psnr":     sum(r["psnr"]   for r in records) / n,
        "avg_msssim":   sum(r["msssim"] for r in records) / n,
        "total_t":      total_comp + total_decomp,
        "avg_comp_t":   total_comp   / n,
        "avg_decomp_t": total_decomp / n,
        "avg_t":        (total_comp + total_decomp) / n,
    }


def run_codec(fmt, quality, images):
    records = []
    n = len(images)
    for i, img_path in enumerate(images, 1):
        img = load_image(img_path)
        orig_bytes = img_path.stat().st_size

        buf = io.BytesIO()
        t0 = time.time()
        img.save(buf, format=fmt, quality=quality)
        comp_time = time.time() - t0
        comp_bytes = buf.tell()

        buf.seek(0)
        t0 = time.time()
        decoded = Image.open(buf).convert("RGB")
        decomp_time = time.time() - t0

        x_orig = to_tensor(img).unsqueeze(0)
        x_hat  = to_tensor(decoded).unsqueeze(0)
        psnr   = compute_psnr(x_orig, x_hat)
        msssim = compute_msssim(x_orig, x_hat)
        total_t = comp_time + decomp_time

        print(f"  [{i:>3}/{n}] {img_path.name:<16}  {total_t:.2f}s "
              f"(c={comp_time:.2f}s d={decomp_time:.2f}s)  "
              f"PSNR={psnr:.2f}dB  MS-SSIM={msssim:.4f}", flush=True)

        records.append({
            "orig_bytes":  orig_bytes,
            "comp_bytes":  comp_bytes,
            "psnr":        psnr,
            "msssim":      msssim,
            "comp_time":   comp_time,
            "decomp_time": decomp_time,
        })
    return records


def print_detail(label, s):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Images        : {s['n']}")
    print(f"  Input         : {s['in_mb']:.3f} MB")
    print(f"  Compressed    : {s['comp_mb']:.3f} MB")
    print(f"  Ratio         : {s['ratio']:.2f}x")
    print(f"  Avg PSNR      : {s['avg_psnr']:.2f} dB")
    print(f"  Avg MS-SSIM   : {s['avg_msssim']:.4f}")
    print(f"  Total time    : {s['total_t']:.2f}s")
    print(f"  Avg/image     : {s['avg_t']:.2f}s")
    print(f"  Avg compress  : {s['avg_comp_t']:.2f}s")
    print(f"  Avg decompress: {s['avg_decomp_t']:.2f}s")


def main():
    images = scan_images(IMAGE_DIR, recursive=False)
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Device : {DEVICE}")
    print(f"Images : {len(images)}  ({IMAGE_DIR})")

    rows = []

    # ── Built-in models ───────────────────────────────────────────────────────
    for model_name, quality in BUILTIN_CONFIGS:
        label = f"{model_name}  q{quality}"
        print(f"\n[{label}] loading…")
        try:
            model = load_model(model_name, quality, DEVICE)
            s = summarise(run_model(model, images))
            print_detail(label, s)
            rows.append((label, s))
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Custom checkpoints (.pth) ─────────────────────────────────────────────
    for ckpt in list_custom_checkpoints(APP_DIR):
        path    = ckpt["path"]
        quality = ckpt["quality"] or 4
        detected = infer_checkpoint_base(path)
        base    = detected.get("base") or "tiny-hyperprior"
        label   = ckpt["label"]
        print(f"\n[{label}] loading…  (base={base}, q={quality})")
        try:
            model = load_model(base, quality, DEVICE, checkpoint_path=path)
            s = summarise(run_model(model, images))
            print_detail(label, s)
            rows.append((label, s))
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Traditional codecs (AVIF, WebP) ──────────────────────────────────────
    for fmt, quality in CODEC_CONFIGS:
        label = f"{fmt.lower()}  q{quality}"
        print(f"\n[{label}] encoding…")
        try:
            s = summarise(run_codec(fmt, quality, images))
            print_detail(label, s)
            rows.append((label, s))
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 52
    SEP = W + 83
    print(f"\n\n{'═'*SEP}")
    print(
        f"{'Model':<{W}} {'PSNR(dB)':>9} {'MS-SSIM':>9}"
        f" {'In(MB)':>8} {'Comp(MB)':>9} {'Ratio':>7}"
        f" {'TotalT(s)':>10} {'AvgC(s)':>8} {'AvgD(s)':>8}"
    )
    print(f"{'─'*SEP}")
    for label, s in rows:
        print(
            f"{label:<{W}} {s['avg_psnr']:>9.2f} {s['avg_msssim']:>9.4f}"
            f" {s['in_mb']:>8.3f} {s['comp_mb']:>9.3f} {s['ratio']:>7.2f}"
            f" {s['total_t']:>10.2f} {s['avg_comp_t']:>8.2f} {s['avg_decomp_t']:>8.2f}"
        )
    print(f"{'═'*SEP}")


if __name__ == "__main__":
    main()
