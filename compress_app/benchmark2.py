"""
Benchmark AVIF and WebP at slowest/highest-effort settings on test_images/.
Reports avg PSNR, avg MS-SSIM, timing, and size metrics.

AVIF: speed=0, max_threads=1, subsampling="4:4:4"  (maximum effort)
WebP: method=6                                       (maximum effort)
"""

import io
import time
from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import to_tensor

from utils import (
    compute_msssim,
    compute_psnr,
    load_image,
    scan_images,
)

APP_DIR   = Path(__file__).parent
IMAGE_DIR = APP_DIR / "test_images"

CODEC_CONFIGS = [
    ("AVIF", 50),
    ("AVIF", 80),
    ("WEBP", 50),
    ("WEBP", 85),
]


def run_codec(fmt, quality, images):
    records = []
    n = len(images)
    for i, img_path in enumerate(images, 1):
        img        = load_image(img_path)
        orig_bytes = img_path.stat().st_size

        buf = io.BytesIO()
        t0  = time.time()
        if fmt == "AVIF":
            img.save(
                buf,
                format="AVIF",
                quality=quality,
                speed=0,
                max_threads=1,
                subsampling="4:4:4",
            )
        else:  # WEBP
            img.save(buf, format="WEBP", quality=quality, method=6)
        comp_time  = time.time() - t0
        comp_bytes = buf.tell()

        buf.seek(0)
        t0      = time.time()
        decoded = Image.open(buf).convert("RGB")
        decomp_time = time.time() - t0

        x_orig  = to_tensor(img).unsqueeze(0)
        x_hat   = to_tensor(decoded).unsqueeze(0)
        psnr    = compute_psnr(x_orig, x_hat)
        msssim  = compute_msssim(x_orig, x_hat)
        total_t = comp_time + decomp_time

        print(
            f"  [{i:>3}/{n}] {img_path.name:<16}  {total_t:.2f}s "
            f"(c={comp_time:.2f}s d={decomp_time:.2f}s)  "
            f"PSNR={psnr:.2f}dB  MS-SSIM={msssim:.4f}",
            flush=True,
        )

        records.append({
            "orig_bytes":  orig_bytes,
            "comp_bytes":  comp_bytes,
            "psnr":        psnr,
            "msssim":      msssim,
            "comp_time":   comp_time,
            "decomp_time": decomp_time,
        })
    return records


def summarise(records):
    n            = len(records)
    total_in     = sum(r["orig_bytes"]  for r in records)
    total_out    = sum(r["comp_bytes"]  for r in records)
    total_comp   = sum(r["comp_time"]   for r in records)
    total_decomp = sum(r["decomp_time"] for r in records)
    return {
        "n":            n,
        "in_mb":        total_in  / 1024**2,
        "comp_mb":      total_out / 1024**2,
        "ratio":        total_in  / total_out if total_out else 0,
        "avg_psnr":     sum(r["psnr"]   for r in records) / n,
        "avg_msssim":   sum(r["msssim"] for r in records) / n,
        "total_t":      total_comp + total_decomp,
        "avg_comp_t":   total_comp   / n,
        "avg_decomp_t": total_decomp / n,
        "avg_t":        (total_comp + total_decomp) / n,
    }


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
    images = scan_images(IMAGE_DIR, recursive=False)[:15]
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Images : {len(images)}  ({IMAGE_DIR})")
    print(f"AVIF   : speed=0, max_threads=1, subsampling=4:4:4")
    print(f"WebP   : method=6")

    rows = []

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
    W   = 20
    SEP = W + 83
    print(f"\n\n{'═'*SEP}")
    print(
        f"{'Codec':<{W}} {'PSNR(dB)':>9} {'MS-SSIM':>9}"
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
