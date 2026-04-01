"""
Neural Compression Quality Sweep

Sweeps through all quality levels (1-8) for neural compression and reports
bitrate (bpp) and quality (PSNR, MS-SSIM) for each image.

Usage:
    python compression_sweep.py --input-dir ./high_quality_images
    python compression_sweep.py --input-dir ./high_quality_images --qualities 3 4 5
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import compressai
from compressai.ops import compute_padding
from compressai.zoo import models

EXAMPLES_DIR = Path(__file__).resolve().parent

# AVIF support
try:
    import pillow_avif
    AVIF_AVAILABLE = True
except ImportError:
    try:
        import pillow_heif
        pillow_heif.register_avif_opener()
        AVIF_AVAILABLE = True
    except ImportError:
        AVIF_AVAILABLE = False
        print("Warning: AVIF support not available. Install pillow-heif: pip install pillow-heif")


@dataclass
class CompressionMetrics:
    """Metrics from compressing a single image at a single quality level"""
    image: str
    quality: int
    psnr: float
    ms_ssim: float
    bpp: float
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    encode_time_ms: float
    decode_time_ms: float
    width: int
    height: int


def psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate PSNR between two tensors"""
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0) - 10 * torch.log10(mse).item()


def ms_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate MS-SSIM between two tensors"""
    try:
        from pytorch_msssim import ms_ssim as compute_ms_ssim
        return compute_ms_ssim(original, reconstructed, data_range=1.0).item()
    except ImportError:
        return 0.0


def compress_image(
    img: Image.Image,
    model,
    device: str = "cpu"
) -> Dict:
    """Compress an image and return raw metrics"""
    to_tensor = transforms.ToTensor()
    
    x = to_tensor(img).unsqueeze(0).to(device)
    h, w = x.size(2), x.size(3)
    
    # Pad to multiple of 64
    pad, unpad = compute_padding(h, w, min_div=64)
    x_padded = F.pad(x, pad)
    
    with torch.inference_mode():
        # Encode
        t0 = time.perf_counter()
        out_enc = model.compress(x_padded)
        encode_time = (time.perf_counter() - t0) * 1000  # ms
        
        # Compressed size
        compressed_bytes = sum(len(s[0]) for s in out_enc["strings"])
        
        # Decode
        t0 = time.perf_counter()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        decode_time = (time.perf_counter() - t0) * 1000  # ms
        
        x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)
    
    psnr_val = psnr(x, x_hat)
    ms_ssim_val = ms_ssim(x, x_hat)
    bpp = compressed_bytes * 8 / (h * w)
    
    return {
        "psnr": psnr_val,
        "ms_ssim": ms_ssim_val,
        "bpp": bpp,
        "compressed_bytes": compressed_bytes,
        "encode_time_ms": encode_time,
        "decode_time_ms": decode_time,
        "width": w,
        "height": h,
    }


def run_sweep(
    input_dir: Path,
    output_file: Path,
    model_name: str = "bmshj2018-factorized",
    quality_levels: List[int] = None,
    device: str = "cpu"
) -> Dict:
    """Run compression sweep across all images and quality levels"""
    
    if quality_levels is None:
        quality_levels = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Find images
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    if AVIF_AVAILABLE:
        extensions.extend([".avif", ".heif", ".heic"])
    
    images = sorted([p for p in input_dir.iterdir() 
                     if p.suffix.lower() in extensions])
    
    if not images:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {extensions}")
        return {}
    
    print("=" * 80)
    print("NEURAL COMPRESSION QUALITY SWEEP")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Quality levels: {quality_levels}")
    print(f"Images: {len(images)} from {input_dir}")
    print(f"Device: {device}")
    print("=" * 80)
    
    all_results = []
    summary_by_quality = {q: {"bpp": [], "psnr": [], "ms_ssim": [], "ratio": [], "encode_time": [], "decode_time": []} 
                          for q in quality_levels}
    
    # Initialize entropy coder
    compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
    
    for q in quality_levels:
        print(f"\n{'='*60}")
        print(f"Quality Level {q}")
        print(f"{'='*60}")
        
        # Load model for this quality
        model = models[model_name](quality=q, metric="mse", pretrained=True)
        model = model.to(device).eval()
        model.update()
        
        for i, img_path in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] {img_path.name}...", end=" ", flush=True)
            
            try:
                img = Image.open(img_path).convert("RGB")
                original_size = img_path.stat().st_size
                
                metrics = compress_image(img, model, device)
                
                result = CompressionMetrics(
                    image=img_path.name,
                    quality=q,
                    psnr=metrics["psnr"],
                    ms_ssim=metrics["ms_ssim"],
                    bpp=metrics["bpp"],
                    original_size_bytes=original_size,
                    compressed_size_bytes=metrics["compressed_bytes"],
                    compression_ratio=original_size / metrics["compressed_bytes"],
                    encode_time_ms=metrics["encode_time_ms"],
                    decode_time_ms=metrics["decode_time_ms"],
                    width=metrics["width"],
                    height=metrics["height"],
                )
                
                all_results.append(asdict(result))
                
                summary_by_quality[q]["bpp"].append(metrics["bpp"])
                summary_by_quality[q]["psnr"].append(metrics["psnr"])
                summary_by_quality[q]["ms_ssim"].append(metrics["ms_ssim"])
                summary_by_quality[q]["ratio"].append(result.compression_ratio)
                summary_by_quality[q]["encode_time"].append(metrics["encode_time_ms"])
                summary_by_quality[q]["decode_time"].append(metrics["decode_time_ms"])
                
                print(f"{metrics['bpp']:.4f} bpp | {metrics['psnr']:.2f} dB | "
                      f"MS-SSIM: {metrics['ms_ssim']:.4f} | "
                      f"Enc: {metrics['encode_time_ms']:.0f}ms | Dec: {metrics['decode_time_ms']:.0f}ms")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY BY QUALITY LEVEL")
    print("=" * 80)
    print(f"{'Quality':<8} {'Avg BPP':<10} {'Avg PSNR':<10} {'Avg MS-SSIM':<12} {'Enc (ms)':<10} {'Dec (ms)':<10}")
    print("-" * 80)
    
    summary_table = {}
    for q in quality_levels:
        data = summary_by_quality[q]
        if data["bpp"]:
            avg_bpp = sum(data["bpp"]) / len(data["bpp"])
            avg_psnr = sum(data["psnr"]) / len(data["psnr"])
            avg_ssim = sum(data["ms_ssim"]) / len(data["ms_ssim"])
            avg_ratio = sum(data["ratio"]) / len(data["ratio"])
            avg_enc = sum(data["encode_time"]) / len(data["encode_time"])
            avg_dec = sum(data["decode_time"]) / len(data["decode_time"])
            
            summary_table[q] = {
                "avg_bpp": avg_bpp,
                "avg_psnr": avg_psnr,
                "avg_ms_ssim": avg_ssim,
                "avg_compression_ratio": avg_ratio,
                "avg_encode_time_ms": avg_enc,
                "avg_decode_time_ms": avg_dec,
            }
            
            print(f"{q:<8} {avg_bpp:<10.4f} {avg_psnr:<10.2f} {avg_ssim:<12.4f} {avg_enc:<10.0f} {avg_dec:<10.0f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    for q, s in summary_table.items():
        if s["avg_psnr"] >= 30:
            rec = "✓ Good quality (PSNR ≥ 30 dB)"
        elif s["avg_psnr"] >= 28:
            rec = "~ Acceptable quality"
        else:
            rec = "⚠ Low quality (visible artifacts likely)"
        
        print(f"Quality {q}: {s['avg_psnr']:.1f} dB, {s['avg_bpp']:.3f} bpp -> {rec}")
    
    # Best quality for space savings
    good_qualities = [(q, s) for q, s in summary_table.items() if s["avg_psnr"] >= 30]
    if good_qualities:
        best = min(good_qualities, key=lambda x: x[1]["avg_bpp"])
        print(f"\n★ Best space-saving quality with good quality (≥30 dB): Quality {best[0]}")
        print(f"  -> {best[1]['avg_bpp']:.3f} bpp at {best[1]['avg_psnr']:.1f} dB")
    
    # Save results
    output_data = {
        "config": {
            "model": model_name,
            "quality_levels": quality_levels,
            "input_dir": str(input_dir),
            "num_images": len(images),
            "device": device,
        },
        "per_image_results": all_results,
        "summary_by_quality": summary_table,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Neural compression quality sweep")
    parser.add_argument("--input-dir", type=str, 
                        default=str(EXAMPLES_DIR / "high_quality_images"),
                        help="Directory containing images")
    parser.add_argument("--output", type=str, default="sweep_results.json",
                        help="Output JSON file")
    parser.add_argument("--model", type=str, default="bmshj2018-factorized",
                        choices=["bmshj2018-factorized", "bmshj2018-hyperprior"],
                        help="Neural compression model")
    parser.add_argument("--qualities", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Quality levels to test")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    run_sweep(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output),
        model_name=args.model,
        quality_levels=args.qualities,
        device=args.device,
    )


if __name__ == "__main__":
    main()
