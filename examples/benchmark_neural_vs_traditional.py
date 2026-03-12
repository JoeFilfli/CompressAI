"""
Benchmark: Neural Compression (bmshj2018) vs Traditional Codecs (JPEG/WebP/AVIF)

This script verifies the claim:
"Neural compression beats JPEG/WebP/AVIF by 40-60% at same quality"

Methodology:
- At matched PSNR quality, compare the bits-per-pixel (BPP) between codecs
- Bitrate savings = (traditional_bpp - neural_bpp) / traditional_bpp * 100%

Usage:
    python benchmark_neural_vs_traditional.py --input-dir ./kodak --quality-levels 1 2 3 4 5 6 7 8
"""

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import compressai
from compressai.ops import compute_padding
from compressai.zoo import models

# Optional: AVIF support
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
        print("Warning: AVIF support not available. Install pillow-avif-plugin or pillow-heif for AVIF benchmarks.")


@dataclass
class CompressionResult:
    """Result of compressing a single image"""
    psnr: float
    ms_ssim: float
    bpp: float
    encode_time: float
    decode_time: float
    file_size_bytes: int


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
        return 0.0  # Return 0 if pytorch_msssim not available


class NeuralCodec:
    """Neural image codec using CompressAI models"""
    
    def __init__(self, model_name: str, quality: int, metric: str = "mse", device: str = "cpu"):
        self.model_name = model_name
        self.quality = quality
        self.device = device
        
        compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
        self.model = models[model_name](quality=quality, metric=metric, pretrained=True)
        self.model = self.model.to(device).eval()
        self.model.update()
    
    def compress_image(self, img: Image.Image) -> CompressionResult:
        """Compress an image and return metrics"""
        to_tensor = transforms.ToTensor()
        
        x = to_tensor(img).unsqueeze(0).to(self.device)
        h, w = x.size(2), x.size(3)
        
        # Pad to multiple of 64
        pad, unpad = compute_padding(h, w, min_div=64)
        x_padded = F.pad(x, pad)
        
        with torch.inference_mode():
            # Encode
            t0 = time.perf_counter()
            out_enc = self.model.compress(x_padded)
            encode_time = time.perf_counter() - t0
            
            # Calculate compressed size
            compressed_bytes = sum(len(s[0]) for s in out_enc["strings"])
            
            # Decode
            t0 = time.perf_counter()
            out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])
            decode_time = time.perf_counter() - t0
            
            x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)
        
        # Calculate metrics
        psnr_val = psnr(x, x_hat)
        ms_ssim_val = ms_ssim(x, x_hat)
        bpp = compressed_bytes * 8 / (h * w)
        
        return CompressionResult(
            psnr=psnr_val,
            ms_ssim=ms_ssim_val,
            bpp=bpp,
            encode_time=encode_time,
            decode_time=decode_time,
            file_size_bytes=compressed_bytes
        )


class TraditionalCodec:
    """Traditional image codec (JPEG, WebP, AVIF)"""
    
    def __init__(self, codec_name: str, quality: int):
        self.codec_name = codec_name.upper()
        self.quality = quality
    
    def compress_image(self, img: Image.Image) -> CompressionResult:
        """Compress image and return metrics"""
        h, w = img.size[1], img.size[0]
        to_tensor = transforms.ToTensor()
        original_tensor = to_tensor(img).unsqueeze(0)
        
        # Compress to memory buffer
        buffer = io.BytesIO()
        
        t0 = time.perf_counter()
        if self.codec_name == "JPEG":
            img.save(buffer, format="JPEG", quality=self.quality, subsampling=0, optimize=True)
        elif self.codec_name == "WEBP":
            img.save(buffer, format="WEBP", quality=self.quality, method=6)
        elif self.codec_name == "AVIF":
            if not AVIF_AVAILABLE:
                raise RuntimeError("AVIF not available")
            img.save(buffer, format="AVIF", quality=self.quality)
        else:
            raise ValueError(f"Unknown codec: {self.codec_name}")
        encode_time = time.perf_counter() - t0
        
        compressed_bytes = buffer.tell()
        
        # Decompress
        buffer.seek(0)
        t0 = time.perf_counter()
        reconstructed = Image.open(buffer).convert("RGB")
        decode_time = time.perf_counter() - t0
        
        reconstructed_tensor = to_tensor(reconstructed).unsqueeze(0)
        
        # Calculate metrics
        psnr_val = psnr(original_tensor, reconstructed_tensor)
        ms_ssim_val = ms_ssim(original_tensor, reconstructed_tensor)
        bpp = compressed_bytes * 8 / (h * w)
        
        return CompressionResult(
            psnr=psnr_val,
            ms_ssim=ms_ssim_val,
            bpp=bpp,
            encode_time=encode_time,
            decode_time=decode_time,
            file_size_bytes=compressed_bytes
        )


def find_quality_for_target_psnr(
    img: Image.Image,
    codec_name: str,
    target_psnr: float,
    tolerance: float = 0.5,
    min_quality: int = 1,
    max_quality: int = 100
) -> Tuple[int, CompressionResult]:
    """Binary search to find quality setting that achieves target PSNR"""
    
    best_quality = min_quality
    best_result = None
    best_diff = float('inf')
    
    lo, hi = min_quality, max_quality
    
    while lo <= hi:
        mid = (lo + hi) // 2
        
        try:
            codec = TraditionalCodec(codec_name, mid)
            result = codec.compress_image(img)
            diff = abs(result.psnr - target_psnr)
            
            if diff < best_diff:
                best_diff = diff
                best_quality = mid
                best_result = result
            
            if result.psnr < target_psnr:
                lo = mid + 1
            else:
                hi = mid - 1
                
        except Exception as e:
            # Some quality values may fail
            lo = mid + 1
    
    # Refine around best quality
    for q in range(max(min_quality, best_quality - 5), min(max_quality + 1, best_quality + 6)):
        try:
            codec = TraditionalCodec(codec_name, q)
            result = codec.compress_image(img)
            diff = abs(result.psnr - target_psnr)
            
            if diff < best_diff:
                best_diff = diff
                best_quality = q
                best_result = result
        except:
            pass
    
    return best_quality, best_result


def benchmark_image(
    img_path: Path,
    neural_model: str,
    neural_quality: int,
    device: str = "cpu"
) -> Dict:
    """Benchmark a single image with neural and traditional codecs"""
    
    img = Image.open(img_path).convert("RGB")
    
    # Neural compression
    neural_codec = NeuralCodec(neural_model, neural_quality, device=device)
    neural_result = neural_codec.compress_image(img)
    
    target_psnr = neural_result.psnr
    
    results = {
        "image": img_path.name,
        "neural_model": neural_model,
        "neural_quality": neural_quality,
        "target_psnr": target_psnr,
        "neural": {
            "psnr": neural_result.psnr,
            "ms_ssim": neural_result.ms_ssim,
            "bpp": neural_result.bpp,
            "encode_time": neural_result.encode_time,
            "decode_time": neural_result.decode_time,
        },
        "traditional": {}
    }
    
    # Traditional codecs - match quality (PSNR)
    traditional_codecs = ["JPEG", "WEBP"]
    if AVIF_AVAILABLE:
        traditional_codecs.append("AVIF")
    
    for codec_name in traditional_codecs:
        try:
            quality, trad_result = find_quality_for_target_psnr(
                img, codec_name, target_psnr
            )
            
            # Calculate savings
            savings = (trad_result.bpp - neural_result.bpp) / trad_result.bpp * 100
            
            results["traditional"][codec_name] = {
                "quality_setting": quality,
                "psnr": trad_result.psnr,
                "psnr_diff": abs(trad_result.psnr - target_psnr),
                "ms_ssim": trad_result.ms_ssim,
                "bpp": trad_result.bpp,
                "encode_time": trad_result.encode_time,
                "decode_time": trad_result.decode_time,
                "bitrate_savings_percent": savings
            }
        except Exception as e:
            results["traditional"][codec_name] = {"error": str(e)}
    
    return results


def run_benchmark(
    input_dir: Path,
    output_file: Path,
    neural_model: str = "bmshj2018-factorized",
    quality_levels: List[int] = None,
    device: str = "cpu",
    max_images: int = None
):
    """Run full benchmark across quality levels and images"""
    
    if quality_levels is None:
        quality_levels = [1, 2, 3, 4, 5, 6, 7, 8]
    
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".bmp"]])
    
    if max_images:
        images = images[:max_images]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"=" * 80)
    print(f"NEURAL COMPRESSION vs TRADITIONAL CODECS BENCHMARK")
    print(f"=" * 80)
    print(f"Model: {neural_model}")
    print(f"Quality levels: {quality_levels}")
    print(f"Images: {len(images)} from {input_dir}")
    print(f"Device: {device}")
    print(f"=" * 80)
    
    all_results = {
        "config": {
            "model": neural_model,
            "quality_levels": quality_levels,
            "images": [str(p) for p in images],
            "device": device,
        },
        "per_image_results": [],
        "summary": {}
    }
    
    # Run benchmarks
    for q in quality_levels:
        print(f"\n{'='*60}")
        print(f"Quality Level {q}")
        print(f"{'='*60}")
        
        q_results = []
        
        for i, img_path in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] {img_path.name}...", end=" ", flush=True)
            
            result = benchmark_image(img_path, neural_model, q, device)
            q_results.append(result)
            
            neural_bpp = result["neural"]["bpp"]
            neural_psnr = result["neural"]["psnr"]
            print(f"Neural: {neural_bpp:.4f} bpp @ {neural_psnr:.2f} dB")
            
            for codec, data in result["traditional"].items():
                if "error" not in data:
                    print(f"       {codec}: {data['bpp']:.4f} bpp @ {data['psnr']:.2f} dB -> " +
                          f"Savings: {data['bitrate_savings_percent']:.1f}%")
        
        all_results["per_image_results"].extend(q_results)
    
    # Calculate summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY: Average Bitrate Savings at Matched PSNR")
    print(f"{'='*80}")
    
    summary_by_quality = {}
    
    for q in quality_levels:
        q_data = [r for r in all_results["per_image_results"] if r["neural_quality"] == q]
        
        summary_by_quality[q] = {
            "neural_avg_bpp": sum(r["neural"]["bpp"] for r in q_data) / len(q_data),
            "neural_avg_psnr": sum(r["neural"]["psnr"] for r in q_data) / len(q_data),
            "codecs": {}
        }
        
        for codec in ["JPEG", "WEBP", "AVIF"]:
            valid = [r["traditional"].get(codec) for r in q_data 
                     if r["traditional"].get(codec) and "error" not in r["traditional"].get(codec, {})]
            
            if valid:
                avg_savings = sum(v["bitrate_savings_percent"] for v in valid) / len(valid)
                avg_bpp = sum(v["bpp"] for v in valid) / len(valid)
                avg_psnr = sum(v["psnr"] for v in valid) / len(valid)
                avg_psnr_diff = sum(v["psnr_diff"] for v in valid) / len(valid)
                
                summary_by_quality[q]["codecs"][codec] = {
                    "avg_savings_percent": avg_savings,
                    "avg_bpp": avg_bpp,
                    "avg_psnr": avg_psnr,
                    "avg_psnr_diff": avg_psnr_diff,
                }
    
    all_results["summary"] = summary_by_quality
    
    # Print summary table
    print(f"\n{'Quality':<10} {'Neural BPP':<12} {'Neural PSNR':<14} ", end="")
    for codec in ["JPEG", "WEBP", "AVIF"]:
        print(f"| {codec} Savings", end="")
    print()
    print("-" * 80)
    
    for q in quality_levels:
        s = summary_by_quality[q]
        print(f"{q:<10} {s['neural_avg_bpp']:<12.4f} {s['neural_avg_psnr']:<14.2f}", end="")
        for codec in ["JPEG", "WEBP", "AVIF"]:
            if codec in s["codecs"]:
                savings = s["codecs"][codec]["avg_savings_percent"]
                print(f"| {savings:>10.1f}%", end="")
            else:
                print(f"| {'N/A':>10}", end="")
        print()
    
    # Overall average savings
    print(f"\n{'='*80}")
    print(f"OVERALL AVERAGE SAVINGS (across all quality levels)")
    print(f"{'='*80}")
    
    for codec in ["JPEG", "WEBP", "AVIF"]:
        all_savings = []
        for q in quality_levels:
            if codec in summary_by_quality[q]["codecs"]:
                all_savings.append(summary_by_quality[q]["codecs"][codec]["avg_savings_percent"])
        
        if all_savings:
            avg_overall = sum(all_savings) / len(all_savings)
            min_savings = min(all_savings)
            max_savings = max(all_savings)
            print(f"{codec:>6}: {avg_overall:>6.1f}% average ({min_savings:.1f}% - {max_savings:.1f}% range)")
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Verdict
    print(f"\n{'='*80}")
    print("CLAIM VERIFICATION: Neural compression beats JPEG/WebP/AVIF by 40-60% at same quality")
    print(f"{'='*80}")
    
    for codec in ["JPEG", "WEBP", "AVIF"]:
        all_savings = []
        for q in quality_levels:
            if codec in summary_by_quality[q]["codecs"]:
                all_savings.append(summary_by_quality[q]["codecs"][codec]["avg_savings_percent"])
        
        if all_savings:
            avg = sum(all_savings) / len(all_savings)
            if avg >= 40:
                verdict = "✓ VERIFIED"
            elif avg >= 30:
                verdict = "~ CLOSE (30-40%)"
            else:
                verdict = "✗ NOT MET"
            print(f"  vs {codec}: {avg:.1f}% savings -> {verdict}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark neural vs traditional codecs")
    parser.add_argument("--input-dir", type=str, default="C:/Users/User/Downloads/AUB/Fyp/high_quality_images",
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--model", type=str, default="bmshj2018-factorized",
                        choices=["bmshj2018-factorized", "bmshj2018-hyperprior"],
                        help="Neural compression model")
    parser.add_argument("--quality-levels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Quality levels to test")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to test")
    
    args = parser.parse_args()
    
    run_benchmark(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output),
        neural_model=args.model,
        quality_levels=args.quality_levels,
        device=args.device,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
