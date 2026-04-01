"""
Compare Neural Compression Models: mbt2018-mean vs bmshj2018-factorized

Compares:
- Encoding time
- Decoding time
- Compression ratio (BPP)
- Quality (PSNR, MS-SSIM)

Usage:
    python compare_models.py --input-dir ../kodak
    python compare_models.py --input-dir ../high_quality_images --quality 4
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

# AVIF/HEIF support for input images
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except (ImportError, AttributeError):
    pass


@dataclass
class ModelResult:
    """Results for a single model on a single image"""
    model_name: str
    image_name: str
    quality: int
    width: int
    height: int
    bpp: float
    psnr: float
    ms_ssim: float
    compressed_bytes: int
    encode_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    num_parameters: int


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


def count_parameters(model) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters())


def compress_with_model(
    img: Image.Image,
    model,
    model_name: str,
    quality: int,
    device: str = "cpu"
) -> Dict:
    """Compress image with a model and return metrics"""
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
        encode_time = (time.perf_counter() - t0) * 1000
        
        # Compressed size
        compressed_bytes = sum(len(s[0]) for s in out_enc["strings"])
        
        # Decode
        t0 = time.perf_counter()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        decode_time = (time.perf_counter() - t0) * 1000
        
        x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)
    
    psnr_val = psnr(x, x_hat)
    ms_ssim_val = ms_ssim(x, x_hat)
    bpp = compressed_bytes * 8 / (h * w)
    
    return {
        "model_name": model_name,
        "quality": quality,
        "width": w,
        "height": h,
        "bpp": bpp,
        "psnr": psnr_val,
        "ms_ssim": ms_ssim_val,
        "compressed_bytes": compressed_bytes,
        "encode_time_ms": encode_time,
        "decode_time_ms": decode_time,
        "total_time_ms": encode_time + decode_time,
        "num_parameters": count_parameters(model),
    }


def run_comparison(
    input_dir: Path,
    output_file: Path,
    quality_levels: List[int] = None,
    device: str = "cpu",
    max_images: int = None
):
    """Run comparison between models"""
    
    if quality_levels is None:
        quality_levels = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Models to compare
    model_names = ["bmshj2018-factorized", "mbt2018"]
    
    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".avif", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in extensions])
    
    if max_images:
        images = images[:max_images]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print("=" * 90)
    print("MODEL COMPARISON: bmshj2018-factorized vs mbt2018-mean")
    print("=" * 90)
    print(f"Quality levels: {quality_levels}")
    print(f"Images: {len(images)} from {input_dir}")
    print(f"Device: {device}")
    print("=" * 90)
    
    # Initialize entropy coder
    compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
    
    all_results = []
    summary = {q: {m: {"bpp": [], "psnr": [], "ms_ssim": [], "encode": [], "decode": []} 
                   for m in model_names} for q in quality_levels}
    
    for quality in quality_levels:
        print(f"\n{'='*70}")
        print(f"Quality Level {quality}")
        print(f"{'='*70}")
        
        # Load models for this quality
        loaded_models = {}
        for model_name in model_names:
            print(f"Loading {model_name} (quality={quality})...", end=" ", flush=True)
            model = models[model_name](quality=quality, metric="mse", pretrained=True)
            model = model.to(device).eval()
            model.update()
            loaded_models[model_name] = model
            params = count_parameters(model)
            print(f"{params:,} parameters")
        
        print()
        
        for idx, img_path in enumerate(images, 1):
            print(f"[{idx}/{len(images)}] {img_path.name}")
            
            try:
                img = Image.open(img_path).convert("RGB")
                
                for model_name in model_names:
                    model = loaded_models[model_name]
                    result = compress_with_model(img, model, model_name, quality, device)
                    result["image_name"] = img_path.name
                    all_results.append(result)
                    
                    # Update summary
                    summary[quality][model_name]["bpp"].append(result["bpp"])
                    summary[quality][model_name]["psnr"].append(result["psnr"])
                    summary[quality][model_name]["ms_ssim"].append(result["ms_ssim"])
                    summary[quality][model_name]["encode"].append(result["encode_time_ms"])
                    summary[quality][model_name]["decode"].append(result["decode_time_ms"])
                    
                    short_name = "factorized" if "factorized" in model_name else "mbt2018"
                    print(f"    {short_name:12}: {result['bpp']:.4f} bpp | "
                          f"{result['psnr']:.2f} dB | "
                          f"Enc: {result['encode_time_ms']:>6.0f}ms | "
                          f"Dec: {result['decode_time_ms']:>5.0f}ms")
                
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Clear models to free memory
        del loaded_models
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Print summary comparison
    print(f"\n{'='*90}")
    print("SUMMARY COMPARISON")
    print("=" * 90)
    
    print(f"\n{'Quality':<8} {'Model':<22} {'Avg BPP':<10} {'Avg PSNR':<10} {'Avg MS-SSIM':<12} "
          f"{'Enc (ms)':<10} {'Dec (ms)':<10}")
    print("-" * 90)
    
    for quality in quality_levels:
        for model_name in model_names:
            data = summary[quality][model_name]
            if data["bpp"]:
                avg_bpp = sum(data["bpp"]) / len(data["bpp"])
                avg_psnr = sum(data["psnr"]) / len(data["psnr"])
                avg_ssim = sum(data["ms_ssim"]) / len(data["ms_ssim"])
                avg_enc = sum(data["encode"]) / len(data["encode"])
                avg_dec = sum(data["decode"]) / len(data["decode"])
                
                short_name = "bmshj2018-factorized" if "factorized" in model_name else "mbt2018"
                print(f"{quality:<8} {short_name:<22} {avg_bpp:<10.4f} {avg_psnr:<10.2f} "
                      f"{avg_ssim:<12.4f} {avg_enc:<10.0f} {avg_dec:<10.0f}")
        print()
    
    # Head-to-head comparison
    print(f"\n{'='*90}")
    print("HEAD-TO-HEAD: mbt2018-mean vs bmshj2018-factorized")
    print("=" * 90)
    print(f"\n{'Quality':<8} {'BPP Δ':<12} {'PSNR Δ':<12} {'Encode Δ':<15} {'Decode Δ':<15} {'Winner':<15}")
    print("-" * 90)
    
    overall_comparison = {"bpp": [], "psnr": [], "encode": [], "decode": []}
    
    for quality in quality_levels:
        fact = summary[quality]["bmshj2018-factorized"]
        mbt = summary[quality]["mbt2018"]
        
        if fact["bpp"] and mbt["bpp"]:
            fact_bpp = sum(fact["bpp"]) / len(fact["bpp"])
            mbt_bpp = sum(mbt["bpp"]) / len(mbt["bpp"])
            bpp_diff = ((mbt_bpp - fact_bpp) / fact_bpp) * 100  # negative = mbt smaller
            
            fact_psnr = sum(fact["psnr"]) / len(fact["psnr"])
            mbt_psnr = sum(mbt["psnr"]) / len(mbt["psnr"])
            psnr_diff = mbt_psnr - fact_psnr  # positive = mbt better
            
            fact_enc = sum(fact["encode"]) / len(fact["encode"])
            mbt_enc = sum(mbt["encode"]) / len(mbt["encode"])
            enc_ratio = mbt_enc / fact_enc  # < 1 = mbt faster
            
            fact_dec = sum(fact["decode"]) / len(fact["decode"])
            mbt_dec = sum(mbt["decode"]) / len(mbt["decode"])
            dec_ratio = mbt_dec / fact_dec
            
            # Determine winner (considering quality and size trade-off)
            mbt_wins = 0
            if bpp_diff < -2:  # mbt is >2% smaller
                mbt_wins += 1
            elif bpp_diff > 2:  # factorized is >2% smaller
                mbt_wins -= 1
            if psnr_diff > 0.1:  # mbt has better PSNR
                mbt_wins += 1
            elif psnr_diff < -0.1:
                mbt_wins -= 1
            
            if mbt_wins > 0:
                winner = "mbt2018"
            elif mbt_wins < 0:
                winner = "factorized"
            else:
                winner = "TIE"
            
            bpp_str = f"{bpp_diff:+.1f}%" if bpp_diff != 0 else "same"
            psnr_str = f"{psnr_diff:+.2f} dB"
            enc_str = f"{enc_ratio:.2f}x"
            dec_str = f"{dec_ratio:.2f}x"
            
            print(f"{quality:<8} {bpp_str:<12} {psnr_str:<12} {enc_str:<15} {dec_str:<15} {winner:<15}")
            
            overall_comparison["bpp"].append(bpp_diff)
            overall_comparison["psnr"].append(psnr_diff)
            overall_comparison["encode"].append(enc_ratio)
            overall_comparison["decode"].append(dec_ratio)
    
    # Overall verdict
    print(f"\n{'='*90}")
    print("OVERALL VERDICT")
    print("=" * 90)
    
    if overall_comparison["bpp"]:
        avg_bpp_diff = sum(overall_comparison["bpp"]) / len(overall_comparison["bpp"])
        avg_psnr_diff = sum(overall_comparison["psnr"]) / len(overall_comparison["psnr"])
        avg_enc_ratio = sum(overall_comparison["encode"]) / len(overall_comparison["encode"])
        avg_dec_ratio = sum(overall_comparison["decode"]) / len(overall_comparison["decode"])
        
        print(f"\nmbt2018 vs bmshj2018-factorized:")
        print(f"  BPP:      {avg_bpp_diff:+.1f}% {'(mbt smaller)' if avg_bpp_diff < 0 else '(factorized smaller)'}")
        print(f"  PSNR:     {avg_psnr_diff:+.2f} dB {'(mbt better)' if avg_psnr_diff > 0 else '(factorized better)'}")
        print(f"  Encoding: {avg_enc_ratio:.2f}x {'(mbt faster)' if avg_enc_ratio < 1 else '(factorized faster)'}")
        print(f"  Decoding: {avg_dec_ratio:.2f}x {'(mbt faster)' if avg_dec_ratio < 1 else '(factorized faster)'}")
        
        print(f"\nRecommendation:")
        if avg_bpp_diff < -5 and avg_psnr_diff > 0:
            print("  → mbt2018-mean: Better compression AND quality")
        elif avg_bpp_diff < -5:
            print("  → mbt2018-mean: Better compression (smaller files)")
        elif avg_psnr_diff > 0.5:
            print("  → mbt2018-mean: Better quality (higher PSNR)")
        elif avg_enc_ratio > 2:
            print("  → bmshj2018-factorized: Much faster encoding")
        else:
            print("  → Both models perform similarly; choose based on speed vs quality needs")
    
    # Save results
    output_data = {
        "config": {
            "models": model_names,
            "quality_levels": quality_levels,
            "num_images": len(images),
            "device": device,
        },
        "per_image_results": all_results,
        "summary": {str(q): {m: {
            "avg_bpp": sum(summary[q][m]["bpp"]) / len(summary[q][m]["bpp"]) if summary[q][m]["bpp"] else 0,
            "avg_psnr": sum(summary[q][m]["psnr"]) / len(summary[q][m]["psnr"]) if summary[q][m]["psnr"] else 0,
            "avg_ms_ssim": sum(summary[q][m]["ms_ssim"]) / len(summary[q][m]["ms_ssim"]) if summary[q][m]["ms_ssim"] else 0,
            "avg_encode_ms": sum(summary[q][m]["encode"]) / len(summary[q][m]["encode"]) if summary[q][m]["encode"] else 0,
            "avg_decode_ms": sum(summary[q][m]["decode"]) / len(summary[q][m]["decode"]) if summary[q][m]["decode"] else 0,
        } for m in model_names} for q in quality_levels}
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Compare neural compression models")
    parser.add_argument("--input-dir", type=str, default=str(EXAMPLES_DIR / "kodak"),
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, default="model_comparison.json",
                        help="Output JSON file")
    parser.add_argument("--quality-levels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Quality levels to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum images to test")
    
    args = parser.parse_args()
    
    run_comparison(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output),
        quality_levels=args.quality_levels,
        device=args.device,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
