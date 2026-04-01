"""
Phase 2B: Multiprocessing Optimization
Expected speedup: 4-8x (on 8-core CPU)
Quality impact: ZERO (identical to baseline)
"""
import time
from pathlib import Path
from compressai.zoo import models
from examples.codec import encode_image, CodecInfo, get_header, CodecType
import compressai
import torch
from multiprocessing import Pool, cpu_count
import os

EXAMPLES_DIR = Path(__file__).resolve().parent

# -------- CONFIG --------
INPUT_DIR = EXAMPLES_DIR / "kodak"
OUTPUT_DIR = EXAMPLES_DIR / "images2_parallel_compressed"
MODEL = "bmshj2018-factorized"
QUALITY = 3
METRIC = "mse"
CODER = compressai.available_entropy_coders()[0]
DEVICE = "cpu"
NUM_WORKERS = max(1, cpu_count() - 2)  # Use 14 of 16 cores
# ------------------------

# Global worker state (initialized once per worker)
_worker_net = None
_worker_codec_info = None

def init_worker(model_name, quality, metric, coder, device):
    """Initialize model once per worker process"""
    global _worker_net, _worker_codec_info
    compressai.set_entropy_coder(coder)
    _worker_net = models[model_name](quality=quality, metric=metric, pretrained=True).to(device).eval()
    codec_header = get_header(model_name, metric, quality, -1, CodecType.IMAGE_CODEC)
    _worker_codec_info = CodecInfo(codec_header, None, None, _worker_net, device)

def encode_single_image(args):
    """Worker function to encode a single image in separate process"""
    img_path, output_path = args
    
    # Reuse the model loaded in init_worker
    global _worker_codec_info
    
    # Encode
    out_file = output_path / img_path.with_suffix(".bin").name
    
    with torch.inference_mode():
        result = encode_image(str(img_path), _worker_codec_info, str(out_file))
    
    return img_path.name, result

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"])
    
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return
    
    print(f"{'='*60}")
    print(f"MULTIPROCESSING OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Quality: {QUALITY}")
    print(f"Images: {len(images)}")
    print(f"CPU cores: {cpu_count()}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"{'='*60}\n")
    
    # Prepare arguments for each worker
    tasks = [
        (img, OUTPUT_DIR)
        for img in images
    ]
    
    print(f"Starting parallel encoding with {NUM_WORKERS} workers...\n")
    
    t0_total = time.perf_counter()
    t0_encode = time.perf_counter()  # Model already loaded in workers
    
    # Use Pool for parallel processing with initializer
    with Pool(processes=NUM_WORKERS, 
              initializer=init_worker, 
              initargs=(MODEL, QUALITY, METRIC, CODER, DEVICE)) as pool:
        results = []
        completed = 0
        
        # Use imap_unordered for better progress tracking
        for result in pool.imap_unordered(encode_single_image, tasks):
            completed += 1
            results.append(result)
            if completed % 10 == 0 or completed == len(images):
                print(f"  Progress: {completed}/{len(images)} images...")
    
    encode_time = time.perf_counter() - t0_encode
    total_time = time.perf_counter() - t0_total
    throughput = len(images) / encode_time
    
    # Calculate actual speedup (baseline is ~0.69 images/sec for 144 images at Q3)
    baseline_throughput = 0.69  # Update this based on your baseline run
    actual_speedup = throughput / baseline_throughput
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTS - PARALLEL")
    print(f"{'='*70}")
    print(f"Method:              Parallel (Multiprocessing)")
    print(f"Images processed:    {len(images)}")
    print(f"Workers:             {NUM_WORKERS}")
    print(f"Model load time:     (per-worker, included in total)")
    print(f"Total encode time:   {encode_time:.2f}s")
    print(f"Avg time per image:  {encode_time/len(images):.3f}s")
    print(f"Throughput:          {throughput:.2f} images/sec")
    print(f"Speedup factor:      {actual_speedup:.2f}x (actual vs baseline)")
    print(f"Efficiency:          {(actual_speedup/NUM_WORKERS)*100:.1f}% (of theoretical {NUM_WORKERS}x)")
    print(f"{'='*70}")
    print(f"\n💡 Actual speedup: {throughput:.2f} ÷ {baseline_throughput:.2f} = {actual_speedup:.2f}x")
    print(f"⚠️  Not {NUM_WORKERS}x due to: process overhead, model loading, sequential bottlenecks\n")

if __name__ == "__main__":
    # Set start method to spawn for Windows compatibility
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
