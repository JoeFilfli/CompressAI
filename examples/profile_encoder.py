"""
Profile ONNX Runtime vs PyTorch vs Quantized for the g_a encoder.
Also tests thread tuning.
The g_a CNN is 82% of compression time - this finds the fastest way to run it.
"""
import time
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from compressai.zoo import models
from compressai.ops import compute_padding
import compressai
import warnings
warnings.filterwarnings("ignore")

# -------- CONFIG --------
INPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\images22")
MODEL = "bmshj2018-factorized"
QUALITY = 1
METRIC = "mse"
DEVICE = "cpu"
ONNX_DIR = Path("model_onnx")
NUM_IMAGES = 5  # Profile on first N images
WARMUP = 2
# ------------------------

compressai.set_entropy_coder(compressai.available_entropy_coders()[0])

# Load image and prepare input tensor
images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"])[:NUM_IMAGES]

def prepare_input(img_path):
    img = Image.open(img_path).convert("RGB")
    x = ToTensor()(img).unsqueeze(0)
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=64)
    x = F.pad(x, pad, mode="constant", value=0)
    return x

inputs = [prepare_input(p) for p in images]
print(f"Profiling {len(inputs)} images, {inputs[0].shape}")
print(f"Model: {MODEL}, Quality: {QUALITY}")
print(f"=" * 70)

results = {}

# =====================================================
# 1. PyTorch baseline (current default threads)
# =====================================================
print(f"\n--- 1. PyTorch Baseline (threads={torch.get_num_threads()}) ---")
net = models[MODEL](quality=QUALITY, metric=METRIC, pretrained=True).eval()

with torch.inference_mode():
    # Warmup
    for _ in range(WARMUP):
        _ = net.g_a(inputs[0])
    
    t0 = time.perf_counter()
    for x in inputs:
        _ = net.g_a(x)
    elapsed = time.perf_counter() - t0
    per_img = elapsed / len(inputs)
    results["PyTorch baseline"] = per_img
    print(f"  Total: {elapsed:.3f}s | Per image: {per_img*1000:.1f}ms")

# =====================================================
# 2. PyTorch with different thread counts
# =====================================================
print(f"\n--- 2. Thread Tuning ---")
cpu_count = os.cpu_count()
thread_counts = sorted(set([1, 2, 4, cpu_count // 2, cpu_count, cpu_count * 2]))
thread_counts = [t for t in thread_counts if t > 0 and t <= cpu_count * 2]

best_threads = torch.get_num_threads()
best_time = per_img

with torch.inference_mode():
    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)
        # Warmup
        for _ in range(WARMUP):
            _ = net.g_a(inputs[0])
        
        t0 = time.perf_counter()
        for x in inputs:
            _ = net.g_a(x)
        elapsed = time.perf_counter() - t0
        t_per = elapsed / len(inputs)
        marker = " ◀ BEST" if t_per < best_time else ""
        print(f"  threads={n_threads:<3}: {t_per*1000:.1f}ms/img{marker}")
        if t_per < best_time:
            best_time = t_per
            best_threads = n_threads

results["PyTorch best threads"] = best_time
print(f"  → Best: {best_threads} threads ({best_time*1000:.1f}ms/img)")

# Reset to best
torch.set_num_threads(best_threads)

# =====================================================
# 3. PyTorch Dynamic Quantization (INT8)
# =====================================================
print(f"\n--- 3. PyTorch Dynamic Quantization (INT8) ---")
try:
    net_q = torch.ao.quantization.quantize_dynamic(
        net, {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear}, dtype=torch.qint8
    )
    
    with torch.inference_mode():
        # Warmup
        for _ in range(WARMUP):
            _ = net_q.g_a(inputs[0])
        
        t0 = time.perf_counter()
        for x in inputs:
            _ = net_q.g_a(x)
        elapsed = time.perf_counter() - t0
        per_img = elapsed / len(inputs)
        results["PyTorch INT8 quantized"] = per_img
        print(f"  Total: {elapsed:.3f}s | Per image: {per_img*1000:.1f}ms")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# =====================================================
# 4. PyTorch Static Quantization
# =====================================================
print(f"\n--- 4. PyTorch Static Quantization ---")
try:
    from torch.ao.quantization import get_default_qconfig, prepare, convert
    
    net_sq = models[MODEL](quality=QUALITY, metric=METRIC, pretrained=True).eval()
    net_sq.qconfig = get_default_qconfig('x86')
    net_sq_prepared = prepare(net_sq, inplace=False)
    
    # Calibrate with a few images
    with torch.inference_mode():
        for x in inputs[:3]:
            _ = net_sq_prepared.g_a(x)
    
    net_sq_converted = convert(net_sq_prepared, inplace=False)
    
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = net_sq_converted.g_a(inputs[0])
        
        t0 = time.perf_counter()
        for x in inputs:
            _ = net_sq_converted.g_a(x)
        elapsed = time.perf_counter() - t0
        per_img = elapsed / len(inputs)
        results["PyTorch static quantized"] = per_img
        print(f"  Total: {elapsed:.3f}s | Per image: {per_img*1000:.1f}ms")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# =====================================================
# 5. ONNX Runtime
# =====================================================
print(f"\n--- 5. ONNX Runtime ---")
try:
    import onnxruntime as ort
    
    onnx_encoder_path = ONNX_DIR / f"{MODEL}_q{QUALITY}_encoder.onnx"
    onnx_full_path = ONNX_DIR / f"{MODEL}_q{QUALITY}.onnx"
    
    onnx_path = None
    if onnx_encoder_path.exists():
        onnx_path = onnx_encoder_path
        print(f"  Using encoder-only ONNX: {onnx_path}")
    elif onnx_full_path.exists():
        onnx_path = onnx_full_path
        print(f"  Using full model ONNX: {onnx_path}")
    
    if onnx_path is None:
        # Export encoder to ONNX on the fly
        print(f"  No ONNX model found, exporting encoder...")
        onnx_path = Path("model_onnx") / f"{MODEL}_q{QUALITY}_encoder.onnx"
        onnx_path.parent.mkdir(exist_ok=True)
        
        dummy = torch.randn(1, 3, 1024, 1024)
        torch.onnx.export(
            net.g_a, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "h", 3: "w"}},
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"  Exported to {onnx_path}")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0  # auto
    sess_options.inter_op_num_threads = 0
    
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(WARMUP):
        _ = session.run(None, {input_name: inputs[0].numpy()})
    
    t0 = time.perf_counter()
    for x in inputs:
        _ = session.run(None, {input_name: x.numpy()})
    elapsed = time.perf_counter() - t0
    per_img = elapsed / len(inputs)
    results["ONNX Runtime"] = per_img
    print(f"  Total: {elapsed:.3f}s | Per image: {per_img*1000:.1f}ms")
    
    # Test with different thread counts
    print(f"\n  ONNX Thread Tuning:")
    best_onnx_time = per_img
    best_onnx_threads = 0
    for n_threads in [1, 2, 4, cpu_count // 2, cpu_count]:
        if n_threads <= 0:
            continue
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = n_threads
        so.inter_op_num_threads = 1
        
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        
        for _ in range(WARMUP):
            _ = sess.run(None, {input_name: inputs[0].numpy()})
        
        t0 = time.perf_counter()
        for x in inputs:
            _ = sess.run(None, {input_name: x.numpy()})
        elapsed = time.perf_counter() - t0
        t_per = elapsed / len(inputs)
        marker = " ◀ BEST" if t_per < best_onnx_time else ""
        print(f"    threads={n_threads:<3}: {t_per*1000:.1f}ms/img{marker}")
        if t_per < best_onnx_time:
            best_onnx_time = t_per
            best_onnx_threads = n_threads
    
    results["ONNX best threads"] = best_onnx_time
    print(f"  → Best: {best_onnx_threads} threads ({best_onnx_time*1000:.1f}ms/img)")

except ImportError:
    print("  ❌ onnxruntime not installed")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback; traceback.print_exc()

# =====================================================
# SUMMARY
# =====================================================
print(f"\n{'='*70}")
print(f"SUMMARY - g_a encoder speed (per image)")
print(f"{'='*70}")

baseline = results.get("PyTorch baseline", 1.0)
for name, t in sorted(results.items(), key=lambda x: x[1]):
    speedup = baseline / t
    bar = "█" * int(speedup * 10)
    print(f"  {name:<30} {t*1000:>7.1f}ms  {speedup:>5.2f}x  {bar}")

print(f"\n{'='*70}")
print(f"PROJECTED FULL-PIPELINE TIMES (16 images)")
print(f"{'='*70}")

# Original full pipeline: 12.94s for 16 images = 809ms/img
# g_a is 82.4% of that = 635ms, rest = 142ms overhead per image
overhead_per_img = 0.809 - 0.635  # seconds
for name, t in sorted(results.items(), key=lambda x: x[1]):
    projected = (t + overhead_per_img) * 16
    speedup = 12.94 / projected
    print(f"  {name:<30} {projected:>6.1f}s  ({speedup:.2f}x vs baseline 12.94s)")

winner = min(results, key=results.get)
winner_time = results[winner]
projected_winner = (winner_time + overhead_per_img) * 16
print(f"\n🏆 WINNER: {winner}")
print(f"   Per image: {winner_time*1000:.1f}ms (g_a only)")
print(f"   Projected 16 images: {projected_winner:.1f}s (vs 12.94s baseline)")
print(f"   Speedup: {12.94/projected_winner:.2f}x")
print(f"\n   + Multiprocessing ({cpu_count} cores): ~{projected_winner/cpu_count:.1f}s")
