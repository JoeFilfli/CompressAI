import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pandas as pd
import kagglehub
import pillow_avif  # Critical for AVIF support
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

# Importing from 'train' as per your manual edit
from train import SmallStudent 

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def run_avif_full_benchmark(epoch_num=30, max_images=10, avif_quality=40, avif_speed=0):
    device = 'cpu'
    print(f"Benchmarking on: {device} | AVIF Quality: {avif_quality} | AVIF Speed: {avif_speed}")

    # 1. Fetch from Kaggle
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Setup Model using your scratch path
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_compression_focus.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model = SmallStudent(N=64, M=96).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True)

    supported_ext = ('.jpg', '.jpeg', '.png')
    all_files = [os.path.join(r, f) for r, _, fs in os.walk(dataset_path) for f in fs if f.lower().endswith(supported_ext)]
    image_paths = all_files[:max_images]

    # Create directory for reconstructed outputs
    output_base_dir = "reconstructed_results"
    epoch_dir = os.path.join(output_base_dir, f"epoch_{epoch_num}")
    os.makedirs(epoch_dir, exist_ok=True)

    results = []

    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            
            # --- PHASE 0: Original Reference ---
            orig_img = Image.open(img_path).convert("RGB")
            x_orig = ToTensor()(orig_img).unsqueeze(0).to(device)

            # --- PHASE 1: AVIF Comparison & Timing ---
            temp_avif = "temp_input.avif"
            try:
                # Time the AVIF Encoding
                start_avif_enc = time.perf_counter()
                orig_img.save(temp_avif, "AVIF", quality=avif_quality, speed=avif_speed)
                avif_enc_time_ms = (time.perf_counter() - start_avif_enc) * 1000
                
                avif_size_kb = os.path.getsize(temp_avif) / 1024
                
                # Time the AVIF Decoding (Force load to measure decompression)
                start_avif_dec = time.perf_counter()
                avif_img = Image.open(temp_avif).convert("RGB")
                avif_img.load() 
                avif_dec_time_ms = (time.perf_counter() - start_avif_dec) * 1000

                w, h = avif_img.size
                x_avif = ToTensor()(avif_img).unsqueeze(0).to(device)
                psnr_avif = calculate_psnr(x_orig, x_avif)
            except Exception as e:
                print(f"Skipping {filename}: AVIF error ({e})")
                continue

            # --- PHASE 2: Model Compression & Timing ---
            pad_w = (64 - w % 64) % 64
            pad_h = (64 - h % 64) % 64
            x_padded = F.pad(x_avif, (0, pad_w, 0, pad_h), "constant", 0)

            start_comp = time.perf_counter()
            compressed_data = model.compress(x_padded) 
            comp_time_ms = (time.perf_counter() - start_comp) * 1000

            tmp_bin = "temp_bits.bin"
            with open(tmp_bin, "wb") as f:
                pickle.dump(compressed_data, f)
            model_size_kb = os.path.getsize(tmp_bin) / 1024

            start_decomp = time.perf_counter()
            out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
            decomp_time_ms = (time.perf_counter() - start_decomp) * 1000
            
            x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
            psnr_model = calculate_psnr(x_orig, x_hat)

            # --- PHASE 3: Save Reconstructed Image ---
            recon_pil = ToPILImage()(x_hat.squeeze(0).cpu())
            recon_filename = f"recon_{os.path.splitext(filename)[0]}.png"
            recon_pil.save(os.path.join(epoch_dir, recon_filename))

            results.append({
                "file": filename,
                "avif_kb": avif_size_kb,
                "avif_psnr": psnr_avif,
                "avif_enc_ms": avif_enc_time_ms,
                "avif_dec_ms": avif_dec_time_ms,
                "model_kb": model_size_kb,
                "model_psnr": psnr_model,
                "model_enc_ms": comp_time_ms,
                "model_dec_ms": decomp_time_ms
            })

    # 3. Final Comparison Summary
    df = pd.DataFrame(results)
    
    print("\n" + "="*115)
    print(f"RESULTS FOR EPOCH {epoch_num} (AVIF Speed: {avif_speed}, Quality: {avif_quality})")
    header = f"{'Filename':<15} | {'A-KB':<7} | {'A-PSNR':<7} | {'A-Enc(ms)':<10} | {'M-KB':<7} | {'M-PSNR':<7} | {'M-Enc(ms)':<10}"
    print(header)
    print("-" * 115)
    
    for _, r in df.iterrows():
        print(f"{r['file'][:15]:<15} | {r['avif_kb']:<7.1f} | {r['avif_psnr']:<7.2f} | {r['avif_enc_ms']:<10.1f} | "
              f"{r['model_kb']:<7.1f} | {r['model_psnr']:<7.2f} | {r['model_enc_ms']:<10.1f}")
    
    print("="*115)
    print(f"AVG PSNR: Model {df['model_psnr'].mean():.2f} dB vs AVIF {df['avif_psnr'].mean():.2f} dB")
    print(f"AVG ENCODE: Model {df['model_enc_ms'].mean():.1f} ms vs AVIF {df['avif_enc_ms'].mean():.1f} ms")
    print(f"AVG DECODE: Model {df['model_dec_ms'].mean():.1f} ms vs AVIF {df['avif_dec_ms'].mean():.1f} ms")
    print(f"Images saved to: {epoch_dir}")

    # Cleanup temporary bitstreams
    for f in [temp_avif, tmp_bin]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # Benchmarking across requested epoch range
	"""
    for i in range(20, 31):
        run_avif_full_benchmark(epoch_num=i, max_images=10, avif_quality=60, avif_speed=0)
	"""
	run_avif_full_benchmark(epoch_num=29, max_images=1, avif_quality=40,avif_speed=0)
