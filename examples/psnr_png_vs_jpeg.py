import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ms_ssim

import compressai
from compressai.zoo import models
from compressai.ops import compute_padding


# ================= CONFIG =================
INPUT_DIR = Path("C:/Users/User/Downloads/AUB/Fyp/images2")
TMP_DIR = Path("C:/Users/User/Downloads/AUB/Fyp/tmp_outputs")

MODEL = "bmshj2018-factorized"
QUALITY = 1
METRIC = "mse"
DEVICE = "cpu"   # or "cuda"
JPEG_QUALITY = 95
# ==========================================

TMP_DIR.mkdir(parents=True, exist_ok=True)
compressai.set_entropy_coder(compressai.available_entropy_coders()[0])

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# ---------- PSNR ----------
def psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    return 20 * math.log10(1.0) - 10 * torch.log10(mse)


# ---------- LOAD MODEL ONCE ----------
t0 = time.perf_counter()
model = models[MODEL](quality=QUALITY, metric=METRIC, pretrained=True)
model = model.to(DEVICE).eval()
print(f"Model loaded in {time.perf_counter() - t0:.2f}s")


# ---------- IMAGE LIST ----------
images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"])
assert images, "No images found"


psnr_mem_list = []
psnr_png_list = []
psnr_jpg_list = []


with torch.inference_mode():
    for img_path in images:
        # ----- Read original -----
        img = Image.open(img_path).convert("RGB")
        x = to_tensor(img).to(DEVICE).unsqueeze(0)

        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=64)
        x_padded = F.pad(x, pad)

        # ----- Compress & Decompress (CompressAI) -----
        out_enc = model.compress(x_padded)
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1)

        x = x.squeeze(0)
        x_hat = x_hat.squeeze(0)

        # ----- PSNR: CompressAI (in memory) -----
        psnr_mem = psnr(x, x_hat).item()
        psnr_mem_list.append(psnr_mem)

        # ----- Save as PNG (lossless) -----
        png_path = TMP_DIR / f"{img_path.stem}.png"
        to_pil(x_hat).save(png_path, format="PNG")

        png_img = to_tensor(Image.open(png_path).convert("RGB")).to(DEVICE)
        psnr_png = psnr(x, png_img).item()
        psnr_png_list.append(psnr_png)

        # ----- Save as JPEG (lossy) -----
        jpg_path = TMP_DIR / f"{img_path.stem}.jpg"
        to_pil(x_hat).save(
            jpg_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0,
            optimize=True,
        )

        jpg_img = to_tensor(Image.open(jpg_path).convert("RGB")).to(DEVICE)
        psnr_jpg = psnr(x, jpg_img).item()
        psnr_jpg_list.append(psnr_jpg)


# ---------- RESULTS ----------
avg_mem = sum(psnr_mem_list) / len(psnr_mem_list)
avg_png = sum(psnr_png_list) / len(psnr_png_list)
avg_jpg = sum(psnr_jpg_list) / len(psnr_jpg_list)

print("\n===== PSNR RESULTS (Average over folder) =====")
print(f"CompressAI (in memory): {avg_mem:.2f} dB")
print(f"PNG (lossless save):    {avg_png:.2f} dB")
print(f"JPEG (Q={JPEG_QUALITY}): {avg_jpg:.2f} dB")
print(f"PSNR drop due to JPEG:  {avg_mem - avg_jpg:.2f} dB")
