import time
from pathlib import Path
from compressai.zoo import models
from examples.codec import encode_image, CodecInfo, get_header, CodecType
import compressai
import torch

# -------- CONFIG --------
INPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\images2")
OUTPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\images2_compressed")
MODEL = "bmshj2018-factorized"
QUALITY = 1
METRIC = "mse"
CODER = compressai.available_entropy_coders()[0]
DEVICE = "cpu"
# ------------------------

compressai.set_entropy_coder(CODER)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model ONCE
t0 = time.perf_counter()
net = models[MODEL](quality=QUALITY, metric=METRIC, pretrained=True).to(DEVICE).eval()
model_load_time = time.perf_counter() - t0

codec_header = get_header(MODEL, METRIC, QUALITY, -1, CodecType.IMAGE_CODEC)
codec_info = CodecInfo(codec_header, None, None, net, DEVICE)

images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"])

# Encode whole folder
t0 = time.perf_counter()
with torch.inference_mode():
    for img in images:
        out_file = OUTPUT_DIR / img.with_suffix(".bin").name
        encode_image(str(img), codec_info, str(out_file))
total_time = time.perf_counter() - t0

print(f"Images: {len(images)}")
print(f"Model load time: {model_load_time:.2f}s")
print(f"Total encode time: {total_time:.2f}s")
print(f"Avg time per image: {total_time/len(images):.3f}s")
