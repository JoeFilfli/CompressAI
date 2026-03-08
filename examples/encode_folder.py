import time
from pathlib import Path
from compressai.zoo import models
from examples.codec import encode_image, CodecInfo, get_header, CodecType
import compressai
import torch
from PIL import Image

# -------- CONFIG --------
INPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\kodak")
OUTPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\images2_compressed")
CROP_SIZE = 512  # Crop images to 512x512
MODEL = "bmshj2018-factorized"
QUALITY = 1
METRIC = "mse"
CODER = compressai.available_entropy_coders()[0]
DEVICE = "cpu"
# ------------------------

compressai.set_entropy_coder(CODER)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create temp directory for cropped images
TEMP_DIR = OUTPUT_DIR / "temp_cropped"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def center_crop_image(img_path, output_path, crop_size=512):
    """Center crop image to crop_size x crop_size"""
    img = Image.open(img_path)
    width, height = img.size
    
    # Calculate crop coordinates (center crop)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # Crop image
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(output_path)
    return output_path

images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"])

if not images:
    print(f"No images found in {INPUT_DIR}")
    exit(1)

print(f"\n{'='*70}")
print(f"PRE-PROCESSING: Cropping images to {CROP_SIZE}x{CROP_SIZE}")
print(f"{'='*70}")

# Crop all images BEFORE measuring encoding time
cropped_images = []
for i, img in enumerate(images, 1):
    out_crop_path = TEMP_DIR / img.name
    center_crop_image(img, out_crop_path, CROP_SIZE)
    cropped_images.append(out_crop_path)
    if i % 10 == 0 or i == len(images):
        print(f"  Cropped: {i}/{len(images)} images...")

print(f"✓ Cropping complete!\n")

# Load model ONCE
t0 = time.perf_counter()
net = models[MODEL](quality=QUALITY, metric=METRIC, pretrained=True).to(DEVICE).eval()
model_load_time = time.perf_counter() - t0

codec_header = get_header(MODEL, METRIC, QUALITY, -1, CodecType.IMAGE_CODEC)
codec_info = CodecInfo(codec_header, None, None, net, DEVICE)

print(f"\n{'='*70}")
print(f"ENCODING (Sequential Processing with {CROP_SIZE}x{CROP_SIZE} cropped images)")
print(f"{'='*70}")
print(f"Model: {MODEL} | Quality: {QUALITY} | Device: {DEVICE}")
print(f"Images: {len(cropped_images)}")
print(f"{'='*70}\n")

# Encode whole folder
t0 = time.perf_counter()
with torch.inference_mode():
    for i, img in enumerate(cropped_images, 1):
        out_file = OUTPUT_DIR / img.with_suffix(".bin").name
        encode_image(str(img), codec_info, str(out_file))
        if i % 10 == 0 or i == len(cropped_images):
            print(f"  Progress: {i}/{len(cropped_images)} images...")
total_time = time.perf_counter() - t0

print(f"\n{'='*70}")
print(f"📊 RESULTS - {CROP_SIZE}x{CROP_SIZE} CROPPED IMAGES")
print(f"{'='*70}")
print(f"Method:              Sequential with center crop")
print(f"Crop size:           {CROP_SIZE}x{CROP_SIZE}")
print(f"Images processed:    {len(cropped_images)}")
print(f"Model load time:     {model_load_time:.2f}s")
print(f"Total encode time:   {total_time:.2f}s")
print(f"Avg time per image:  {total_time/len(cropped_images):.3f}s")
print(f"Throughput:          {len(cropped_images)/total_time:.2f} images/sec")
print(f"{'='*70}")
print(f"\n💡 Images were center-cropped to {CROP_SIZE}x{CROP_SIZE} before encoding")
print(f"💡 Cropped images saved to: {TEMP_DIR}\n")
