import time
from pathlib import Path
from compressai.zoo import models
from examples.codec import decode_image, CodecInfo, parse_header, read_uints, read_uchars
import compressai
import torch

# -------- CONFIG --------
INPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\images2_compressed")
OUTPUT_DIR = Path("C:\\Users\\User\\Downloads\\AUB\\Fyp\\images2_decoded")
CODER = compressai.available_entropy_coders()[0]
DEVICE = "cpu"  # or "cuda"
# ------------------------

compressai.set_entropy_coder(CODER)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bitstreams = sorted(p for p in INPUT_DIR.iterdir() if p.suffix == ".bin")

if not bitstreams:
    raise RuntimeError("No .bin files found")

# ---- Read header from first file to know which model to load ----
with bitstreams[0].open("rb") as f:
    model_name, metric, quality = parse_header(f.read(2))

# ---- Load model ONCE ----
t0 = time.perf_counter()
net = (
    models[model_name](quality=quality, metric=metric, pretrained=True)
    .to(DEVICE)
    .eval()
)
model_load_time = time.perf_counter() - t0

print(f"Loaded model: {model_name}, metric={metric}, quality={quality}")

# ---- Decode whole folder ----
t0 = time.perf_counter()
with torch.inference_mode():
    for bitstream in bitstreams:
        with bitstream.open("rb") as f:
            # read header again
            parse_header(f.read(2))
            original_size = read_uints(f, 2)
            original_bitdepth = read_uchars(f, 1)[0]

            codec_info = CodecInfo(
                codec_header=None,
                original_size=original_size,
                original_bitdepth=original_bitdepth,
                net=net,
                device=DEVICE,
            )

            output_img = OUTPUT_DIR / bitstream.with_suffix(".png").name
            decode_image(f, codec_info, str(output_img))

total_time = time.perf_counter() - t0

print(f"Images decoded: {len(bitstreams)}")
print(f"Model load time: {model_load_time:.2f}s")
print(f"Total decode time: {total_time:.2f}s")
print(f"Avg time per image: {total_time/len(bitstreams):.3f}s")
