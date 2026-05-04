import io
import json
import struct
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from utils import (
    compute_msssim,
    compute_psnr,
    crop_to_original,
    load_image,
    load_model,
    pad_to_multiple,
    scan_images,
)


def compress_folder(
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    quality: int,
    device: str,
    recursive: bool,
    metrics: bool,
    delete_source: bool,
    checkpoint_path: Optional[Path] = None,
) -> tuple[int, Iterator[dict]]:
    images = scan_images(input_folder, recursive)
    total = len(images)

    def _generate() -> Iterator[dict]:
        model = load_model(model_name, quality, device, checkpoint_path=checkpoint_path)

        for image_path in images:
            rel_path = image_path.relative_to(input_folder)
            out_path = output_folder / rel_path.with_suffix(".bin")

            result: dict = {
                "file": str(rel_path),
                "status": "ok",
                "error": None,
                "orig_kb": image_path.stat().st_size / 1024,
                "comp_kb": None,
                "jpeg_kb": None,
                "ratio": None,
                "bpp": None,
                "psnr": None,
                "msssim": None,
                "orig_image": None,
                "recon_image": None,
                "comp_time": None,
                "decomp_time": None,
            }

            try:
                img = load_image(image_path)

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                result["jpeg_kb"] = buf.tell() / 1024

                x = to_tensor(img).unsqueeze(0).to(device)
                x_padded, orig_h, orig_w = pad_to_multiple(x)

                t_comp = time.time()
                with torch.no_grad():
                    compressed = model.compress(x_padded)
                result["comp_time"] = time.time() - t_comp

                strings = compressed["strings"]
                shape = compressed["shape"]

                string_bytes = [s[0] for s in strings]
                header = {
                    "model": model_name,
                    "quality": quality,
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                    "shape": list(shape),
                    "string_lengths": [len(b) for b in string_bytes],
                }
                if checkpoint_path:
                    header["checkpoint"] = str(checkpoint_path)
                    header["checkpoint_name"] = checkpoint_path.name
                header_bytes = json.dumps(header).encode("utf-8")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(struct.pack(">I", len(header_bytes)))
                    f.write(header_bytes)
                    for b in string_bytes:
                        f.write(b)

                comp_kb = out_path.stat().st_size / 1024
                num_pixels = orig_h * orig_w
                result["comp_kb"] = comp_kb
                result["ratio"] = result["orig_kb"] / comp_kb if comp_kb > 0 else 0.0
                result["bpp"] = (out_path.stat().st_size * 8) / num_pixels

                if metrics:
                    t_decomp = time.time()
                    with torch.no_grad():
                        decompressed = model.decompress(strings, shape)
                    result["decomp_time"] = time.time() - t_decomp
                    x_hat = crop_to_original(decompressed["x_hat"], orig_h, orig_w)
                    x_orig = crop_to_original(x_padded, orig_h, orig_w)
                    result["psnr"] = compute_psnr(x_orig, x_hat)
                    result["msssim"] = compute_msssim(x_orig, x_hat)
                    result["orig_image"] = img
                    result["recon_image"] = to_pil_image(x_hat.squeeze(0).clamp(0, 1))

                if delete_source:
                    image_path.unlink()

            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)

            yield result

    return total, _generate()
