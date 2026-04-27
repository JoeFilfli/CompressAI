import json
import struct
from pathlib import Path
from typing import Iterator

import torch

from utils import crop_to_original, load_model, save_jpeg, scan_bins


def decompress_folder(
    input_folder: Path,
    output_folder: Path,
    recursive: bool,
) -> tuple[int, Iterator[dict]]:
    bin_files = scan_bins(input_folder, recursive)
    total = len(bin_files)

    def _generate() -> Iterator[dict]:
        for bin_path in bin_files:
            rel_path = bin_path.relative_to(input_folder)
            out_path = output_folder / rel_path.with_suffix(".jpg")

            result: dict = {
                "file": str(rel_path),
                "status": "ok",
                "error": None,
                "comp_kb": bin_path.stat().st_size / 1024,
                "out_kb": None,
                "model": None,
                "quality": None,
            }

            try:
                with open(bin_path, "rb") as f:
                    header_len = struct.unpack(">I", f.read(4))[0]
                    header = json.loads(f.read(header_len).decode("utf-8"))
                    string_bytes = [f.read(length) for length in header["string_lengths"]]

                model_name = header["model"]
                quality = header["quality"]
                orig_h = header["orig_h"]
                orig_w = header["orig_w"]
                shape = header["shape"]
                strings = [[b] for b in string_bytes]

                result["model"] = model_name
                result["quality"] = quality

                model = load_model(model_name, quality, "cpu")

                with torch.no_grad():
                    decompressed = model.decompress(strings, shape)

                x_hat = crop_to_original(decompressed["x_hat"], orig_h, orig_w)
                save_jpeg(x_hat, out_path)

                result["out_kb"] = out_path.stat().st_size / 1024

            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)

            yield result

    return total, _generate()
