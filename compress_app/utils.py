from pathlib import Path

import torch
import torch.nn.functional as F
from compressai.zoo import image_models
from PIL import Image
from torchvision.transforms.functional import to_pil_image

MODELS = [
    "bmshj2018-factorized",
    "bmshj2018-hyperprior",
    "mbt2018",
    "mbt2018-mean",
    "cheng2020-anchor",
    "cheng2020-attn",
]

_MODEL_CACHE: dict = {}


def load_model(name: str, quality: int, device: str) -> torch.nn.Module:
    key = (name, quality, device)
    if key not in _MODEL_CACHE:
        model = image_models[name](quality=quality, pretrained=True)
        model.eval().to(device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def scan_images(folder: Path, recursive: bool) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    glob = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in glob if p.is_file() and p.suffix.lower() in exts)


def scan_bins(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.bin" if recursive else "*.bin"
    return sorted(folder.glob(pattern))


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_jpeg(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_pil_image(tensor.squeeze(0).clamp(0, 1))
    img.save(path, format="JPEG", quality=95)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> tuple:
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, h, w


def crop_to_original(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return tensor[:, :, :h, :w]


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10 * torch.log10(torch.tensor(mse)).item()


def compute_msssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    from pytorch_msssim import ms_ssim
    return ms_ssim(original, reconstructed, data_range=1.0).item()
