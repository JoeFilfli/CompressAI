from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from compressai.zoo import image_models
from compressai.zoo.image import cfgs as _ZOO_CFGS
from PIL import Image
from torchvision.transforms.functional import to_pil_image

BUILTIN_MODELS = [
    "bmshj2018-factorized",
    "bmshj2018-hyperprior",
    "mbt2018",
    "mbt2018-mean",
    "cheng2020-anchor",
    "cheng2020-attn",
]

CUSTOM_BASE_MODELS = BUILTIN_MODELS + ["tiny-hyperprior"]

MODELS = BUILTIN_MODELS

_MODEL_CACHE: dict = {}

_DISTILL_SEP = "_distilled_from_"

_SCALE_HYPERPRIOR_CFGS = {}
for _arch, _quality_map in _ZOO_CFGS.items():
    for _cfg in _quality_map.values():
        if isinstance(_cfg, tuple) and len(_cfg) == 2:
            _SCALE_HYPERPRIOR_CFGS.setdefault(_arch, set()).add(_cfg)

_ORDERED_SCALE_ARCHS = [
    arch for arch in CUSTOM_BASE_MODELS if arch in _SCALE_HYPERPRIOR_CFGS
]


def parse_distilled_checkpoint_name(path: Path) -> Optional[Tuple[str, int, str]]:
    name = path.name
    if not name.endswith(".pth") or _DISTILL_SEP not in name:
        return None
    left, teacher_part = name.rsplit(_DISTILL_SEP, 1)
    teacher = teacher_part[:-4]
    if "_q" not in left:
        return None
    student, q_part = left.rsplit("_q", 1)
    try:
        quality = int(q_part)
    except ValueError:
        return None
    return student, quality, teacher


def list_custom_checkpoints(base_dir: Path) -> list[dict]:
    checkpoints = []
    for path in sorted(base_dir.glob("*.pth")):
        info = {
            "path": path,
            "label": path.name,
            "student": "tiny-hyperprior",
            "quality": None,
            "teacher": None,
        }
        parsed = parse_distilled_checkpoint_name(path)
        if parsed:
            student, quality, teacher = parsed
            info["student"] = student
            info["quality"] = quality
            info["teacher"] = teacher
            info["label"] = f"{student} q{quality} (distilled from {teacher})"
        checkpoints.append(info)
    return checkpoints


def _extract_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "resume_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unsupported checkpoint format")


def _strip_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    student_keys = {k for k in state_dict if k.startswith("student.")}
    if student_keys:
        return {k[len("student."):]: v for k, v in state_dict.items() if k in student_keys}
    return state_dict


def _infer_nm_from_state(state_dict: dict) -> Optional[Tuple[int, int]]:
    if "g_a.0.weight" in state_dict and "g_a.6.weight" in state_dict:
        n = int(state_dict["g_a.0.weight"].shape[0])
        m = int(state_dict["g_a.6.weight"].shape[0])
        return n, m
    return None


def infer_checkpoint_base(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)
    nm = _infer_nm_from_state(state)

    n = m = None
    matches = []
    if nm:
        n, m = nm
        for arch in _ORDERED_SCALE_ARCHS:
            if (n, m) in _SCALE_HYPERPRIOR_CFGS[arch]:
                matches.append(arch)

    base = matches[0] if len(matches) == 1 else None
    return {"n": n, "m": m, "matches": matches, "base": base}


def load_model(
    name: str,
    quality: int,
    device: str,
    checkpoint_path: Optional[Path] = None,
) -> torch.nn.Module:
    ckpt_key = str(checkpoint_path) if checkpoint_path else None
    key = (name, quality, device, ckpt_key)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if checkpoint_path:
        model = image_models[name](quality=quality, pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = _strip_module_prefix(_extract_state_dict(ckpt))
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        if hasattr(model, "update"):
            model.update(force=True)
    else:
        model = image_models[name](quality=quality, pretrained=True)
        model.eval().to(device)

    _MODEL_CACHE[key] = model
    return model


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
