import sys
import math
from PIL import Image, ImageChops
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    print("Installing pytorch-msssim...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-msssim"])
    from pytorch_msssim import ms_ssim


def compute_psnr(x, y):
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare.py <input_image> <recon_image>")
        sys.exit(1)

    input_path, recon_path = sys.argv[1], sys.argv[2]

    # Load and convert to tensors
    img1 = Image.open(input_path).convert("RGB")
    img2 = Image.open(recon_path).convert("RGB")

    x = TF.to_tensor(img1).unsqueeze(0)
    y = TF.to_tensor(img2).unsqueeze(0)

    # Compute metrics
    psnr = compute_psnr(x, y)
    ssim_val = ms_ssim(x, y, data_range=1.0).item()

    print(f"PSNR: {psnr:.2f} dB")
    print(f"MS-SSIM: {ssim_val:.4f}")

    # Visual comparison
    diff = ImageChops.difference(img1, img2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1)
    axes[0].set_title("Original")
    axes[1].imshow(img2)
    axes[1].set_title("Reconstructed")
    axes[2].imshow(diff)
    axes[2].set_title("Difference (highlighted)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
