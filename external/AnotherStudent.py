"""
My notes:
1) I changed the model from Bilal to another Model
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# Specific imports for the MBT2018 teacher
from compressai.zoo import mbt2018 
from compressai.models.google import ScaleHyperprior
from compressai.layers.layers import GDN
from compressai.models.utils import conv, deconv

# --- 1. MODEL DEFINITION (Student) ---
class SmallStudent(ScaleHyperprior):
    def __init__(self, N=64, M=96):
        super().__init__(N=N, M=M)
        self.M = M
        self.g_a = nn.Sequential(
            conv(3, N, stride=2), GDN(N),
            conv(N, N, stride=2), GDN(N),
            conv(N, N, stride=2), GDN(N),
            conv(N, M, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, stride=2), GDN(N, inverse=True),
            deconv(N, N, stride=2), GDN(N, inverse=True),
            deconv(N, N, stride=2), GDN(N, inverse=True),
            deconv(N, 3, stride=2),
        )

# --- 2. DATASET HANDLER WITH OFFSET ---
class HPCDatasetOffset(Dataset):
    def __init__(self, root_dir, max_images, offset, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        all_paths = []

        # Collect all paths from subdirectories
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in files:
                all_paths.append(os.path.join(subdir_path, f))

        # Select the specific chunk based on offset
        self.image_paths = all_paths[offset : offset + max_images]
        print(f"Dataset: Loaded {len(self.image_paths)} images (Starting from index {offset})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --- 3. DISTILLED TRAINER (MBT2018 TEACHER) ---
class DistilledTrainer(nn.Module):
    def __init__(self, student, teacher_quality=3, lmbda=0.09):
        super().__init__()
        self.student = student
        # Switched to MBT2018 teacher
        self.teacher = mbt2018(quality=teacher_quality, pretrained=True).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.lmbda = lmbda
        
        # Dynamically match teacher's latent channels
        teacher_channels = self.teacher.g_a[-1].out_channels
        self.adapter = nn.Conv2d(student.M, teacher_channels, kernel_size=1)
        print(f"Distillation: Student({student.M}ch) -> MBT Teacher({teacher_channels}ch)")

    def forward(self, x):
        with torch.no_grad():
            y_teacher = self.teacher.g_a(x)

        y_student = self.student.g_a(x)
        z_student = self.student.h_a(y_student)
        z_hat, z_likelihoods = self.student.entropy_bottleneck(z_student)
        scales_hat = self.student.h_s(z_hat)
        y_hat, y_likelihoods = self.student.gaussian_conditional(y_student, scales_hat)
        x_hat = self.student.g_s(y_hat)

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp_loss = sum(torch.log(lik).sum() for lik in [y_likelihoods, z_likelihoods]) / (-math.log(2) * num_pixels)
        mse_dist = F.mse_loss(x_hat, x)
        
        distill_loss = F.mse_loss(self.adapter(y_student), y_teacher)

        # Scale MSE by 255^2
        total_loss = bpp_loss + self.lmbda * (mse_dist * 255**2) + 0.5 * distill_loss
        return total_loss, bpp_loss, mse_dist

# --- 4. RESUMED TRAINING EXECUTION ---
def run_training():
    # Detect device and handle driver mismatch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = "./MBT2018Checkpoints" 
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Setup Data using Offset
    transform = transforms.Compose([transforms.RandomCrop(256), transforms.ToTensor()])
    dataset = HPCDatasetOffset(root_dir='/home/egb11/scratch/my_images', 
                               max_images=5000, offset=5000, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 2. Setup Models
    student = SmallStudent(N=64, M=96).to(device)
    trainer = DistilledTrainer(student, teacher_quality=3, lmbda=0.09).to(device)
    
    optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-4)
    aux_optimizer = torch.optim.Adam(student.entropy_bottleneck.parameters(), lr=1e-3)

    # 3. Resume from previous checkpoints
    load_path = os.path.join(checkpoint_dir, "mbt_distilled_latest.pth")
    if os.path.exists(load_path):
        print(f"Resuming from: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])

    # 4. Training Loop
    for epoch in range(1, 21): 
        student.train()
        for i, images in enumerate(loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            loss, bpp, mse = trainer(images)
            loss.backward()
            optimizer.step()

            aux_optimizer.zero_grad()
            student.aux_loss().backward()
            aux_optimizer.step()

            if i % 20 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] | Loss: {loss.item():.4f} | MSE: {mse.item():.5f}")

        # Update CDFs and Save (Aligned correctly outside the 'i' loop)
        student.update(force=True)
        save_path = os.path.join(checkpoint_dir, f"mbt_distilled_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    run_training()