# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from compressai.registry import register_dataset


@register_dataset("ImageFolder")
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        
        max_attempts = 10  # Safety guard to prevent infinite loops on a fully corrupted dataset
        attempt = 0

        while attempt < max_attempts:
            file_path = self.samples[index]
            try:
                # Attempt to open and convert the image
                img = Image.open(file_path).convert("RGB")
                
                # Verify the file content (optional but helpful)
                # This line might also raise UnidentifiedImageError or OSError
                img.verify()
                
                # Reopen after verify (verify closes the file)
                img = Image.open(file_path).convert("RGB")
                
                # If successful, break the loop and proceed
                if self.transform:
                    return self.transform(img)
                return img
                
            except (UnidentifiedImageError, OSError) as e:
                # This catches the error you encountered (and related file reading errors)
                print(f"⚠️ Corrupt file skipped: {file_path} (Error: {e})")
                
                # Move to the next index (circularly)
                index = (index + 1) % len(self.samples)
                attempt += 1

        # If the loop finishes without returning (too many consecutive corrupt files)
        raise RuntimeError(
            f"Failed to load a valid image after {max_attempts} consecutive attempts. "
            "Please clean your dataset or increase max_attempts."
        )

    def __len__(self):
        return len(self.samples)
