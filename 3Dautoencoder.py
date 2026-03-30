#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 3D Autoencoder for paired T1/T2 brain MRI (MONAI + PyTorch)
# - Input: 2 channels (T1, T2) stacked as [2, D, H, W]
# - Output: reconstructed [2, D, H, W]
# - Loss: MSE (Mean Square Error)
#
# Dataset folder structure :
#   HCP/
#     100206/
#       T1w_acpc_dc_restore_brain.nii.gz
#       T2w_acpc_dc_restore_brain.nii.gz
#     100307/
#       T1w_acpc_dc_restore_brain.nii.gz
#       T2w_acpc_dc_restore_brain.nii.gz


# In[ ]:


# Packages

import os, glob, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, NormalizeIntensityd, ResizeWithPadOrCropd
)


# In[ ]:


# Config

DATA_ROOT = "/cluster/home/sowada23/MedAI/UAD/DATA/HCP"
OUTPUT_DIR = "./Output"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
NUM_WORKERS = 4
LEARNING_RATE= 1e-4
EPOCHS = 5
VAL_FRAC = 0.2
TARGET_SPATIAL_SIZE = (160, 192, 160)  # (D, H, W)
USE_AMP = (DEVICE.type == "cuda")


# In[ ]:


# Set seed
random.seed(SEED)


# In[ ]:


# Build patient file list

def build_file_list(root: str):
    # Find all T1 files and pair with corresponding T2 in the same folder
    t1_paths = sorted(glob.glob(os.path.join(root, "**", "T1w_acpc_dc_restore_brain.nii.gz"), recursive=True))
    items = []
    for t1 in t1_paths:
        folder = os.path.dirname(t1)
        t2 = os.path.join(folder, "T2w_acpc_dc_restore_brain.nii.gz")
        if os.path.exists(t2):
            items.append({"t1": t1, "t2": t2})
    return items

files = build_file_list(DATA_ROOT)
if len(files) == 0:
    raise RuntimeError(f"No paired T1/T2 files found under: {DATA_ROOT}")

random.shuffle(files)
n_val = int(len(files) * VAL_FRAC)
val_files = files[:n_val]
train_files = files[n_val:]

print(f"Found paired subjects: {len(files)} | train: {len(train_files)} | val: {len(val_files)}")


# In[ ]:


# Transforms
# - EnsureChannelFirstd makes each image [1, D, H, W]
# - ConcatItemsd stacks them into a single tensor key "x": [2, D, H, W]
# - NormalizeIntensityd(channel_wise=True, nonzero=True) normalizes each channel


# In[ ]:


train_tf = Compose([
    LoadImaged(keys=["t1", "t2"]),
    EnsureChannelFirstd(keys=["t1", "t2"]),
    ResizeWithPadOrCropd(keys=["t1", "t2"], spatial_size=TARGET_SPATIAL_SIZE),
    ConcatItemsd(keys=["t1", "t2"], name="x", dim=0),   # -> [2, D, H, W]
    NormalizeIntensityd(keys=["x"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["x"], device=None),
])

val_tf = Compose([
    LoadImaged(keys=["t1", "t2"]),
    EnsureChannelFirstd(keys=["t1", "t2"]),
    ResizeWithPadOrCropd(keys=["t1", "t2"], spatial_size=TARGET_SPATIAL_SIZE),
    ConcatItemsd(keys=["t1", "t2"], name="x", dim=0),
    NormalizeIntensityd(keys=["x"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["x"], device=None),
])

train_ds = Dataset(train_files, transform=train_tf)
val_ds   = Dataset(val_files, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))


# In[ ]:


# 3D Autoencoder model

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )

class AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=2, base=32, out_channels=2):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_channels, base)          # -> base
        self.down1 = nn.Conv3d(base, base*2, kernel_size=3, stride=2, padding=1)
        self.enc2 = conv_block(base*2, base*2)             # -> base*2

        self.down2 = nn.Conv3d(base*2, base*4, kernel_size=3, stride=2, padding=1)
        self.enc3 = conv_block(base*4, base*4)             # -> base*4

        self.down3 = nn.Conv3d(base*4, base*8, kernel_size=3, stride=2, padding=1)
        self.enc4 = conv_block(base*8, base*8)             # -> base*8

        self.down4 = nn.Conv3d(base*8, base*16, kernel_size=3, stride=2, padding=1)
        self.bottleneck = conv_block(base*16, base*16)     # -> base*16

        # Decoder
        self.up4 = nn.ConvTranspose3d(base*16, base*8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base*8, base*8)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base*4, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base*2, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = conv_block(base, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.enc4(self.down3(x3))
        xb = self.bottleneck(self.down4(x4))

        y = self.dec4(self.up4(xb))
        y = self.dec3(self.up3(y))
        y = self.dec2(self.up2(y))
        y = self.dec1(self.up1(y))
        y = self.out(y)
        return y

model = AutoEncoder3D(in_channels=2, base=32, out_channels=2).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss()
torch.amp.GradScaler('cuda')
# scaler = GradScaler(enabled=USE_AMP)


# In[ ]:


# Train / Validate

def run_val():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(DEVICE)  # [B, 2, D, H, W]
            with autocast(enabled=USE_AMP):
                x_hat = model(x)
                loss = criterion(x_hat, x)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")

best_val = float("inf")
best_path = os.path.join(OUTPUT_DIR, "best_ae3d_t1t2.pt")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_losses = []

    for batch in train_loader:
        x = batch["x"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            x_hat = model(x)
            loss = criterion(x_hat, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

    train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
    val_loss = run_val()

    dt = time.time() - t0
    print(f"Epoch {epoch:03d}/{EPOCHS} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | {dt:.1f}s")

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "epoch": EPOCHS,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "val_loss": best_val,
            "target_spatial_size": TARGET_SPATIAL_SIZE,
        }, best_path)
        print(f"  ✓ saved best: {best_path} (val {best_val:.6f})")

print("Training done. Best val:", best_val)


# In[ ]:


# Anomaly map

def compute_anomaly_map(x, x_hat):
    # x, x_hat: [1, 2, D, H, W]
    err = (x - x_hat) ** 2
    amap = err.mean(dim=1, keepdim=True)  # -> [1, 1, D, H, W]
    return amap


# In[ ]:


model.eval()
with torch.no_grad():
    batch = next(iter(val_loader))
    x = batch["x"].to(device)
    x_hat = model(x)
    amap = compute_anomaly_map(x, x_hat)
    print("x:", tuple(x.shape), "x_hat:", tuple(x_hat.shape), "amap:", tuple(amap.shape))

