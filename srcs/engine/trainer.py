import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from srcs.data.datamodule import build_train_val_test_loaders
from srcs.models.autoencoder import AutoEncoder2D
from srcs.utils.ens_dir import ensure_dir
from srcs.utils.device import get_device
from srcs.utils.seed import set_seed
from srcs.plot.save_loss_curve import save_loss_curve
from srcs.utils.run_dir import make_run_dir


def run_val(model, val_loader, criterion, device, use_amp):
    if val_loader is None:
        return float("nan")

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)
                loss = criterion(y_hat, y)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def train(cfg: dict):
    set_seed(cfg["train"]["seed"])
    device = get_device()
    use_amp = bool(cfg["train"]["amp"] and device.type == "cuda")

    run_info = make_run_dir(cfg["train"]["output_dir"])
    ensure_dir(cfg["train"]["output_dir"])
    ensure_dir(run_info["weight_dir"])

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))

    print(f"Run directory: {run_info['run_dir']}")
    print(f"Using device: {device}")
    print(f"Scanning dataset root: {cfg['data']['root']}")
    print("Building train/val/test loaders...")

    train_loader, val_loader, test_loader, n_all, n_train, n_val, n_test = build_train_val_test_loaders(cfg, device.type)

    print(f"Found paired subjects: {n_all} | train: {n_train} | val: {n_val} | test: {n_test}")
    print(f"Train slice samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val slice samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"Test slice samples: {len(test_loader.dataset)}")

    model = AutoEncoder2D(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.MSELoss()
    scaler = GradScaler(device.type, enabled=use_amp)

    best_val = float("inf")
    best_path = os.path.join(run_info["weight_dir"], "best_ae25d_t1t2.pt")

    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            if batch_idx == 1:
                print(f"Epoch {epoch:03d}: first batch OK | x={tuple(x.shape)} y={tuple(y.shape)}")

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = run_val(model, val_loader, criterion, device, use_amp)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{cfg['train']['epochs']} | "
            f"train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | {dt:.1f}s"
        )

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "val_loss": best_val,
                    "model_cfg": cfg["model"],
                    "data_cfg": cfg["data"],
                },
                best_path,
            )
            print(f"saved best: {best_path} (val {best_val:.6f})")

    save_loss_curve(history, run_info["train_dir"], filename="mse_loss_curve.png")
    print("Training done.")
