import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast

from srcs.data.hcp_dataset import make_loaders
from srcs.models.autoencoder import AutoEncoder2D
from srcs.utils.device import get_device
from srcs.utils.seed import set_seed



def test(cfg: dict, checkpoint_path: str):
    device = get_device()
    set_seed(cfg["train"]["seed"])

    use_amp = bool(cfg["train"]["amp"] and device.type == "cuda")

    _, _, test_loader, _, _, _, _ = make_loaders(cfg, device.type)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = AutoEncoder2D(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.MSELoss()
    losses = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)
                loss = criterion(y_hat, y)

            losses.append(loss.item())

    test_loss = float(np.mean(losses)) if losses else float("inf")
    print(f"Test MSE: {test_loss:.6f}")
    return test_loss
