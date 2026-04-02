import torch

from srcs.data.hcp_dataset import make_loaders
from srcs.models.autoencoder import AutoEncoder2D
from srcs.utils.device import get_device


def compute_anomaly_map(y, y_hat):
    err = (y - y_hat) ** 2
    amap = err.mean(dim=1, keepdim=True)
    return amap


def test_once(cfg: dict, ckpt_path: str):
    device = get_device()
    _, val_loader, _, _, _ = make_loaders(cfg, device.type)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_cfg = checkpoint.get("model_cfg", cfg["model"])

    model = AutoEncoder2D(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        batch = next(iter(val_loader))
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_hat = model(x)
        amap = compute_anomaly_map(y, y_hat)

    print("x:", tuple(x.shape))
    print("y:", tuple(y.shape))
    print("y_hat:", tuple(y_hat.shape))
    print("amap:", tuple(amap.shape))
    print("checkpoint val_loss:", checkpoint.get("val_loss"))
