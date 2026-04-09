from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from srcs.data.datamodule import build_train_val_test_loaders
from srcs.models.autoencoder import AutoEncoder2D
from srcs.plot.save_test_samples import save_test_examples_svg
from srcs.utils.device import get_device
from srcs.utils.ens_dir import ensure_dir
from srcs.utils.seed import set_seed
from srcs.utils.save_test_metrics import save_test_metrics


def _resolve_test_output_dir(checkpoint_path: str) -> str:
    ckpt_path = Path(checkpoint_path).resolve()
    run_dir = ckpt_path.parent.parent
    test_dir = run_dir / "test"
    ensure_dir(str(test_dir))
    return str(test_dir)


def _select_subject_samples(test_dataset, limit=3):
    selected = []
    for subject_idx, subject in enumerate(test_dataset.subjects[:limit]):
        min_z = test_dataset.radius
        max_z = subject["depth"] - test_dataset.radius - 1
        z = subject["depth"] // 2
        z = max(min_z, min(z, max_z))

        sample_idx = None
        for idx, sample in enumerate(test_dataset.samples):
            if sample["subject_idx"] == subject_idx and sample["z"] == z:
                sample_idx = idx
                break

        if sample_idx is not None:
            selected.append(sample_idx)

    return selected


def _collect_examples(model, test_dataset, sample_indices, device, use_amp):
    examples = []
    radius = test_dataset.radius

    with torch.no_grad():
        for sample_idx in sample_indices:
            sample = test_dataset[sample_idx]
            x = sample["x"].unsqueeze(0).to(device, non_blocking=True)
            y = sample["y"].unsqueeze(0).to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)

            x_np = x[0].detach().cpu().numpy()
            y_np = y[0].detach().cpu().numpy()
            y_hat_np = y_hat[0].detach().cpu().numpy()

            true_t1 = y_np[0]
            pred_t1 = y_hat_np[0]

            examples.append(
                {
                    "input_center_t1": x_np[radius],
                    "true_t1": true_t1,
                    "pred_t1": pred_t1,
                    "abs_error_t1": np.abs(pred_t1 - true_t1),
                    "z": int(sample["z"]),
                    "t1_path": sample["t1_path"],
                }
            )

    return examples


def test(cfg: dict, checkpoint_path: str):
    device = get_device()
    set_seed(cfg["train"]["seed"])
    use_amp = bool(cfg["train"]["amp"] and device.type == "cuda")

    _, _, test_loader, _, _, _, n_test = build_train_val_test_loaders(cfg, device.type)

    if test_loader is None or n_test == 0:
        raise RuntimeError("Test split is empty. Check test_frac and dataset size.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = AutoEncoder2D(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    mse_losses = []
    mae_losses = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)
                mse_loss = mse_criterion(y_hat, y)
                mae_loss = mae_criterion(y_hat, y)

            mse_losses.append(mse_loss.item())
            mae_losses.append(mae_loss.item())

    mse_loss = float(np.mean(mse_losses)) if mse_losses else float("inf")
    mae_loss = float(np.mean(mae_losses)) if mae_losses else float("inf")

    test_out_dir = _resolve_test_output_dir(checkpoint_path)

    sample_indices = _select_subject_samples(test_loader.dataset, limit=3)
    if sample_indices:
        examples = _collect_examples(model, test_loader.dataset, sample_indices, device, use_amp)
        out_path = save_test_examples_svg(
            examples,
            _resolve_test_output_dir(checkpoint_path),
            filename="t1_test_examples.svg",
        )
        print(f"Saved test SVG: {out_path}")

    print(f"Test MSE: {mse_loss:.6f}")
    print(f"Test MAE: {mae_loss:.6f}")
    save_test_metrics(mse_loss, mae_loss, test_out_dir, filename="test_metrics.txt")


    return mse_loss, mae_loss
