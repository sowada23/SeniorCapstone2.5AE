from pathlib import Path

import matplotlib.pyplot as plt


def save_mse_curve(history, out_dir, filename="train_val_mse_curve.png"):
    """
    history: list of dicts like
        [{"epoch": 1, "train_loss": 0.12, "val_loss": 0.15}, ...]
    out_dir: directory where plot will be saved
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / filename

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train MSE")
    plt.plot(epochs, val_losses, marker="o", label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train/Val MSE Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
