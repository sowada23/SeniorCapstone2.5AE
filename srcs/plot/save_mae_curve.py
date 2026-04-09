from pathlib import Path

import matplotlib.pyplot as plt


def save_mae_curve(history, out_dir, filename="mae_loss_curve.svg"):
    """
    history: list of dicts like
        [{"epoch": 1, "train_mae": 0.12, "val_mae": 0.15}, ...]
    out_dir: directory where plot will be saved
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / filename

    epochs = [item["epoch"] for item in history]
    train_mae = [item["train_mae"] for item in history]
    val_mae = [item["val_mae"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mae, marker="o", label="Train MAE")
    plt.plot(epochs, val_mae, marker="o", label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.title("Train/Val MAE Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.close()

