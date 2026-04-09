from pathlib import Path


def save_test_metrics(mse, mae, out_dir, filename="test_metrics.txt"):
    """
    Save final test metrics to a plain text file.

    Args:
        mse: final test MSE (float)
        mae: final test MAE (float)
        out_dir: directory where the file will be saved
        filename: output text filename
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Test MAE: {mae:.6f}\n")
