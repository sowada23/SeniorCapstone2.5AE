from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_brats_overlay(
    t1_slice,
    seg_slice,
    anomaly_map,
    out_dir,
    subject_id,
    z,
    filename=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{subject_id}_z{int(z):03d}_overlay.png"

    out_path = out_dir / filename

    t1_slice = np.asarray(t1_slice, dtype=np.float32)
    seg_slice = np.asarray(seg_slice, dtype=np.float32)
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(t1_slice, cmap="gray")
    axes[0].set_title("T1")
    axes[0].axis("off")

    axes[1].imshow(t1_slice, cmap="gray")
    axes[1].imshow(anomaly_map, cmap="hot", alpha=0.5)
    axes[1].set_title("Anomaly Overlay")
    axes[1].axis("off")

    axes[2].imshow(seg_slice, cmap="gray")
    axes[2].set_title("Tumor Mask")
    axes[2].axis("off")

    axes[3].imshow(t1_slice, cmap="gray")
    axes[3].contour(seg_slice > 0, colors="lime", linewidths=1.0)
    axes[3].imshow(anomaly_map, cmap="hot", alpha=0.45)
    axes[3].set_title("Mask + Anomaly")
    axes[3].axis("off")

    fig.suptitle(f"{subject_id} | z={int(z)}")
    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return str(out_path)
