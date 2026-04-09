from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _patient_label(path_or_subject_id: str) -> str:
    path = Path(path_or_subject_id)
    return path.parent.name or path.stem or str(path_or_subject_id)


def _compute_shared_crop_bounds(examples, pad=30):
    mask = None

    for example in examples:
        for key in ("original", "recon", "seg"):
            image = np.asarray(example[key])
            image_mask = np.isfinite(image) & (image > 0)
            mask = image_mask if mask is None else (mask | image_mask)

    if mask is None or not np.any(mask):
        first = np.asarray(examples[0]["original"])
        return 0, first.shape[0], 0, first.shape[1]

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad + 1, mask.shape[0])
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def save_brats_focus_svg(examples, out_dir, filename="t1_brats_overlays.svg", modality_label="T1"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(filename).with_suffix(".svg")

    if not examples:
        raise ValueError("No examples were provided for BraTS SVG export.")

    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    y0, y1, x0, x1 = _compute_shared_crop_bounds(examples)

    col_titles = [
        f"Original {modality_label}",
        f"Reconstructed {modality_label}",
        "Anomaly Overlay",
        "Tumor Mask + Anomaly",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title)

    for row_idx, example in enumerate(examples):
        original = np.asarray(example["original"])[y0:y1, x0:x1]
        recon = np.asarray(example["recon"])[y0:y1, x0:x1]
        anomaly = np.asarray(example["anomaly"])[y0:y1, x0:x1]
        seg = np.asarray(example["seg"])[y0:y1, x0:x1]

        axes[row_idx, 0].imshow(original, cmap="gray")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(recon, cmap="gray")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(original, cmap="gray")
        axes[row_idx, 2].imshow(anomaly, cmap="hot", alpha=0.5)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(original, cmap="gray")
        axes[row_idx, 3].contour(seg > 0, colors="lime", linewidths=1.0)
        axes[row_idx, 3].imshow(anomaly, cmap="hot", alpha=0.45)
        axes[row_idx, 3].axis("off")

        patient = example.get("subject_id") or _patient_label(example.get("path", "unknown"))
        z = int(example["z"])

        axes[row_idx, 0].text(
            0.02,
            0.98,
            f"Patient: {patient} | z={z}",
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, pad=3, edgecolor="none"),
        )

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)
