from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _patient_label(t1_path: str) -> str:
    path = Path(t1_path)
    return path.parent.name or path.stem


def _compute_shared_crop_bounds(examples, pad=30):
    mask = None

    for example in examples:
        for key in ("input_center", "true", "pred"):
            image = np.asarray(example[key])
            image_mask = np.isfinite(image) & (image > 0)
            mask = image_mask if mask is None else (mask | image_mask)

    if mask is None or not np.any(mask):
        first = np.asarray(examples[0]["input_center"])
        return 0, first.shape[0], 0, first.shape[1]

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad + 1, mask.shape[0])
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def save_test_examples_svg(examples, out_dir, filename="test_examples.svg", modality_label="T1"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if not examples:
        raise ValueError("No examples were provided for SVG export.")

    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    y0, y1, x0, x1 = _compute_shared_crop_bounds(examples)

    col_titles = [
        f"Input Center {modality_label}",
        f"True {modality_label}",
        f"Predicted {modality_label}",
        f"Absolute Error {modality_label}",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title)

    for row_idx, example in enumerate(examples):
        images = [
            example["input_center"],
            example["true"],
            example["pred"],
            example["abs_error"],
        ]
        patient = _patient_label(example["path"])
        z = example["z"]

        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            cropped = image[y0:y1, x0:x1]

            if col_idx == 3:
                im = ax.imshow(cropped, cmap="hot")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label("Absolute Error", rotation=90, labelpad=8)
                cbar.ax.tick_params(labelsize=8)
            else:
                ax.imshow(cropped, cmap="gray")

            ax.axis("off")

        # Put patient number at the top-left of the row
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
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
