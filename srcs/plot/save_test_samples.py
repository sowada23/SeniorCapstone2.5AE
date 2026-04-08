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
        for key in ("input_center_t1", "true_t1", "pred_t1"):
            image = np.asarray(example[key])
            image_mask = np.isfinite(image) & (image > 0)
            mask = image_mask if mask is None else (mask | image_mask)

    if mask is None or not np.any(mask):
        first = np.asarray(examples[0]["input_center_t1"])
        return 0, first.shape[0], 0, first.shape[1]

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad + 1, mask.shape[0])
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def save_test_examples_svg(examples, out_dir, filename="t1_test_examples.svg"):
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

    col_titles = ["Input Center T1", "True T1", "Predicted T1", "Absolute Error T1"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title)

    for row_idx, example in enumerate(examples):
        images = [
            example["input_center_t1"],
            example["true_t1"],
            example["pred_t1"],
            example["abs_error_t1"],
        ]
        patient = _patient_label(example["t1_path"])
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

        axes[row_idx, 0].set_ylabel(f"{patient}\nz={z}", rotation=0, labelpad=48, va="center")

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
