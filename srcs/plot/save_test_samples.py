from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _patient_label(t1_path: str) -> str:
    path = Path(t1_path)
    return path.parent.name or path.stem


def save_test_examples_svg(examples, out_dir, filename="t1_test_examples.svg"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if not examples:
        raise ValueError("No examples were provided for SVG export.")

    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

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
            if col_idx == 3:
                ax.imshow(image, cmap="hot")
            else:
                ax.imshow(image, cmap="gray")
            ax.axis("off")

        axes[row_idx, 0].set_ylabel(f"{patient}\nz={z}", rotation=0, labelpad=48, va="center")

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
