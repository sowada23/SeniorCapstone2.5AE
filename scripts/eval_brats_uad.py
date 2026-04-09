import argparse
import os
import sys
from pathlib import Path
import re

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))

from srcs.data.brats_dataset import BraTSDataset, build_brats_file_list
from srcs.models.autoencoder import AutoEncoder2D
from srcs.plot.save_brats_overlays import save_brats_focus_svg
from srcs.utils.device import get_device
from srcs.utils.seed import set_seed
from srcs.utils.config import load_config

DEFAULT_BRATS_ROOT = "/cluster/home/sowada23/MedAI/UAD/DATA/BraTS/BraTS2021/test"
DEFAULT_T1_NAME = "t1.nii.gz"
DEFAULT_T2_NAME = "t2.nii.gz"
DEFAULT_SEG_NAME = "seg.nii.gz"
DEFAULT_USE_AMP = True
DEFAULT_SAVE_OVERLAYS = True


def infer_output_dir_from_ckpt(ckpt_path: str):
    ckpt_path = Path(ckpt_path).resolve()
    return str(ckpt_path.parent.parent)


def find_latest_run_dir(output_root="./Output"):
    output_root = Path(output_root)
    if not output_root.exists():
        return str(output_root / "brats_uad_eval")

    run_dirs = []
    pattern = re.compile(r"^Output_(\d+)$")

    for path in output_root.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                run_idx = int(match.group(1))
                run_dirs.append((run_idx, path))

    if not run_dirs:
        return str(output_root / "brats_uad_eval")

    latest_dir = max(run_dirs, key=lambda x: x[0])[1]
    return str(latest_dir / "brats_uad_eval")


DEFAULT_OUTPUT_DIR = find_latest_run_dir("./Output")


def dice_score(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(np.float32)
    true_mask = true_mask.astype(np.float32)

    intersection = np.sum(pred_mask * true_mask)
    denom = np.sum(pred_mask) + np.sum(true_mask)
    return float((2.0 * intersection + eps) / (denom + eps))


def reduce_anomaly_map(y_hat, y, mode="abs"):
    if mode == "abs":
        residual = torch.abs(y_hat - y)
    elif mode == "sq":
        residual = (y_hat - y) ** 2
    else:
        raise ValueError("mode must be 'abs' or 'sq'")

    anomaly_map = residual.mean(dim=1, keepdim=True)
    return anomaly_map


def compute_modality_anomaly_map(pred_slice, true_slice, mode="abs"):
    if mode == "abs":
        return np.abs(pred_slice - true_slice)
    if mode == "sq":
        return (pred_slice - true_slice) ** 2
    raise ValueError("mode must be 'abs' or 'sq'")


def find_best_dice_threshold(scores, labels, num_thresholds=200):
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    best_dice = -1.0
    best_thr = float(thresholds[0])

    for thr in thresholds:
        pred = (scores >= thr).astype(np.uint8)
        dsc = dice_score(pred, labels)
        if dsc > best_dice:
            best_dice = dsc
            best_thr = float(thr)

    return best_thr, float(best_dice)


def save_metrics(out_dir, metrics):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "brats_uad_metrics.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    return str(out_path)


def _select_subject_samples(dataset, limit=3):
    selected = []

    for subject_idx, subject in enumerate(dataset.subjects[:limit]):
        min_z = dataset.radius
        max_z = subject["depth"] - dataset.radius - 1
        z = subject["depth"] // 2
        z = max(min_z, min(z, max_z))

        sample_idx = None
        for idx, sample in enumerate(dataset.samples):
            if sample["subject_idx"] == subject_idx and sample["z"] == z:
                sample_idx = idx
                break

        if sample_idx is not None:
            selected.append(sample_idx)

    return selected


def _collect_center_slice_examples(model, dataset, sample_indices, device, use_amp, anomaly_mode):
    examples = {"t1": [], "t2": []}

    with torch.no_grad():
        for sample_idx in sample_indices:
            sample = dataset[sample_idx]
            x = sample["x"].unsqueeze(0).to(device, non_blocking=True)
            y = sample["y"].unsqueeze(0).to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)

            y_np = y[0].detach().cpu().numpy()
            y_hat_np = y_hat[0].detach().cpu().numpy()
            seg_np = sample["seg"][0].detach().cpu().numpy()
            z = int(sample["z"])
            subject_id = sample["subject_id"]

            true_t1 = y_np[0]
            true_t2 = y_np[1]
            pred_t1 = y_hat_np[0]
            pred_t2 = y_hat_np[1]

            examples["t1"].append(
                {
                    "original": true_t1,
                    "recon": pred_t1,
                    "anomaly": compute_modality_anomaly_map(pred_t1, true_t1, mode=anomaly_mode),
                    "seg": seg_np,
                    "z": z,
                    "subject_id": subject_id,
                    "path": sample["t1_path"],
                }
            )

            examples["t2"].append(
                {
                    "original": true_t2,
                    "recon": pred_t2,
                    "anomaly": compute_modality_anomaly_map(pred_t2, true_t2, mode=anomaly_mode),
                    "seg": seg_np,
                    "z": z,
                    "subject_id": subject_id,
                    "path": sample["t2_path"],
                }
            )

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    brats_cfg = cfg["brats"]
    data_cfg = cfg["data"]

    brats_root = brats_cfg["root"]
    max_subjects = brats_cfg.get("max_subjects")
    target_size = data_cfg["target_spatial_size"]
    slice_axis = data_cfg["slice_axis"]
    num_adjacent_slices = data_cfg["num_adjacent_slices"]
    target_depth = brats_cfg.get("target_depth", 160)

    set_seed(brats_cfg["seed"])
    device = get_device()
    use_amp = True

    files = build_brats_file_list(brats_root)
    if not files:
        raise RuntimeError(f"No BraTS cases found under: {brats_root}")

    files = sorted(files, key=lambda x: x["t1"])
    if max_subjects is not None:
        files = files[:max_subjects]

    print(f"BraTS cases selected: {len(files)}")

    dataset = BraTSDataset(
        files=files,
        target_spatial_size=target_size,
        slice_axis=slice_axis,
        num_adjacent_slices=num_adjacent_slices,
        target_depth=target_depth,
    )

    loader = DataLoader(
        dataset,
        batch_size=brats_cfg["batch_size"],
        shuffle=False,
        num_workers=brats_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(brats_cfg["num_workers"] > 0),
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = AutoEncoder2D(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_scores = []
    all_labels = []
    anomaly_mode = cfg["brats"].get("anomaly_mode", "abs")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            print(f"Processing batch {batch_idx}/{len(loader)}")

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)
                anomaly_map = reduce_anomaly_map(y_hat, y, mode=anomaly_mode)

            scores = anomaly_map.detach().cpu().numpy()[:, 0]
            labels = (seg.detach().cpu().numpy()[:, 0] > 0).astype(np.uint8)

            for i in range(scores.shape[0]):
                all_scores.append(scores[i].reshape(-1))
                all_labels.append(labels[i].reshape(-1))

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if np.unique(all_labels).size < 2:
        raise RuntimeError("AUROC/AP need both positive and negative pixels in the labels.")

    auroc = float(roc_auc_score(all_labels, all_scores))
    ap = float(average_precision_score(all_labels, all_scores))
    best_thr, best_dice = find_best_dice_threshold(all_scores, all_labels, num_thresholds=200)

    metrics = {
        "num_cases": len(files),
        "num_slices": len(dataset),
        "anomaly_mode": anomaly_mode,
        "pixel_auroc": f"{auroc:.6f}",
        "pixel_ap": f"{ap:.6f}",
        "best_threshold": f"{best_thr:.6f}",
        "best_dice": f"{best_dice:.6f}",
    }

    output_dir = infer_output_dir_from_ckpt(args.ckpt)
    print(f"Using output directory: {output_dir}")

    metrics_path = save_metrics(output_dir, metrics)

    print(f"Cases: {len(files)}")
    print(f"Slices: {len(dataset)}")
    print(f"Pixel AUROC: {auroc:.6f}")
    print(f"Pixel AP: {ap:.6f}")
    print(f"Best threshold: {best_thr:.6f}")
    print(f"Best Dice: {best_dice:.6f}")
    print(f"Saved metrics: {metrics_path}")

    overlay_dir = Path(output_dir) / "overlays"
    sample_indices = _select_subject_samples(dataset, limit=3)

    if sample_indices:
        examples_by_modality = _collect_center_slice_examples(
            model=model,
            dataset=dataset,
            sample_indices=sample_indices,
            device=device,
            use_amp=use_amp,
            anomaly_mode=anomaly_mode,
        )

        t1_path = save_brats_focus_svg(
            examples_by_modality["t1"],
            out_dir=overlay_dir,
            filename="t1_center_slice_overlays.svg",
            modality_label="T1",
        )
        print(f"Saved overlay SVG: {t1_path}")

        t2_path = save_brats_focus_svg(
            examples_by_modality["t2"],
            out_dir=overlay_dir,
            filename="t2_center_slice_overlays.svg",
            modality_label="T2",
        )
        print(f"Saved overlay SVG: {t2_path}")
    else:
        print("No center-slice examples were selected for overlay SVG export.")


if __name__ == "__main__":
    main()
