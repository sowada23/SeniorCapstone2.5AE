import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))

from srcs.data.brats_dataset import BraTSDataset, build_brats_file_list
from srcs.models.autoencoder import AutoEncoder2D
from srcs.plot.save_brats_overlays import save_brats_overlay
from srcs.utils.device import get_device
from srcs.utils.seed import set_seed

DEFAULT_BRATS_ROOT = "/cluster/home/sowada23/MedAI/UAD/DATA/BraTS/BraTS2021/test"
DEFAULT_T1_NAME = "t1.nii.gz"
DEFAULT_T2_NAME = "t2.nii.gz"
DEFAULT_SEG_NAME = "seg.nii.gz"
DEFAULT_USE_AMP = True
DEFAULT_SAVE_OVERLAYS = True


from pathlib import Path
import re

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

    # Average across reconstructed channels -> [B, 1, H, W]
    anomaly_map = residual.mean(dim=1, keepdim=True)
    return anomaly_map


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats-root", type=str, default=DEFAULT_BRATS_ROOT)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--t1-name", type=str, default=DEFAULT_T1_NAME)
    parser.add_argument("--t2-name", type=str, default=DEFAULT_T2_NAME)
    parser.add_argument("--seg-name", type=str, default=DEFAULT_SEG_NAME)
    parser.add_argument("--target-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--target-depth", type=int, default=160)
    parser.add_argument("--slice-axis", type=int, default=1)
    parser.add_argument("--num-adjacent-slices", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--anomaly-mode", type=str, default="abs", choices=["abs", "sq"])
    parser.add_argument("--max-overlays", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # Optional override flags
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=DEFAULT_USE_AMP)

    parser.add_argument("--save-overlays", dest="save_overlays", action="store_true")
    parser.add_argument("--no-save-overlays", dest="save_overlays", action="store_false")
    parser.set_defaults(save_overlays=DEFAULT_SAVE_OVERLAYS)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    files = build_brats_file_list(
        root=args.brats_root,
        t1_name=args.t1_name,
        t2_name=args.t2_name,
        seg_name=args.seg_name,
    )
    if not files:
        raise RuntimeError(f"No BraTS cases found under: {args.brats_root}")

    dataset = BraTSDataset(
        files=files,
        target_spatial_size=args.target_size,
        slice_axis=args.slice_axis,
        num_adjacent_slices=args.num_adjacent_slices,
        target_depth=args.target_depth,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = AutoEncoder2D(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_scores = []
    all_labels = []
    tumor_examples = []
    overlay_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                y_hat = model(x)
                anomaly_map = reduce_anomaly_map(y_hat, y, mode=args.anomaly_mode)

            scores = anomaly_map.detach().cpu().numpy()[:, 0]
            labels = (seg.detach().cpu().numpy()[:, 0] > 0).astype(np.uint8)
            y_cpu = y.detach().cpu().numpy()

            for i in range(scores.shape[0]):
                score_map = scores[i]
                label_map = labels[i]

                all_scores.append(score_map.reshape(-1))
                all_labels.append(label_map.reshape(-1))

                if np.any(label_map > 0):
                    tumor_examples.append(
                        {
                            "subject_id": batch["subject_id"][i],
                            "z": int(batch["z"][i]),
                            "t1": y_cpu[i, 0],
                            "seg": label_map,
                            "anomaly": score_map,
                        }
                    )

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
        "anomaly_mode": args.anomaly_mode,
        "pixel_auroc": f"{auroc:.6f}",
        "pixel_ap": f"{ap:.6f}",
        "best_threshold": f"{best_thr:.6f}",
        "best_dice": f"{best_dice:.6f}",
    }

    metrics_path = save_metrics(args.output_dir, metrics)

    print(f"Cases: {len(files)}")
    print(f"Slices: {len(dataset)}")
    print(f"Pixel AUROC: {auroc:.6f}")
    print(f"Pixel AP: {ap:.6f}")
    print(f"Best threshold: {best_thr:.6f}")
    print(f"Best Dice: {best_dice:.6f}")
    print(f"Saved metrics: {metrics_path}")

    if args.save_overlays:
        overlay_dir = Path(args.output_dir) / "overlays"
        tumor_examples = sorted(
            tumor_examples,
            key=lambda x: float(np.max(x["anomaly"])),
            reverse=True,
        )

        for example in tumor_examples[: args.max_overlays]:
            save_brats_overlay(
                t1_slice=example["t1"],
                seg_slice=example["seg"],
                anomaly_map=example["anomaly"],
                out_dir=overlay_dir,
                subject_id=example["subject_id"],
                z=example["z"],
            )
            overlay_count += 1

        print(f"Saved overlays: {overlay_count}")


if __name__ == "__main__":
    main()
