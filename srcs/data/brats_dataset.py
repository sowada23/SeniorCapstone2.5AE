import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, NormalizeIntensityd, Resized
from torch.utils.data import Dataset


def build_brats_file_list(root: str, t1_name: str, t2_name: str, seg_name: str):
    t1_paths = sorted(glob.glob(os.path.join(root, "**", t1_name), recursive=True))
    items = []

    for t1 in t1_paths:
        folder = os.path.dirname(t1)
        t2 = os.path.join(folder, t2_name)
        seg = os.path.join(folder, seg_name)

        if os.path.exists(t2) and os.path.exists(seg):
            items.append({"t1": t1, "t2": t2, "seg": seg})

    return items


def build_brats_transforms(target_spatial_size, target_depth=160):
    return Compose(
        [
            Resized(
                keys=["t1", "t2", "seg"],
                spatial_size=(target_depth, *target_spatial_size),
                mode=("trilinear", "trilinear", "nearest"),
                anti_aliasing=True,
            ),
            NormalizeIntensityd(
                keys=["t1", "t2"],
                nonzero=True,
                channel_wise=True,
            ),
        ]
    )


class BraTSDataset(Dataset):
    def __init__(
        self,
        files,
        target_spatial_size,
        slice_axis=1,
        num_adjacent_slices=3,
        target_depth=160,
    ):
        if num_adjacent_slices % 2 == 0:
            raise ValueError("num_adjacent_slices must be odd, e.g. 3 or 5")
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")

        self.files = files
        self.slice_axis = slice_axis
        self.num_adjacent_slices = num_adjacent_slices
        self.radius = num_adjacent_slices // 2
        self.target_depth = target_depth
        self.array_tf = build_brats_transforms(
            target_spatial_size=target_spatial_size,
            target_depth=target_depth,
        )

        self.subjects = []
        self.samples = []
        self._build_cache_and_index()

    def _load_nifti(self, path: str):
        return nib.load(path).get_fdata(dtype=np.float32)

    def _load_subject(self, item):
        t1 = self._load_nifti(item["t1"])
        t2 = self._load_nifti(item["t2"])
        seg = self._load_nifti(item["seg"])

        t1 = np.expand_dims(t1, axis=0)
        t2 = np.expand_dims(t2, axis=0)
        seg = np.expand_dims(seg, axis=0)

        data = self.array_tf({"t1": t1, "t2": t2, "seg": seg})

        t1 = np.asarray(data["t1"][0], dtype=np.float32)
        t2 = np.asarray(data["t2"][0], dtype=np.float32)
        seg = np.asarray(data["seg"][0], dtype=np.float32)

        # Match the same orientation logic used in the HCP dataset.
        t1 = np.rot90(t1, k=1, axes=(1, 2)).copy()
        t2 = np.rot90(t2, k=1, axes=(1, 2)).copy()
        seg = np.rot90(seg, k=1, axes=(1, 2)).copy()

        seg = (seg > 0).astype(np.float32)
        return t1, t2, seg

    def _axis_depth(self, vol):
        return vol.shape[self.slice_axis]

    def _extract_stack(self, vol, z):
        vol_by_slice = np.moveaxis(vol, self.slice_axis, 0)
        return vol_by_slice[z - self.radius : z + self.radius + 1]

    def _extract_slice(self, vol, z):
        return np.take(vol, z, axis=self.slice_axis)

    def _build_cache_and_index(self):
        for item in self.files:
            subject_idx = len(self.subjects)
            t1, t2, seg = self._load_subject(item)

            depth = self._axis_depth(t1)
            self.subjects.append(
                {
                    "t1": t1,
                    "t2": t2,
                    "seg": seg,
                    "paths": item,
                    "depth": depth,
                    "subject_id": Path(item["t1"]).parent.name,
                }
            )

            for z in range(self.radius, depth - self.radius):
                self.samples.append({"subject_idx": subject_idx, "z": z})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        z = sample["z"]

        subject = self.subjects[sample["subject_idx"]]
        t1 = subject["t1"]
        t2 = subject["t2"]
        seg = subject["seg"]
        paths = subject["paths"]

        t1_stack = self._extract_stack(t1, z)
        t2_stack = self._extract_stack(t2, z)

        x = np.concatenate([t1_stack, t2_stack], axis=0).astype(np.float32)
        y = np.stack(
            [
                self._extract_slice(t1, z),
                self._extract_slice(t2, z),
            ],
            axis=0,
        ).astype(np.float32)

        seg_slice = self._extract_slice(seg, z).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "seg": torch.from_numpy(seg_slice).unsqueeze(0),
            "z": z,
            "subject_id": subject["subject_id"],
            "t1_path": paths["t1"],
            "t2_path": paths["t2"],
            "seg_path": paths["seg"],
        }
