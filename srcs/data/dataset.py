import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from srcs.data.transforms import build_hcp_transforms


def build_file_list(root: str, t1_name: str, t2_name: str):
    t1_paths = sorted(glob.glob(os.path.join(root, "**", t1_name), recursive=True))
    items = []
    for t1 in t1_paths:
        folder = os.path.dirname(t1)
        t2 = os.path.join(folder, t2_name)
        if os.path.exists(t2):
            items.append({"t1": t1, "t2": t2})
    return items


class HCPDataset(Dataset):
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
        self.array_tf = build_hcp_transforms(
            target_spatial_size=target_spatial_size,
            target_depth=target_depth,
        )

        self.subjects = []
        self.samples = []
        self._build_cache_and_index()

    def _load_subject(self, item):
        t1 = np.load(item["t1"]).astype(np.float32)
        t2 = np.load(item["t2"]).astype(np.float32)

        t1 = np.expand_dims(t1, axis=0)
        t2 = np.expand_dims(t2, axis=0)

        data = self.array_tf({"t1": t1, "t2": t2})
        t1 = np.asarray(data["t1"][0], dtype=np.float32)
        t2 = np.asarray(data["t2"][0], dtype=np.float32)

        t1 = np.rot90(t1, k=1, axes=(1, 2)).copy()
        t2 = np.rot90(t2, k=1, axes=(1, 2)).copy()

        return t1, t2

    def _axis_depth(self, vol):
        return vol.shape[self.slice_axis]

    def _extract_stack(self, vol, z):
        vol_by_slice = np.moveaxis(vol, self.slice_axis, 0)
        return vol_by_slice[z - self.radius:z + self.radius + 1]

    def _extract_slice(self, vol, z):
        return np.take(vol, z, axis=self.slice_axis)

    def _build_cache_and_index(self):
        for item in self.files:
            subject_idx = len(self.subjects)
            t1, t2 = self._load_subject(item)

            depth = self._axis_depth(t1)
            self.subjects.append({"t1": t1, "t2": t2, "paths": item, "depth": depth})

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

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "z": z,
            "t1_path": paths["t1"],
            "t2_path": paths["t2"],
        }
