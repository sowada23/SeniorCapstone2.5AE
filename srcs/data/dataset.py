import glob
import os

import numpy as np
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
        slice_axis=0,
        num_adjacent_slices=3,
        cache_subjects=False,
        target_depth=160,
    ):
        if num_adjacent_slices % 2 == 0:
            raise ValueError("num_adjacent_slices must be odd, e.g. 3 or 5")
        if slice_axis != 0:
            raise NotImplementedError("Currently only slice_axis=0 is supported.")

        self.files = files
        self.slice_axis = slice_axis
        self.num_adjacent_slices = num_adjacent_slices
        self.radius = num_adjacent_slices // 2
        self.cache_subjects = cache_subjects
        self.target_depth = target_depth
        self.array_tf = build_hcp_transforms(
            target_spatial_size=target_spatial_size,
            target_depth=target_depth,
        )

        self.subjects = []
        self.samples = []
        self._build_index()

    def _load_subject(self, item):
        t1 = np.load(item["t1"]).astype(np.float32)
        t2 = np.load(item["t2"]).astype(np.float32)

        t1 = np.expand_dims(t1, axis=0)
        t2 = np.expand_dims(t2, axis=0)

        data = self.array_tf({"t1": t1, "t2": t2})
        t1 = np.asarray(data["t1"][0], dtype=np.float32)
        t2 = np.asarray(data["t2"][0], dtype=np.float32)
        return t1, t2

    def _build_index(self):
        for item in self.files:
            subject_idx = len(self.subjects)

            if self.cache_subjects:
                t1, t2 = self._load_subject(item)
                self.subjects.append({"t1": t1, "t2": t2, "paths": item})
            else:
                self.subjects.append({"paths": item})

            for z in range(self.radius, self.target_depth - self.radius):
                self.samples.append({"subject_idx": subject_idx, "z": z})

    def __len__(self):
        return len(self.samples)

    def _get_subject_arrays(self, subject_idx):
        subject = self.subjects[subject_idx]
        if self.cache_subjects:
            return subject["t1"], subject["t2"], subject["paths"]

        paths = subject["paths"]
        t1, t2 = self._load_subject(paths)
        return t1, t2, paths

    def _extract_stack(self, vol, z):
        return vol[z - self.radius:z + self.radius + 1]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        z = sample["z"]
        t1, t2, paths = self._get_subject_arrays(sample["subject_idx"])

        t1_stack = self._extract_stack(t1, z)
        t2_stack = self._extract_stack(t2, z)

        x = np.concatenate([t1_stack, t2_stack], axis=0).astype(np.float32)
        y = np.stack([t1[z], t2[z]], axis=0).astype(np.float32)

        return {
            "x": x,
            "y": y,
            "z": z,
            "t1_path": paths["t1"],
            "t2_path": paths["t2"],
        }
