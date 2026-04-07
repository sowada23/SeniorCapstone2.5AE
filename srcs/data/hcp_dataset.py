import glob
import os
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, NormalizeIntensityd, ResizeWithPadOrCropd


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
        cache_subjects=True,
    ):
        if num_adjacent_slices % 2 == 0:
            raise ValueError("num_adjacent_slices must be odd, e.g. 3 or 5")
        if slice_axis != 0:
            raise NotImplementedError("This implementation currently supports slice_axis=0 only.")

        self.files = files
        self.slice_axis = slice_axis
        self.num_adjacent_slices = num_adjacent_slices
        self.radius = num_adjacent_slices // 2
        self.cache_subjects = cache_subjects

        self.array_tf = Compose([
            ResizeWithPadOrCropd(keys=["t1", "t2"], spatial_size=(160, *target_spatial_size)),
            NormalizeIntensityd(keys=["t1", "t2"], nonzero=True, channel_wise=True),
        ])

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
                depth = t1.shape[self.slice_axis]
                self.subjects.append({"t1": t1, "t2": t2, "paths": item})
            else:
                t1, _ = self._load_subject(item)
                depth = t1.shape[self.slice_axis]
                self.subjects.append({"paths": item})

            for z in range(self.radius, depth - self.radius):
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


def make_loaders(cfg: dict, device_type: str):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    files = build_file_list(
        root=data_cfg["root"],
        t1_name=data_cfg["t1_name"],
        t2_name=data_cfg["t2_name"],
    )
    if not files:
        raise RuntimeError(f"No paired T1/T2 files found under: {data_cfg['root']}")

    random.shuffle(files)

    n_all = len(files)
    n_test = int(n_all * data_cfg["test_frac"])
    n_val = int(n_all * data_cfg["val_frac"])

    test_files = files[:n_test]
    val_files = files[n_test:n_test + n_val]
    train_files = files[n_test + n_val:]

    if not train_files:
        raise RuntimeError("Train split is empty. Reduce val_frac/test_frac or add more data.")

    ds_kwargs = dict(
        target_spatial_size=data_cfg["target_spatial_size"],
        slice_axis=data_cfg["slice_axis"],
        num_adjacent_slices=data_cfg["num_adjacent_slices"],
        cache_subjects=data_cfg.get("cache_subjects", True),
    )

    train_ds = HCPDataset(train_files, **ds_kwargs)
    val_ds = HCPDataset(val_files, **ds_kwargs) if val_files else None
    test_ds = HCPDataset(test_files, **ds_kwargs) if test_files else None

    persistent_workers = bool(train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    loader_kwargs = dict(
        num_workers=train_cfg["num_workers"],
        pin_memory=(device_type == "cuda"),
        persistent_workers=persistent_workers,
    )
    if train_cfg["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = train_cfg["prefetch_factor"]

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    ) if val_ds is not None else None
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    ) if test_ds is not None else None

    return train_loader, val_loader, test_loader, n_all, len(train_files), len(val_files), len(test_files)
