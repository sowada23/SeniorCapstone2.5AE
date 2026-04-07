import random

from torch.utils.data import DataLoader

from srcs.data.dataset import HCPDataset, build_file_list


def build_train_val_test_loaders(cfg, device_type: str):
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
        cache_subjects=data_cfg.get("cache_subjects", False),
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
