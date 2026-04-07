#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# scripts/cache_hcp_numpy.py
import argparse
import os
from pathlib import Path
import sys
import nibabel as nib
import numpy as np

sys.path.append(os.path.abspath("."))

from srcs.data.hcp_dataset import build_file_list


def subject_id_from_path(path: str) -> str:
    return Path(path).parent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--t1-name", default="T1w_acpc_dc_restore_brain.nii.gz")
    parser.add_argument("--t2-name", default="T2w_acpc_dc_restore_brain.nii.gz")
    args = parser.parse_args()

    files = build_file_list(args.root, args.t1_name, args.t2_name)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for item in files:
        sid = subject_id_from_path(item["t1"])
        subject_dir = out_root / sid
        subject_dir.mkdir(parents=True, exist_ok=True)

        t1_out = subject_dir / "t1.npy"
        t2_out = subject_dir / "t2.npy"

        if not t1_out.exists():
            t1 = nib.load(item["t1"]).get_fdata(dtype=np.float32)
            np.save(t1_out, t1.astype(np.float32))

        if not t2_out.exists():
            t2 = nib.load(item["t2"]).get_fdata(dtype=np.float32)
            np.save(t2_out, t2.astype(np.float32))

        print(f"cached {sid}")


if __name__ == "__main__":
    main()

