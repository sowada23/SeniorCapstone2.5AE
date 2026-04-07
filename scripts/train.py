import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

from srcs.engine.trainer import train
from srcs.utils.config import load_config


def main():
    import numpy as np
    x = np.load("/cluster/home/sowada23/MedAI/UAD/DATA/HCP_numpy/100206/t1.npy")
    print(x.shape)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
