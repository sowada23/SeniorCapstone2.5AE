import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

from srcs.engine.trainer import train
from srcs.utils.load import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
