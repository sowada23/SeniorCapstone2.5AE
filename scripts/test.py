import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

from srcs.engine.tester import test_once
from srcs.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_once(cfg, args.ckpt)


if __name__ == "__main__":
    main()
