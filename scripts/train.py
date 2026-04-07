import argparse
import os
import sys
import traceback

sys.path.append(os.path.abspath("."))

from srcs.engine.trainer import train
from srcs.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        train(cfg)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
