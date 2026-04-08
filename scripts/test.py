import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

from srcs.engine.tester import test
from srcs.utils.config import load_config
from srcs.utils.checkpoint import resolve_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="./Output")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.ckpt, args.output_root)
    cfg = load_config(args.config)
    test(cfg, ckpt_path)


if __name__ == "__main__":
    main()
