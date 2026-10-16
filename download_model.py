"""Download the pretrained Qwen3 0.6B weights and tokenizer.

Follows "Build a Reasoning Model (From Scratch)", sections 2.4 and 2.6.

    python download_model.py                  # base model + tokenizer (~1.4 GB)
    python download_model.py --tokenizer-only # tokenizer only (~6 MB)
    python download_model.py --kind reasoning # official Qwen3 reasoning variant
"""

import argparse

from reasoning_from_scratch.qwen3 import download_qwen3_small

OUT_DIR = "qwen3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("base", "reasoning"),
        default="base",
        help="base is the pretrained model the book builds on (default)",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="skip the 1.4 GB weight file and fetch only tokenizer-*.json",
    )
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    download_qwen3_small(
        kind=args.kind,
        tokenizer_only=args.tokenizer_only,
        out_dir=args.out_dir,
    )
    print(f"\nFiles are in ./{args.out_dir}/")


if __name__ == "__main__":
    main()
