from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from datasets import load_from_disk


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/datasets/PITA/AIME_Gemma.hf",
        help="Path to a dataset saved with save_to_disk",
    )
    parser.add_argument("-n", "--num", type=int, default=5, help="Rows to show")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=["dataset", "model_family", "model_alias", "question", "answer", "prompt", "t", "context", "y_sample", "y_ref", "preferred"],
        help="Keys to print if present",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=240,
        help="Max characters per string field",
    )
    args = parser.parse_args()

    ds_path = Path(args.path)
    ds = load_from_disk(str(ds_path))

    print(f"Path: {ds_path}")
    print(f"Rows: {len(ds)}")
    print(f"Columns: {list(ds.features.keys())}")

    rows_to_show = min(int(args.num), len(ds))
    for row_index in range(rows_to_show):
        example = ds[row_index]
        print(f"\n[{row_index}]")
        for key in args.keys:
            if key in example:
                value = example[key]
                if isinstance(value, str):
                    value = truncate_text(value, int(args.width))
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()


