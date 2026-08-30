#!/usr/bin/env python3
"""Randomly copy images listed in a COCO JSON file."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly copy images referenced by a COCO JSON file."
    )
    parser.add_argument(
        "--json",
        required=True,
        type=Path,
        help="COCO JSON file containing images[*].file_name.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Root directory containing the original images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to which sampled images will be copied.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3000,
        help="Number of images to sample (default: 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to make sampling reproducible (default: 0).",
    )
    return parser.parse_args()


def load_file_names(json_path: Path) -> list[Path]:
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list):
        raise ValueError("Invalid COCO JSON: 'images' must be a list")

    file_names: list[Path] = []
    seen: set[str] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
            raise ValueError(f"Invalid image record at images[{index}]: missing file_name")

        normalized_name = image["file_name"].replace("\\", "/")
        relative_path = Path(normalized_name)
        if (
            not normalized_name
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                f"Unsafe or invalid file_name at images[{index}]: {image['file_name']!r}"
            )

        key = relative_path.as_posix()
        if key not in seen:
            seen.add(key)
            file_names.append(relative_path)

    if not file_names:
        raise ValueError("The COCO JSON does not contain any image file names")
    return file_names


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")
    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {args.source_dir}")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir:
        raise ValueError("--output-dir must be different from --source-dir")

    file_names = load_file_names(args.json)
    available = [name for name in file_names if (source_dir / name).is_file()]
    missing_count = len(file_names) - len(available)
    if len(available) < args.count:
        raise ValueError(
            f"Cannot sample {args.count} images: only {len(available)} of "
            f"{len(file_names)} JSON images exist under {source_dir} "
            f"({missing_count} missing)"
        )

    selected = random.Random(args.seed).sample(available, args.count)
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in selected:
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / relative_path, destination)

    print(f"JSON images: {len(file_names)}")
    print(f"Available source images: {len(available)}")
    print(f"Missing source images: {missing_count}")
    print(f"Copied images: {len(selected)}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
