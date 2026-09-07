"""Draw COCO GT boxes: training classes in green, unknown classes in red.

Run from the project root:
    python tools/visualize_gt_known_unknown.py
    python tools/visualize_gt_known_unknown.py -o output/my_gt --boxes-only

Only Pillow is required; no model weights, PyTorch, or GPU are needed.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "OXPID_M"
KNOWN_COLOR = (0, 170, 80)
UNKNOWN_COLOR = (230, 45, 45)


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_gt(image, annotations, categories, known_names, font, line_width, boxes_only):
    """Render every valid GT, including crowd/difficult annotations if present."""
    draw = ImageDraw.Draw(image)
    counts = {"known": 0, "unknown": 0, "invalid": 0}
    for annotation in annotations:
        name = categories[annotation["category_id"]]
        is_known = name in known_names
        color = KNOWN_COLOR if is_known else UNKNOWN_COLOR
        group = "known" if is_known else "unknown"
        # COCO bbox is [x, y, width, height], in original image pixels.
        x, y, width, height = map(float, annotation["bbox"])
        if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
            counts["invalid"] += 1
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.width, x + width), min(image.height, y + height)
        if x2 <= x1 or y2 <= y1:
            counts["invalid"] += 1
            continue
        box = (min(x1, image.width - 1), min(y1, image.height - 1),
               min(x2, image.width - 1), min(y2, image.height - 1))
        draw.rectangle(box, outline=color, width=line_width)
        counts[group] += 1
        if boxes_only:
            continue
        label = f"{group}: {name}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = right - left + 8, bottom - top + 8
        label_x = min(x1, max(0, image.width - label_width))
        label_y = min(max(0, y1 - label_height), max(0, image.height - label_height))
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height), fill=color)
        draw.text((label_x + 4 - left, label_y + 4 - top), label, font=font, fill="white")
    return counts


def main(args):
    train = load_json(args.train_json)
    test = load_json(args.test_json)
    # Match class names instead of assuming the two JSONs use identical IDs.
    known_names = {category["name"] for category in train["categories"]}
    categories = {category["id"]: category["name"] for category in test["categories"]}
    unknown_names = set(categories.values()) - known_names
    print(f"Known classes ({len(known_names)}), GREEN: {', '.join(sorted(known_names))}", flush=True)
    print(f"Unknown classes ({len(unknown_names)}), RED: {', '.join(sorted(unknown_names))}", flush=True)

    image_dir = args.img_folder.resolve()
    output_dir = args.output_dir.resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_dir}")
    if output_dir == image_dir:
        raise ValueError("Use a different output directory to preserve the original images.")

    image_ids = {info["id"] for info in test["images"]}
    if len(image_ids) != len(test["images"]):
        raise ValueError("Duplicate image IDs in test JSON.")
    annotations_by_image = defaultdict(list)
    for annotation in test["annotations"]:
        if annotation["category_id"] not in categories:
            raise ValueError(f"Undefined category_id: {annotation['category_id']}")
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"Undefined image_id: {annotation['image_id']}")
        annotations_by_image[annotation["image_id"]].append(annotation)

    images = test["images"][:args.limit] if args.limit else test["images"]
    if not images:
        raise ValueError("No test images to visualize.")
    paths = []
    destinations = set()
    for info in images:
        relative_path = Path(info["file_name"].replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Expected a relative image filename: {info['file_name']}")
        source = (image_dir / relative_path).resolve()
        destination = (output_dir / relative_path).resolve()
        if image_dir not in source.parents or output_dir not in destination.parents:
            raise ValueError(f"Image path leaves its root directory: {relative_path}")
        if source == destination:
            raise ValueError(f"Output would overwrite an original image: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing test image: {source}")
        if destination in destinations:
            raise ValueError(f"Duplicate output filename: {destination}")
        destinations.add(destination)
        paths.append((info, source, destination))

    font = load_font(args.font_size)
    totals = {"known": 0, "unknown": 0, "invalid": 0}
    for index, (info, source, destination) in enumerate(paths, 1):
        with Image.open(source) as original:
            image = original.convert("RGB")
        counts = draw_gt(image, annotations_by_image[info["id"]], categories,
                         known_names, font, args.line_width, args.boxes_only)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_options = {"quality": 95, "subsampling": 0} if destination.suffix.lower() in (".jpg", ".jpeg") else {}
        image.save(destination, **save_options)
        for group in totals:
            totals[group] += counts[group]
        if index % 100 == 0 or index == len(paths):
            print(f"Saved {index}/{len(paths)} images", flush=True)

    summary = {
        "train_json": str(args.train_json.resolve()),
        "test_json": str(args.test_json.resolve()),
        "image_folder": str(image_dir),
        "image_count": len(paths),
        "known_classes": sorted(known_names),
        "unknown_classes": sorted(unknown_names),
        "known_color_rgb": KNOWN_COLOR,
        "unknown_color_rgb": UNKNOWN_COLOR,
        "box_counts": totals,
    }
    (output_dir / "gt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"GT boxes: {totals}", flush=True)
    print(f"Output: {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-json", type=Path, default=DATA_ROOT / "train.json")
    parser.add_argument("--test-json", type=Path, default=DATA_ROOT / "test.json")
    parser.add_argument("--img-folder", type=Path, default=DATA_ROOT / "val")
    parser.add_argument("-o", "--output-dir", type=Path, default=PROJECT_ROOT / "output" / "oxpid_m_gt_known_unknown")
    parser.add_argument("--line-width", type=int, default=3, help="box line width in pixels")
    parser.add_argument("--font-size", type=int, default=20, help="label font size in pixels")
    parser.add_argument("--boxes-only", action="store_true", help="draw colored boxes without text labels")
    parser.add_argument("--limit", type=int, default=0, help="preview only the first N images; 0 means all")
    args = parser.parse_args()
    if args.line_width < 1 or args.font_size < 1 or args.limit < 0:
        parser.error("line width/font size must be positive and limit must be nonnegative")
    return args


if __name__ == "__main__":
    main(parse_args())

'''
conda activate HL_rtdetr

python tools/z_visualize_gt_known_unknow.py
'''