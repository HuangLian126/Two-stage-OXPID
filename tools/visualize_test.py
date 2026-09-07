"""Save predictions and ground-truth comparisons for the full object test set."""

import argparse
import html
import math
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PRED_COLOR = (230, 45, 45)
GT_COLOR = (0, 170, 80)


def load_font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def clip_box(box, size):
    """Clip only the drawing coordinates; predictions remain in original pixels."""
    if not all(math.isfinite(value) for value in box):
        return None
    width, height = size
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    x1, x2 = [min(max(x, 0), width - 1) for x in (x1, x2)]
    y1, y2 = [min(max(y, 0), height - 1) for y in (y1, y2)]
    return x1, y1, x2, y2


def draw_boxes(image, boxes, labels, color, font, line_width):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        box = clip_box(box, image.size)
        if box is None:
            continue
        draw.rectangle(box, outline=color, width=line_width)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = right - left + 8, bottom - top + 8
        x = min(max(0, box[0]), max(0, image.width - text_width))
        y = min(max(0, box[1] - text_height), max(0, image.height - text_height))
        draw.rectangle((x, y, x + text_width, y + text_height), fill=color)
        draw.text((x + 4 - left, y + 4 - top), label, fill="white", font=font)
    return image


def save_visualizations(image, gt_boxes, pred_boxes, scores, output_dir,
                        filename, threshold, font_size, line_width):
    prediction_path = output_dir / "predictions" / filename
    comparison_path = output_dir / "comparison" / filename
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"quality": 95, "subsampling": 0} if prediction_path.suffix.lower() in (".jpg", ".jpeg") else {}
    font = load_font(font_size)
    prediction = draw_boxes(
        image, pred_boxes, [f"object {score:.2f}" for score in scores],
        PRED_COLOR, font, line_width,
    )
    truth = draw_boxes(
        image, gt_boxes, ["object"] * len(gt_boxes), GT_COLOR, font, line_width,
    )
    prediction.save(prediction_path, **save_options)

    header_height = max(font_size + 24, 48)
    comparison = Image.new("RGB", (2 * image.width, image.height + header_height), "white")
    comparison.paste(truth, (0, header_height))
    comparison.paste(prediction, (image.width, header_height))
    draw = ImageDraw.Draw(comparison)
    draw.text((12, 10), f"Ground truth: {len(gt_boxes)} object(s)", font=font, fill=GT_COLOR)
    draw.text(
        (image.width + 12, 10),
        f"Prediction: {len(scores)} object(s), score >= {threshold:g}",
        font=font, fill=PRED_COLOR,
    )
    comparison.save(comparison_path, **save_options)


def write_gallery(output_dir, rows, threshold):
    """A local, lazy-loading gallery keeps all test images easy to browse."""
    with (output_dir / "index.html").open("w", encoding="utf-8") as handle:
        handle.write(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Object test results</title><style>'
            'body{font:16px sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;'
            'background:#f3f4f6;color:#202124}figure{margin:20px 0;padding:12px;background:white}'
            'img{width:100%;height:auto}figcaption{padding:8px 0;overflow-wrap:anywhere}'
            'a{color:#185abc}</style><h1>Object 测试集可视化</h1>'
            f'<p>共 {len(rows)} 张；置信度阈值 {threshold:g}。'
            '左侧绿色为标注框，右侧红色为预测框及置信度。点击图片查看原尺寸。</p>'
        )
        for row in rows:
            filename = quote(row["output_file"], safe="/")
            caption = html.escape(row["file_name"])
            handle.write(
                f'<figure><figcaption>{row["image_id"]}: {caption} | '
                f'GT: {row["gt_count"]} | Pred: {row["prediction_count"]} | '
                f'<a href="predictions/{filename}">仅预测框</a></figcaption>'
                f'<a href="comparison/{filename}"><img loading="lazy" '
                f'src="comparison/{filename}" alt="{caption}"></a></figure>\n'
            )
        handle.write('</html>')


def main(args):
    # Keep --help and the drawing utilities usable without the training runtime.
    import torch
    from src.core import YAMLConfig

    if not Path(args.resume).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    cfg = YAMLConfig(args.config, resume=args.resume)
    if cfg.yaml_cfg.get("num_classes") != 1 or cfg.yaml_cfg.get("remap_mscoco_category", False):
        raise ValueError("Use an object config with num_classes: 1 and remap_mscoco_category: False.")

    loader_cfg = cfg.yaml_cfg["val_dataloader"]
    dataset_cfg = loader_cfg["dataset"]
    if args.img_folder is not None:
        dataset_cfg["img_folder"] = args.img_folder
    if args.ann_file is not None:
        dataset_cfg["ann_file"] = args.ann_file
    if not Path(dataset_cfg["ann_file"]).is_file():
        raise FileNotFoundError(f"Annotation file not found: {dataset_cfg['ann_file']}. Use --ann-file.")
    if not Path(dataset_cfg["img_folder"]).is_dir():
        raise FileNotFoundError(f"Image folder not found: {dataset_cfg['img_folder']}. Use --img-folder.")
    # Include the last partial batch and iterate every COCO image, even with no annotations.
    loader_cfg.update(shuffle=False, drop_last=False, batch_size=args.batch_size,
                      num_workers=args.num_workers)
    # A full trained checkpoint replaces backbone pretraining; no download is needed.
    if "PResNet" in cfg.yaml_cfg:
        cfg.yaml_cfg["PResNet"]["pretrained"] = False

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the training environment or --device cpu.")
    loader = cfg.val_dataloader
    dataset = loader.dataset
    if set(dataset.coco.cats) != {0}:
        raise ValueError("This single-object config requires category_id=0 in the annotations.")
    if not len(dataset):
        raise ValueError("The test annotation file contains no images.")

    # Match DetSolver.evaluate: eval mode, the normal postprocessor, and no extra NMS.
    model = cfg.model
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    use_ema = cfg.use_ema and checkpoint.get("ema") is not None
    state = checkpoint["ema"]["module"] if use_ema else checkpoint["model"]
    model.load_state_dict(state)
    del state, checkpoint
    model = model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    output_dir = Path(args.output_dir)
    for name in ("predictions", "comparison"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    print(f"Weights: {args.resume} ({'ema' if use_ema else 'model'})", flush=True)
    print(f"Test set: {dataset_cfg['ann_file']} ({len(dataset)} images)", flush=True)
    print(f"Images: {dataset_cfg['img_folder']}", flush=True)
    print(f"Output: {output_dir.resolve()} | confidence >= {args.conf:g}", flush=True)

    rows = []
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            # This repository uses [width, height], not [height, width].
            orig_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = postprocessor(model(samples), orig_sizes)
            for target, result in zip(targets, results):
                image_id = int(target["image_id"].item())
                info = dataset.coco.imgs[image_id]
                # Keep the original filename, extension, and any relative subdirectories.
                filename = Path(info["file_name"].replace("\\", "/"))
                if filename.is_absolute() or ".." in filename.parts:
                    raise ValueError(f"Expected a relative image filename: {info['file_name']}")
                with Image.open(Path(dataset.img_folder) / filename) as source:
                    original = source.convert("RGB")
                keep = result["scores"] >= args.conf
                boxes = result["boxes"][keep].cpu().tolist()
                scores = result["scores"][keep].cpu().tolist()
                # Prepare GT exactly as CocoDetection does, in original image coordinates.
                annotations = dataset.coco.imgToAnns.get(image_id, [])
                _, ground_truth = dataset.prepare(
                    original, {"image_id": image_id, "annotations": annotations},
                )
                gt_boxes = ground_truth["boxes"].tolist()
                save_visualizations(
                    original, gt_boxes, boxes, scores, output_dir, filename,
                    args.conf, args.font_size, args.line_width,
                )
                rows.append(dict(image_id=image_id, file_name=info["file_name"],
                                 output_file=filename.as_posix(), gt_count=len(gt_boxes),
                                 prediction_count=len(scores)))
            print(f"Saved {len(rows)}/{len(dataset)} images", flush=True)

    if len(rows) != len(dataset):
        raise RuntimeError(f"Incomplete export: {len(rows)}/{len(dataset)} images.")
    write_gallery(output_dir, rows, args.conf)
    print(f"Done. Open {(output_dir / 'index.html').resolve()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="the same YAML used for training")
    parser.add_argument("-r", "--resume", required=True, help="trained checkpoint.pth")
    parser.add_argument("-o", "--output-dir", default="output/oxpid_s_test_vis")
    parser.add_argument("-d", "--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--conf", type=float, default=0.3, help="draw scores >= this threshold")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-folder", help="override val_dataloader.dataset.img_folder")
    parser.add_argument("--ann-file", help="override val_dataloader.dataset.ann_file")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--font-size", type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers must be nonnegative")
    if args.line_width < 1 or args.font_size < 1:
        parser.error("--line-width and --font-size must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
