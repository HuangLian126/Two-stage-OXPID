import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)


def _load_draw_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_rendered_image(image, output_path, dpi):
    dpi = max(300, int(dpi))
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)

    suffix = os.path.splitext(output_path)[1].lower()
    save_kwargs = {"dpi": (dpi, dpi)}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs.update(quality=95, subsampling=0, optimize=True)
    elif suffix == ".png":
        save_kwargs["compress_level"] = 1
    image.save(output_path, **save_kwargs)


def draw(images, labels, boxes, scores, thrh=0.6, path="", dpi=300,
         line_width=5, font_size=14):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i].detach().cpu().numpy() if torch.is_tensor(scores[i]) else np.asarray(scores[i])
        box = boxes[i].detach().cpu().numpy() if torch.is_tensor(boxes[i]) else np.asarray(boxes[i])
        keep = scr.reshape(-1) > float(thrh)
        box = box.reshape(-1, 4)[keep]
        scrs = scr.reshape(-1)[keep]

        min_side = min(im.size)
        width = max(int(line_width), int(round(min_side / 300)))
        font = _load_draw_font(max(8, int(font_size)))
        padding = max(2, width // 2)
        box_color = (43, 187, 216)

        for b, score in zip(box, scrs):
            x1, y1, x2, y2 = [float(value) for value in b]
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=width)

            text = f"object {float(score):.2f}"
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=font)

            label_width = text_width + padding * 2
            label_height = text_height + padding * 2
            label_x = min(max(0, int(round(x1))), max(0, im.width - label_width))
            label_y = int(round(y1)) - label_height
            if label_y < 0:
                label_y = min(max(0, int(round(y1)) + width), max(0, im.height - label_height))

            draw.rectangle(
                [label_x, label_y, label_x + label_width, label_y + label_height],
                fill=box_color,
            )
            draw.text(
                (label_x + padding, label_y + padding),
                text=text,
                font=font,
                fill="black",
            )

        output_path = path or f"results_{i}.jpg"
        _save_rendered_image(im, output_path, dpi)
            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)
    if args.sliced:
        num_boxes = args.numberofboxes
        
        aspect_ratio = w / h
        num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
        num_rows = int(num_boxes / num_cols)
        slice_height = h // num_rows
        slice_width = w // num_cols
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        for i, slice_img in enumerate(slices):
            slice_tensor = transforms(slice_img)[None].to(args.device)
            with autocast():  # Use AMP for each slice
                output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
            torch.cuda.empty_cache() 
            labels, boxes, scores = output
            
            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            predictions.append((labels, boxes, scores))
        
        merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
        labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
    else:
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        
    draw(
        [im_pil], labels, boxes, scores,
        thrh=args.conf,
        path=args.output,
        dpi=args.dpi,
        line_width=args.line_width,
        font_size=args.font_size,
    )
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    parser.add_argument('--conf', type=float, default=0.6, help='visualization confidence threshold')
    parser.add_argument('-o', '--output', type=str, default='results_0.jpg')
    parser.add_argument('--dpi', type=int, default=300, help='output DPI metadata (minimum 300)')
    parser.add_argument('--line-width', type=int, default=5, help='minimum bounding-box width in pixels')
    parser.add_argument('--font-size', type=int, default=14, help='label font size in pixels')
    args = parser.parse_args()
    main(args)









