"""
train_classifier_eval_yolov8_edl_pue.py — 两阶段开集目标检测（单文件版，CLIP + OWEL + EDL）

整体设计
--------
阶段 1：冻结的单类 class-agnostic RT-DETR(官方 rtdetr_pytorch) 检测器，只检测一类 "object"，生成候选框。
阶段 2：对候选框 crop，用 CLIP image encoder 提取特征：
  - 分类头 = 冻结的类别文本嵌入 + 共享文本适配器（见下"文本适配器版"）；
  - 损失   = EDL（Evidential Deep Learning）证据损失 + 弱 KL 正则（线性退火）
             + 可选 ETF-style 几何稳定正则（吸收 ENC/Neural Collapse 思想，不替换 EDL 监督）；
  - 开集   = EDL 不确定度（拒识 NOOD）+ Pseudo Unknown Embedding（发现 FOOD）。

★本版新增：开集评估指标 WI / AOSE / U-Recall（遵循 opendet2/OWOD 定义，IoU=0.5）
----------------------------------------------------------------------
  - AOSE  : 跨所有已知类，"未知物体被判为某已知类"的检测框绝对数量（整数）。
  - WI@0.8: 已知类召回=0.8 的操作点上，mean_k(FP_open)/mean_k(TP+FP)，再×100。
            等价于 (P_K / P_{K∪U} − 1)。FP_open = 判为已知却压在未知GT上的框。
  - U-Recall: 被检出的未知 GT 占比；不受"未知未穷尽标注"导致的FP影响，比 APU 可靠。
  评估时务必用低 --conf（如 0.05），否则召回到不了 0.8、WI 操作点不可比。

★文本适配器版改动（相对上一版，配合"视觉 backbone 必须解冻"的现实）
----------------------------------------------------------------------
背景：X光域差距主要在视觉侧，backbone 不解冻就检不出已知类（已由实验确认）。
      但 backbone 解冻后图像特征 f 漂出 CLIP 原始空间，而上一版把 w0 作为 buffer
      钉死在旧文本空间，导致 PUE（wU = w0 − α·w̄）几何不自洽、pue_hit 全触发/全不触发。

本版做法（让文本侧也能迁移到适配空间，且保持 w_k 与 w0 几何一致）：
  - 文本编码器仍"用完即弃"：只在初始化时编码类名/通用词，得到原始嵌入后整体 del。
  - 原始类嵌入 text_wk、通用词嵌入 w0 都冻结为 buffer（不再是可训练 class_embeds）。
  - 新增共享 TextAdapter（CLIP-Adapter 式残差瓶颈）：
        w_k  = Adapter(text_wk)        # 已知类锚点
        w0'  = Adapter(w0)             # 通用/未知方向
        w̄   = normalize(mean_k w_k)
        wU   = normalize( w0' − α·w̄ )  # 两端都过同一适配器 → 同一空间 → PUE 自洽
  - 训练参数 = TextAdapter + logit_scale + （解冻的）visual。文本塔不参与训练。

CLIP backbone 冻结开关
----------------------
  --unfreeze-backbone     解冻 CLIP image encoder 一起微调（X光建议解冻）。
  --backbone-lr           解冻时 backbone 的学习率（建议 1e-5~1e-6）。
  --freeze-epochs N       解冻模式下，前 N 个 epoch 仍冻结 backbone（先让适配器/温度收敛）。

几何稳定正则（ENC/Neural Collapse 思想的轻量接入）
---------------------------------------------------
  --geom-weight           约束 Adapter 后的类原型近似 Simplex ETF 均匀分布，默认 0.01。
  --geom-include-pue      把 PUE 伪未知方向也纳入原型集合，形成 K+1 个开集锚点，默认开。
  --geom-feature-weight   可选特征紧致项，把 crop 特征轻量拉向对应类锚点，默认 0（关闭）。

QuickGELU 一致性（★3，沿用上一版）
----------------------------------
  open_clip 用 pretrained="openai" 建模时激活是 QuickGELU；pretrained=None 重建得到普通
  GELU——激活不在 state_dict 里，加载会静默成功但特征全错。ckpt 保存 clip_pretrained 与
  visual 前向指纹，加载时自检并在不匹配时翻转 quick_gelu 重试。

依赖: torch, torchvision, pillow, numpy, open_clip_torch + lyuwenyu 官方 rtdetr_pytorch(src/)。
  pip install open_clip_torch
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
try:
    import open_clip
except ImportError as e:  # pragma: no cover
    raise ImportError("缺少 open_clip_torch，请先安装：pip install open_clip_torch") from e

# CLIP（OpenAI 权重）的归一化常数，与 ImageNet 不同
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ===========================================================================
# 0. 阶段1：单类 class-agnostic RT-DETR 检测器（lyuwenyu 官方 rtdetr_pytorch）
# ===========================================================================
# 与 ultralytics 封装不同，官方代码需要 (YAML config + checkpoint) 两样来建模，流程
# 照搬官方 tools/infer.py：
#   cfg = YAMLConfig(config, resume=ckpt)
#   state = ckpt['ema']['module'] if 'ema' else ckpt['model']
#   cfg.model.load_state_dict(state)
#   model = cfg.model.deploy(); postprocessor = cfg.postprocessor.deploy()
#   前向: outputs = model(images); labels,boxes,scores = postprocessor(outputs, orig_size)
# 关键差异（务必注意）：
#   1) 官方预处理只有 Resize(640)+ToTensor，没有 ImageNet/CLIP 归一化（0~1 输入）。
#   2) postprocessor(deploy) 直接给【原图像素 xyxy】，scores 已按分降序（top num_top_queries）。
#   3) RT-DETR 是 NMS-free：默认不做 NMS；本封装在 iou_thresh<1 时才做一次兜底去重。
#   4) orig_size 传的是 [w, h]（宽在前），与官方一致。

_RTDETR_DEFAULT_INPUT = 640  # 官方默认输入分辨率

class Stage1RTDETR:
    """官方 RT-DETR(deploy 模式) 单类检测器的薄封装。"""
    def __init__(self, deploy_model: nn.Module, device, input_size: int = 640):
        self.model = deploy_model          # forward(images, orig_sizes) -> (labels, boxes, scores)
        self.device = device
        self.input_size = int(input_size)
        # 官方 infer.py 的预处理：只 Resize+ToTensor，无归一化
        self.tf = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ])

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

def _ensure_rtdetr_on_path(rtdetr_root: str | None) -> str:
    """把官方 rtdetr_pytorch 目录（含 src/）加入 sys.path，便于 import src.core。"""
    import sys
    root = rtdetr_root or os.environ.get("RTDETR_ROOT")
    if not root:
        raise RuntimeError(
            "未指定 RT-DETR 官方代码目录。请用 --rtdetr-root 指向 .../RT-DETR/rtdetr_pytorch，"
            "或设置环境变量 RTDETR_ROOT。")
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "src")):
        raise RuntimeError(f"'{root}' 下找不到 src/ 目录，确认它是 rtdetr_pytorch 根目录。")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

def stage1_load_model(ckpt_path: str, device, cfg_path: str | None = None,
                      rtdetr_root: str | None = None,
                      input_size: int = _RTDETR_DEFAULT_INPUT) -> Tuple[Stage1RTDETR, int]:
    """加载官方 RT-DETR(deploy) 单类检测器。

    ckpt_path : 官方训练保存的 .pth（det_solver 保存，含 'model' 或 'ema'）。
    cfg_path  : 对应的 YAML config（如 configs/rtdetr/rtdetr_r50vd_6x_coco.yml，
                num_classes 改成 1）。官方代码必须靠它建模，缺了会报错。
    rtdetr_root: 官方 rtdetr_pytorch 根目录（含 src/）。也可用环境变量 RTDETR_ROOT。
    """
    _ensure_rtdetr_on_path(rtdetr_root)
    from src.core import YAMLConfig  # 来自官方仓库

    if not cfg_path:
        raise RuntimeError("官方 RT-DETR 需要 YAML config，请用 --rtdetr-config 指定。")

    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:                       # 优先用 EMA 权重（官方 infer.py 同逻辑）
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint                        # 兜底：裸 state_dict
    cfg.model.load_state_dict(state)

    class _DeployModel(nn.Module):
        def __init__(self, m, p):
            super().__init__()
            self.model = m.deploy()
            self.postprocessor = p.deploy()
        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    deploy = _DeployModel(cfg.model, cfg.postprocessor).to(device).eval()
    print(f"[stage1] RT-DETR(官方) 已加载 | config={cfg_path} | input={input_size}")
    return Stage1RTDETR(deploy, device, input_size), input_size

@torch.no_grad()
def stage1_detect(model: Stage1RTDETR, img: Image.Image, imgsz, device,
                  conf_thresh: float, iou_thresh: float, max_det: int) -> List[list]:
    """单图推理。返回 [[cls, score, x1, y1, x2, y2], ...]，xyxy 为原图像素坐标。

    注：imgsz 形参仅为兼容旧签名，实际输入分辨率由 model.input_size 决定（官方=640）。
        cls 一律记为 0（class-agnostic 单类 "object"）。"""
    img = img.convert("RGB")
    w, h = img.size
    im = model.tf(img)[None].to(device)
    orig = torch.tensor([[w, h]], dtype=torch.float32, device=device)  # 宽在前
    labels, boxes, scores = model.model(im, orig)   # deploy 模式 -> 三个张量 [1, N(,4)]

    boxes = boxes[0].detach().cpu().numpy()          # [N,4] 原图像素 xyxy
    scores = scores[0].detach().cpu().numpy()        # [N]   已按分降序

    keep = scores >= conf_thresh
    boxes, scores = boxes[keep], scores[keep]
    if boxes.shape[0] == 0:
        return []

    # RT-DETR 本身 NMS-free；仅当显式给了 iou_thresh(<1) 才做一次 class-agnostic 兜底去重
    if iou_thresh and iou_thresh < 1.0 and boxes.shape[0] > 1:
        idx = tvops.nms(torch.from_numpy(boxes).float(),
                        torch.from_numpy(scores).float(), float(iou_thresh)).numpy()
        boxes, scores = boxes[idx], scores[idx]

    order = np.argsort(-scores)[:max_det]            # 取 top max_det
    out = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        out.append([0, float(scores[i]), float(x1), float(y1), float(x2), float(y2)])
    return out

# ===========================================================================
# 1. 读取 COCO 标注
# ===========================================================================
def _resolve_path(file_name: str, img_root: str | None) -> str:
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name
    if img_root:
        cand = os.path.join(img_root, file_name)
        if os.path.exists(cand) or not os.path.exists(file_name):
            return cand
    return file_name

def load_coco(json_path: str, img_root: str | None):
    """返回 (samples, class_names)。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict) and "categories" in data and "annotations" in data and "images" in data, \
        "不是 COCO 格式（需要 images/annotations/categories）。"

    cats = sorted(data["categories"], key=lambda c: c["id"])
    catid2idx = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [str(c["name"]) for c in cats]

    imgid2file = {im["id"]: im["file_name"] for im in data["images"]}
    by_img: Dict[int, dict] = defaultdict(lambda: {"boxes": [], "labels": []})
    for a in data["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        if a.get("image_id") not in imgid2file:
            continue
        if a.get("category_id") not in catid2idx:
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        by_img[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        by_img[a["image_id"]]["labels"].append(catid2idx[a["category_id"]])

    samples = []
    for img_id, d in by_img.items():
        if not d["boxes"]:
            continue
        samples.append({
            "image_path": _resolve_path(imgid2file[img_id], img_root),
            "boxes": np.asarray(d["boxes"], dtype=np.float32),
            "labels": np.asarray(d["labels"], dtype=np.int64),
        })

    print(f"[data] 图片数: {len(samples)}  已知类别数: {len(class_names)}")
    print(f"[data] 已知类别: {class_names}")
    return samples, class_names

# ===========================================================================
# 2. 阶段1在线出框 + IoU 匹配 GT 贴标签
# ===========================================================================
def build_items(samples, yw, imgsz, device, args, tag="", add_gt=True):
    """对一批图片：阶段1出框 -> 与 GT 做 IoU 匹配 -> 生成 (path, box, label) 列表。"""
    items: List[Tuple[str, np.ndarray, int]] = []
    n_prop_kept, n_gt_added, n_img_ok = 0, 0, 0
    matched_gt_total, total_gt = 0, 0

    for si, s in enumerate(samples):
        path = s["image_path"]
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        gt_boxes = s["boxes"]
        gt_labels = s["labels"]
        total_gt += len(gt_boxes)

        dets = stage1_detect(yw, img, imgsz, device,conf_thresh=args.prop_conf, iou_thresh=args.prop_iou, max_det=args.prop_max_det)
        prop_boxes = np.asarray([[d[2], d[3], d[4], d[5]] for d in dets], dtype=np.float32) \
            if dets else np.zeros((0, 4), dtype=np.float32)

        matched_gt_mask = np.zeros(len(gt_boxes), dtype=bool)
        if len(prop_boxes) and len(gt_boxes):
            iou = tvops.box_iou(torch.from_numpy(prop_boxes), torch.from_numpy(gt_boxes)).numpy()
            max_iou = iou.max(axis=1)
            gt_idx = iou.argmax(axis=1)
            for n in range(len(prop_boxes)):
                if max_iou[n] >= args.pos_iou:
                    g = gt_idx[n]
                    items.append((path, prop_boxes[n], int(gt_labels[g])))
                    n_prop_kept += 1
                    matched_gt_mask[g] = True

        matched_gt_total += int(matched_gt_mask.sum())

        if add_gt:
            for b, l in zip(gt_boxes, gt_labels):
                items.append((path, b.astype(np.float32), int(l)))
                n_gt_added += 1

        n_img_ok += 1
        if (si + 1) % 200 == 0:
            print(f"  [{tag}] 已处理 {si + 1}/{len(samples)} 图")

    print(f"[stage1-{tag}] 有效图 {n_img_ok} | 匹配到的候选框 {n_prop_kept} | "
          f"并入GT框 {n_gt_added} | 总样本 {len(items)}")
    if total_gt:
        print(f"[stage1-{tag}] GT 召回率(被候选框命中比例): "
              f"{matched_gt_total}/{total_gt} = {matched_gt_total / total_gt:.3f}")
    return items

# ===========================================================================
# 3. 裁剪数据集
# ===========================================================================
@lru_cache(maxsize=64)
def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def build_transform(imgsz: int, train: bool):
    """CLIP 归一化（注意：不是 ImageNet 的 mean/std），BICUBIC 与 CLIP 预处理一致。"""
    if train:
        return transforms.Compose([
            transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def _jitter_expand_box(box, W, H, jitter, expand):
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand
    y2 += bh * expand
    if jitter > 0:
        x1 += random.uniform(-jitter, jitter) * bw
        x2 += random.uniform(-jitter, jitter) * bw
        y1 += random.uniform(-jitter, jitter) * bh
        y2 += random.uniform(-jitter, jitter) * bh
    x1 = max(0, min(W - 2, x1))
    x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 2, y1))
    y2 = max(y1 + 1, min(H, y2))
    return int(x1), int(y1), int(x2), int(y2)

class BoxCropDataset(Dataset):
    """直接吃 (path, box(xyxy), label) 列表，返回 (crop张量, label)。"""

    def __init__(self, items, imgsz=224, train=True, jitter=0.1, expand=0.1):
        self.items = items
        self.imgsz = imgsz
        self.jitter = jitter if train else 0.0
        self.expand = expand
        self.tf = build_transform(imgsz, train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box, lab = self.items[i]
        img = _open_rgb(path)
        W, H = img.size
        x1, y1, x2, y2 = _jitter_expand_box(box, W, H, self.jitter, self.expand)
        return self.tf(img.crop((x1, y1, x2, y2))), int(lab)

    def label_counts(self, num_classes):
        cnt = np.zeros(num_classes, dtype=np.int64)
        for _, _, lab in self.items:
            cnt[int(lab)] += 1
        return cnt

# ===========================================================================
# 4. CLIP + OWEL 分类器：CLIP image encoder + 冻结文本嵌入 + 共享文本适配器 + EDL
# ===========================================================================
# X 光安检定制 prompt 模板（多模板取平均后归一化，CLIP 标准做法）
PROMPT_TEMPLATES = {"xray": ["an X-ray image of a {}", "an X-ray scan of a {}", "a {} in an X-ray security scan", "a photo of a {}"], "plain": ["a photo of a {}"]}
# 通用 objectness 词（论文：w0 用 "object" 这类一般性词）
GENERIC_OBJECT_PROMPTS = {
    "xray": [
        "an X-ray image of an object",
        "an X-ray scan of an item",
        "a object in an X-ray security scan",
        "a photo of an object",
    ],
    "plain": ["a photo of an object"],}

@torch.no_grad()
def encode_text_embeddings(clip_model, tokenizer, class_names, prompt_style, device):
    """用 CLIP text encoder 编码类名（多模板平均）-> 初始已知类嵌入 W_K，以及通用词 w0。

    返回 (W_K [N, D] 已归一化, w0 [D] 已归一化)。本版：W_K/w0 之后冻结为 buffer，
    由共享 TextAdapter 学习到适配空间的映射；text encoder 用完即弃（disposable）。
    """
    templates = PROMPT_TEMPLATES[prompt_style]

    def _encode(prompts: List[str]) -> torch.Tensor:
        toks = tokenizer(prompts).to(device)
        emb = clip_model.encode_text(toks).float()        # [T, D]
        emb = F.normalize(emb, dim=-1).mean(dim=0)        # 模板平均
        return F.normalize(emb, dim=-1)                   # [D]

    wk = torch.stack([_encode([t.format(name.replace("_", " ")) for t in templates]) for name in class_names])           # [N, D]
    w0 = _encode(GENERIC_OBJECT_PROMPTS[prompt_style])    # [D]
    return wk, w0

class TextAdapter(nn.Module):
    """共享文本适配器（CLIP-Adapter 式残差瓶颈）。

        out = normalize( x + beta * MLP(x) ),   MLP: D -> D/ratio -> D

    已知类锚点 w_k 与通用方向 w0 共用同一个适配器：二者被同一映射送进同一适配空间，
    从而 PUE 的 wU = w0' − α·w̄ 几何重新自洽（这是相对自由 class_embeds 的核心好处）。
    用残差 + 小瓶颈是为了：监督信号少时不易过拟合，且初始（MLP≈0）时退化回原始文本嵌入。
    """
    def __init__(self, dim: int, ratio: int = 4, beta: float = 0.5):
        super().__init__()
        hidden = max(dim // ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        self.beta = float(beta)
        # 末层零初始化：训练初期 out≈normalize(x)，等价于"未适配的原始文本嵌入"，稳起步
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):                         # x: [..., D]（应已归一化）
        return F.normalize(x + self.beta * self.fc(x), dim=-1)

class CLIPEDLClassifier(nn.Module):
    """
    结构（对应论文 Fig.1 的"分类那一半"）：
        f      = normalize( CLIP_visual(crop) )           # [B, D] 图像嵌入
        w_k    = Adapter(text_wk)                         # [N, D] 适配后的类锚点（已归一化）
        cos_k  = f · w_k                                  # 余弦相似度
        logits = s * cos                                  # s = exp(logit_scale)，可学习温度
        e      = softplus(logits)；alpha = e + 1          # EDL 证据 -> Dirichlet 参数
        p_k    = alpha_k / S；u = N / S                   # 期望概率 / 不确定度

    可学习参数：TextAdapter + logit_scale +（解冻时）visual。
    冻结 buffer：text_wk（原始类文本嵌入）、w0（原始通用词嵌入）。文本塔初始化后即丢弃。

    Pseudo Unknown Embedding（仅推理用，论文 Eq.1/2，本版在适配空间内构造）：
        w_k = Adapter(text_wk)；w0' = Adapter(w0)
        w̄  = normalize(mean_k w_k)
        wU = normalize( w0' − alpha_pue · w̄ )
        cos_u = f · wU；若 cos_u > max_k cos_k 则判未知（FOOD 路径）。
    """

    def __init__(self, class_names: List[str], clip_model_name: str = "ViT-B-16",
                 clip_pretrained: str | None = "openai", prompt_style: str = "xray",
                 freeze_backbone: bool = True, logit_scale_init: float = 30.0,
                 force_quick_gelu: bool = False,
                 adapter_ratio: int = 4, adapter_beta: float = 0.5,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.clip_model_name = clip_model_name
        self.prompt_style = prompt_style
        self.freeze_backbone = freeze_backbone

        # ---- 构建 CLIP；text encoder 仅用于初始化嵌入，之后整体丢弃 ----
        clip_full = open_clip.create_model(clip_model_name, pretrained=clip_pretrained, force_quick_gelu=force_quick_gelu)
        clip_full = clip_full.to(device).eval()
        if clip_pretrained is not None:
            tokenizer = open_clip.get_tokenizer(clip_model_name)
            wk, w0 = encode_text_embeddings(clip_full, tokenizer, class_names, prompt_style, device)
        else:
            # eval/加载 ckpt 路径：随机占位，稍后被 load_state_dict 覆盖
            d = clip_full.visual.output_dim
            wk = F.normalize(torch.randn(self.num_classes, d), dim=-1)
            w0 = F.normalize(torch.randn(d), dim=-1)

        self.visual = clip_full.visual                    # 只保留图像塔
        del clip_full

        # ---- 本版：原始文本嵌入冻结为 buffer；共享适配器为唯一可学习的"文本侧" ----
        self.register_buffer("text_wk", wk.clone())       # [N, D] 冻结的原始类文本嵌入
        self.register_buffer("w0", w0.clone())            # [D]   冻结的原始通用词嵌入
        self.text_adapter = TextAdapter(wk.shape[1], ratio=adapter_ratio, beta=adapter_beta)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(logit_scale_init))))

        self.set_backbone_frozen(freeze_backbone)

    # ---- backbone 冻结开关（文本适配器始终可训练） ----
    def set_backbone_frozen(self, frozen: bool):
        self.freeze_backbone = frozen
        for p in self.visual.parameters():
            p.requires_grad_(not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.visual.eval()   # 冻结时 backbone 永远 eval（不更新 norm 统计/不启用 dropout）
        return self

    # ---- 前向 ----
    def encode_image(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                f = self.visual(x)
        else:
            f = self.visual(x)
        return F.normalize(f.float(), dim=-1)             # [B, D]

    def adapted_class_embeds(self) -> torch.Tensor:
        """w_k = Adapter(冻结文本嵌入)，已归一化。[N, D]"""
        return self.text_adapter(self.text_wk)

    def cos_logits(self, f):
        wk = self.adapted_class_embeds()                  # [N, D] 已归一化
        cos = f @ wk.t()                                  # [B, N]
        scale = self.logit_scale.exp().clamp(max=100.0)
        return cos, scale * cos

    def forward_feats(self, x):
        """返回 (logits, f)。logits = s·cos，f 为归一化图像嵌入。"""
        f = self.encode_image(x)
        _, logits = self.cos_logits(f)
        return logits, f

    def forward(self, x):
        return self.forward_feats(x)[0]

    # ---- Pseudo Unknown Embedding（论文 Eq.1/2，测试时构造；本版在适配空间内） ----
    def pseudo_unknown_embedding(self, pue_alpha: float, detach: bool = True) -> torch.Tensor:
        # w_k 与 w0 都过同一个 text_adapter -> 同一适配空间，PUE 几何自洽
        wk = self.adapted_class_embeds()                         # [N, D] 已归一化
        if detach:
            wk = wk.detach()
        w_bar = F.normalize(wk.mean(dim=0), dim=-1)              # 归一化均值方向
        w0_adapted = self.text_adapter(self.w0)                  # [D] 适配后的通用方向（已归一化）
        if detach:
            w0_adapted = w0_adapted.detach()
        wu = w0_adapted - pue_alpha * w_bar
        return F.normalize(wu, dim=-1)                           # [D]

    # ---- ★3 visual 前向指纹：保存/加载两侧用同一确定性探针对齐结构 ----
    @torch.no_grad()
    def visual_fingerprint(self, imgsz: int) -> torch.Tensor:
        """用全零探针过一遍 visual，取前 8 维作指纹（CPU float32）。"""
        was_training = self.training
        self.eval()
        device = next(self.visual.parameters()).device
        probe = torch.zeros(1, 3, imgsz, imgsz, device=device)
        fp = self.visual(probe).float().flatten()[:8].cpu().clone()
        self.train(was_training)
        return fp

# ===========================================================================
# 5. EDL 损失与开集分数
# ===========================================================================
def edl_alpha(logits: torch.Tensor) -> torch.Tensor:
    """evidence = softplus(logits) >= 0；alpha = evidence + 1。"""
    return F.softplus(logits) + 1.0

def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) )，逐样本。"""
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    dg = torch.digamma(alpha) - torch.digamma(S)
    kl = (torch.lgamma(S.squeeze(1)) - torch.lgamma(alpha).sum(dim=1)
          - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          + (alpha - 1.0).mul(dg).sum(dim=1))
    return kl

def edl_loss(alpha: torch.Tensor, labels: torch.Tensor, kl_w: float, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """EDL digamma 形式 + 误导证据的 KL 正则。"""
    K = alpha.size(1)
    y = F.one_hot(labels, K).float()
    S = alpha.sum(dim=1, keepdim=True)
    ll = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    alpha_tilde = y + (1.0 - y) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde)
    per_sample = ll + kl_w * kl
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
        return per_sample.sum() / sample_weight.sum().clamp_min(1e-8)
    return per_sample.mean()

def edl_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """u = K / S ∈ (0, 1]，越大越"不知道"。known_score = 1 − u。"""
    K = alpha.size(1)
    return K / alpha.sum(dim=1)

def msp_score(logits: torch.Tensor):
    return F.softmax(logits, dim=1).max(dim=1).values

def energy_score(logits: torch.Tensor):
    return torch.logsumexp(logits, dim=1)

def prototype_etf_loss(model: CLIPEDLClassifier, include_pue: bool = True,
                       pue_alpha: float = 1.0) -> torch.Tensor:
    """ETF-style 原型几何正则：让类锚点尽量均匀分布在单位球面上。

    不改变 EDL 的监督形式，只约束 Adapter 后的文本原型几何。若 include_pue=True，
    则把推理使用的 PUE 伪未知方向也纳入原型集合，相当于形成 K+1 个开放世界锚点。
    """
    wk = model.adapted_class_embeds()                            # [K, D]
    protos = [wk]
    if include_pue:
        wu = model.pseudo_unknown_embedding(pue_alpha, detach=False).unsqueeze(0)
        protos.append(wu)
    proto = torch.cat(protos, dim=0)                              # [M, D]
    m = proto.size(0)
    if m <= 1:
        return proto.new_zeros(())

    gram = proto @ proto.t()
    off_diag = ~torch.eye(m, dtype=torch.bool, device=proto.device)
    target = -1.0 / float(m - 1)
    return (gram.masked_select(off_diag) - target).pow(2).mean()

def feature_anchor_loss(model: CLIPEDLClassifier, feats: torch.Tensor, labels: torch.Tensor,
                        sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """NC-style 特征紧致项：把样本特征轻量拉向其类别文本锚点。"""
    wk = model.adapted_class_embeds()
    target = wk[labels]
    per_sample = 1.0 - (feats * target).sum(dim=1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
        return per_sample.sum() / sample_weight.sum().clamp_min(1e-8)
    return per_sample.mean()

# ===========================================================================
# 6. 训练与统计
# ===========================================================================
def run_epoch(model, loader, device, optimizer=None, cfg=None, class_weight=None):
    """一个 epoch。损失 = EDL（含 KL 正则）+ 可选几何稳定正则。"""
    assert cfg is not None
    train = optimizer is not None
    model.train(train)
    total, correct = 0, 0
    loss_sum = edl_loss_sum = proto_geo_sum = feat_geo_sum = u_sum = 0.0
    torch.set_grad_enabled(train)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, feats = model.forward_feats(x)
        alpha = edl_alpha(logits)

        sw = class_weight[y] if class_weight is not None else None
        loss_edl = edl_loss(alpha, y, cfg["kl_w"], sample_weight=sw)
        proto_geo = logits.new_zeros(())
        feat_geo = logits.new_zeros(())
        loss = loss_edl

        if cfg.get("geom_w", 0.0) > 0:
            proto_geo = prototype_etf_loss(
                model,
                include_pue=bool(cfg.get("geom_include_pue", True)),
                pue_alpha=float(cfg.get("geom_pue_alpha", 1.0)),
            )
            loss = loss + float(cfg["geom_w"]) * proto_geo

        if cfg.get("geom_feature_w", 0.0) > 0:
            feat_geo = feature_anchor_loss(model, feats, y, sample_weight=sw)
            loss = loss + float(cfg["geom_feature_w"]) * feat_geo

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = alpha.argmax(dim=1)
            correct += (pred == y).sum().item()
            bs = x.size(0)
            total += bs
            loss_sum += loss.item() * bs
            edl_loss_sum += loss_edl.item() * bs
            proto_geo_sum += proto_geo.item() * bs
            feat_geo_sum += feat_geo.item() * bs
            u_sum += edl_uncertainty(alpha).mean().item() * bs

    torch.set_grad_enabled(True)
    n = max(total, 1)
    return {
        "loss": loss_sum / n,
        "edl_loss": edl_loss_sum / n,
        "proto_geo": proto_geo_sum / n,
        "feat_geo": feat_geo_sum / n,
        "u": u_sum / n,
        "acc": correct / n,
    }

@torch.no_grad()
def _collect_scores(model, loader, device):
    """返回 (logits, known_score=1-u, labels)，全部在 CPU。"""
    model.eval()
    all_logits, all_score, all_y = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _ = model.forward_feats(x)
        alpha = edl_alpha(logits)
        score = 1.0 - edl_uncertainty(alpha)
        all_logits.append(logits.cpu())
        all_score.append(score.cpu())
        all_y.append(y)
    if not all_logits:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    return torch.cat(all_logits), torch.cat(all_score), torch.cat(all_y)

def train_main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples, class_names = load_coco(args.json, args.img_root)
    num_classes = len(class_names)
    if num_classes < 1:
        raise RuntimeError("没有任何已知类。")

    random.shuffle(samples)
    train_samples = samples
    print(f"[data] 训练图 {len(train_samples)}（无验证集）")

    print(f"\n[stage1] 加载冻结的单类 RT-DETR(官方) 检测器: {args.yw_ckpt}")
    yw, det_imgsz = stage1_load_model(args.yw_ckpt, device, cfg_path=args.rtdetr_config,
                                      rtdetr_root=args.rtdetr_root, input_size=args.det_input_size)
    yw.eval()
    for p in yw.parameters():
        p.requires_grad_(False)

    print("[stage1] 生成训练候选框 ...")
    train_items = build_items(train_samples, yw, det_imgsz, device, args, tag="train", add_gt=args.add_gt)

    del yw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not train_items:
        raise RuntimeError("没有任何训练样本，检查 --pos-iou / --prop-conf / 图片路径。")

    train_ds = BoxCropDataset(train_items, args.imgsz, train=True, jitter=args.jitter, expand=args.expand)
    counts = train_ds.label_counts(num_classes)
    print("[data] 各已知类训练样本数:", {class_names[i]: int(counts[i]) for i in range(num_classes)})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # ---- 模型：CLIP backbone + 冻结文本嵌入 + 共享文本适配器 ----
    freeze = not args.unfreeze_backbone
    model = CLIPEDLClassifier(
        class_names,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        prompt_style=args.prompt_style,
        freeze_backbone=True,   # 先冻结；解冻在 freeze_epochs 之后切换
        logit_scale_init=args.logit_scale_init,
        force_quick_gelu=False,  # ★3 训练侧不强制；openai 路径 open_clip 自动用 QuickGELU
        adapter_ratio=args.adapter_ratio,
        adapter_beta=args.adapter_beta,
        device=device,
    ).to(device)
    print(f"[model] CLIP={args.clip_model}({args.clip_pretrained})  prompt={args.prompt_style}  "
          f"head=冻结文本嵌入(N={num_classes}, D={model.text_wk.shape[1]}) + 共享TextAdapter"
          f"(ratio={args.adapter_ratio}, beta={args.adapter_beta}) + EDL")
    print(f"[model] backbone {'解冻微调(backbone_lr=%g, 前%d轮先冻结)' % (args.backbone_lr, args.freeze_epochs) if not freeze else '冻结（论文 OWEL 做法）'}")

    class_weight = None
    if args.class_balanced:
        w = 1.0 / np.clip(counts, 1, None)
        w = w / w.sum() * num_classes
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("[loss] 使用类别均衡权重:", {class_names[i]: float(w[i]) for i in range(num_classes)})

    # ---- 优化器：文本适配器 + 温度 一组；backbone 单独小学习率 ----
    head_params = list(model.text_adapter.parameters()) + [model.logit_scale]
    param_groups = [{"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if not freeze:
        param_groups.append({"params": model.visual.parameters(), "lr": args.backbone_lr, "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[loss] EDL(digamma) + KL 正则 (kl_weight={args.kl_weight}, 前{args.kl_anneal:.0%}个epoch线性渐入)")
    print(f"[loss] 几何稳定正则: proto_etf_weight={args.geom_weight} "
          f"(include_pue={args.geom_include_pue}, pue_alpha={args.pue_alpha}, anneal={args.geom_anneal:.0%}); "
          f"feature_anchor_weight={args.geom_feature_weight}")

    def save_ckpt(epoch: int, path: Path):
        """统计训练集 EDL/MSP/Energy 分数分位（建议阈值）并保存当前权重到 path。"""
        stat_ds = BoxCropDataset(train_items, args.imgsz, train=False, expand=args.expand)
        stat_ld = DataLoader(stat_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        logits, edl_s, _ = _collect_scores(model, stat_ld, device)
        msp = msp_score(logits)
        energy = energy_score(logits)
        pcts = [1, 5, 10, 25, 50]
        score_pct, suggested = {}, {}
        for name, s in [("edl", edl_s), ("msp", msp), ("energy", energy)]:
            arr = s.numpy()
            score_pct[name] = {int(p): float(np.percentile(arr, p)) for p in pcts}
            suggested[name] = float(np.percentile(arr, 5))
        # ★3 保存 visual 前向指纹 + clip_pretrained，eval 加载时自检结构一致性
        fp = model.visual_fingerprint(args.imgsz)
        torch.save({
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,          # ★3
            "visual_fingerprint": fp,                         # ★3
            "prompt_style": args.prompt_style,
            "imgsz": args.imgsz,
            "expand": args.expand,
            "logit_scale_init": args.logit_scale_init,
            "pue_alpha": args.pue_alpha,
            "adapter_ratio": args.adapter_ratio,              # 适配器结构（影响参数形状，必须存）
            "adapter_beta": args.adapter_beta,                # 残差权重（非参数，必须存）
            "geometry": {
                "proto_etf_weight": args.geom_weight,
                "proto_etf_include_pue": args.geom_include_pue,
                "proto_etf_anneal": args.geom_anneal,
                "feature_anchor_weight": args.geom_feature_weight,
            },
            "head": "clip_edl_adapter",
            "epoch": epoch,
            "osr": {"suggested_thresh": suggested, "score_percentiles": score_pct}}, str(path))
        print(f"[ckpt] epoch {epoch}: 已保存 {path}  (edl known_score 建议阈值 p5={suggested['edl']:.4f})")

    E = args.epochs
    for epoch in range(1, E + 1):
        # 解冻模式：前 freeze_epochs 轮只训文本适配器+温度，之后放开 backbone
        if not freeze and epoch == args.freeze_epochs + 1:
            model.set_backbone_frozen(False)
            print(f"[model] epoch {epoch}: 解冻 CLIP backbone（lr={args.backbone_lr}）")

        # KL 正则线性渐入：先让证据/判别收敛，再逐步压制误导证据（弱正则）
        ramp = float(np.clip(epoch / max(1.0, args.kl_anneal * E), 0.0, 1.0))
        geom_ramp = float(np.clip(epoch / max(1.0, args.geom_anneal * E), 0.0, 1.0))
        cfg = {
            "kl_w": args.kl_weight * ramp,
            "geom_w": args.geom_weight * geom_ramp,
            "geom_include_pue": args.geom_include_pue,
            "geom_pue_alpha": args.pue_alpha,
            "geom_feature_w": args.geom_feature_weight * geom_ramp,
        }

        tr = run_epoch(model, train_loader, device, optimizer, cfg, class_weight)
        scheduler.step()

        print(f"epoch {epoch:3d}/{E} | kl_w {cfg['kl_w']:.4f} | geom_w {cfg['geom_w']:.5f} "
              f"| feat_w {cfg['geom_feature_w']:.5f} | logit_scale {model.logit_scale.exp().item():.2f}")
        print(f"    train | loss {tr['loss']:.4f}  edl {tr['edl_loss']:.4f}  "
              f"proto_geo {tr['proto_geo']:.4f}  feat_geo {tr['feat_geo']:.4f}  "
              f"mean_u {tr['u']:.4f}  acc {tr['acc']:.3f}")

        if epoch % 10 == 0:
            ckpt_path = out_path.with_name(f"{out_path.stem}_ep{epoch}{out_path.suffix}")
            save_ckpt(epoch, ckpt_path)

    save_ckpt(E, out_path)
    print(f"\n[done] 训练完成，最终模型: {out_path}")
    print("       推理默认 --osr-method edl_pue（EDL不确定度 + 伪未知嵌入）；")
    print("       本版 PUE 的 w0 已过同一适配器（与 w_k 同空间）。若 pue_hit 仍异常，")
    print("       退回 --osr-method edl（对 backbone 漂移免疫，不依赖 PUE 几何）。")

# ===========================================================================
# 7. 推理：阶段1出框 + EDL 分类 + (EDL不确定度 / 伪未知嵌入) 开集判别
# ===========================================================================
class OpenSetCLIPEDLClassifier:
    def __init__(self, ckpt_path, device, pue_alpha: float | None = None):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck.get("head") in ("clip_edl_adapter", "clip_edl"), \
            "ckpt 不是本脚本 train 产出的权重（head 不匹配）"
        if ck.get("head") == "clip_edl":
            raise RuntimeError(
                "检测到旧版 ckpt（head=clip_edl，自由 class_embeds，无文本适配器）。\n"
                "本版结构为 冻结文本嵌入 + 共享适配器，state_dict 不兼容。\n"
                "请用本版脚本重新训练，或单独用旧版脚本评估旧 ckpt。")
        self.class_names = ck["class_names"]
        self.imgsz = int(ck["imgsz"])
        self.expand = float(ck.get("expand", 0.0))
        self.device = device
        self.pue_alpha = float(pue_alpha if pue_alpha is not None else ck.get("pue_alpha", 1.0))
        self.adapter_ratio = int(ck.get("adapter_ratio", 4))
        self.adapter_beta = float(ck.get("adapter_beta", 0.5))

        # ★3 重建结构时对齐激活函数（QuickGELU/GELU），加载后用指纹自检
        ck_pretrained = ck.get("clip_pretrained")
        if ck_pretrained is None:
            print("[ckpt][警告] 旧版 ckpt 未记录 clip_pretrained，假设训练时用的是 openai 权重(QuickGELU)。")
            quick_gelu = True
        else:
            quick_gelu = (str(ck_pretrained) == "openai")

        self.model = self._build_and_load(ck, quick_gelu)

        fp_ref = ck.get("visual_fingerprint")
        if fp_ref is not None:
            if not self._fingerprint_ok(fp_ref):
                print(f"[ckpt][警告] visual 指纹不匹配(quick_gelu={quick_gelu})，翻转激活函数重建重试 ...")
                quick_gelu = not quick_gelu
                self.model = self._build_and_load(ck, quick_gelu)
                if not self._fingerprint_ok(fp_ref):
                    raise RuntimeError(
                        "visual 前向输出与训练时不一致（QuickGELU/GELU 两种都试过）。\n"
                        "可能原因：open_clip 版本差异 / clip_model 名不一致 / ckpt 损坏。\n"
                        "请确认 eval 环境的 open_clip 版本与训练时相同。")
            print(f"[ckpt] visual 指纹自检通过 (quick_gelu={quick_gelu})")
        else:
            print("[ckpt][警告] 旧版 ckpt 无 visual 指纹，无法自检结构一致性。")

        self.model.to(device).eval()
        self.tf = build_transform(self.imgsz, train=False)
        osr = ck.get("osr", {})
        self.suggested = osr.get("suggested_thresh", {})
        self.percentiles = osr.get("score_percentiles", {})
        # 测试时构造伪未知嵌入（本版：w0 与 w_k 同过适配器）
        with torch.no_grad():
            self.wu = self.model.pseudo_unknown_embedding(self.pue_alpha).to(device)  # [D]

    # ---- ★3 辅助：按指定 quick_gelu 重建结构并加载权重 ----
    def _build_and_load(self, ck, quick_gelu: bool) -> CLIPEDLClassifier:
        model = CLIPEDLClassifier(
            self.class_names,
            clip_model_name=ck["clip_model"],
            clip_pretrained=None,
            prompt_style=ck.get("prompt_style", "xray"),
            freeze_backbone=True,
            logit_scale_init=float(ck.get("logit_scale_init", 30.0)),
            force_quick_gelu=quick_gelu,
            adapter_ratio=self.adapter_ratio,
            adapter_beta=self.adapter_beta,
            device="cpu",
        )
        model.load_state_dict(ck["state_dict"])
        return model

    def _fingerprint_ok(self, fp_ref: torch.Tensor) -> bool:
        fp_now = self.model.visual_fingerprint(self.imgsz)
        return torch.allclose(fp_now, fp_ref.float(), atol=1e-2, rtol=1e-2)

    def default_thresh(self, method: str) -> float:
        base = {"edl_pue": "edl", "edl": "edl", "pue": None, "msp": "msp", "energy": "energy"}[method]
        if base is None:
            return 0.0  # pue 路径不需要阈值
        t = self.suggested.get(base)
        return float(t) if t is not None else 0.5

    @torch.no_grad()
    def classify(self, img: Image.Image, boxes_xyxy, method="edl_pue", thresh=None):
        """
        判别规则：
            edl_pue : 未知 <= (cos_u > max_k cos_k)  OR  (1-u < thresh)
            edl     : 未知 <= 1-u < thresh
            pue     : 未知 <= cos_u > max_k cos_k
            msp/energy : 仅作对比基线
        """
        if thresh is None:
            thresh = self.default_thresh(method)

        W, H = img.size
        crops, valid = [], []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = _jitter_expand_box(box, W, H, 0.0, self.expand)
            if x2 <= x1 or y2 <= y1:
                valid.append(False)
                continue
            crops.append(self.tf(img.crop((x1, y1, x2, y2))))
            valid.append(True)

        results = [dict(known=False, cls_idx=-1, cls_name="未知", cls_prob=0.0,
                        osr_score=0.0, u=1.0, cos_unknown=0.0, pue_hit=False)
                   for _ in boxes_xyxy]
        if not crops:
            return results

        batch = torch.stack(crops).to(self.device)
        f = self.model.encode_image(batch)                       # [B, D]
        cos, logits = self.model.cos_logits(f)                   # [B, N]
        alpha = edl_alpha(logits)
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha / S                                         # EDL 期望概率
        u = edl_uncertainty(alpha)                               # [B]
        known_score = 1.0 - u
        cos_u = f @ self.wu                                      # [B] 与伪未知嵌入的相似度
        pue_hit = cos_u > cos.max(dim=1).values                  # FOOD：更像 "unknown"

        pred_prob, pred_idx = prob.max(dim=1)
        scores = {
            "edl_pue": known_score,
            "edl": known_score,
            "pue": known_score,           # 仅展示用；pue 的判别不走阈值
            "msp": prob.max(dim=1).values,
            "energy": energy_score(logits),
        }[method]

        if method == "edl_pue":
            unknown_mask = pue_hit | (known_score < thresh)
        elif method == "edl":
            unknown_mask = known_score < thresh
        elif method == "pue":
            unknown_mask = pue_hit
        else:  # msp / energy
            unknown_mask = scores < thresh

        cos = cos.cpu(); u = u.cpu(); cos_u = cos_u.cpu()
        pue_hit = pue_hit.cpu(); unknown_mask = unknown_mask.cpu()
        scores = scores.cpu(); pred_prob = pred_prob.cpu(); pred_idx = pred_idx.cpu()

        j = 0
        for i, ok in enumerate(valid):
            if not ok:
                continue
            known = not bool(unknown_mask[j])
            ci = int(pred_idx[j])
            results[i] = dict(
                known=known,
                cls_idx=ci if known else -1,
                cls_name=self.class_names[ci] if known else "unknown",
                cls_prob=float(pred_prob[j]),
                osr_score=float(scores[j]),
                u=float(u[j]),
                cos_unknown=float(cos_u[j]),
                pue_hit=bool(pue_hit[j]),
            )
            j += 1
        return results

def _get_font(size=16):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_dets(img, dets):
    """dets: list of (label, score, x1, y1, x2, y2, known)。已知=红框，未知=橙框。"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = _get_font(16)
    for label, score, x1, y1, x2, y2, known in dets:
        color = "green" if known else "red"
        text = f"{label} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 120, 18
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
        draw.text((x1 + 2, ty + 2), text, fill="white", font=font)
    return img

def predict_main(args):
    device = torch.device(args.device)
    yw, imgsz = stage1_load_model(args.yw_ckpt, device, cfg_path=args.rtdetr_config,
                                  rtdetr_root=args.rtdetr_root, input_size=args.det_input_size)
    img = Image.open(args.image).convert("RGB")

    dets = stage1_detect(yw, img, imgsz, device, conf_thresh=args.conf, iou_thresh=args.iou, max_det=args.max_det)
    print(f"[stage1] 候选框: {len(dets)}")
    if not dets:
        print("无候选框，结束。")
        return

    boxes = [(d[2], d[3], d[4], d[5]) for d in dets]
    det_scores = [d[1] for d in dets]

    clf = OpenSetCLIPEDLClassifier(args.clf_ckpt, device, pue_alpha=args.pue_alpha)
    thr = args.osr_thresh if args.osr_thresh is not None else clf.default_thresh(args.osr_method)
    print(f"[stage2] OSR 方法={args.osr_method}  阈值={thr:.4f}  pue_alpha={clf.pue_alpha}")
    res = clf.classify(img, boxes, method=args.osr_method, thresh=thr)

    final, n_known, n_unknown = [], 0, 0
    for det_s, box, r in zip(det_scores, boxes, res):
        if r["known"]:
            disp = det_s * r["cls_prob"]
            n_known += 1
        else:
            disp = det_s
            n_unknown += 1
        final.append((r["cls_name"], disp, box[0], box[1], box[2], box[3], r["known"]))

    vis = draw_dets(img, final)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(str(out_path))

    print(f"\n[result] 已知 {n_known} 个，未知 {n_unknown} 个:")
    for (name, s, x1, y1, x2, y2, known), r in zip(final, res):
        tag = "" if known else f"  <未知{'(PUE)' if r['pue_hit'] else '(EDL)'}>"
        print(f"  {name:12s} score={s:.3f} u={r['u']:.3f} cos_u={r['cos_unknown']:.3f} "
              f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]{tag}")
    print(f"\n保存可视化: {out_path}")

    if args.save_txt:
        Path(args.txt_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.txt_out, "w", encoding="utf-8") as f:
            for (name, s, x1, y1, x2, y2, _), r in zip(final, res):
                f.write(f"{name} {s:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                        f"u={r['u']:.6f} cos_u={r['cos_unknown']:.6f} pue={int(r['pue_hit'])}\n")
        print(f"保存 txt: {args.txt_out}")

# ===========================================================================
# 8. 批量评估：已知类 mAP + 未知类 mAP + 开集指标(WI / AOSE / U-Recall)
# ===========================================================================
def _voc_ap(recall, precision):
    """全点插值 AP（VOC2010+/COCO 风格）。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def _ap_for_class(preds, gt_by_img, iou_thresh):
    npos = int(sum(len(v) for v in gt_by_img.values()))
    if npos == 0:
        return None, 0
    if not preds:
        return 0.0, npos
    preds = sorted(preds, key=lambda x: -x[1])
    matched = {img: np.zeros(len(b), dtype=bool) for img, b in gt_by_img.items()}
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)
    for i, (img, _score, box) in enumerate(preds):
        gboxes = gt_by_img.get(img)
        if gboxes is None or len(gboxes) == 0:
            fp[i] = 1
            continue
        ious = tvops.box_iou(
            torch.tensor([box], dtype=torch.float32),
            torch.tensor(gboxes, dtype=torch.float32),
        ).numpy()[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thresh and not matched[img][j]:
            tp[i] = 1
            matched[img][j] = True
        else:
            fp[i] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / npos
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    return _voc_ap(recall, precision), npos

# ---------------------------------------------------------------------------
# 开集指标（遵循 opendet2/OWOD 定义，固定 IoU=0.5）
#   AOSE   : 跨所有已知类，"未知物体被判为某已知类"的检测框数（绝对计数）。
#   WI@R   : 已知类召回=R 的操作点上，mean_k(FP_open)/mean_k(TP+FP)，再×100。
#            FP_open = 被判为该已知类、却压在某未知GT上(IoU>=thresh)的框。
#   U-Recall: 被检出的未知GT占比，不受未知未穷尽标注的FP影响，比 APU 可靠。
# ---------------------------------------------------------------------------
def _eval_known_class_osr(preds, gt_by_img, gt_unknown_by_img, iou_thresh):
    """对一个已知类：算 VOC AP，同时统计开集错误（被判为该已知类、却压在未知GT上的框）。

    返回 (ap, npos, rec, prec, tp_plus_fp_cum, fp_open_cum, aose_count)，曲线按置信度降序累积。
    - ap: npos>0 时为 VOC AP，否则 None
    - rec/prec/tp_plus_fp_cum/fp_open_cum: 累积数组（无预测时为空数组）
    - aose_count: 该类下"未知被判为此已知类"的框数（AOSE 分量）
    """
    npos = int(sum(len(v) for v in gt_by_img.values()))
    if not preds:
        return (0.0 if npos > 0 else None), npos, \
               np.array([]), np.array([]), np.array([]), np.array([]), 0

    preds = sorted(preds, key=lambda x: -x[1])
    matched = {img: np.zeros(len(b), dtype=bool) for img, b in gt_by_img.items()}
    nd = len(preds)
    tp = np.zeros(nd); fp = np.zeros(nd); is_unk = np.zeros(nd)

    for i, (img, _score, box) in enumerate(preds):
        bt = torch.tensor([box], dtype=torch.float32)
        # 闭集 TP/FP：与该已知类 GT 匹配
        gboxes = gt_by_img.get(img)
        if gboxes is not None and len(gboxes) > 0:
            ious = tvops.box_iou(bt, torch.tensor(gboxes, dtype=torch.float32)).numpy()[0]
            j = int(ious.argmax())
            if ious[j] >= iou_thresh and not matched[img][j]:
                tp[i] = 1; matched[img][j] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
        # 开集错误：该框是否压在某个未知 GT 上（IoU>=thresh）
        ub = gt_unknown_by_img.get(img)
        if ub is not None and len(ub) > 0:
            uious = tvops.box_iou(bt, torch.tensor(ub, dtype=torch.float32)).numpy()[0]
            if uious.max() >= iou_thresh:
                is_unk[i] = 1

    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    fp_open_c = np.cumsum(is_unk)
    tp_plus_fp_c = tp_c + fp_c
    rec = tp_c / npos if npos > 0 else np.zeros(nd)
    prec = tp_c / np.maximum(tp_plus_fp_c, 1e-12)
    ap = _voc_ap(rec, prec) if npos > 0 else None
    return ap, npos, rec, prec, tp_plus_fp_c, fp_open_c, int(is_unk.sum())

def _compute_wi(curves, known_names, recall_level=0.8):
    """WI = mean_k(FP_open) / mean_k(TP+FP) at the given known-recall level（opendet2 用 0.8）。

    curves[name] = {"rec":, "tp_plus_fp":, "fp_open":}（均为累积数组）。
    """
    tp_plus_fps, fps = [], []
    for name in known_names:
        c = curves.get(name)
        if c is None or len(c["rec"]) == 0:
            continue
        rec = c["rec"]
        idx = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
        tp_plus_fps.append(c["tp_plus_fp"][idx])
        fps.append(c["fp_open"][idx])
    if not tp_plus_fps or np.mean(tp_plus_fps) == 0:
        return 0.0
    return float(np.mean(fps) / np.mean(tp_plus_fps))

def _unknown_recall(preds_unknown, gt_unknown_by_img, iou_thresh):
    """U-Recall：被检出的未知 GT 占比（不受未标注未知导致的 FP 影响，比 APU 可靠）。"""
    npos = int(sum(len(v) for v in gt_unknown_by_img.values()))
    if npos == 0:
        return None, 0
    matched = {img: np.zeros(len(b), dtype=bool) for img, b in gt_unknown_by_img.items()}
    hit = 0
    for img, _score, box in sorted(preds_unknown, key=lambda x: -x[1]):
        ub = gt_unknown_by_img.get(img)
        if ub is None or len(ub) == 0:
            continue
        ious = tvops.box_iou(torch.tensor([box], dtype=torch.float32),
                             torch.tensor(ub, dtype=torch.float32)).numpy()[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thresh and not matched[img][j]:
            matched[img][j] = True; hit += 1
    return hit / npos, npos

def load_coco_eval(json_path, img_root, known_names):
    """读测试集 COCO；类别名在 known_names 里算已知，否则合并为 __unknown__。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = {c["id"]: str(c["name"]) for c in data["categories"]}
    known_set = set(known_names)

    # ★1 防御：测试集类名必须能和训练时的已知类对上（字符串精确匹配）。
    test_names = set(cats.values())
    overlap = known_set & test_names
    print(f"[eval-data] 测试集中匹配到的已知类({len(overlap)}/{len(known_set)}): {sorted(overlap)}")
    missing = known_set - test_names
    if missing:
        print(f"[eval-data][警告] 以下已知类在测试集中找不到同名类别: {sorted(missing)}")
    if not overlap:
        raise RuntimeError(
            "测试集类名与训练时的已知类完全不匹配（精确字符串比较）！\n"
            f"  已知类(来自ckpt): {sorted(known_set)}\n"
            f"  测试集类别:       {sorted(test_names)}\n"
            "  请检查大小写、下划线/空格、中英文是否一致。")

    imgid2file = {im["id"]: im["file_name"] for im in data["images"]}
    gts = defaultdict(list)
    n_known_gt, n_unknown_gt = 0, 0
    for a in data["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        iid, cid = a.get("image_id"), a.get("category_id")
        if iid not in imgid2file or cid not in cats:
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        name = cats[cid]
        box = [x, y, x + w, y + h]
        if name in known_set:
            gts[iid].append((name, box)); n_known_gt += 1
        else:
            gts[iid].append(("__unknown__", box)); n_unknown_gt += 1
    images = [(iid, _resolve_path(fn, img_root)) for iid, fn in imgid2file.items()]
    unknown_cats = sorted({n for n in cats.values() if n not in known_set})
    print(f"[eval-data] 图片 {len(images)} | 已知GT {n_known_gt} | 未知GT {n_unknown_gt}")
    print(f"[eval-data] 已知类: {list(known_names)}")
    print(f"[eval-data] 未知类(合并为 unknown): {unknown_cats}")
    if n_known_gt == 0:
        raise RuntimeError("测试集中已知类 GT 数为 0，无法评估已知 mAP。请检查 test.json 的类名。")
    return images, gts

def eval_main(args):
    device = torch.device(args.device)
    clf = OpenSetCLIPEDLClassifier(args.clf_ckpt, device, pue_alpha=args.pue_alpha)
    known_names = list(clf.class_names)
    images, gts = load_coco_eval(args.json, args.img_root, known_names)

    yw, imgsz = stage1_load_model(args.yw_ckpt, device, cfg_path=args.rtdetr_config,
                                  rtdetr_root=args.rtdetr_root, input_size=args.det_input_size)
    yw.eval()

    thr = args.osr_thresh if args.osr_thresh is not None else clf.default_thresh(args.osr_method)
    if clf.suggested:
        print(f"[eval] ckpt 内建议阈值(训练集p5): {clf.suggested}")
    print(f"[eval] OSR={args.osr_method} 阈值={thr:.4f} pue_alpha={clf.pue_alpha}")

    preds_known = defaultdict(list)
    preds_unknown = []
    n_img = 0
    dbg_score, dbg_pue, dbg_u = [], [], []
    for iid, path in images:
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        dets = stage1_detect(yw, img, imgsz, device, conf_thresh=args.conf, iou_thresh=args.iou, max_det=args.max_det)

        n_img += 1
        if dets:
            boxes = [(d[2], d[3], d[4], d[5]) for d in dets]
            det_scores = [float(d[1]) for d in dets]
            res = clf.classify(img, boxes, method=args.osr_method, thresh=thr)
            for ds, box, r in zip(det_scores, boxes, res):
                dbg_score.append(r["osr_score"])
                dbg_pue.append(r["pue_hit"])
                dbg_u.append(r["u"])
                b = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                if r["known"]:
                    preds_known[r["cls_name"]].append((iid, ds * r["cls_prob"], b))
                else:
                    preds_unknown.append((iid, ds, b))
        if n_img % 100 == 0:
            print(f"  已评估 {n_img}/{len(images)} 图")

    if dbg_score:
        s = np.asarray(dbg_score, dtype=np.float64)
        u = np.asarray(dbg_u, dtype=np.float64)
        pue_rate = float(np.mean(dbg_pue))
        n_known_pred = int(sum(len(v) for v in preds_known.values()))
        print(f"\n[debug] 总候选框 {len(s)} | 判已知 {n_known_pred} | 判未知 {len(preds_unknown)}")
        print(f"[debug] osr_score 分位: p5={np.percentile(s,5):.4f}  p25={np.percentile(s,25):.4f}  "
              f"p50={np.percentile(s,50):.4f}  p75={np.percentile(s,75):.4f}  p95={np.percentile(s,95):.4f}")
        print(f"[debug] u(不确定度) 分位: p5={np.percentile(u,5):.4f}  p50={np.percentile(u,50):.4f}  "
              f"p95={np.percentile(u,95):.4f}")
        print(f"[debug] pue_hit 触发比例: {pue_rate:.3f} | 使用阈值: {thr:.4f}")
        if n_known_pred == 0:
            print("[debug][警告] 没有任何框被判为已知！可能原因：")
            print("              1) 阈值过高（对比上面 osr_score 分位与阈值）；")
            print("              2) pue_hit 比例≈1（PUE 路径全触发，尝试 --osr-method edl 或调小 --pue-alpha）；")
            print("              3) u≈1（特征坍掉，多半是 QuickGELU/结构不一致——但本版已有指纹自检）。")

    gt_known = defaultdict(lambda: defaultdict(list))
    gt_unknown = defaultdict(list)
    for iid, anns in gts.items():
        for name, box in anns:
            if name == "__unknown__":
                gt_unknown[iid].append(box)
            else:
                gt_known[name][iid].append(box)
    gt_known = {c: {i: np.asarray(b, np.float32) for i, b in d.items()} for c, d in gt_known.items()}
    gt_unknown = {i: np.asarray(b, np.float32) for i, b in gt_unknown.items()}

    def map_at(iou):
        per_class = {}
        for name in known_names:
            ap, npos = _ap_for_class(preds_known.get(name, []), gt_known.get(name, {}), iou)
            if ap is not None:
                per_class[name] = (ap, npos)
        known_map = float(np.mean([v[0] for v in per_class.values()])) if per_class else 0.0
        u_ap, u_npos = _ap_for_class(preds_unknown, gt_unknown, iou)
        return known_map, per_class, (u_ap, u_npos)

    print("\n========== 评估结果 ==========")
    known_map, per_class, (u_ap, u_npos) = map_at(args.map_iou)
    print(f"\n[已知类 AP @IoU={args.map_iou}]")
    for name in known_names:
        if name in per_class:
            ap, npos = per_class[name]
            print(f"  {name:14s} AP={ap:.4f}  (GT={npos})")
        else:
            print(f"  {name:14s} (测试集无该类GT，跳过)")
    print(f"  -> 已知类 mAP@{args.map_iou} (mAPK) = {known_map:.4f}")
    if u_ap is None:
        print(f"\n[未知类] 测试集无未知GT，无法计算 APU")
    else:
        print(f"\n[未知类] AP@{args.map_iou} (APU) = {u_ap:.4f}  "
              f"(GT={u_npos}, 检出未知框 {len(preds_unknown)})")

    if args.coco_map:
        ious = [round(0.5 + 0.05 * k, 2) for k in range(10)]
        kmaps, umaps = [], []
        for iou in ious:
            km, _, (ua, _) = map_at(iou)
            kmaps.append(km)
            if ua is not None:
                umaps.append(ua)
        print(f"\n[mAP@[.5:.95]] 已知类 = {np.mean(kmaps):.4f}", end="")
        if umaps:
            print(f" | 未知类 = {np.mean(umaps):.4f}", end="")
        print()

    # ---- 开集指标 WI / AOSE / U-Recall（固定 IoU=0.5，遵循 opendet2/OWOD）----
    OSR_IOU = float(args.osr_iou)
    curves, aose_total = {}, 0
    for name in known_names:
        _, _, rec, _, tpfp, fpo, aose_c = _eval_known_class_osr(
            preds_known.get(name, []), gt_known.get(name, {}), gt_unknown, OSR_IOU)
        curves[name] = {"rec": rec, "tp_plus_fp": tpfp, "fp_open": fpo}
        aose_total += aose_c

    wi_main = _compute_wi(curves, known_names, recall_level=args.wi_recall)
    u_recall, n_unk_gt = _unknown_recall(preds_unknown, gt_unknown, OSR_IOU)
    wi_sweep = {round(r / 10, 1): round(_compute_wi(curves, known_names, r / 10) * 100, 2)
                for r in range(1, 10)}

    print(f"\n========== 开集指标 (IoU={OSR_IOU}) ==========")
    print(f"[AOSE]   未知被判为已知的框数 = {aose_total}")
    print(f"[WI@{args.wi_recall}] (×100) = {wi_main * 100:.2f}")
    print(f"[WI sweep] (×100, recall 0.1~0.9): {wi_sweep}")
    if u_recall is not None:
        print(f"[U-Recall] = {u_recall:.4f}  (未知GT={n_unk_gt}, 检出未知框={len(preds_unknown)})")
    else:
        print("[U-Recall] 测试集无未知GT，无法计算")
    print("\n================================")
    apu_str = f"{u_ap:.4f}" if u_ap is not None else "N/A"
    ur_str = f"{u_recall:.4f}" if u_recall is not None else "N/A"
    # 建议将以下代码统一缩进4个空格
    print(f"mAPK: {known_map:.4f}")
    print(f"WI: {wi_main * 100:.2f}")
    print(f"AOSE: {aose_total}")
    print(f"APU: {apu_str}")
    print(f"U-Recall: {ur_str}")
    print("================================\n")
# ===========================================================================
# 9. 命令行
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(description="两阶段开集检测（单文件）：RT-DETR(官方) 出框 + CLIP冻结文本嵌入+共享适配器 分类 + EDL/PUE 开集判别")
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="训练 CLIP + 冻结文本嵌入 + 共享适配器 + EDL 候选框分类器")
    t.add_argument("--json", required=True, help="COCO 格式标注（已知类，训练）")
    t.add_argument("--img-root", default=None, help="图片根目录")
    t.add_argument("--yw-ckpt", required=True, help="冻结的阶段1单类 RT-DETR(官方) 权重(.pth，含 model/ema)")
    t.add_argument("--rtdetr-config", required=True, help="官方 RT-DETR 的 YAML config（num_classes 改为 1）")
    t.add_argument("--rtdetr-root", default=None, help="官方 rtdetr_pytorch 根目录(含 src/)；也可用环境变量 RTDETR_ROOT")
    t.add_argument("--det-input-size", type=int, default=640, help="阶段1 RT-DETR 输入分辨率（官方默认 640）")
    t.add_argument("--out", default="out/box_classifier_clip_edl.pt")
    t.add_argument("--imgsz", type=int, default=224, help="CLIP 输入分辨率（与所选模型匹配，ViT-B-16=224）")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3, help="文本适配器 + 温度的学习率")
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--jitter", type=float, default=0.1, help="训练时框抖动")
    t.add_argument("--expand", type=float, default=0.1, help="裁剪外扩上下文")
    t.add_argument("--class-balanced", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # CLIP backbone
    t.add_argument("--clip-model", default="ViT-B-16", help="open_clip 模型名，如 ViT-B-16 / ViT-L-14")
    t.add_argument("--clip-pretrained", default="openai", help="open_clip 预训练标识（openai/laion2b_s34b_b88k...）或本地 .bin/.pt 权重路径")
    t.add_argument("--prompt-style", default="xray", choices=["xray", "plain"], help="类名编码用的 prompt 模板：xray=安检定制，plain=a photo of a {}")
    t.add_argument("--unfreeze-backbone", action="store_true", help="解冻 CLIP image encoder 一起微调（X光建议开）")
    t.add_argument("--backbone-lr", type=float, default=1e-5, help="解冻时 backbone 学习率")
    t.add_argument("--freeze-epochs", type=int, default=3,  help="解冻模式下，前 N 轮仍冻结 backbone，先让适配器/温度收敛")
    t.add_argument("--logit-scale-init", type=float, default=30.0, help="余弦logits温度初值（可学习）")

    # 文本适配器
    t.add_argument("--adapter-ratio", type=int, default=4, help="文本适配器瓶颈压缩比（hidden=D/ratio）")
    t.add_argument("--adapter-beta", type=float, default=0.5, help="文本适配器残差权重 out=norm(x+beta·MLP(x))")

    # EDL
    t.add_argument("--kl-weight", type=float, default=0.1, help="EDL KL 正则最终权重（弱正则）")
    t.add_argument("--kl-anneal", type=float, default=0.5, help="KL 权重线性渐入所占总 epoch 比例")

    # 几何稳定正则（吸收 ENC/Neural Collapse 思想，但不替换 EDL 监督）
    t.add_argument("--geom-weight", type=float, default=0.00,
                   help="适配后类原型的 ETF-style 几何正则权重；设 0 退回原始 EDL 损失")
    t.add_argument("--geom-anneal", type=float, default=0.5,
                   help="几何正则线性渐入所占总 epoch 比例")
    t.add_argument("--geom-include-pue", dest="geom_include_pue", action="store_true", default=True,
                   help="ETF 正则中纳入 PUE 伪未知方向，形成 K+1 开集原型，默认开")
    t.add_argument("--no-geom-include-pue", dest="geom_include_pue", action="store_false",
                   help="ETF 正则只约束已知类文本原型，不纳入 PUE 伪未知方向")
    t.add_argument("--geom-feature-weight", type=float, default=0.00,
                   help="可选 NC-style 特征到类别锚点收缩项权重，默认 0 表示关闭")

    # PUE（存进 ckpt 作为默认，测试时可覆盖）
    t.add_argument("--pue-alpha", type=float, default=1.0, help="伪未知嵌入 wU = w0' − α·w̄ 的 α（论文 Eq.2，本版 w0' 已过适配器）")

    # 阶段1出框 / 匹配
    t.add_argument("--prop-conf", type=float, default=0.05, help="出框objectness阈值，训练建议放低，多召回")
    t.add_argument("--prop-iou", type=float, default=0.7, help="出框NMS IoU")
    t.add_argument("--prop-max-det", type=int, default=300)
    t.add_argument("--pos-iou", type=float, default=0.5, help="候选框与GT匹配为正样本的IoU阈值")
    t.add_argument("--add-gt", dest="add_gt", action="store_true", default=True, help="训练集把GT框并入正样本，默认开")
    t.add_argument("--no-add-gt", dest="add_gt", action="store_false")

    q = sub.add_parser("predict", help="RT-DETR 出框 + EDL 分类 + EDL/PUE 开集判别")
    q.add_argument("--yw-ckpt", required=True, help="阶段1单类 RT-DETR(官方) 权重(.pth)")
    q.add_argument("--rtdetr-config", required=True, help="官方 RT-DETR 的 YAML config（num_classes=1）")
    q.add_argument("--rtdetr-root", default=None, help="官方 rtdetr_pytorch 根目录(含 src/)；或用环境变量 RTDETR_ROOT")
    q.add_argument("--det-input-size", type=int, default=640, help="阶段1 RT-DETR 输入分辨率（官方默认 640）")
    q.add_argument("--clf-ckpt", required=True)
    q.add_argument("--image", required=True)
    q.add_argument("--out", default="pred_clip_edl.jpg")
    q.add_argument("--conf", type=float, default=0.25, help="最终检测 objectness 阈值")
    q.add_argument("--iou", type=float, default=0.5)
    q.add_argument("--max-det", type=int, default=300)
    q.add_argument("--osr-method", default="edl_pue", choices=["edl_pue", "edl", "pue", "msp", "energy"], help="默认 edl_pue：EDL不确定度(NOOD) OR 伪未知嵌入(FOOD)")
    q.add_argument("--osr-thresh", type=float, default=None, help="不填则用训练时建议阈值(p5)")
    q.add_argument("--pue-alpha", type=float, default=None, help="覆盖 ckpt 内的 pue α（测试时可调）")
    q.add_argument("--save-txt", action="store_true")
    q.add_argument("--txt-out", default="pred_clip_edl.txt")
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ev = sub.add_parser("eval", help="批量评估测试集：已知类 mAP + 未知类 mAP + WI/AOSE/U-Recall")
    ev.add_argument("--yw-ckpt", required=True, help="阶段1单类 RT-DETR(官方) 权重(.pth)")
    ev.add_argument("--rtdetr-config", required=True, help="官方 RT-DETR 的 YAML config（num_classes=1）")
    ev.add_argument("--rtdetr-root", default=None, help="官方 rtdetr_pytorch 根目录(含 src/)；或用环境变量 RTDETR_ROOT")
    ev.add_argument("--det-input-size", type=int, default=640, help="阶段1 RT-DETR 输入分辨率（官方默认 640）")
    ev.add_argument("--clf-ckpt", required=True)
    ev.add_argument("--json", required=True, help="测试集 COCO 标注（含已知+未知类别）")
    ev.add_argument("--img-root", default=None, help="测试图片文件夹")
    ev.add_argument("--conf", type=float, default=0.05, help="检测 objectness 阈值（WI/AOSE 建议用低conf如0.05）")
    ev.add_argument("--iou", type=float, default=0.5, help="检测 NMS IoU")
    ev.add_argument("--max-det", type=int, default=300)
    ev.add_argument("--osr-method", default="edl_pue", choices=["edl_pue", "edl", "pue", "msp", "energy"])
    ev.add_argument("--osr-thresh", type=float, default=None)
    ev.add_argument("--pue-alpha", type=float, default=None)
    ev.add_argument("--map-iou", type=float, default=0.5, help="计算 mAP 用的 IoU 阈值")
    ev.add_argument("--coco-map", action="store_true", help="额外算 mAP@[.5:.95]")
    # 开集指标参数（遵循 opendet2：IoU=0.5、WI 在已知召回 0.8 处）
    ev.add_argument("--osr-iou", type=float, default=0.5, help="WI/AOSE/U-Recall 的 IoU 阈值（opendet2=0.5）")
    ev.add_argument("--wi-recall", type=float, default=0.8, help="WI 报告所在的已知类召回水平（opendet2=0.8）")
    ev.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p

def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        train_main(args)
    elif args.mode == "eval":
        eval_main(args)
    else:
        predict_main(args)

if __name__ == "__main__":
    main()

'''
# 可选：激活你自己的 Python 环境
# conda activate your-env

# 从仓库根目录运行。可按本机目录覆盖这些环境变量；默认均使用仓库内的相对目录。
export RTDETR_ROOT="${RTDETR_ROOT:-$(pwd)}"
export OXPID_DATA_ROOT="${OXPID_DATA_ROOT:-${RTDETR_ROOT}/dataSet/OXPID_M}"
export RTDETR_OUTPUT_ROOT="${RTDETR_OUTPUT_ROOT:-${RTDETR_ROOT}/output}"
export TWO_STAGE_OUTPUT_ROOT="${TWO_STAGE_OUTPUT_ROOT:-${RTDETR_ROOT}/outputTwoStage}"
export DETECTOR_CKPT="${DETECTOR_CKPT:-${RTDETR_OUTPUT_ROOT}/rtdetr_r50vd_6x_pidray/checkpoint.pth}"
export CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${TWO_STAGE_OUTPUT_ROOT}/box_classifier_clip_edl_rtdetr.pt}"

# ★ 阶段1改用 lyuwenyu 官方 RT-DETR(rtdetr_pytorch)。需要额外提供：
#   --rtdetr-config  官方 YAML config（把 num_classes 改成 1，单类 "object"）
#   --rtdetr-root    官方 rtdetr_pytorch 根目录(含 src/)，或设环境变量 RTDETR_ROOT
#   --yw-ckpt        官方 det_solver 保存的 .pth（含 model/ema，优先用 ema）

# ---------- 训练（默认：冻结 CLIP backbone，只训 文本适配器+温度） ----------
python twoStage-geo.py train \
  --json "${OXPID_DATA_ROOT}/train.json" \
  --img-root "${OXPID_DATA_ROOT}/train" \
  --yw-ckpt "${DETECTOR_CKPT}" \
  --rtdetr-config "${RTDETR_ROOT}/configs/rtdetr/rtdetr_r50vd_6x_pidray.yml" \
  --rtdetr-root "${RTDETR_ROOT}" \
  --out "${CLASSIFIER_CKPT}" \
  --clip-model ViT-B-16 --clip-pretrained openai --prompt-style xray \
  --epochs 25 --batch-size 64 --imgsz 224 --lr 1e-3 \
  --adapter-ratio 4 --adapter-beta 0.5 --kl-weight 0.1 \
  --prop-conf 0.05 --prop-iou 0.7 --prop-max-det 300 --pos-iou 0.5 --expand 0.1

# ---------- 训练（解冻 CLIP backbone 微调：X光域差距大，推荐这条） ----------
python twoStage-geo.py train \
  --json "${OXPID_DATA_ROOT}/train.json" \
  --img-root "${OXPID_DATA_ROOT}/train" \
  --yw-ckpt "${DETECTOR_CKPT}" \
  --rtdetr-config "${RTDETR_ROOT}/configs/rtdetr/rtdetr_r50vd_6x_pidray.yml" \
  --rtdetr-root "${RTDETR_ROOT}" \
  --out "${CLASSIFIER_CKPT}" \
  --clip-model ViT-B-16 --clip-pretrained openai --prompt-style xray \
  --unfreeze-backbone --backbone-lr 1e-5 --freeze-epochs 3 --lr 5e-4 \
  --epochs 50 --batch-size 64 --adapter-ratio 4 --adapter-beta 0.5 --kl-weight 0.1 \
  --geom-weight 0.01 --geom-anneal 0.5 --geom-feature-weight 0.0

# ---------- 单图预测 ----------
python twoStage-geo.py predict \
  --yw-ckpt "${DETECTOR_CKPT}" \
  --rtdetr-config "${RTDETR_ROOT}/configs/rtdetr/rtdetr_r50vd_6x_pidray.yml" \
  --rtdetr-root "${RTDETR_ROOT}" \
  --clf-ckpt "${CLASSIFIER_CKPT}" \
  --image "${OXPID_DATA_ROOT}/val/01273.jpg" \
  --out "${RTDETR_ROOT}/out/pred_01273.jpg" \
  --conf 0.25 --iou 0.5 --osr-method edl_pue \
  --save-txt --txt-out "${RTDETR_ROOT}/out/pred_01273.txt"
  
  

# ---------- 批量评估（mAPK + APU + WI + AOSE + U-Recall） ----------
# 注意：WI/AOSE 对 conf 敏感，固定用低 conf(0.05)；WI 默认在已知召回 0.8 处报告。
python twoStage-geo.py eval \
  --yw-ckpt "${DETECTOR_CKPT}" \
  --rtdetr-config "${RTDETR_ROOT}/configs/rtdetr/rtdetr_r50vd_6x_pidray.yml" \
  --rtdetr-root "${RTDETR_ROOT}" \
  --clf-ckpt "${CLASSIFIER_CKPT}" \
  --json "${OXPID_DATA_ROOT}/test.json" \
  --img-root "${OXPID_DATA_ROOT}/val" \
  --conf 0.05 --iou 0.5 --osr-method edl_pue \
  --map-iou 0.5 --coco-map --osr-iou 0.5 --wi-recall 0.8
'''
