import argparse
import glob
import math
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchinfo import summary
from tqdm.auto import tqdm

from config import SegConfig
from seg import SEG

# DSEC 11-classes 训练标签映射 (trainId → 颜色)，确保相同 trainId 共享颜色
DSEC_TRAIN_ID_COLORS = {
    0: (0, 0, 0),        # sky rendered as black
    1: (70, 70, 70),     # building
    2: (190, 153, 153),  # fence
    3: (220, 20, 60),    # person / rider
    4: (153, 153, 153),  # pole
    5: (128, 64, 128),   # road
    6: (244, 35, 232),   # sidewalk
    7: (107, 142, 35),   # vegetation / terrain
    8: (0, 0, 142),      # vehicle
    9: (102, 102, 156),  # wall
    10: (220, 220, 0),   # traffic light / sign (yellow)
    11: (0, 0, 0),       # ignore / padding
}


def build_palette_from_mapping(color_mapping: dict):
    """
    从 color_mapping (RGB → class_id) 构建一个 [num_classes x 3] 的颜色表。
    例如 { (128,64,128):5, (244,35,232):6 } → colors[5] = [128,64,128]
    """
    num_classes = max(color_mapping.values()) + 1
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for rgb, cls in color_mapping.items():
        if cls < num_classes:
            palette[cls] = np.array(rgb, dtype=np.uint8)
    return palette


def build_trainid_palette(num_classes: int,
                          trainid_colors: dict,
                          default_color=(0, 0, 0)):
    """按照 trainId 共享颜色构建调色板。"""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    default_rgb = np.array(default_color, dtype=np.uint8)
    for cls_idx in range(num_classes):
        palette[cls_idx] = np.array(trainid_colors.get(cls_idx, default_rgb), dtype=np.uint8)
    return palette

def tensor_to_color(mask: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """把预测/标签 tensor (H, W) 转为彩色 mask。"""
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    return palette[np.clip(mask_np, 0, len(palette) - 1)]


def render_event_tensor(event: torch.Tensor) -> torch.Tensor:
    """复制 SEG._render_event_tensor 方便单独调用。"""
    if event.ndim == 3:
        event = event.unsqueeze(0)
    if event.ndim != 4:
        raise ValueError(f"Expected tensor with shape [B, C, H, W], got {event.shape}")
    _, c, _, _ = event.shape
    if c >= 3:
        vis = event[:, :3]
    else:
        repeat = int(math.ceil(3 / c))
        vis = event.repeat(1, repeat, 1, 1)[:, :3]
    vis = vis - vis.amin(dim=(2, 3), keepdim=True)
    denom = vis.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return vis / denom


def recover_input_image(image_tensor: torch.Tensor, cfg: SegConfig) -> np.ndarray:
    """把经过 transform 的张量恢复成 0-255 RGB 便于可视化。"""
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    img = image_tensor.detach().cpu()
    if cfg.type == "EL":
        if img.shape[1] == len(cfg.ME):
            mean = torch.tensor(cfg.ME, dtype=img.dtype).view(1, -1, 1, 1)
            std = torch.tensor(cfg.SE, dtype=img.dtype).view(1, -1, 1, 1)
            vis = img * std + mean
        else:
            vis = render_event_tensor(img)
    else:
        mean = torch.tensor(cfg.MI, dtype=img.dtype).view(1, -1, 1, 1)
        std = torch.tensor(cfg.SI, dtype=img.dtype).view(1, -1, 1, 1)
        vis = img * std + mean
    vis = vis.clamp(0.0, 1.0)
    vis = vis[0].permute(1, 2, 0).numpy()
    return (vis * 255.0).astype(np.uint8)


def blend_prediction(image_rgb: np.ndarray, pred_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """把原图与预测 mask 进行透明度叠加。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = image_rgb.astype(np.float32) * (1.0 - alpha) + pred_rgb.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def colorize_confidence(conf_tensor: torch.Tensor) -> np.ndarray:
    """把置信度张量映射为彩色热力图。"""
    conf_np = conf_tensor.detach().cpu().numpy()
    conf_norm = np.clip(conf_np, 0.0, 1.0)
    conf_u8 = (conf_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(conf_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def load_checkpoint(model: SEG, checkpoint_path: str):
    """严格按照训练时的模块划分加载权重。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "event_encoder" in ckpt:
        model.event_encoder.load_state_dict(ckpt["event_encoder"], strict=True)
    if "transformer" in ckpt and not isinstance(model.transformer, nn.Identity):
        model.transformer.load_state_dict(ckpt["transformer"], strict=True)
    if model.is_ecddp:
        decoder_state = ckpt.get("decoder", {})
        if model.ecddp_head is not None and "main" in decoder_state:
            model.ecddp_head.load_state_dict(decoder_state["main"], strict=True)
        if model.ecddp_aux_head is not None and "aux" in decoder_state:
            model.ecddp_aux_head.load_state_dict(decoder_state["aux"], strict=True)
    else:
        if model.decoder is not None and "decoder" in ckpt:
            model.decoder.load_state_dict(ckpt["decoder"], strict=True)
        if getattr(model, "pyramid_head", None) is not None and "pyramid_head" in ckpt:
            model.pyramid_head.load_state_dict(ckpt["pyramid_head"], strict=True)
    return ckpt


def run_model(model: SEG, img: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """返回 (B, C, H, W) logits。"""
    with torch.no_grad():
        if use_tta:
            return model._predict_with_tta(img)
        preds, _ = model(img)
        return preds[0] if isinstance(preds, tuple) else preds


def prepare_palette(cfg: SegConfig) -> np.ndarray:
    num_palette_classes = max(cfg.ignore_index, cfg.C - 1) + 1
    if cfg.dataset == "dsec":
        return build_trainid_palette(num_palette_classes, DSEC_TRAIN_ID_COLORS)
    return build_palette_from_mapping(cfg.color_mapping)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation inference & visualization helper.")
    parser.add_argument("--checkpoint", type=str, default="/data/storage/jianwen/cache/ckpts/2025-11-14-02:04_seg/epoch8000_0.8904.pt", help="训练好的权重路径 (.pt)")
    parser.add_argument("--image-glob", type=str, default="/data/storage/jianwen/DSEC/test_images/zurich_city_14_c/images/left/eventImage/*.png", help="输入图片 glob，比如 '/path/*.png'")
    parser.add_argument("--label-glob", type=str, default="/data/storage/jianwen/DSEC/test_semantic_segmentation/test/zurich_city_14_c/11classes/*.png", help="标签 glob，需与图片数量一致")
    parser.add_argument("--frame-dir", type=str, default="14c_mem", help="保存逐帧结果的目录")
    parser.add_argument("--output-video", type=str, default="14c_mem.mp4", help="输出视频文件名")
    parser.add_argument("--fps", type=int, default=30, help="输出视频帧率")
    parser.add_argument("--no-video", action="store_true", help="只保存图片，不写视频")
    parser.add_argument("--force-tta", action="store_true", help="强制启用 TTA（忽略配置文件）")
    parser.add_argument("--disable-tta", action="store_true", help="禁用 TTA")
    parser.add_argument("--alpha", type=float, default=0.6, help="overlay 中预测颜色的透明度")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 张图片，方便调试")
    parser.add_argument("--save-confidence", action="store_true", help="额外保存/拼接置信度热力图")
    parser.add_argument("--device", type=str, default="cuda:2", help="覆盖 SegConfig.device (例如 cuda:1)")
    parser.add_argument("--no-summary", action="store_true", help="跳过 torchinfo summary，节省时间")
    return parser.parse_args()


def main():
    args = parse_args()
    image_names = sorted(glob.glob(args.image_glob))
    label_names = sorted(glob.glob(args.label_glob))
    if not image_names:
        raise FileNotFoundError(f"未找到匹配 {args.image_glob} 的图片")
    if not label_names:
        raise FileNotFoundError(f"未找到匹配 {args.label_glob} 的标签")
    if len(image_names) != len(label_names):
        raise ValueError(f"图片数量({len(image_names)}) 与标签数量({len(label_names)}) 不一致")
    if args.limit is not None:
        image_names = image_names[: args.limit]
        label_names = label_names[: args.limit]
    print(f"将对 {len(image_names)} 张图片进行推理与可视化。")

    cfg = SegConfig()
    if args.device is not None:
        cfg.device = args.device
    os.makedirs(args.frame_dir, exist_ok=True)

    model = SEG(cfg)
    load_checkpoint(model, args.checkpoint)
    model.to(cfg.device).eval()
    if not args.no_summary:
        summary(model)

    use_tta = getattr(model, "tta_enable", False)
    if args.force_tta:
        use_tta = True
    if args.disable_tta:
        use_tta = False

    palette = prepare_palette(cfg)
    transform = cfg.valid_preprocessors

    sample_img = Image.open(image_names[0]).convert("RGB")
    sample_lbl = Image.open(label_names[0]).convert("L")
    _, sample_lbl_tensor = transform(sample_img, sample_lbl)
    h, w = sample_lbl_tensor.shape
    include_conf = args.save_confidence
    num_columns = 3 + 1 + int(include_conf)  # input, pred, overlay, label, (optional conf)
    frame_h, frame_w = h, w * num_columns

    video_writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (frame_w, frame_h))
        if not video_writer.isOpened():
            raise RuntimeError(f"无法打开视频写入器: {args.output_video}")
        print(f"开始生成视频 ({args.output_video})，帧率 {args.fps} ...")

    frame_dir = Path(args.frame_dir)
    print(f"帧图片将保存至：{frame_dir.resolve()}")

    for idx in tqdm(range(len(image_names)), desc="Infer", unit="img"):
        img_path = image_names[idx]
        lbl_path = label_names[idx]
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        img_tensor, lbl_tensor = transform(img, lbl)
        img_for_model = img_tensor.unsqueeze(0).to(cfg.device)

        logits = run_model(model, img_for_model, use_tta)
        pred = torch.argmax(logits, dim=1).squeeze(0)
        pred_color = tensor_to_color(pred, palette)

        img_rgb = recover_input_image(img_tensor, cfg)
        overlay = blend_prediction(img_rgb, pred_color, args.alpha)
        lbl_color = tensor_to_color(lbl_tensor, palette)

        conf_color = None
        if include_conf:
            probs = torch.softmax(logits, dim=1)
            conf_map = probs.max(dim=1).values.squeeze(0)
            conf_color = colorize_confidence(conf_map)

        panels: List[np.ndarray] = [img_rgb, pred_color, overlay, lbl_color]
        if include_conf and conf_color is not None:
            panels.append(conf_color)
        concat = np.concatenate(panels, axis=1)

        if video_writer is not None:
            video_writer.write(cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

        frame_basename = frame_dir / f"frame_{idx:05d}"
        Image.fromarray(img_rgb).save(frame_basename.with_name(f"{frame_basename.name}_img.png"))
        Image.fromarray(pred_color).save(frame_basename.with_name(f"{frame_basename.name}_pred.png"))
        Image.fromarray(overlay).save(frame_basename.with_name(f"{frame_basename.name}_overlay.png"))
        Image.fromarray(lbl_color).save(frame_basename.with_name(f"{frame_basename.name}_lbl.png"))
        if include_conf and conf_color is not None:
            Image.fromarray(conf_color).save(frame_basename.with_name(f"{frame_basename.name}_conf.png"))

    if video_writer is not None:
        video_writer.release()
        print(f"✅ 视频保存完成：{args.output_video}")
    print(f"✅ 共保存 {len(image_names)} 张帧图片到：{frame_dir}")


if __name__ == "__main__":
    main()
