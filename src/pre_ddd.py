import os
import glob
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm

import sys

from utils import accumulate_to_rgb
sys.path.append("dinov2")


import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from dinov2.models.vision_transformer import vit_small
from dataset import PadToMinSide, PairedProcessor, Normalize, RandomCrop, RandomHorizontalFlip, RandomSwapEventRedBlue, ResizeKeepRatio, ToTensor, CenterCrop

# --- 1. 配置模块 ---
# !! 重要: 请在此处修改为包含所有 dir* 文件夹的顶级'data'目录的绝对路径 !!
TOP_LEVEL_DATA_DIR = '/data/storage/jianwen/ddd17_seg/data'

# --- 2. 数据加载模块 ---

def load_data(directory_path):
    """
    为单个目录加载所有必需的数据文件。
    """
    try:
        events_xyp_file = os.path.join(directory_path, 'events.dat.xyp')
        events_t_file = os.path.join(directory_path, 'events.dat.t')
        index_file = os.path.join(directory_path, 'index/index_250ms.npy')

        if not all(map(os.path.exists, [events_xyp_file, events_t_file, index_file])):
            raise FileNotFoundError("一个或多个关键数据文件（events.dat.* 或 index_50ms.npy）未找到。")

        num_events = int(os.path.getsize(events_t_file) / 8)
        xyp_events = np.memmap(events_xyp_file, dtype="int16", mode="r", shape=(num_events, 3))
        index_data = np.load(index_file)
        
        return xyp_events, index_data

    except FileNotFoundError as e:
        print(f"错误: {e}")
        return None, None # 返回 None 表示加载失败

def _frame_id_from_name(path_or_name):
    """
    提取文件名后缀中的帧编号（例如 img_00000012.png -> 12）。
    返回 None 表示无法解析。
    """
    name = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(name)
    if "_" not in stem:
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None

def get_semantic_frame_ids(dir_path):
    """
    读取 segmentation_masks 目录，返回拥有语义标签的帧编号集合（从 1 开始）。
    """
    seg_dir = os.path.join(dir_path, 'segmentation_masks')
    if not os.path.isdir(seg_dir):
        return set()
    frame_ids = set()
    for mask_path in glob.glob(os.path.join(seg_dir, "segmentation_*.png")):
        frame_id = _frame_id_from_name(mask_path)
        if frame_id is not None:
            frame_ids.add(frame_id)
    return frame_ids
    
# --- 3. 事件聚合模块 ---

def accumulate_to_rgb(x, y, p, shape, pct=95):
    """
    将事件(x, y, p)累积成一张白底、红/蓝色的RGB图像。
    """
    H, W = shape
    
    pos = np.zeros((H, W), dtype=np.float32)
    neg = np.zeros((H, W), dtype=np.float32)
    p = p.astype(bool)
    
    if x.size:
        np.add.at(pos, (y[p], x[p]), 1)
        np.add.at(neg, (y[~p], x[~p]), 1)

    def norm_with_percentile(a):
        if a.max() == 0: return a
        thr = np.percentile(a[a > 0], pct) if np.any(a > 0) else 1.0
        if thr <= 0: thr = float(a.max())
        return np.clip(a, 0, thr) / thr

    pos_n = norm_with_percentile(pos)
    neg_n = norm_with_percentile(neg)

    dominate_pos = pos_n >= neg_n
    inten_pos = pos_n * dominate_pos
    inten_neg = neg_n * (~dominate_pos)

    R, G, B = np.ones((H, W)), np.ones((H, W)), np.ones((H, W))
    G -= inten_pos; B -= inten_pos
    R -= inten_neg; G -= inten_neg
    
    img = np.stack([np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)], axis=-1)
    return (img * 255).astype(np.uint8)

# --- 4. 单帧处理模块 (线程工作函数) ---

def process_and_save_frame(args):
    """
    处理单个帧：提取事件、生成图像并保存。
    """
    frame_idx, xyp_events, index_data, output_dir, image_shape = args
    _, event_idx_end, event_idx_start = index_data[frame_idx]
    event_idx_start, event_idx_end = int(event_idx_start), int(event_idx_end)

    if event_idx_start >= event_idx_end:
        event_image = np.full((image_shape[0], image_shape[1], 3), 255, dtype=np.uint8)
    else:
        events_for_frame = xyp_events[event_idx_start:event_idx_end]
        x_coords, y_coords, p_values = events_for_frame.T
        event_image = accumulate_to_rgb(x_coords, y_coords, p_values, shape=image_shape)

    # if event_image.mean() > 250:
    #     return False
    # else:
    output_filename = f"img_{frame_idx + 1:08d}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    Image.fromarray(event_image).save(output_path)
    return True

# --- 5. 单目录处理模块 ---

def process_directory(dir_path, semantic_only=False):
    """
    协调处理单个dir*目录的完整流程。
    """
    dir_name = os.path.basename(dir_path)
    print(f"\n{'='*20} 开始处理目录: {dir_name} {'='*20}")
    
    output_dir = os.path.join(dir_path, 'imgs')
    if not os.path.isdir(output_dir):
        print(f"跳过: 在'{dir_name}'中未找到输出目录'imgs'。")
        return

    print(f"数据加载中...")
    xyp_events, index_data = load_data(dir_path)
    
    # 如果加载失败，则跳过此目录
    if xyp_events is None or index_data is None:
        print(f"跳过: 目录'{dir_name}'数据加载失败。")
        return

    labeled_frame_ids = None
    if semantic_only:
        labeled_frame_ids = sorted(get_semantic_frame_ids(dir_path))
        if not labeled_frame_ids:
            print(f"跳过: semantic_only=True 但在 '{dir_name}' 中未找到语义标签。")
            return

    print(f"数据加载成功。开始批量生成 {len(index_data)} 张图像...")
    image_shape = (260, 346)
    num_frames = len(index_data)

    if semantic_only:
        frame_indices = [
            frame_id - 1
            for frame_id in labeled_frame_ids
            if 1 <= frame_id <= num_frames
        ]
        frame_indices = sorted(set(idx for idx in frame_indices if idx >= 0))
    else:
        frame_indices = list(range(num_frames))

    if not frame_indices:
        print(f"跳过: semantic_only=True 但没有和索引文件匹配的语义帧。")
        return

    tasks = [(i, xyp_events, index_data, output_dir, image_shape) for i in frame_indices]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_and_save_frame, tasks), total=len(tasks), desc=f"处理 {dir_name}"))

    print(f"--- 目录 {dir_name} 处理完成！---")

# --- 6. 主执行模块 ---

def main(semantic_only=False):
    """
    主函数，查找所有dir*目录并循环处理它们。
    """
    # 使用glob查找所有匹配的目录
    all_dirs = sorted(glob.glob(os.path.join(TOP_LEVEL_DATA_DIR, 'dir*')))

    if not all_dirs:
        print(f"错误: 在'{TOP_LEVEL_DATA_DIR}'中没有找到任何以'dir'开头的子目录。")
        print("请检查 TOP_LEVEL_DATA_DIR 路径是否正确。")
        return

    print(f"成功找到 {len(all_dirs)} 个目录进行处理。")
    
    # 循环处理每一个找到的目录
    for dir_path in all_dirs:
        process_directory(dir_path, semantic_only=semantic_only)

    print(f"\n{'='*20} 所有目录均已处理完毕！ {'='*20}")

import os
import glob
import torch
from PIL import Image
from tqdm import tqdm

def extract_features(
        top_level_data_dir: str,
        image_encoder_ckpt: str = "/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth",
        event_encoder_ckpt: str = "/data/storage/jianwen/cache/ckpt_matters/gra_mixture_4x.pt",
        H: int = 224, W: int = 224,
        train_nums = [0, 3, 4, 5, 6, 7],
        valid_nums = [1],
        semantic_only: bool = False,
    ):
    """
    为数据集 B（每个 dir*/imgs/img_XXXXXXXX.jpg）抽取特征。
    - 根据 dir 后数字判断训练集或验证集。
    - semantic_only=True 时，仅处理存在 segmentation_masks/segmentation_*.png 的帧。
    """
    ME, SE = [0.9635178446769714, 0.9372559189796448, 0.9606495499610901], [0.10666034370660782, 0.14702685177326202, 0.10695996135473251]
    MI, SI = [0.37694597244262695, 0.37694597244262695, 0.37694597244262695], [0.2800583243370056, 0.2800583243370056, 0.2800583243370056]
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    type_flag = "EI"

    # 构建 processor
    train_processor = PairedProcessor([
        ToTensor(type=type_flag),
        Normalize(ME, SE, MI, SI, type=type_flag),
        RandomSwapEventRedBlue(type=type_flag),
        RandomHorizontalFlip(p=0.5),    
        PadToMinSide(target=(H, W), pad_x1=0, pad_x2=0),
        RandomCrop(crop_size=(H, W), type=type_flag),
    ])
    valid_processor = PairedProcessor([
        ToTensor(type=type_flag),
        Normalize(ME, SE, MI, SI, type=type_flag),
        PadToMinSide(target=(H, W), pad_x1=0, pad_x2=0),
        CenterCrop((H, W)),
    ])

    # 初始化 encoder
    image_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(DEVICE)
    event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(DEVICE)
    print("加载 image encoder 权重：", image_encoder_ckpt)
    image_encoder.load_state_dict(torch.load(image_encoder_ckpt, weights_only=True), strict=True)
    print("加载 event encoder 权重：", event_encoder_ckpt)
    ev_ckpt = torch.load(event_encoder_ckpt, weights_only=True)
    if isinstance(ev_ckpt, dict) and "event_encoder" in ev_ckpt:
        event_encoder.load_state_dict(ev_ckpt["event_encoder"], strict=True)
    else:
        event_encoder.load_state_dict(ev_ckpt, strict=True)
    image_encoder.eval()
    event_encoder.eval()

    all_dirs = sorted(glob.glob(os.path.join(top_level_data_dir, 'dir*')))
    if not all_dirs:
        raise RuntimeError(f"在 {top_level_data_dir} 中未找到任何 dir* 目录，请检查路径。")

    for dir_path in all_dirs:
        dir_name = os.path.basename(dir_path)
        try:
            dir_num = int(dir_name.replace("dir", ""))
        except ValueError:
            print(f"警告: {dir_name} 无法解析数字，跳过。")
            continue

        # 判断训练/验证集
        if dir_num in train_nums:
            processor = train_processor
            split = "train"
        elif dir_num in valid_nums:
            processor = valid_processor
            split = "valid"
        else:
            print(f"警告: {dir_name} 不在 train_nums 或 valid_nums 中，将按验证集处理。")
            processor = valid_processor
            split = "valid"

        imgs_dir = os.path.join(dir_path, 'imgs')
        if not os.path.isdir(imgs_dir):
            print(f"跳过 {dir_name}：未找到 imgs 子目录。")
            continue

        event_feat_dir = os.path.join(dir_path, "event_features")
        image_feat_dir = os.path.join(dir_path, "image_features")
        os.makedirs(event_feat_dir, exist_ok=True)
        os.makedirs(image_feat_dir, exist_ok=True)

        image_files = [f for f in sorted(os.listdir(imgs_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"跳过 {dir_name}：imgs 目录中没有图片文件。")
            continue

        labeled_frame_ids = None
        if semantic_only:
            labeled_frame_ids = get_semantic_frame_ids(dir_path)
            if not labeled_frame_ids:
                print(f"跳过 {dir_name}：semantic_only=True 但未找到语义标签。")
                continue

            filtered_files = []
            seen_ids = set()
            for fname in image_files:
                frame_id = _frame_id_from_name(fname)
                if frame_id is None:
                    continue
                if frame_id not in labeled_frame_ids or frame_id in seen_ids:
                    continue
                filtered_files.append(fname)
                seen_ids.add(frame_id)

            if not filtered_files:
                print(f"跳过 {dir_name}：semantic_only=True 但 imgs 目录中没有与标签匹配的帧。")
                continue
            image_files = filtered_files

        print(f"\n处理 {dir_name} （帧数={len(image_files)}，split={split}）")

        for i, fname in enumerate(tqdm(image_files, desc=f"features {dir_name}")):
            img_path = os.path.join(imgs_dir, fname)
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"无法打开图片 {img_path}: {e}. 跳过。")
                continue

            try:
                event_tensor, image_tensor = processor(pil_img, pil_img)
            except Exception as e:
                print(f"处理图片 {img_path} 失败（transform）：{e}. 跳过。")
                continue

            event_tensor = event_tensor.unsqueeze(0).to(DEVICE)
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                ev_feat = event_encoder.forward_features(event_tensor)["x_norm_patchtokens"].squeeze(0)
                im_feat = image_encoder.forward_features(image_tensor)["x_norm_patchtokens"].squeeze(0)

            torch.save(ev_feat.cpu(), os.path.join(event_feat_dir, f"{i:08d}.pt"))
            torch.save(im_feat.cpu(), os.path.join(image_feat_dir, f"{i:08d}.pt"))

        print(f"目录 {dir_name} 特征提取完成！")

    print("所有目录处理完毕。")

if __name__ == "__main__":
    semantic_only = True
    main(semantic_only=semantic_only)
    # extract_features(f"/data/storage/jianwen/ddd17_seg/data", semantic_only=semantic_only)
