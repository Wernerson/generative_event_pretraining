from __future__ import annotations
import argparse
import os
import sys
import shutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# local imports for encoders and paired transforms
sys.path.append("dinov2")
from dataset import PadToMinSide, RandomResizedCrop
from dinov2.models.vision_transformer import vit_small
from dataset import (
    PairedProcessor,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomSwapEventRedBlue,
    CenterCrop,
)

from utils import accumulate_to_rgb


def _extract_xypt(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robustly extract x,y,t,p from a structured event array.
    Accepts typical field names (case-insensitive): 'x','y','t','p'.
    Falls back to first four fields in order if names are ambiguous.
    """
    names = list(data.dtype.names or [])
    if not names:
        raise ValueError("event array has no named fields")
    name_map = {n.lower(): n for n in names}
    try:
        x = data[name_map.get('x', names[0])]
        y = data[name_map.get('y', names[1 if len(names) > 1 else 0])]
        # prefer explicit 't' and 'p' fields if present
        t = data[name_map.get('t', names[2 if len(names) > 2 else 0])]
        p = data[name_map.get('p', names[3 if len(names) > 3 else 0])]
    except Exception:
        # ultimate fallback: positional order
        x = data[names[0]]
        y = data[names[1]] if len(names) > 1 else np.zeros_like(x)
        t = data[names[2]] if len(names) > 2 else np.zeros_like(x)
        p = data[names[3]] if len(names) > 3 else np.zeros_like(x)
    return x, y, t, p

def list_event_files(root: str) -> List[str]:
    # 验证: extracted_val/class/*.npz
    out = []
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith('.npz'):
                out.append(os.path.join(cls_dir, fname))
    return out

def find_image_for_event(event_path: str, img_train_root: str, img_val_root: str) -> Optional[str]:
    """尝试匹配事件文件对应原始图像以获取尺寸.
    规则尝试顺序:
      1) 训练: 直接 img_train_root/<basename>.JPEG
      2) 训练: img_train_root/<class>/<basename>.JPEG (若 train 仍保留类目录)
      3) 验证: img_val_root/<class>/<basename>.JPEG
      4) 验证特殊: 若 basename 形如 nXXXXXXXX_xxx, 尝试提取末尾数字构造 ILSVRC2012_val_000xxxxx.JPEG
    若仍失败返回 None.
    """
    base = os.path.splitext(os.path.basename(event_path))[0]
    # 类 ID 优先取父目录名(兼容 val: ILSVRC2012_val_XXXXXX.npz), 回退为旧规则
    parent = os.path.basename(os.path.dirname(event_path))
    if parent.startswith('n') and parent[1:].isdigit():
        cls = parent
    else:
        # 旧规则: 取第一个 '_' 前部分 (训练集文件名往往以类ID开头)
        cls = base.split('_')[0]
    # 1)
    cand = os.path.join(img_train_root, base + '.JPEG')
    if os.path.isfile(cand):
        return cand
    # 支持 .JPEG/.jpg
    cand_jpg = cand[:-5] + '.jpg'
    if os.path.isfile(cand_jpg):
        return cand_jpg
    # 2)
    cand2 = os.path.join(img_train_root, cls, base + '.JPEG')
    if os.path.isfile(cand2):
        return cand2
    cand2_jpg = cand2[:-5] + '.jpg'
    if os.path.isfile(cand2_jpg):
        return cand2_jpg
    # 3)
    cand3 = os.path.join(img_val_root, cls, base + '.JPEG')
    if os.path.isfile(cand3):
        return cand3
    cand3_jpg = cand3[:-5] + '.jpg'
    if os.path.isfile(cand3_jpg):
        return cand3_jpg
    # 4) 构造 ILSVRC2012_val_ 形式 (猜测) - 提取最后 '_' 后数字部分
    if '_' in base:
        tail = base.split('_')[-1]
        if tail.isdigit():
            num = int(tail)
            val_name = f"ILSVRC2012_val_{num:08d}.JPEG"
            cand4 = os.path.join(img_val_root, cls, val_name)
            if os.path.isfile(cand4):
                return cand4
    return None

def process_one(event_path: str, img_train_root: str, img_val_root: str, percentile: float = 99.0, overwrite: bool = False, link_image: bool = False) -> Tuple[str, bool, str]:
    base = os.path.splitext(os.path.basename(event_path))[0]
    out_dir = os.path.dirname(event_path)
    out_path = os.path.join(out_dir, base + '.png')
    old_npy = os.path.join(out_dir, base + '.npy')
    if os.path.isfile(old_npy):
        try:
            os.remove(old_npy)
        except OSError:
            pass
    if (not overwrite) and os.path.isfile(out_path):
        return (event_path, True, 'skip_exists')
    img_path = find_image_for_event(event_path, img_train_root, img_val_root)
    try:
        with np.load(event_path) as npz:
            if 'event_data' in npz:
                data = npz['event_data']
            else:
                data = list(npz.values())[0]
        x, y, t, p = _extract_xypt(data)
        # Determine output shape: prefer original image size when available,
        # otherwise infer from event coordinates.
        if img_path and os.path.isfile(img_path):
            with Image.open(img_path) as im:
                shape = im.size[1], im.size[0]  # (H,W)
        else:
            raise ValueError("未找到对应原图")
        # Correct argument order: accumulate_to_rgb(x, y, p, shape, pct)
        rgb = accumulate_to_rgb(x, y, p, (480, 640), pct=percentile)
        Image.fromarray(rgb).resize(im.size).save(out_path)

        if img_path and os.path.isfile(img_path):
            img_ext = os.path.splitext(img_path)[1]
            local_img_path = os.path.join(os.path.dirname(event_path), base + img_ext)
            if not os.path.exists(local_img_path):
                if link_image:
                    try:
                        os.symlink(img_path, local_img_path)
                    except OSError:
                        shutil.copy2(img_path, local_img_path)
                else:
                    shutil.copy2(img_path, local_img_path)
        return (event_path, True, 'ok')
    except Exception as e:  # noqa
        return (event_path, False, f'err:{e.__class__.__name__}')

def run_batch(args):
    if getattr(args, 'val_only', True):
        train_events = []
        val_events = list_event_files(args.valid_root) if args.valid_root and os.path.isdir(args.valid_root) else []
        all_events = val_events
    else:
        train_events = list_event_files(args.train_root) if os.path.isdir(args.train_root) else []
        val_events = list_event_files(args.valid_root) if args.valid_root and os.path.isdir(args.valid_root) else []
        all_events = train_events + val_events

    if not all_events:
        print('未找到事件文件, 请检查路径.', file=sys.stderr)
        return
    print(f'发现事件文件数量: train={len(train_events)}, val={len(val_events)}, total={len(all_events)}')
    worker = partial(process_one, img_train_root=args.img_train_root, img_val_root=args.img_val_root,
                     percentile=args.percentile, overwrite=args.overwrite, link_image=args.link)
    total = len(all_events)
    results = []
    ok = skip = 0
    miss = []
    err = []
    def show_progress(i):
        pct = i * 100.0 / total
        msg = f"\r进度: {i}/{total} ({pct:5.1f}%) ok:{ok} skip:{skip} miss:{len(miss)} err:{len(err)}"
        print(msg, end='', flush=True)
    if args.num_workers <= 1:
        for i, p in enumerate(all_events, 1):
            res = worker(p)
            results.append(res)
            _, success, status = res
            if success and status == 'ok':
                ok += 1
            elif success and status == 'skip_exists':
                skip += 1
            elif (not success) and status == 'image_not_found':
                if len(miss) < 20:
                    miss.append(p)
            elif (not success) and status.startswith('err:'):
                if len(err) < 20:
                    err.append(p)
            show_progress(i)
    else:
        with mp.Pool(args.num_workers) as pool:
            for i, res in enumerate(pool.imap(worker, all_events, chunksize=8), 1):
                results.append(res)
                pth, success, status = res
                if success and status == 'ok':
                    ok += 1
                elif success and status == 'skip_exists':
                    skip += 1
                elif (not success) and status == 'image_not_found':
                    if len(miss) < 20:
                        miss.append(pth)
                elif (not success) and status.startswith('err:'):
                    if len(err) < 20:
                        err.append(pth)
                show_progress(i)
    print()  # 换行
    print(f'完成: 成功 {ok}, 跳过 {skip}, 缺失图像 {len(miss)}, 错误 {len(err)}')
    if miss:
        print('缺失图像示例(最多20):')
        for m in miss:
            print('  ', m)
    if err:
        print('错误示例(最多20):')
        for e in err:
            print('  ', e)

@torch.no_grad()
def process_tokens(args):
    # NIMA dataset RGB stats (0..1)
    EVT_MEAN = [0.9673029496145361, 0.929740832760733, 0.9624378831461544]
    EVT_STD  = [0.12036860792540319, 0.1674319634709885, 0.12649585555644863]
    IMG_MEAN = [0.4802686970882532, 0.45750728990737405, 0.40818174243273203]
    IMG_STD  = [0.28073999251715753, 0.2736791173289334, 0.28782502739532]

    def list_event_image_pair(root: str) -> List[str]:
        events, images = [], []
        if not os.path.isdir(root):
            return events
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.png'):
                    events.append(os.path.join(cls_dir, fname))
                elif fname.endswith('.JPEG'):
                    images.append(os.path.join(cls_dir, fname))
        return events, images

    envs_train, imgs_train = list_event_image_pair(args.train_root)
    envs_valid, imgs_valid = list_event_image_pair(args.valid_root)
    envs = envs_train + envs_valid
    imgs = imgs_train + imgs_valid
    if not envs:
        print('未找到事件PNG用于特征提取.')
        return

    # announce save locations (example from first sample)
    example_dir = os.path.dirname(envs[0])
    print(f"将为 {len(envs)} 个样本提取 tokens")
    print("保存位置示例：")
    print(f"  图像 tokens: {os.path.join(example_dir, 'imageToken')} ")
    print(f"  事件 tokens: {os.path.join(example_dir, 'eventToken')} ")

    # transforms
    NIMA_H, NIMA_W        = 224, 224
    nima_scale_range            = (0.5, 1.0)
    type = "EI"
    train_preprocessors   = PairedProcessor([
                                        ToTensor(type),
                                        Normalize(EVT_MEAN, EVT_STD, IMG_MEAN, IMG_STD, type=type),
                                        PadToMinSide(NIMA_H),
                                        CenterCrop(size=(NIMA_H, NIMA_W)),
                                    ])
    valid_preprocessors     = PairedProcessor([
                                        ToTensor(type),
                                        Normalize(EVT_MEAN, EVT_STD, IMG_MEAN, IMG_STD, type=type),
                                        PadToMinSide(NIMA_H),
                                        CenterCrop(size=(NIMA_H, NIMA_W)),
                                    ])

    # encoders
    device = args.device
    image_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(device)
    event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(device)
    if args.image_ckpt:
        sd = torch.load(args.image_ckpt, map_location=device, weights_only=True)
        image_encoder.load_state_dict(sd, strict=True)
    else:
        print('警告: 未提供 image_ckpt, 使用随机初始化权重')
    sd = torch.load(args.event_ckpt, map_location=device, weights_only=True)["event_encoder"]
    event_encoder.load_state_dict(sd, strict=True)
    image_encoder.eval(); event_encoder.eval()

    # Dataset + DataLoader (batched)
    class NIMAPairedDataset(Dataset):
        def __init__(self, envs, imgs):
            self.envs = envs
            self.imgs = imgs

        def __len__(self):
            return len(self.envs)

        def __getitem__(self, idx):
            env_dir = self.envs[idx]
            img_dir = self.imgs[idx]
            fn_ev = os.path.splitext(os.path.basename(env_dir))[0]
            fn_im = os.path.splitext(os.path.basename(img_dir))[0]
            assert fn_ev == fn_im, f'事件与图像文件名不匹配: {fn_ev} != {fn_im}'
            evt_img = Image.open(env_dir).convert('RGB')
            img_img = Image.open(img_dir).convert('RGB')
            if "train" in env_dir:
                evt_t, img_t = train_preprocessors(evt_img, img_img)
            elif "val" in env_dir:
                evt_t, img_t = valid_preprocessors(evt_img, img_img)
            else:
                raise ValueError(f'无法识别样本路径 (非 train/val): {env_dir}')
            return evt_t, img_t, env_dir, img_dir

    loader = DataLoader(
        NIMAPairedDataset(envs, imgs),
        batch_size=getattr(args, 'batch_size', 32),
        shuffle=False,
        num_workers=max(0, args.token_workers),
        pin_memory=str(args.device).startswith('cuda'),
    )

    errs = 0
    processed = 0
    for i, batch in tqdm(enumerate(loader), total=len(loader), unit='batch'):
        if i <= 80086:
            processed += len(batch[2])
            continue
        evts, imgs, env_dirs, img_dirs = batch
        try:
            with torch.no_grad():
                img_tok = image_encoder.forward_features(imgs.to(device))["x_norm_patchtokens"].cpu().to(torch.float32)
                evt_tok = event_encoder.forward_features(evts.to(device))["x_norm_patchtokens"].cpu().to(torch.float32)
            for i, (ed_, id_) in enumerate(zip(env_dirs, img_dirs)):
                fn_ev = os.path.splitext(os.path.basename(ed_))[0]
                fn_im = os.path.splitext(os.path.basename(id_))[0]
                assert fn_ev == fn_im, f'事件与图像文件名不匹配: {fn_ev} != {fn_im}'

                event_base = "/".join(ed_.split("/")[:-1]) + "/" + "eventToken"
                image_base = "/".join(id_.split("/")[:-1]) + "/" + "imageToken"
                out_evt = os.path.join(event_base, fn_ev + ".pt")
                out_img = os.path.join(image_base, fn_im + ".pt")
                os.makedirs(event_base, exist_ok=True)
                os.makedirs(image_base, exist_ok=True)

                torch.save(img_tok[i].clone(), out_img)
                torch.save(evt_tok[i].clone(), out_evt)
            processed += len(ed_)
        except Exception:
            errs += len(ed_)

    print(f'特征提取完成: 成功={processed}, err={errs}')


def _stats_worker(event_png_path: str) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, int, bool, bool]:
    """统计一个样本(同目录事件PNG与同名原图)的RGB和平方和及像素计数.
    入参为事件PNG路径。
    返回: (ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count, ev_ok, img_ok)
    """
    base = os.path.splitext(os.path.basename(event_png_path))[0]
    d = os.path.dirname(event_png_path)
    ev_png = event_png_path

    ev_sum = np.zeros(3, dtype=np.float64)
    ev_sq_sum = np.zeros(3, dtype=np.float64)
    ev_count = 0
    img_sum = np.zeros(3, dtype=np.float64)
    img_sq_sum = np.zeros(3, dtype=np.float64)
    img_count = 0
    ev_ok = False
    img_ok = False

    # 事件 .png
    if os.path.isfile(ev_png):
        try:
            arr = np.array(Image.open(ev_png).convert('RGB'), dtype=np.float64) / 255.0
            h, w, _ = arr.shape
            r = arr.reshape(-1, 3)
            ev_sum += r.sum(axis=0)
            ev_sq_sum += (r * r).sum(axis=0)
            ev_count += h * w
            ev_ok = True
        except Exception:
            pass

    # 图像: 同目录同basename的图像副本/链接
    # 尝试多种扩展名
    cand_exts = ['.JPEG', '.jpg', '.jpeg', '.png']
    local_img = None
    for ext in cand_exts:
        p = os.path.join(d, base + ext)
        if os.path.isfile(p):
            local_img = p
            break
    img_path = local_img
    if img_path and os.path.isfile(img_path):
        try:
            arr = np.array(Image.open(img_path).convert('RGB'), dtype=np.float64) / 255.0
            h, w, _ = arr.shape
            r = arr.reshape(-1, 3)
            img_sum += r.sum(axis=0)
            img_sq_sum += (r * r).sum(axis=0)
            img_count += h * w
            img_ok = True
        except Exception:
            pass

    return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count, ev_ok, img_ok


def parse_args():
    p = argparse.ArgumentParser(
        description='批量事件到RGB预处理 (同目录输出, 默认零参数)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--train_root', type=str, default='/data/storage/jianwen/N_ImageNet/extracted_train')
    p.add_argument('--valid_root', type=str, default='/data/storage/jianwen/N_ImageNet/extracted_val')
    p.add_argument('--img-train-root', type=str, default='/data/storage/jianwen/ILSVRC2012/train')
    p.add_argument('--img-val-root', type=str, default='/data/storage/jianwen/ILSVRC2012/valid')
    p.add_argument('--percentile', type=float, default=99.0, help='强度裁剪分位数')
    p.add_argument('--num_workers', type=int, default=8, help='并行进程数')
    p.add_argument('--overwrite', action='store_true', help='覆盖已存在的 .png')
    p.add_argument('--link', action='store_true', help='使用符号链接保存原始图像 (默认复制/已有则跳过)')
    p.add_argument('--stats-out', type=str, default=None, help='将统计结果保存到YAML文件')
    p.add_argument('--val-only', action='store_true', help='仅处理验证集')
    # tokenization options
    p.add_argument('--process-tokens', action='store_true', help='对NIMA提取图像/事件tokens')
    p.add_argument('--device', type=str, default='cuda:2')
    p.add_argument('--token-workers', type=int, default=8)
    p.add_argument('--image-ckpt', type=str, default="/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth")
    p.add_argument('--event-ckpt', type=str, default="/data/storage/jianwen/cache/ckpt_matters/gra_mixture_16x.pt")
    p.add_argument('--batch-size', type=int, default=16, help='DataLoader batch size for token extraction')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # run_batch(args)
    process_tokens(args)


