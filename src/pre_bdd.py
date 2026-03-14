import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import math
import sys

from utils import accumulate_to_rgb
import h5py
import numpy as np

# make local packages importable when running this file directly
sys.path.append("dinov2")

import torch
from dinov2.models.vision_transformer import vit_small

# paired transforms from our repo (src/dataset.py)
try:
    from dataset import (
        PairedProcessor,
        ToTensor,
        Normalize,
        RandomHorizontalFlip,
        RandomSwapEventRedBlue,
        ResizeKeepRatio,
        PadToMinSide,
        RandomCrop,
        CenterCrop,
    )
except ModuleNotFoundError:
    # allow running from repo root
    sys.path.append("src")
    from dataset import (
        PairedProcessor,
        ToTensor,
        Normalize,
        RandomHorizontalFlip,
        RandomSwapEventRedBlue,
        ResizeKeepRatio,
        PadToMinSide,
        RandomCrop,
        CenterCrop,
    )

def _file_rgb_stats(path: str):
    """Worker: compute sum, sumsq, count for an RGB image at `path`.
    Returns (sum[3], sumsq[3], count, path, err_msg)
    """
    try:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        sum_c = flat.sum(axis=0, dtype=np.float64)
        sumsq_c = (flat * flat).sum(axis=0, dtype=np.float64)
        count = flat.shape[0]
        return sum_c, sumsq_c, int(count), path, None
    except Exception as e:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0, path, str(e)

class BDDPairer:
    def __init__(
        self,
        event_root="/data/storage/datasets/bdd100k/bdd100k_t0_0.5_events",
        image_root="/data/storage/jianwen/bdd100k/images/100k",
        paired_root="/data/storage/jianwen/bdd100k/paired",
        resize_factor: float = 0.333333333,
        crop_x: int = 61,
        crop_y: int = 0,
        height: int = 240,
        width: int = 304,
        nn_max_divisor: int = 32,
        max_workers: int = None,
        save_original: bool = False,
        split: str = 'train',
    ):
        self.event_root = event_root
        self.image_root = image_root
        self.paired_root = paired_root
        self.resize_factor = resize_factor
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.height = height
        self.width = width
        self.nn_max_divisor = nn_max_divisor
        self.max_workers = max_workers or min(32, (os.cpu_count() or 8) * 4)
        self.save_original = save_original
        self.split = split

        # directories to exclude/allow
        self.excluded_image_dirs = [
            "/data/storage/jianwen/bdd100k/images/100k/train/testB",
            "/data/storage/jianwen/bdd100k/images/100k/train/trainB",
        ]
        self.allowed_train_dirs = [
            "/data/storage/jianwen/bdd100k/images/100k/train/trainA",
            "/data/storage/jianwen/bdd100k/images/100k/train/testA",
        ]

        # build file lists/maps
        self.event_files = sorted(glob.glob(os.path.join(self.event_root, self.split, '**/*.h5'), recursive=True))
        self.image_files = sorted(glob.glob(os.path.join(self.image_root, self.split, '**/*.jpg'), recursive=True))
        # filter out excluded image directories
        self.image_files = [p for p in self.image_files if not self._is_excluded_image(p)]
        self.image_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.image_files}
        # keep only event files that have a non-excluded corresponding image
        self.event_files = [
            ev for ev in self.event_files if self._ev_sample_name(ev) in self.image_map
        ]

        # defaults for tokenization/normalization (BDDD-specific stats, 0..1 range)
        # event (PNG) stats
        self.evt_mean = [0.98062167, 0.96161081, 0.98098914]
        self.evt_std = [0.08805939, 0.12125193, 0.08766055]
        # image (JPG) stats
        self.img_mean = [0.39277547, 0.43453864, 0.44106239]
        self.img_std = [0.24503397, 0.25585564, 0.26836405]

    def _is_excluded_image(self, path: str) -> bool:
        ap = os.path.abspath(path)
        # If under the train root, only allow trainA and testA; exclude others
        train_root = os.path.abspath("/data/storage/jianwen/bdd100k/images/100k/train")
        if ap == train_root or ap.startswith(train_root + os.sep):
            for ad in self.allowed_train_dirs:
                ad_abs = os.path.abspath(ad)
                if ap == ad_abs or ap.startswith(ad_abs + os.sep):
                    break
            else:
                return True
        # Explicit excluded directories (redundant for trainB/testB, but kept for clarity)
        for d in self.excluded_image_dirs:
            ad = os.path.abspath(d)
            if ap == ad or ap.startswith(ad + os.sep):
                return True
        return False

    def _ev_sample_name(self, ev_path: str) -> str:
        return os.path.basename(os.path.dirname(ev_path))

    def _clear_paired(self):
        """Clear paired output only for the current split and ensure structure exists.

        Creates paired_root/train and paired_root/val if missing, but only wipes
        the directory for self.split to avoid removing other split results.
        """
        os.makedirs(self.paired_root, exist_ok=True)
        # ensure both subfolders exist
        for sub in ("train", "val"):
            os.makedirs(os.path.join(self.paired_root, sub), exist_ok=True)
        # clear only current split
        split_dir = os.path.join(self.paired_root, self.split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    def _pad_to_divisor(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        div = self.nn_max_divisor
        target_w = int(math.ceil(w / div) * div)
        target_h = int(math.ceil(h / div) * div)
        if target_w == w and target_h == h:
            return img
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        new_img.paste(img, (0, 0))
        return new_img

    def _process_image(self, img_path: str) -> Image.Image:
        img = Image.open(img_path).convert("RGB")
        resize_factor = 0.333333333
        crop_x, crop_y = 61, 0
        height, width = 240, 304
        nn_max_divisor = 32
        # Step 1: 等比缩放
        scaled_w = round(img.width * resize_factor)
        scaled_h = round(img.height * resize_factor)
        img_resized = img.resize((scaled_w, scaled_h), Image.BILINEAR)

        # Step 2: ROI 裁剪
        crop1 = img_resized.crop((crop_x, crop_y, crop_x + width, crop_y + height))

        # Step 3: 直接中心裁剪到 32 的倍数
        new_w = (crop1.width // nn_max_divisor) * nn_max_divisor
        new_h = (crop1.height // nn_max_divisor) * nn_max_divisor
        dx = (crop1.width - new_w) // 2
        dy = (crop1.height - new_h) // 2
        crop2 = crop1.crop((dx, dy, dx + new_w, dy + new_h))

        return crop2

    def _copy_pair(self, ev_path: str):
        sample_name = os.path.basename(os.path.dirname(ev_path))
        img_path = self.image_map.get(sample_name)
        if not img_path:
            # excluded or missing image; skip
            return
        dst_base = os.path.join(self.paired_root, self.split)
        os.makedirs(dst_base, exist_ok=True)
        dst_img = os.path.join(dst_base, f"{sample_name}.jpg")
        dst_h5 = os.path.join(dst_base, f"{sample_name}.h5")
        # copy h5
        shutil.copy2(ev_path, dst_h5)
        # save image (original or processed)
        if self.save_original:
            shutil.copy2(img_path, dst_img)
        else:
            proc = self._process_image(img_path)
            proc.save(dst_img, format="JPEG", quality=95)

    def pairing(self):
        self._clear_paired()
        total = len(self.event_files)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._copy_pair, ev) for ev in self.event_files]
            for fut in tqdm(as_completed(futures), total=total, desc="Pairing", unit="pair"):
                exc = fut.exception()
                if exc:
                    print("Error:", exc)
        # pairing finished

    def generate_event_images(
        self,
        event_dir: str = None,
        out_dir: str = None,
        h: int = 224,
        w: int = 288,
        n: int = 50000,
        n_start: int = 20000,
        ext: str = ".h5",
        max_workers: int = None,
        ):
        """Convert .h5 event files to PNG event images in parallel.

        Parameters:
        - event_dir: directory containing .h5 files (defaults to self.paired_root)
        - out_dir: directory to write pngs (defaults to event_dir)
        - h,w: output image size
        - n: number of events to use from each file
        - ext: input file extension to look for
        - max_workers: thread pool size (defaults to self.max_workers)
        """
        event_dir = event_dir or os.path.join(self.paired_root, self.split)
        out_dir = out_dir or event_dir
        files = sorted(glob.glob(os.path.join(event_dir, f"*{ext}")))
        # filter out excluded samples based on name
        def _name_from_path(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]
        files = [p for p in files if _name_from_path(p) in self.image_map]
        if not files:
            print("No event files found in", event_dir)
            return

        def _convert_one(h5_path: str):
            try:
                name = os.path.splitext(os.path.basename(h5_path))[0]
                out_path = os.path.join(out_dir, f"{name}.png")
                with h5py.File(h5_path, 'r') as f:
                    x = np.array(f['events']['x'])
                    y = np.array(f['events']['y'])
                    # t = np.array(f['events']['t'])
                    p = np.array(f['events']['p'])
                # limit events
                x, y, p = x[n_start:n+n_start], y[n_start:n+n_start], p[n_start:n+n_start]
                event_image = accumulate_to_rgb(x, y, (p + 1) // 2, (h, w))
                Image.fromarray(event_image).save(out_path)
            except Exception as e:
                # return exception for logging by caller
                return (h5_path, e)
            return None

        workers = max_workers or self.max_workers

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_convert_one, p) for p in files]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Event->PNG", unit="file"):
                res = fut.result()
                if res:
                    path, exc = res
                    print(f"Error converting {path}: {exc}")

    @torch.no_grad()
    def process_tokens(
                        self,
                        device: str = "cuda:0",
                        workers: int = 1,
                        image_ckpt: str | None = None,
                        event_ckpt: str | None = None,
                        h: int = 224,
                        w: int = 224,
    ):
        split_dir = os.path.join(self.paired_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Paired split folder not found: {split_dir}. Run pairing/gen first.")

        # find intersection of jpg and png basenames
        jpgs = glob.glob(os.path.join(split_dir, "*.jpg"))
        pngs = glob.glob(os.path.join(split_dir, "*.png"))
        bn_jpg = {os.path.splitext(os.path.basename(p))[0]: p for p in jpgs}
        bn_png = {os.path.splitext(os.path.basename(p))[0]: p for p in pngs}
        common = sorted(set(bn_jpg.keys()) & set(bn_png.keys()))
        if not common:
            print(f"No paired (.jpg/.png) files found in {split_dir}.")
            return

        # out dirs
        tok_img_dir = os.path.join(split_dir, "imageToken")
        tok_evt_dir = os.path.join(split_dir, "eventToken")
        os.makedirs(tok_img_dir, exist_ok=True)
        os.makedirs(tok_evt_dir, exist_ok=True)

        # preprocessing pipelines
        type_tag = 'EI'

        train_proc = PairedProcessor([
            ToTensor(origi_H=None, type=type_tag),
            Normalize(self.evt_mean, self.evt_std, self.img_mean, self.img_std, type=type_tag),
            RandomSwapEventRedBlue(type=type_tag),
            RandomHorizontalFlip(p=0.5),
            PadToMinSide(target=(h, w), pad_x1=0, pad_x2=0),
            RandomCrop(crop_size=(h, w), type=type_tag),
        ])
        valid_proc = PairedProcessor([
            ToTensor(origi_H=None, type=type_tag),
            Normalize(self.evt_mean, self.evt_std, self.img_mean, self.img_std, type=type_tag),
            # deterministic val path; no scale aug
            PadToMinSide(target=(h, w), pad_x1=0, pad_x2=0),
            CenterCrop((h, w)),
        ])

        # encoders
        image_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(device)
        event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(device)

        # weights
        if image_ckpt is None:
            print("Warning: image_ckpt not provided. Encoders will be randomly initialized.")
        else:
            sd = torch.load(image_ckpt, map_location=device, weights_only=True)
            image_encoder.load_state_dict(sd, strict=True)
        if event_ckpt is None and image_ckpt is not None:
            event_ckpt = image_ckpt
            print("Warning: event_ckpt not provided. Using image_ckpt weights for event encoder.")
        if event_ckpt is not None:
            sd = torch.load(event_ckpt, map_location=device, weights_only=True)
            # support nested dicts like {"event_encoder": state_dict}
            if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "event_encoder")):
                sd = sd.get("event_encoder", sd.get("state_dict", sd))
            event_encoder.load_state_dict(sd, strict=True)

        image_encoder.eval(); event_encoder.eval()

        # processing function
        def _process_one(name: str):
            try:
                jpg_path = bn_jpg[name]
                png_path = bn_png[name]
                img = Image.open(jpg_path).convert("RGB")
                evt = Image.open(png_path).convert("RGB")
                if self.split == 'train':
                    evt_t, img_t = train_proc(evt, img)
                else:
                    evt_t, img_t = valid_proc(evt, img)

                with torch.no_grad():
                    img_tok = image_encoder.forward_features(img_t.unsqueeze(0).to(device))["x_norm_patchtokens"].squeeze(0).cpu()
                    evt_tok = event_encoder.forward_features(evt_t.unsqueeze(0).to(device))["x_norm_patchtokens"].squeeze(0).cpu()

                out_img_pt = os.path.join(tok_img_dir, f"{name}.pt")
                out_evt_pt = os.path.join(tok_evt_dir, f"{name}.pt")
                if not os.path.exists(out_img_pt):
                    torch.save(img_tok, out_img_pt)
                if not os.path.exists(out_evt_pt):
                    torch.save(evt_tok, out_evt_pt)
            except Exception as e:
                return (name, e)
            return None

        # run
        errs = []
        if workers is None or workers <= 1:
            for name in tqdm(common, desc=f"Tokens [{self.split}]", unit="img"):
                res = _process_one(name)
                if res:
                    errs.append(res)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_process_one, n) for n in common]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Tokens [{self.split}]", unit="img"):
                    res = fut.result()
                    if res:
                        errs.append(res)
        if errs:
            print(f"{len(errs)} errors during tokenization. Examples:")
            for n, e in errs[:10]:
                print(f" - {n}: {e}")

    def compute_stats(self, split: str = None):
        """Compute per-channel mean and std for event PNGs and image JPGs.

        By default computes from the 'train' split as requested.
        """
        split = split or 'train'
        base_dir = os.path.join(self.paired_root, split)
        evt_files = sorted(glob.glob(os.path.join(base_dir, "*.png")))
        img_files = sorted(glob.glob(os.path.join(base_dir, "*.jpg")))
        # Build allowed names from image_root while respecting exclusions
        allowed_images = sorted(glob.glob(os.path.join(self.image_root, split, '**/*.jpg'), recursive=True))
        allowed_images = [p for p in allowed_images if not self._is_excluded_image(p)]
        allowed_names = {os.path.splitext(os.path.basename(p))[0] for p in allowed_images}
        # filter stats inputs to exclude unwanted samples
        def _bn(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]
        evt_files = [p for p in evt_files if _bn(p) in allowed_names]
        img_files = [p for p in img_files if _bn(p) in allowed_names]
        from concurrent.futures import ProcessPoolExecutor

        def _accumulate_parallel(files, desc: str):
            total_sum = np.zeros(3, dtype=np.float64)
            total_sumsq = np.zeros(3, dtype=np.float64)
            total_count = 0
            if not files:
                return np.zeros(3), np.zeros(3)
            workers = self.max_workers
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_file_rgb_stats, p) for p in files]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="img"):
                    sum_c, sumsq_c, count, path, err = fut.result()
                    if err:
                        print(f"Skip {path} due to error: {err}")
                        continue
                    total_sum += sum_c
                    total_sumsq += sumsq_c
                    total_count += count
            if total_count == 0:
                return np.zeros(3), np.zeros(3)
            mean = total_sum / total_count
            var = total_sumsq / total_count - mean ** 2
            std = np.sqrt(np.clip(var, 0, None))
            return mean, std

        evt_mean, evt_std = _accumulate_parallel(evt_files, desc=f"Stats EVT [{split}]")
        img_mean, img_std = _accumulate_parallel(img_files, desc=f"Stats IMG [{split}]")

        print(f"[{split}] Event PNG mean: {evt_mean}, std: {evt_std}")
        print(f"[{split}] Image JPG mean: {img_mean}, std: {img_std}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pair BDD100k events with images and generate event images.")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both", help="Dataset split(s) to process.")
    parser.add_argument("--save-original", action="store_true", help="Save original images instead of processed ones.")
    parser.add_argument("--max-workers", type=int, default=8, help="Max worker threads.")
    parser.add_argument("--only", choices=["pair", "gen", "ext", "both"], default="ext", help="Which step(s) to run.")
    parser.add_argument("--compute-stats", action="store_true", help="Compute mean/std for event and image (from train split by default).")
    # tokenization options
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--token-workers", default=1, type=int, help="Thread workers for token extraction.")
    parser.add_argument("--image-ckpt", default="/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", type=str, help="Path to DINOv2 ViT-S/14 image encoder weights.")
    parser.add_argument("--event-ckpt", default="/data/storage/jianwen/cache/ckpt_matters/gra_mixture_cos_8x.pt", type=str, help="Path to event encoder weights (optional; defaults to image-ckpt).")
    args = parser.parse_args()
    if args.compute_stats:
        # Only compute stats on train split; skip pairing and image generation
        pairer = BDDPairer(save_original=args.save_original, max_workers=args.max_workers, split='train')
        pairer.compute_stats(split='train')
    else:
        splits = [args.split] if args.split in ("train", "val") else ["train", "val"]
        for sp in splits:
            pairer = BDDPairer(save_original=args.save_original, max_workers=args.max_workers, split=sp)
            if args.only in ("pair", "both"):
                pairer.pairing()
            if args.only in ("gen", "both"):
                pairer.generate_event_images()
            if args.only in ("ext", "both"):
                pairer.process_tokens(
                    device=args.device,
                    workers=args.token_workers,
                    image_ckpt=args.image_ckpt,
                    event_ckpt=args.event_ckpt,
                )
