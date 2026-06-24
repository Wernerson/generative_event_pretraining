import math
from typing import Literal, Optional, Sized
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import math
from typing import Dict, Tuple, List, Iterator
import h5py
from numba import jit
import numpy as np
import hdf5plugin


def get_param_groups(model, wd: float, encoder_lr_mult: float = 1.0, transformer_lr_mult = 1.0):
    """
    Create param groups so the backbone runs with (base_lr * backbone_lr_mult)
    and the rest (decode head) with base_lr.

    Parameters
    ----------
    model : nn.Module
    base_lr : float
        The learning-rate you want to use for the task head.
    wd : float
        Global weight-decay (will be set to 0 for 1-D params / biases).
    backbone_lr_mult : float, optional
        Ratio applied to `base_lr` for backbone groups.
    """
    en = []
    tr = []
    reg_hd, noreg_hd = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:                 #  frozen params
            continue
        if "encoder" in name:
            en.append(p)
            # if encoder_lr_mult < 1.0:
            #     print(f"shrink lr for {name}")
        elif "transformer" in name:
            tr.append(p)
            # if transformer_lr_mult < 1.0:
                # print(f"shrink lr for {name}")
        else:
            if p.ndim < 2 or name.endswith(".bias"):
                noreg_hd.append(p)
            else:
                reg_hd.append(p)

    return ({'params': en,  'weight_decay': 0., 'lr_mult': encoder_lr_mult}, {'params': tr,'weight_decay': 0., 'lr_mult': transformer_lr_mult}, 
            {'params': reg_hd,  'weight_decay': wd, 'lr_mult': 1.0}, {'params': noreg_hd,'weight_decay': 0., 'lr_mult': 1.0},)

def get_lr(step, warmup_steps, lr, lr_decay_steps, min_lr):
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return lr * step / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if step > lr_decay_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)

def spatiotemporal_aggregate(
    slot: torch.Tensor,
    ids: torch.Tensor,
    tokens_per_image: int = 224,
    images_per_group: int = 8,
    valid_ids: Optional[set[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce token sequences by averaging 8 consecutive images (each made of `tokens_per_image` tokens)
    along the temporal dimension and then pooling spatially.
    """
    if valid_ids is None:
        valid_ids = {0, 1, 2, 3, 4}
    valid_ids = set(valid_ids)

    assert slot.ndim == 3 and ids.ndim == 2, "slot must be (B,T,D) and ids (B,T)"
    assert slot.shape[:2] == ids.shape, "slot and ids must align on batch/time"
    assert tokens_per_image > 0 and images_per_group > 0

    B, _, D = slot.shape
    aggregated_slots: list[torch.Tensor] = []
    aggregated_ids: list[torch.Tensor] = []

    for b in range(B):
        tokens_b = slot[b]
        ids_b = ids[b]
        total_tokens = tokens_b.shape[0]

        if total_tokens == 0:
            aggregated_slots.append(tokens_b.new_zeros((0, D)))
            aggregated_ids.append(ids_b.new_zeros((0,), dtype=ids.dtype))
            continue

        num_classes = max(valid_ids) + 1

        num_images = math.ceil(total_tokens / tokens_per_image)
        pad_tokens = num_images * tokens_per_image - total_tokens

        pad_feat = tokens_b.new_zeros((pad_tokens, D))
        pad_ids = ids_b.new_zeros((pad_tokens,), dtype=ids.dtype)
        pad_mask = tokens_b.new_zeros((pad_tokens, 1))

        tokens_padded = torch.cat((tokens_b, pad_feat), dim=0)
        ids_padded = torch.cat((ids_b, pad_ids), dim=0)
        mask_padded = torch.cat(
            (
                torch.ones((total_tokens, 1), device=tokens_b.device, dtype=tokens_b.dtype),
                pad_mask,
            ),
            dim=0,
        )

        tokens_images = tokens_padded.reshape(num_images, tokens_per_image, D)
        ids_images = ids_padded.reshape(num_images, tokens_per_image)
        mask_images = mask_padded.reshape(num_images, tokens_per_image, 1)

        groups = math.ceil(num_images / images_per_group)
        pad_images = groups * images_per_group - num_images
        if pad_images > 0:
            tokens_images = torch.cat(
                (tokens_images, tokens_images.new_zeros((pad_images, tokens_per_image, D))), dim=0
            )
            ids_images = torch.cat(
                (ids_images, ids_images.new_zeros((pad_images, tokens_per_image))), dim=0
            )
            mask_images = torch.cat(
                (mask_images, mask_images.new_zeros((pad_images, tokens_per_image, 1))), dim=0
            )

        tokens_groups = tokens_images.view(groups, images_per_group, tokens_per_image, D)
        ids_groups = ids_images.view(groups, images_per_group, tokens_per_image)
        mask_groups = mask_images.view(groups, images_per_group, tokens_per_image, 1)

        ids_one_hot = torch.nn.functional.one_hot(
            ids_groups.clamp(min=0).long(), num_classes=num_classes
        ).to(tokens_groups.dtype)

        valid_mask = (mask_groups > 0).squeeze(-1)  # (groups, images_per_group, tokens_per_image)

        time_numer = (tokens_groups * mask_groups).sum(dim=1)  # (groups, tokens_per_image, D)
        time_denom = valid_mask.sum(dim=1)  # (groups, tokens_per_image)
        time_mean = time_numer / time_denom.clamp_min(1.0).unsqueeze(-1)
        time_mean = torch.where(
            time_denom.unsqueeze(-1) > 0, time_mean, time_numer.new_zeros(time_mean.shape)
        )

        counts_time = (ids_one_hot * valid_mask.unsqueeze(-1)).sum(dim=1)  # (groups, tokens_per_image, num_classes)
        time_ids = counts_time.argmax(dim=-1)
        time_ids = time_ids.masked_fill(time_denom == 0, 0)

        spatial_numer = (tokens_groups * mask_groups).sum(dim=2)  # (groups, images_per_group, D)
        spatial_denom = valid_mask.sum(dim=2)  # (groups, images_per_group)
        spatial_mean = spatial_numer / spatial_denom.clamp_min(1.0).unsqueeze(-1)
        spatial_mean = torch.where(
            spatial_denom.unsqueeze(-1) > 0, spatial_mean, spatial_numer.new_zeros(spatial_mean.shape)
        )

        counts_spatial = (ids_one_hot * valid_mask.unsqueeze(-1)).sum(dim=2)  # (groups, images_per_group, num_classes)
        spatial_ids = counts_spatial.argmax(dim=-1)
        spatial_ids = spatial_ids.masked_fill(spatial_denom == 0, 0)

        group_tokens = torch.cat((time_mean, spatial_mean), dim=1)  # (groups, tokens_per_image + images_per_group, D)
        group_ids = torch.cat((time_ids, spatial_ids), dim=1)       # (groups, tokens_per_image + images_per_group)

        valid_time_flags = time_denom > 0
        valid_spatial_flags = spatial_denom > 0
        group_valid = torch.cat(
            (valid_time_flags, valid_spatial_flags), dim=1
        )  # (groups, tokens_per_image + images_per_group)

        tokens_flat = group_tokens.reshape(-1, D)
        ids_flat = group_ids.reshape(-1)
        valid_flat = group_valid.reshape(-1)

        aggregated_tokens = tokens_flat[valid_flat]
        aggregated_ids_seq = ids_flat[valid_flat]

        aggregated_slots.append(aggregated_tokens)
        aggregated_ids.append(aggregated_ids_seq)

    max_len = max(seq.shape[0] for seq in aggregated_slots) if aggregated_slots else 0
    if max_len == 0:
        return (
            slot.new_zeros((B, 0, D)),
            ids.new_zeros((B, 0), dtype=ids.dtype),
        )

    padded_slots: list[torch.Tensor] = []
    padded_ids: list[torch.Tensor] = []

    for token_seq, id_seq in zip(aggregated_slots, aggregated_ids):
        if token_seq.shape[0] < max_len:
            pad_len = max_len - token_seq.shape[0]
            pad_tokens = slot.new_zeros((pad_len, D))
            pad_ids = ids.new_zeros((pad_len,), dtype=ids.dtype)
            token_seq = torch.cat((token_seq, pad_tokens), dim=0)
            id_seq = torch.cat((id_seq, pad_ids), dim=0)
        padded_slots.append(token_seq)
        padded_ids.append(id_seq)

    return torch.stack(padded_slots, dim=0), torch.stack(padded_ids, dim=0)

def multiclass_focal_loss(pred_logit: torch.Tensor,
               label: torch.Tensor,
               gamma: Optional[float] = 2.0,
               alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the multi class focal loss.
    :param pred_logit: Logits from the model, shape [B, C] or [B, C, X1, X2, ...].
    :param label: Ground truth labels, shape [B] or [B, X1, X2, ...].
    :return: Computed focal loss.
    """
    B, C = pred_logit.shape[:2]  # batch size and number of categories
    if pred_logit.dim() > 2:
        # e.g. pred_logit.shape is [B, C, X1, X2]
        pred_logit = pred_logit.reshape(B, C, -1)  # [B, C, X1, X2] => [B, C, X1*X2]
        pred_logit = pred_logit.transpose(1, 2)  # [B, C, X1*X2] => [B, X1*X2, C]
        pred_logit = pred_logit.reshape(-1, C)  # [B, X1*X2, C] => [B*X1*X2, C] set N = B*X1*X2
        label = label.reshape(-1)  # [N, ]

    log_p = torch.log_softmax(pred_logit, dim=-1)  # [N, C]
    log_p = log_p.gather(1, label[:, None]).squeeze()  # [N,]
    p = torch.exp(log_p)  # [N,]

    if alpha is None:
        alpha = torch.ones((C,), dtype=torch.float, device=pred_logit.device)
    alpha = alpha.gather(0, label)  # [N,]

    loss = -1 * alpha * torch.pow(1 - p, gamma) * log_p
    return loss.sum() / alpha.sum()

def multiclass_dice_loss(
    logits: torch.Tensor,             # (B, C, H, W)  — raw model output
    target: torch.Tensor,             # (B, H, W) int  or  (B, C, H, W) one-hot
    ignore_index: int | None = None,
    smooth: float = 1e-6,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Multi-class Dice loss with a *real* ignore_index implementation.

    Steps
    -----
    1. Convert target to integer labels if it is one-hot.
    2. Build a boolean mask of valid pixels (target != ignore_index).
    3. Flatten logits → (N, C) and keep only valid rows.
    4. One-hot the valid integer labels → (N, C).
    5. Compute per-class Dice; optionally average.

    Returned value
    --------------
    • "mean": scalar Dice loss  (default, typical for training)  
    • "none": per-class Dice loss vector shape (C,)
    """
    B, C, H, W = logits.shape
    assert logits.device == target.device
    # ------------------------------------------------------------------ #
    # 1. target → integer labels                                         #
    # ------------------------------------------------------------------ #
    if target.dim() == 4:                 # given as one-hot
        target_int = target.argmax(dim=1)     # (B, H, W)
    elif target.dim() == 3:               # given as integer
        target_int = target.long()
    else:
        raise ValueError("target must have shape (B,H,W) or (B,C,H,W)")

    # ------------------------------------------------------------------ #
    # 2. build mask of pixels to keep                                    #
    # ------------------------------------------------------------------ #
    if ignore_index is None:
        valid_mask = torch.ones_like(target_int, dtype=torch.bool)
    else:
        valid_mask = target_int != ignore_index            # (B,H,W)

    # ------------------------------------------------------------------ #
    # 3. flatten logits and mask                                         #
    # ------------------------------------------------------------------ #
    probs = nn.functional.softmax(logits, dim=1)           # (B,C,H,W)
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)       # (N,C)  N = B*H*W
    target_int = target_int.reshape(-1)                    # (N,)
    valid_mask = valid_mask.reshape(-1)                    # (N,)

    probs = probs[valid_mask]                              # (M,C)
    target_int = target_int[valid_mask]                    # (M,)
    if probs.numel() == 0:                                 # all pixels ignored
        raise ValueError("All pixels are ignore_index; Dice undefined.")

    # ------------------------------------------------------------------ #
    # 4. one-hot the *filtered* labels                                   #
    # ------------------------------------------------------------------ #
    target_1h = F.one_hot(target_int, num_classes=probs.size(1))  # (M,C)

    # ------------------------------------------------------------------ #
    # 5. per-class Dice                                                  #
    # ------------------------------------------------------------------ #
    inter = (probs * target_1h).sum(dim=0)                 # (C,)
    union = probs.sum(dim=0) + target_1h.sum(dim=0)        # (C,)

    dice_per_class = (2 * inter + smooth) / (union + smooth)
    loss_per_class = 1 - dice_per_class                    # (C,)

    if reduction == "mean":
        return loss_per_class.mean()                       # scalar
    elif reduction == "none":
        return loss_per_class                              # vector (C,)
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

def multiclass_ce_loss(pred, target, ignore_index=None):
    """
    Computes Cross-Entropy Loss for multi-class segmentation.
    Args:
        pred: Logits from the model, shape [B, C] or [B, C, X1, X2, ...].
        target: Ground truth labels, shape [B] or [B, X1, X2, ...].
    Returns:
        Scalar Cross-Entropy Loss.
    """
    B, C = pred.shape[:2]
    if pred.dim() > 2:
        # e.g. pred_logit.shape is [B, C, X1, X2]
        pred = pred.reshape(B, C, -1)  # [B, C, X1, X2] => [B, C, X1*X2]
        pred = pred.transpose(1, 2)  # [B, C, X1*X2] => [B, X1*X2, C]
        pred = pred.reshape(-1, C)  # [B, X1*X2, C] => [B*X1*X2, C] set N = B*X1*X2
        target = target.reshape(-1)  # [N, ]
    if ignore_index is not None:
        index = target != ignore_index
        pred = pred[index]
        target = target[index]
    return nn.functional.cross_entropy(pred, target)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class IOU:
    def __init__(self, num_classes: int, ignore_index: int = None, device=None):
        """
        Args:
            num_classes: total number of classes
            ignore_index: if not None, that class label is skipped in both intersection and union
            device: where to keep the accumulators (e.g. 'cpu' or 'cuda')
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device('cpu')
        self.reset()

    def reset(self):
        """Zero out all accumulated intersections and unions."""
        # shape: [num_classes]
        self.intersections = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.unions        = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate intersection and union counts from a batch.
        
        Args:
            preds: predicted class indices, any shape
            targets: ground-truth class indices, same shape as preds
        """
        preds = preds.flatten()
        targets = targets.flatten()

        # optionally mask out ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        # loop per class
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            # boolean masks
            pred_mask = preds == cls
            gt_mask   = targets == cls

            intersection = torch.logical_and(pred_mask, gt_mask).sum()
            union        = torch.logical_or(pred_mask, gt_mask).sum()

            self.intersections[cls] += intersection
            self.unions[cls]         += union

    def compute(self) -> torch.Tensor:
        # avoid divide‑by‑zero
        unions = self.unions.float()
        intersections = self.intersections.float()
        
        # compute IoU, with nan when union is zero
        ious = intersections / unions
        ious[unions == 0] = float('nan')
        return ious

class PixelAccuracy:
    def __init__(self, num_classes: int, ignore_index: int = None, device=None):
        """
        Args:
            num_classes: 类别总数。
            ignore_index: 如果设置，该类别标签在计算时将被忽略。
            device: 累加器所在的设备 (例如 'cpu' 或 'cuda')。
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device('cpu')
        self.reset()

    def reset(self):
        """将所有累加的正确像素数和总像素数清零。"""
        # 形状: [num_classes]
        self.correct_pixels = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.total_pixels   = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        从一个批次中按类别累加正确像素数和总像素数。
        
        Args:
            preds: 预测的类别索引，任意形状。
            targets: 真实的类别索引，与preds形状相同。
        """
        preds = preds.flatten()
        targets = targets.flatten()

        # 可选地，屏蔽掉ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        # 按类别循环
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            # 布尔掩码
            pred_mask = preds == cls
            gt_mask   = targets == cls

            # 正确分类的像素是预测和真实标签都为当前类别的像素
            self.correct_pixels[cls] += torch.logical_and(pred_mask, gt_mask).sum()
            # 总像素是真实标签中属于当前类别的所有像素
            self.total_pixels[cls]   += gt_mask.sum()

    def compute(self) -> torch.Tensor:
        """
        计算每个类别的像素准确率。
        """
        # 避免除以零
        total_pixels = self.total_pixels.float()
        correct_pixels = self.correct_pixels.float()
        
        # 计算每个类别的准确率，当某类别的总像素为0时，结果为nan
        class_accuracies = correct_pixels / total_pixels
        class_accuracies[total_pixels == 0] = float('nan')
        return class_accuracies
    
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res

def compute_mean_std(
        dataset,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
):
    """
    Compute per-channel mean and std for a 3-channel image dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset whose __getitem__ returns an image (C,H,W) or (image, label, …).
        Images must already be tensors in [0, 1] or [0, 255] range; the function
        works in whatever scale you supply.
    batch_size : int, optional
        Batch size for the internal DataLoader (default 64).
    num_workers : int, optional
        Passed straight to DataLoader (defaults to 0 for portability).
    pin_memory : bool, optional
        Passed straight to DataLoader.

    Returns
    -------
    mean : torch.Tensor, shape (3,)
        Channel-wise means.
    std  : torch.Tensor, shape (3,)
        Channel-wise standard deviations.
    """
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory)

    # Running totals
    n_pixels = 0
    channel_sum = torch.zeros(3, dtype=torch.double)
    channel_sum_sq = torch.zeros(3, dtype=torch.double)

    for i, batch in enumerate(tqdm(loader)):
        # Accept (img) or (img, *extra)
        if i >= 0.1 * len(loader):
            break
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Make sure the tensor shape is (B, C, H, W)
        if imgs.ndim == 3:                       # single image
            imgs = imgs.unsqueeze(0)
        elif imgs.ndim == 4 and imgs.shape[1] != 3:
            raise ValueError("Images must be 3-channel (C=3) tensors.")

        b, c, h, w = imgs.shape
        imgs = imgs.to(dtype=torch.double)       # accumulate in high precision
        imgs = imgs.view(b, c, -1)               # flatten H×W

        n_pixels += b * h * w
        channel_sum += imgs.sum(dim=(0, 2))
        channel_sum_sq += (imgs ** 2).sum(dim=(0, 2))

    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    mean = mean.float().tolist()
    std = std.float().tolist()
    print(f"mean: {mean}, std: {std}")
    return mean, std

def accumulate_to_rgb(x, y, p, shape, pct=99.0):
    """白底红/蓝事件图 (参考单文件版本)."""
    if len(shape) == 3:
        H, W = shape[:2]
    else:
        H, W = shape
    m = (x < W) & (y < H) & (x >= 0) & (y >= 0)
    x = x[m]; y = y[m]; p = p[m].astype(bool)
    pos = np.zeros((H, W), dtype=np.float32)
    neg = np.zeros((H, W), dtype=np.float32)
    if x.size:
        np.add.at(pos, (y[p], x[p]), 1)
        np.add.at(neg, (y[~p], x[~p]), 1)

    def norm_with_percentile(a):
        if a.max() == 0:
            return a
        thr = np.percentile(a[a > 0], pct) if np.any(a > 0) else 1.0
        if thr <= 0:
            thr = float(a.max())
        a = np.clip(a, 0, thr) / thr
        return a

    pos_n = norm_with_percentile(pos)
    neg_n = norm_with_percentile(neg)

    dominate_pos = pos_n >= neg_n
    inten_pos = pos_n * dominate_pos
    inten_neg = neg_n * (~dominate_pos)

    R = np.ones((H, W), dtype=np.float32)
    G = np.ones((H, W), dtype=np.float32)
    B = np.ones((H, W), dtype=np.float32)
    G -= inten_pos; B -= inten_pos  # 红
    R -= inten_neg; G -= inten_neg  # 蓝
    img = np.stack([np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)], axis=-1)
    return (img * 255).astype(np.uint8)

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000, for ms > 0
        # (2) t[ms_to_idx[ms] - 1] < ms*1000, for ms > 0
        # (3) ms_to_idx[0] == 0
        # , where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx]).astype(np.uint64)
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = self.events[dset_str][t_start_us_idx:t_end_us_idx]
            assert events[dset_str].size == events['t'].size
        return events

    def iter_events_batched(
        self,
        t_start_us: int,
        t_end_us: int,
        batch_duration_us: int = 50_000,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Yield events in fixed-duration time windows to avoid loading all events at once."""
        t = t_start_us
        while t < t_end_us:
            t_next = min(t + batch_duration_us, t_end_us)
            events = self.get_events(t, t_next)
            if events is not None and events['t'].size > 0:
                yield events
            t = t_next


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]

def info_nce_loss(query_features, key_features, temperature):
    """
    计算 InfoNCE Loss。
    Args:
        query_features (torch.Tensor): 查询特征，形状 (N, D)，来自可调整的event分支。
        key_features (torch.Tensor): 键特征，形状 (N, D)，来自冻结的image分支。
        temperature (float): 温度参数。
    Returns:
        torch.Tensor: InfoNCE Loss。
    """
    query_features = F.normalize(query_features, dim=1)
    key_features = F.normalize(key_features, dim=1)
    similarity_matrix = torch.matmul(query_features, key_features.T) / temperature

    targets = torch.arange(similarity_matrix.shape[0], device=query_features.device)
    
    loss = F.cross_entropy(similarity_matrix, targets)
    return loss

def kl_loss(logits_teacher, logits_student, T):
    logits_teacher = F.normalize(logits_teacher, dim=-1)
    logits_student = F.normalize(logits_student, dim=-1)
    p_teacher = F.softmax(logits_teacher / T, dim=-1)
    p_student = F.log_softmax(logits_student / T, dim=-1)
    loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (T ** 2)
    return loss

def masked_l1_loss(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L1 loss on the valid pixels specified by the mask.
    """
    diff = torch.abs(pred - label)
    masked_diff = diff * mask
    
    num_valid_pixels = mask.sum()
    if num_valid_pixels == 0:
        return torch.tensor(0.0, device=pred.device)
        
    loss = masked_diff.sum() / num_valid_pixels
    return loss

class FlowMetrics:
    def __init__(self, n_vals: Optional[List[int]] = None, device=None):
        """
        A class to accumulate and compute optical flow metrics (EPE, NPE, AE)
        over a large dataset, supporting multiple N-Pixel Error thresholds.

        Args:
            n_vals (list[int], optional): A list of integer thresholds for N-Pixel 
                                          Error calculation. Defaults to [1, 3, 5].
            device: The device to store accumulator tensors on (e.g., 'cpu' or 'cuda').
        """
        if n_vals is None:
            self.n_vals = [1, 2, 3]
        else:
            # Sort the list to ensure consistent output order in the dictionary
            self.n_vals = sorted(n_vals)
            
        self.device = device or torch.device('cpu')
        self.reset()

    def reset(self):
        """
        Resets all accumulators to zero.
        Using float64 for sums to maintain precision over many updates.
        """
        self.total_epe_sum = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.total_ae_sum = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        # Create a list of accumulators, one for each n_val
        self.total_npe_pixels = [
            torch.tensor(0, dtype=torch.long, device=self.device) for _ in self.n_vals
        ]
        
        self.total_valid_pixels = torch.tensor(0, dtype=torch.long, device=self.device)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Accumulates statistics from a single batch of predictions and labels.

        Args:
            pred (torch.Tensor): Predicted flow, shape (B, 2, H, W).
            label (torch.Tensor): Ground truth flow, shape (B, 2, H, W).
            mask (torch.Tensor): Validity mask, shape (B, 1, H, W), where 1 is valid.
        """
        label, mask = target[:, :2], target[:, 2:]
        batch_valid_pixels = mask.sum()
        
        if batch_valid_pixels == 0:
            return

        # --- EPE Calculation (calculated once per batch) ---
        # print(pred.shape, label.shape)
        pixel_epe = torch.sqrt(torch.sum((pred - label) ** 2, dim=1, keepdim=True))
        masked_epe = pixel_epe * mask
        self.total_epe_sum += masked_epe.sum().to(self.total_epe_sum)

        # --- NPE Calculation (iterate over all n_vals) ---
        for i, n_val in enumerate(self.n_vals):
            error_pixels_mask = (pixel_epe > n_val).float()
            masked_npe_pixels = error_pixels_mask * mask
            self.total_npe_pixels[i] += masked_npe_pixels.sum().to(self.total_npe_pixels[i])

        # --- AE Calculation ---
        pred_u, pred_v = pred[:, 0, ...], pred[:, 1, ...]
        label_u, label_v = label[:, 0, ...], label[:, 1, ...]
        
        numerator = (pred_u * label_u) + (pred_v * label_v) + 1
        denominator = torch.sqrt(pred_u**2 + pred_v**2 + 1) * torch.sqrt(label_u**2 + label_v**2 + 1)
        cosine_similarity = torch.clamp(numerator / (denominator + 1e-8), -1.0, 1.0)
        
        angle_rad = torch.acos(cosine_similarity).unsqueeze(1)
        masked_angle = angle_rad * mask
        self.total_ae_sum += masked_angle.sum().to(self.total_ae_sum)
        
        self.total_valid_pixels += batch_valid_pixels.to(self.total_valid_pixels)

    def compute(self) -> Dict[str, float]:
        """
        Computes the final metrics from all accumulated statistics.

        Returns:
            A dictionary containing the final EPE, AE, and all requested NPE values.
            Returns NaNs if no valid pixels were ever processed.
        """
        if self.total_valid_pixels == 0:
            nan_val = float('nan')
            results = {
                'epe': nan_val,
                'ae': nan_val
            }
            for n_val in self.n_vals:
                results[f'{n_val}pe'] = nan_val
            return results

        final_epe = self.total_epe_sum / self.total_valid_pixels
        final_ae = self.total_ae_sum / self.total_valid_pixels
        
        results = {
            'epe': final_epe.item(),
            'ae': final_ae.item()
        }
        
        for i, n_val in enumerate(self.n_vals):
            final_npe = self.total_npe_pixels[i] / self.total_valid_pixels
            results[f'{n_val}pe'] = final_npe.item()
        
        return results
