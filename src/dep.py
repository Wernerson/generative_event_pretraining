import sys
sys.path.append("dinov2")
sys.path.append("dinov3")

# Deterministic training setup
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)

import argparse
import itertools
import math
import os
from datetime import datetime
from pathlib import Path
import time

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from torchinfo import summary
from tqdm import tqdm

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.use_deterministic_algorithms(True)

from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
from dinov3.eval.dense.depth.models.dpt_head import DPTHead

from config import DEPConfig
from model import Block, Transformer

try:
    from ecddp import ECDDPEncoder, ECDDPEncoderConfig
except Exception:
    ECDDPEncoder = None
    ECDDPEncoderConfig = None
from utils import get_lr, get_param_groups


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean over valid pixels only."""
    if x.shape != mask.shape:
        mask = mask.expand_as(x)
    denom = mask.sum()
    if denom <= eps:
        return x.new_zeros(())
    return (x * mask).sum() / denom


def silog_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lam: float = 0.85, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al.)."""
    valid = mask > 0.5
    if not valid.any():
        return pred.new_zeros(())
    log_diff = torch.log(pred[valid] + eps) - torch.log(target[valid] + eps)
    loss = torch.sqrt((log_diff ** 2).mean() - lam * (log_diff.mean() ** 2))
    return loss


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs()
    return masked_mean(diff, mask)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encourage aligned depth gradients."""
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    targ_dx = target[..., :, 1:] - target[..., :, :-1]
    targ_dy = target[..., 1:, :] - target[..., :-1, :]

    mask_dx = mask[..., :, 1:] * mask[..., :, :-1]
    mask_dy = mask[..., 1:, :] * mask[..., :-1, :]

    loss_x = masked_mean((pred_dx - targ_dx).abs(), mask_dx)
    loss_y = masked_mean((pred_dy - targ_dy).abs(), mask_dy)
    return loss_x + loss_y


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity index loss."""
    if pred.ndim != 4 or target.ndim != 4 or mask.ndim != 4:
        raise ValueError("pred, target, and mask must be 4-D tensors (B,1,H,W)")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    pad = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    pred_sq = pred * pred
    target_sq = target * target
    pred_target = pred * target

    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    sigma_pred = F.avg_pool2d(pred_sq, window_size, stride=1, padding=pad) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target_sq, window_size, stride=1, padding=pad) - mu_target ** 2
    sigma_cross = F.avg_pool2d(pred_target, window_size, stride=1, padding=pad) - mu_pred * mu_target

    ssim_n = (2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)
    ssim_d = (mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2)
    ssim_map = ssim_n / ssim_d.clamp_min(1e-6)
    ssim_map = torch.clamp((1 - ssim_map) / 2, 0, 1)
    return masked_mean(ssim_map.squeeze(1), mask.squeeze(1))


def normalized_log_to_metric(log_depth_norm: torch.Tensor,
                             min_depth: float,
                             max_depth: float,
                             eps: float = 1e-6) -> torch.Tensor:
    """Invert normalized log depth (in [0, 1]) back to metric depth."""
    min_depth = max(min_depth, eps)
    max_depth = max(max_depth, min_depth + eps)
    log_min = math.log(min_depth)
    log_max = math.log(max_depth)
    log_range = max(log_max - log_min, eps)
    log_min_t = torch.tensor(log_min, device=log_depth_norm.device, dtype=log_depth_norm.dtype)
    log_range_t = torch.tensor(log_range, device=log_depth_norm.device, dtype=log_depth_norm.dtype)
    return torch.exp(log_depth_norm * log_range_t + log_min_t)


def _downsample_tensor(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    if x.ndim == 3:
        x = x.unsqueeze(1)
        squeezed = True
    elif x.ndim == 4:
        squeezed = False
    else:
        raise ValueError("Expected tensor with 3 or 4 dims for downsampling")
    pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    return pooled.squeeze(1) if squeezed else pooled


def multi_scale_si_gradient_loss(pred: torch.Tensor,
                                 target: torch.Tensor,
                                 mask: torch.Tensor,
                                 scales: tuple[int, ...] = (1, 2, 4, 8),
                                 eps: float = 1e-6) -> torch.Tensor:
    """Multi-scale scale-invariant gradient matching loss."""
    total = pred.new_zeros(())
    valid_levels = 0
    mask = mask.float()
    for scale in scales:
        pred_s = _downsample_tensor(pred, scale)
        target_s = _downsample_tensor(target, scale)
        mask_s = _downsample_tensor(mask, scale)
        mask_s = (mask_s > 0.5).float()
        if mask_s.sum() <= 0:
            continue
        log_pred = torch.log(pred_s.clamp_min(eps))
        log_target = torch.log(target_s.clamp_min(eps))
        total = total + gradient_loss(log_pred, log_target, mask_s)
        valid_levels += 1
    if valid_levels == 0:
        return pred.new_zeros(())
    return total / valid_levels


def replace_batchnorm_with_groupnorm(module: nn.Module) -> None:
    """Recursively replace BatchNorm layers with GroupNorm layers."""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, child in module.named_children():
        if isinstance(child, bn_types):
            num_channels = child.num_features
            if num_channels <= 0:
                setattr(module, name, nn.Identity())
                continue
            num_groups = max(1, num_channels // 2)
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            gn = nn.GroupNorm(num_groups, num_channels, eps=child.eps, affine=True)
            setattr(module, name, gn)
        else:
            replace_batchnorm_with_groupnorm(child)


@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor,
                          target: torch.Tensor,
                          mask: torch.Tensor,
                          min_depth: float | None,
                          max_depth: float | None,
                          eps: float = 1e-6) -> dict[str, float]:
    """Compute δ metrics, absolute error, RMSE, RMSElog on valid LiDAR pixels."""
    if min_depth is not None:
        pred = torch.clamp(pred, min=min_depth)
        target = torch.clamp(target, min=min_depth)
    if max_depth is not None:
        pred = torch.clamp(pred, max=max_depth)
        target = torch.clamp(target, max=max_depth)

    if mask.shape != target.shape:
        mask = mask.expand_as(target)
    valid = mask > 0.5
    if not valid.any():
        return {k: float("nan") for k in ["delta1", "delta2", "delta3", "abs", "rmse", "rmse_log"]}

    p = pred[valid]
    t = target[valid]

    ratio = torch.max(p / (t + eps), t / (p + eps))
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25 ** 2).float().mean()
    delta3 = (ratio < 1.25 ** 3).float().mean()

    abs_err = (p - t).abs()
    rmse = torch.sqrt(((p - t) ** 2).mean())
    p_nonneg = torch.clamp(p, min=0.0)
    t_nonneg = torch.clamp(t, min=0.0)
    rmse_log = torch.sqrt(((torch.log1p(p_nonneg) - torch.log1p(t_nonneg)) ** 2).mean())

    return {
        "delta1": delta1.item(),
        "delta2": delta2.item(),
        "delta3": delta3.item(),
        "abs": abs_err.mean().item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
    }


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleDepthDecoder(nn.Module):
    def __init__(self, config: DEPConfig, stage_dims: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        if not stage_dims:
            raise ValueError("stage_dims must contain at least one entry")
        self.config = config
        self.stage_dims = stage_dims
        self.base_h = config.H // config.patch_size
        self.base_w = config.W // config.patch_size
        self.full_h = config.H
        self.full_w = config.W

        self.proj = nn.Linear(config.n_embed, stage_dims[0])

        stages: list[DecoderBlock] = []
        for idx, out_dim in enumerate(stage_dims):
            in_dim = stage_dims[idx - 1] if idx > 0 else stage_dims[0]
            stages.append(DecoderBlock(in_dim, out_dim))
        self.stages = nn.ModuleList(stages)

        fusion_in_ch = sum(stage_dims)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _target_shapes(self) -> list[tuple[int, int]]:
        targets: list[tuple[int, int]] = []
        cur_h, cur_w = self.base_h, self.base_w
        total_stages = len(self.stage_dims)
        for idx in range(total_stages):
            if idx == 0:
                targets.append((cur_h, cur_w))
                continue
            if idx == total_stages - 1:
                targets.append((self.full_h, self.full_w))
                continue
            cur_h = min(cur_h * 2, self.full_h)
            cur_w = min(cur_w * 2, self.full_w)
            targets.append((cur_h, cur_w))
        return targets

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, _ = tokens.shape
        h = self.base_h
        w = self.base_w
        proj = self.proj(tokens)
        feat = einops.rearrange(proj, "b (h w) c -> b c h w", h=h, w=w)

        targets = self._target_shapes()
        features: list[torch.Tensor] = []
        x = feat
        for stage, target_shape in zip(self.stages, targets):
            x = stage(x)
            if x.shape[-2:] != target_shape:
                x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
            features.append(x)

        fused = torch.cat(
            [
                F.interpolate(feat_map, size=(self.full_h, self.full_w), mode="bilinear", align_corners=False)
                if feat_map.shape[-2:] != (self.full_h, self.full_w)
                else feat_map
                for feat_map in features
            ],
            dim=1,
        )
        return self.fusion(fused)


class Dinov3DPTDecoder(nn.Module):
    def __init__(self, config: DEPConfig):
        super().__init__()
        self.config = config
        self.layer_indices = list(getattr(config, "dpt_layer_indices", [2, 5, 8, 11]))
        if not self.layer_indices:
            raise ValueError("dpt_layer_indices must contain at least one layer index.")
        self.num_stages = len(self.layer_indices)
        in_channels = getattr(config, "dpt_in_channels", None)
        if in_channels is None:
            in_channels = [config.n_embed] * self.num_stages
        if len(in_channels) != self.num_stages:
            raise ValueError("Length of dpt_in_channels must match dpt_layer_indices.")
        post_channels = getattr(config, "dpt_post_process_channels", [128, 256, 512, 1024])
        if len(post_channels) != self.num_stages:
            raise ValueError("Length of dpt_post_process_channels must match dpt_layer_indices.")
        self.base_h = config.H // config.patch_size
        self.base_w = config.W // config.patch_size
        self.head = DPTHead(
            in_channels=in_channels,
            channels=getattr(config, "dpt_channels", 256),
            post_process_channels=post_channels,
            readout_type="ignore",
            n_output_channels=1,
            use_batchnorm=getattr(config, "dpt_use_batchnorm", False),
        )

    def forward(self, stage_inputs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if stage_inputs is None or len(stage_inputs) != self.num_stages:
            raise ValueError("stage_inputs must match the configured number of DPT stages.")
        processed = []
        for tokens, cls_token in stage_inputs:
            feat = einops.rearrange(tokens, "b (h w) c -> b c h w", h=self.base_h, w=self.base_w)
            processed.append((feat, cls_token))
        return self.head(processed)


class DepthEstimator(Transformer):
    def __init__(self, config: DEPConfig):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.device)
        self.encoder_mode = getattr(config, "encoder_mode", "ours")
        self.is_ecddp = self.encoder_mode == "ecddp"
        if self.is_ecddp:
            self.num_modalities = 1
        else:
            self.num_modalities = int(config.use_events) + int(config.use_rgb)
        if self.num_modalities == 0:
            raise ValueError("At least one modality (events or RGB) must be enabled.")

        self.vit_backbone = getattr(config, "vit_backbone", "dinov2")
        self.vit_size = getattr(config, "vit", "base")
        self.stage_layer_indices = list(getattr(config, "dpt_layer_indices", [2, 5, 8, 11]))
        self.decoder_name = getattr(config, "depth_decoder", "baseline")
        self.use_dpt_decoder = (self.decoder_name == "dinov3_dpt") and (not self.is_ecddp)

        if self.is_ecddp:
            self.event_encoder = self._build_ecddp_encoder()
            self.image_encoder = None
        else:
            self.event_encoder = self._build_backbone(use_register_tokens=False)
            self.image_encoder = self._build_backbone(use_register_tokens=True)

        if (not self.is_ecddp) and config.event_encoder_weight is not None:
            load_result = self.event_encoder.load_state_dict(config.event_encoder_weight, strict=False)
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
            if missing:
                print("[event encoder] missing keys:", missing)
            if unexpected:
                print("[event encoder] unexpected keys:", unexpected)
            print("*" * 40 + " loaded event encoder weights")
        if (not self.is_ecddp) and config.image_encoder_weight is not None and self.image_encoder is not None:
            load_result = self.image_encoder.load_state_dict(config.image_encoder_weight, strict=False)
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
            if missing:
                print("[image encoder] missing keys:", missing)
            if unexpected:
                print("[image encoder] unexpected keys:", unexpected)
            print("*" * 40 + " loaded image encoder weights")

        if not self.is_ecddp:
            replace_batchnorm_with_groupnorm(self.event_encoder)
            replace_batchnorm_with_groupnorm(self.image_encoder)

        transformer_weight = getattr(config, "transformer_weight", None)
        self.use_transformer = (transformer_weight is not None) and (not self.is_ecddp)
        if self.use_transformer:
            print("using transformer")
            self.transformer = nn.ModuleDict(
                dict(
                    modality_embed=nn.Embedding(5, config.n_embed),
                    pos_embed=nn.Embedding(config.window_size, config.n_embed),
                    blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    norm=nn.LayerNorm(config.n_embed),
                )
            )
            load_result = self.transformer.load_state_dict(transformer_weight, strict=False)
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
            if missing:
                print("[transformer] missing keys:", missing)
            if unexpected:
                print("[transformer] unexpected keys:", unexpected)
            print("*" * 40 + " loaded transformer weights")
            replace_batchnorm_with_groupnorm(self.transformer)
        else:
            self.transformer = None

        if self.num_modalities > 1:
            self.fusion = nn.Sequential(
                nn.Linear(config.n_embed * self.num_modalities, config.n_embed),
                nn.GELU(),
                nn.Linear(config.n_embed, config.n_embed),
            )
        else:
            self.fusion = nn.Identity()

        if self.use_dpt_decoder:
            self.decoder = Dinov3DPTDecoder(config)
        else:
            self.decoder = MultiScaleDepthDecoder(config)
        self.ray_encoder = None if self.is_ecddp else nn.Linear(2, config.n_embed)
        self._ray_grid_cache: dict[tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        self.active_stage = None
        self.validation_configs = config.validation_splits
        self.validation_loaders: dict[str, DataLoader] = {}
        for name, val_cfg in self.validation_configs.items():
            self.validation_loaders[name] = DataLoader(
                val_cfg["dataset"],
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.n_workers,
                pin_memory=True,
                drop_last=False,
            )

        self.training_stages = []
        for stage_cfg in config.training_stages:
            stage_entry = dict(stage_cfg)
            stage_entry["train_loader"] = DataLoader(
                stage_cfg["train_dataset"],
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.n_workers,
                pin_memory=True,
                drop_last=False,
            )
            self.training_stages.append(stage_entry)

        self.amp = torch.amp.autocast(device_type="cuda")
        self.scaler = torch.amp.GradScaler(device="cuda")

        self.optimizer = torch.optim.AdamW(
            get_param_groups(self, config.wd, encoder_lr_mult=config.encoder_lr_mult, transformer_lr_mult=config.transformer_lr_mult)
        )

        now = datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.run_dir = Path("src/runs") / f"{now}_dep"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path("/data/storage/jianwen/cache/ckpts") / f"{now}_dep"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        if config.encoder_frozen:
            print("Freezing encoder parameters.")
            for name, param in self.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False

        self._summary_written = False

        self.to(self.device)

    def log_scalar(self, category: str, group: str, name: str, value: float, step: int):
        """Write scalar values with consistent TensorBoard naming."""
        if group:
            tag = f"{category}/{group}/{name}"
        else:
            tag = f"{category}/{name}"
        self.writer.add_scalar(tag, value, step)

    def _build_backbone(self, use_register_tokens: bool) -> nn.Module:
        vit_size = getattr(self.config, "vit", "base")
        if self.vit_backbone == "dinov3":
            from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

            builders = {
                "small": dinov3_vits16,
                "base": dinov3_vitb16,
                "large": dinov3_vitl16,
            }
            builder = builders.get(vit_size)
            if builder is None:
                raise ValueError(f"Unsupported DINOv3 ViT size '{vit_size}'.")
            return builder(pretrained=False)

        builders = {
            "small": vit_small,
            "base": vit_base,
            "large": vit_large,
        }
        builder = builders.get(vit_size)
        if builder is None:
            raise ValueError(f"Unsupported DINOv2 ViT size '{vit_size}'.")
        kwargs = dict(
            patch_size=self.config.patch_size,
            img_size=518,
            block_chunks=0,
            init_values=1e-6,
            num_register_tokens=4,
        )
        if use_register_tokens:
            kwargs["num_register_tokens"] = 4
        return builder(**kwargs)

    def _build_ecddp_encoder(self) -> nn.Module:
        if ECDDPEncoder is None or ECDDPEncoderConfig is None:
            raise ImportError("ECDDP module is not available.")
        cfg = ECDDPEncoderConfig(
            ckpt_path=getattr(self.config, "ecddp_ckpt_path"),
            image_size=getattr(self.config, "ecddp_image_size", (self.config.H, self.config.W)),
            patch_size=getattr(self.config, "ecddp_patch_size", 4),
            in_chans=getattr(self.config, "ecddp_in_chans", 3),
            embed_dim=getattr(self.config, "ecddp_embed_dim", 96),
            depths=tuple(getattr(self.config, "ecddp_depths", (2, 2, 6, 2))),
            num_heads=tuple(getattr(self.config, "ecddp_num_heads", (3, 6, 12, 24))),
            window_size=getattr(self.config, "ecddp_window_size", 7),
            mlp_ratio=getattr(self.config, "ecddp_mlp_ratio", 4.0),
            drop_rate=getattr(self.config, "ecddp_drop_rate", 0.0),
            attn_drop_rate=getattr(self.config, "ecddp_attn_drop_rate", 0.0),
            drop_path_rate=getattr(self.config, "ecddp_drop_path_rate", 0.0),
            device=self.config.device,
            freeze=getattr(self.config, "encoder_frozen", True),
        )
        encoder = ECDDPEncoder(cfg)
        if not getattr(self.config, "encoder_frozen", True):
            encoder.backbone.train()
            for param in encoder.parameters():
                param.requires_grad_(True)
        return encoder

    def _encode_backbone_tokens(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
        ray_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        if self.use_dpt_decoder:
            outputs = encoder.get_intermediate_layers(
                x,
                n=self.stage_layer_indices,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
            stage_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for patch_tokens, cls_token in outputs:
                tokens = patch_tokens
                if ray_tokens is not None:
                    tokens = tokens + ray_tokens
                stage_pairs.append((tokens, cls_token))
            final_tokens = stage_pairs[-1][0]
            return final_tokens, stage_pairs

        feat = encoder.forward_features(x)["x_norm_patchtokens"]
        if ray_tokens is not None:
            feat = feat + ray_tokens
        return feat, None

    def _fuse_modalities(
        self,
        event_feat: torch.Tensor | None,
        image_feat: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if event_feat is None and image_feat is None:
            return None
        if event_feat is None:
            return image_feat
        if image_feat is None:
            return event_feat
        mode = getattr(self.config, "fuse_mode", "concat")
        if mode == "add":
            return (event_feat + image_feat) / 2
        fused = torch.cat([event_feat, image_feat], dim=-1)
        return self.fusion(fused)

    def _prepare_stage_inputs(
        self,
        stage_features: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if not self.use_dpt_decoder:
            return None
        if stage_features is None:
            raise ValueError("Stage features required for DPT decoder.")
        if self.num_modalities == 1:
            key = "event" if self.config.use_events else "image"
            stages = stage_features.get(key)
            if stages is None:
                raise ValueError(f"Missing stage features for modality '{key}'.")
            return stages

        event_stages = stage_features.get("event")
        image_stages = stage_features.get("image")
        if event_stages is None or image_stages is None:
            raise ValueError("Both event and image stage features are required for fusion.")
        fused_stages: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx in range(self.decoder.num_stages):
            event_tokens, event_cls = event_stages[idx]
            image_tokens, image_cls = image_stages[idx]
            fused_tokens = self._fuse_modalities(event_tokens, image_tokens)
            fused_cls = self._fuse_modalities(event_cls, image_cls)
            fused_stages.append((fused_tokens, fused_cls))
        return fused_stages

    def _prepare_intrinsics(self, intrinsics: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        if intrinsics is None:
            return None
        return {k: v.to(self.device, non_blocking=True) for k, v in intrinsics.items()}

    def _resolve_depth_bounds(self, min_depth: float | None, max_depth: float | None) -> tuple[float, float]:
        base_min = self.config.min_depth if getattr(self.config, "min_depth", None) is not None else 0.1
        resolved_min = float(min_depth if min_depth is not None else base_min)
        fallback_max = getattr(self.config, "depth_normalizer_max", None)
        if fallback_max is None:
            fallback_max = resolved_min + 1.0
        resolved_max = float(max_depth if max_depth is not None else fallback_max)
        if resolved_max <= resolved_min:
            resolved_max = resolved_min + 1e-3
        return resolved_min, resolved_max

    def _ray_tokens(
        self,
        intrinsics: dict[str, torch.Tensor] | None,
        spatial_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if intrinsics is None or spatial_shape is None or self.ray_encoder is None:
            return None
        required = ("fx", "fy", "cx", "cy")
        if not all(k in intrinsics for k in required):
            return None
        device = self.device
        fx = intrinsics["fx"].float().view(-1, 1, 1)
        fy = intrinsics["fy"].float().view(-1, 1, 1)
        cx = intrinsics["cx"].float().view(-1, 1, 1)
        cy = intrinsics["cy"].float().view(-1, 1, 1)
        B = fx.shape[0]
        h, w = spatial_shape
        if h % self.config.patch_size != 0 or w % self.config.patch_size != 0:
            return None
        cache_key = (device, h, w)
        if cache_key not in self._ray_grid_cache:
            y_coords = torch.arange(h, device=device, dtype=torch.float32) + 0.5
            x_coords = torch.arange(w, device=device, dtype=torch.float32) + 0.5
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self._ray_grid_cache[cache_key] = (grid_y.unsqueeze(0), grid_x.unsqueeze(0))
        grid_y, grid_x = self._ray_grid_cache[cache_key]
        x_norm = (grid_x - cx) / fx
        y_norm = (grid_y - cy) / fy
        rays = torch.stack([x_norm, y_norm], dim=1)  # (B, 2, H, W)
        patch_rays = einops.reduce(
            rays,
            "b c (hp p1) (wp p2) -> b (hp wp) c",
            "mean",
            p1=self.config.patch_size,
            p2=self.config.patch_size,
        )
        return self.ray_encoder(patch_rays)

    def forward_backbones(
        self,
        event: torch.Tensor | None,
        image: torch.Tensor | None,
        intrinsics: dict[str, torch.Tensor] | None,
    ):
        if self.is_ecddp:
            if event is None:
                raise ValueError("ECDDP encoder requires event inputs.")
            feats = self.event_encoder.extract_features(event)
            tokens = feats["patch_tokens"]
            ids = torch.zeros((tokens.size(0), tokens.size(1)), device=tokens.device, dtype=torch.long)
            slices = {"event": slice(0, tokens.size(1))}
            return tokens, ids, slices, None

        tokens = []
        modality_ids = []
        slices = {}
        cursor = 0
        stage_features: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] | None = {} if self.use_dpt_decoder else None

        spatial_shape = None
        if event is not None:
            spatial_shape = event.shape[-2:]
        elif image is not None:
            spatial_shape = image.shape[-2:]

        ray_tokens = self._ray_tokens(intrinsics, spatial_shape)

        if self.config.use_events and event is not None:
            feat_e, stages_e = self._encode_backbone_tokens(self.event_encoder, event, ray_tokens)
            tokens.append(feat_e)
            modality_ids.append(torch.full((feat_e.size(0), feat_e.size(1)), 2, device=feat_e.device, dtype=torch.long))
            slices["event"] = slice(cursor, cursor + feat_e.size(1))
            cursor += feat_e.size(1)
            if stage_features is not None and stages_e is not None:
                stage_features["event"] = stages_e

        if self.config.use_rgb and image is not None:
            feat_i, stages_i = self._encode_backbone_tokens(self.image_encoder, image, ray_tokens)
            tokens.append(feat_i)
            modality_ids.append(torch.full((feat_i.size(0), feat_i.size(1)), 1, device=feat_i.device, dtype=torch.long))
            slices["image"] = slice(cursor, cursor + feat_i.size(1))
            cursor += feat_i.size(1)
            if stage_features is not None and stages_i is not None:
                stage_features["image"] = stages_i

        if not tokens:
            raise ValueError("No tokens produced. Check modality settings.")

        tokens_cat = torch.cat(tokens, dim=1)
        ids_cat = torch.cat(modality_ids, dim=1)
        return tokens_cat, ids_cat, slices, stage_features

    def forward_transformer(self, tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = tokens.shape
        pos_idx = torch.arange(T, device=tokens.device)
        pos_emb = self.transformer["pos_embed"](pos_idx).unsqueeze(0)
        modal_emb = self.transformer["modality_embed"](ids)
        x = tokens + pos_emb + modal_emb
        for blk in self.transformer["blocks"]:
            x = blk(x)
        x = self.transformer["norm"](x)
        return x

    def forward_decoder(
        self,
        tokens: torch.Tensor,
        min_depth: float,
        max_depth: float,
        stage_inputs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if self.use_dpt_decoder:
            if stage_inputs is None:
                raise ValueError("DPT decoder requires stage inputs.")
            norm_log_depth = torch.sigmoid(self.decoder(stage_inputs))
        else:
            _B, _L, _ = tokens.shape
            norm_log_depth = torch.sigmoid(self.decoder(tokens))
        depth = normalized_log_to_metric(norm_log_depth, min_depth, max_depth)
        return depth

    def forward(
        self,
        event: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        intrinsics: dict[str, torch.Tensor] | None = None,
        depth_bounds: tuple[float | None, float | None] | None = None,
    ):
        tokens, ids, slices, stage_features = self.forward_backbones(event, image, intrinsics)
        if self.use_transformer:
            tokens = self.forward_transformer(tokens, ids)
        fused_tokens = tokens

        if self.num_modalities == 2:
            event_tokens = fused_tokens[:, slices["event"], :]
            image_tokens = fused_tokens[:, slices["image"], :]
            fused = self._fuse_modalities(event_tokens, image_tokens)
        else:
            fused = fused_tokens

        if depth_bounds is None:
            stage = self.active_stage or {}
            bounds = (stage.get("min_depth"), stage.get("max_depth"))
        else:
            bounds = depth_bounds
        min_depth, max_depth = self._resolve_depth_bounds(bounds[0], bounds[1])
        decoder_stage_inputs = self._prepare_stage_inputs(stage_features)
        depth = self.forward_decoder(fused, min_depth, max_depth, decoder_stage_inputs)
        return depth

    def compute_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        min_depth: float | None = None,
        max_depth: float | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        pred = pred.squeeze(1)
        target = target.squeeze(1) if target.ndim == 4 else target
        mask = mask.squeeze(1) if mask.ndim == 4 else mask
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1).float(), size=target.shape[-2:], mode="nearest").squeeze(1)
        if min_depth is None or max_depth is None:
            stage_cfg = self.active_stage
            if stage_cfg is not None:
                if min_depth is None:
                    min_depth = stage_cfg.get("min_depth", self.config.min_depth)
                if max_depth is None:
                    max_depth = stage_cfg.get("max_depth", self.config.max_depth)
            else:
                if min_depth is None:
                    min_depth = self.config.min_depth
                if max_depth is None:
                    max_depth = self.config.max_depth
        if min_depth is not None:
            pred = torch.clamp(pred, min=min_depth)
            target = torch.clamp(target, min=min_depth)
        if max_depth is not None:
            pred = torch.clamp(pred, max=max_depth)
            target = torch.clamp(target, max=max_depth)
        losses["silog"] = silog_loss(pred, target, mask)
        losses["ms_grad"] = multi_scale_si_gradient_loss(pred, target, mask)
        return losses

    def train_step(self, batch: dict, global_step: int, local_step: int, stage_cfg: dict):
        self.active_stage = stage_cfg
        event = batch["event"].to(self.device) if self.config.use_events else None
        image = batch["image"].to(self.device) if self.config.use_rgb else None
        depth = batch["depth"].to(self.device)
        mask = batch["mask"].to(self.device)
        intrinsics = self._prepare_intrinsics(batch.get("intrinsics"))

        if not self._summary_written:
            with torch.no_grad():
                summary_inputs: list[torch.Tensor] = []
                if event is not None:
                    summary_inputs.append(event)
                if image is not None:
                    summary_inputs.append(image)
                if summary_inputs:
                    input_data = summary_inputs[0] if len(summary_inputs) == 1 else tuple(summary_inputs)
                    summary(self, input_data=input_data, device=self.config.device, depth=2)
            self._summary_written = True

        start = time.time()
        lr_offset = stage_cfg.get("start_step", 0)
        total_decay_steps = getattr(self.config, "total_steps", 0)
        if total_decay_steps <= 0:
            total_decay_steps = lr_offset + stage_cfg["steps"]
        decay_steps = max(total_decay_steps, lr_offset + stage_cfg["steps"])
        current_lr = get_lr(
            lr_offset + local_step,
            self.config.warmup_steps,
            self.config.lr,
            decay_steps,
            self.config.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr * group.get("lr_mult", 1.0)

        self.train()
        stage_bounds = (stage_cfg.get("min_depth"), stage_cfg.get("max_depth"))
        with self.amp:
            pred = self.forward(event, image, intrinsics, depth_bounds=stage_bounds)
            loss_dict = self.compute_losses(pred, depth, mask)
            total_loss = sum(self.config.depth_loss_weights[k] * loss_dict[k] for k in loss_dict)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if (global_step + 1) % self.config.log_every == 0:
            elapsed = time.time() - start
            log_step = global_step + 1
            self.log_scalar("train", "", "total_loss", total_loss.item(), log_step)
            for name in ("silog", "ms_grad"):
                if name in loss_dict:
                    self.log_scalar("train", "", name, loss_dict[name].item(), log_step)
            print(f"[stage {stage_cfg['name']}] step={log_step} loss={total_loss.item():.4f} silog={loss_dict['silog'].item():.4f} lr={current_lr:.6e} dt={elapsed:.2f}s")

        return total_loss.item()

    @torch.no_grad()
    def visualize(
        self,
        batch: dict,
        pred: torch.Tensor,
        step: int,
        split: str,
        image_stats: dict | None = None,
        event_stats: dict | None = None,
        depth_bounds: tuple[float, float] | None = None,
    ):
        if depth_bounds is None:
            depth_bounds = (self.config.min_depth, self.config.max_depth)
        min_depth, max_depth = depth_bounds
        depth_gt = batch["depth"].to(self.device)
        mask = batch["mask"].to(self.device)
        if min_depth is not None:
            pred = torch.clamp(pred, min=min_depth)
            depth_gt = torch.clamp(depth_gt, min=min_depth)
        if max_depth is not None:
            pred = torch.clamp(pred, max=max_depth)
            depth_gt = torch.clamp(depth_gt, max=max_depth)
        pred = pred.squeeze(1)
        depth_gt = depth_gt.squeeze(1) if depth_gt.ndim == 4 else depth_gt
        mask = mask.squeeze(1) if mask.ndim == 4 else mask

        # stretch depth values per sample using the valid region for clearer contrast
        pred_norm = torch.zeros_like(pred)
        gt_norm = torch.zeros_like(depth_gt)
        for b in range(pred.shape[0]):
            valid = mask[b] > 0.5
            if valid.any():
                combined = torch.cat([
                    pred[b][valid],
                    depth_gt[b][valid],
                ])
                vmin = combined.min()
                vmax = combined.max()
                if (vmax - vmin) < 1e-6:
                    vmax = vmin + 1e-6
                scale = vmax - vmin
                pred_b = pred[b].clamp(vmin, vmax)
                gt_b = depth_gt[b].clamp(vmin, vmax)
                pred_out = torch.zeros_like(pred_b)
                gt_out = torch.zeros_like(gt_b)
                pred_out[valid] = (pred_b[valid] - vmin) / scale
                gt_out[valid] = (gt_b[valid] - vmin) / scale
            else:
                pred_out = torch.zeros_like(pred[b])
                gt_out = torch.zeros_like(depth_gt[b])
            pred_norm[b] = pred_out
            gt_norm[b] = gt_out

        n_vis = min(4, pred_norm.shape[0])
        pred_vis = pred_norm[:n_vis].unsqueeze(1).cpu().repeat(1, 3, 1, 1)
        gt_vis = gt_norm[:n_vis].unsqueeze(1).cpu().repeat(1, 3, 1, 1)

        rgb_vis = None
        raw_image = batch.get("image")
        if self.config.use_rgb and isinstance(raw_image, torch.Tensor):
            rgb_vis = raw_image[:n_vis].clone()
            if image_stats is not None:
                mean_vals = image_stats.get("mean", self.config.MVSEC_IMAGE_ME)
                std_vals = image_stats.get("std", self.config.MVSEC_IMAGE_SE)
            else:
                mean_vals = self.config.MVSEC_IMAGE_ME
                std_vals = self.config.MVSEC_IMAGE_SE
            mean = torch.tensor(mean_vals, dtype=rgb_vis.dtype, device=rgb_vis.device).view(1, -1, 1, 1)
            std = torch.tensor(std_vals, dtype=rgb_vis.dtype, device=rgb_vis.device).view(1, -1, 1, 1)
            rgb_vis = rgb_vis * std + mean
            rgb_vis = rgb_vis.clamp(0, 1).cpu()

        event_vis = None
        raw_event = batch.get("event")
        if self.config.use_events and isinstance(raw_event, torch.Tensor):
            event_vis = raw_event[:n_vis].clone()
            mean_vals = None
            std_vals = None
            if event_stats is not None:
                mean_vals = event_stats.get("mean")
                std_vals = event_stats.get("std")
            if mean_vals is None or std_vals is None:
                mean_vals = getattr(self.config, "MVSEC_EVENT_ME", None)
                std_vals = getattr(self.config, "MVSEC_EVENT_SE", None)
            if mean_vals is not None and std_vals is not None:
                mean = torch.tensor(mean_vals, dtype=event_vis.dtype, device=event_vis.device).view(1, -1, 1, 1)
                std = torch.tensor(std_vals, dtype=event_vis.dtype, device=event_vis.device).view(1, -1, 1, 1)
                if mean.shape[1] != event_vis.shape[1]:
                    mean = mean.mean(dim=1, keepdim=True).expand_as(event_vis)
                    std = std.mean(dim=1, keepdim=True).expand_as(event_vis)
                event_vis = event_vis * std + mean
            event_vis = event_vis.cpu().clamp(0.0, 1.0)
            if event_vis.shape[1] == 1:
                event_vis = event_vis.repeat(1, 3, 1, 1)
            elif event_vis.shape[1] != 3:
                gray = event_vis.mean(dim=1, keepdim=True)
                event_vis = gray.repeat(1, 3, 1, 1)

            # Boost contrast so sparse events remain visible once saved to disk.
            contrast = getattr(self.config, "event_visual_contrast", 1.6)
            if contrast > 1.0:
                event_vis = 1.0 - torch.clamp((1.0 - event_vis) * contrast, 0.0, 1.0)

        tiles = [pred_vis, gt_vis]
        if rgb_vis is not None:
            tiles.insert(0, rgb_vis)
        if event_vis is not None:
            tiles.insert(0, event_vis)

        grid = make_grid(torch.cat(tiles, dim=0), nrow=n_vis, normalize=False)
        grid_cpu = grid.cpu()

        save_path = self.run_dir / f"vis_{split}_{step:06d}.png"
        save_image(grid_cpu, save_path)
        self.writer.add_image(f"visual/{split}", grid_cpu, step)

    @torch.no_grad()
    def validate(self, global_step: int, stage_cfg: dict, valid_name: str, max_batches: int | None = None):
        self.active_stage = stage_cfg
        self.eval()
        val_cfg = self.validation_configs[valid_name]
        metrics_accum = {k: 0.0 for k in self.config.depth_metrics}
        metrics_count = {k: 0 for k in self.config.depth_metrics}
        loss_accum = {k: 0.0 for k in self.config.depth_loss_weights}
        count = 0

        base_loader = self.validation_loaders[valid_name]
        loader = base_loader
        if max_batches is not None:
            loader = itertools.islice(loader, max_batches)
            total = max_batches
        else:
            total = None

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Validating[{valid_name}]", leave=False, total=total)):
            event = batch["event"].to(self.device) if self.config.use_events else None
            image = batch["image"].to(self.device) if self.config.use_rgb else None
            depth = batch["depth"].to(self.device)
            mask = batch["mask"].to(self.device)
            intrinsics = self._prepare_intrinsics(batch.get("intrinsics"))

            pred = self.forward(event, image, intrinsics, depth_bounds=(val_cfg["min_depth"], val_cfg["max_depth"]))
            loss_dict = self.compute_losses(
                pred,
                depth,
                mask,
                min_depth=val_cfg["min_depth"],
                max_depth=val_cfg["max_depth"],
            )
            metrics = compute_depth_metrics(
                pred.squeeze(1), depth, mask,
                val_cfg["min_depth"],
                val_cfg["max_depth"],
            )

            for k, v in loss_dict.items():
                loss_accum[k] += v.item()
            for k, v in metrics.items():
                if math.isnan(v):
                    continue
                metrics_accum[k] += v
                metrics_count[k] += 1
            count += 1

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

        if count == 0:
            return

        for k in loss_accum:
            loss_accum[k] /= count
        total_val_loss = 0.0
        for name, weight in self.config.depth_loss_weights.items():
            if name in loss_accum:
                total_val_loss += weight * loss_accum[name]
        self.log_scalar("valid", "", "total_loss", total_val_loss, global_step)
        for name in ("silog", "ms_grad"):
            if name in loss_accum:
                self.log_scalar("valid", "", name, loss_accum[name], global_step)
        for k in metrics_accum:
            if metrics_count[k] > 0:
                metrics_accum[k] /= metrics_count[k]
            else:
                metrics_accum[k] = float("nan")
        for k in self.config.depth_metrics:
            val = metrics_accum.get(k, float("nan"))
            if math.isnan(val):
                continue
            self.log_scalar("valid", "metric", k, val, global_step)

        formatted = ", ".join(
            f"{k}={metrics_accum[k]:.4f}" if not math.isnan(metrics_accum[k]) else f"{k}=nan"
            for k in self.config.depth_metrics
        )
        print(f"[stage {stage_cfg['name']}] validation on {valid_name} @ step {global_step}: {formatted}")

        vis_batch = next(iter(base_loader))
        event_vis = vis_batch["event"].to(self.device) if self.config.use_events else None
        image_vis = vis_batch["image"].to(self.device) if self.config.use_rgb else None
        intr_vis = self._prepare_intrinsics(vis_batch.get("intrinsics"))
        pred_vis = self.forward(event_vis, image_vis, intr_vis, depth_bounds=(val_cfg["min_depth"], val_cfg["max_depth"]))
        self.visualize(
            vis_batch,
            pred_vis,
            global_step,
            split=f"{stage_cfg['name']}_{valid_name}",
            image_stats={"mean": val_cfg["image_mean"], "std": val_cfg["image_std"]} if self.config.use_rgb else None,
            event_stats={"mean": val_cfg["event_mean"], "std": val_cfg["event_std"]} if self.config.use_events else None,
            depth_bounds=(val_cfg["min_depth"], val_cfg["max_depth"]),
        )

    def save(self, global_step: int, stage_index: int, stage_step: int):
        ckpt = {
            "step": global_step,
            "global_step": global_step,
            "stage_index": stage_index,
            "stage_step": stage_step,
            "stage_name": None if stage_index >= len(self.training_stages) else self.training_stages[stage_index]["name"],
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config_class": self.config.__class__.__name__,
        }
        path = self.ckpt_dir / f"checkpoint_{global_step:06d}.pt"
        torch.save(ckpt, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        print(f"Loaded checkpoint from {path}")
        stage_index = ckpt.get("stage_index", 0)
        stage_step = ckpt.get("stage_step", ckpt.get("step", 0))
        global_step = ckpt.get("global_step", ckpt.get("step", 0))
        return int(stage_index), int(stage_step), int(global_step)

    def start(self, resume_path: str | None = None, dry_run: bool = False):
        start_stage_idx = 0
        start_stage_step = 0
        global_step = 0
        if resume_path is not None and os.path.isfile(resume_path):
            start_stage_idx, start_stage_step, global_step = self.load(resume_path, strict=True)

        total_stages = len(self.training_stages)
        if start_stage_idx >= total_stages:
            print("All training stages already completed.")
            return

        for stage_idx in range(start_stage_idx, total_stages):
            stage_cfg = self.training_stages[stage_idx]
            stage_name = stage_cfg["name"]
            stage_steps = stage_cfg["steps"]
            if stage_steps <= 0:
                print(f"Skipping stage '{stage_name}' (no steps).")
                start_stage_step = 0
                continue

            local_step = start_stage_step if stage_idx == start_stage_idx else 0
            if local_step >= stage_steps:
                print(f"Stage '{stage_name}' already completed (step {local_step}/{stage_steps}).")
                start_stage_step = 0
                continue

            print(f"Starting stage '{stage_name}' from step {local_step}/{stage_steps}.")
            loader = stage_cfg["train_loader"]
            train_iter = iter(loader)
            last_validation_step = -1

            while local_step < stage_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(loader)
                    batch = next(train_iter)

                self.active_stage = stage_cfg
                loss_val = self.train_step(batch, global_step, local_step, stage_cfg)
                local_step += 1
                global_step += 1

                if dry_run and local_step >= min(2, stage_steps):
                    for valid_name in stage_cfg["valid_splits"]:
                        self.validate(global_step, stage_cfg, valid_name, max_batches=1)
                    print(f"[dry-run] stage '{stage_name}' loss={loss_val:.4f} at step {global_step}")
                    return

                if not dry_run and global_step % self.config.valid_every == 0:
                    max_batches = 64
                    for valid_name in stage_cfg["valid_splits"]:
                        self.validate(global_step, stage_cfg, valid_name, max_batches=max_batches)
                    last_validation_step = global_step

                if not dry_run and global_step % (self.config.valid_every * 5) == 0:
                    self.save(global_step, stage_idx, local_step)

            if not dry_run:
                if last_validation_step != global_step:
                    for valid_name in stage_cfg["valid_splits"]:
                        self.validate(global_step, stage_cfg, valid_name, max_batches=max_batches)
                self.save(global_step, stage_idx, local_step)

            start_stage_step = 0


def main():
    parser = argparse.ArgumentParser(description="Supervised depth estimation on MVSEC.")
    parser.add_argument("--device", type=str, default=None, help="Override GPU device id (e.g., cuda:0)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming")
    parser.add_argument("--dry-run", action="store_true", help="Run a short sanity check without full training.")
    args = parser.parse_args()

    config = DEPConfig()
    if args.device is not None:
        config.device = args.device
    if args.dry_run:
        config.batch_size = min(config.batch_size, 2)
        config.n_workers = 0

    trainer = DepthEstimator(config)
    trainer.start(resume_path=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
