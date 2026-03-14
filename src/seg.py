import sys
if "dinov2" not in sys.path:
    sys.path.append("dinov2")
if "dinov3" not in sys.path:
    sys.path.append("dinov3")
# --------------------------------deterministic setting-------------------------------- #
import numpy as np
seed = 0
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from torch import nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
g = torch.Generator()
g.manual_seed(seed)
torch.use_deterministic_algorithms(True)
# --------------------------------------------------------------------------------------- #
import itertools
import math
from datetime import datetime
import os
import time
from pathlib import Path
import importlib.util
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import einops
import torchvision
from torch import nn
import torch
import torch.nn.functional as F
from torchinfo import summary
from torchvision.models import swin_t
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_small, vit_base, vit_small_plus
from model import Block, Transformer
from utils import IOU, PixelAccuracy, get_lr, get_param_groups, multiclass_ce_loss, multiclass_dice_loss
try:
    from mem import MEMEncoderConfig, MemEncoder
except ImportError:  # fall back to local src/mem.py when package name clashes
    _mem_spec = importlib.util.spec_from_file_location(
        "seg_mem_module", Path(__file__).resolve().with_name("mem.py")
    )
    _mem_module = importlib.util.module_from_spec(_mem_spec)
    assert _mem_spec and _mem_spec.loader
    sys.modules[_mem_spec.name] = _mem_module
    _mem_spec.loader.exec_module(_mem_module)
    MEMEncoderConfig = _mem_module.MEMEncoderConfig
    MemEncoder = _mem_module.MemEncoder

try:
    from ecddp import ECDDPEncoderConfig, ECDDPEncoder
except ImportError:  # fall back to local src/ecddp.py
    _ec_spec = importlib.util.spec_from_file_location(
        "seg_ecddp_module", Path(__file__).resolve().with_name("ecddp.py")
    )
    _ec_module = importlib.util.module_from_spec(_ec_spec)
    assert _ec_spec and _ec_spec.loader
    sys.modules[_ec_spec.name] = _ec_module
    _ec_spec.loader.exec_module(_ec_module)
    ECDDPEncoderConfig = _ec_module.ECDDPEncoderConfig
    ECDDPEncoder = _ec_module.ECDDPEncoder

from torch.utils.tensorboard import SummaryWriter
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.no_grad()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def _resolve_group_channels(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class BaselineSegHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        out_dim = config.C * (config.P ** 2)
        self.decode = nn.Linear(config.n_embed, out_dim)
        self.post = nn.Sequential(
            nn.Conv2d(self.config.C, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, self.config.C, kernel_size=3, padding=1),
        )

    def forward(self, tokens: torch.Tensor, spatial_hw: tuple[int, int] | None = None) -> torch.Tensor:
        config = self.config
        if spatial_hw is None:
            spatial_hw = (config.H, config.W)
        h_tokens, w_tokens = (spatial_hw[0] // config.P, spatial_hw[1] // config.P)
        if h_tokens * w_tokens != tokens.shape[1]:
            raise ValueError(
                f"Token count {tokens.shape[1]} does not match spatial grid "
                f"{h_tokens}x{w_tokens} derived from {spatial_hw} and patch {config.P}."
            )
        x = self.decode(tokens)
        x = einops.rearrange(
            x,
            "b (l1 l2) (c p1 p2) -> b c (l1 p1) (l2 p2)",
            c=config.C,
            p1=config.P,
            p2=config.P,
            l1=h_tokens,
            l2=w_tokens,
        )
        return self.post(x)


class PyramidSegHead(nn.Module):
    def __init__(self, config, stage_dims):
        super().__init__()
        self.config = config
        self.stage_dims = stage_dims
        self.head_dim = getattr(config, "pyramid_head_dim", 128)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, self.head_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_resolve_group_channels(self.head_dim), self.head_dim),
                    nn.GELU(),
                )
                for in_ch in stage_dims
            ]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_resolve_group_channels(self.head_dim), self.head_dim),
                    nn.GELU(),
                )
                for _ in stage_dims
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.head_dim, config.C, kernel_size=1),
        )

    def forward(self, feats, target_hw: tuple[int, int]) -> torch.Tensor:
        if not feats:
            raise ValueError("PyramidSegHead expects non-empty feature maps.")
        if len(feats) != len(self.stage_dims):
            raise ValueError(f"Expected {len(self.stage_dims)} pyramid features, got {len(feats)}.")
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]
        top = None
        for idx in reversed(range(len(laterals))):
            lateral = laterals[idx]
            if top is None:
                top = lateral
            else:
                top = lateral + F.interpolate(
                    top, size=lateral.shape[-2:], mode="bilinear", align_corners=False
                )
            top = self.smooth_convs[idx](top)
        top = F.interpolate(top, size=target_hw, mode="bilinear", align_corners=False)
        return self.project(top)


def _deterministic_adaptive_avg_pool(x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    """Mimic adaptive_avg_pool2d via deterministic avg_pool2d + replicate padding."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    target_h, target_w = output_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError("output_size must be positive")
    B, C, H, W = x.shape
    pad_h = (target_h - (H % target_h)) % target_h
    pad_w = (target_w - (W % target_w)) % target_w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        H = H + pad_h
        W = W + pad_w
    kernel_h = H // target_h
    kernel_w = W // target_w
    pooled = F.avg_pool2d(
        x,
        kernel_size=(kernel_h, kernel_w),
        stride=(kernel_h, kernel_w),
        ceil_mode=False,
        count_include_pad=True,
    )
    return pooled[..., :target_h, :target_w]

class ConvBNAct(nn.Module):
    """Lightweight Conv-BN-GELU block used inside segmentation heads."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyramidPoolingModule(nn.Module):
    """Pooling pyramid identical to the PSP block used in UPerHead."""

    def __init__(
        self,
        in_channels: int,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        channels: int = 512,
        align_corners: bool = False,
        reduce_input: bool = True,
    ):
        super().__init__()
        self.pool_scales = tuple(int(s) for s in pool_scales)
        self.align_corners = align_corners
        self.input_conv = (
            ConvBNAct(in_channels, channels, kernel_size=1)
            if reduce_input
            else nn.Identity()
        )
        self.ppm_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                )
                for _ in self.pool_scales
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input_conv(x)
        outs = [base]
        for scale, conv in zip(self.pool_scales, self.ppm_convs):
            pooled = _deterministic_adaptive_avg_pool(x, (scale, scale))
            upsampled = F.interpolate(conv(pooled), size=x.shape[-2:], mode="bilinear", align_corners=self.align_corners)
            outs.append(upsampled)
        return torch.cat(outs, dim=1)


class ECDDPUPerHead(nn.Module):
    """Swin feature decoder mirroring the ECDDP segmentation head."""

    def __init__(
        self,
        in_channels: Sequence[int],
        pyramid_channels: int,
        num_classes: int,
        pool_scales: Sequence[int],
        dropout: float = 0.1,
        align_corners: bool = False,
        patch_size: int = 1,
    ):
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("UPer head expects at least two feature levels.")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.channels = int(pyramid_channels)
        self.align_corners = align_corners
        self.patch_size = patch_size
        ppm_in = self.in_channels[-1]
        self.ppm = PyramidPoolingModule(ppm_in, pool_scales, self.channels, align_corners, reduce_input=True)
        ppm_out_channels = self.channels * (len(pool_scales) + 1)
        self.ppm_bottleneck = ConvBNAct(ppm_out_channels, self.channels, kernel_size=3)
        self.lateral_convs = nn.ModuleList(
            [ConvBNAct(in_ch, self.channels, kernel_size=1) for in_ch in self.in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList([ConvBNAct(self.channels, self.channels, kernel_size=3) for _ in self.lateral_convs])
        self.fpn_bottleneck = ConvBNAct(self.channels * len(self.in_channels), self.channels, kernel_size=3)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        if self.patch_size > 1:
            self.classifier = nn.Sequential(
                nn.Conv2d(self.channels, num_classes * patch_size * patch_size, kernel_size=1),
                nn.PixelShuffle(patch_size)
            )
        else:
            self.classifier = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} features, got {len(feats)}")
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats[:-1])]
        top = self.ppm_bottleneck(self.ppm(feats[-1]))
        laterals.append(top)
        for idx in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[idx], size=laterals[idx - 1].shape[-2:], mode="bilinear", align_corners=self.align_corners
            )
            laterals[idx - 1] = laterals[idx - 1] + upsampled
        outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals[:-1])]
        outs.append(laterals[-1])
        target_hw = outs[0].shape[-2:]
        outs = [
            out if out.shape[-2:] == target_hw else F.interpolate(out, size=target_hw, mode="bilinear", align_corners=self.align_corners)
            for out in outs
        ]
        fused = self.fpn_bottleneck(torch.cat(outs, dim=1))
        fused = self.dropout(fused)
        return self.classifier(fused)


class ECDDPFCNAuxHead(nn.Module):
    """Auxiliary FCN head that supervises the penultimate Swin stage."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        layers = [
            ConvBNAct(in_channels, hidden_channels, kernel_size=3),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.block(feat)

class SEG(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pad = CenterPadding(self.config.P)
        encoder_mode = getattr(self.config, "encoder_mode", "ours")
        self.event_backbone = getattr(self.config, "event_backbone", "vit")
        self.encoder_mode = encoder_mode
        self.is_ecddp = self.encoder_mode == "ecddp"
        if self.encoder_mode == "mem":
            mem_cfg = MEMEncoderConfig(
                ckpt_path=self.config.event_encoder_weight,
                image_size=getattr(self.config, "mem_image_size", (self.config.H, self.config.W)),
                patch_size=self.config.P,
                device=self.config.device,
                dtype=torch.float32,
                freeze=self.config.encoder_frozen,
            )
            self.event_encoder = MemEncoder(mem_cfg)
        elif self.encoder_mode == "ecddp":
            ec_cfg = ECDDPEncoderConfig(
                ckpt_path=self.config.event_encoder_weight,
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
                ape=False,
                patch_norm=True,
                device=self.config.device,
                dtype=torch.float32,
                keep_patch_keys=getattr(self.config, "ecddp_keep_patch", False),
                load_teacher=getattr(self.config, "ecddp_load_teacher", False),
                freeze=self.config.encoder_frozen,
            )
            self.event_encoder = ECDDPEncoder(ec_cfg)
        else:
            if self.event_backbone == "swin":
                swin = swin_t()
                self.event_encoder = nn.Sequential(swin.features, swin.norm)
                self.event_encoder.embed_dim = 768
                if getattr(config, "swin_project", False):
                    self.token_proj = nn.Linear(49, 256)
                    if getattr(config, "token_proj_weight", None) is not None:
                        self.token_proj.load_state_dict(config.token_proj_weight)
                        print("*" * 50 + " token_proj loaded")

            else:
                vit_backbone = getattr(self.config, "vit_backbone", "dinov2")
                vit_size = getattr(self.config, "vit", "base")
                if vit_backbone == "dinov3":
                    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16

                    builders = {
                        "small": dinov3_vits16,
                        "base": dinov3_vitb16,
                    }
                    if vit_size not in builders:
                        raise ValueError(f"Unsupported DINOv3 vit size '{vit_size}'.")
                    builder = builders[vit_size]
                    self.event_encoder = builder(pretrained=False)
                else:
                    if vit_size == "small":
                        self.event_encoder = vit_small(
                            patch_size=14,
                            img_size=518,
                            block_chunks=0,
                            init_values=1e-6,
                            num_register_tokens=4,
                        )
                    elif vit_size == "small+":
                        self.event_encoder = vit_small_plus(
                            patch_size=14,
                            img_size=518,
                            block_chunks=0,
                            init_values=1e-6,
                            num_register_tokens=4,
                        )
                    else:
                        self.event_encoder = vit_base(
                            patch_size=14,
                            img_size=518,
                            block_chunks=0,
                            init_values=1e-6,
                            num_register_tokens=4,
                        )
            if getattr(config, "event_encoder_weight", None) is not None:
                self.event_encoder.load_state_dict(config.event_encoder_weight, strict=True)
                print("*" * 50 + " event encoder loaded")
        
        self.dim_proj = nn.Identity()
        if getattr(config, "dim_proj_weight", None) is not None:
            weight = config.dim_proj_weight["weight"]
            self.dim_proj = nn.Linear(weight.shape[1], weight.shape[0])
            self.dim_proj.load_state_dict(config.dim_proj_weight)
            print("*" * 50 + " dim_proj loaded")

        if hasattr(config, "transformer_weight"):
            self.transformer = nn.ModuleDict(dict(
                modality_embed = nn.Embedding(5, config.n_embed),
                pos_embed = nn.Embedding(config.window_size, config.n_embed),
                blocks = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
                norm = nn.LayerNorm(config.n_embed),
            ))
            if config.transformer_weight is not None:
                self.transformer.load_state_dict(config.transformer_weight, strict=True)
                print("*" * 50 + " transformer loaded")
            else:
                print("*" * 50 + " transformer random initialized")
        else:
            self.transformer = nn.Identity()

        self.pyramid_stage_dims = self._default_stage_dims(encoder_mode)
        self.pyramid_head = (
            PyramidSegHead(self.config, self.pyramid_stage_dims)
            if getattr(self.config, "use_pyramid_head", False)
            else None
        )
        self.decoder: nn.Module | None = None
        self.ecddp_head: Optional[ECDDPUPerHead] = None
        self.ecddp_aux_head: Optional[ECDDPFCNAuxHead] = None
        self.ecddp_aux_index: int = int(getattr(self.config, "ecddp_aux_in_index", 2))
        self.ecddp_aux_loss_weight: float = 0.0
        self.use_upernet = getattr(self.config, "use_upernet", False)
        self.ecddp_align_corners: bool = bool(getattr(self.config, "ecddp_align_corners", False))
        default_tta_enable = bool(getattr(self.config, "ecddp_use_tta", False)) if self.is_ecddp else False
        default_tta_scales = (
            tuple(float(s) for s in getattr(self.config, "ecddp_tta_scales", (1.0,)))
            if self.is_ecddp
            else (1.0,)
        )
        default_tta_flip = bool(getattr(self.config, "ecddp_tta_flip", False)) if self.is_ecddp else False
        legacy_enable = getattr(self.config, "use_tta", None)
        legacy_scales = getattr(self.config, "ecddp_tta_scales", None)
        legacy_flip = getattr(self.config, "ecddp_tta_flip", None)
        resolved_enable = getattr(
            self.config,
            "tta_enable",
            legacy_enable if legacy_enable is not None else default_tta_enable,
        )
        resolved_scales = getattr(
            self.config,
            "tta_scales",
            legacy_scales if legacy_scales is not None else default_tta_scales,
        )
        resolved_flip = getattr(
            self.config,
            "tta_flip",
            legacy_flip if legacy_flip is not None else default_tta_flip,
        )
        self.tta_enable: bool = bool(resolved_enable)
        self.tta_scales: Tuple[float, ...] = tuple(float(s) for s in resolved_scales)
        self.tta_flip: bool = bool(resolved_flip)
        if encoder_mode == "ecddp":
            self.pyramid_head = None  # override any legacy flag
            self.ecddp_aux_index = max(0, min(self.ecddp_aux_index, len(self.pyramid_stage_dims) - 1))
            self.ecddp_aux_loss_weight = float(getattr(self.config, "ecddp_aux_loss_weight", 0.4))
            self.decoder = None
            self.ecddp_head = ECDDPUPerHead(
                in_channels=self.pyramid_stage_dims,
                pyramid_channels=getattr(self.config, "ecddp_decoder_channels", 512),
                num_classes=self.config.C,
                pool_scales=getattr(self.config, "ecddp_pool_scales", (1, 2, 3, 6)),
                dropout=getattr(self.config, "ecddp_decoder_dropout", 0.1),
                align_corners=self.ecddp_align_corners,
            )
            aux_hidden = getattr(self.config, "ecddp_aux_channels", 256)
            self.ecddp_aux_head = ECDDPFCNAuxHead(
                in_channels=self.pyramid_stage_dims[self.ecddp_aux_index],
                hidden_channels=aux_hidden,
                num_classes=self.config.C,
                dropout=getattr(self.config, "ecddp_aux_dropout", 0.1),
            )
        elif self.use_upernet:
            self.decoder = None
            self.ecddp_aux_loss_weight = float(getattr(self.config, "ecddp_aux_loss_weight", 0.4))
            self.ecddp_head = ECDDPUPerHead(
                in_channels=self.pyramid_stage_dims,
                pyramid_channels=getattr(self.config, "ecddp_decoder_channels", 512),
                num_classes=self.config.C,
                pool_scales=getattr(self.config, "ecddp_pool_scales", (1, 2, 3, 6)),
                dropout=getattr(self.config, "ecddp_decoder_dropout", 0.1),
                align_corners=self.ecddp_align_corners,
                patch_size=self.config.P if not self.is_ecddp else 1,
            )
            if getattr(self.config, "use_aux_head", False):
                aux_hidden = getattr(self.config, "ecddp_aux_channels", 256)
                self.ecddp_aux_head = ECDDPFCNAuxHead(
                    in_channels=self.pyramid_stage_dims[self.ecddp_aux_index],
                    hidden_channels=aux_hidden,
                    num_classes=self.config.C,
                    dropout=getattr(self.config, "ecddp_aux_dropout", 0.1),
                )
        else:
            self.decoder = BaselineSegHead(self.config)
        self._pyramid_feats: list[torch.Tensor] | None = None
        self._encoder_stage_feats: Optional[List[torch.Tensor]] = None
        self._loss_components: dict[str, float] = {}
        self._encoder_hw: Tuple[int, int] = (self.config.H, self.config.W)
        
        # training related
        self.train_dataloader       = torch.utils.data.DataLoader(self.config.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dataloader       = torch.utils.data.DataLoader(self.config.valid_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.amp = torch.amp.autocast(device_type = "cuda")
        self.scaler = torch.amp.GradScaler(device = "cuda")
        self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd, self.config.encoder_lr_mult, self.config.transformer_lr_mult))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.iou = IOU(num_classes=self.config.C, ignore_index=self.config.ignore_index, device=self.config.device)
        self.acc = PixelAccuracy(num_classes=self.config.C, ignore_index=self.config.ignore_index, device=self.config.device)
        self.writer = None
        if not getattr(self.config, "eval_only", False):
            os.makedirs("src/runs", exist_ok=True)
            self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_seg")
        if self.config.encoder_frozen:
            print("freeze event encoder")
            for name, param in self.named_parameters():
                if "decoder" not in name and "transformer" not in name:
                    param.requires_grad = False
        self._maybe_load_eval_checkpoint()
    
    def _default_stage_dims(self, encoder_mode: str):
        if encoder_mode == "ecddp":
            base = getattr(self.config, "ecddp_embed_dim", 96)
            depths = len(getattr(self.config, "ecddp_depths", (2, 2, 6, 2)))
            return [base * (2 ** i) for i in range(depths)]
        if hasattr(self.config, "out_indices"):
            return [self.config.n_embed] * len(self.config.out_indices)
        return [self.config.n_embed] * 4

    def _tokens_to_pyramid(self, tokens: torch.Tensor):
        if self.pyramid_head is None and self.ecddp_head is None:
            return None
        B, N, C = tokens.shape
        h = max(1, self._encoder_hw[0] // self.config.P)
        w = max(1, self._encoder_hw[1] // self.config.P)
        feat = tokens.transpose(1, 2).reshape(B, C, h, w)
        feats = [feat]
        for _ in range(1, len(self.pyramid_stage_dims)):
            prev = feats[-1]
            if min(prev.shape[-2], prev.shape[-1]) <= 1:
                feats.append(prev)
            else:
                feats.append(F.avg_pool2d(prev, kernel_size=2, stride=2))
        return feats

    def _maybe_set_pyramid_feats(self, *, feats=None, tokens=None):
        if self.pyramid_head is None and self.ecddp_head is None:
            self._pyramid_feats = None
            return
        if feats is not None and len(feats) > 0:
            feat_list = list(feats)
            expected = len(self.pyramid_stage_dims)
            if len(feat_list) > expected:
                feat_list = feat_list[-expected:]
            while len(feat_list) < expected:
                feat_list.append(feat_list[-1])
            self._pyramid_feats = feat_list
            return
        if tokens is not None:
            self._pyramid_feats = self._tokens_to_pyramid(tokens)
        else:
            self._pyramid_feats = None
    
    def _maybe_load_eval_checkpoint(self):
        if not getattr(self.config, "eval_only", False):
            return
        ckpt_path = getattr(self.config, "eval_checkpoint", None)
        if not ckpt_path:
            raise ValueError("SegConfig.eval_only=True requires eval_checkpoint to be set to a .pth file.")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Eval checkpoint not found: {ckpt_path}")
        print(f"Loading segmentation checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.config.device)
        if "event_encoder" in checkpoint:
            self.event_encoder.load_state_dict(checkpoint["event_encoder"], strict=True)
        if "transformer" in checkpoint and not isinstance(self.transformer, nn.Identity):
            self.transformer.load_state_dict(checkpoint["transformer"], strict=True)
        decoder_state = checkpoint.get("decoder")
        if self.ecddp_head is not None:
            if self.ecddp_head is not None and isinstance(decoder_state, dict):
                main_state = decoder_state.get("main")
                if main_state is not None:
                    self.ecddp_head.load_state_dict(main_state, strict=True)
            if self.ecddp_aux_head is not None and isinstance(decoder_state, dict):
                aux_state = decoder_state.get("aux")
                if aux_state is not None:
                    self.ecddp_aux_head.load_state_dict(aux_state, strict=True)
        else:
            if self.decoder is not None and decoder_state is not None:
                self.decoder.load_state_dict(decoder_state, strict=True)
            if self.pyramid_head is not None and "pyramid_head" in checkpoint:
                self.pyramid_head.load_state_dict(checkpoint["pyramid_head"], strict=True)
        print("Checkpoint loaded; running in eval-only mode.")

    def _render_event_tensor(self, event: torch.Tensor) -> torch.Tensor:
        """Project multi-channel event tensors to a 3-channel visualization."""
        if event.ndim != 4:
            raise ValueError(f"Expected event tensor of shape [B, C, H, W], got {event.shape}")
        _, c, _, _ = event.shape
        if c >= 3:
            vis = event[:, :3]
        else:
            repeat = int(math.ceil(3 / c))
            vis = event.repeat(1, repeat, 1, 1)[:, :3]
        vis = vis - vis.amin(dim=(2, 3), keepdim=True)
        denom = vis.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        vis = vis / denom
        return vis
            
    def forward_encoder(self, x):
        self._pyramid_feats = None
        self._encoder_stage_feats = None
        x = self.pad(x)
        self._encoder_hw = (int(x.shape[-2]), int(x.shape[-1]))
        if self.encoder_mode == "mem":
            feats = self.event_encoder(x, return_tokens=True)
            self._maybe_set_pyramid_feats(tokens=feats.get("patch_tokens"))
            return feats["patch_tokens"]
        elif self.encoder_mode == "ecddp":
            feats = self.event_encoder(x, return_tokens=True)
            pyramid_feats = feats.get("pyramid_feats")
            if pyramid_feats:
                self._encoder_stage_feats = list(pyramid_feats)
            self._maybe_set_pyramid_feats(feats=feats.get("pyramid_feats"), tokens=feats.get("patch_tokens"))
            return feats["patch_tokens"]
        
        if self.event_backbone == "swin":
            # Swin output: [B, H, W, C] -> Flatten to [B, N, C]
            out = self.event_encoder(x)
            B, H, W, C = out.shape
            tokens = out.view(B, -1, C)
            
            if getattr(self.config, "swin_project", False) and hasattr(self, "token_proj"):
                # Ensure input to token_proj is 49 tokens (7x7 grid)
                if tokens.shape[1] != 49:
                    # Reshape to [B, C, H, W] for interpolation
                    feat = out.permute(0, 3, 1, 2) 
                    feat = F.interpolate(feat, size=(7, 7), mode='bilinear', align_corners=False)
                    tokens = feat.flatten(2).transpose(1, 2) # [B, 49, C]
                # Project: [B, 49, C] -> [B, 256, C]
                tokens = self.token_proj(tokens.transpose(1, 2)).transpose(1, 2)
                self._encoder_hw = (224, 224) # Force decoder to treat this as 224x224 (16x16 patches)
            self._maybe_set_pyramid_feats(tokens=tokens)
            return tokens
        else:
            if hasattr(self.config, "out_indices"):
                # Multi-level feature extraction for UPerNet
                outs = self.event_encoder.get_intermediate_layers(x, n=self.config.out_indices, reshape=True, norm=True)
                # outs is a list of [B, C, H, W] tensors
                processed_outs = []
                for feat in outs:
                    # Optional projection if backbone dim != n_embed
                    if not isinstance(self.dim_proj, nn.Identity):
                        # [B, C, H, W] -> [B, H*W, C] -> proj -> [B, C, H, W]
                        b, c, h, w = feat.shape
                        feat_flat = feat.flatten(2).transpose(1, 2)
                        feat_flat = self.dim_proj(feat_flat)
                        feat = feat_flat.transpose(1, 2).reshape(b, -1, h, w)
                    processed_outs.append(feat)
                
                self._pyramid_feats = processed_outs
                # Use the last feature map (flattened) as the token representation
                last_feat = processed_outs[-1]
                tokens = last_feat.flatten(2).transpose(1, 2)
                return tokens
            else:
                tokens = self.event_encoder.forward_features(x)["x_norm_patchtokens"]
                tokens = self.dim_proj(tokens)
                self._maybe_set_pyramid_feats(tokens=tokens)
                return tokens
    
    def forward_transformer(self, x):
        B, T, C = x.shape
        ids = torch.ones(B, T, dtype=torch.int64, device=x.device) * 2  # 1 for image, 2 for event
        modality_emb = self.transformer.modality_embed(ids)
        
        # Interpolate positional embeddings if resolution changes (e.g. during TTA)
        H_grid = self._encoder_hw[0] // self.config.P
        W_grid = self._encoder_hw[1] // self.config.P
        H_orig = self.config.H // self.config.P
        W_orig = self.config.W // self.config.P

        if (H_grid != H_orig or W_grid != W_orig) and (H_grid * W_grid == T):
            N_orig = H_orig * W_orig
            pos_weight = self.transformer.pos_embed.weight[:N_orig]
            pos_weight = pos_weight.transpose(0, 1).reshape(1, C, H_orig, W_orig)
            pos_emb = F.interpolate(pos_weight, size=(H_grid, W_grid), mode='bicubic', align_corners=False)
            pos_emb = pos_emb.flatten(2).transpose(1, 2).squeeze(0)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=x.device).clamp(max=self.config.window_size - 1)
            pos_emb = self.transformer.pos_embed(pos)

        x_ = x + pos_emb + modality_emb
        for i, blk in enumerate(self.transformer.blocks):
            x_ = blk(x_)
        x_ = self.transformer.norm(x_)
        return x_

    def forward_decoder(self, x):
        if self.ecddp_head is not None:
            if self.is_ecddp:
                if self._encoder_stage_feats is None:
                    raise RuntimeError("ECDDP decoder requires encoder stage features.")
                stage_feats = list(self._encoder_stage_feats)
            else:
                stage_feats = list(self._pyramid_feats) if self._pyramid_feats is not None else None
            
            if stage_feats is None:
                raise RuntimeError("Decoder requires feature maps (pyramid or stage feats).")

            main_logits = self.ecddp_head(stage_feats)
            main_logits = F.interpolate(
                main_logits,
                size=(self.config.H, self.config.W),
                mode="bilinear",
                align_corners=self.ecddp_align_corners,
            )
            aux_logits = None
            if self.ecddp_aux_head is not None:
                aux_feat = stage_feats[self.ecddp_aux_index]
                aux_logits = self.ecddp_aux_head(aux_feat)
                aux_logits = F.interpolate(
                    aux_logits,
                    size=(self.config.H, self.config.W),
                    mode="bilinear",
                    align_corners=self.ecddp_align_corners,
                )
            return (main_logits, aux_logits) if aux_logits is not None else main_logits
        if self.decoder is None:
            raise RuntimeError("Segmentation decoder is not initialized.")
        logits = self.decoder(x, self._encoder_hw)
        if self.pyramid_head is not None and self._pyramid_feats is not None:
            pyramid_logits = self.pyramid_head(self._pyramid_feats, self._encoder_hw)
            logits = logits + pyramid_logits
        
        if logits.shape[-2:] != (self.config.H, self.config.W):
            logits = F.interpolate(
                logits, size=(self.config.H, self.config.W), mode="bilinear", align_corners=False
            )
            
        return logits

    def _forward_logits(self, x: torch.Tensor):
        tokens = self.forward_encoder(x)
        if not isinstance(self.transformer, nn.Identity):
            tokens = self.forward_transformer(tokens)
        return self.forward_decoder(tokens)

    def _compute_single_seg_loss(self, logits: torch.Tensor, target: torch.Tensor):
        ce_loss = multiclass_ce_loss(logits, target, ignore_index=self.config.ignore_index)
        dice_loss = multiclass_dice_loss(logits, target, ignore_index=self.config.ignore_index)
        return ce_loss + dice_loss, ce_loss, dice_loss
    
    def forward_loss(self, pred, target):
        """
        pred    : logits tensor or (main, aux) tuple
        target  : [N, H, W]
        """
        if target is None:
            self._loss_components = {}
            return None
        if isinstance(pred, tuple):
            main_logits, aux_logits = pred
        else:
            main_logits, aux_logits = pred, None
        main_loss, main_ce, main_dice = self._compute_single_seg_loss(main_logits, target)
        total_loss = main_loss
        components: dict[str, float] = {
            "main_total": float(main_loss.detach().item()),
            "main_ce": float(main_ce.detach().item()),
            "main_dice": float(main_dice.detach().item()),
        }
        if aux_logits is not None:
            aux_loss, aux_ce, aux_dice = self._compute_single_seg_loss(aux_logits, target)
            total_loss = total_loss + self.ecddp_aux_loss_weight * aux_loss
            components.update(
                {
                    "aux_total": float(aux_loss.detach().item()),
                    "aux_ce": float(aux_ce.detach().item()),
                    "aux_dice": float(aux_dice.detach().item()),
                }
            )
        self._loss_components = components
        return total_loss

    def forward(self, x, y=None):
        pred = self._forward_logits(x)
        loss = self.forward_loss(pred, y) if y is not None else None
        return pred, loss

    @torch.no_grad()
    def _predict_with_tta(self, x: torch.Tensor) -> torch.Tensor:
        base_hw = (self.config.H, self.config.W)
        stride = self.config.P
        accum = None
        variants = 0
        flip_options = [False, True] if self.tta_flip else [False]
        scales = tuple(float(s) for s in self.tta_scales) if self.tta_scales else (1.0,)
        if self.encoder_mode == "mem":
            # MEM ViT expects a fixed input resolution; drop non-1.0 scales to avoid shape asserts.
            valid_scales = tuple(s for s in scales if abs(s - 1.0) < 1e-6)
            if len(valid_scales) != len(scales) and not getattr(self, "_mem_tta_warning_shown", False):
                print("MEM encoder only supports scale=1.0 during TTA; skipping additional scales.")
                self._mem_tta_warning_shown = True
            scales = valid_scales or (1.0,)
        for scale in scales:
            scale = float(scale)
            if abs(scale - 1.0) < 1e-6:
                scaled = x
            else:
                target_h = max(1, int(round(base_hw[0] * scale)))
                target_w = max(1, int(round(base_hw[1] * scale)))
                target_h = int(math.ceil(target_h / stride) * stride)
                target_w = int(math.ceil(target_w / stride) * stride)
                scaled = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            for flip in flip_options:
                aug = torch.flip(scaled, dims=[-1]) if flip else scaled
                preds = self._forward_logits(aug)
                main_logits = preds[0] if isinstance(preds, tuple) else preds
                if flip:
                    main_logits = torch.flip(main_logits, dims=[-1])
                if main_logits.shape[-2:] != base_hw:
                    main_logits = F.interpolate(
                        main_logits,
                        size=base_hw,
                        mode="bilinear",
                        align_corners=self.ecddp_align_corners,
                    )
                accum = main_logits if accum is None else accum + main_logits
                variants += 1
        if variants == 0:
            raise RuntimeError("TTA generated zero variants.")
        return accum / variants

    @torch.no_grad()
    def visualize(self, image, pred, labels, name, alpha=0.7, n=8):
        '''logits and labels.shape: [batch_size, height, width]'''
        # full‑res original image (unnormalize)
        if self.config.type == "EL":
            if len(self.config.ME) == image.shape[1]:
                M = torch.tensor(self.config.ME, device=self.config.device)[None, :, None, None]
                S = torch.tensor(self.config.SE, device=self.config.device)[None, :, None, None]
                orig = image * S + M
            else:
                orig = self._render_event_tensor(image)
        else:
            M = torch.tensor(self.config.MI, device=self.config.device)[None, :, None, None]
            S = torch.tensor(self.config.SI, device=self.config.device)[None, :, None, None]
            orig = image * S + M       # [B, 3, H, W]
        # predicted & ground‑truth color maps
        pred_rgb  = self.config.palette[pred].permute(0, 3, 1, 2) / 255.  # [B, 3, H, W]
        label_rgb = self.config.palette[labels].permute(0, 3, 1, 2) / 255.
        # mask out ignored pixels
        mask = (labels == self.config.ignore_index).unsqueeze(1)   # [B, 1, H, W]
        # wherever mask == True, use orig instead of pred_rgb
        pred_rgb = torch.where(mask, orig, pred_rgb)
        # blended overlay
        blend = orig.mul(1 - alpha) + pred_rgb.mul(alpha)
        # build a grid: [orig | pred | blend | gt]
        rendered = torchvision.utils.make_grid(
            torch.cat([orig[:n], pred_rgb[:n], blend[:n], label_rgb[:n]]),
            nrow=8
        )
        torchvision.utils.save_image(rendered, f"src/{name}.png")
    
    def train_step(self, x, y, global_step):
        x, y = x.to(self.config.device), y.to(self.config.device)
        t0 = time.time()
        self.train()
        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]
            
        with self.amp:
            pred, loss = self.forward(x, y)
        main_pred = pred[0] if isinstance(pred, tuple) else pred
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        t1 = time.time()
        if self.writer is not None and (global_step + 1) % self.config.log_every == 0:
            self.writer.add_scalar("loss/train", loss.item(), global_step + 1)
            for key, value in self._loss_components.items():
                self.writer.add_scalar(f"loss/{key}", value, global_step + 1)
            dt = (t1 - t0)
            self.visualize(x, torch.argmax(main_pred, dim=1), y, name="vis_seg_train")
            num_tokens_per_secend =  self.config.batch_size * self.config.n_tokens_per_image / dt
            print(f"step: {global_step + 1}, lr: {current_lr :.8f}, loss: {loss.item() :.4f}, grad_norm: {grad_norm:.4f}, input: {x.shape}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")
        
    def start(self):
        # self.validate(0)
        if getattr(self.config, "eval_only", False):
            print("Segmentation eval-only mode: skipping training loop.")
            return
        train_iter = iter(self.train_dataloader)

        for step in range(self.config.steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                x, y = next(train_iter)

            if step == 0:
                summary(self, input_data=(x, y), device=self.config.device, depth=2)
            self.train_step(x, y, step)

            if (step + 1) % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                self.validate(step)
    
    @ torch.no_grad()
    def validate(self, step):
        self.eval()
        valid_loss = 0.0
        self.iou.reset()
        self.acc.reset()
        last_batch = None
        for i, (x, y) in enumerate(tqdm(self.valid_dataloader)):
            if i > 32 and not getattr(self.config, "eval_only", False):
                break
            x, y = x.to(self.config.device), y.to(self.config.device)
            preds, loss = self.forward(x, y)
            valid_loss += loss.item()
            main_logits = preds[0] if isinstance(preds, tuple) else preds
            eval_logits = main_logits
            if self.tta_enable:
                eval_logits = self._predict_with_tta(x)
            pred_labels = torch.argmax(eval_logits, dim=1)
            self.iou.update(pred_labels, y)
            self.acc.update(pred_labels, y)
            last_batch = (x, pred_labels, y)
        if last_batch is not None:
            self.visualize(*last_batch, name="vis_seg_valid")
        ious = self.iou.compute()
        acc = self.acc.compute()
        miou = ious.mean().item()
        macc = acc.mean().item()

        valid_loss /= (i+1)
        
        print(f"step: {step}, valid loss: {valid_loss:.4f}")
        print("ious:", [round(item, 4) for item in ious.tolist()])
        print("acc:", [round(item, 4) for item in acc.tolist()])
        print(f"mIoU: {miou:.4f}")
        print(f"mAcc: {macc:.4f}")
        
        if self.writer is not None:
            self.writer.add_scalar("loss/valid", valid_loss, step)
            self.writer.add_scalar("mIoU/valid", miou, step)
            self.writer.add_scalar("acc/valid", macc, step)
        
        if getattr(self.config, "eval_only", False):
            return
        if self.ecddp_head is not None:
            decoder_state = {}
            if self.ecddp_head is not None:
                decoder_state["main"] = self.ecddp_head.state_dict()
            if self.ecddp_aux_head is not None:
                decoder_state["aux"] = self.ecddp_aux_head.state_dict()
        else:
            decoder_state = self.decoder.state_dict()
        param_dict = {
            "event_encoder": self.event_encoder.state_dict(),
            "transformer": self.transformer.state_dict(),
            "decoder": decoder_state,
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        if not self.is_ecddp and self.pyramid_head is not None:
            param_dict["pyramid_head"] = self.pyramid_head.state_dict()
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_seg"
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step + 1}_{valid_loss:.4f}.pt"))
        
if __name__ == "__main__":
    from config import SegConfig
    config = SegConfig()
    mae = SEG(config).to(config.device)
    mae.start()
    print("Training completed.")
