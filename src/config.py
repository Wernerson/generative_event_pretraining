import copy
import math
import os
import torch

from dataset import (
    DepthAdditiveGaussianNoise,
    DepthClamp,
    DepthColorJitter,
    DepthEnsureContiguous,
    DepthNormalize,
    DepthRandomGamma,
    DepthRandomHorizontalFlip,
    DepthRandomResizedCrop,
    DepthResize,
    DepthToTensor,
    DepthTransformCompose,
    DSECOpticalDataset,
    DDD17PairedDataset,
    DDD17SegmentDataset,
    EventDataset,
    ImageDataset,
    NIMAClsDataset,
    NIMAPairedDataset,
    RandomResizedCrop,
    RandomSwapEventRedBlue,
    SCAPEPairedDataset,
    DSECSegmentDataset,
    DSECSegmentSequenceDataset,
    DSECECDDPEventDataset,
    SequenceDataset,
    SequencePairedProcessor,
    SequenceToTensor,
    SequenceNormalize,
    SequenceRandomSwapEventRedBlue,
    SequenceRandomHorizontalFlip,
    SequenceResizeKeepRatio,
    SequencePadToMinSide,
    SequenceRandomCrop,
    SequenceCenterCrop,
    ToTensor,
    CenterCrop,
    DSECPairedDataset,
    PairedProcessor,
    Normalize,
    PadToMinSide,
    RandomCrop,
    RandomHorizontalFlip,
    ResizeKeepRatio,
    EnsureTensorPair,
    EventTensorJitter,
    RandomShiftScaleRotateTensor,
    BDDPairedDataset,
    NCALClsDataset,
    MVSECDepthDataset,
    MVSECECDDPDepthDataset,
    EventScapeDepthDataset,
)
from torchvision import transforms

from utils import compute_mean_std

DINOV3_CKPT_PATHS = {
    "small": "/data/storage/jianwen/cache/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "/data/storage/jianwen/cache/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

class Config():
    def __init__(self):
        self.P       = 14
        self.DSEC_ME = [0.8993729784963826, 0.7969581014619264, 0.8928228776286392]
        self.DSEC_SE = [0.22043360769748688, 0.2921656668186188, 0.22049927711486816]
        self.DSEC_MI = [0.23862534365611895, 0.24712072838418375, 0.2574492927024542]
        self.DSEC_SI = [0.2195923498258092, 0.2361895432347472, 0.2633142601113288]

        self.NIMA_ME = [0.9673029496145361, 0.929740832760733, 0.9624378831461544]
        self.NIMA_SE = [0.12036860792540319, 0.1674319634709885, 0.12649585555644863]
        self.NIMA_MI = [0.4802686970882532, 0.45750728990737405, 0.40818174243273203]
        self.NIMA_SI = [0.28073999251715753, 0.2736791173289334, 0.28782502739532]

        self.NCAL_ME = [0.9235784026671774, 0.8068846090902834, 0.8833083972837024]
        self.NCAL_SE = [0.14994221547458192, 0.22561278466801965, 0.16587210234612618]
        self.NCAL_MI = [0.5298536242842665, 0.5104335180603393, 0.4849997112351298]
        self.NCAL_SI = [0.32870640022717573, 0.32238194064808956, 0.33288269944617854]

        self.SCAP_ME = [0.9888106104297153, 0.9747728761936781, 0.9859595939484498]
        self.SCAP_SE = [0.06632214632296055, 0.09744895769725151, 0.0735958979181792]
        self.SCAP_MI = [0.3691855686450492, 0.372362750445305, 0.38055244521714615]
        self.SCAP_SI = [0.21236170116948833, 0.20754919382931097, 0.21394057988512902]

        self.BDDD_ME = [0.98062167, 0.96161081, 0.98098914]
        self.BDDD_SE = [0.08805939, 0.12125193, 0.08766055]
        self.BDDD_MI = [0.39277547, 0.43453864, 0.44106239]
        self.BDDD_SI = [0.24503397, 0.25585564, 0.26836405]

        self.DD17_ME = [0.9635178446769714, 0.9372559189796448, 0.9606495499610901]
        self.DD17_SE = [0.10666034370660782, 0.14702685177326202, 0.10695996135473251]
        self.DD17_MI = [0.37694597244262695, 0.37694597244262695, 0.37694597244262695]
        self.DD17_SI = [0.2800583243370056, 0.2800583243370056, 0.2800583243370056]

        self.MVSEC_EVENT_ME = [0.8993729784963826, 0.7969581014619264, 0.8928228776286392]
        self.MVSEC_EVENT_SE = [0.22043360769748688, 0.2921656668186188, 0.22049927711486816]
        self.MVSEC_IMAGE_ME = [0.17496666312217712, 0.17496666312217712, 0.17496666312217712]
        self.MVSEC_IMAGE_SE = [0.28217577934265137, 0.28217577934265137, 0.28217577934265137]


class MAEConfig(Config):
    def __init__(self):
        super().__init__()
        self.device                 = "cuda:0"
        self.n_embed                = 384
        self.n_head                 = 6
        self.n_layer                = 8
        self.event_encoder_weight   = None
        self.encoder_lr_mult        = 0.01

        self.batch_size             = 8
        self.n_workers              = 8
        self.lr                     = 5e-4
        self.wd                     = 1e-5
        self.min_lr                 = 1e-6
        self.warmup_steps           = 100
        self.steps                  = 100000
        self.log_every              = 100
        self.valid_every            = 1000

        self.mask_ratio             = 0.75
        self.norm_pix_loss          = True

        self.H, self.W              = 224, 322
        self.scale_range            = (0.5, 2.0)
        if encoder_mode == "ecddp":
            self.scale_range        = (0.7, 1.3)
        self.n_tokens_per_image     = (self.H // self.P) * (self.W // self.P)
        self.type                   = "EI"
        self.train_preprocessors    = PairedProcessor([
                                            ToTensor(self.origi_H, type=self.type),
                                            Normalize(self.ME, self.SE, self.MI, self.SI, type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            ResizeKeepRatio(short_side_target_size=self.H, scale_range=self.scale_range, type=self.type),
                                            PadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=0),
                                            RandomCrop(crop_size=(self.H, self.W), type=self.type),
                                            ])
        self.valid_preprocessors    = PairedProcessor([
                                                    ToTensor(self.origi_H, type=self.type),
                                                    Normalize(self.ME, self.SE, self.MI, self.SI, type=self.type),
                                                    ResizeKeepRatio(short_side_target_size=self.H, type=self.type),
                                                    CenterCrop((self.H, self.W)),
                                                    ])
        self.train_dataset          = DSECPairedDataset(root_dir = "/data/storage/jianwen/DSEC", split="train", transform=self.train_preprocessors)
        self.valid_dataset          = DSECPairedDataset(root_dir = "/data/storage/jianwen/DSEC", split="test", transform=self.valid_preprocessors)
        self.train_dataloader       = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=False)
        self.valid_dataloader       = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=False)
    
class SegConfig(Config):
    """
    Top 类别 (class, count, ratio):
    1: 936957007 (68.2417%)
    3: 209452476 (15.2551%)
    0: 148605866 (10.8235%)
    5: 66795293 (4.8649%)
    2: 9453787 (0.6886%)
    4: 1732771 (0.1262%)
    """

    def __init__(self):
        super().__init__()
        self.encoder_mode = "ours"  # set to "ours" / "mem" / "ecddp" manually
        self.event_backbone = "vit" # ["vit", "swin"]
        self.swin_project   = False

        self._init_common_options()
        self._init_mem_defaults()
        self._init_ecddp_defaults()
        self._apply_mode_overrides()
        self._init_palette()
        self._build_preprocessors()
        self._build_datasets()

        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"valid dataset size: {len(self.valid_dataset)}")

    # ------------------------------------------------------------------ #
    # initialization helpers
    # ------------------------------------------------------------------ #

    def _init_common_options(self):
        self.device = "cuda:1"
        self.data = "event"
        self.type = "EL" if self.data == "event" else "IL"
        self.dataset = "dsec"  # ["dsec", "ddd17"]
        self.eval_only = False
        # self.eval_checkpoint = "/data/storage/jianwen/cache/ckpts/2025-11-14-02:04_seg/epoch8000_0.8904.pt"
        self.eval_checkpoint = None
        self.manual_encoder_weight_path = "/data/storage/jianwen/cache/ckpt_matters/gra_nima_16x.pt"

        self.vit = "small"
        self.vit_backbone = "dinov2"  # ["dinov2", "dinov3"]
        self.dinov3_ckpt_paths = dict(DINOV3_CKPT_PATHS)

        self.encoder_frozen = False
        self.lr = 5e-4
        self.wd = 1e-5
        self.transformer_lr_mult = 0.1
        self.encoder_lr_mult = 0.01
        self.batch_size = 16
        self.n_workers = 8
        self.min_lr = 0.0
        self.warmup_steps = 100

        self.window_size = 4096
        self.n_embed = 384
        self.n_head = 6
        self.n_layer = 12
        self.out_indices = [2, 5, 8, 11]

        self.use_pyramid_head = False
        self.pyramid_head_dim = 128
        self.scale_range = (0.5, 2.0)
        self.cat_max_ratio = 0.75
        self.use_upernet = True
        self.use_aux_head = True

        # shared test-time augmentation defaults
        self.tta_enable = False
        self.tta_scales = (0.8, 1.0, 1.2)
        self.tta_flip = False

        if self.dataset == "dsec":
            self.H, self.W = (448, 644)
            if self.vit_backbone == "dinov3":
                self.H, self.W = (448, 640)
            self.origi_H, self.origi_W = (440, 640)
            self.C = 11
            self.ignore_index = 11
            self.ME, self.SE, self.MI, self.SI = self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI
            self.steps = 40000
        else:
            self.H, self.W = (392, 686)
            self.origi_H, self.origi_W = 200, None
            self.C = 6
            self.ignore_index = 6
            self.ME, self.SE, self.MI, self.SI = self.DD17_ME, self.DD17_SE, self.DD17_MI, self.DD17_SI
            self.steps = 4000
        self.log_every = self.steps // 100
        self.valid_every = self.steps // 10

    def _init_mem_defaults(self):
        self.mem_image_size = (448, 640)

    def _init_ecddp_defaults(self):
        self.ecddp_image_size = (448, 640)
        self.ecddp_in_chans = 3
        self.ecddp_patch_size = 4
        self.ecddp_depths = (2, 2, 6, 2)
        self.ecddp_num_heads = (3, 6, 12, 24)
        self.ecddp_embed_dim = 96
        self.ecddp_window_size = 7
        self.ecddp_mlp_ratio = 4.0
        self.ecddp_drop_rate = 0.0
        self.ecddp_attn_drop_rate = 0.0
        self.ecddp_drop_path_rate = 0.0
        self.ecddp_keep_patch = False
        self.ecddp_load_teacher = False
        self.ecddp_pad_extra = (32, 32)
        self.ecddp_tensor_subdir = "eventTensor_ecddp"
        self.ecddp_tensor_exts = (".pt", ".npz", ".npy")
        self.ecddp_decoder_channels = 416
        self.ecddp_decoder_dropout = 0.1
        self.ecddp_pool_scales = (1, 2, 3, 6)
        self.ecddp_aux_in_index = 2
        self.ecddp_aux_channels = 256
        self.ecddp_aux_dropout = 0.1
        self.ecddp_aux_loss_weight = 0.4
        self.ecddp_align_corners = False

    def _apply_mode_overrides(self):
        if self.encoder_mode == "mem":
            self.P = 16
            self.event_encoder_weight = "/data/storage/jianwen/nimagenet-pt-checkpoint-74.pth"
            self.encoder_lr_mult = 0.1
            self.batch_size = 16
            self._align_spatial_to_stride(self.P)
            self.mem_image_size = (self.H, self.W)
            self.steps = 8000
        elif self.encoder_mode == "ecddp":
            self.P = self.ecddp_patch_size * (2 ** (len(self.ecddp_depths) - 1))
            self.event_encoder_weight = "/data/storage/jianwen/pr.pt"
            self.encoder_lr_mult = 0.2
            self.batch_size = 8
            self.scale_range = (0.7, 1.3)
            self.steps = 50000
            if self.dataset == "dsec":
                self.H, self.W = (448, 640)
                self.origi_H, self.origi_W = (448, 640)
            self._align_spatial_to_stride(self.P)
            self.ecddp_image_size = (self.H, self.W)
            self.tta_enable = True
            self.tta_scales = (1.0, 1.5, 2.0)
            self.tta_flip = True
        else:
            vit_backbone = getattr(self, "vit_backbone", "dinov2")
            self.vit = getattr(self, "vit", "base")
            
            if self.event_backbone == "swin":
                ckpt = torch.load(
                    "/data/storage/jianwen/cache/ckpts/2026-01-23-16:08_gpt/epoch25000_0.1571.pt",
                    map_location="cpu",
                )
                self.event_encoder_weight = ckpt.get("event_encoder", None)
                self.token_proj_weight = ckpt.get("token_proj", None)
                self.dim_proj_weight = ckpt.get("dim_proj", None)
                self.transformer_weight = None

                if self.swin_project:
                    self.P = 14
                    self.n_embed = 768
                else:
                    self.P = 32
                    self.n_embed = 768
            else:
                encoder_weight = "/data/storage/jianwen/cache/ckpt_matters/gra_nima_16x.pt"
                # transformer_weight = None

                if 'encoder_weight' in locals() and encoder_weight is not None:
                    enc_ckpt = torch.load(encoder_weight, map_location="cpu")
                    if isinstance(enc_ckpt, dict) and "event_encoder" in enc_ckpt:
                        self.event_encoder_weight = enc_ckpt["event_encoder"]
                        self.token_proj_weight = enc_ckpt.get("token_proj", None)
                        self.dim_proj_weight = enc_ckpt.get("dim_proj", None)
                    else:
                        self.event_encoder_weight = enc_ckpt
                        self.token_proj_weight = None
                        self.dim_proj_weight = None
                elif 'encoder_weight' in locals() and encoder_weight is None:
                    self.event_encoder_weight = None
                    self.token_proj_weight = None
                    self.dim_proj_weight = None

                if 'transformer_weight' in locals():
                    if transformer_weight is not None:
                        ckpt = torch.load(transformer_weight, map_location="cpu")
                        self.transformer_weight = ckpt.get("transformer", ckpt)
                    else:
                        self.transformer_weight = None

                if self.dim_proj_weight is not None:
                    self.n_embed = self.dim_proj_weight['weight'].shape[0]
                else:
                    self.n_embed = 384 if self.vit in ["small", "small+"] else 768

                if vit_backbone == "dinov3":
                    self.P = 16
                else:
                    self.P = 14
            self.encoder_lr_mult = 0.1 if getattr(self, "event_encoder_weight", None) is None else 0.01
            self.tta_enable = True
            self.tta_scales = (0.8, 1.0, 1.2)
            self.tta_flip = True

        self.n_tokens_per_image = (self.H // self.P) * (self.W // self.P)

    def _align_spatial_to_stride(self, stride: int):
        def _align(value: int, step: int) -> int:
            return int(math.ceil(value / step) * step)

        self.H = _align(self.H, stride)
        self.W = _align(self.W, stride)

    def _init_palette(self):
        if self.dataset == "dsec":
            self.color_mapping = {
                (128, 64, 128): 5,
                (244, 35, 232): 6,
                (70, 70, 70): 1,
                (102, 102, 156): 9,
                (190, 153, 153): 2,
                (153, 153, 153): 4,
                (250, 170, 30): 10,
                (220, 220, 0): 10,
                (107, 142, 35): 7,
                (152, 251, 152): 7,
                (70, 130, 180): 0,
                (220, 20, 60): 3,
                (255, 0, 0): 3,
                (0, 0, 142): 8,
                (0, 0, 70): 8,
                (0, 60, 100): 8,
                (0, 80, 100): 8,
                (0, 0, 230): 8,
                (119, 11, 32): 8,
                (0, 0, 0): 11,
            }
        else:
            self.color_mapping = {
                (128, 64, 128): 0,
                (70, 130, 180): 1,
                (70, 70, 70): 2,
                (107, 142, 35): 3,
                (220, 20, 60): 4,
                (0, 0, 142): 5,
                (0, 0, 0): 6,
            }
        self.reversed_color_mapping = {v: k for k, v in self.color_mapping.items()}
        self.palette = torch.tensor(
            [self.reversed_color_mapping[i] for i in range(len(self.reversed_color_mapping.keys()))],
            dtype=torch.uint8,
            device=self.device,
        )

    def _build_preprocessors(self):
        if self.encoder_mode == "ecddp":
            pad_target = (
                self.H + self.ecddp_pad_extra[0],
                self.W + self.ecddp_pad_extra[1],
            )
            self.train_preprocessors = PairedProcessor(
                [
                    EnsureTensorPair(),
                    EventTensorJitter(
                        scale_range=(0.85, 1.15),
                        bias_range=(-0.05, 0.05),
                        noise_std=0.02,
                        channel_wise=True,
                        clamp=(0.0, 1.0),
                        p=0.7,
                    ),
                    RandomHorizontalFlip(p=0.5),
                    ResizeKeepRatio(short_side_target_size=self.H, scale_range=self.scale_range, type=self.type),
                    PadToMinSide(target=pad_target, pad_x1=0, pad_x2=self.ignore_index),
                    RandomCrop(
                        crop_size=(self.H, self.W),
                        type=self.type,
                        cat_max_ratio=self.cat_max_ratio,
                        ignore_index=self.ignore_index,
                    ),
                    RandomShiftScaleRotateTensor(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=15.0,
                        p=0.8,
                        ignore_index=self.ignore_index,
                    ),
                ]
            )
            self.valid_preprocessors = PairedProcessor(
                [
                    EnsureTensorPair(),
                    ResizeKeepRatio(short_side_target_size=self.H, type=self.type),
                    PadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=self.ignore_index),
                    CenterCrop((self.H, self.W)),
                ]
            )
        else:
            common_train = [
                ToTensor(origi_H=self.origi_H, type=self.type),
                Normalize(self.ME, self.SE, self.MI, self.SI, type=self.type),
                RandomSwapEventRedBlue(type=self.type),
                RandomHorizontalFlip(p=0.5),
                ResizeKeepRatio(short_side_target_size=self.H, scale_range=self.scale_range, type=self.type),
                PadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=self.ignore_index),
                RandomCrop(
                    crop_size=(self.H, self.W),
                    cat_max_ratio=self.cat_max_ratio,
                    ignore_index=self.ignore_index,
                    type=self.type,
                ),
            ]
            common_valid = [
                ToTensor(origi_H=self.origi_H, type=self.type),
                Normalize(self.ME, self.SE, self.MI, self.SI, type=self.type),
                ResizeKeepRatio(short_side_target_size=self.H, type=self.type),
                PadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=self.ignore_index),
                CenterCrop((self.H, self.W)),
            ]
            self.train_preprocessors = PairedProcessor(common_train)
            self.valid_preprocessors = PairedProcessor(common_valid)

    def _build_datasets(self):
        if self.dataset == "dsec":
            dsec_root = "/data/storage/jianwen/DSEC"
            if self.encoder_mode == "ecddp":
                self.train_dataset = DSECECDDPEventDataset(
                    root_dir=dsec_root,
                    split="train",
                    C=self.C,
                    tensor_subdir=self.ecddp_tensor_subdir,
                    tensor_exts=self.ecddp_tensor_exts,
                    transform=self.train_preprocessors,
                )
                self.valid_dataset = DSECECDDPEventDataset(
                    root_dir=dsec_root,
                    split="test",
                    C=self.C,
                    tensor_subdir=self.ecddp_tensor_subdir,
                    tensor_exts=self.ecddp_tensor_exts,
                    transform=self.valid_preprocessors,
                )
                if getattr(self.train_dataset, "event_channels", None):
                    self.ecddp_in_chans = int(self.train_dataset.event_channels)
                if getattr(self.train_dataset, "event_hw", None):
                    self.H, self.W = tuple(int(v) for v in self.train_dataset.event_hw)
                    self.ecddp_image_size = (self.H, self.W)
                    if (self.H % self.P) != 0 or (self.W % self.P) != 0:
                        raise ValueError(
                            f"ECDDP tensors have size {(self.H, self.W)} which is not divisible by patch stride {self.P}"
                        )
                    self.n_tokens_per_image = (self.H // self.P) * (self.W // self.P)
            else:
                self.train_dataset = DSECSegmentDataset(
                    root_dir=dsec_root,
                    split="train",
                    data=self.data,
                    C=self.C,
                    transform=self.train_preprocessors,
                )
                self.valid_dataset = DSECSegmentDataset(
                    root_dir=dsec_root,
                    split="test",
                    data=self.data,
                    C=self.C,
                    transform=self.valid_preprocessors,
                )
        elif self.dataset == "ddd17":
            self.train_dataset = DDD17SegmentDataset(
                split="train",
                data=self.data,
                C=self.C,
                transform=self.train_preprocessors,
            )
            self.valid_dataset = DDD17SegmentDataset(
                split="valid",
                data=self.data,
                C=self.C,
                transform=self.valid_preprocessors,
            )

class DEPConfig(Config):
    """Depth estimation pipeline: optional EventScape pre-training and MVSEC fine-tuning."""

    def __init__(self):
        super().__init__()
        self.encoder_mode               = "ours"  # ["ours", "mem", "ecddp"]
        self.ecddp_ckpt_path            = "/data/storage/jianwen/pr.pt"
        self.ecddp_patch_size           = 4
        self.ecddp_depths               = (2, 2, 6, 2)
        self.ecddp_num_heads            = (3, 6, 12, 24)
        self.ecddp_embed_dim            = 96
        self.ecddp_window_size          = 7
        self.ecddp_mlp_ratio            = 4.0
        self.ecddp_drop_rate            = 0.0
        self.ecddp_attn_drop_rate       = 0.0
        self.ecddp_drop_path_rate       = 0.0
        self.ecddp_image_size           = (448, 640)
        self.ecddp_tensor_subdir        = "eventTensor_ecddp"
        self.ecddp_tensor_exts          = (".pt", ".npz", ".npy")

        # ------------------------------------------------------------------
        # General model and optimisation settings
        # ------------------------------------------------------------------
        self.device                     = "cuda:1"
        self.use_events                 = True
        self.use_rgb                    = False
        self.fuse_mode                  = "concat"   # options: concat | add | cross_attention
        self.vit                        = "base"   # ["small", "base", "large"]
        self.vit_backbone               = "dinov2"  # ["dinov2", "dinov3"]
        self.depth_decoder              = "baseline"  # ["baseline", "dinov3_dpt"]
        self.dpt_layer_indices          = [2, 5, 8, 11]
        self.dpt_channels               = 256
        self.dpt_post_process_channels  = [128, 256, 512, 1024]
        self.dpt_use_batchnorm          = True 

        self.event_encoder_weight       = None
        self.image_encoder_weight       = None
        # self.transformer_weight       = None
        # self.transformer_weight         = torch.load("/data/storage/jianwen/cache/ckpt_matters/gpt_mixture_4x.pt", map_location="cpu")["transformer"]

        self.encoder_frozen             = False
        self.encoder_lr_mult            = 0.01
        self.transformer_lr_mult        = 0.05

        self.n_head                     = 6
        self.n_layer                    = 12
        self.window_size                = 2048
        if self.encoder_mode == "ecddp":
            depth_levels = len(self.ecddp_depths)
            self.n_embed = int(self.ecddp_embed_dim * (2 ** (depth_levels - 1)))
            self.patch_size = self.ecddp_patch_size * (2 ** (depth_levels - 1))
        else:
            self.event_encoder_weight = torch.load(
                "/data/storage/jianwen/cache/ckpt_matters/gra_base_16x.pt"
            )["event_encoder"]
            self.patch_size = 14
            self._init_vit_weights()

        self.output_height              = 280
        self.output_width               = 350
        if self.encoder_mode == "ecddp":
            self.output_height, self.output_width = self.ecddp_image_size
        self.n_tokens_per_image         = (self.output_height // self.patch_size) * (self.output_width // self.patch_size)

        self.batch_size                 = 8 if self.encoder_mode == "ours" else 4
        self.n_workers                  = 8
        self.lr                         = 5e-4
        self.wd                         = 1e-5
        self.min_lr                     = 0.0
        self.warmup_steps               = 100
        self.grad_clip                  = 1.0

        # Depth objective follows the fine-tuning recipe described in Sec. 2.4:
        # scale-invariant log loss plus a multi-scale scale-invariant gradient
        # matching loss, weighted 1.0 and 0.25 respectively.
        self.depth_loss_weights         = {
            "silog": 1.0,
            "ms_grad": 0.25,
        }
        self.depth_metrics              = ["delta1", "delta2", "delta3", "abs", "rmse", "rmse_log"]
        # When predicting normalized log depth we still need finite bounds to
        # invert the predictions back to metric depth. Most datasets clip the
        # valid LiDAR depth to < 1000m, so we use that as a safe default.
        self.depth_normalizer_max       = 1000.0

        # ------------------------------------------------------------------
        # EventScape (stage 1) configuration
        # ------------------------------------------------------------------
        self.enable_eventscape_stage    = False
        self.eventscape_root            = "/data/storage/jianwen/EventScape"
        self.eventscape_steps           = 10000
        self.eventscape_train_min_depth = 0.1
        self.eventscape_train_max_depth = None
        self.eventscape_eval_min_depth  = 0.1
        self.eventscape_eval_max_depth  = None
        self.eventscape_depth_scale     = 1.0
        self.eventscape_invalid_depth   = 1000.0

        eventscape_crop_scale           = (0.5, 1.0)
        eventscape_crop_ratio           = (0.9, 1.1)
        self.eventscape_train_transform = DepthTransformCompose([
            DepthToTensor(),
            DepthRandomResizedCrop(size=(self.output_height, self.output_width),
                                   scale=eventscape_crop_scale,
                                   ratio=eventscape_crop_ratio
                                   ),
            DepthRandomHorizontalFlip(p=0.5),
            DepthColorJitter(brightness=0.3,
                             contrast=0.3,
                             saturation=0.2,
                             hue=0.05,
                             p=0.6),
            DepthRandomGamma(gamma_range=(0.8, 1.2), p=0.4),
            DepthAdditiveGaussianNoise(std=0.03, p=0.5),
            DepthNormalize(self.SCAP_ME, self.SCAP_SE, self.SCAP_MI, self.SCAP_SI),
            DepthClamp(self.eventscape_train_min_depth, self.eventscape_train_max_depth),
            DepthEnsureContiguous(),
        ])
        self.eventscape_valid_transform = DepthTransformCompose([
            DepthToTensor(),
            DepthResize(size=(self.output_height, self.output_width)),
            DepthNormalize(self.SCAP_ME, self.SCAP_SE, self.SCAP_MI, self.SCAP_SI),
            DepthClamp(self.eventscape_eval_min_depth, self.eventscape_eval_max_depth),
            DepthEnsureContiguous(),
        ])

        self.eventscape_train_dataset   = EventScapeDepthDataset(
            root_dir=self.eventscape_root,
            split="train",
            transform=self.eventscape_train_transform,
            min_depth=self.eventscape_train_min_depth,
            max_depth=self.eventscape_train_max_depth,
            depth_scale=self.eventscape_depth_scale,
            invalid_depth_value=self.eventscape_invalid_depth,
        )
        self.eventscape_valid_dataset   = EventScapeDepthDataset(
            root_dir=self.eventscape_root,
            split="valid",
            transform=self.eventscape_valid_transform,
            min_depth=self.eventscape_eval_min_depth,
            max_depth=self.eventscape_eval_max_depth,
            depth_scale=self.eventscape_depth_scale,
            invalid_depth_value=self.eventscape_invalid_depth,
        )

        # ------------------------------------------------------------------
        # MVSEC (stage 2) configuration
        # ------------------------------------------------------------------
        self.enable_mvsec_stage         = True
        self.mvsec_root                 = "/data/storage/jianwen/mvsec"
        self.mvsec_calibration_root     = "/data/storage/jianwen/MVSEC"
        self.mvsec_steps                = 10000
        self.log_every                  = 30
        self.valid_every                = 300
        self.mvsec_train_sequences      = ["outdoor_day2"]
        self.mvsec_valid_sequences      = ["outdoor_day1", "outdoor_night1", "outdoor_night2", "outdoor_night3"]
        self.mvsec_train_min_depth      = 0.1
        self.mvsec_train_max_depth      = 30
        self.mvsec_eval_min_depth       = 0.1
        self.mvsec_eval_max_depth       = 30
        self.mvsec_depth_scale          = 1.0 / 100.0

        mvsec_crop_scale                = (0.5, 1.0)
        mvsec_crop_ratio                = (0.9, 1.1)
        self.mvsec_train_transform      = DepthTransformCompose([
            DepthToTensor(),
            DepthRandomResizedCrop(size=(self.output_height, self.output_width),
                                   scale=mvsec_crop_scale,
                                   ratio=mvsec_crop_ratio
                                ),
            DepthRandomHorizontalFlip(p=0.5),
            DepthColorJitter(brightness=0.3,
                             contrast=0.3,
                             saturation=0.2,
                             hue=0.05,
                             p=0.6),
            DepthRandomGamma(gamma_range=(0.8, 1.2), p=0.4),
            DepthAdditiveGaussianNoise(std=0.03, p=0.5),
            DepthNormalize(self.MVSEC_EVENT_ME, self.MVSEC_EVENT_SE, self.MVSEC_IMAGE_ME, self.MVSEC_IMAGE_SE),
            DepthClamp(self.mvsec_train_min_depth, self.mvsec_train_max_depth),
            DepthEnsureContiguous(),
        ])
        self.mvsec_valid_transform      = DepthTransformCompose([
            DepthToTensor(),
            DepthResize(size=(self.output_height, self.output_width)),
            DepthNormalize(self.MVSEC_EVENT_ME, self.MVSEC_EVENT_SE, self.MVSEC_IMAGE_ME, self.MVSEC_IMAGE_SE),
            DepthClamp(self.mvsec_eval_min_depth, self.mvsec_eval_max_depth),
            DepthEnsureContiguous(),
        ])

        mvsec_dataset_cls = MVSECECDDPDepthDataset if self.encoder_mode == "ecddp" else MVSECDepthDataset
        mvsec_dataset_kwargs = {}
        if self.encoder_mode == "ecddp":
            mvsec_dataset_kwargs.update(
                tensor_root=self.mvsec_root,
                tensor_subdir=self.ecddp_tensor_subdir,
                tensor_exts=self.ecddp_tensor_exts,
            )

        self.mvsec_train_dataset        = mvsec_dataset_cls(
            root_dir=self.mvsec_root,
            split="train",
            sequences=self.mvsec_train_sequences,
            transform=self.mvsec_train_transform,
            min_depth=self.mvsec_train_min_depth,
            max_depth=self.mvsec_train_max_depth,
            depth_scale=self.mvsec_depth_scale,
            calibration_root=self.mvsec_calibration_root,
            **mvsec_dataset_kwargs,
        )
        self.mvsec_valid_dataset        = mvsec_dataset_cls(
            root_dir=self.mvsec_root,
            split="valid",
            sequences=self.mvsec_valid_sequences,
            transform=self.mvsec_valid_transform,
            min_depth=self.mvsec_eval_min_depth,
            max_depth=self.mvsec_eval_max_depth,
            depth_scale=self.mvsec_depth_scale,
            calibration_root=self.mvsec_calibration_root,
            **mvsec_dataset_kwargs,
        )
        if self.encoder_mode == "ecddp":
            detected_channels = getattr(self.mvsec_train_dataset, "event_channels", None)
            if detected_channels is not None:
                self.ecddp_in_chans = int(detected_channels)

        # ------------------------------------------------------------------
        # Backward compatibility shortcuts (MVSEC defaults)
        # ------------------------------------------------------------------
        self.dataset_root               = self.mvsec_root
        self.train_sequences            = self.mvsec_train_sequences
        self.valid_sequences            = self.mvsec_valid_sequences
        self.min_depth                  = self.mvsec_train_min_depth
        self.max_depth                  = self.mvsec_train_max_depth
        self.depth_scale                = self.mvsec_depth_scale
        self.H, self.W                  = self.output_height, self.output_width
        self.train_transform            = self.mvsec_train_transform
        self.valid_transform            = self.mvsec_valid_transform
        self.train_dataset              = self.mvsec_train_dataset
        self.valid_dataset              = self.mvsec_valid_dataset

        if self.enable_eventscape_stage:
            print(f"EventScape train dataset size: {len(self.eventscape_train_dataset)}")
            print(f"EventScape valid dataset size: {len(self.eventscape_valid_dataset)}")
        print(f"MVSEC train dataset size: {len(self.mvsec_train_dataset)}")
        print(f"MVSEC valid dataset size: {len(self.mvsec_valid_dataset)}")

        # Validation splits that DepthEstimator will iterate over
        self.validation_splits = {
            "eventscape": {
                "dataset": self.eventscape_valid_dataset,
                "min_depth": self.eventscape_eval_min_depth,
                "max_depth": self.eventscape_eval_max_depth,
                "event_mean": self.SCAP_ME,
                "event_std": self.SCAP_SE,
                "image_mean": self.SCAP_MI,
                "image_std": self.SCAP_SI,
            },
            "mvsec": {
                "dataset": self.mvsec_valid_dataset,
                "min_depth": self.mvsec_eval_min_depth,
                "max_depth": self.mvsec_eval_max_depth,
                "event_mean": self.MVSEC_EVENT_ME,
                "event_std": self.MVSEC_EVENT_SE,
                "image_mean": self.MVSEC_IMAGE_ME,
                "image_std": self.MVSEC_IMAGE_SE,
            },
        }
        if self.encoder_mode == "ecddp":
            self.validation_splits = {
                k: v for k, v in self.validation_splits.items() if k == "mvsec"
            }

        # Stage definitions consumed by the trainer
        cumulative_steps = 0
        self.training_stages = []
        eventscape_valids = ("eventscape", "mvsec")
        mvsec_valids = ("eventscape", "mvsec")
        if self.encoder_mode == "ecddp":
            eventscape_valids = ("mvsec",)
            mvsec_valids = ("mvsec",)

        if self.enable_eventscape_stage and self.eventscape_steps > 0:
            self.training_stages.append({
                "name": "eventscape",
                "train_dataset": self.eventscape_train_dataset,
                "min_depth": self.eventscape_train_min_depth,
                "max_depth": self.eventscape_train_max_depth,
                "steps": self.eventscape_steps,
                "start_step": cumulative_steps,
                "valid_splits": eventscape_valids,
            })
            cumulative_steps += self.eventscape_steps
        if self.enable_mvsec_stage and self.mvsec_steps > 0:
            self.training_stages.append({
                "name": "mvsec",
                "train_dataset": self.mvsec_train_dataset,
                "min_depth": self.mvsec_train_min_depth,
                "max_depth": self.mvsec_train_max_depth,
                "steps": self.mvsec_steps,
                "start_step": cumulative_steps,
                "valid_splits": mvsec_valids,
            })
            cumulative_steps += self.mvsec_steps

        # Fallback for utilities expecting a single step count
        self.total_steps = cumulative_steps
        self.steps = cumulative_steps

    def _init_vit_weights(self):
        vit_backbone = getattr(self, "vit_backbone", "dinov2")
        vit_size = getattr(self, "vit", "base")
        embed_map = {
            "small": 384,
            "base": 768,
            "large": 1024,
        }
        if vit_size not in embed_map:
            raise ValueError(f"Unsupported ViT size '{vit_size}' for DEPConfig.")
        self.n_embed = embed_map[vit_size]
        self.patch_size = 16 if vit_backbone == "dinov3" else 14

class SegAggConfig(Config):
    """
    Segmentation configuration that feeds short DSEC frame sequences through
    the encoder and aggregates them with `spatiotemporal_aggregate` before the
    transformer.
    """

    def __init__(self):
        super().__init__()
        self.device                 = "cuda:3"
        self.data                   = "event"
        self.type                   = "EL"
        self.dataset                = "dsec"
        self.frames_per_sample      = 2
        self.images_per_group       = 2
        self.modalities             = ("event", "event")

        self.event_encoder_weight   = torch.load("/data/storage/jianwen/cache/ckpt_matters/gra_mixture_16x.pt", map_location="cpu")["event_encoder"]
        self.transformer_weight     = torch.load("/data/storage/jianwen/cache/ckpt_matters/gpt_aggregrator.pt", map_location="cpu")["transformer"]
        self.image_encoder_weight   = None

        self.encoder_frozen         = True
        self.lr                     = 1e-3
        self.batch_size             = 16
        self.wd                     = 1e-5
        self.encoder_lr_mult        = 0.01
        self.transformer_lr_mult    = 0.01

        self.window_size            = 4096
        self.n_embed                = 384
        self.n_head                 = 6
        self.n_layer                = 12
        self.C                      = 11
        self.ignore_index           = 11

        self.H, self.W              = (448, 644)
        self.origi_H, self.origi_W  = (440, 640)
        self.ME, self.SE            = self.DSEC_ME, self.DSEC_SE
        self.MI, self.SI            = self.DSEC_MI, self.DSEC_SI

        self.scale_range            = (0.5, 2.0)
        self.n_tokens_per_image     = (self.H // self.P) * (self.W // self.P)

        self.color_mapping          = {
            (128,  64, 128): 5,    # road
            (244,  35, 232): 6,    # sidewalk
            ( 70,  70,  70): 1,    # building
            (102, 102, 156): 9,    # wall
            (190, 153, 153): 2,    # fence
            (153, 153, 153): 4,    # pole
            (250, 170,  30): 10,   # traffic light
            (220, 220,   0): 10,   # traffic sign
            (107, 142,  35): 7,    # vegetation
            (152, 251, 152): 7,    # terrain
            ( 70, 130, 180): 0,    # sky
            (220,  20,  60): 3,    # person
            (255,   0,   0): 3,    # rider
            (  0,   0, 142): 8,    # car
            (  0,   0,  70): 8,    # truck
            (  0,  60, 100): 8,    # bus
            (  0,  80, 100): 8,    # train
            (  0,   0, 230): 8,    # motorcycle
            (119,  11,  32): 8,    # bicycle
            (  0,   0,   0): 11,   # pad / void
        }
        self.reversed_color_mapping = {v: k for k, v in self.color_mapping.items()}
        self.palette                = torch.tensor(
            [self.reversed_color_mapping[i] for i in range(len(self.reversed_color_mapping.keys()))],
            dtype=torch.uint8,
            device=self.device,
        )
        self.cat_max_ratio          = 0.75

        self.n_workers              = 8
        self.min_lr                 = 0.
        self.warmup_steps           = 100
        self.steps                  = 4000
        self.log_every              = 40
        self.valid_every            = 400

        self.train_preprocessors    = SequencePairedProcessor([
                                                SequenceToTensor(type=self.type, modalities=self.modalities, origi_H=self.origi_H),
                                                SequenceNormalize(self.ME, self.SE, self.MI, self.SI, type=self.type, modalities=self.modalities),
                                                SequenceRandomSwapEventRedBlue(modalities=self.modalities),
                                                SequenceRandomHorizontalFlip(p=0.5),
                                                SequenceResizeKeepRatio(short_side_target_size=self.H, scale_range=self.scale_range, type=self.type),
                                                SequencePadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=self.ignore_index),
                                                SequenceRandomCrop(crop_size=(self.H, self.W),
                                                                   cat_max_ratio=self.cat_max_ratio,
                                                                   ignore_index=self.ignore_index,
                                                                   type=self.type)
                                                ])
        self.valid_preprocessors    = SequencePairedProcessor([
                                                SequenceToTensor(type=self.type, modalities=self.modalities, origi_H=self.origi_H),
                                                SequenceNormalize(self.ME, self.SE, self.MI, self.SI, type=self.type, modalities=self.modalities),
                                                SequenceResizeKeepRatio(short_side_target_size=self.H, type=self.type),
                                                SequencePadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=self.ignore_index),
                                                SequenceCenterCrop((self.H, self.W)),
                                                ])
        self.train_dataset          = DSECSegmentSequenceDataset(
                                                root_dir="/data/storage/jianwen/DSEC",
                                                split="train",
                                                C=self.C,
                                                frames_per_sample=self.frames_per_sample,
                                                transform=self.train_preprocessors)
        self.valid_dataset          = DSECSegmentSequenceDataset(
                                                root_dir="/data/storage/jianwen/DSEC",
                                                split="test",
                                                C=self.C,
                                                frames_per_sample=self.frames_per_sample,
                                                transform=self.valid_preprocessors)
        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"valid dataset size: {len(self.valid_dataset)}")

class GraConfig(Config):
    def __init__(self):
        super().__init__()
        self.device                 = "cuda:1"
        self.vit                    = "small"   # ["small", "base", "large"， “small+]
        self.vit_backbone           = "dinov2"  # ["dinov2", "dinov3"]
        self.event_backbone         = "vit"     # ["vit", "swin"]
        self.teacher_vit            = "small"      # ["small", "base", "large"] or None (same as vit)
        self.swin_project           = False
        self.dinov3_ckpt_paths      = dict(DINOV3_CKPT_PATHS)
        
        self.event_channels         = 20        # 3 or 20
        self.loss_types             = {"mse": 1.0, "mse_patch": 1.0} # "mse_patch" for patch_embed alignment
        self.loss_grad_ratio        = None      # Each auxiliary loss targets 10% of cosine-loss gradient norm
        self.loss_grad_eps          = 1e-8

        # Weights
        self.image_encoder_weight   = torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_reg4_pretrain.pth", map_location="cpu")
        self.event_encoder_weight   = torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_reg4_pretrain.pth", map_location="cpu")
        self.restore_ckpt           = None
        
        # Model specs
        self.n_embed                = 384 if self.vit == "small" else 768
        self.n_head                 = 6 if self.vit == "small" else 12
        self.n_layer                = 12
        self.window_size            = 4096
        
        self.gra_style              = "pure_dsec" # ["pure_nima", "pure_bddd", "pure_dsec", "pure_scape", "pure_dd17", "mixture", "pretrain"]
        self.batch_size             = 64
        self.n_workers              = 8
        self.lr                     = 2e-5
        self.wd                     = 0.
        self.min_lr                 = 0.
        self.warmup_steps           = 1000
        self.steps                  = int(24e4)  # standard int(3e4)
        self.log_every              = 100
        self.valid_every            = 1000
        self.type                   = "EI"
        self.rescale                = 1

        self.DSEC_H, self.DSEC_W         = 448, 644
        self.dsec_n_tokens_per_image     = (self.DSEC_H // self.P) * (self.DSEC_W // self.P)
        self.dsec_scale_range            = (0.25, 1.0)
        self.train_dsec_preprocessors    = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.DSEC_H, self.DSEC_W), scale=self.dsec_scale_range, type=self.type, interpolation="bicubic"),
                                            ])
        self.valid_dsec_preprocessors    = PairedProcessor([
                                                    ToTensor(type=self.type),
                                                    Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                                    ResizeKeepRatio(short_side_target_size=224, type=self.type),
                                                    CenterCrop((224, 224)),
                                                    ])

        # self.SCAP_H, self.SCAP_W        = 224 // self.rescale, 448 // self.rescale
        self.SCAP_H, self.SCAP_W        = 224, 224
        self.scape_n_tokens_per_image     = (self.SCAP_H // self.P) * (self.SCAP_W // self.P)
        self.scape_scale_range            = (0.25, 1.0)
        self.train_scap_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.SCAP_ME, self.SCAP_SE, self.SCAP_MI, self.SCAP_SI, type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),                                           
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.SCAP_H, self.SCAP_W), scale=self.scape_scale_range, type=self.type, interpolation="bicubic"),
                                            ])
        self.valid_scap_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.SCAP_ME, self.SCAP_SE, self.SCAP_MI, self.SCAP_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=self.SCAP_H, type=self.type),
                                            CenterCrop((self.SCAP_H, self.SCAP_W)),
                                            ])

        self.NIMA_H, self.NIMA_W        = 224, 224
        self.nima_n_tokens_per_image    = (self.NIMA_H // self.P) * (self.NIMA_W // self.P)
        self.train_nima_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.NIMA_ME, self.NIMA_SE, self.NIMA_MI, self.NIMA_SI, type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.NIMA_H, self.NIMA_W), scale=(0.5, 1.0), type=self.type, interpolation="bicubic"),
                                        ])
        self.valid_nima_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.NIMA_ME, self.NIMA_SE, self.NIMA_MI, self.NIMA_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=224, type=self.type),
                                            CenterCrop((224, 224)),
                                        ])
        
        self.DD17_H, self.DD17_W        = 224, 224
        self.dd17_n_tokens_per_image    = (self.DD17_H // self.P) * (self.DD17_W // self.P)
        self.dd17_scale_range           = (0.5, 1.0)
        self.train_dd17_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.DD17_ME, self.DD17_SE, self.DD17_MI, self.DD17_SI, type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.DD17_H, self.DD17_W), scale=self.dd17_scale_range, type=self.type, interpolation="bicubic"),
                                        ])
        self.valid_dd17_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.DD17_ME, self.DD17_SE, self.DD17_MI, self.DD17_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=self.DD17_H, type=self.type),
                                            CenterCrop((self.DD17_H, self.DD17_W)),
                                        ])

        self.train_dsec_dataset          = DSECPairedDataset(root_dir = "/data/storage/jianwen/DSEC", split="train", transform=self.train_dsec_preprocessors, event_channels=self.event_channels, event_subdir="eventTensor_ecddp_way")
        self.valid_dsec_dataset          = DSECPairedDataset(root_dir = "/data/storage/jianwen/DSEC", split="test", transform=self.valid_dsec_preprocessors, event_channels=self.event_channels, event_subdir="eventTensor_ecddp_way")
        self.train_scap_dataset         = SCAPEPairedDataset(root_dir = "/data/storage/jianwen/EventScape", split="train", transform=self.train_scap_preprocessors)
        self.valid_scap_dataset         = SCAPEPairedDataset(root_dir = "/data/storage/jianwen/EventScape", split="valid", transform=self.valid_scap_preprocessors)
        self.train_nima_dataset          = NIMAPairedDataset(root_dir = "/data/storage/jianwen/N_ImageNet", split="train", transform=self.train_nima_preprocessors, event_channels=self.event_channels)
        self.valid_nima_dataset          = NIMAPairedDataset(root_dir = "/data/storage/jianwen/N_ImageNet", split="valid", transform=self.valid_nima_preprocessors, event_channels=self.event_channels)
        # self.train_bddd_dataset          = BDDPairedDataset(root_dir = "/data/storage/jianwen/bdd100k/paired", split="train", transform=self.train_bddd_preprocessors)
        # self.valid_bddd_dataset          = BDDPairedDataset(root_dir = "/data/storage/jianwen/bdd100k/paired", split="valid", transform=self.valid_bddd_preprocessors)
        # self.train_dd17_dataset          = DDD17PairedDataset(split="train", transform=self.train_dd17_preprocessors)
        # self.valid_dd17_dataset          = DDD17PairedDataset(split="valid", transform=self.valid_dd17_preprocessors)
        # compute_mean_std(self.train_dsec_dataset, num_workers=8)
        # compute_mean_std(self.train_dd17_dataset, num_workers=8)
        # exit()
        # compute_mean_std(self.train_scap_dataset, num_workers=8)
        # compute_mean_std(self.train_nima_dataset, num_workers=8)
        # compute_mean_std(self.train_bddd_dataset, num_workers=8)
        print(f"train_dsec_dataset length: {len(self.train_dsec_dataset)}")
        print(f"valid_dsec_dataset length: {len(self.valid_dsec_dataset)}")
        print(f"train_scap_dataset length: {len(self.train_scap_dataset)}")
        print(f"valid_scap_dataset length: {len(self.valid_scap_dataset)}")
        print(f"train_nima_dataset length: {len(self.train_nima_dataset)}")
        print(f"valid_nima_dataset length: {len(self.valid_nima_dataset)}")
        # print(f"train_bddd_dataset length: {len(self.train_bddd_dataset)}")
        # print(f"valid_bddd_dataset length: {len(self.valid_bddd_dataset)}")
        # print(f"train_dd17_dataset length: {len(self.train_dd17_dataset)}")
        # print(f"valid_dd17_dataset length: {len(self.valid_dd17_dataset)}")
        '''
        train_dsec_dataset length: 52727        52727   * 0.3   = 15818
        valid_dsec_dataset length: 11204
        train_scap_dataset length: 122329       122329  * 0.05  = 6116
        valid_scap_dataset length: 22493
        train_nima_dataset length: 1281167      1281167 * 0.5   = 640584
        valid_nima_dataset length: 50000
        train_bddd_dataset length: 41346        41346   * 0.1   = 4134
        valid_bddd_dataset length: 9999
        train_dd17_dataset length: 356830       356830  * 0.05  = 17841
        valid_dd17_dataset length: 143360
        '''

    def _init_vit_weights(self):
        vit_backbone = getattr(self, "vit_backbone", "dinov2")
        if vit_backbone == "dinov3":
            ckpt_path = self.dinov3_ckpt_paths.get(self.vit)
            if ckpt_path is None:
                raise ValueError(f"No DINOv3 checkpoint defined for vit='{self.vit}'.")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.event_encoder_weight = state_dict
            self.image_encoder_weight = copy.deepcopy(state_dict)
            self.P = 16
            self.n_embed = 384 if self.vit == "small" else 768
            return

        def _get_info(size):
            if size == "small":
                return "/data/storage/jianwen/cache/dinov2/dinov2_vits14_reg4_pretrain.pth", 384
            elif size == "base":
                return "/data/storage/jianwen/cache/dinov2/dinov2_vitb14_pretrain.pth", 768
            elif size == "large":
                return "/data/storage/jianwen/cache/dinov2/dinov2_vitl14_pretrain.pth", 1024
            elif size == "small+":
                return None, 384
            else:
                raise ValueError(f"Unsupported vit size '{size}'.")

        ev_ckpt, self.n_embed = _get_info(self.vit)
        if ev_ckpt:
            self.event_encoder_weight = torch.load(ev_ckpt, map_location="cpu")
        else:
            self.event_encoder_weight = None

        if hasattr(self, "finetuned_image_encoder_path") and self.finetuned_image_encoder_path and os.path.exists(self.finetuned_image_encoder_path):
            print(f"Loading fine-tuned image encoder from {self.finetuned_image_encoder_path}")
            ckpt = torch.load(self.finetuned_image_encoder_path, map_location="cpu")
            if isinstance(ckpt, dict) and "encoder" in ckpt:
                self.image_encoder_weight = ckpt["encoder"]
            else:
                self.image_encoder_weight = ckpt
        else:
            teacher_size = self.teacher_vit if self.teacher_vit is not None else self.vit
            im_ckpt, _ = _get_info(teacher_size)
            if im_ckpt:
                self.image_encoder_weight = torch.load(im_ckpt, map_location="cpu")
            else:
                self.image_encoder_weight = None

        self.P = 14

class RECConfig(Config):
    def __init__(self):
        super().__init__()
        self.device                 = "cuda:0"
        self.n_embed                = 384
        self.n_head                 = 6
        self.n_layer                = 8
        self.event_encoder_weight   = torch.load("/data/storage/jianwen/cache/ckpt_matters/gra_mixture_4x.pt", map_location="cpu")["event_encoder"]
        # self.event_encoder_weight   = torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", map_location="cpu")
        self.image_encoder_weight   = torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", map_location="cpu")

        self.batch_size             = 16
        self.n_workers              = 8
        self.lr                     = 1e-4
        self.wd                     = 1e-5
        self.min_lr                 = 0.
        self.warmup_steps           = 100
        self.steps                  = int(10e4)
        self.log_every              = 100
        self.valid_every            = 1000
        self.type                   = "EI"
        self.style                  = "mixture"   # ["pure_nima", "pure_bddd", "pure_dsec", "pure_scape", "pure_dd17", "mixture", "pretrain"]

        self.DSEC_H, self.DSEC_W         = 224, 224
        self.dsec_n_tokens_per_image     = (self.DSEC_H // self.P) * (self.DSEC_W // self.P)
        self.dsec_scale_range            = (0.25, 1.0)
        self.train_dsec_preprocessors    = PairedProcessor([
                                            ToTensor(type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.DSEC_H, self.DSEC_W), scale=self.dsec_scale_range, type=self.type),
                                            ])
        self.valid_dsec_preprocessors    = PairedProcessor([
                                                    ToTensor(type=self.type),
                                                    Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                                    ResizeKeepRatio(short_side_target_size=self.DSEC_H, type=self.type),
                                                    CenterCrop((self.DSEC_H, self.DSEC_W)),
                                                    ])

        # self.SCAP_H, self.SCAP_W        = 224 // self.rescale, 448 // self.rescale
        self.SCAP_H, self.SCAP_W        = 224, 224
        self.scape_n_tokens_per_image     = (self.SCAP_H // self.P) * (self.SCAP_W // self.P)
        self.scape_scale_range            = (0.25, 1.0)
        self.train_scap_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            Normalize(self.SCAP_ME, self.SCAP_SE, self.SCAP_MI, self.SCAP_SI, type=self.type),                                           
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.SCAP_H, self.SCAP_W), scale=self.scape_scale_range, type=self.type),
                                            ])
        self.valid_scap_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.SCAP_ME, self.SCAP_SE, self.SCAP_MI, self.SCAP_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=self.SCAP_H, type=self.type),
                                            CenterCrop((self.SCAP_H, self.SCAP_W)),
                                            ])

        self.NIMA_H, self.NIMA_W        = 224, 224
        self.nima_n_tokens_per_image    = (self.NIMA_H // self.P) * (self.NIMA_W // self.P)
        self.nima_scale_range            = (0.5, 1.0)
        self.train_nima_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            Normalize(self.NIMA_ME, self.NIMA_SE, self.NIMA_MI, self.NIMA_SI, type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.NIMA_H, self.NIMA_W), scale=self.nima_scale_range, type=self.type),
                                        ])
        self.valid_nima_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.NIMA_ME, self.NIMA_SE, self.NIMA_MI, self.NIMA_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=self.NIMA_H, type=self.type),
                                            CenterCrop((self.NIMA_H, self.NIMA_W)),
                                        ])

        # self.BDDD_H, self.BDDD_W        = 224, 266
        self.BDDD_H, self.BDDD_W        = 224, 224
        self.bddd_n_tokens_per_image    = (self.BDDD_H // self.P) * (self.BDDD_W // self.P)
        self.bddd_scale_range            = (0.5, 1.0)
        self.train_bddd_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            Normalize(self.BDDD_ME, self.BDDD_SE, self.BDDD_MI, self.BDDD_SI, type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.BDDD_H, self.BDDD_W), scale=self.bddd_scale_range, type=self.type),
                                        ])
        self.valid_bddd_preprocessors = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.BDDD_ME, self.BDDD_SE, self.BDDD_MI, self.BDDD_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=self.BDDD_H, type=self.type),
                                            CenterCrop((self.BDDD_H, self.BDDD_W)),
                                            ])
        
        self.DD17_H, self.DD17_W        = 224, 224
        self.dd17_n_tokens_per_image    = (self.DD17_H // self.P) * (self.DD17_W // self.P)
        self.dd17_scale_range           = (0.5, 1.0)
        self.train_dd17_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            RandomSwapEventRedBlue(type=self.type),
                                            Normalize(self.DD17_ME, self.DD17_SE, self.DD17_MI, self.DD17_SI, type=self.type),
                                            RandomHorizontalFlip(p=0.5),
                                            RandomResizedCrop(size=(self.DD17_H, self.DD17_W), scale=self.dd17_scale_range, type=self.type),
                                        ])
        self.valid_dd17_preprocessors   = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.DD17_ME, self.DD17_SE, self.DD17_MI, self.DD17_SI, type=self.type),
                                            ResizeKeepRatio(short_side_target_size=self.DD17_H, type=self.type),
                                            CenterCrop((self.DD17_H, self.DD17_W)),
                                        ])

        self.train_dsec_dataset          = DSECPairedDataset(root_dir = "/data/storage/jianwen/DSEC", split="train", transform=self.train_dsec_preprocessors)
        self.valid_dsec_dataset          = DSECPairedDataset(root_dir = "/data/storage/jianwen/DSEC", split="test", transform=self.valid_dsec_preprocessors)
        self.train_scap_dataset         = SCAPEPairedDataset(root_dir = "/data/storage/jianwen/EventScape", split="train", transform=self.train_scap_preprocessors)
        self.valid_scap_dataset         = SCAPEPairedDataset(root_dir = "/data/storage/jianwen/EventScape", split="valid", transform=self.valid_scap_preprocessors)
        self.train_nima_dataset          = NIMAPairedDataset(root_dir = "/data/storage/jianwen/N_ImageNet", split="train", transform=self.train_nima_preprocessors, event_channels=3)
        self.valid_nima_dataset          = NIMAPairedDataset(root_dir = "/data/storage/jianwen/N_ImageNet", split="valid", transform=self.valid_nima_preprocessors, event_channels=3)
        self.train_bddd_dataset          = BDDPairedDataset(root_dir = "/data/storage/jianwen/bdd100k/paired", split="train", transform=self.train_bddd_preprocessors)
        self.valid_bddd_dataset          = BDDPairedDataset(root_dir = "/data/storage/jianwen/bdd100k/paired", split="valid", transform=self.valid_bddd_preprocessors)
        self.train_dd17_dataset          = DDD17PairedDataset(split="train", transform=self.train_dd17_preprocessors)
        self.valid_dd17_dataset          = DDD17PairedDataset(split="valid", transform=self.valid_dd17_preprocessors)
        # compute_mean_std(self.train_dsec_dataset, num_workers=8)
        # compute_mean_std(self.train_dd17_dataset, num_workers=8)
        # compute_mean_std(self.train_scap_dataset, num_workers=8)
        # compute_mean_std(self.train_nima_dataset, num_workers=8)
        # compute_mean_std(self.train_bddd_dataset, num_workers=8)
        print(f"train_dsec_dataset length: {len(self.train_dsec_dataset)}")
        print(f"valid_dsec_dataset length: {len(self.valid_dsec_dataset)}")
        print(f"train_scap_dataset length: {len(self.train_scap_dataset)}")
        print(f"valid_scap_dataset length: {len(self.valid_scap_dataset)}")
        print(f"train_nima_dataset length: {len(self.train_nima_dataset)}")
        print(f"valid_nima_dataset length: {len(self.valid_nima_dataset)}")
        print(f"train_bddd_dataset length: {len(self.train_bddd_dataset)}")
        print(f"valid_bddd_dataset length: {len(self.valid_bddd_dataset)}")
        print(f"train_dd17_dataset length: {len(self.train_dd17_dataset)}")
        print(f"valid_dd17_dataset length: {len(self.valid_dd17_dataset)}")

class GPTConfig(Config):
    def __init__(self):
        super().__init__()
        self.device                 = "cuda:1"
        self.mae                    = False
        self.modality               = "event"   # ["both", "image", "event"]
        self.H, self.W              = 224, 224
        self.n_tokens_per_image     = (self.H // self.P) * (self.W // self.P)
        self.window_size            = 4096 if not self.mae else 1472
        self.n_embed                = 384
        self.n_decoder_embed        = 384
        self.n_head                 = 6
        self.n_layer                = 12 
        self.n_decoder_layer        = 0

        self.mask_ratio             = 0.
        self.mask_style             = "random"   # ["random", "image", "event"]
        self.batch_size             = 8
        self.n_workers              = 8
        self.lr                     = 1e-4 if not self.mae else 5e-4
        self.wd                     = 1e-5
        self.min_lr                 = 0.
        self.warmup_steps           = 100
        self.steps                  = int(1E6)
        self.log_every              = 100
        self.valid_every            = 1000

        self.train_dataset          = SequenceDataset(split="train", window_size=self.window_size, n_tokens_per_image = self.n_tokens_per_image, n_embed=self.n_embed, modality=self.modality)
        self.valid_dataset          = SequenceDataset(split="test", window_size=self.window_size, n_tokens_per_image = self.n_tokens_per_image, n_embed=self.n_embed, modality=self.modality)
        print(f"train_dataset length: {len(self.train_dataset)}")
        print(f"valid_dataset length: {len(self.valid_dataset)}")

        # self.infer                  = torch.load("/data/storage/jianwen/cache/ckpts/2025-07-28-06:57_gpt/epoch54001_0.4139.pt", weights_only=True)
        self.infer                  = None
        self.decoder_weight         = torch.load("/data/storage/jianwen/cache/ckpt_matters/rec_mixture_4x.pt", weights_only=True)["decoder"]

class CLSConfig(Config):
    def __init__(self):
        super().__init__()
        self.device                 = "cuda:2"
        self.window_size            = 4096
        self.vit                    = "small"  # ["small", "base"]
        self.vit_backbone           = "dinov2"  # ["dinov2", "dinov3"]
        
        # Model specs
        if self.vit_backbone == "dinov3":
            self.P = 16
            self.H, self.W = 256, 256
        else:
            self.P = 14
            self.H, self.W = 224, 224

        if self.vit == "small":
            self.n_embed = 384
        elif self.vit == "base":
            self.n_embed = 768
        else:
            raise ValueError(f"Unsupported vit size '{self.vit}'.")
        
        self.n_head                 = 6 if self.vit == "small" else 12
        self.n_layer                = 12
        self.n_tokens_per_image     = (self.H // self.P) * (self.W // self.P)

        self.modality               = "event"
        self.type                   = "IL" if self.modality == "image" else "EL"
        self.transfer               = "finetune"   # ["finetune", "linear"]
        self.dataset_name           = "nima"       # ["nima", "ncal"]
        if self.dataset_name == "nima":
            self.n_cls              = 1000
            dataset_cls             = NIMAClsDataset
            dataset_root            = "/data/storage/jianwen/N_ImageNet"
            stats_image             = (self.NIMA_MI, self.NIMA_SI)
            stats_event             = (self.NIMA_ME, self.NIMA_SE)
        else:
            self.n_cls              = 100
            dataset_cls             = NCALClsDataset
            dataset_root            = "/data/storage/jianwen/ncaltech101/event_frames"
            stats_image             = (self.NCAL_MI, self.NCAL_SI)
            stats_event             = (self.NCAL_ME, self.NCAL_SE)
        
        # Weights
        ckpt = torch.load("/data/storage/jianwen/cache/ckpts/2026-01-26-22:52_gra/epoch169000_0.6132.pt", map_location="cpu")
        if "event_encoder" in ckpt:
            self.encoder_weight = ckpt["event_encoder"]
        else:
            self.encoder_weight = ckpt

        self.use_projected_encoder = False
        self.proj_dim = 0
        if isinstance(self.encoder_weight, dict):
            has_encoder = any(k.startswith("encoder.") for k in self.encoder_weight.keys())
            has_proj = any(k.startswith("proj.") for k in self.encoder_weight.keys())
            if has_encoder and has_proj:
                self.use_projected_encoder = True
                # Infer projection dim
                for k, v in self.encoder_weight.items():
                    if k == "proj.weight":
                        self.proj_dim = v.shape[0]
                        break
                if self.proj_dim == 0:
                     for k, v in self.encoder_weight.items():
                        if k == "proj.bias":
                             self.proj_dim = v.shape[0]
                             break

        self.decoder_weight         = torch.load("/data/storage/jianwen/cache/ckpts/2026-01-25-00:49_cls/epoch12500_0.6816.pt", map_location="cpu")["decoder"]
        # self.transformer_weight     = None
        self.restore_ckpt           = None

        self.batch_size             = 64
        self.n_workers              = 8
        self.lr                     = 1e-5
        self.wd                     = 0
        self.encoder_lr_mult        = 0.01 if self.encoder_weight is not None else 1
        self.transformer_lr_mult    = 0.01
        self.min_lr                 = 0.
        self.warmup_steps           = 1000
        self.steps                  = 12500  if self.dataset_name == "nima" else 2500  # standard 12500, base 5000
        self.log_every              = 100 if self.dataset_name == "nima" else 10
        self.valid_every            = 1000 if self.dataset_name == "nima" else 100

        self.M, self.S              = stats_image if self.modality == "image" else stats_event
        if self.modality == "image":
            self.train_transform    = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.M, self.S),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomResizedCrop((self.H, self.W), interpolation=transforms.InterpolationMode.BICUBIC),
                                                ])
            self.valid_transform      = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.M, self.S),
                                                transforms.Resize(self.H),
                                                transforms.CenterCrop((self.H, self.W)),
                                                ])
            self.train_dataset    = dataset_cls(root_dir = dataset_root, split="train", transform=self.train_transform)
            self.valid_dataset    = dataset_cls(root_dir = dataset_root, split="valid", transform=self.valid_transform)
            print(f"train_image_dataset length: {len(self.train_dataset)}")
            print(f"valid_image_dataset length: {len(self.valid_dataset)}")

        elif self.modality == "event":
            self.train_transform     = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.M, self.S),
                                                RandomSwapEventRedBlue(type=self.type),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomResizedCrop((self.H, self.W), scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                                                ])
            self.valid_transform      = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.M, self.S),
                                                transforms.Resize((self.H, self.W), interpolation=transforms.InterpolationMode.BICUBIC),
                                                ])
            self.train_dataset    = dataset_cls(root_dir = dataset_root, split="train", transform=self.train_transform, modality=self.modality)
            self.valid_dataset    = dataset_cls(root_dir = dataset_root, split="valid", transform=self.valid_transform, modality=self.modality)
            print(f"train_event_dataset length: {len(self.train_dataset)}")
            print(f"valid_event_dataset length: {len(self.valid_dataset)}")   
        self.dataset_root           = dataset_root
        self.class_names            = getattr(self.train_dataset, "class_names", None)
        if self.class_names:
            self.n_cls              = len(self.class_names)

class OPTConfig(Config):
    def __init__(self):
        super().__init__()
        self.device                 = "cuda:0"
        self.data                   = "event"
        self.type                   = "EO"
        
        self.event_encoder_weight   = torch.load("/data/storage/jianwen/cache/ckpt_matters/gra_mixture_cos_4x.pt", map_location="cpu")["event_encoder"]
        # self.event_encoder_weight   = torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", map_location="cpu") 
        # self.event_encoder_weight   = None
        # self.transformer_weight     = None
        # self.transformer_weight     = torch.load("/data/storage/jianwen/cache/ckpts/2025-08-18-16:52_gpt/epoch99000_0.5560.pt", map_location="cpu")["transformer"] 

        self.encoder_frozen         = False
        self.lr                     = 1e-3
        self.batch_size             = 8
        self.wd                     = 0.
        self.encoder_lr_mult        = 0.1 if self.event_encoder_weight is None else 0.01
        self.transformer_lr_mult    = 0.1

        self.window_size            = 4096
        self.n_embed                = 384
        self.n_head                 = 6
        self.n_layer                = 12

        self.H, self.W              = (448, 644)
        self.origi_H, self.origi_W  = (440, 640)
        self.ME, self.SE, self.MI, self.SI = self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI
        self.scale_range            = (0.5, 2.0)
        self.n_workers              = 8
        self.n_tokens_per_image     = (self.H // self.P) * (self.W // self.P)
        
        self.cat_max_ratio          = 0.75
        self.min_lr                 = 0.
        self.warmup_steps           = 100
        self.steps                  = 10000
        self.log_every              = 100
        self.valid_every            = 1000
        
        self.train_preprocessors    = PairedProcessor([ToTensor(type=self.type),
                                                        Normalize(self.ME, self.SE, self.MI, self.SI, type=self.type),
                                                        RandomHorizontalFlip(p=0.5),
                                                        ResizeKeepRatio(short_side_target_size=self.H, scale_range=self.scale_range, type=self.type),
                                                        PadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=0),
                                                        RandomCrop(crop_size=(self.H, self.W),
                                                                   cat_max_ratio=self.cat_max_ratio,
                                                                   type=self.type)
                                                                   ])
        self.valid_preprocessors    = PairedProcessor([ToTensor(origi_H=self.origi_H, type=self.type),
                                                        Normalize(self.ME, self.SE, self.MI, self.SI, type=self.type),
                                                        ResizeKeepRatio(short_side_target_size=self.H, type=self.type),
                                                        PadToMinSide(target=(self.H, self.W), pad_x1=0, pad_x2=0),
                                                        CenterCrop((self.H, self.W)),])
        self.train_dataset          = DSECOpticalDataset(root_dir = "/data/storage/jianwen/DSEC", split="train", transform=self.train_preprocessors)
        self.valid_dataset          = DSECOpticalDataset(root_dir = "/data/storage/jianwen/DSEC", split="valid", transform=self.valid_preprocessors)
        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"valid dataset size: {len(self.valid_dataset)}")
