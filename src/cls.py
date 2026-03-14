import sys
if "dinov2" not in sys.path:
    sys.path.append("dinov2")
if "dinov3" not in sys.path:
    sys.path.append("dinov3")
# --------------------------------deterministic setting-------------------------------- #
import einops
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
from datetime import datetime
import os
import time
import torchvision
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from tqdm import tqdm
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
import ast
import json

from dinov2.models.vision_transformer import vit_small, vit_base
from model import Block, Transformer
from utils import accuracy, get_lr, get_param_groups

from torch.utils.tensorboard import SummaryWriter

class Cls(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        vit_backbone = getattr(config, "vit_backbone", "dinov2")
        vit_size = getattr(config, "vit", "small")
        if vit_backbone == "dinov3":
            from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16

            builders = {
                "small": dinov3_vits16,
                "base": dinov3_vitb16,
            }
            if vit_size not in builders:
                raise ValueError(f"Unsupported DINOv3 vit size '{vit_size}'.")
            self.encoder = builders[vit_size](pretrained=False)
        else:
            if vit_size == "small":
                self.encoder = vit_small(
                    patch_size=14,
                    img_size=518,
                    block_chunks=0,
                    init_values=1e-6,
                    num_register_tokens=4,
                )
            elif vit_size == "base":
                self.encoder = vit_base(
                    patch_size=14,
                    img_size=518,
                    block_chunks=0,
                    init_values=1e-6,
                    num_register_tokens=4,
                )
            else:
                raise ValueError(f"Unsupported vit size '{vit_size}'.")

        if getattr(config, "use_projected_encoder", False):
            from gra import ProjectedEncoder
            out_dim = getattr(config, "proj_dim", config.n_embed)
            if out_dim == 0: out_dim = config.n_embed
            print(f"Wrapping encoder with ProjectedEncoder: {self.encoder.embed_dim} -> {out_dim}")
            self.encoder = ProjectedEncoder(self.encoder, self.encoder.embed_dim, out_dim)

        self.transformer    = nn.ModuleDict(dict(   modality_embed = nn.Embedding(5, config.n_embed),
                                                    pos_embed = nn.Embedding(config.window_size, config.n_embed),
                                                    blocks = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
                                                    norm = nn.LayerNorm(config.n_embed),
                                                )) if hasattr(config, "transformer_weight") else nn.Identity()
        self.decoder = nn.Linear(config.n_embed, config.n_cls)
        self.class_map = None  # lazily loaded class id -> name mapping

        if config.encoder_weight is not None:
            self.encoder.load_state_dict(config.encoder_weight, strict=True)
            self.decoder.load_state_dict(config.decoder_weight, strict=True)
            print("*" * 50 + "encoder loaded")
        if hasattr(config, "transformer_weight") and config.transformer_weight is not None:
            self.transformer.load_state_dict(config.transformer_weight, strict=True)
            print("*" * 50 + "transformer loaded")

        # training related
        self.train_loader = torch.utils.data.DataLoader(
            self.config.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.n_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.config.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.n_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.amp = torch.amp.autocast(device_type="cuda")
        self.scaler = torch.amp.GradScaler(device="cuda")
        self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd, encoder_lr_mult=self.config.encoder_lr_mult, transformer_lr_mult=self.config.transformer_lr_mult))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")

        if self.config.transfer == "linear":
            for name, params in self.encoder.named_parameters():
                params.requires_grad = False

        os.makedirs("src/runs", exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_cls")

    def _load_class_mapping(self):
        """Load class id->name mapping from cached JSON or raw mapping.txt.

        Priority:
        1) src/imagenet_mapping.json
        2) /data/storage/jianwen/N_ImageNet/mapping.txt (Python dict literal)
        If neither is available, returns an empty dict and we fallback to id strings.
        """
        if self.class_map is not None:
            return self.class_map

        class_names = getattr(self.config, "class_names", None)
        if class_names:
            self.class_map = {idx: name for idx, name in enumerate(class_names)}
            return self.class_map

        cache_json = os.path.join(os.path.dirname(__file__), "imagenet_mapping.json")
        if os.path.exists(cache_json):
            try:
                with open(cache_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # keys may be strings in JSON; normalize to int keys
                self.class_map = {int(k): v for k, v in data.items()}
                return self.class_map
            except Exception:
                pass

        mapping_txt = "/data/storage/jianwen/N_ImageNet/mapping.txt"
        if os.path.exists(mapping_txt):
            try:
                with open(mapping_txt, "r", encoding="utf-8") as f:
                    txt = f.read()
                # mapping.txt looks like a Python dict literal
                parsed = ast.literal_eval(txt)
                # Normalize names: keep the first alias before comma for brevity
                norm = {}
                for k, v in parsed.items():
                    try:
                        key = int(k)
                    except Exception:
                        # k may already be int
                        key = k
                    name = str(v)
                    short = name.split(",")[0].strip()
                    norm[key] = short
                self.class_map = norm
                # Cache as JSON for faster later runs
                try:
                    with open(cache_json, "w", encoding="utf-8") as f:
                        json.dump({int(k): v for k, v in norm.items()}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                return self.class_map
            except Exception:
                pass

        # Fallback: empty mapping
        self.class_map = {}
        return self.class_map

    def _annotate_with_text(self, img_tensor: torch.Tensor, caption: str) -> torch.Tensor:
        """Append a black bar with white text to the bottom of an image tensor [C,H,W] in [0,1]."""
        # Ensure 3 channels for visualization
        if img_tensor.dim() != 3:
            raise ValueError("img_tensor must be [C,H,W]")
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif img_tensor.shape[0] == 2:
            img_tensor = torch.cat([img_tensor, img_tensor[0:1, ...]], dim=0)  # pad to 3
            img_tensor = img_tensor[:3]
        elif img_tensor.shape[0] > 3:
            img_tensor = img_tensor[:3]

        pil_img = TF.to_pil_image(img_tensor.clamp(0, 1))
        w, h = pil_img.size
        # Choose bar height based on image height
        bar_h = max(32, int(h * 0.12))
        bar = Image.new("RGB", (w, bar_h), color=(0, 0, 0))

        # Compose
        combined = Image.new("RGB", (w, h + bar_h))
        combined.paste(pil_img, (0, 0))
        combined.paste(bar, (0, h))

        # Draw text centered vertically in the bar
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text = caption
        # If too long, truncate gracefully
        max_chars = 64
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        # Measure text size (Pillow 10 removed textsize/getsize; prefer textbbox/textlength)
        if font is not None:
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            elif hasattr(draw, "textsize"):
                # Fallback for older Pillow
                tw, th = draw.textsize(text, font=font)
            else:
                # Last resort: approximate width via textlength, fixed height
                tl = draw.textlength(text, font=font) if hasattr(draw, "textlength") else len(text) * 6
                tw, th = int(tl), 12
        else:
            tl = draw.textlength(text) if hasattr(draw, "textlength") else len(text) * 6
            tw, th = int(tl), 12
        x = 6
        y = h + (bar_h - th) // 2
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return TF.to_tensor(combined)

    def forward_encoder(self, x):
        x = self.encoder.forward_features(x)

        patch_tokens = x["x_norm_patchtokens"]
        cls_token = x["x_norm_clstoken"]

        if hasattr(self.encoder, "proj"):
            patch_tokens = self.encoder.proj(patch_tokens)
            cls_token = self.encoder.proj(cls_token)

        if hasattr(self.config, "transformer_weight"):
            return patch_tokens, cls_token
        else:
            return cls_token

    def forward_transformer(self, x):
        x, x_cls = x
        B, T, C = x.shape
        ids = torch.ones(B, T, dtype=torch.int64, device=x.device) * 4  # 1 for image, 2 for event
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.pos_embed(pos)
        modality_emb = self.transformer.modality_embed(ids)
        x = x + pos_emb + modality_emb
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x)
        x = self.transformer.norm(x)
        return x.mean(dim=1) + x_cls

    def forward_decoder(self, x):
        x = self.decoder(x)
        return x
    
    def forward_loss(self, x, y):
        loss = F.cross_entropy(x, y, reduction='mean')
        return loss

    def forward(self, x, y):
        x = self.forward_encoder(x)
        if hasattr(self.config, "transformer_weight"):
            x = self.forward_transformer(x)
        x = self.forward_decoder(x)
        loss = self.forward_loss(x, y)
        return x, loss

    @torch.no_grad()
    def visualize(self, x, logits, y, modality, name, nrow=8):
        # Denormalize inputs for visualization
        imgs = x.detach().clone()
        M, S = torch.tensor(self.config.M, device=imgs.device)[None, :, None, None], torch.tensor(self.config.S, device=imgs.device)[None, :, None, None]
        imgs = imgs * S + M
        imgs = imgs.clamp(0, 1)

        # Prepare labels
        class_map = self._load_class_mapping()
        pred_ids = logits.argmax(dim=1).detach().cpu().tolist()
        gt_ids = y.detach().cpu().tolist()

        annotated = []
        count = min(nrow, imgs.shape[0])
        for i in range(count):
            pred_id = int(pred_ids[i])
            gt_id = int(gt_ids[i])
            pred_name = class_map.get(pred_id, str(pred_id))
            gt_name = class_map.get(gt_id, str(gt_id))
            caption = f"pred: {pred_name} ({pred_id}) | gt: {gt_name} ({gt_id})"
            annotated.append(self._annotate_with_text(imgs[i].cpu(), caption))

        if len(annotated) == 0:
            return
        grid = torchvision.utils.make_grid(torch.stack(annotated, dim=0), nrow=nrow, padding=2)
        os.makedirs("src", exist_ok=True)
        torchvision.utils.save_image(grid, f"src/{name}_{modality}.png")
    
    def train_step(self, x, y, global_step):
        self.train()
        x = x.to(self.config.device)
        y = y.to(self.config.device)
        t0 = time.time()

        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]

        with self.amp:
            pred, loss = self.forward(x, y)
        self.outputs.append(pred)
        self.targets.append(y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.0,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        t1 = time.time()
        if (global_step) % self.config.log_every == 0:
            outputs = torch.cat(self.outputs, dim=0)
            targets = torch.cat(self.targets, dim=0)
            acc = accuracy(outputs, targets, topk=(1, 5))
            top1, top5 = acc
            dt = (t1 - t0)
            self.visualize(x, pred, y, modality=self.config.modality, name="vis_cls_train")
            num_tokens_per_secend =  self.config.batch_size * self.config.n_tokens_per_image / dt
            print(f"step: {global_step}, lr: {current_lr :.8f}, loss: {loss.item() :.4f}, acc@1: {top1 :.4f}, acc@5: {top5 :.4f}, grad_norm: {grad_norm:.4f}, input: {(x.shape)}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")
            self.writer.add_scalar("loss/train", loss.item(), global_step)
            self.writer.add_scalar("top1/train", top1, global_step)
            self.writer.add_scalar("top5/train", top5, global_step)
        self.h, self.w = None, None

    def start(self):
        self.validate(0)
        train_iter = iter(self.train_loader)
        re_step = 0
        if self.config.restore_ckpt is not None:
            ckpt = self.config.restore_ckpt
            re_step = ckpt["epoch"] + 1
            optim = ckpt["optimizer"]
            encoder = ckpt["encoder"]
            decoder = ckpt["decoder"]
            self.decoder.load_state_dict(decoder, strict=True)
            self.encoder.load_state_dict(encoder, strict=True)
            self.optimizer.load_state_dict(optim)

        for step in range(re_step, self.config.steps):
            try:
                x, y = next(train_iter)
                self.outputs, self.targets = [], []
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)
                self.outputs, self.targets = [], []

            if step == 0:
                summary(self, input_data=(x, y), device=self.config.device, depth=2)
            self.train_step(x, y, step)

            if (step + 1) % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                self.validate(step)
    
    @ torch.no_grad()
    def validate(self, step):
        self.eval()
        valid_loss = 0.0
        outputs, targets = [], []
        for i, (x, y) in enumerate(tqdm(self.valid_dataloader)):
            x, y = x.to(self.config.device), y.to(self.config.device)
            pred, loss = self.forward(x, y)
            outputs.append(pred)
            targets.append(y)

            valid_loss += loss.item()
            # if i >= 32 and ((step + 1) != self.config.steps or step!=0):
            #     break
        valid_loss /= (i + 1)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        acc = accuracy(outputs, targets, topk=(1, 5))
        top1, top5 = acc
        print(f"step: {step}, valid loss: {valid_loss:.4f}, top1: {top1:.4f}, top5: {top5:.4f}")
        self.writer.add_scalar("loss/valid", valid_loss, step)
        self.writer.add_scalar("top1/valid", top1, step)
        self.writer.add_scalar("top5/valid", top5, step)
        self.visualize(x, pred, y, modality=self.config.modality, name="vis_cls_valid")
        
        param_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_cls"
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step + 1}_{valid_loss:.4f}.pt"))
        
if __name__ == "__main__":
    from config import CLSConfig
    config = CLSConfig()
    model = Cls(config).to(config.device)
    model.start()
    print("Training completed.")
