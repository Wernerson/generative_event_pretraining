
import sys
import argparse
import math
if "dinov2" not in sys.path:
    sys.path.append("dinov2")
if "dinov3" not in sys.path:
    sys.path.append("dinov3")
import torch._dynamo.config as dynamo_config
dynamo_config.cache_size_limit = 64 
dynamo_config.accumulated_cache_size_limit = 512 
# --------------------------------deterministic setting-------------------------------- #
import numpy as np
seed = 0
np.random.seed(seed)
import os
import time
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
import json
import torchvision
from datetime import datetime
import os
import time
from torch import nn
import torch
import torch.nn.functional as F
from torchinfo import summary
from tqdm import tqdm

from torchvision.models import swin_t, Swin_T_Weights
from dinov2.models.vision_transformer import vit_large, vit_small, vit_base, vit_small_plus
from model import Transformer
from utils import get_lr, get_param_groups, info_nce_loss, kl_loss
from model import Block

from torch.utils.tensorboard import SummaryWriter

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)

class ProjectedEncoder(nn.Module):
    """
    Wraps an encoder with a linear projection layer.
    Exposes underlying encoder methods like forward_features.
    """
    def __init__(self, encoder, in_dim, out_dim):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(in_dim, out_dim)
        self.embed_dim = out_dim # Expose projected dim

    def forward(self, x):
        # Assume encoder returns tensor (CLS)
        x = self.encoder(x)
        return self.proj(x)

    def forward_features(self, x, masks=None):
        return self.encoder.forward_features(x, masks)
    
    def get_last_self_attention(self, x, masks=None):
        return self.encoder.get_last_self_attention(x, masks)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)

class Gra(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        vit_backbone = getattr(config, "vit_backbone", "dinov2")
        self.event_backbone = getattr(config, "event_backbone", "vit")
        teacher_vit = getattr(config, "teacher_vit", None) or config.vit
        
        # Initialize Image Encoder (Frozen DINOv2/DINOv3)
        if vit_backbone == "dinov3":
            from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16
            builders = {
                "small": dinov3_vits16,
                "base": dinov3_vitb16,
            }
            if config.vit not in builders:
                raise ValueError(f"Unsupported DINOv3 vit size '{config.vit}'.")
            builder = builders[config.vit]
            self.image_encoder = builder(pretrained=False)
        else:
            if teacher_vit == "small":
                self.image_encoder  = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4)
            elif teacher_vit == "large":
                self.image_encoder  = vit_large(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4)
            elif teacher_vit == "base":
                self.image_encoder  = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4)
            else:
                raise ValueError(f"Unsupported teacher vit size '{teacher_vit}'.")
        
        # Initialize Event Encoder
        self.target_layers = ['norm']
        if any("patch" in key for key in config.loss_types):
             self.target_layers.append('patch_embed')

        in_chans = getattr(config, 'event_channels', 3)

        if self.event_backbone == "swin":
            weights = Swin_T_Weights.IMAGENET1K_V1
            swin = swin_t(weights=weights)
            # Swin-T features -> norm gives [B, H, W, C]
            self.event_encoder = nn.Sequential(swin.features, swin.norm)
            self.event_encoder.embed_dim = 768
        else:
            # Default ViT initialization for event encoder
            if vit_backbone == "dinov3":
                self.event_encoder = builder(pretrained=False)
            else:
                if config.vit == "small":
                    self.event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4, in_chans=in_chans)
                elif config.vit == "small+":
                    self.event_encoder = vit_small_plus(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, in_chans=in_chans)
                    ckpt_path = "/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth"
                    if os.path.exists(ckpt_path):
                        state_dict = torch.load(ckpt_path, map_location="cpu")
                        msg = self.event_encoder.load_state_dict(state_dict, strict=False)
                        print(f"Initialized vit-small+ (first 12 layers) from {ckpt_path}. Missing keys: {len(msg.missing_keys)}")
                elif config.vit == "large":
                    self.event_encoder = vit_large(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4, in_chans=in_chans)
                elif config.vit == "base":
                    self.event_encoder = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4, in_chans=in_chans)

        # Wrap with projection if needed
        self.dim_proj = nn.Identity()
        if self.event_encoder.embed_dim != self.image_encoder.embed_dim:
            print(f"Wrapping event encoder with projection: {self.event_encoder.embed_dim} -> {self.image_encoder.embed_dim}")
            self.event_encoder = ProjectedEncoder(self.event_encoder, self.event_encoder.embed_dim, self.image_encoder.embed_dim)
            # dim_proj is now handled inside event_encoder.proj
        
        if "nce" in config.loss_types.keys():
            # Project to image encoder dim because event features will be projected to match it
            self.proj = ProjectionHead(self.image_encoder.embed_dim, self.image_encoder.embed_dim * 4, self.image_encoder.embed_dim)
        else:
            self.proj = nn.Identity()
            
        # Projection for Swin token matching (49 -> 256)
        if self.event_backbone == "swin" and getattr(config, "swin_project", False):
            self.token_proj = nn.Linear(49, 256)

        # for name, module in self.event_encoder.named_modules():
        #     print(name)
        # assert False
        
        # Determine the encoder object to load weights into and register hooks on
        if isinstance(self.event_encoder, ProjectedEncoder):
            inner_event_encoder = self.event_encoder.encoder
        else:
            inner_event_encoder = self.event_encoder

        if config.event_encoder_weight is not None and self.event_backbone != "swin":
            # If weight file contains "encoder" + "proj" we might need to load differently
            # For now assume we load ViT weights into inner_event_encoder
            if in_chans != 3:
                # Filter out mismatching shapes
                model_state = inner_event_encoder.state_dict()
                filtered_state_dict = {}
                for k, v in config.event_encoder_weight.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state_dict[k] = v
                        else:
                            print(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state[k].shape}")
                    else:
                        filtered_state_dict[k] = v

                msg = inner_event_encoder.load_state_dict(filtered_state_dict, strict=False)
                print(f"*" * 50 + f" event encoder loaded (strict=False due to {in_chans} channels). Missing keys: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            else:
                inner_event_encoder.load_state_dict(config.event_encoder_weight, strict=True)
                print("*" * 50 + " event encoder loaded")

        if config.image_encoder_weight is not None:
            self.image_encoder.load_state_dict(config.image_encoder_weight, strict=True)
            print("*" * 50 + " image encoder loaded")
        self.event_features, self.image_features = {}, {}
        self.event_hooks, self.image_hooks = [], []
        
        for name, param in self.event_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        
        if self.event_backbone != "swin":
            # Hooks on inner encoder
            for name, module in inner_event_encoder.named_modules():
                if name in self.target_layers:
                    hook_fn = self.create_event_hook(name)
                    handle = module.register_forward_hook(hook_fn)
                    self.event_hooks.append(handle)
        
        for name, module in self.image_encoder.named_modules():
            if name in self.target_layers:
                hook_fn = self.create_image_hook(name)
                handle = module.register_forward_hook(hook_fn)
                self.image_hooks.append(handle)

        for name, param in self.event_encoder.named_parameters():
            param.requires_grad = True
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False

        # training related
        self.train_dsec_dataloader       = torch.utils.data.DataLoader(self.config.train_dsec_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dsec_dataloader       = torch.utils.data.DataLoader(self.config.valid_dsec_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.train_scap_dataloader      = torch.utils.data.DataLoader(self.config.train_scap_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_scap_dataloader      = torch.utils.data.DataLoader(self.config.valid_scap_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.train_nima_dataloader       = torch.utils.data.DataLoader(self.config.train_nima_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_nima_dataloader       = torch.utils.data.DataLoader(self.config.valid_nima_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        # self.train_bddd_dataloader       = torch.utils.data.DataLoader(self.config.train_bddd_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        # self.valid_bddd_dataloader       = torch.utils.data.DataLoader(self.config.valid_bddd_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        # self.train_dd17_dataloader       = torch.utils.data.DataLoader(self.config.train_dd17_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        # self.valid_dd17_dataloader       = torch.utils.data.DataLoader(self.config.valid_dd17_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)    
        self.amp = torch.amp.autocast(device_type = "cuda")
        self.scaler = torch.amp.GradScaler(device = "cuda")
        
        if config.vit == "small+":
            p_low_lr_encoder = []
            p_high_lr_encoder = []
            p_high_lr_other_wd = []
            p_high_lr_other_nowd = []
            
            for name, p in self.named_parameters():
                if not p.requires_grad: continue
                
                if "event_encoder" in name:
                    is_low = False
                    if any(x in name for x in ["patch_embed", "cls_token", "pos_embed"]):
                        is_low = True
                    elif "blocks" in name:
                        try:
                            if int(name.split(".")[2]) < 12: is_low = True
                        except: pass
                    
                    if is_low: p_low_lr_encoder.append(p)
                    else: p_high_lr_encoder.append(p)
                else:
                    if p.ndim < 2 or name.endswith(".bias"): p_high_lr_other_nowd.append(p)
                    else: p_high_lr_other_wd.append(p)
            
            self.optimizer = torch.optim.AdamW([
                {'params': p_low_lr_encoder, 'weight_decay': 0.0, 'lr_mult': 0.1},
                {'params': p_high_lr_encoder, 'weight_decay': 0.0, 'lr_mult': 1.0},
                {'params': p_high_lr_other_wd, 'weight_decay': self.config.wd, 'lr_mult': 1.0},
                {'params': p_high_lr_other_nowd, 'weight_decay': 0.0, 'lr_mult': 1.0},
            ])
        else:
            self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")

        self.n_last = 0
        self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_gra")

    def create_event_hook(self, name):
        """Creates a hook function that saves the output to a dictionary."""
        def hook(model, input, output):
            self.event_features[name] = output
        return hook

    def create_image_hook(self, name):
        """Creates a hook function that saves the output to a dictionary."""
        def hook(model, input, output):
            self.image_features[name] = output
        return hook

    def close(self):
        """Removes all registered hooks. It's important to call this when you're done to avoid memory leaks."""
        for handle in self.event_hooks:
            handle.remove()
        for handle in self.image_hooks:
            handle.remove()

    def forward_encoder(self, event, image, compute_attention=False):
        self.event_features.clear()
        self.image_features.clear()
        
        if self.event_backbone == "swin":
            # Swin output is [B, 7, 7, 768] -> flatten to [B, 49, 768]
            x = self.event_encoder(event)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            event_features = {'norm': x, 'map': x}
        else:
            _ = self.event_encoder(event)
            event_features = {name: feat for name, feat in self.event_features.items()}
            
        with torch.no_grad():
            _ = self.image_encoder(image)
        image_features = {name: feat for name, feat in self.image_features.items()}

        event_attn, image_attn = None, None
        if compute_attention and self.event_backbone != "swin":
            event_attn = self.event_encoder.get_last_self_attention(event)
            with torch.no_grad():
                image_attn = self.image_encoder.get_last_self_attention(image)

        return event_features, image_features, event_attn, image_attn
    
    def get_nce_temperature(self, dataset):
        default_temperature = getattr(self.config, "nce_temperature_default", 0.2)
        image_temperature = getattr(self.config, "nce_temperature_image", 0.2)
        video_temperature = getattr(self.config, "nce_temperature_video", 0.5)
        raw_map = getattr(self.config, "nce_temperature_map", None) or {}
        if hasattr(raw_map, "items"):
            dataset_temperature_map = {k.lower(): v for k, v in raw_map.items()}
        else:
            dataset_temperature_map = {}

        if dataset is None:
            return default_temperature

        dataset_key = dataset.lower()
        if dataset_key in dataset_temperature_map:
            return dataset_temperature_map[dataset_key]

        if dataset_key in {"nima", "n-imagenet"}:
            return getattr(self.config, "nce_temperature_nimagenet", image_temperature)
        if dataset_key in {"dsec", "scap", "eventscape"}:
            return getattr(self.config, "nce_temperature_eventscape", video_temperature)
        return default_temperature

    def _loss_grad_norm(self, loss_term, inputs):
        if not torch.is_grad_enabled():
            return torch.tensor(0.0, device=loss_term.device)
        valid_inputs = [tensor for tensor in inputs if tensor is not None and tensor.requires_grad]
        if not valid_inputs:
            return torch.tensor(0.0, device=loss_term.device)
        grads = torch.autograd.grad(
            loss_term,
            valid_inputs,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        total = None
        for grad in grads:
            if grad is None:
                continue
            grad_norm = grad.norm()
            squared = grad_norm.pow(2)
            total = squared if total is None else total + squared
        if total is None:
            return torch.tensor(0.0, device=loss_term.device)
        return torch.sqrt(total)

    def _sum_loss_terms(self, loss_terms):
        total = None
        for term in loss_terms.values():
            total = term if total is None else total + term
        if total is None:
            return torch.tensor(0.0, device=self.config.device)
        return total

    def _combine_losses(self, loss_terms, loss_inputs):
        if not loss_terms:
            return torch.tensor(0.0, device=self.config.device)
        grad_ratio = getattr(self.config, "loss_grad_ratio", None)
        eps = getattr(self.config, "loss_grad_eps", 1e-8)
        if (
            grad_ratio is None
            or grad_ratio <= 0
            or "cos" not in loss_terms
            or not torch.is_grad_enabled()
        ):
            return self._sum_loss_terms(loss_terms)

        base_norm = self._loss_grad_norm(loss_terms["cos"], loss_inputs.get("cos", {}).values())
        base_value = float(base_norm.detach().item())
        if base_value <= eps:
            return self._sum_loss_terms(loss_terms)

        total_loss = loss_terms["cos"]
        target_norm = grad_ratio * base_value
        if target_norm <= 0:
            return self._sum_loss_terms(loss_terms)

        for name, term in loss_terms.items():
            if name == "cos" or term is None:
                continue
            grad_norm = self._loss_grad_norm(term, loss_inputs.get(name, {}).values())
            grad_value = float(grad_norm.detach().item())
            if grad_value <= eps:
                continue
            scale = target_norm / (grad_value + eps)
            total_loss = total_loss + term * scale
        return total_loss

    def forward(self, event, image, dataset=None):
        compute_attention = any(name == "att" and weight != 0 for name, weight in self.config.loss_types.items())
        event_features, image_features, event_attn, image_attn = self.forward_encoder(
            event, image, compute_attention=compute_attention
        )
        loss_terms = {}
        loss_inputs = {}

        def register_loss_input(loss_name, tensor):
            if tensor is None or not tensor.requires_grad:
                return
            tensor_dict = loss_inputs.setdefault(loss_name, {})
            tensor_dict[id(tensor)] = tensor
        for target_layer in self.target_layers:
            e, i = event_features[target_layer], image_features[target_layer]
            
            if self.event_backbone == "swin" and target_layer == 'norm':
                # If image encoder (ViT) has CLS token, drop it to match Swin's patch-only output
                if i.shape[1] == (image.shape[-2] // 14) * (image.shape[-1] // 14) + 1:
                    i = i[:, 1:, :]

                if not getattr(self.config, "swin_project", False):
                    e = e.mean(dim=1, keepdim=True)
                    i = i.mean(dim=1, keepdim=True)
                else:
                    if e.shape[1] != 49:
                        if 'map' in event_features:
                            e = event_features['map'].permute(0, 3, 1, 2) # [B, C, H, W]
                            e = F.interpolate(e, size=(7, 7), mode='bilinear', align_corners=False)
                            e = e.flatten(2).transpose(1, 2)

                    # e: [B, 49, 768] -> [B, 256, 768] via Linear(49, 256) on dim 1
                    e = self.token_proj(e.transpose(1, 2)).transpose(1, 2)

                    if i.shape[1] != 256:
                        B, N, C = i.shape
                        H_img, W_img = image.shape[-2:]
                        h_p, w_p = H_img // 14, W_img // 14
                        i = i.transpose(1, 2).view(B, C, h_p, w_p)
                        i = F.interpolate(i, size=(16, 16), mode='bilinear', align_corners=False)
                        i = i.flatten(2).transpose(1, 2)

            # Apply projection if we are using a ProjectedEncoder and looking at the final norm layer
            # or if self.dim_proj is active (old way, though now Identity if wrapped)
            if isinstance(self.event_encoder, ProjectedEncoder) and hasattr(self.event_encoder, 'proj'):
                 # We assume both 'norm' and 'patch_embed' might need projection if dims differ?
                 # Usually patch_embed dim == norm dim. Teacher has larger dim.
                 # So we project both.
                 e = self.event_encoder.proj(e)
            else:
                 e = self.dim_proj(e)

            if "nce" in self.config.loss_types.keys() and target_layer == 'norm':
                e_proj = self.proj(e)
            else:
                e_proj = e # Fallback for patch_embed if NCE requested (though unlikely)

            for name, weight in self.config.loss_types.items():
                if weight == 0:
                    continue

                # Filter losses by layer
                if target_layer == 'norm':
                     if 'patch' in name: continue
                elif target_layer == 'patch_embed':
                     if 'patch' not in name: continue
                else:
                     # Unknown layer, skip or process? Assume only norm/patch_embed for now
                     pass

                effective_name = name.replace('_patch', '')

                if effective_name == "nce":
                    temperature = self.get_nce_temperature(dataset)
                    ei_loss = info_nce_loss(e_proj.mean(dim=1), i.mean(dim=1), temperature=temperature)
                elif effective_name == "mse":
                    ei_loss = F.mse_loss(e, i)
                elif effective_name == "cos":
                    ei_loss = (1 - F.cosine_similarity(e, i, dim=-1).mean())
                elif effective_name == "kld":
                    ei_loss = kl_loss(e, i, T=2.0)
                elif effective_name == "att":
                    if event_attn is None or image_attn is None:
                        raise RuntimeError("Attention features were not computed despite att loss being requested.")
                    ei_loss = F.mse_loss(event_attn, image_attn)
                else:
                    continue
                if effective_name == "att":
                    register_loss_input(name, event_attn)
                else:
                    register_loss_input(name, e)
                weighted_loss = weight * ei_loss
                if name in loss_terms:
                    loss_terms[name] = loss_terms[name] + weighted_loss
                else:
                    loss_terms[name] = weighted_loss
        loss = self._combine_losses(loss_terms, loss_inputs)
        return event, image, loss, loss_terms
    
    def train_step(self, event, image, global_step, dataset):
        self.train()
        event, image = event.to(self.config.device), image.to(self.config.device)
        t0 = time.time()
        
        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]
            
        with self.amp:
            event, image, loss, loss_terms = self.forward(event, image, dataset=dataset)
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.0,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=2.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        t1 = time.time()
        if (global_step + 1) % self.config.log_every == 0:
            self.writer.add_scalar("loss/train", loss.item(), global_step + 1)
            # Log individual loss terms
            loss_str = f"loss: {loss.item() :.4f}"
            for k, v in loss_terms.items():
                if isinstance(v, torch.Tensor):
                    val = v.item()
                else:
                    val = v
                self.writer.add_scalar(f"loss/train_{k}", val, global_step + 1)
                loss_str += f", {k}: {val:.4f}"

            dt = (t1 - t0)
            num_tokens_per_secend =  event.shape[0] * event.shape[2] * event.shape[3] / self.config.P / self.config.P / dt
            print(f"step: {global_step + 1}, lr: {current_lr :.8f}, {loss_str}, grad_norm: {grad_norm:.4f}, input: {(event.shape, image.shape)}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")
            ME, SE, MI, SI = torch.tensor(self.config.NIMA_ME, device=self.config.device)[None, :, None, None], torch.tensor(self.config.NIMA_SE, device=self.config.device)[None, :, None, None], torch.tensor(self.config.NIMA_MI, device=self.config.device)[None, :, None, None], torch.tensor(self.config.NIMA_SI, device=self.config.device)[None, :, None, None]
            if event.shape[1] == 3: # Only visualize if 3 channels
                torchvision.utils.save_image(event * SE + ME, f"src/vis_gra_train_event_{dataset}.png")
            torchvision.utils.save_image(image * SI + MI, f"src/vis_gra_train_image_{dataset}.png")

    def get_batch(self, step):
        if self.config.gra_style == "mixture":
            number = random.random()
            if number <= 0.7:
                dataset = "NIMA"
            elif number < 0.9:
                dataset = "DSEC"
            else:
                dataset = "SCAP"

            if dataset == "DSEC":
                event, image = next(self.dsec_iter)
                H, W = self.config.DSEC_H, self.config.DSEC_W
            elif dataset == "SCAP":
                event, image = next(self.scap_iter)
                H, W = self.config.SCAP_H, self.config.SCAP_W
            elif dataset == "NIMA":
                event, image = next(self.nima_iter)
                H, W = self.config.NIMA_H, self.config.NIMA_W
            # elif dataset == "BDDD":
            #     event, image = next(self.bddd_iter)
            #     H, W = self.config.BDDD_H, self.config.BDDD_W
            elif dataset == "DD17":
                event, image = next(self.dd17_iter)
                H, W = self.config.DD17_H, self.config.DD17_W
            else:
                raise ValueError("Unknown dataset")
            self.n_last -= 1
        
        # elif self.config.gra_style == "pretrain":
        #     ratio = step / self.config.steps
        #     if ratio < 0.4:
        #         event, image = next(self.bddd_iter)
        #         H, W = self.config.BDDD_H, self.config.BDDD_W
        #     else:
        #         event, image = next(self.dsec_iter)
        #         H, W = self.config.DSEC_H, self.config.DSEC_W

        elif self.config.gra_style == "pure_dsec":
            event, image = next(self.dsec_iter)
            H, W = self.config.DSEC_H, self.config.DSEC_W
            dataset = "DSEC"
        
        elif self.config.gra_style == "pure_scap":
            event, image = next(self.scap_iter)
            H, W = self.config.SCAP_H, self.config.SCAP_W
            dataset = "SCAP"
        
        elif self.config.gra_style == "pure_nima":
            event, image = next(self.nima_iter)
            H, W = self.config.NIMA_H, self.config.NIMA_W
            dataset = "NIMA"
        
        # elif self.config.gra_style == "pure_bddd":
        #     event, image = next(self.bddd_iter)
        #     H, W = self.config.BDDD_H, self.config.BDDD_W
        #     dataset = "BDDD"
        
        elif self.config.gra_style == "pure_dd17":
            event, image = next(self.dd17_iter)
            H, W = self.config.DD17_H, self.config.DD17_W
            dataset = "DD17"

        return event, image, H, W, dataset

    def start(self):
        self.dsec_iter = iter(self.train_dsec_dataloader)
        self.scap_iter = iter(self.train_scap_dataloader)
        self.nima_iter = iter(self.train_nima_dataloader)
        # self.bddd_iter = iter(self.train_bddd_dataloader)
        # self.dd17_iter = iter(self.train_dd17_dataloader)

        if self.config.restore_ckpt is not None:
            ckpt = self.config.restore_ckpt
            re_step = ckpt["epoch"] + 1
            optim = ckpt["optimizer"]
            model = ckpt["event_encoder"]
            self.event_encoder.load_state_dict(model, strict=True)
            if hasattr(self, "token_proj") and "token_proj" in ckpt:
                self.token_proj.load_state_dict(ckpt["token_proj"])
            if not isinstance(self.dim_proj, nn.Identity) and "dim_proj" in ckpt:
                self.dim_proj.load_state_dict(ckpt["dim_proj"])
            self.optimizer.load_state_dict(optim)
        else:
            re_step = 0
        # self.validate(0)
        self.train()
        # exit()

        for step in range(re_step, self.config.steps):
            try:
                event, image, H, W, dataset = self.get_batch(step)
            except StopIteration:
                self.dsec_iter = iter(self.train_dsec_dataloader)
                self.scap_iter = iter(self.train_scap_dataloader)
                self.nima_iter = iter(self.train_nima_dataloader)
                # self.bddd_iter = iter(self.train_bddd_dataloader)
                # self.dd17_iter = iter(self.train_dd17_dataloader)
                event, image, H, W, dataset = self.get_batch(step)

            # if self.training:
            #     random_number = random.random()
            #     # When using 'project' match method, we must ensure input is 224x224 
            #     # so that Swin-T outputs 49 tokens and DINOv2 (P=14) outputs 256 tokens.
            #     force_resize = getattr(self.config, "swin_project", False)
            #     if force_resize or random_number < 0.7:                    
            #         H = 224; W = 224
            #         event = F.interpolate(event, size=(H, W), mode='bilinear')
            #         image = F.interpolate(image, size=(H, W), mode='bilinear')
            #         # Update tokens count estimate for logging if needed, though strictly strictly n_tokens_per_image is config-based
            #         # self.config.n_tokens_per_image = event.shape[0] * (H // self.config.P) * (W // self.config.P)

            if step == 0:
                summary(self, input_data=(event, image), device=self.config.device, depth=2)
                os.makedirs("src/runs", exist_ok=True)
            self.train_step(event, image, step, dataset)

            if (step + 1) % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                self.validate(step)
                self.train()
    
    @ torch.no_grad()
    def validate(self, step):
        self.eval()

        valid_loss = 0.0
        valid_loss_terms = {}
        count = 0
        for i, (event, image) in enumerate(tqdm(self.valid_dsec_dataloader)):
            if i >= 32 and (step+1) != self.config.steps and step!=0:
                break
            event, image = event.to(self.config.device), image.to(self.config.device)
            event, image, loss, loss_terms = self.forward(event, image, dataset="DSEC")
            valid_loss += loss.item()
            for k, v in loss_terms.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                valid_loss_terms[k] = valid_loss_terms.get(k, 0.0) + val
            count += 1
            
        valid_loss /= count
        for k in valid_loss_terms:
            valid_loss_terms[k] /= count
            
        print(f"dsec step: {step}, valid loss: {valid_loss:.4f}, breakdown: {valid_loss_terms}")
        self.writer.add_scalar("loss/valid_dsec", valid_loss, step)
        for k, v in valid_loss_terms.items():
            self.writer.add_scalar(f"loss/valid_dsec_{k}", v, step)

        # valid_loss = 0.0
        # for i, (event, image) in enumerate(tqdm(self.valid_bddd_dataloader)):
        #     if i >= 32 and (step+1) != self.config.steps and step!=0:
        #         break
        #     event, image = event.to(self.config.device), image.to(self.config.device)
        #     event, image, loss = self.forward(event, image)
        #     valid_loss += loss.item()
        # valid_loss /= (i + 1)
        # print(f"bddd step: {step}, valid loss: {valid_loss:.4f}")
        # self.writer.add_scalar("loss/valid_bddd", valid_loss, step)

        valid_loss = 0.0
        count = 0
        for i, (event, image) in enumerate(tqdm(self.valid_scap_dataloader)):
            if i >= 32 and (step+1) != self.config.steps and step!=0:
                break
            event, image = event.to(self.config.device), image.to(self.config.device)
            event, image, loss, _ = self.forward(event, image, dataset="SCAP")
            valid_loss += loss.item()
            count += 1
        valid_loss /= count
        print(f"scap step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_scap", valid_loss, step)

        # valid_loss = 0.0
        # for i, (event, image) in enumerate(tqdm(self.valid_dd17_dataloader)):
        #     if i >= 32 and (step+1) != self.config.steps and step!=0:
        #         break
        #     event, image = event.to(self.config.device), image.to(self.config.device)
        #     event, image, loss = self.forward(event, image)
        #     valid_loss += loss.item()
        # valid_loss /= (i + 1)
        # print(f"dd17 step: {step}, valid loss: {valid_loss:.4f}")
        # self.writer.add_scalar("loss/valid_dd17", valid_loss, step)

        valid_loss = 0.0
        count = 0
        for i, (event, image) in enumerate(tqdm(self.valid_nima_dataloader)):
            if i >= 32 and (step+1) != self.config.steps and step!=0:
                break
            event, image = event.to(self.config.device), image.to(self.config.device)
            event, image, loss, _ = self.forward(event, image, dataset="NIMA")
            valid_loss += loss.item()
            count += 1
        valid_loss /= count
        print(f"nima step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_nima", valid_loss, step)

        param_dict = {
            "event_encoder": self.event_encoder.state_dict(),
            # "proj": self.proj.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        if hasattr(self, "token_proj"):
            param_dict["token_proj"] = self.token_proj.state_dict()
        if not isinstance(self.dim_proj, nn.Identity):
            param_dict["dim_proj"] = self.dim_proj.state_dict()
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_gra"
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step + 1}_{valid_loss:.4f}.pt"))


def _parse_losses_type(arg: str):
    """Allow JSON dict, comma separated tokens, or single loss name."""
    if arg is None:
        return None
    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(k): float(v) for k, v in parsed.items()}
    losses = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, weight = item.split(":", 1)
            losses[name.strip()] = float(weight.strip())
        else:
            losses[item] = 1.0
    if not losses:
        raise ValueError(f"Unable to parse --losses_type value: {arg}")
    return losses

if __name__ == "__main__":
    from config import GraConfig
    parser = argparse.ArgumentParser(description="Train Gra with selectable loss weights.")
    parser.add_argument(
        "--losses_type",
        type=str,
        default=None,
        help="Override loss configuration (JSON dict or comma separated pairs, e.g. mse or mse:1.0,cos:0.5).",
    )
    args = parser.parse_args()
    config = GraConfig()
    override_losses = _parse_losses_type(args.losses_type)
    if override_losses is not None:
        config.loss_types = override_losses
        print(f"Overriding loss_types with CLI flag: {config.loss_types}")
    model = Gra(config).to(config.device)
    # model.compile()
    model.start()
    print("Training completed.")
