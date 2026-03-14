import sys
sys.path.append("dinov2")
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

from dinov2.models.vision_transformer import vit_small
from model import Block, Transformer
from utils import get_lr, get_param_groups

from torch.utils.tensorboard import SummaryWriter

class Rec(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.event_encoder  = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6)
        self.image_encoder  = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6)

        self.decoder        = nn.ModuleDict(dict(   project = nn.Linear(self.event_encoder.embed_dim, config.n_embed) if config.n_embed != self.event_encoder.embed_dim else nn.Identity(),
                                                    blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                                    norm    = nn.LayerNorm(config.n_embed),
                                                    head    = nn.Linear(config.n_embed, 3 * config.P ** 2)))

        if config.event_encoder_weight is not None:
            self.event_encoder.load_state_dict(config.event_encoder_weight, strict=True)
            print("*" * 50 + " event encoder loaded")
        if config.image_encoder_weight is not None:
            self.image_encoder.load_state_dict(config.image_encoder_weight, strict=True)
            print("*" * 50 + " image encoder loaded")
        for name, param in self.event_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        
        self.train_dsec_dataloader       = torch.utils.data.DataLoader(self.config.train_dsec_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dsec_dataloader       = torch.utils.data.DataLoader(self.config.valid_dsec_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.train_scap_dataloader      = torch.utils.data.DataLoader(self.config.train_scap_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_scap_dataloader      = torch.utils.data.DataLoader(self.config.valid_scap_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.train_nima_dataloader       = torch.utils.data.DataLoader(self.config.train_nima_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_nima_dataloader       = torch.utils.data.DataLoader(self.config.valid_nima_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.train_bddd_dataloader       = torch.utils.data.DataLoader(self.config.train_bddd_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_bddd_dataloader       = torch.utils.data.DataLoader(self.config.valid_bddd_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.train_dd17_dataloader       = torch.utils.data.DataLoader(self.config.train_dd17_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dd17_dataloader       = torch.utils.data.DataLoader(self.config.valid_dd17_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)  
        
        self.amp = torch.amp.autocast(device_type = "cuda")
        self.scaler = torch.amp.GradScaler(device = "cuda")
        self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")

    def get_batch(self, step):
        if self.config.style == "mixture":
            number = random.random()
            if number < 1.:
                dataset = "DSEC"
            # elif number < 0.85:
            #     dataset = "NIMA"
            # elif number <= 0.95:
            #     dataset = "DD17"
            # else:
            #     dataset = "SCAP"

            if dataset == "DSEC":
                event, image = next(self.dsec_iter)
                H, W = self.config.DSEC_H, self.config.DSEC_W
            elif dataset == "SCAP":
                event, image = next(self.scap_iter)
                H, W = self.config.SCAP_H, self.config.SCAP_W
            elif dataset == "NIMA":
                event, image = next(self.nima_iter)
                H, W = self.config.NIMA_H, self.config.NIMA_W
            elif dataset == "BDDD":
                event, image = next(self.bddd_iter)
                H, W = self.config.BDDD_H, self.config.BDDD_W
            elif dataset == "DD17":
                event, image = next(self.dd17_iter)
                H, W = self.config.DD17_H, self.config.DD17_W
            else:
                raise ValueError("Unknown dataset")
        
        elif self.config.style == "pretrain":
            ratio = step / self.config.steps
            if ratio < 0.4:
                event, image = next(self.bddd_iter)
                H, W = self.config.BDDD_H, self.config.BDDD_W
            else:
                event, image = next(self.dsec_iter)
                H, W = self.config.DSEC_H, self.config.DSEC_W

        elif self.config.style == "pure_dsec":
            event, image = next(self.dsec_iter)
            H, W = self.config.DSEC_H, self.config.DSEC_W
        
        elif self.config.style == "pure_scap":
            event, image = next(self.scap_iter)
            H, W = self.config.SCAP_H, self.config.SCAP_W
        
        elif self.config.style == "pure_nima":
            event, image = next(self.nima_iter)
            H, W = self.config.NIMA_H, self.config.NIMA_W
        
        elif self.config.style == "pure_bddd":
            event, image = next(self.bddd_iter)
            H, W = self.config.BDDD_H, self.config.BDDD_W
        
        elif self.config.style == "pure_dd17":
            event, image = next(self.dd17_iter)
            H, W = self.config.DD17_H, self.config.DD17_W

        return event, image, H, W
    
    def forward_encoder(self, data, modality):
        if modality == "event":
            z = self.event_encoder.forward_features(data)["x_norm_patchtokens"]
        elif modality == "image":
            z = self.image_encoder.forward_features(data)["x_norm_patchtokens"]
        return z

    def forward_decoder(self, z, H, W):
        z = self.decoder.project(z)
        for blk in self.decoder.blocks:
            z = blk(z)
        z = self.decoder.norm(z)
        z = self.decoder.head(z)
        h, w = H // self.config.P, W // self.config.P
        rec = einops.rearrange(z, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=self.config.P, p2=self.config.P, c=3)
        return rec
    
    def forward_loss(self, rec, ori):
        loss = F.mse_loss(rec, ori, reduction='mean')
        return loss

    def forward(self, data, modality, H, W):
        z = self.forward_encoder(data, modality)
        rec = self.forward_decoder(z, H, W)
        loss = self.forward_loss(rec, data)
        return rec, loss

    @torch.no_grad()
    def visualize(self, rec, ori, M, S, name, nrow=8):
        M, S = torch.tensor(M, device=self.config.device)[None, :, None, None], torch.tensor(S, device=self.config.device)[None, :, None, None]
        rec, ori = (rec * S + M), (ori * S + M)
        torchvision.utils.save_image(torch.cat([ori[:nrow], rec[:nrow]], dim=0), f"src/{name}.png", nrow=nrow)
    
    def train_step(self, data, modality, H, W, global_step):
        self.train()
        data = data.to(self.config.device)
        t0 = time.time()
        
        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]
            
        with self.amp:
            rec, loss = self.forward(data, modality, H, W)
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.0,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        self.writer.add_scalar("loss/train", loss.item(), global_step)
        t1 = time.time()

        if (global_step) % self.config.log_every == 0:
            dt = (t1 - t0)
            if modality == "event":
                M, S = self.config.NIMA_ME, self.config.NIMA_SE
            else:
                M, S = self.config.NIMA_MI, self.config.NIMA_SI
            self.visualize(rec, data, M, S, name="vis_rec_train")
            num_tokens_per_secend =  self.config.batch_size * self.config.dsec_n_tokens_per_image / dt
            print(f"step: {global_step}, lr: {current_lr :.8f}, loss: {loss.item() :.4f}, grad_norm: {grad_norm:.4f}, input: {(data.shape)}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")


    def start(self):
        self.dsec_iter = iter(self.train_dsec_dataloader)
        self.scap_iter = iter(self.train_scap_dataloader)
        self.nima_iter = iter(self.train_nima_dataloader)
        self.bddd_iter = iter(self.train_bddd_dataloader)
        self.dd17_iter = iter(self.train_dd17_dataloader)

        for step in range(self.config.steps):
            try:
                event, image, H, W = self.get_batch(step)
            except StopIteration:
                self.dsec_iter = iter(self.train_dsec_dataloader)
                self.scap_iter = iter(self.train_scap_dataloader)
                self.nima_iter = iter(self.train_nima_dataloader)
                self.bddd_iter = iter(self.train_bddd_dataloader)
                self.dd17_iter = iter(self.train_dd17_dataloader)
                event, image, H, W = self.get_batch(step)
            if random.random() < 0.5:
                data = event
                modality = "event"
            else:
                data = image
                modality = "image"

            if step == 0:
                # summary(self, input_data=(data, modality, H, W), device=self.config.device, depth=2)
                os.makedirs("src/runs", exist_ok=True)
                self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_rec")
            self.train_step(data, modality, H, W, step)

            if (step + 1) % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                self.validate(step)
    
    @ torch.no_grad()
    def validate(self, step):
        self.eval()

        # ---------------------- DSEC ----------------------
        valid_loss = 0.0
        H, W = self.config.DSEC_H, self.config.DSEC_W
        for i, (event, image) in enumerate(tqdm(self.valid_dsec_dataloader)):
            event, image = event.to(self.config.device), image.to(self.config.device)
            if random.random() < 0.5:
                modality = "event"
                rec, loss = self.forward(event, modality, H, W)
            else:
                modality = "image"
                rec, loss = self.forward(image, modality, H, W)
            valid_loss += loss.item()
            if i >= 32 and (step+1) != self.config.steps and step != 0:
                break
        valid_loss /= (i + 1)
        if modality == "event":
            M, S = self.config.DSEC_ME, self.config.DSEC_SE
            data = event
        else:
            M, S = self.config.DSEC_MI, self.config.DSEC_SI
            data = image
        self.visualize(rec, data, M, S, name="vis_rec_valid_dsec")
        print(f"dsec step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_dsec", valid_loss, step)

        # ---------------------- BDDD ----------------------
        valid_loss = 0.0
        H, W = self.config.BDDD_H, self.config.BDDD_W
        for i, (event, image) in enumerate(tqdm(self.valid_bddd_dataloader)):
            event, image = event.to(self.config.device), image.to(self.config.device)
            if random.random() < 0.5:
                modality = "event"
                rec, loss = self.forward(event, modality, H, W)
            else:
                modality = "image"
                rec, loss = self.forward(image, modality, H, W)
            valid_loss += loss.item()
            if i >= 32 and (step+1) != self.config.steps and step != 0:
                break
        valid_loss /= (i + 1)
        if modality == "event":
            M, S = self.config.BDDD_ME, self.config.BDDD_SE
            data = event
        else:
            M, S = self.config.BDDD_MI, self.config.BDDD_SI
            data = image
        self.visualize(rec, data, M, S, name="vis_rec_valid_bddd")
        print(f"bddd step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_bddd", valid_loss, step)

        # ---------------------- SCAP ----------------------
        valid_loss = 0.0
        H, W = self.config.SCAP_H, self.config.SCAP_W
        for i, (event, image) in enumerate(tqdm(self.valid_scap_dataloader)):
            event, image = event.to(self.config.device), image.to(self.config.device)
            if random.random() < 0.5:
                modality = "event"
                rec, loss = self.forward(event, modality, H, W)
            else:
                modality = "image"
                rec, loss = self.forward(image, modality, H, W)
            valid_loss += loss.item()
            if i >= 32 and (step+1) != self.config.steps and step != 0:
                break
        valid_loss /= (i + 1)
        if modality == "event":
            M, S = self.config.SCAP_ME, self.config.SCAP_SE
            data = event
        else:
            M, S = self.config.SCAP_MI, self.config.SCAP_SI
            data = image
        self.visualize(rec, data, M, S, name="vis_rec_valid_scap")
        print(f"scap step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_scap", valid_loss, step)

        # ---------------------- DD17 ----------------------
        valid_loss = 0.0
        H, W = self.config.DD17_H, self.config.DD17_W
        for i, (event, image) in enumerate(tqdm(self.valid_dd17_dataloader)):
            event, image = event.to(self.config.device), image.to(self.config.device)
            if random.random() < 0.5:
                modality = "event"
                rec, loss = self.forward(event, modality, H, W)
            else:
                modality = "image"
                rec, loss = self.forward(image, modality, H, W)
            valid_loss += loss.item()
            if i >= 32 and (step+1) != self.config.steps and step != 0:
                break
        valid_loss /= (i + 1)
        if modality == "event":
            M, S = self.config.DD17_ME, self.config.DD17_SE
            data = event
        else:
            M, S = self.config.DD17_MI, self.config.DD17_SI
            data = image
        self.visualize(rec, data, M, S, name="vis_rec_valid_dd17")
        print(f"dd17 step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_dd17", valid_loss, step)

        # ---------------------- NIMA ----------------------
        valid_loss = 0.0
        H, W = self.config.NIMA_H, self.config.NIMA_W
        for i, (event, image) in enumerate(tqdm(self.valid_nima_dataloader)):
            event, image = event.to(self.config.device), image.to(self.config.device)
            if random.random() < 0.5:
                modality = "event"
                rec, loss = self.forward(event, modality, H, W)
            else:
                modality = "image"
                rec, loss = self.forward(image, modality, H, W)
            valid_loss += loss.item()
            if i >= 32 and (step+1) != self.config.steps and step != 0:
                break
        valid_loss /= (i + 1)
        if modality == "event":
            M, S = self.config.NIMA_ME, self.config.NIMA_SE
            data = event
        else:
            M, S = self.config.NIMA_MI, self.config.NIMA_SI
            data = image
        self.visualize(rec, data, M, S, name="vis_rec_valid_nima")
        print(f"nima step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid_nima", valid_loss, step)
        
        param_dict = {
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_rec"
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step + 1}_{valid_loss:.4f}.pt"))
        
if __name__ == "__main__":
    from config import RECConfig
    config = RECConfig()
    model = Rec(config).to(config.device)
    model.start()
    print("Training completed.")