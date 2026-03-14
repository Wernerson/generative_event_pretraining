import sys
sys.path.append("dinov2")
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
from datetime import datetime
import os
import time
import einops
import torchvision
from torch import nn
import torch
import torch.nn.functional as F
from torchinfo import summary
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_small
from utils import get_lr, get_param_groups
from model import Block, Transformer

from torch.utils.tensorboard import SummaryWriter

class MAE(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6)
        if config.event_encoder_weight is not None:
            self.event_encoder.load_state_dict(config.event_encoder_weight, strict=True)
            print("*" * 50 + " event encoder loaded")

        self.decoder = nn.ModuleDict(dict(  project = nn.Linear(self.event_encoder.embed_dim, config.n_embed) if config.n_embed != self.event_encoder.embed_dim else nn.Identity(),
                                            pos_embed = nn.Embedding(config.n_tokens_per_image + 1, config.n_embed),
                                            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                            norm = nn.LayerNorm(config.n_embed),
                                            head = nn.Linear(config.n_embed, 3 * config.P ** 2),
                                            ))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.n_embed))

        # training related
        self.train_dataloader       = torch.utils.data.DataLoader(self.config.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dataloader       = torch.utils.data.DataLoader(self.config.valid_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.amp = torch.amp.autocast(device_type = "cuda")
        self.scaler = torch.amp.GradScaler(device = "cuda")
        self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd, self.config.encoder_lr_mult))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")

        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.event_encoder.cls_token, std=.02)
        w = self.event_encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random noise. x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.config.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        assert self.event_encoder.num_register_tokens == 0, "MAE does not support register tokens"

        # regular positional embedding
        B, C, H, W = x.shape
        x_cls, x_patch = self.event_encoder.cls_token.expand(x.shape[0], -1, -1), self.event_encoder.patch_embed(x)
        x = torch.cat((x_cls, x_patch), dim=1)
        x = x + self.event_encoder.interpolate_pos_encoding(x, H, W)

        # random masking
        x_cls, x_patch = x[:, :1], x[:, 1:]
        x_patch, mask, ids_restore = self.random_masking(x_patch)
        x = torch.cat((x_cls, x_patch), dim=1)

        # regular encoder forward
        for blk in self.event_encoder.blocks:
            x = blk(x)
        x = self.event_encoder.norm(x)

        return x[:, 1:, :], mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        x = self.decoder.project(x)  
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # regular decoder forward
        B, T, C = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = x + self.decoder.pos_embed(pos)
        for blk in self.decoder.blocks:
            x = blk(x)
        x = self.decoder.norm(x)
        x = self.decoder.head(x)
        x = einops.rearrange(x, "b (l1 l2) (c p1 p2) -> b c (l1 p1) (l2 p2)", c=3, p1=self.config.P, p2=self.config.P, l1=self.config.H // self.config.P, l2=self.config.W // self.config.P)
        return x
    
    def forward_loss(self, pred, target, mask):
        """
        pred    : [N, 3, H, W]
        target  : [N, 3, H, W]
        mask    : [N, L], 0 is keep, 1 is remove, 
        """
        if self.config.norm_pix_loss:
            mean = target.flatten(2).mean(dim=-1, keepdim=True)[:, :, None]
            var = target.flatten(2).var(dim=-1, keepdim=True)[:, :, None]
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = einops.rearrange(loss, "b c (l1 p1) (l2 p2) -> b (l1 l2) (c p1 p2)", p1=self.config.P, p2=self.config.P, l1=self.config.H // self.config.P, l2=self.config.W // self.config.P)
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        z, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(z, ids_restore)
        loss = self.forward_loss(pred, x, mask)
        return pred, loss, ids_restore
    
    @torch.no_grad()
    def visualize(self, x, pred, ids_restore, M, S, name, n=8):
        M, S = torch.tensor(M, device=self.config.device)[None, :, None, None], torch.tensor(S, device=self.config.device)[None, :, None, None]
        x, pred, ids_restore = x[:n], pred[:n], ids_restore[:n]
        x_flatten = einops.rearrange(x, "b c (l1 p1) (l2 p2) -> b (l1 l2) (c p1 p2)", p1=self.config.P, p2=self.config.P, l1=self.config.H // self.config.P, l2=self.config.W // self.config.P)
        ids_restore = ids_restore[:, :int((1 - self.config.mask_ratio) * x_flatten.shape[1])]

        x_flatten[:, ids_restore] = torch.zeros_like(x_flatten[:, ids_restore])
        x_masked = einops.rearrange(x_flatten, "b (l1 l2) (c p1 p2) -> b c (l1 p1) (l2 p2)", c=3, p1=self.config.P, p2=self.config.P, l1=self.config.H // self.config.P, l2=self.config.W // self.config.P)
        x, x_masked, pred = x * S + M, x_masked * S + M, pred * S + M
        torchvision.utils.save_image(torch.cat([x, x_masked, pred], dim=0), f"src/{name}.png", nrow=n)
    
    def train_step(self, x, global_step):
        x = x.to(self.config.device)
        t0 = time.time()
        self.train()
        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]
            
        with self.amp:
            pred, loss, ids_restore = self.forward(x)
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        self.writer.add_scalar("loss/train", loss.item(), global_step + 1)
        t1 = time.time()

        if (global_step + 1) % self.config.log_every == 0:
            dt = (t1 - t0)
            self.visualize(x, pred, ids_restore, M=self.config.ME, S=self.config.SE, name="vis_mae_train", n=8)
            num_tokens_per_secend =  self.config.batch_size * self.config.n_tokens_per_image / dt
            print(f"step: {global_step + 1}, lr: {current_lr :.8f}, loss: {loss.item() :.4f}, grad_norm: {grad_norm: .4f}, input: {x.shape}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")
        
    def start(self):
        train_iter = iter(self.train_dataloader)

        for step in range(self.config.steps):
            try:
                event, image = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                event, image = next(train_iter)

            if step == 0:
                summary(self, input_data=event, device=self.config.device, depth=2)
                os.makedirs("src/runs", exist_ok=True)
                self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_mae")
            self.train_step(event, step)

            if (step + 1) % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                self.validate(step)
    
    def validate(self, step):
        self.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (event, image) in enumerate(tqdm(self.config.valid_dataloader)):
                event, image = event.to(self.config.device), image.to(self.config.device)
                if i > 100:
                    break
                pred, loss, ids_restore = self.forward(event)
                valid_loss += loss.item()
            self.visualize(event, pred, ids_restore, M=self.config.ME, S=self.config.SE, name="vis_mae_valid", n=8)
        valid_loss /= i
        print(f"step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid", valid_loss, step)
        
        param_dict = {
            "event_encoder": self.event_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mask_token": self.mask_token,
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_mae"
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step}_{valid_loss:.4f}.pt"))
        
if __name__ == "__main__":
    from config import MAEConfig
    config = MAEConfig()
    mae = MAE(config).to(config.device)
    mae.start()
    print("Training completed.")