import sys
sys.path.append("dinov2")
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
from config import RECConfig
from rec import Rec
from datetime import datetime
import os
import time
import torchvision
from torch import nn
import torch
import torch.nn.functional as F
from torchinfo import summary
from tqdm import tqdm

from model import Block, Transformer
from utils import get_lr, get_param_groups, spatiotemporal_aggregate 

from torch.utils.tensorboard import SummaryWriter

class GPT(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer    = nn.ModuleDict(dict(   modality_embed = nn.Embedding(5, config.n_embed),
                                                    pos_embed = nn.Embedding(config.window_size, config.n_embed),
                                                    blocks = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
                                                    norm = nn.LayerNorm(config.n_embed),
                                                ))
        self.decoder        = nn.ModuleDict(dict(   
            pos_embed = nn.Embedding(config.window_size, config.n_embed),
            blocks = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
            norm = nn.LayerNorm(config.n_embed),
            head = nn.Linear(config.n_embed, config.n_embed),
                                                ))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.n_embed)) if config.mae else None
        # self.initialize_weights()
        self.rec            = Rec(RECConfig())
        self.rec.decoder.load_state_dict(self.config.decoder_weight, strict=True)
        self.rec.h, self.rec.w = self.config.H, self.config.W
        self.rec.eval()
        for name, param in self.rec.named_parameters():
            param.requires_grad = False
        
        # training related
        self.train_dataloader       = torch.utils.data.DataLoader(self.config.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dataloader       = torch.utils.data.DataLoader(self.config.valid_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.amp = torch.amp.autocast(device_type = "cuda")
        self.scaler = torch.amp.GradScaler(device = "cuda")
        self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")
        
        self.train_image_context = torch.load("/data/storage/jianwen/DSEC/train_images/interlaken_00_c/images/left/imageToken/51808300513.pt").unsqueeze(0).to(self.config.device)
        self.train_event_context = torch.load("/data/storage/jianwen/DSEC/train_images/interlaken_00_c/images/left/eventToken/51808300513.pt").unsqueeze(0).to(self.config.device)
        self.valid_image_context = torch.load("/data/storage/jianwen/DSEC/test_images/zurich_city_12_a/images/left/imageToken/57924157588.pt").unsqueeze(0).to(self.config.device)
        self.valid_event_context = torch.load("/data/storage/jianwen/DSEC/test_images/zurich_city_12_a/images/left/eventToken/57924157588.pt").unsqueeze(0).to(self.config.device)
        
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
    
    def forward_mae_transformer(self, x):
        x, mask, ids_restore = self.random_masking(x)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)
        return x, mask, ids_restore
    
    def forward_transformer(self, slot, ids, is_infer=False):
        slot, ids = spatiotemporal_aggregate(
            slot,
            ids,
            tokens_per_image=self.config.n_tokens_per_image,
            images_per_group=8,
        )
        if is_infer:
            x, y = slot.clone().detach(), slot.clone().detach()
            ids = ids.clone().detach()
        else:
            x, y = slot[:, :-1].clone().detach(), slot[:, 1:].clone().detach()
            ids = ids[:, :-1]
        
        B, T, C = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = x + self.transformer.modality_embed(ids)  # add modality embedding
        x = x + self.transformer.pos_embed(pos)  # add position embedding
        
        if self.config.mask_ratio > 0. and not is_infer:
            if hasattr(self.config, 'mask_style') and self.config.mask_style in ['image', 'event']:
                kept_x_list = []
                kept_y_list = []
                
                # 步骤 1: 遍历批次，为每个样本独立计算保留的索引，并提取数据
                for i in range(B):
                    sample_ids = ids[i]
                    mask_token_id = 1 if self.config.mask_style == 'image' else 2

                    unmaskable_indices = torch.where(sample_ids != mask_token_id)[0]
                    maskable_indices = torch.where(sample_ids == mask_token_id)[0]

                    if maskable_indices.numel() > 0:
                        num_to_keep = int(maskable_indices.numel() * (1 - self.config.mask_ratio))
                        perm = torch.randperm(maskable_indices.numel(), device=x.device)
                        kept_maskable_indices = maskable_indices[perm[:num_to_keep]]
                        keep_indices = torch.cat((unmaskable_indices, kept_maskable_indices)).sort()[0]
                    else:
                        keep_indices = unmaskable_indices

                    # 将处理后的（长度不一的）序列存入列表
                    kept_x_list.append(x[i, keep_indices, :])
                    kept_y_list.append(slot[i, keep_indices + 1, :])

                # 步骤 2: 找出批次中被保留序列的最短长度
                min_len = min(t.shape[0] for t in kept_x_list)

                # 步骤 3: 以最短长度为基准，截断所有序列并重新堆叠成一个批次
                x_truncated = [t[:min_len] for t in kept_x_list]
                y_truncated = [t[:min_len] for t in kept_y_list]

                x = torch.stack(x_truncated)
                y = torch.stack(y_truncated)

            else:
                keep_indices = torch.randint(0, T, (int(T * (1 - self.config.mask_ratio)), ), device=slot.device).sort()[0]
                x, y = x[:, keep_indices, :].clone().detach(), slot[:, keep_indices + 1, :].clone().detach()


        for blk in self.transformer.blocks:
            x = blk(x, is_causal=True)
        x = self.transformer.norm(x)
        return x, y
        
    def forward_mae_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
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
        return x
    
    def forward_decoder(self, x):
        x = self.decoder.head(x)
        return x
    
    def forward_mae_loss(self, pred, x, mask):
        loss = (pred - x) ** 2
        # loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss.mean()
        return loss
    
    def forward_loss(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss

    def forward(self, slot, ids, is_infer=False):
        if self.config.mae:
            if not is_infer:
                slot = slot[:, :-1]
            z, mask, ids_restore = self.forward_mae_transformer(slot)
            pred = self.forward_mae_decoder(z, ids_restore)
            loss = self.forward_mae_loss(pred, slot, mask)
        else:
            x, y = self.forward_transformer(slot, ids)
            pred = self.forward_decoder(x)
            loss = self.forward_loss(pred, y)
        return pred, loss
    
    def inference(self, x: torch.Tensor, ids, num_new_tokens: int) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.config.n_embed, "Embedding dim mismatch"
        target_length = N + num_new_tokens
        while x.size(1) < target_length:
            ctx = x[:, -self.config.window_size:]
            ids = ids[:, -self.config.window_size:]

            ctx, _ = self.forward_transformer(ctx, ids, is_infer=True)  # (B, window_size, D)
            ctx = self.forward_decoder(ctx)
            next_emb = ctx[:, -1:, :]              # (B, 1, D)
            next_ids = (torch.tensor([[1]]) if ids[:, 0] == 2 else torch.tensor([[2]])).to(x.device)  # (B, 1)
            x = torch.cat((x, next_emb), dim=1)       # append
            ids = torch.cat((ids, next_ids), dim=1)
        return x
        
    @torch.no_grad()
    def generate(self, x: torch.Tensor, ids, num_new_tokens: int):
        assert num_new_tokens % self.config.n_tokens_per_image == 0
        input_length = x.shape[1]
        n_input_images = input_length // self.config.n_tokens_per_image
        x = self.inference(x, ids, num_new_tokens=num_new_tokens)[-num_new_tokens:]
        print(x.shape, self.config.n_tokens_per_image)
        x = x.reshape(-1, self.config.n_tokens_per_image, self.config.n_embed)[n_input_images:]
        return x
    
    @torch.no_grad()
    def visualize_mae(self):
        train_event_pred, loss1 = self.forward(self.train_event_context, is_infer=True)
        train_image_pred, loss2 = self.forward(self.train_image_context, is_infer=True)
        valid_event_pred, loss3 = self.forward(self.valid_event_context, is_infer=True)
        valid_image_pred, loss4 = self.forward(self.valid_image_context, is_infer=True)
        train_event_context, train_image_context, valid_event_context, valid_image_context = self.train_event_context, self.train_image_context, self.valid_event_context, self.valid_image_context
        train_event_pred = self.rec.forward_decoder(train_event_pred)
        train_image_pred = self.rec.forward_decoder(train_image_pred)
        valid_event_pred = self.rec.forward_decoder(valid_event_pred)
        valid_image_pred = self.rec.forward_decoder(valid_image_pred)
        train_event_context = self.rec.forward_decoder(train_event_context)
        train_image_context = self.rec.forward_decoder(train_image_context)
        valid_event_context = self.rec.forward_decoder(valid_event_context)
        valid_image_context = self.rec.forward_decoder(valid_image_context)
        ME, SE, MI, SI = torch.tensor(self.config.ME, device=self.config.device)[None, :, None, None], torch.tensor(self.config.SE, device=self.config.device)[None, :, None, None], torch.tensor(self.config.MI, device=self.config.device)[None, :, None, None], torch.tensor(self.config.SI, device=self.config.device)[None, :, None, None]
        train_event_pred, train_image_pred = (train_event_pred * SE + ME), (train_image_pred * SI + MI)
        train_event_context, train_image_context = (train_event_context * SE + ME), (train_image_context * SI + MI)
        valid_event_pred, valid_image_pred = (valid_event_pred * SE + ME), (valid_image_pred * SI + MI)
        valid_event_context, valid_image_context = (valid_event_context * SE + ME), (valid_image_context * SI + MI)
        torchvision.utils.save_image(torch.cat([train_event_context, train_image_context, train_event_pred, train_image_pred], dim=0), f"src/vis_gpt_mae_train.png", nrow=2)
        torchvision.utils.save_image(torch.cat([valid_event_context, valid_image_context, valid_event_pred, valid_image_pred], dim=0), f"src/vis_gpt_mae_valid.png", nrow=2)
        print(f"Train event feature loss: {loss1.item():.4f}, Train image feature loss: {loss2.item():.4f}")
        print(f"Valid event feature loss: {loss3.item():.4f}, Valid image feature loss: {loss4.item():.4f}")
    
    @torch.no_grad()
    def visualize(self):
        train_event_pred = self.generate(self.train_image_context, ids=torch.ones(self.train_image_context.shape[:2], device=self.config.device, dtype=torch.int64), num_new_tokens=self.config.n_tokens_per_image)
        train_image_pred = self.generate(self.train_event_context, ids=torch.ones(self.train_event_context.shape[:2], device=self.config.device, dtype=torch.int64)*2, num_new_tokens=self.config.n_tokens_per_image)
        valid_event_pred = self.generate(self.valid_image_context, ids=torch.ones(self.valid_image_context.shape[:2], device=self.config.device, dtype=torch.int64), num_new_tokens=self.config.n_tokens_per_image)
        valid_image_pred = self.generate(self.valid_event_context, ids=torch.ones(self.valid_event_context.shape[:2], device=self.config.device, dtype=torch.int64)*2, num_new_tokens=self.config.n_tokens_per_image)
        train_event_context, train_image_context, valid_event_context, valid_image_context = self.train_event_context, self.train_image_context, self.valid_event_context, self.valid_image_context
        train_event_pred = self.rec.forward_decoder(train_event_pred, self.config.H, self.config.W)
        train_image_pred = self.rec.forward_decoder(train_image_pred, self.config.H, self.config.W)
        valid_event_pred = self.rec.forward_decoder(valid_event_pred, self.config.H, self.config.W)
        valid_image_pred = self.rec.forward_decoder(valid_image_pred, self.config.H, self.config.W)
        train_event_context = self.rec.forward_decoder(train_event_context, self.config.H, self.config.W)
        train_image_context = self.rec.forward_decoder(train_image_context, self.config.H, self.config.W)
        valid_event_context = self.rec.forward_decoder(valid_event_context, self.config.H, self.config.W)
        valid_image_context = self.rec.forward_decoder(valid_image_context, self.config.H, self.config.W)
        ME, SE, MI, SI = torch.tensor(self.config.DSEC_ME, device=self.config.device)[None, :, None, None], torch.tensor(self.config.DSEC_SE, device=self.config.device)[None, :, None, None], torch.tensor(self.config.DSEC_MI, device=self.config.device)[None, :, None, None], torch.tensor(self.config.DSEC_SI, device=self.config.device)[None, :, None, None]
        train_event_pred, train_image_pred = (train_event_pred * SE + ME), (train_image_pred * SI + MI)
        train_event_context, train_image_context = (train_event_context * SE + ME), (train_image_context * SI + MI)
        valid_event_pred, valid_image_pred = (valid_event_pred * SE + ME), (valid_image_pred * SI + MI)
        valid_event_context, valid_image_context = (valid_event_context * SE + ME), (valid_image_context * SI + MI)
        torchvision.utils.save_image(torch.cat([train_event_context, train_image_context, train_image_pred, train_event_pred], dim=0), f"src/vis_gpt_train.png", nrow=2)
        torchvision.utils.save_image(torch.cat([valid_event_context, valid_image_context, valid_image_pred, valid_event_pred], dim=0), f"src/vis_gpt_valid.png", nrow=2)
    
    def train_step(self, slot, ids, global_step):
        slot = slot.to(self.config.device)
        ids = ids.to(self.config.device)
        t0 = time.time()
        self.train()
        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]
            
        with self.amp:
            pred, loss = self.forward(slot, ids)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.0,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        t1 = time.time()
        if (global_step + 1) % self.config.log_every == 0:
            self.writer.add_scalar("loss/train", loss.item(), global_step + 1)
            dt = (t1 - t0)
            num_tokens_per_secend =  self.config.batch_size * self.config.n_tokens_per_image / dt
            print(f"step: {global_step + 1}, lr: {current_lr :.8f}, loss: {loss.item() :.4f}, grad_norm: {grad_norm:.4f}, input: {slot.shape}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")
        
    def start(self):
        if self.config.infer is not None:
            self.transformer.load_state_dict(self.config.infer["transformer"])
            self.decoder.load_state_dict(self.config.infer["decoder"])
            # self.visualize()
        else:
            train_iter = iter(self.train_dataloader)

            for step in range(self.config.steps):
                try:
                    slot, ids = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dataloader)
                    slot, ids = next(train_iter)
                if step == 0:
                    summary(self, input_data=(slot, ids), device=self.config.device, depth=2)
                    os.makedirs("src/runs", exist_ok=True)
                    self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_gpt")
                self.train_step(slot, ids, step)

                if step % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                    self.validate(step)
    
    @ torch.no_grad()
    def validate(self, step):
        self.eval()
        valid_loss = 0.0
        for i, (slot, ids) in enumerate(tqdm(self.valid_dataloader)):
            slot = slot.to(self.config.device)
            ids = ids.to(self.config.device)
            pred, loss = self.forward(slot, ids)
            valid_loss += loss.item()
            if i > 100:
                break
        valid_loss /= i
        print(f"step: {step}, valid loss: {valid_loss:.4f}")
        self.writer.add_scalar("loss/valid", valid_loss, step)
        
        param_dict = {
            "transformer": self.transformer.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_gpt"
        # if self.config.mae:
        #     self.visualize_mae()
        # else:
        #     self.visualize()
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step}_{valid_loss:.4f}.pt"))
        
if __name__ == "__main__":
    from config import GPTConfig
    config = GPTConfig()
    model = GPT(config).to(config.device)
    model.compile()
    model.start()
    print("Training completed.")
